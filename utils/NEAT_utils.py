import sys
import neat
import pickle
import time
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


# --- NEATWrapper Class (Modified to accept config_file_path) ---
class NEATWrapper:
    def __init__(self, config_file_path, X_train, y_train, X_test, y_test,
                 generations,seed, log_path = None, log_level=None, **full_params):

        # Convert tensors to numpy arrays for NEAT compatibility if needed
        # Assuming NEAT's activate method can handle iterables (lists/arrays)
        torch.manual_seed(seed)


        # Define datasets for data loaders
        train_ds_not_norm = TensorDataset(X_train, y_train)
        test_ds_not_norm = TensorDataset(X_test, y_test)

        X_train, y_train = train_ds_not_norm.tensors
        X_test, y_test = test_ds_not_norm.tensors

        mean = X_train.mean(dim=0) 
        std = X_train.std(dim=0) 
        std[std == 0] = 1.0  

        X_train_normalized = (X_train - mean) / std
        X_test_normalized = (X_test - mean) / std
        
        
        
        self.X_train = X_train_normalized.numpy() 
        self.y_train = y_train.numpy() 
        self.X_test = X_test_normalized.numpy()
        self.y_test = y_test.numpy() 

        self.generations = generations
        self.seed = seed
        self.config_file_path = config_file_path # Store the path

        self.log_level=log_level
        self.log_path = log_path

        self.best_fitness_per_generation = []  # Track best fitness each generation
        self.best_test_fitness_per_generation = []  # Track best test fitness
        self.generation_times = []  # Track time per generation
        
        # I use config first so I am sure that all parameters are set correctl
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.config_file_path) # Use the stored path

        neat_parmms = ['fitness_criterion', 'fitness_threshold', 'pop_size','reset_on_extinction']

        
        # Apply fixed_params to config
        for param, value in full_params.items():
            # This assumes fixed_params are genome_config parameters.
            # If they belong to other sections (e.g., NEAT, DefaultStagnation),
            # additional logic would be needed here.
            if param in config.genome_config.__dict__:
                setattr(config.genome_config, param, value)
            
            elif param in config.reproduction_config.__dict__:
                setattr(config.reproduction_config, param, value)
            
            elif param in config.species_set_config.__dict__:
                setattr(config.species_set_config, param, value)
            
            elif param in config.stagnation_config.__dict__:
                setattr(config.stagnation_config, param, value)
            
            elif param in neat_parmms:
                setattr(config, param, value)
                

        # Create population with seed
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(int(generations/3)))

        # Determine if NEAT's verbose output should be redirected
        log_level = full_params.get("log_level", 0)
        neat_output_redirected = False
        if log_level > 1: # Redirect if log_level is sufficiently high (e.g., 2 for outer loop)
            timestamp = int(time.time() * 1000)
            neat_output_file = f"neat_run_stdout_{timestamp}.txt"
            print(f"Redirecting NEAT stdout to {neat_output_file}")
            with open(neat_output_file, 'w') as f:
                original_stdout = sys.stdout
                sys.stdout = f
                try:
                    self.winner = p.run(self.eval_genomes, generations)
                    neat_output_redirected = True

                finally:
                    sys.stdout = original_stdout
        else:
            # If not redirecting, still suppress some default NEAT output if needed
            # For simplicity, we just run directly if not redirecting to a file
            self.winner = p.run(self.eval_genomes, generations)

        # Calculate final fitness values for grid search compatibility
        self._calculate_final_fitness()

        self.stats = stats

        # Logging
        if self.log_level == 2:
            self.logger(self.log_path)
        else: 
            pass

    def eval_rmse(self, net, X, y):
        '''
        Auxiliary function to evaluate the RMSE.
        '''
        fit = 0.
        # Ensure y is a list of single values for direct comparison if it came from torch.Tensor
        y_list = [val.item() if isinstance(val, torch.Tensor) else val for val in y]

        for xi, xo in zip(X, y_list):
            output = net.activate(xi)
            fit += (output[0] - xo)**2
        # RMSE
        return (fit/len(y_list))**.5

    def eval_genomes(self, genomes, config):
        '''
        The function used by NEAT-Python to evaluate the fitness of the genomes.
        -> It has to have the two first arguments genomes and config.
        -> It has to update the `fitness` attribute of the genome.
        '''

        generation_start = time.time()
        best_fitness = -float('inf')
        best_test_fitness = -float('inf')

        for genome_id, genome in genomes:
            # Define the network
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            # Train fitness (negative RMSE for maximization)
            genome.fitness = -self.eval_rmse(net, self.X_train, self.y_train)

            # Test fitness (using X_test, y_test from grid search)
            genome.fitness_val = -self.eval_rmse(net, self.X_test, self.y_test)

            # Track best fitness in this generation
            if genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_test_fitness = genome.fitness_val
        
        # Store generation results
        self.best_fitness_per_generation.append(best_fitness)
        self.best_test_fitness_per_generation.append(best_test_fitness)
        self.generation_times.append(time.time() - generation_start)

    def _calculate_final_fitness(self):
        '''
        Calculate final fitness values for grid search compatibility.
        Returns positive RMSE values as expected by the grid search framework.
        '''
        # Create network from winner
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.config_file_path)  # Use the stored path

        net = neat.nn.FeedForwardNetwork.create(self.winner, config)

        # Calculate positive RMSE values (grid search expects positive values)
        self.fitness = self.eval_rmse(net, self.X_train, self.y_train)
        self.test_fitness = self.eval_rmse(net, self.X_test, self.y_test)


    def item(self):
        '''
        Compatibility method for grid search framework that expects .item() calls
        '''
        # This method is not directly used for fitness/test_fitness values themselves,
        # but ensures the object has a .item() method if called on other attributes.
        # For actual fitness values, they are directly accessed as self.fitness and self.test_fitness.
        return self

    def logger(self, log_path): 
        
        df = pd.DataFrame(index=range(self.generations))
        df['algorithm'] = 'NEAT'
        df['Instance ID'] = 1 #PLACEHOLDER
        df['dataset'] = 2 #PLACEHOLDER 
        df['seed'] = self.seed
        df['generations'] = range(1, self.generations + 1)
        df['fitness'] = self.best_fitness_per_generation
        df['running time'] = self.generation_times
        df['population nodes'] = 7 #PLACEHOLDER
        df['test_fitness'] = self.best_test_fitness_per_generation
        df['Elite nodes'] = 9 #PLACEHOLDER
        df['niche entropy'] = 10 #PLACEHOLDER
        df['sd(pop.fit)'] = 11 #PLACEHOLDER
        df['Log Level'] = 12 #PLACEHOLDER
        df['params'] = 'TREZE' #PLACEHOLDER

        filename_object = f'{log_path}.pkl'

        # Save the custom object
        try:
            with open(filename_object, 'wb') as file:
                pickle.dump(self.stats, file)
            print(f"Custom object successfully saved to {filename_object}")
        except Exception as e:
            print(f"Error saving custom object: {e}")

        # If 
        df.to_csv(log_path, index=False, header = False)
