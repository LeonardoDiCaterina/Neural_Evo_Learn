# library imports
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

criterion = RMSELoss()



class Net_Arc(nn.Module):
    
    def __init__(self, input_size=2, output_size=1,
                 hidden_layer_sizes:tuple = (3,4,5), activation=nn.ReLU()):
        
        super(Net_Arc, self).__init__()
        #
        # 1. 1. Network architecture
        
        self.add_module(f'fc{1}', nn.Linear(input_size, hidden_layer_sizes[0]))
        
        for i in range(1,len(hidden_layer_sizes)):
            self.add_module(f'fc{i+1}', nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
        
        self.add_module(f'fc{len(hidden_layer_sizes) +1 }', nn.Linear(hidden_layer_sizes[-1], output_size))

        # Weights initialisation
        # The apply method applies the function passed as the apply() argument
        # to each element in the object, that in this case is the neural network.
        self.apply(self._init_weights)
        # Store the parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        
        self.n_forward_calls = 0
        
    #
    # 1. 2. Weights and bias initialization
    #
    def _init_weights(self, attribute):
        if isinstance(attribute, nn.Linear):
          torch.nn.init.xavier_uniform_(attribute.weight)
          torch.nn.init.zeros_(attribute.bias)
    #
    # 1. 3. Forward pass
    """    
    def forward(self, x):
        # For each layer, the output will be the ReLu activation applied to the output of the linear operation
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # For the last layer, the sigmoid function will be the activation
        x = torch.sigmoid(self.fc3(x))
        return x"""
    
    def forward(self, x):
        # Forward pass through all layers
        for i in range(1, len(self.hidden_layer_sizes) + 2):
            #print(f'forward pass layer {i}')
            layer = getattr(self, f'fc{i}')
            x = layer(x)
            if i < len(self.hidden_layer_sizes):
                self.n_forward_calls += 1
                x = self.activation(x)
        # Apply sigmoid activation to the output layer
        self.n_forward_calls += 1
        x = torch.relu(x)
        return x
    
    #
    # 1. 4. Training loop
    # For details, see Machine Learning with PyTorch and Scikit-Learn.
    #
    def train(self, num_epochs, loss_fn, optimizer, train_dl, train_size, batch_size, x_valid, y_valid):
        # Initialize weights
        self.apply(self._init_weights)
    
        n_calls_array = np.zeros(num_epochs)

        # Loss and accuracy history objects initialization
        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        accuracy_hist_valid = [0] * num_epochs
        delta_times = [0] * num_epochs
        self.n_forward_calls = 0
        
        # Learning loop
        for epoch in tqdm(range(num_epochs)):
            start_time = time.time()
            # Batch learn
            for x_batch, y_batch in train_dl:
                #print('*'*20)
                # ---
                # 1.4.1. Get the predictions, the [:,0] reshapes from (batch_size,1) to (batch_size)
                pred = self(x_batch)[:,0]
                # 1.4.2. Compute the loss
                loss = loss_fn(pred, y_batch)
                # 1.4.3. Back propagate the gradients
                # The `backward()` method, already available in PyTroch, calculates the 
                # derivative of the Error in respect to the NN weights
                # applying the chain rule for hidden neurons.
                loss.backward()
                # 1.4.4. Update the weights based on the computed gradients
                optimizer.step()
                # ---
                
                # Reset to zero the gradients so they will not accumulate over the mini-batches
                optimizer.zero_grad()
                
                # Update performance metrics
                loss_hist_train[epoch] += loss.item()
                is_correct = ((pred>=0.5).float() == y_batch).float()
                accuracy_hist_train[epoch] += is_correct.mean()
            
            n_calls_array[epoch] = self.n_forward_calls
            self.n_forward_calls = 0
            delta_times[epoch] = time.time() - start_time
            # Average the results
            loss_hist_train[epoch] /= train_size/batch_size
            accuracy_hist_train[epoch] /= train_size/batch_size
            
            # Predict the validation set
            pred = self(x_valid)[:, 0]
            loss_hist_valid[epoch] = loss_fn(pred, y_valid).item()
            is_correct = ((pred>=0.5).float() == y_valid).float()
            accuracy_hist_valid[epoch] += is_correct.mean()
            
        return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid, n_calls_array, delta_times

    # Not needed normaly, it is just for mlextend plots
    def predict(self, x):
        print(f'predict with input shape: {x.shape}')
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        print(f'finished predict with output shape: {pred.shape}')
        return (pred>=0.5).float()


class nn_model():
    def __init__(self,
                X_train,
                X_test,
                y_train,
                y_test,
                input_size, 
                output_size,
                hidden_layer_sizes, 
                optimizer_name:str,
                num_epochs,  
                train_size,
                batch_size,
                learning_rate,
                log_path = None,
                train_dl = None,
                log_level = None,
                seed = 42,
                activation = nn.ReLU(),
                loss_fn = RMSELoss()
                ):
        

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer_name = optimizer_name
        self.num_epochs = num_epochs
        self.train_dl = train_dl 
        self.train_size = train_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_path = log_path
        self.log_level = log_level
        self.seed = seed 
        self.activation = activation
        self.loss_fn = loss_fn

        
        self.arcthicture = Net_Arc(input_size=input_size,
                                  output_size=output_size,
                                  hidden_layer_sizes=hidden_layer_sizes,
                                  activation=activation)
        
        self.arcthicture.to('cpu')
        
        
        self.fitnesses, self.test_fitnesses, self.accuracy, self.accuracy_valid, self.n_forward_calls, self.delta_times = self.fit(
                X_train,
                X_test,
                y_train,
                y_test,
                num_epochs=num_epochs, 
                loss_fn=loss_fn, 
                optimizer_name=optimizer_name,
                batch_size=batch_size, 
                learning_rate=learning_rate,
                seed=seed)
    
        if self.log_level == 2:
            self.logger(self.log_path)
        else: 
            pass


        self.fitness = torch.tensor(self.fitnesses[-1])
        self.test_fitness = torch.tensor(self.test_fitnesses[-1])


    def logger(self, log_path): 
        

        df = pd.DataFrame(index=range(self.num_epochs))
        df['algorithm'] = str(self.optimizer_name)
        df['Instance ID'] = 1 #PLACEHOLDER
        df['dataset'] = 2 #PLACEHOLDER 
        df['seed'] = self.seed
        df['epochs'] = range(1, self.num_epochs + 1)
        df['fitness'] = self.fitnesses
        df['running time'] = self.delta_times
        df['population nodes'] = 7 #PLACEHOLDER
        df['test_fitness'] = self.test_fitnesses
        df['Elite nodes'] = 9 #PLACEHOLDER
        df['niche entropy'] = 10 #PLACEHOLDER
        df['sd(pop.fit)'] = 11 #PLACEHOLDER
        df['Log Level'] = 12 #PLACEHOLDER
        df['params'] = 'TREZE' #PLACEHOLDER
        df['n_forward_calls'] = self.n_forward_calls


        # If 
        df.to_csv(log_path, index=False, header = False)

        
        
       

    def fit(self,X_train,X_test,y_train,y_test,
                num_epochs, loss_fn, optimizer_name, batch_size, learning_rate, seed):
        """
        Train the model with the given parameters.
        
        Parameters:
        - num_epochs: Number of epochs to train the model.
        - loss_fn: Loss function to use for training.
        - optimizer: Optimizer to use for training.
        - batch_size: Size of each batch during training.
        - x_valid: Validation input data.
        - y_valid: Validation target data.
        - learning_rate: Learning rate for the optimizer.
        
        Returns:
        - history: Training history containing loss and accuracy metrics.
        """
        
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

        train_ds= TensorDataset(X_train_normalized, y_train)
        test_ds = TensorDataset(X_test_normalized, y_test)

        train_size = len(train_ds)
        if optimizer_name == 'GD':
            batch_size = train_size
            train_dl = DataLoader(train_ds, batch_size, shuffle=True)
            #val_dl = DataLoader(val_ds, batch_size, shuffle=True)
        
        elif optimizer_name == 'SGD':
            batch_size = 1
            train_dl = DataLoader(train_ds, batch_size, shuffle=True)
            #val_dl = DataLoader(val_ds, batch_size, shuffle=True)

        else:
            batch_size = batch_size
            train_dl = DataLoader(train_ds, batch_size, shuffle=True)
            #val_dl = DataLoader(val_ds, batch_size, shuffle=True)
        
        optimizer_choiche = {
        'GD': torch.optim.SGD(self.arcthicture.parameters(), lr=learning_rate),
        'SGD': torch.optim.SGD(self.arcthicture.parameters(), lr=learning_rate),
        'MiniSGD': torch.optim.SGD(self.arcthicture.parameters(), lr=learning_rate),
        'ASGD': torch.optim.ASGD(self.arcthicture.parameters(), lr=learning_rate),
        'RMSprop': torch.optim.RMSprop(self.arcthicture.parameters(), lr=learning_rate),
        'Adam': torch.optim.Adam(self.arcthicture.parameters(), lr=learning_rate)
        }
        optimizer_instance = optimizer_choiche[optimizer_name]
        
        return self.arcthicture.train(
            num_epochs=num_epochs, 
            loss_fn=loss_fn, 
            optimizer=optimizer_instance, 
            train_dl=train_dl, 
            train_size=train_size, 
            batch_size=batch_size,
            x_valid=X_test_normalized,     
            y_valid=y_test )
        

