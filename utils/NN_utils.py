
import torch
import torch.nn as nn
from tqdm import tqdm
# Neural Network utilities for PyTorch



class Net_group_Y(nn.Module):
    
    def __init__(self, input_size=2, output_size=1,
                 hidden_layer_sizes=[3,4,5], activation=nn.ReLU()):
        
        super(Net_group_Y, self).__init__()
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
                x = self.activation(x)
        # Apply sigmoid activation to the output layer
        x = torch.sigmoid(x)
        return x
    
    #
    # 1. 4. Training loop
    # For details, see Machine Learning with PyTorch and Scikit-Learn.
    #
    def train(self, num_epochs, loss_fn, optimizer, train_dl, train_size, batch_size, x_valid, y_valid):
        # Initialize weights
        self.apply(self._init_weights)
    
        # Loss and accuracy history objects initialization
        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        accuracy_hist_valid = [0] * num_epochs
        
        # Learning loop
        for epoch in tqdm(range(num_epochs)):
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
                
            # Average the results
            loss_hist_train[epoch] /= train_size/batch_size
            accuracy_hist_train[epoch] /= train_size/batch_size
            
            # Predict the validation set
            pred = self(x_valid)[:, 0]
            loss_hist_valid[epoch] = loss_fn(pred, y_valid).item()
            is_correct = ((pred>=0.5).float() == y_valid).float()
            accuracy_hist_valid[epoch] += is_correct.mean()
            
        return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

    # Not needed normaly, it is just for mlextend plots
    def predict(self, x):
        print(f'predict with input shape: {x.shape}')
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        print(f'finished predict with output shape: {pred.shape}')
        return (pred>=0.5).float()
        