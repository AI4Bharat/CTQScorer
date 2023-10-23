import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size, activation_func):
        super(NeuralNet, self).__init__()
        
        # Create a list to hold the hidden layers
        layers = []
        
        # Add the input layer
        layers.append(nn.Linear(input_size, hidden_size))
        
        # Add the hidden layers
        for i in range(hidden_layers):
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_func())
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Add the output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Create a Sequential model using the layers list
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

    
def get_activation_func(config_activation):
    if config_activation == 'relu':
        activation_func = nn.ReLU
    elif config_activation == 'sigmoid':
        activation_func = nn.Sigmoid
    elif config_activation == 'tanh':
        activation_func = nn.Tanh
    return activation_func


def get_optimizer(model, config_optimizer, config_learning_rate, config_weight_decay):
    if config_optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config_learning_rate, weight_decay=config_weight_decay)
    elif config_optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config_learning_rate, weight_decay=config_weight_decay)
    elif config_optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config_learning_rate, weight_decay=config_weight_decay)
    return optimizer