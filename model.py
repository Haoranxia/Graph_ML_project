import torch as th 
import torch.nn as nn
from torch_geometric.nn import TAGConv
import numpy as np 

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        """ 
        input_dim:      
        outpit_dim:     
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Network layers
        self.module_list = []

        # Input layer:
        self.module_list.append(TAGConv(input_dim, hidden_dims[0]))

        # Intermediate layers
        for i in range(1, len(hidden_dims - 1)):
            layer = TAGConv(hidden_dims[i] - 1, hidden_dims[i])
            self.module_list.append(layer)

        # Output layers
        self.module_list.append(TAGConv(hidden_dims[-1], output_dim))

    
    def forward(self, data):
        """ 
        data:          the full graph   
        """
        x = data.x 
        edge_index = data.edge_index

        for module in self.module_list():
            x = module(x=x, edge_index = edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.PReLU()(x)
        
        return x 



class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[])
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims 
