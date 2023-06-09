import torch as th 
import torch.nn as nn
from torch_geometric.nn import TAGConv
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np 

class Generator(nn.Module):
    """ 
    WGAN-GP Generator

    Goal: make Discriminator maximize its output (score)
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        """ 
        input_dim:      
        outpit_dim:     
        """
        super().__init__()

        self.hidden_dims = hidden_dims

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
    """ 
    WGAN-GP Discriminator
    Goal: minimize critic score for real samples
          maximize critic score for generator samples
    """
    def __init__(self, input_dim, hidden_dims=[]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims 

        # scalar value (score) indicating how real the input is
        # in WGAN-GP we dont use sigmoid to turn it into a probability
        self.output_dim = 1     

        self.module_list = []
        
        # Input layer 
        self.module_list.append(TAGConv(input_dim, hidden_dims[0]))

        # Intermediate layers 
        for i in range(1, len(hidden_dims)):
            layer = TAGConv(hidden_dims[i - 1], hidden_dims[i])
            self.module_list.append(layer)

        # Output layer 
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    
    def forward(self, data):
        """ 
        data:          graph  
        """
        x = data.x 
        edge_index = data.edge_index

        for module in self.module_list():
            x = module(x=x, edge_index = edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.PReLU()(x)

        # Pool graph node features
        x = global_add_pool(x)

        # Predict WGAN score
        x = self.output_layer(x)
        
        return x 




def gradient_penalty(discriminator, real, fake):
    """ 
    real: torch.geometric Batch object
    fake: torch.geometric Batch object
    NOTE: real, fake are geometric Batch objects (not tensors!). I hope I did it properly
    """

    # Create interpolated graph over node features and make it into a batch 
    interpolated_batch = []
    for (r_data, f_data) in zip(real, fake):       
        N, F = r_data.x.shape
        alpha = th.rand((N, 1)).repeat(1, F)
        interpolated_x = alpha * r_data.x  + (1 - alpha) * f_data.x
        i_data = Data(x=interpolated_x, edge_index=r_data.edge_index)
        interpolated_batch.append(i_data)

    interpolated_batch = Batch.from_data_list(interpolated_batch)

    # Critic score of interpolated features
    interpolation_score = discriminator(interpolated_batch)

    # Compute gradient of (interpolated) score wrt features
    # Note that (inputs, outputs) linked through a function above (interpolated_batched_features)
    gradient = th.autograd.grad(
        inputs=interpolated_batch,                              # what we compute the gradients wrt to
        output=interpolation_score,                             # output we want gradients of
        grad_outputs=th.ones_like(interpolation_score),
        create_graph=True,
        retain_graph=True
    )[0]

    # Change shape so we can compute penalty over the non batch dimension
    gradient = gradient.view(gradient.shape[0], -1)         
    gradient_norm = gradient_norm(2, dim=-1)
    gradient_penalty = th.mean((gradient_norm - 1)**2)
    
    return gradient_penalty


