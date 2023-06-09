import torch as th 
import torch.nn as nn
from torch_geometric.nn import TAGConv
from torch_geometric.nn.pool import global_add_pool
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
    Assumes data comes in shape: (BATCH_SIZE, Nodes, Features) 
    If this shape changes adjust the code
    """

    assert len(real.shape) != 3, "shape of tensors (real, fake) must be adjusted. Adjust code below to match shape"

    # Create interpolated outputs
    BATCH_SIZE, N, F = real.shape 
    alpha = th.rand((BATCH_SIZE, 1, 1)).repeat(1, N, F) 
    interpolated_batched_features = real * alpha + fake * (1 - alpha)

    # Critic score of interpolated features
    interpolation_score = discriminator(interpolated_batched_features)

    # Compute gradient of (interpolated) score wrt features
    # Note that (inputs, outputs) linked through a function above (interpolated_batched_features)
    gradient = th.autograd.grad(
        inputs=interpolated_batched_features,       # what we compute the gradients wrt to
        output=interpolation_score,                 # output we want gradients of
        grad_outputs=th.ones_like(interpolation_score),
        create_graph=True,
        retain_graph=True
    )[0]

    # Change shape so we can compute penalty over the non batch dimension
    gradient = gradient.view(gradient.shape[0], -1)         
    gradient_norm = gradient_norm(2, dim=-1)
    gradient_penalty = th.mean((gradient_norm - 1)**2)
    
    return gradient_penalty


