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
    def __init__(self, input_dim, output_dim, hidden_dims=[], device='cpu'):
        """ 
        input_dim:      
        outpit_dim:     
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = device

        self.dropout_rate = 0.1

        # Network layers
        self.module_list = nn.ModuleList()

        # Input layer:
        self.module_list.append(TAGConv(input_dim, hidden_dims[0]))

        # Intermediate layers
        for i in range(1, len(hidden_dims) - 1):
            layer = TAGConv(hidden_dims[i - 1], hidden_dims[i])
            self.module_list.append(layer)

        # Output layers
        self.module_list.append(TAGConv(hidden_dims[-1], output_dim))

    
    def forward(self, data, noise):
        """ 
        data:          the full graph   
        """
        category = data.category
        x = th.concat((category.to(self.device), noise.to(self.device)), dim=1)
        for module in self.module_list:
            x = module(x=x, edge_index=data.edge_index.to(self.device))
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.PReLU()(x)
        
        return x 



class Discriminator(nn.Module):
    """ 
    WGAN-GP Discriminator
    Goal: minimize critic score for real samples
          maximize critic score for generator samples
    """
    def __init__(self, input_dim, hidden_dims=[], device='cpu'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device

        self.dropout_rate = 0.1

        # scalar value (score) indicating how real the input is
        # in WGAN-GP we dont use sigmoid to turn it into a probability
        self.output_dim = 1     

        self.module_list = nn.ModuleList()
        
        # Input layer 
        self.module_list.append(TAGConv(input_dim, hidden_dims[0]))

        # Intermediate layers 
        for i in range(1, len(hidden_dims)):
            layer = TAGConv(hidden_dims[i - 1], hidden_dims[i])
            self.module_list.append(layer)

        # Output layer 
        output_layer = nn.Linear(hidden_dims[-1], 1)
        self.module_list.append(output_layer)

    
    def forward(self, data):
        """ 
        data:          graph  
        """
        x = data.geometry.to(self.device)

        for i in range(len(self.module_list) - 1):
            x = self.module_list[i](x=x , edge_index=data.edge_index.to(self.device))
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.PReLU()(x)

        # Pool graph node features
        x = global_add_pool(x, data.batch)

        # Predict WGAN score
        x = self.module_list[-1](x)

        return x 
    

    
def gradient_penalty(discriminator, real, fake):
    """ 
    real: torch.geometric Batch object
    fake: torch.geometric Batch object
    NOTE: real, fake are geometric Batch objects (not tensors!). I hope I did it properly

    Constructs a batch (big graph) with features interpolated between real data and fake (generated) data
    Then compute the gradients of the loss of the discriminator over the interpolated constructed data
    
    This gradient penalty is a type of regularization that encourages (enforces) lipschitz continuity
    Lipschitz continuity is a property of functions where the rate of change is bounded by a constant.

    In WGAN-GP the gradient penalty term tries to enforce 1-lipschitz continuity with respect to the interpolated features.
    The interpolation can be seen as a form of distance metric between the real and fake data. We sample points along this
    distance through interpolation. 

    We then use the discriminator to score this interpolated data and try to keep the gradients of the discriminator wrt 
    the interpolated image close to 1. The idea is that using this interpolated data as extra information we can indirectly
    encourage the gradients to be smoother for both real and fake data resulting in some lipschitz continuity regularization
    This is because the interpolated data provides additional information about the local behaviour of the discriminator.
    Local as in around the real and fake data. 

    The interpolated data capture the transition between the real and fake data distribution, the gradients wrt this interpolation
    provides information about how the discriminator perceives the transition between real and fake data. We try to keep this gradient
    close to 1 to encourage smoothness which improves training stability (big issue for GANs) 

    """
    assert real.geometry.shape == fake.geometry.shape, "real and fake geometry shapes dont match"
    assert real.edge_index.shape == fake.edge_index.shape, "reral and fake edge index dont match"
    assert real.batch.shape == fake.batch.shape, "real and fake batch shape doesn't match"

    real.geometry.requires_grad = True
    #fake.geometry.requires_grad = True

    # Construct interpolated geometry/features and compute score
    N, F = fake.geometry.shape
    alpha = th.rand((N, 1)).repeat(1, F)
    interpolated_geometry = alpha * real.geometry + (1 - alpha) * fake.geometry
    #interpolated_geometry.requires_grad = True 

    interpolated_batch = Batch(geometry=interpolated_geometry, edge_index=real.edge_index, batch=real.batch)
    interpolation_score = discriminator(interpolated_batch)

    # Compute gradient of interpolated score wrt interpolated features
    # Note that (inputs, outputs) linked through a function above (discriminator is our function in this case)
    gradient = th.autograd.grad(
        inputs=interpolated_geometry,                            # what we compute the gradients wrt to
        outputs=interpolation_score,                             # output we want gradients of
        grad_outputs=th.ones_like(interpolation_score),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Change shape so we can compute penalty over the non batch dimension
    gradient = gradient.view(gradient.shape[0], -1)       
    gradient_norm = gradient.norm(2, dim=-1)

    # (norm - 1) ** 2 enforces 1 lipschitz continutiy as we aim for this value to be as small as possible (0)
    # which is only achieved when the gradient_norm = 1
    return ((gradient_norm - 1 ) ** 2).mean()