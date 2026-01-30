import torch
import copy

"""
Manifold Mapping Utilities.

These functions act as coordinate charts, mapping between the high-level 
representation of the Neural Network (structured layers, weights, biases) 
and the flat vector space required by the Symplectic Integrator.

In Differential Geometry terms:
- `get_flat_params`: $\phi: \mathcal{M} \to \mathbb{R}^D$ (Coordinate Map)
- `load_flat_params`: $\phi^{-1}: \mathbb{R}^D \to \mathcal{M}$ (Inverse Map)
"""

def get_flat_params(model):
    """
    Flattens the model's parameter manifold into a single Euclidean vector $\theta \in \mathbb{R}^D$.
    
    Args:
        model (nn.Module): The PyTorch model.
        
    Returns:
        torch.Tensor: A 1D tensor containing all trainable parameters.
    """
    return torch.cat([p.view(-1) for p in model.parameters()])

def load_flat_params(model, flat_params):
    """
    Projects a point $\theta \in \mathbb{R}^D$ back onto the Neural Network's parameter manifold.
    This modifies the model in-place.
    """
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset+numel].view_as(p))
        offset += numel

def unflatten_params(model, flat_params):
    """
    Reconstructs the structured parameter dictionary (state_dict) from a flat vector 
    without modifying the model itself.
    
    Essential for `functional_call` where we evaluate gradients at arbitrary points 
    in phase space without mutating the global state.
    """
    params_dict = {}
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        params_dict[name] = flat_params[offset:offset+numel].view_as(p)
        offset += numel
    return params_dict
