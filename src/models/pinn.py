import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    A Physics-Informed Neural Network (PINN) architecture serving as the parametric approximation ansatz.
    
    This network acts as a function approximator $\hat{u}(x, t; \theta)$ mapping the spatiotemporal 
    domain $\Omega 	imes [0, T]$ to the solution manifold of the PDE.
    
    **Geometric Perspective:**
    The vector of weights and biases $\theta 
    \in 
    \mathbb{R}^N$ defines a coordinate system on the 
    statistical manifold of hypothesis functions. The MCMC sampler traverses this specific manifold, 
    guided by the curvature induced by the PDE constraints.
    
    **Mathematical Properties:**
      - **Differentiability ($C^\infty$):** Uses `Tanh` activation functions to ensure the existence 
        of non-trivial higher-order derivatives (Hessians) required for computing physics residuals 
        (e.g., $\partial^2 u / \partial x^2$). ReLU is avoided as its second derivative is zero almost everywhere.
      - **Universal Approximation:** Guarantees that with sufficient width/depth, there exists a 
        $\theta^*$ such that $||
        \hat{u} - u_{exact}|| < \epsilon$.
    """
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_layers=4):
        """
        Initialize the Deep Neural Network Ansatz.

        Args:
            input_dim (int): Dimensionality of the domain (e.g., 2 for $\mathbb{R}^2 
            \ni (x, t)$).
            hidden_dim (int): Width of the hidden layers (neurons).
            output_dim (int): Dimensionality of the codomain (solution variables).
            num_layers (int): Depth of the network (number of hidden transformations).
        """
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot initialization for better convergence
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

    def get_flat_params(self):
        """Returns all parameters as a single flat vector (for MCMC state)."""
        return torch.cat([p.view(-1) for p in self.parameters()])

    def load_flat_params(self, flat_params):
        """Loads parameters from a single flat vector (for MCMC state restoration)."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view_as(p))
            offset += numel