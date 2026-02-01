import torch
from dataclasses import dataclass

@dataclass
class PhysicsData:
    """Container for the spatiotemporal training set."""
    X_boundary: torch.Tensor
    Y_boundary: torch.Tensor
    X_collocation: torch.Tensor

class DataGenerator:
    """
    The Physics Data Foundry.
    
    Responsible for discretizing the spatiotemporal domain $\Omega \times [0, T]$ into 
    observation points that define the constraints for the Neural PDE Solver.
    
    Generates two distinct sets of data:
    1.  **Boundary Conditions (BC/IC):** "Hard" constraints where the solution $u(x,t)$ is known exactly.
        - Initial Condition: $u(x, 0) = -\sin(\pi x)$
        - Boundary Condition: $u(-1, t) = u(1, t) = 0$
        
    2.  **Collocation Points:** "Soft" constraints scattered across the interior domain. These are the 
        "sensors" where the physics residual $\mathcal{F}$ is minimized.
    """
    def __init__(self, num_boundary: int = 50, num_collocation: int = 2000):
        self.num_boundary = num_boundary
        self.num_collocation = num_collocation

    def generate_burgers(self) -> PhysicsData:
        """
        Synthesizes the training manifold for the Viscous Burgers' benchmark.
        
        Returns:
            PhysicsData: Structured container carrying boundary coordinates and domain collocation points.
        """
        # 1. Boundary / Initial Conditions
        # IC: t=0, u = -sin(pi*x)
        x_ic = torch.linspace(-1, 1, self.num_boundary)
        t_ic = torch.zeros_like(x_ic)
        u_ic = -torch.sin(torch.pi * x_ic)
        
        # BC: x=-1 and x=1, u = 0
        t_bc = torch.linspace(0, 1, self.num_boundary)
        x_bc_left = -torch.ones_like(t_bc)
        u_bc_left = torch.zeros_like(t_bc)
        
        x_bc_right = torch.ones_like(t_bc)
        u_bc_right = torch.zeros_like(t_bc)
        
        # Combine
        x_b = torch.cat([x_ic, x_bc_left, x_bc_right])
        t_b = torch.cat([t_ic, t_bc, t_bc])
        u_b = torch.cat([u_ic, u_bc_left, u_bc_right])
        
        X_boundary = torch.stack([x_b, t_b], dim=1)
        Y_boundary = u_b.unsqueeze(1)
        
        # 2. Collocation Points (Domain Interior)
        # Latin Hypercube Sampling (LHS) is usually better, but uniform random is fine for now.
        x_col = torch.rand(self.num_collocation) * 2 - 1 # [-1, 1]
        t_col = torch.rand(self.num_collocation) # [0, 1]
        X_collocation = torch.stack([x_col, t_col], dim=1)
        
        return PhysicsData(X_boundary, Y_boundary, X_collocation)