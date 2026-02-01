import torch
from torch.func import grad, vmap, functional_call

class Burgers1D:
    """
    Defines the physical laws governing the 1D Viscous Burgers' Equation.
    
    The PDE acts as the "Likelihood Function" in our Bayesian setting. It is given by:
    $$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0 $$
    
    Where:
      - $u(x,t)$: Velocity field (The State Variable).
      - $\nu$ (nu): Kinematic viscosity (diffusion coefficient), controlling shock thickness.
      
    **Computational Graph Implementation:**
    Uses `torch.func` (Jacobian/Hessian vector products) to compute **exact** derivatives of the 
    neural network $\hat{u}$ with respect to inputs $(x, t)$. This avoids the memory overhead of 
    standard `torch.autograd.grad` (create_graph=True) and allows for efficient vectorization (`vmap`)
    over the collocation batch.
    """
    def __init__(self, nu=0.01/torch.pi):
        self.nu = nu

    def physics_residual(self, model, params, x_t):
        """
        Computes the PDE residual $\mathcal{F}(x, t) = u_t + u u_x - \nu u_{xx}$.
        
        In the Bayesian interpretation, minimizing the norm of this residual maximizes the 
        posterior probability $p(\theta | \text{Physics})$.
        
        Args:
            model (nn.Module): The stateless PINN model instance.
            params (dict): The parameters $\theta$ of the model (weights/biases).
            x_t (torch.Tensor): Spatiotemporal coordinates $(x, t)$.
            
        Returns:
            torch.Tensor: The PDE error residual at each point in the batch.
        """
        # We need first and second derivatives of u w.r.t x and t.
        # We use torch.func.grad to get derivatives w.r.t inputs.
        
        def u_fn(x_t_single):
            # Helper to call model with params on a single input
            # Output must be scalar for grad
            out = functional_call(model, params, (x_t_single.unsqueeze(0),))
            return out.squeeze()

        # First derivatives (Gradient of u w.r.t x and t)
        # grad(u_fn) returns a vector of size 2 (du/dx, du/dt)
        d_u = grad(u_fn)
        
        # Second derivatives (Hessian of u w.r.t x and t)
        # hessian(u_fn) returns 2x2 matrix
        from torch.func import hessian
        d2_u = hessian(u_fn)

        # We need to vectorize this over the batch of points
        # vmap transforms the single-point function to a batch function
        d_u_vmap = vmap(d_u)
        d2_u_vmap = vmap(d2_u)
        
        u_val = functional_call(model, params, (x_t,))
        du_val = d_u_vmap(x_t)
        d2u_val = d2_u_vmap(x_t)
        
        u = u_val.squeeze()
        u_x = du_val[:, 0]
        u_t = du_val[:, 1]
        u_xx = d2u_val[:, 0, 0]
        
        # Burgers' Equation Residual
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss(self, model, params, data_boundary, data_collocation):
        """
        Computes the Total Potential Energy $U(\theta)$ (Negative Log Posterior).
        
        $$ U(\theta) = \lambda_b \mathcal{L}_{boundary} + \lambda_f \mathcal{L}_{physics} $$
        
        Valleys in this potential landscape correspond to valid physical solutions.
        """
        # 1. Boundary/Initial Conditions Loss
        x_b, y_b = data_boundary
        y_pred_b = functional_call(model, params, (x_b,))
        loss_b = torch.mean((y_pred_b - y_b) ** 2)
        
        # 2. Physics Residual Loss (Collocation points)
        # Since we enforce f=0, the target is 0.
        f_pred = self.physics_residual(model, params, data_collocation)
        loss_f = torch.mean(f_pred ** 2)
        
        # Total Loss
        return loss_b + loss_f