import torch
import numpy as np
from torch.func import grad, vmap

class Hamiltonian:
    """
    Defines the physical energy system of the probability space: $H(\theta, p) = U(\theta) + K(p)$.
    
    In the context of Bayesian Inference on Riemannian Manifolds, the Hamiltonian represents the 
    total energy of a particle moving across the high-dimensional loss landscape.
    
    Components:
      - **Potential Energy $U(\theta)$**: Corresponds to the Negative Log Posterior (Physics Loss). 
        Valleys in the landscape correspond to high-probability regions (valid physics solutions).
      - **Kinetic Energy $K(p)$**: Represents the momentum of the particle. 
        $K(p) = \frac{1}{2} p^T M^{-1} p$, where $M$ is the Mass Matrix (Metric Tensor).
        
    This encapsulation allows for the separation of conservative dynamics from the specific 
    geometry of the parameter space.
    """
    def __init__(self, potential_fn, mass_matrix_diag=None):
        """
        Initialize the Hamiltonian System.

        Args:
            potential_fn (Callable): The scalar field $U(\theta)$ representing the geometry of the target distribution.
            mass_matrix_diag (torch.Tensor, optional): The diagonal of the Inverse Metric Tensor $M^{-1}$. 
                                                       If None, Euclidean geometry (Identity Metric) is assumed.
        """
        self.potential_fn = potential_fn
        self.mass_matrix_diag = mass_matrix_diag

    def potential(self, params):
        return self.potential_fn(params)

    def kinetic(self, momentum):
        """
        Computes Kinetic Energy: $K(p) = \frac{1}{2} p^T M^{-1} p$.
        """
        if self.mass_matrix_diag is None:
            return 0.5 * torch.sum(momentum ** 2)
        else:
            return 0.5 * torch.sum(momentum ** 2 / self.mass_matrix_diag)

    def hamiltonian(self, params, momentum):
        """Total Energy $H = U + K$. Preserved by symplectic integration."""
        return self.potential(params) + self.kinetic(momentum)

    def grad_potential(self, params):
        """
        Computes the Conservative Force field: $\nabla_\theta U(\theta)$.
        This drives the particle towards high-probability regions.
        """
        return grad(self.potential_fn)(params)


class LeapfrogIntegrator:
    """
    A Symplectic Numerical Solver for Hamilton's Equations of Motion.
    
    Unlike standard integrators (Euler, Runge-Kutta), the Leapfrog integrator is **Symplectic**,
    meaning it strictly preserves the symplectic 2-form $dp \wedge dq$. 
    
    Key Properties:
      1. **Volume Preservation:** Liouville's Theorem holds (no shrinking/expanding of phase space).
      2. **Time Reversibility:** The map $T(t) \to T(t+dt)$ is exactly reversible.
      3. **Energy Stability:** No long-term energy drift, allowing for valid MCMC sampling even 
         with large step sizes.
         
    The update scheme is a Strang Splitting of the Hamiltonian flow:
    1. Kick (Momentum): $p_{1/2} = p_0 - \frac{\epsilon}{2} \nabla U(\theta_0)$
    2. Drift (Position): $\theta_1 = \theta_0 + \epsilon M^{-1} p_{1/2}$
    3. Kick (Momentum): $p_1 = p_{1/2} - \frac{\epsilon}{2} \nabla U(\theta_1)$
    """
    def __init__(self, hamiltonian, step_size, num_steps):
        self.H = hamiltonian
        self.step_size = step_size
        self.num_steps = num_steps

    def step(self, params, momentum):
        """
        Evolves the system through Phase Space for a fixed trajectory length.
        
        Args:
            params (torch.Tensor): Initial coordinates $\theta(t)$.
            momentum (torch.Tensor): Initial momentum $p(t)$.
            
        Returns:
            tuple: $(\theta(t+\tau), p(t+\tau))$ after `num_steps` of symplectic evolution.
        """
        dt = self.step_size
        
        # Half step for momentum (Kick)
        grad_u = self.H.grad_potential(params)
        momentum = momentum - 0.5 * dt * grad_u
        
        # Full steps for position and momentum (Drift + Kick)
        for i in range(self.num_steps):
            # Full step for position
            if self.H.mass_matrix_diag is None:
                params = params + dt * momentum
            else:
                params = params + dt * (momentum / self.H.mass_matrix_diag)
            
            # Recalculate Gradient at new position
            grad_u = self.H.grad_potential(params)
            
            # Full step for momentum (except last step)
            if i != self.num_steps - 1:
                momentum = momentum - dt * grad_u
        
        # Final Half step for momentum (Kick)
        momentum = momentum - 0.5 * dt * grad_u
        
        return params, momentum

class DualAveragingStepSize:
    """
    Nesterov's Dual Averaging algorithm for adaptive step size tuning.
    Targets a specific acceptance rate (usually 0.8) during burn-in.
    """
    def __init__(self, initial_step_size, target_accept=0.8, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_step_size = np.log(initial_step_size)
        self.log_step_size_bar = self.log_step_size
        self.m = 1

    def update(self, accept_prob):
        # H_m = (1 - 1/(m+t0)) H_{m-1} + 1/(m+t0) (delta - alpha)
        eta = 1.0 / (self.m + self.t0)
        self.error_sum = (1 - eta) * self.error_sum + eta * (self.target_accept - accept_prob)
        
        # log epsilon_m = mu - sqrt(m)/gamma * H_m
        self.log_step_size = self.mu - (np.sqrt(self.m) / self.gamma) * self.error_sum
        
        # log epsilon_bar_m = m^{-kappa} log epsilon_m + (1 - m^{-kappa}) log epsilon_bar_{m-1}
        m_power = self.m ** -self.kappa
        self.log_step_size_bar = m_power * self.log_step_size + (1 - m_power) * self.log_step_size_bar
        
        self.m += 1
        return np.exp(self.log_step_size)
    
    def get_final_epsilon(self):
        return np.exp(self.log_step_size_bar)

class HMCSampler:
    """
    The High-Fidelity "Rolls Royce" of MCMC Samplers.
    
    Utilizes Hamiltonian Dynamics to propose distant, uncorrelated samples with high acceptance probabilities.
    
    Algorithm:
    1. **Gibbs Sampling of Momentum:** Sample $p \sim \mathcal{N}(0, M)$.
    2. **Hamiltonian Dynamics:** Simulate the particle trajectory for time $\tau = L \epsilon$.
    3. **Metropolis-Hastings Correction:** Accept/Reject based on Energy Error $\Delta H$.
       Because the integrator is symplectic, $\Delta H \approx 0$ and Acceptance $\approx 100\%$.
    
    Features:
    - **Riemannian Adaptation:** Can adapt the Mass Matrix $M$ (Metric Tensor) during burn-in to 
      normalize the curvature of the loss landscape (preconditioning).
    - **Dual Averaging Step Size:** Implements Nesterov's Dual Averaging to automatically tune 
      the step size $\epsilon$ during burn-in, targeting an optimal acceptance rate (typically 0.8).
    """
    def __init__(self, potential_fn, step_size=1e-3, num_steps=10, mass_matrix_diag=None):
        self.H = Hamiltonian(potential_fn, mass_matrix_diag)
        self.integrator = LeapfrogIntegrator(self.H, step_size, num_steps)
        self.initial_step_size = step_size
        
    def sample(self, current_params, num_samples=100, burn_in=0, adapt_mass_matrix=False):
        samples = []
        accepted = 0
        
        # We assume params is a dictionary (for functional call)
        # But our integrator works best with flat tensors. 
        # For now, let's assume current_params is FLATTENED tensor.
        
        params = current_params.clone()
        
        # Adaptation buffer
        if adapt_mass_matrix:
            burn_in_samples = []
            
        # Step Size Adaptation
        step_size_adapter = DualAveragingStepSize(self.initial_step_size)
        
        for i in range(num_samples + burn_in):
            # 1. Resample Momentum
            momentum = self._sample_momentum(params)
            
            # 2. Simulate Dynamics
            current_H = self.H.hamiltonian(params, momentum)
            new_params, new_momentum = self.integrator.step(params, momentum)
            
            # 3. Metropolis Correction
            # Note: We negate new_momentum because the proposal is symmetric 
            # only if we flip momentum, though Hamiltonian is even in p so H(p) = H(-p).
            proposed_H = self.H.hamiltonian(new_params, new_momentum)
            
            # Acceptance Probability = min(1, exp(H_old - H_new))
            # (Note: Hamiltonian is Energy, so Prob ~ exp(-H))
            log_accept_prob = current_H - proposed_H
            accept_prob = torch.exp(torch.min(torch.tensor(0.0), log_accept_prob)).item()
            
            if torch.rand(1) < accept_prob:
                params = new_params
                accepted += 1
            
            # Adaptation Step (during burn-in)
            if i < burn_in:
                # Step Size Adaptation
                new_step_size = step_size_adapter.update(accept_prob)
                self.integrator.step_size = new_step_size
                
                if adapt_mass_matrix:
                    burn_in_samples.append(params.detach())
                    # Update Mass Matrix every 20 steps if we have enough samples
                    # Increased window for stability
                    if i > 20 and i % 20 == 0:
                        stacked_burn = torch.stack(burn_in_samples)
                        # Variance of parameters + small jitter for stability
                        var = torch.var(stacked_burn, dim=0) + 1e-5
                        # Inverse variance is the diagonal mass matrix (Metric)
                        self.H.mass_matrix_diag = 1.0 / var
            elif i == burn_in:
                # End of burn-in: Lock in the averaged step size
                final_step_size = step_size_adapter.get_final_epsilon()
                self.integrator.step_size = final_step_size
                print(f"   >>> Burn-in Complete. Final Step Size: {final_step_size:.2e}")
            
            if i >= burn_in:
                samples.append(params.detach().cpu())
                
            if i % 10 == 0:
                print(f"Sample {i}/{num_samples+burn_in} | Accept Prob: {accept_prob:.2f} | Step: {self.integrator.step_size:.2e} | Energy: {current_H.item():.4f}")
                
        return torch.stack(samples)

    def _sample_momentum(self, params):
        """Samples momentum from N(0, M)."""
        if self.H.mass_matrix_diag is None:
            return torch.randn_like(params)
        else:
            return torch.randn_like(params) * torch.sqrt(self.H.mass_matrix_diag)
