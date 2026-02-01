import torch
import numpy as np
import logging
from torch.autograd import grad

logger = logging.getLogger("NeuroManifold")

def hvp(potential_fn, params, v):
    """
    Computes the Hessian-Vector Product (HVP) efficiently via automatic differentiation.
    
    Mathematically, for a potential $U(\theta)$ and a vector $v$, this computes:
    $$ \nabla^2 U(\theta) \cdot v = \nabla_\theta (\nabla_\theta U(\theta) \cdot v) $$ 
    
    This operation has $O(N)$ complexity, avoiding the $O(N^2)$ storage and $O(N^3)$ computation
    required to instantiate the full Hessian matrix.
    
    Args:
        potential_fn (callable): The scalar field $U(\theta)$.
        params (torch.Tensor): The parameter vector $\theta$ at which to evaluate the curvature.
        v (torch.Tensor): The vector $v$ to multiply.
        
    Returns:
        torch.Tensor: The vector result of $H \cdot v$.
    """
    # Enable grad for HVP
    params = params.detach().requires_grad_(True)
    u = potential_fn(params)
    g = grad(u, params, create_graph=True)[0]
    return grad(torch.sum(g * v), params, retain_graph=False)[0].detach()

class LanczosApprox:
    """
    Lanczos Iteration for Low-Rank Hessian Approximation.
    
    The Lanczos algorithm constructs an orthogonal basis (Krylov subspace) that captures the 
    dominant curvature directions of the loss landscape. This allows us to approximate the 
    Hessian $H$ as:
    
    $$ H \approx V T V^T $$
    
    Where:
      - $V \in \mathbb{R}^{N \times k}$ is the matrix of Lanczos vectors (orthonormal basis).
      - $T \in \mathbb{R}^{k \times k}$ is the tridiagonal matrix of coefficients.
      
    By diagonalizing $T = Q \Lambda Q^T$, we obtain the top-$k$ eigenvalues $\Lambda$ and 
    eigenvectors $V Q$ of the full Hessian $H$.
    
    Args:
        k (int): Rank of the approximation (Krylov subspace size).
    """
    def __init__(self, k=10):
        self.k = k

    def compute(self, potential_fn, params):
        """
        Executes the Lanczos Iteration.
        
        Args:
            potential_fn (callable): The potential energy function $U(\theta)$.
            params (torch.Tensor): Current parameter state $\theta$.
            
        Returns:
            tuple: (eigenvalues, eigenvectors) of the top-$k$ curvature directions.
        """
        n = params.numel()
        k = min(self.k, n)
        
        # Storage
        V = torch.zeros((k + 1, n), device=params.device)
        alphas = torch.zeros(k, device=params.device)
        betas = torch.zeros(k, device=params.device)
        
        # Initial random vector
        v = torch.randn_like(params)
        v = v / torch.norm(v)
        V[0] = v
        
        for i in range(k):
            # HVP: w = H * v_i
            w = hvp(potential_fn, params, V[i])
            
            # Orthogonalize against previous vector (Gram-Schmidt)
            if i > 0:
                w = w - betas[i-1] * V[i-1]
            
            # Compute alpha_i = v_i^T H v_i (Diagonal element)
            alpha = torch.dot(w, V[i])
            alphas[i] = alpha
            
            # Orthogonalize against current vector
            w = w - alpha * V[i]
            
            # Compute beta_i = ||w|| (Off-diagonal element)
            if i < k - 1:
                beta = torch.norm(w)
                betas[i] = beta
                if beta < 1e-6:
                    break # Krylov subspace exhausted
                V[i+1] = w / beta
        
        # Construct Tridiagonal matrix T
        T = torch.diag(alphas) + torch.diag(betas[:-1], 1) + torch.diag(betas[:-1], -1)
        
        # Eigen decomposition of the small k x k matrix T
        eig_vals, eig_vecs_T = torch.linalg.eigh(T)
        
        # Map back to high-dimensional space: U = V_{1:k} @ Q
        # V is (k+1, n), eig_vecs_T is (k, k). We use first k Lanczos vectors.
        # Note: torch.linalg.eigh returns eigenvectors in columns.
        eig_vecs = torch.matmul(eig_vecs_T.T, V[:k])
        
        return eig_vals, eig_vecs

class LowRankMassMatrix:
    """
    Riemannian Metric Tensor based on Low-Rank Hessian Approximation.
    
    Implements a "SoftAbs" metric (Betancourt, 2013) generalized for low-rank structures.
    The goal is to precondition the Hamiltonian Dynamics by "straightening out" the 
    high-curvature directions (eigenvectors of the Hessian).
    
    The inverse metric tensor $M^{-1}$ is constructed as:
    
    $$ M^{-1} = V \left( |\Lambda|^{-1} - \alpha^{-1} I \right) V^T + \alpha^{-1} I $$
    
    Where:
      - $V, \Lambda$ are the top-$k$ eigenpairs of the Hessian.
      - $\alpha$ is a regularization term (representing the isotropic curvature of the null space).
      - $|\Lambda|$ denotes the element-wise absolute value (SoftAbs) to ensure positive definiteness.
    
    This allows us to simulate dynamics on the Riemannian Manifold induced by the loss landscape.
    """
    def __init__(self, rank=10, alpha=1.0):
        self.rank = rank
        self.alpha = alpha # Regularization (diagonal term)
        self.lanczos = LanczosApprox(k=rank)
        
        self.eig_vals = None
        self.eig_vecs = None # Shape (k, n)
        self.metric_vals = None # |lambda| + alpha

    def update(self, potential_fn, params):
        """
        Recomputes the local metric tensor at the current position in phase space.
        This is typically done periodically during the burn-in phase.
        """
        logger.info("   [Riemannian] Updating Metric Tensor via Lanczos...")
        vals, vecs = self.lanczos.compute(potential_fn, params)
        
        # SoftAbs: Take absolute values + regularization to ensure positive definite Metric
        # M_eigenvalues = abs(lambda) + alpha
        self.eig_vals = vals
        self.eig_vecs = vecs
        self.metric_vals = torch.abs(vals) + self.alpha
        
        logger.info(f"   [Riemannian] Top Eigenspectrum (Curvature): {vals[-5:].cpu().numpy()}")

    def kinetic_energy(self, momentum):
        """
        Computes the Riemannian Kinetic Energy:
        $$ K(p) = \frac{1}{2} p^T M^{-1} p $$
        
        Utilizes the Woodbury Matrix Identity structure to compute this in $O(Nk) $ 
        instead of $O(N^2)$.
        """
        if self.eig_vecs is None:
            # Euclidean Geometry fallback
            return 0.5 * torch.sum(momentum ** 2)
            
        # Term 1: Isotropic contribution (alpha^{-1} * I)
        term1 = torch.sum(momentum ** 2) / self.alpha
        
        # Term 2: Low-rank correction along curvature directions
        # p_proj = V @ p (Result is size k)
        p_proj = torch.mv(self.eig_vecs, momentum) 
        
        inv_correction = (1.0 / self.metric_vals) - (1.0 / self.alpha)
        term2 = torch.sum((p_proj ** 2) * inv_correction)
        
        return 0.5 * (term1 + term2)

    def sample_momentum(self, params):
        """
        Samples momentum from the conditional Gaussian distribution defined by the Metric:
        $$ p \sim \mathcal{N}(0, M) $$
        
        Generative Process:
        1. Sample $z \sim \mathcal{N}(0, I)$
        2. Transform $p = M^{1/2} z$
        
        Where $M^{1/2} = V (\sqrt{|\Lambda|} - \sqrt{\alpha}) V^T + \sqrt{\alpha} I$.
        """
        z = torch.randn_like(params)
        
        if self.eig_vecs is None:
            return z
            
        sqrt_alpha = np.sqrt(self.alpha)
        
        # Term 1: Isotropic component
        term1 = sqrt_alpha * z
        
        # Term 2: Curvature correction
        z_proj = torch.mv(self.eig_vecs, z)
        correction = torch.sqrt(self.metric_vals) - sqrt_alpha
        
        # Back project: V^T @ (correction * z_proj)
        term2 = torch.matmul(z_proj * correction, self.eig_vecs)
        
        return term1 + term2

class NUTSSampler:
    """
    Riemannian No-U-Turn Sampler (NUTS).
    
    The "Gold Standard" of MCMC algorithms, adapted for High-Dimensional Neural Landscapes.
    
    ## Theoretical Foundation
    NUTS eliminates the need to manually tune the trajectory length $L$ (or `num_steps`) in HMC.
    Instead of a fixed path, it recursively builds a binary tree of Leapfrog steps that expands 
    forward and backward in time until the trajectory makes a "U-Turn" (i.e., the distance between 
    the start and end points stops increasing).
    
    ## Key Features
    1.  **Riemannian Manifold Hamiltonian Monte Carlo (RMHMC):**
        Uses the `LowRankMassMatrix` to adapt the symplectic integration to the local geometry 
        of the loss surface. This allows the sampler to traverse "narrow canyons" (high curvature) 
        and "flat plains" (low curvature) with equal efficiency.
        
    2.  **Dual Averaging Step Size Adaptation:**
        Automatically tunes the integrator step size $\epsilon$ during burn-in to achieve a 
        target acceptance statistic (typically $\delta = 0.8$), using Nesterov's Primal-Dual 
        Averaging scheme.
        
    3.  **Hoffman-Gelman Tree Building:**
        Constructs a trajectory that satisfies Detailed Balance (Reversibility) without fixing 
        integration time a priori.
    """
    def __init__(self, potential_fn, step_size=1e-3, max_tree_depth=10, adapt_mass_matrix=True):
        self.potential_fn = potential_fn
        self.step_size = step_size
        self.max_tree_depth = max_tree_depth
        self.mass_matrix = LowRankMassMatrix(rank=10, alpha=1.0)
        self.adapt_mass = adapt_mass_matrix

    def _compute_potential_and_grad(self, params):
        params = params.detach().requires_grad_(True)
        u = self.potential_fn(params)
        g = grad(u, params)[0]
        return u.detach(), g.detach()

    def sample(self, current_params, num_samples=100, burn_in=50):
        """
        Executes the NUTS sampling loop.
        
        Args:
            current_params (torch.Tensor): Initial position $\theta_0$.
            num_samples (int): Number of valid posterior samples to collect.
            burn_in (int): Number of adaptation steps (discarded).
            
        Returns:
            torch.Tensor: A tensor of shape (num_samples, num_params) containing the posterior chain.
        """
        samples = []
        params = current_params.clone()
        
        # Initial gradient and potential
        current_u, current_grad = self._compute_potential_and_grad(params)
        
        # Dual Averaging state initialization (Nesterov's method)
        mu = np.log(10 * self.step_size)
        log_step = np.log(self.step_size)
        log_step_bar = np.log(1.0)
        H_bar = 0
        gamma = 0.05
        t0 = 10
        kappa = 0.75
        
        # Initialize Metric
        if self.adapt_mass:
            self.mass_matrix.update(self.potential_fn, params)

        logger.info(f"   [NUTS] Starting Sampling: {num_samples} samples, {burn_in} burn-in")

        for m in range(1, num_samples + burn_in + 1):
            # 1. Resample Momentum p ~ N(0, M)
            r0 = self.mass_matrix.sample_momentum(params)
            
            # Joint Energy H(theta, p) = U(theta) + K(p)
            kinetic = self.mass_matrix.kinetic_energy(r0)
            H0 = current_u + kinetic
            
            # Slice Sampling Variable u ~ Uniform([0, exp(-H)])
            # We work in log-space: log_u = log(Uniform) - H
            log_u = torch.log(torch.rand(1)) - H0
            
            # Initialize Tree State
            params_minus = params.clone()
            params_plus = params.clone()
            r_minus = r0.clone()
            r_plus = r0.clone()
            
            grad_minus = current_grad.clone()
            grad_plus = current_grad.clone()
            
            j = 0 # Current Tree Depth
            n = 1 # Number of valid nodes in the tree
            s = 1 # Stop condition (1 = Continue, 0 = Stop)
            
            params_next = params.clone()
            
            while s == 1 and j < self.max_tree_depth:
                # Choose direction: -1 (backward in time) or +1 (forward in time)
                v = int(2 * (torch.rand(1) < 0.5) - 1)
                
                if v == -1:
                    params_minus, r_minus, grad_minus, _, _, _, params_prime, n_prime, s_prime, alpha, n_alpha = \
                        self._build_tree(params_minus, r_minus, grad_minus, log_u, v, j, self.step_size, H0)
                else:
                    _, _, _, params_plus, r_plus, grad_plus, params_prime, n_prime, s_prime, alpha, n_alpha = \
                        self._build_tree(params_plus, r_plus, grad_plus, log_u, v, j, self.step_size, H0)
                
                # Metropolis-Hastings acceptance step for the subtree
                if s_prime == 1:
                    if torch.rand(1) < float(n_prime) / n:
                        params_next = params_prime.clone()
                        
                n = n + n_prime
                
                # Update Stop Condition (UTurn check)
                s = s_prime and self._stop_criterion(params_minus, params_plus, r_minus, r_plus)
                j = j + 1
                
                # Dual Averaging Adaptation (during burn-in)
                if m <= burn_in:
                    eta = 1.0 / (m + t0)
                    avg_accept_prob = alpha / n_alpha if n_alpha > 0 else 0
                    H_bar = (1 - eta) * H_bar + eta * (0.8 - avg_accept_prob) # Target alpha = 0.8
                    
                    log_step = mu - (np.sqrt(m) / gamma) * H_bar
                    m_power = m ** -kappa
                    log_step_bar = m_power * log_step + (1 - m_power) * log_step_bar
                    self.step_size = np.exp(log_step)
            
            # Update state for next iteration
            params = params_next
            current_u, current_grad = self._compute_potential_and_grad(params)
            
            # ---------------------------------------------------------
            # Burn-in Management
            # ---------------------------------------------------------
            if m <= burn_in:
                if m == burn_in:
                    self.step_size = np.exp(log_step_bar)
                    logger.info(f"   [NUTS] Burn-in Complete. Final Step Size: {self.step_size:.2e}")
                
                # Periodically update Riemannian Metric
                # We stop updating slightly before burn-in ends to let the chain stabilize
                if self.adapt_mass and m % 20 == 0 and m < burn_in * 0.9:
                     self.mass_matrix.update(self.potential_fn, params)
            else:
                samples.append(params.detach().cpu())
            
            if m % 10 == 0:
                logger.info(f"Sample {m}/{num_samples+burn_in} | Depth: {j} | Step: {self.step_size:.2e}")

        return torch.stack(samples)

    def _stop_criterion(self, theta_minus, theta_plus, r_minus, r_plus):
        """
        The "No-U-Turn" criterion. 
        Checks if the trajectory is starting to turn back on itself.
        
        $$ \Delta \theta \cdot r_- \ge 0 \quad \text{AND} \quad \Delta \theta \cdot r_+ \ge 0 $$
        """
        d_theta = theta_plus - theta_minus
        return (torch.dot(d_theta, r_minus) >= 0) and (torch.dot(d_theta, r_plus) >= 0)

    def _leapfrog(self, params, r, grad_u, epsilon):
        """
        Symplectic Integrator Step (Velocity Verlet).
        
        Updates (q, p) -> (q', p') while preserving volume in phase space.
        With Riemannian Metric, the position update becomes:
        
        $$ \theta_{t+1} = \theta_t + \epsilon M^{-1} p_{t+1/2} $$
        """
        # Half step momentum (Kick)
        r = r - 0.5 * epsilon * grad_u
        
        # Full step position (Drift)
        # We need M^{-1} r. Using autograd on kinetic energy is a robust way to get this
        # since grad_p(0.5 p^T M^{-1} p) = M^{-1} p
        r_temp = r.detach().requires_grad_(True)
        k = self.mass_matrix.kinetic_energy(r_temp)
        v = grad(k, r_temp)[0] 
        
        params = params + epsilon * v
        
        # Re-evaluate gradient at new position
        u, new_grad = self._compute_potential_and_grad(params)
        
        # Half step momentum (Kick)
        r = r - 0.5 * epsilon * new_grad
        
        return params, r, new_grad, u

    def _build_tree(self, params, r, g, log_u, v, j, epsilon, H0):
        """
        Recursively builds the NUTS trajectory binary tree. 
        
        Returns a tuple containing:
        - params_minus, r_minus, g_minus: State at the backward edge of the tree.
        - params_plus, r_plus, g_plus: State at the forward edge of the tree.
        - params_prime: The proposal state sampled from the subtree.
        - n_prime: Number of valid nodes in the subtree.
        - s_prime: Stop condition for the subtree.
        - alpha_prime: Sum of acceptance probabilities (for Dual Averaging).
        - n_alpha_prime: Count of acceptance probabilities.
        """
        if j == 0:
            # Base case: Take a single Leapfrog step
            params_prime, r_prime, g_prime, u_prime = self._leapfrog(params, r, g, v * epsilon)
            
            # Energy of new state
            kinetic = self.mass_matrix.kinetic_energy(r_prime)
            H_prime = u_prime + kinetic
            
            n_prime = int(log_u <= -H_prime) # Valid node check
            s_prime = int(log_u < -H_prime + 1000) # Divergence check
            
            # Metropolis acceptance probability for this leaf
            alpha_prime = min(1.0, torch.exp(H0 - H_prime).item())
            
            return params_prime, r_prime, g_prime, params_prime, r_prime, g_prime, params_prime, n_prime, s_prime, alpha_prime, 1

        else:
            # Recursion: Build left/right subtrees
            params_minus, r_minus, g_minus, params_plus, r_plus, g_plus, params_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self._build_tree(params, r, g, log_u, v, j - 1, epsilon, H0)
                
            if s_prime == 1:
                if v == -1:
                    params_minus, r_minus, g_minus, _, _, _, params_prime2, n_prime2, s_prime2, alpha_prime2, n_alpha_prime2 = \
                        self._build_tree(params_minus, r_minus, g_minus, log_u, v, j - 1, epsilon, H0)
                else:
                    _, _, _, params_plus, r_plus, g_plus, params_prime2, n_prime2, s_prime2, alpha_prime2, n_alpha_prime2 = \
                        self._build_tree(params_plus, r_plus, g_plus, log_u, v, j - 1, epsilon, H0)
                
                # Progressive Sampling: Choose between left/right subtree proposals
                if torch.rand(1) < float(n_prime2) / max(1, (n_prime + n_prime2)):
                    params_prime = params_prime2
                
                # Accumulate statistics
                alpha_prime = alpha_prime + alpha_prime2
                n_alpha_prime = n_alpha_prime + n_alpha_prime2
                s_prime = s_prime and s_prime2 and self._stop_criterion(params_minus, params_plus, r_minus, r_plus)
                n_prime = n_prime + n_prime2
            
            return params_minus, r_minus, g_minus, params_plus, r_plus, g_plus, params_prime, n_prime, s_prime, alpha_prime, n_alpha_prime