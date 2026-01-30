import torch
import torch.optim as optim
import logging
from omegaconf import DictConfig
from src.models.pinn import PINN
from src.physics.burgers import Burgers1D
from src.mcmc.hmc import HMCSampler
from src.data import DataGenerator
from src.validator import Validator
from src.utils import get_flat_params, unflatten_params

# Configure Logging to look "Pro"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("NeuroManifold")

class BayesianPipeline:
    """
    The High-Fidelity Orchestrator.
    
    Manages the lifecycle of the Bayesian Physics Inversion process. This pipeline coordinates
    the interaction between the Differential Geometry engine (HMC), the Physics laws (Burgers), 
    and the Neural Approximation (PINN).
    
    **Lifecycle Phases:**
      1. **Initialization:** Sets up the manifold geometry and random seeds.
      2. **Phase 1 (Data Synthesis):** Generates the boundary and collocation constraints.
      3. **Phase 2 (MAP Optimization):** Uses `Adam` to rapidly descend to the "Typical Set" 
         (the region of high probability mass), providing a warm start for the sampler.
      4. **Phase 3 (HMC Sampling):** The "Gold Standard" phase. A particle explores the posterior 
         distribution using Hamiltonian Dynamics, collecting uncorrelated samples.
      5. **Phase 4 (Validation):** Computes diagnostics (ESS, Trace Plots) and quantifies uncertainty.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        torch.manual_seed(self.cfg.seed)
        
        # Initialize Components
        logger.info("Initializing NeuroManifold Engine (Hydra Powered)...")
        self.model = PINN(
            hidden_dim=self.cfg.model.hidden_dim, 
            num_layers=self.cfg.model.num_layers
        )
        self.physics = Burgers1D()
        self.generator = DataGenerator()
        
        # Validator outputs to the current working directory (managed by Hydra)
        self.validator = Validator(self.model, output_dir=".")
        
        # Placeholders for data
        self.data = None

    def run(self):
        """
        Executes the Master Pipeline.
        """
        logger.info(f">>> Starting Pipeline Execution [Step Size: {self.cfg.hmc.step_size}]")
        
        # 1. Data Generation
        self._generate_data()
        
        # 2. Pre-training (MAP)
        self._pretrain_map()
        
        # 3. HMC Sampling
        samples = self._run_hmc()
        
        # 4. Validation & Visualization
        self.validator.plot_uncertainty(samples)
        self.validator.compute_diagnostics(samples)
        
        logger.info(">>> Pipeline Complete. Shutting down.")

    def _generate_data(self):
        logger.info("Phase 1: Generating 5-D Exciter Data (Synthetic Physics)...")
        self.data = self.generator.generate_burgers()
        logger.info(f"   Boundary Points: {len(self.data.X_boundary)}")
        logger.info(f"   Collocation Points: {len(self.data.X_collocation)}")

    def _pretrain_map(self):
        logger.info("Phase 2: MAP Optimization (Finding the Typical Set)...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        
        for epoch in range(self.cfg.train.pretrain_epochs):
            optimizer.zero_grad()
            # Standard loss calculation
            params_dict = dict(self.model.named_parameters())
            loss = self.physics.loss(
                self.model, 
                params_dict, 
                (self.data.X_boundary, self.data.Y_boundary), 
                self.data.X_collocation
            )
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.6f}")
        
        logger.info(f"   Pre-training converged. Final Loss: {loss.item():.6f}")

    def _run_hmc(self):
        logger.info("Phase 3: Hamiltonian Monte Carlo (The Rolls Royce Sampler)...")
        
        # Define Potential Energy U(theta)
        def potential_fn(flat_params):
            params_dict = unflatten_params(self.model, flat_params)
            loss_val = self.physics.loss(
                self.model, 
                params_dict, 
                (self.data.X_boundary, self.data.Y_boundary), 
                self.data.X_collocation
            )
            return loss_val * self.cfg.hmc.beta

        sampler = HMCSampler(
            potential_fn, 
            step_size=self.cfg.hmc.step_size, 
            num_steps=self.cfg.hmc.num_steps
        )
        
        current_flat_params = get_flat_params(self.model)
        
        logger.info(f"   Starting Chain: {self.cfg.hmc.num_samples} Samples + {self.cfg.hmc.burn_in} Burn-in")
        samples = sampler.sample(
            current_flat_params, 
            num_samples=self.cfg.hmc.num_samples, 
            burn_in=self.cfg.hmc.burn_in, 
            adapt_mass_matrix=self.cfg.hmc.adapt_mass_matrix
        )
        logger.info(f"   Sampling Complete. Posterior Shape: {samples.shape}")
        return samples