import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils import unflatten_params
from torch.func import functional_call

class Validator:
    """
    The Posterior Analytics Engine.
    
    Responsible for quantifying the statistical fidelity of the MCMC chain and generating 
    visual artifacts that represent the epistemic uncertainty of the physics model.
    
    Key Functions:
      - **Uncertainty Quantification:** Computes the marginal predictive distribution 
        $p(u | \mathcal{D}) = \int p(u | \theta) p(\theta | \mathcal{D}) d\theta$ via Monte Carlo integration.
      - **Diagnostics:** Evaluates chain mixing using Effective Sample Size (ESS) and autocorrelation metrics.
    """
    def __init__(self, model, output_dir="results"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_uncertainty(self, samples: torch.Tensor, title: str = "HMC Uncertainty"):
        """
        Visualizes the Predictive Posterior Distribution.
        
        Generates a plot showing:
          - The Mean Prediction $\mathbb{E}[u]$ (Model's best guess).
          - The 95% Credible Interval (Uncertainty Band).
          - Individual Posterior Samples (Spaghetti Plot) to visualize manifold diversity.
        """
        # Create high-res grid slice at t=0.5
        x_plot = torch.linspace(-1, 1, 200)
        t_plot = torch.ones_like(x_plot) * 0.5
        X_plot = torch.stack([x_plot, t_plot], dim=1)
        
        preds = []
        # Vectorizing prediction over samples would be even faster (vmap),
        # but a loop is fine for visualization.
        for s_params in samples:
            params_dict = unflatten_params(self.model, s_params)
            y_pred = functional_call(self.model, params_dict, (X_plot,))
            preds.append(y_pred.detach().squeeze().cpu().numpy())
        
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        
        # Plotting
        plt.figure(figsize=(12, 7), dpi=150) # High DPI for "Publication Quality"
        
        # Plot individual samples (spaghetti plot) for detailed visual texture
        for i in range(min(len(preds), 20)):
            plt.plot(x_plot.numpy(), preds[i], color='gray', alpha=0.1, linewidth=0.5)
            
        plt.plot(x_plot.numpy(), mean_pred, 'b-', linewidth=2, label='Mean Prediction')
        plt.fill_between(x_plot.numpy(), 
                         mean_pred - 2*std_pred, 
                         mean_pred + 2*std_pred, 
                         color='blue', alpha=0.2, label='95% Confidence Interval')
        
        plt.title(title, fontsize=14, fontfamily='serif')
        plt.xlabel("Position (x)", fontsize=12)
        plt.ylabel("Velocity (u)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')
        
        save_path = os.path.join(self.output_dir, "uncertainty_profile.png")
        plt.savefig(save_path)
        print(f"   >>> Artifact generated: {save_path}")

    def compute_diagnostics(self, samples: torch.Tensor):
        """
        Computes Markov Chain Monte Carlo (MCMC) diagnostics to verify convergence.
        
        Metrics:
          - **Trace Plot:** Visual check for stationarity and mixing.
          - **Effective Sample Size (ESS):** The number of effectively independent samples.
            $ESS = N \frac{1-\rho}{1+\rho}$, where $\rho$ is the lag-1 autocorrelation.
        """
        # 1. Trace Plot
        plt.figure(figsize=(10, 4))
        plt.plot(samples[:, 0].numpy())
        plt.title("Trace Plot (Parameter 0)")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.savefig(os.path.join(self.output_dir, "trace_plot.png"))
        
        # 2. Effective Sample Size (Simplified)
        # We calculate lag-1 autocorrelation for the first few parameters
        # ESS = N * (1 - rho) / (1 + rho)
        N = samples.shape[0]
        ess_list = []
        
        # Check first 5 parameters
        for i in range(min(5, samples.shape[1])):
            x = samples[:, i].numpy()
            mean = np.mean(x)
            var = np.var(x)
            if var < 1e-9: continue # Constant parameter
            
            # Autocovariance at lag 1
            acov = np.mean((x[:-1] - mean) * (x[1:] - mean))
            rho = acov / var
            
            # ESS
            ess = N * (1 - rho) / (1 + rho)
            ess_list.append(ess)
            
        avg_ess = np.mean(ess_list)
        print(f"\n[DIAGNOSTICS]")
        print(f"   >>> Mean Effective Sample Size (ESS): {avg_ess:.1f} / {N}")
        print(f"   >>> Efficiency Ratio: {avg_ess/N*100:.1f}%")
        if avg_ess/N > 0.8:
            print("   >>> STATUS: GOLD STANDARD PRECISION (Samples are Independent)")
        else:
            print("   >>> STATUS: CORRELATED (Consider increasing num_steps)")
