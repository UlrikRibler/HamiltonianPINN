# HamiltonianPINN 🌌
### The "Gold Standard" for Bayesian Physics-Informed Neural Networks
![License](https://img.shields.io/badge/license-MIT-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-God%20Tier-red) ![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)

**HamiltonianPINN** is a research-grade framework that solves the **"Physics Inversion Problem"** with mathematical rigor. It combines **Physics-Informed Neural Networks (PINNs)** with **Hamiltonian Monte Carlo (HMC)** to quantify epistemic uncertainty in high-dimensional solution manifolds.

---

## 💡 Why This Is A Game Changer

### 1. Beyond the "Point Estimate" Illusion
Standard PINNs trained with `Adam` or `L-BFGS` produce a single solution. This is dangerous in scientific computing because it masks non-uniqueness and ill-posedness. **HamiltonianPINN** does not give you *an* answer; it gives you the **distribution of all valid answers** consistent with the observed data and physical laws.

### 2. Solving the "Curse of Dimensionality"
Classical Bayesian methods (like Random Walk Metropolis) fail when the parameter space $D > 100$. A PINN has $D > 10,000$.
*   **The Solution:** We treat the loss surface as a physical terrain. Instead of blindly stumbling around (Random Walk), our sampler uses the **gradient of the physics** ($\nabla \mathcal{L}$) to "kick" a virtual particle across the landscape.
*   **The Result:** We can propose samples that are far apart in parameter space but still have high acceptance rates (~80%), allowing us to explore high-dimensional manifolds that are impossible for traditional samplers.

### 3. The "Small Steps, Long Paths" Strategy
We employ a rigorous symplectic integration scheme:
*   **Small Steps ($dt$):** Ensures energy conservation errors are negligible (Machine Precision).
*   **Long Paths ($L$):** We integrate for hundreds of steps per sample. This forces the particle to travel to distant modes of the posterior, ensuring that Sample $N$ and Sample $N+1$ are statistically independent (**High Effective Sample Size**).

---

## 🚀 The Mathematical Engine

We utilize **Symplectic Geometry** to traverse the probability landscape:

1.  **The Hamiltonian ($H$):** The total energy of the system, conserved by nature.
    $$H(\theta, p) = U(\theta) + K(p) = -\log \mathcal{L}_{physics}(\theta) + \frac{1}{2} p^T M^{-1} p$$
    *   $U(\theta)$: Potential Energy (The Physics Loss).
    *   $K(p)$: Kinetic Energy (The Momentum of the Sampler).

2.  **Dual Averaging Adaptation:**
Implementation of Nesterov's Dual Averaging to automatically tune the step size $\epsilon$ during burn-in, targeting an optimal acceptance rate of 80%.

3.  **Exact Physics Derivatives:**
Powered by `torch.func` (JAX-like functional API), we compute exact Jacobians and Hessians of the PDE without the memory overhead of standard autograd.

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.10+
*   PyTorch 2.0+ (with `torch.func` support)
*   Hydra (`pip install hydra-core`)

### Installation
```bash
git clone https://github.com/UlrikRibler/HamiltonianPINN.git
cd HamiltonianPINN
pip install -r requirements.txt
```

### ⚙️ Configuration
The project is configured via **Hydra** in `conf/config.yaml`. The **"Gold Standard"** preset is designed for maximum statistical fidelity:

*   **Precision Settings:**
    *   `num_samples: 200` (High-density posterior)
    *   `num_steps: 100` (Long trajectories for decorrelation)
    *   `step_size: 1.5e-3` (Initial step size, tuned automatically)
    *   `adapt_mass_matrix: true` (Riemannian Metric adaptation during burn-in)
    *   `burn_in: 200` (Warm-up period for adaptation)

## 🏃 Usage & Execution

### Option 1: Robust Background Execution (Recommended)
High-precision sampling is computationally intensive. Run the pipeline as a background process to keep your terminal free and ensure it persists.

**Step 1: Start the Process**
Copy and paste this command into your PowerShell terminal:
```powershell
Start-Process python -ArgumentList "main.py" -RedirectStandardOutput "training.log" -RedirectStandardError "training_error.log" -NoNewWindow
```

**Step 2: Monitor Progress**
To watch the training logs in real-time (press `Ctrl+C` to stop watching, the process will continue):
```powershell
Get-Content training.log -Wait
```

**Step 3: Check Errors**
If nothing seems to be happening, check the error log:
```powershell
Get-Content training_error.log
```

### Option 2: Interactive Run
For debugging or shorter runs where you want to see output directly in the terminal:
```bash
python main.py
```

## 📊 Posterior Analytics & Artifacts
All results are automatically saved to `results/` and the date-structured `outputs/` directory.

| Artifact | Description |
| :--- | :--- |
| **`uncertainty_profile.png`** | **The "God Tier" Plot.** Visualizes the mean prediction $\mathbb{E}[u]$ and the 95% Credible Interval (uncertainty bands). Shows exactly where the physics is uncertain. |
| **`trace_plot.png`** | **Chain Diagnostics.** Visualizes the mixing of the Hamiltonian particle across multiple parameters (start, middle, end) to verify global convergence. |
| **`training.log`** | Detailed telemetry including loss convergence, HMC energy errors, and **Effective Sample Size (ESS)** diagnostics. |

## 📂 Project Structure

```
C:\Users\ulrik\Documents\VSCODE\Markov-chain-optimalization\
├───main.py                 # The Entry Point / Hydra Runtime
├───conf\                   # Hydra Configuration (Metric Tensors, Step Sizes)
├───src\
│   ├───pipeline.py         # The Orchestrator (MAP -> Burn-in -> Sampling)
│   ├───validator.py        # The Posterior Analytics Engine (ESS, Uncertainty)
│   ├───data.py             # The Physics Data Foundry (Manifold Synthesis)
│   ├───physics\            # The Laws of Nature (Burgers PDE via torch.func)
│   ├───mcmc\               # The Engine (Hamiltonian, Leapfrog, HMCSampler)
│   ├───models\             # The Brain (PINN Neural Ansatz)
│   └───utils.py            # Manifold Mapping Utilities (Coordinate Charts)
└───results\                # High-Resolution Artifacts
```

## 📄 License
MIT License.

Built with **PyTorch 2.0**, **Symplectic Geometry**, and **Hamiltonian Dynamics**. 🌌