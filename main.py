import hydra
from omegaconf import DictConfig, OmegaConf
from src.pipeline import BayesianPipeline

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    The Entry Point for the NeuroManifold Runtime.
    
    This function initializes the Hydra configuration engine and bootstraps the 
    `BayesianPipeline`. It serves as the interface between the CLI user and the 
    high-fidelity physics inversion core.
    """
    # Print the configuration for debugging/logging
    print(OmegaConf.to_yaml(cfg))
    
    # Instantiate the Orchestrator
    pipeline = BayesianPipeline(cfg)
    
    # Ignite
    pipeline.run()

if __name__ == "__main__":
    main()
