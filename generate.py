"""Data generation entry point."""

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from main import init_wandb, run_data_gen


@hydra.main(version_base=None, config_path="configs", config_name="generate")
def main(cfg: DictConfig) -> None:
    """Run data generation pipeline."""
    init_wandb(cfg)
    print(OmegaConf.to_yaml(cfg))

    if cfg.stage == "data_gen":
        run_data_gen(cfg)

    wandb.finish()


if __name__ == "__main__":
    main()
