"""Main entry point with Hydra config and wandb logging."""

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig) -> None:
    """Initialize wandb with config."""
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function."""
    init_wandb(cfg)

    print(OmegaConf.to_yaml(cfg))

    wandb.finish()


if __name__ == "__main__":
    main()
