"""Main entry point with Hydra config and wandb logging."""

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig) -> None:
    """Initialize wandb with config."""
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def run_data_gen(cfg: DictConfig) -> None:
    """Run the data generation pipeline."""
    from src.data_gen import Orchestrator, VLLMClient

    # Instantiate prompt source from config
    prompt_source = instantiate(cfg.prompt_source)

    # Create vLLM client
    client = VLLMClient(
        model=cfg.vllm.model,
        temperature=cfg.vllm.temperature,
        max_tokens=cfg.vllm.max_tokens,
    )

    # Create orchestrator
    orchestrator = Orchestrator(
        client=client,
        batch_size=cfg.batching.batch_size,
        output_path=cfg.output.path,
        resume=cfg.output.resume,
        generator_info={
            "model": cfg.vllm.model,
            "temperature": cfg.vllm.temperature,
            "max_tokens": cfg.vllm.max_tokens,
        },
    )

    # Run pipeline
    results = list(orchestrator.run(prompt_source))
    print(f"Generated {len(results)} results to {cfg.output.path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function."""
    init_wandb(cfg)

    print(OmegaConf.to_yaml(cfg))

    # Stage dispatch
    if cfg.stage == "data_gen":
        run_data_gen(cfg)
    elif cfg.stage is not None:
        raise ValueError(f"Unknown stage: {cfg.stage}")

    wandb.finish()


if __name__ == "__main__":
    main()
