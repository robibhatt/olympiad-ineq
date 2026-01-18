"""Data generation entry point."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data_gen import Orchestrator, TemplatedPromptSource, VLLMClient, warn_on_config_issues


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run data generation pipeline."""
    print(OmegaConf.to_yaml(cfg))

    # Resolve output path relative to Hydra's output directory
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    output_path = output_dir / cfg.output.path

    # Validate vLLM config before proceeding
    warn_on_config_issues(OmegaConf.to_container(cfg.vllm))

    # Print pipeline start info
    print("=" * 60)
    print("Starting data generation pipeline")
    print("=" * 60)
    print(f"GPU type: {cfg.vllm.gpu_type}")
    print(f"Model: {cfg.vllm.model}")
    print(f"Prompts: {cfg.prompt_plan.n}")
    print(f"Output: {output_path}")

    # Create prompt source from prompt_plan config
    prompt_source = TemplatedPromptSource(
        template=cfg.prompt_plan.template,
        system_prefix=cfg.prompt_plan.system_prefix,
        format_system=cfg.format.system,
        diversity_config=OmegaConf.to_container(cfg.prompt_plan.diversity),
        n=cfg.prompt_plan.n,
        seed=cfg.prompt_plan.seed,
    )

    # Create vLLM client with full config
    client = VLLMClient(
        model=cfg.vllm.model,
        temperature=cfg.vllm.sampling.temperature,
        max_tokens=cfg.vllm.sampling.max_tokens,
        top_p=cfg.vllm.sampling.top_p,
        tensor_parallel_size=cfg.vllm.tensor_parallel_size,
        dtype=cfg.vllm.dtype,
        max_model_len=cfg.vllm.max_model_len,
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
    )

    # Create orchestrator with nested config
    orchestrator = Orchestrator(
        client=client,
        batch_size=cfg.batching.batch_size,
        output_path=output_path,
        resume=cfg.output.resume,
        generator_info={
            "model": cfg.vllm.model,
            "temperature": cfg.vllm.sampling.temperature,
            "max_tokens": cfg.vllm.sampling.max_tokens,
            "top_p": cfg.vllm.sampling.top_p,
        },
    )

    # Run pipeline
    results = list(orchestrator.run(prompt_source))
    print(f"Generated {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
