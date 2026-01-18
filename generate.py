"""Data generation entry point."""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_gen import Orchestrator, TemplatedPromptSource, VLLMClient


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run data generation pipeline."""
    print(OmegaConf.to_yaml(cfg))

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
    )

    # Create orchestrator with nested config
    orchestrator = Orchestrator(
        client=client,
        batch_size=cfg.batching.batch_size,
        output_path=cfg.output.path,
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
    print(f"Generated {len(results)} results to {cfg.output.path}")


if __name__ == "__main__":
    main()
