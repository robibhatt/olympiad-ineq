"""Tests for Hydra config."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def config_dir():
    """Return the absolute path to the configs directory."""
    return str(Path(__file__).parent.parent / "configs")


class TestNewConfigStructure:
    """Tests for the new nested config structure."""

    def test_load_config(self, config_dir):
        """config.yaml loads without error."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")
            assert cfg is not None

    def test_output_section(self, config_dir):
        """output.path and output.resume are accessible."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert cfg.output.path == "data/raw/generated.jsonl"
            assert cfg.output.resume is True

    def test_batching_section(self, config_dir):
        """batching.batch_size is accessible."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert cfg.batching.batch_size == 16

    def test_vllm_section(self, config_dir):
        """vllm.model, vllm.sampling.temperature, etc. are accessible."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert cfg.vllm.model == "Qwen/Qwen2.5-Math-72B-Instruct"
            assert cfg.vllm.tensor_parallel_size == 4
            assert cfg.vllm.dtype == "bfloat16"
            assert cfg.vllm.max_model_len == 8192
            assert cfg.vllm.sampling.temperature == 0.75
            assert cfg.vllm.sampling.top_p == 0.95
            assert cfg.vllm.sampling.max_tokens == 1600

    def test_prompt_plan_section(self, config_dir):
        """prompt_plan.template, prompt_plan.diversity, etc. are accessible."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert cfg.prompt_plan.seed == 0
            assert cfg.prompt_plan.n == 2000
            assert "olympiad" in cfg.prompt_plan.system_prefix
            assert "{primary_technique}" in cfg.prompt_plan.template
            assert "primary_technique" in cfg.prompt_plan.diversity
            assert cfg.prompt_plan.diversity.primary_technique.choice[0] == "AM-GM"

    def test_format_section(self, config_dir):
        """format.system is accessible."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert "Q:" in cfg.format.system
            assert "A:" in cfg.format.system

    def test_config_override_nested(self, config_dir):
        """Hydra overrides work with nested config paths."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "prompt_plan.n=100",
                    "batching.batch_size=32",
                    "vllm.sampling.temperature=0.5",
                ],
            )

            assert cfg.prompt_plan.n == 100
            assert cfg.batching.batch_size == 32
            assert cfg.vllm.sampling.temperature == 0.5


def test_hydra_saves_config(tmp_path, config_dir):
    """Hydra saves config.yaml to outputs directory."""
    import subprocess

    generate_path = Path(__file__).parent.parent / "generate.py"
    result = subprocess.run(
        [
            "python",
            str(generate_path),
            f"hydra.run.dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )

    # This will fail because VLLMClient requires a real model, but config should still be saved
    # Check that config.yaml was saved
    config_file = tmp_path / ".hydra" / "config.yaml"
    assert config_file.exists(), f"Config not saved. Contents: {list(tmp_path.rglob('*'))}"

    # Verify saved config has expected nested structure
    saved_cfg = OmegaConf.load(config_file)
    assert saved_cfg.prompt_plan.n == 2000
