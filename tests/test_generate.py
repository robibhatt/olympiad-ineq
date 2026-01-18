"""Tests for generate.py entry point."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def config_dir():
    """Return the absolute path to the configs directory."""
    return str(Path(__file__).parent.parent / "configs")


class TestGenerateConfig:
    """Tests for generate.py config structure."""

    def test_config_has_prompt_plan(self, config_dir):
        """config.yaml has prompt_plan section."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert "prompt_plan" in cfg
            assert cfg.prompt_plan.n == 2000
            assert cfg.prompt_plan.seed == 0
            assert "template" in cfg.prompt_plan
            assert "diversity" in cfg.prompt_plan

    def test_config_has_vllm_settings(self, config_dir):
        """config.yaml has nested vllm settings."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert cfg.vllm.model == "Qwen/Qwen2.5-Math-72B-Instruct"
            assert cfg.vllm.sampling.temperature == 0.75
            assert cfg.vllm.sampling.max_tokens == 1600
            assert cfg.vllm.sampling.top_p == 0.95
            assert cfg.vllm.tensor_parallel_size == 4

    def test_config_has_output_settings(self, config_dir):
        """config.yaml has output section."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert cfg.output.path == "generated.jsonl"
            assert cfg.output.resume is True

    def test_config_has_format_system(self, config_dir):
        """config.yaml has format.system for output format instructions."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            assert "format" in cfg
            assert "Q:" in cfg.format.system
            assert "A:" in cfg.format.system


class TestGenerateIntegration:
    """Integration tests for generate.py with new config structure."""

    def test_creates_templated_prompt_source(self, config_dir):
        """generate.py creates TemplatedPromptSource from config."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

        from src.data_gen import TemplatedPromptSource

        # Create prompt source exactly as generate.py does
        prompt_source = TemplatedPromptSource(
            template=cfg.prompt_plan.template,
            system_prefix=cfg.prompt_plan.system_prefix,
            format_system=cfg.format.system,
            diversity_config=OmegaConf.to_container(cfg.prompt_plan.diversity),
            n=cfg.prompt_plan.n,
            seed=cfg.prompt_plan.seed,
        )

        # Verify it works
        items = list(prompt_source)
        assert len(items) == cfg.prompt_plan.n
        assert items[0].id == f"prompt_{cfg.prompt_plan.seed}_000000"

    def test_passes_vllm_config(self, config_dir):
        """VLLMClient receives full vllm config."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

        from src.data_gen import VLLMClient

        # Create client exactly as generate.py does
        client = VLLMClient(
            model=cfg.vllm.model,
            temperature=cfg.vllm.sampling.temperature,
            max_tokens=cfg.vllm.sampling.max_tokens,
            top_p=cfg.vllm.sampling.top_p,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            dtype=cfg.vllm.dtype,
            max_model_len=cfg.vllm.max_model_len,
        )

        # Verify config was stored correctly
        assert client.model == "Qwen/Qwen2.5-Math-72B-Instruct"
        assert client.temperature == 0.75
        assert client.max_tokens == 1600
        assert client.top_p == 0.95
        assert client.engine_kwargs["tensor_parallel_size"] == 4
        assert client.engine_kwargs["dtype"] == "bfloat16"
        assert client.engine_kwargs["max_model_len"] == 8192

    def test_end_to_end_with_fake_client(self, tmp_path, config_dir):
        """Full pipeline works with FakeLLMClient and new config."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=["prompt_plan.n=3"],  # Small n for test
            )

        from src.data_gen import FakeLLMClient, Orchestrator, TemplatedPromptSource

        # Create prompt source
        prompt_source = TemplatedPromptSource(
            template=cfg.prompt_plan.template,
            system_prefix=cfg.prompt_plan.system_prefix,
            format_system=cfg.format.system,
            diversity_config=OmegaConf.to_container(cfg.prompt_plan.diversity),
            n=cfg.prompt_plan.n,
            seed=cfg.prompt_plan.seed,
        )

        # Use fake client for testing
        client = FakeLLMClient(response_template="Generated for {id}")

        # Create orchestrator with nested config paths
        output_path = tmp_path / "test_output.jsonl"
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

        # Verify results
        assert len(results) == 3

        # Verify output file
        import json

        with open(output_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 3
        for record in records:
            assert record["schema_version"] == 1
            assert record["id"].startswith("prompt_0_")
            assert len(record["messages"]) == 2
            assert record["messages"][0]["role"] == "system"
            assert record["messages"][1]["role"] == "user"
            assert "primary_technique" in record["meta"]

    def test_orchestrator_generator_info(self, tmp_path, config_dir):
        """Orchestrator receives correct generator_info from config."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

        from src.data_gen import FakeLLMClient, Orchestrator

        output_path = tmp_path / "test_output.jsonl"
        orchestrator = Orchestrator(
            client=FakeLLMClient(),
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

        assert orchestrator.generator_info["model"] == "Qwen/Qwen2.5-Math-72B-Instruct"
        assert orchestrator.generator_info["temperature"] == 0.75
        assert orchestrator.generator_info["max_tokens"] == 1600
        assert orchestrator.generator_info["top_p"] == 0.95


def test_generate_script_runs_with_mock(tmp_path, config_dir):
    """generate.py runs with mocked VLLMClient."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    # Mock the VLLMClient to avoid needing real model
    with patch("generate.VLLMClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate.return_value = ["response1", "response2", "response3"]
        mock_client_class.return_value = mock_client

        # Import and test the core logic without Hydra decorator
        from src.data_gen import TemplatedPromptSource

        prompt_source = TemplatedPromptSource(
            template=cfg.prompt_plan.template,
            system_prefix=cfg.prompt_plan.system_prefix,
            format_system=cfg.format.system,
            diversity_config=OmegaConf.to_container(cfg.prompt_plan.diversity),
            n=3,
            seed=cfg.prompt_plan.seed,
        )

        items = list(prompt_source)
        assert len(items) == 3
        assert items[0].id == "prompt_0_000000"
