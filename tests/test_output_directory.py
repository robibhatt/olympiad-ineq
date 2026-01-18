"""Tests for output directory configuration and behavior."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.fixture
def config_dir():
    """Return the absolute path to the configs directory."""
    return str(Path(__file__).parent.parent / "configs")


class TestOutputPathConfig:
    """Tests for output.path configuration."""

    def test_config_output_path_is_filename_only(self, config_dir):
        """output.path should be 'generated.jsonl' (no directory prefix)."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")
            assert cfg.output.path == "generated.jsonl"


class TestOrchestratorWritesToPath:
    """Tests for Orchestrator writing to specified path."""

    def test_orchestrator_writes_to_specified_path(self, tmp_path):
        """Orchestrator writes to whatever path is given."""
        from src.data_gen.client import FakeLLMClient
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        # Use a nested path to verify it writes to the exact location
        output_file = tmp_path / "subdir" / "output.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        items = [
            PromptItem(id="test1", messages=[{"role": "user", "content": "Hello"}]),
        ]

        client = FakeLLMClient(response_template="Response for {id}")
        orchestrator = Orchestrator(
            client=client,
            batch_size=10,
            output_path=output_file,
            resume=False,
        )
        list(orchestrator.run(items))

        assert output_file.exists()
        with open(output_file) as f:
            content = f.read()
        assert "test1" in content
