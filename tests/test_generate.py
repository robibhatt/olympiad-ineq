"""Tests for generate.py entry point."""

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir


@pytest.fixture
def config_dir():
    """Return the absolute path to the configs directory."""
    return str(Path(__file__).parent.parent / "configs")


def test_generate_config_has_stage_data_gen(config_dir):
    """generate.yaml sets stage to data_gen."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="generate")

        assert cfg.stage == "data_gen"


def test_generate_config_inherits_defaults(config_dir):
    """generate.yaml inherits all defaults from config.yaml."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="generate")

        # Check inherited values from config.yaml
        assert cfg.wandb.project == "olympiad-ineq"
        assert cfg.data_gen.n_samples == 100
        assert cfg.batching.batch_size == 8


def test_generate_script_runs(tmp_path):
    """generate.py script executes without error.

    Uses stage=null to skip actual generation (config tests verify default).
    """
    import subprocess

    generate_path = Path(__file__).parent.parent / "generate.py"
    result = subprocess.run(
        [
            "python",
            str(generate_path),
            f"hydra.run.dir={tmp_path}",
            "wandb.mode=disabled",
            "stage=null",  # Skip actual generation for test
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )

    assert result.returncode == 0, f"generate.py failed: {result.stderr}"

    # Verify script ran with generate.yaml (check saved config exists)
    config_file = tmp_path / ".hydra" / "config.yaml"
    assert config_file.exists(), f"Config not saved. Contents: {list(tmp_path.rglob('*'))}"


def test_generate_reuses_run_data_gen():
    """generate.py imports run_data_gen from main.py (no duplication)."""
    import generate
    from main import run_data_gen

    # Verify generate imports the same function
    assert generate.run_data_gen is run_data_gen
