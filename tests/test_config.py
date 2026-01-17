"""Tests for Hydra config and wandb initialization."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def config_dir():
    """Return the absolute path to the configs directory."""
    return str(Path(__file__).parent.parent / "configs")


def test_config_loads(config_dir):
    """Config loads with Hydra and has expected structure."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

        # Check top-level keys exist
        assert "wandb" in cfg
        assert "data_gen" in cfg

        # Check wandb config
        assert cfg.wandb.project == "olympiad-ineq"
        assert cfg.wandb.mode == "offline"

        # Check data_gen defaults (small.yaml)
        assert cfg.data_gen.n_samples == 100
        assert cfg.data_gen.seed == 42


def test_data_gen_override(config_dir):
    """data_gen=full uses full.yaml values."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["data_gen=full"])

        # Check full.yaml values
        assert cfg.data_gen.n_samples == 10000
        assert cfg.data_gen.seed == 42


def test_wandb_init(config_dir):
    """wandb.init is called with correct project name."""
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    with patch("wandb.init") as mock_init:
        # Import and call the function that initializes wandb
        from main import init_wandb

        init_wandb(cfg)

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["project"] == "olympiad-ineq"
        assert call_kwargs["mode"] == "offline"


def test_hydra_saves_config(tmp_path, config_dir):
    """Hydra saves config.yaml to outputs directory."""
    # Run main.py with hydra.run.dir pointing to tmp_path
    import subprocess

    main_path = Path(__file__).parent.parent / "main.py"
    result = subprocess.run(
        [
            "python",
            str(main_path),
            f"hydra.run.dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )

    assert result.returncode == 0, f"main.py failed: {result.stderr}"

    # Check that config.yaml was saved
    config_file = tmp_path / ".hydra" / "config.yaml"
    assert config_file.exists(), f"Config not saved. Contents: {list(tmp_path.rglob('*'))}"

    # Verify saved config has expected values
    saved_cfg = OmegaConf.load(config_file)
    assert saved_cfg.wandb.project == "olympiad-ineq"
    assert saved_cfg.data_gen.n_samples == 100
