"""Tests for generate.sh and script.sh scripts."""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest


class TestGenerateShScript:
    """Tests for generate.sh wrapper script."""

    @pytest.fixture
    def repo_root(self):
        """Return the repo root directory."""
        return Path(__file__).parent.parent

    def test_script_exists(self, repo_root):
        """generate.sh exists in repo root."""
        script_path = repo_root / "generate.sh"
        assert script_path.exists(), f"generate.sh not found at {script_path}"

    def test_script_is_executable(self, repo_root):
        """generate.sh has executable permissions."""
        script_path = repo_root / "generate.sh"
        if script_path.exists():
            mode = os.stat(script_path).st_mode
            assert mode & stat.S_IXUSR, "generate.sh should be executable by owner"

    def test_script_creates_outputs_directory_structure(self, repo_root):
        """generate.sh creates outputs/YYYY-MM-DD/HH-MM-SS structure."""
        script_path = repo_root / "generate.sh"
        if not script_path.exists():
            pytest.skip("generate.sh not yet created")

        content = script_path.read_text()

        # Check for timestamped directory creation
        assert "outputs/" in content or "outputs" in content, "Script should use outputs directory"
        # Check for mkdir -p to create the directory
        assert "mkdir -p" in content, "Script should create directory with mkdir -p"
        # Check for date command to create timestamp
        assert "date" in content, "Script should use date for timestamp"


class TestScriptShUpdates:
    """Tests for script.sh updates."""

    @pytest.fixture
    def repo_root(self):
        """Return the repo root directory."""
        return Path(__file__).parent.parent

    def test_no_hardcoded_log_paths(self, repo_root):
        """script.sh has no hardcoded --output/--error paths in SBATCH directives."""
        script_path = repo_root / "script.sh"
        content = script_path.read_text()

        # Check that hardcoded log paths are removed from SBATCH directives
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('#SBATCH'):
                assert '--output=' not in line, f"Found hardcoded --output in: {line}"
                assert '--error=' not in line, f"Found hardcoded --error in: {line}"

    def test_requires_output_dir(self, repo_root):
        """script.sh validates OUTPUT_DIR environment variable."""
        script_path = repo_root / "script.sh"
        content = script_path.read_text()

        # Should check for OUTPUT_DIR
        assert "OUTPUT_DIR" in content, "Script should reference OUTPUT_DIR"
        # Should have error handling for missing OUTPUT_DIR
        assert "ERROR" in content or "error" in content or "-z" in content, \
            "Script should validate OUTPUT_DIR is set"

    def test_passes_hydra_override(self, repo_root):
        """script.sh passes hydra.run.dir to generate.py."""
        script_path = repo_root / "script.sh"
        content = script_path.read_text()

        # Should pass hydra.run.dir override
        assert "hydra.run.dir" in content, "Script should pass hydra.run.dir override"
        assert "OUTPUT_DIR" in content, "Script should use OUTPUT_DIR variable"
