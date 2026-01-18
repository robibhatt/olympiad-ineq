"""Unit tests for config validation (no GPU required)."""

import warnings

import pytest

from src.data_gen import GPU_DTYPE_SUPPORT, validate_vllm_config, warn_on_config_issues


class TestGPUDtypeSupport:
    """Tests for GPU_DTYPE_SUPPORT constant."""

    def test_v100_supports_float16(self):
        """V100 supports float16."""
        assert "float16" in GPU_DTYPE_SUPPORT["v100"]

    def test_v100_supports_float32(self):
        """V100 supports float32."""
        assert "float32" in GPU_DTYPE_SUPPORT["v100"]

    def test_v100_does_not_support_bfloat16(self):
        """V100 does NOT support bfloat16."""
        assert "bfloat16" not in GPU_DTYPE_SUPPORT["v100"]

    def test_a100_supports_bfloat16(self):
        """A100 supports bfloat16."""
        assert "bfloat16" in GPU_DTYPE_SUPPORT["a100"]

    def test_a100_supports_float16(self):
        """A100 supports float16."""
        assert "float16" in GPU_DTYPE_SUPPORT["a100"]

    def test_h100_supports_fp8(self):
        """H100 supports fp8."""
        assert "fp8" in GPU_DTYPE_SUPPORT["h100"]


class TestValidateVLLMConfig:
    """Tests for validate_vllm_config function."""

    def test_valid_v100_config_no_warnings(self):
        """Valid V100 config with float16 produces no warnings."""
        config = {"gpu_type": "v100", "dtype": "float16", "model": "test"}
        warnings_list = validate_vllm_config(config)
        assert warnings_list == []

    def test_valid_a100_config_no_warnings(self):
        """Valid A100 config with bfloat16 produces no warnings."""
        config = {"gpu_type": "a100", "dtype": "bfloat16", "model": "test"}
        warnings_list = validate_vllm_config(config)
        assert warnings_list == []

    def test_v100_with_bfloat16_warns(self):
        """V100 with bfloat16 produces warning."""
        config = {"gpu_type": "v100", "dtype": "bfloat16", "model": "test"}
        warnings_list = validate_vllm_config(config)
        assert len(warnings_list) == 1
        assert "bfloat16" in warnings_list[0]
        assert "v100" in warnings_list[0]

    def test_unknown_gpu_type_warns(self):
        """Unknown GPU type produces warning."""
        config = {"gpu_type": "unknown_gpu", "dtype": "float16"}
        warnings_list = validate_vllm_config(config)
        assert len(warnings_list) == 1
        assert "unknown_gpu" in warnings_list[0]
        assert "Unknown gpu_type" in warnings_list[0]

    def test_missing_gpu_type_no_warning(self):
        """Missing gpu_type does not produce warning."""
        config = {"dtype": "float16", "model": "test"}
        warnings_list = validate_vllm_config(config)
        assert warnings_list == []

    def test_missing_dtype_no_warning(self):
        """Missing dtype does not produce warning."""
        config = {"gpu_type": "v100", "model": "test"}
        warnings_list = validate_vllm_config(config)
        assert warnings_list == []

    def test_gpu_type_case_insensitive(self):
        """GPU type matching is case insensitive."""
        config = {"gpu_type": "V100", "dtype": "float16"}
        warnings_list = validate_vllm_config(config)
        assert warnings_list == []

        config = {"gpu_type": "A100", "dtype": "bfloat16"}
        warnings_list = validate_vllm_config(config)
        assert warnings_list == []

    def test_suggests_supported_dtypes(self):
        """Warning suggests supported dtypes."""
        config = {"gpu_type": "v100", "dtype": "bfloat16"}
        warnings_list = validate_vllm_config(config)
        assert "float16" in warnings_list[0]
        assert "float32" in warnings_list[0]


class TestWarnOnConfigIssues:
    """Tests for warn_on_config_issues function."""

    def test_emits_python_warning(self):
        """warn_on_config_issues emits Python UserWarning."""
        config = {"gpu_type": "v100", "dtype": "bfloat16"}
        with pytest.warns(UserWarning, match="bfloat16"):
            warn_on_config_issues(config)

    def test_no_warning_for_valid_config(self):
        """No warning is emitted for valid config."""
        config = {"gpu_type": "v100", "dtype": "float16"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_on_config_issues(config)
            assert len(w) == 0

    def test_multiple_warnings_emitted(self):
        """Multiple issues produce multiple warnings (if applicable)."""
        # Currently only one type of warning per config, but structure supports multiple
        config = {"gpu_type": "unknown", "dtype": "invalid"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_on_config_issues(config)
            # Unknown gpu_type warning
            assert len(w) == 1
