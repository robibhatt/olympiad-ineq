"""Config validation for vLLM settings."""

import warnings
from typing import Any

# GPU type to supported dtypes mapping
GPU_DTYPE_SUPPORT: dict[str, set[str]] = {
    "v100": {"float16", "float32"},
    "a100": {"float16", "bfloat16", "float32"},
    "h100": {"float16", "bfloat16", "float32", "fp8"},
}


def validate_vllm_config(vllm_config: dict[str, Any]) -> list[str]:
    """Validate vLLM config and return list of warning messages.

    Args:
        vllm_config: Dictionary containing vllm configuration with keys like
            gpu_type, dtype, model, etc.

    Returns:
        List of warning messages (empty if no issues).
    """
    warnings_list = []

    gpu_type = vllm_config.get("gpu_type")
    dtype = vllm_config.get("dtype")

    if gpu_type and dtype:
        gpu_type_lower = gpu_type.lower()
        if gpu_type_lower in GPU_DTYPE_SUPPORT:
            supported = GPU_DTYPE_SUPPORT[gpu_type_lower]
            if dtype not in supported:
                warnings_list.append(
                    f"dtype '{dtype}' may not be supported on {gpu_type}. "
                    f"Supported dtypes: {sorted(supported)}"
                )
        else:
            warnings_list.append(
                f"Unknown gpu_type '{gpu_type}'. "
                f"Known types: {sorted(GPU_DTYPE_SUPPORT.keys())}"
            )

    return warnings_list


def warn_on_config_issues(vllm_config: dict[str, Any]) -> None:
    """Emit Python warnings for any config issues.

    Args:
        vllm_config: Dictionary containing vllm configuration.
    """
    for msg in validate_vllm_config(vllm_config):
        warnings.warn(msg, UserWarning)
