"""Data generation pipeline package."""

from src.data_gen.client import FakeLLMClient, LLMClient, VLLMClient
from src.data_gen.io import append_jsonl, read_existing_ids
from src.data_gen.models import PromptItem
from src.data_gen.orchestrator import Orchestrator
from src.data_gen.prompt_source import (
    ExplicitPromptSource,
    PromptSource,
    TemplatedPromptSource,
)
from src.data_gen.validation import (
    GPU_DTYPE_SUPPORT,
    validate_vllm_config,
    warn_on_config_issues,
)

__all__ = [
    "PromptItem",
    "PromptSource",
    "ExplicitPromptSource",
    "TemplatedPromptSource",
    "LLMClient",
    "FakeLLMClient",
    "VLLMClient",
    "read_existing_ids",
    "append_jsonl",
    "Orchestrator",
    "GPU_DTYPE_SUPPORT",
    "validate_vllm_config",
    "warn_on_config_issues",
]
