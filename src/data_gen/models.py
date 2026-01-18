"""Data models for the data generation pipeline."""

from dataclasses import dataclass, field


@dataclass
class PromptItem:
    """A single prompt item to be processed by the LLM.

    Attributes:
        id: Unique identifier for this prompt.
        messages: List of message dicts with 'role' and 'content' keys.
        meta: Optional metadata dict.
    """

    id: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)
