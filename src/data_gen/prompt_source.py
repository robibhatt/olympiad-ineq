"""Prompt source interfaces and implementations."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.data_gen.models import PromptItem


class PromptSource(ABC):
    """Abstract base class for prompt sources."""

    @abstractmethod
    def __iter__(self) -> Iterator[PromptItem]:
        """Iterate over prompt items."""
        pass


class ExplicitPromptSource(PromptSource):
    """A prompt source that yields from an explicit list of items.

    Attributes:
        items: List of dicts with 'id', 'messages', and optional 'meta' keys.
    """

    def __init__(self, items: list[dict]):
        """Initialize with a list of item dicts.

        Args:
            items: List of dicts with 'id', 'messages', and optional 'meta'.
        """
        self.items = items

    def __iter__(self) -> Iterator[PromptItem]:
        """Yield PromptItem instances from the item list."""
        for item in self.items:
            yield PromptItem(
                id=item["id"],
                messages=item["messages"],
                meta=item.get("meta", {}),
            )
