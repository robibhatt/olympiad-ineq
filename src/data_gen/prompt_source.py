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


class TemplatePromptSource(PromptSource):
    """A prompt source that expands a template n times.

    Args:
        system: System message content.
        template: User message template with {var} placeholders.
        n: Number of PromptItems to generate.
        id_prefix: Prefix for deterministic ID generation.
        vars: Dict of variable substitutions for template.
        meta: Optional metadata to attach to each item.
    """

    def __init__(
        self,
        system: str,
        template: str,
        n: int,
        id_prefix: str,
        vars: dict | None = None,
        meta: dict | None = None,
    ):
        self.system = system
        self.template = template
        self.n = n
        self.id_prefix = id_prefix
        self.vars = vars or {}
        self.meta = meta or {}

    def __iter__(self) -> Iterator[PromptItem]:
        for k in range(self.n):
            item_id = f"{self.id_prefix}-{k:06d}"

            # Render template with vars
            try:
                user_content = self.template.format(**self.vars)
            except KeyError as e:
                raise ValueError(f"Missing template variable: {e}")

            messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user_content},
            ]

            item_meta = {
                "id_prefix": self.id_prefix,
                "index": k,
                **self.meta,
                **self.vars,
            }

            yield PromptItem(id=item_id, messages=messages, meta=item_meta)
