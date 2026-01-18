"""Prompt source interfaces and implementations."""

import random
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


class TemplatedPromptSource(PromptSource):
    """A prompt source that generates prompts from a template with diversity sampling."""

    def __init__(
        self,
        template: str,
        system_prefix: str,
        format_system: str,
        diversity_config: dict,
        n: int,
        seed: int = 0,
    ):
        """Initialize the templated prompt source.

        Args:
            template: The prompt template string.
            system_prefix: System message prefix.
            format_system: System message format instructions.
            diversity_config: Dict mapping attribute names to {choice: [...], weights: [...]}.
            n: Number of prompts to generate.
            seed: Random seed for reproducibility.
        """
        self.template = template
        self.system_prefix = system_prefix
        self.format_system = format_system
        self.diversity_config = diversity_config
        self.n = n
        self.seed = seed
        self._rng = random.Random(seed)

    def _sample_attributes(self) -> dict:
        """Sample one set of attributes from the diversity config.

        Returns:
            Dict mapping attribute names to sampled values.
        """
        result = {}
        for key, config in self.diversity_config.items():
            choices = config["choice"]
            weights = config.get("weights")
            # Use random.choices for weighted sampling
            sampled = self._rng.choices(choices, weights=weights, k=1)[0]
            result[key] = sampled
        return result

    def _fill_template(self, attributes: dict) -> str:
        """Fill the template with sampled attribute values.

        Args:
            attributes: Dict mapping attribute names to values.

        Returns:
            The template string with placeholders filled in.
        """
        # Convert None to "None" string for readable output
        formatted_attrs = {
            key: "None" if value is None else value
            for key, value in attributes.items()
        }
        return self.template.format(**formatted_attrs)

    def _compose_system_message(self) -> str:
        """Compose the final system message from prefix and format parts.

        Returns:
            The composed system message string.
        """
        parts = []
        prefix_stripped = self.system_prefix.rstrip()
        format_stripped = self.format_system.rstrip()

        if prefix_stripped:
            parts.append(prefix_stripped)
        if format_stripped:
            parts.append(format_stripped)

        return "\n\n".join(parts)

    def __iter__(self) -> Iterator[PromptItem]:
        """Iterate over prompt items.

        Yields:
            PromptItem instances with filled templates and composed system messages.
        """
        # Reset RNG for reproducibility on re-iteration
        self._rng = random.Random(self.seed)

        # Pre-compose system message (same for all items)
        system_message = self._compose_system_message()

        for index in range(self.n):
            attributes = self._sample_attributes()
            user_content = self._fill_template(attributes)

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ]

            meta = {
                "index": index,
                "seed": self.seed,
                **attributes,
            }

            prompt_id = f"prompt_{self.seed}_{index:06d}"

            yield PromptItem(id=prompt_id, messages=messages, meta=meta)
