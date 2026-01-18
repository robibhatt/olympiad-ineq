"""Main pipeline orchestrator."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from src.data_gen.io import append_jsonl, read_existing_ids
from src.data_gen.models import PromptItem


class Orchestrator:
    """Orchestrates the data generation pipeline.

    Handles batching, resume logic, and writing results to JSONL.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        client,
        batch_size: int = 8,
        output_path: Path | str | None = None,
        resume: bool = True,
        generator_info: dict | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            client: LLM client with generate(prompts) method.
            batch_size: Number of prompts per batch.
            output_path: Path to output JSONL file (optional).
            resume: If True, skip already-processed IDs.
            generator_info: Optional metadata about the generator.
        """
        self.client = client
        self.batch_size = batch_size
        self.output_path = Path(output_path) if output_path else None
        self.resume = resume
        self.generator_info = generator_info or {}

    def _batch_items(self, items: Iterable[PromptItem]) -> Iterator[list[PromptItem]]:
        """Yield batches of items up to batch_size."""
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def process_items(self, items: Iterable[PromptItem]) -> Iterator[dict]:
        """Process items in batches, yielding result records.

        Does not write to file. Use run() for full pipeline with file I/O.

        Args:
            items: Iterable of PromptItem instances.

        Yields:
            Result dictionaries with schema_version, id, messages, meta, raw_completion.
        """
        for batch in self._batch_items(items):
            completions = self.client.generate(batch)

            for prompt, completion in zip(batch, completions):
                record = {
                    "schema_version": self.SCHEMA_VERSION,
                    "id": prompt.id,
                    "messages": prompt.messages,
                    "meta": prompt.meta,
                    "raw_completion": completion,
                }
                if self.generator_info:
                    record["generator"] = self.generator_info
                yield record

    def run(self, items: Iterable[PromptItem]) -> Iterator[dict]:
        """Run the full pipeline with resume logic and file I/O.

        Args:
            items: Iterable of PromptItem instances.

        Yields:
            Result dictionaries after writing to file.
        """
        # Load existing IDs if resuming
        existing_ids = set()
        if self.resume and self.output_path and self.output_path.exists():
            existing_ids = read_existing_ids(self.output_path)

        # Filter out already-processed items
        filtered_items = (item for item in items if item.id not in existing_ids)

        # Process and write
        for record in self.process_items(filtered_items):
            if self.output_path:
                append_jsonl(self.output_path, record)
            yield record
