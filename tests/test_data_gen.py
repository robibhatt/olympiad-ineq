"""Tests for data generation pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_jsonl(tmp_path):
    """Return a path to a temporary JSONL file."""
    return tmp_path / "results.jsonl"


class TestPromptSourceExplicit:
    """Tests for ExplicitPromptSource."""

    def test_prompt_source_explicit_items(self):
        """ExplicitPromptSource yields correct PromptItems."""
        from src.data_gen.models import PromptItem
        from src.data_gen.prompt_source import ExplicitPromptSource

        items = [
            {"id": "q1", "messages": [{"role": "user", "content": "Hello"}]},
            {"id": "q2", "messages": [{"role": "user", "content": "World"}], "meta": {"difficulty": 1}},
        ]
        source = ExplicitPromptSource(items=items)
        result = list(source)

        assert len(result) == 2
        assert all(isinstance(item, PromptItem) for item in result)
        assert result[0].id == "q1"
        assert result[0].messages == [{"role": "user", "content": "Hello"}]
        assert result[0].meta == {}
        assert result[1].id == "q2"
        assert result[1].meta == {"difficulty": 1}


class TestBatching:
    """Tests for batching logic."""

    def test_batching_respects_batch_size(self):
        """5 items with batch_size=2 results in 3 calls [2,2,1]."""
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        items = [
            PromptItem(id=f"q{i}", messages=[{"role": "user", "content": f"msg{i}"}])
            for i in range(5)
        ]

        # Track batch sizes seen by client
        batch_sizes = []

        class TrackingClient:
            def generate(self, prompts):
                batch_sizes.append(len(prompts))
                return [f"response_{i}" for i in range(len(prompts))]

        orchestrator = Orchestrator(
            client=TrackingClient(),
            batch_size=2,
        )
        # Run without writing to file
        results = list(orchestrator.process_items(items))

        assert batch_sizes == [2, 2, 1], f"Expected [2,2,1] but got {batch_sizes}"
        assert len(results) == 5


class TestResume:
    """Tests for resume functionality."""

    def test_resume_skips_existing_ids(self, temp_jsonl):
        """Existing ids are skipped, only new ids processed."""
        from src.data_gen.io import append_jsonl, read_existing_ids
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        # Pre-populate with existing id
        existing_record = {
            "schema_version": 1,
            "id": "existing_1",
            "messages": [],
            "meta": {},
            "raw_completion": "old response",
        }
        append_jsonl(temp_jsonl, existing_record)

        # Verify existing ids are read
        existing_ids = read_existing_ids(temp_jsonl)
        assert "existing_1" in existing_ids

        # Create items with one existing and one new
        items = [
            PromptItem(id="existing_1", messages=[{"role": "user", "content": "old"}]),
            PromptItem(id="new_1", messages=[{"role": "user", "content": "new"}]),
        ]

        processed_ids = []

        class TrackingClient:
            def generate(self, prompts):
                for p in prompts:
                    processed_ids.append(p.id)
                return [f"response" for _ in prompts]

        orchestrator = Orchestrator(
            client=TrackingClient(),
            batch_size=10,
            output_path=temp_jsonl,
            resume=True,
        )
        list(orchestrator.run(items))

        # Only new_1 should be processed
        assert processed_ids == ["new_1"]


class TestEndToEnd:
    """End-to-end tests."""

    def test_end_to_end_writes_jsonl_schema(self, temp_jsonl):
        """JSONL output has required keys and no duplicates."""
        from src.data_gen.client import FakeLLMClient
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        items = [
            PromptItem(id="q1", messages=[{"role": "user", "content": "Hello"}], meta={"tag": "test"}),
            PromptItem(id="q2", messages=[{"role": "user", "content": "World"}]),
        ]

        client = FakeLLMClient(response_template="Fake response for {id}")
        orchestrator = Orchestrator(
            client=client,
            batch_size=10,
            output_path=temp_jsonl,
            resume=False,
        )
        list(orchestrator.run(items))

        # Read and verify JSONL
        records = []
        with open(temp_jsonl) as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 2

        # Check required keys
        required_keys = {"schema_version", "id", "messages", "meta", "raw_completion"}
        for record in records:
            assert required_keys.issubset(record.keys()), f"Missing keys in {record}"
            assert record["schema_version"] == 1

        # Check no duplicate ids
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids)), "Duplicate ids found"

        # Verify content
        assert records[0]["id"] == "q1"
        assert records[0]["meta"] == {"tag": "test"}
        assert "Fake response" in records[0]["raw_completion"]


class TestTemplatePromptSource:
    """Tests for TemplatePromptSource."""

    def test_template_prompt_source_expands_n(self):
        """Given n=3, yields exactly 3 PromptItems."""
        from src.data_gen.models import PromptItem
        from src.data_gen.prompt_source import TemplatePromptSource

        source = TemplatePromptSource(
            system="You are a helpful assistant.",
            template="Solve this problem: {problem}",
            n=3,
            id_prefix="test",
            vars={"problem": "2+2"},
        )
        result = list(source)

        assert len(result) == 3
        assert all(isinstance(item, PromptItem) for item in result)

    def test_template_prompt_source_deterministic_ids(self):
        """Same config twice produces identical ordered IDs."""
        from src.data_gen.prompt_source import TemplatePromptSource

        config = {
            "system": "You are a helper.",
            "template": "Hello {name}",
            "n": 5,
            "id_prefix": "det-test",
            "vars": {"name": "World"},
        }

        source1 = TemplatePromptSource(**config)
        source2 = TemplatePromptSource(**config)

        ids1 = [item.id for item in source1]
        ids2 = [item.id for item in source2]

        assert ids1 == ids2
        # Check ID pattern follows {id_prefix}-{k:06d}
        assert ids1 == ["det-test-000000", "det-test-000001", "det-test-000002", "det-test-000003", "det-test-000004"]

    def test_template_prompt_source_renders_template_vars(self):
        """Template placeholders are substituted with vars."""
        from src.data_gen.prompt_source import TemplatePromptSource

        source = TemplatePromptSource(
            system="System message",
            template="Hello {name}, solve {problem}",
            n=1,
            id_prefix="render-test",
            vars={"name": "Alice", "problem": "2+2"},
        )
        result = list(source)

        assert len(result) == 1
        user_message = result[0].messages[1]["content"]
        assert user_message == "Hello Alice, solve 2+2"

    def test_orchestrator_works_with_template_source_and_resume(self, temp_jsonl):
        """Resume=true skips existing IDs, no LLM calls for processed items."""
        from src.data_gen.orchestrator import Orchestrator
        from src.data_gen.prompt_source import TemplatePromptSource

        source = TemplatePromptSource(
            system="You are a math tutor.",
            template="What is {problem}?",
            n=2,
            id_prefix="resume-test",
            vars={"problem": "1+1"},
        )

        # Track generate calls
        call_count = 0

        class CountingClient:
            def generate(self, prompts):
                nonlocal call_count
                call_count += len(prompts)
                return [f"response_{p.id}" for p in prompts]

        # First run - should process both items
        orchestrator = Orchestrator(
            client=CountingClient(),
            batch_size=10,
            output_path=temp_jsonl,
            resume=False,
        )
        list(orchestrator.run(source))

        assert call_count == 2

        # Verify 2 lines in JSONL
        with open(temp_jsonl) as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Second run with resume=True - should skip all
        call_count = 0
        source2 = TemplatePromptSource(
            system="You are a math tutor.",
            template="What is {problem}?",
            n=2,
            id_prefix="resume-test",
            vars={"problem": "1+1"},
        )
        orchestrator2 = Orchestrator(
            client=CountingClient(),
            batch_size=10,
            output_path=temp_jsonl,
            resume=True,
        )
        list(orchestrator2.run(source2))

        # No new calls should be made
        assert call_count == 0

        # Still 2 lines (no duplicates)
        with open(temp_jsonl) as f:
            lines = f.readlines()
        assert len(lines) == 2
