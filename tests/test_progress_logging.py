"""Tests for progress logging in VLLMClient and Orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestVLLMClientProgress:
    """Tests for VLLMClient progress messages."""

    def test_model_loading_prints_start_message(self, capsys):
        """VLLMClient prints when starting model load."""
        from src.data_gen.client import VLLMClient
        from src.data_gen.models import PromptItem

        client = VLLMClient(model="test-model")

        # Mock the vllm imports
        mock_llm_class = MagicMock()
        mock_sampling_params_class = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock the chat output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="response")]
        mock_llm_instance.chat.return_value = [mock_output]

        with patch.dict(
            "sys.modules",
            {"vllm": MagicMock(LLM=mock_llm_class, SamplingParams=mock_sampling_params_class)},
        ):
            prompts = [PromptItem(id="test", messages=[{"role": "user", "content": "Hello"}])]
            client.generate(prompts)

        captured = capsys.readouterr()
        assert "Loading model: test-model" in captured.out

    def test_model_loading_prints_complete_message(self, capsys):
        """VLLMClient prints when model load completes."""
        from src.data_gen.client import VLLMClient
        from src.data_gen.models import PromptItem

        client = VLLMClient(model="test-model")

        # Mock the vllm imports
        mock_llm_class = MagicMock()
        mock_sampling_params_class = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock the chat output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="response")]
        mock_llm_instance.chat.return_value = [mock_output]

        with patch.dict(
            "sys.modules",
            {"vllm": MagicMock(LLM=mock_llm_class, SamplingParams=mock_sampling_params_class)},
        ):
            prompts = [PromptItem(id="test", messages=[{"role": "user", "content": "Hello"}])]
            client.generate(prompts)

        captured = capsys.readouterr()
        assert "Model loaded: test-model" in captured.out


class TestOrchestratorProgress:
    """Tests for Orchestrator progress messages."""

    @pytest.fixture
    def temp_jsonl(self, tmp_path):
        """Return a path to a temporary JSONL file."""
        return tmp_path / "results.jsonl"

    def test_total_items_count(self, tmp_path, capsys):
        """Orchestrator prints total items to process."""
        from src.data_gen.client import FakeLLMClient
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        output_file = tmp_path / "output.jsonl"
        items = [
            PromptItem(id=f"test{i}", messages=[{"role": "user", "content": f"Hello {i}"}])
            for i in range(5)
        ]

        client = FakeLLMClient()
        orchestrator = Orchestrator(
            client=client,
            batch_size=2,
            output_path=output_file,
            resume=False,
        )
        list(orchestrator.run(items))

        captured = capsys.readouterr()
        assert "Total prompts: 5" in captured.out

    def test_resume_filtering_message(self, tmp_path, capsys):
        """Orchestrator prints resume filtering info."""
        from src.data_gen.client import FakeLLMClient
        from src.data_gen.io import append_jsonl
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        output_file = tmp_path / "output.jsonl"

        # Pre-populate with existing record
        existing_record = {
            "schema_version": 1,
            "id": "existing_1",
            "messages": [],
            "meta": {},
            "raw_completion": "old response",
        }
        append_jsonl(output_file, existing_record)

        items = [
            PromptItem(id="existing_1", messages=[{"role": "user", "content": "old"}]),
            PromptItem(id="new_1", messages=[{"role": "user", "content": "new"}]),
        ]

        client = FakeLLMClient()
        orchestrator = Orchestrator(
            client=client,
            batch_size=10,
            output_path=output_file,
            resume=True,
        )
        list(orchestrator.run(items))

        captured = capsys.readouterr()
        assert "Resume: found 1 existing records" in captured.out
        assert "Skipping 1 already-processed items" in captured.out

    def test_batch_progress_message(self, tmp_path, capsys):
        """Orchestrator prints 'Batch N/M' progress."""
        from src.data_gen.client import FakeLLMClient
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        output_file = tmp_path / "output.jsonl"
        items = [
            PromptItem(id=f"test{i}", messages=[{"role": "user", "content": f"Hello {i}"}])
            for i in range(5)
        ]

        client = FakeLLMClient()
        orchestrator = Orchestrator(
            client=client,
            batch_size=2,
            output_path=output_file,
            resume=False,
        )
        list(orchestrator.run(items))

        captured = capsys.readouterr()
        # With 5 items and batch_size=2, we expect 3 batches: [2, 2, 1]
        assert "Batch 1/3" in captured.out
        assert "Batch 2/3" in captured.out
        assert "Batch 3/3" in captured.out

    def test_completion_message(self, tmp_path, capsys):
        """Orchestrator prints completion message."""
        from src.data_gen.client import FakeLLMClient
        from src.data_gen.models import PromptItem
        from src.data_gen.orchestrator import Orchestrator

        output_file = tmp_path / "output.jsonl"
        items = [
            PromptItem(id=f"test{i}", messages=[{"role": "user", "content": f"Hello {i}"}])
            for i in range(3)
        ]

        client = FakeLLMClient()
        orchestrator = Orchestrator(
            client=client,
            batch_size=10,
            output_path=output_file,
            resume=False,
        )
        list(orchestrator.run(items))

        captured = capsys.readouterr()
        assert "Complete: 3 items written" in captured.out
