"""Unit tests for VLLMClient (no GPU required - uses mocking)."""

from unittest.mock import MagicMock

import pytest

from src.data_gen import PromptItem, VLLMClient


class TestVLLMClientInit:
    """Test VLLMClient initialization (no GPU needed - lazy loading)."""

    def test_stores_model_name(self):
        """VLLMClient stores the model name."""
        client = VLLMClient(model="test/model")
        assert client.model == "test/model"

    def test_stores_sampling_params(self):
        """VLLMClient stores sampling parameters."""
        client = VLLMClient(model="m", temperature=0.5, max_tokens=100, top_p=0.9)
        assert client.temperature == 0.5
        assert client.max_tokens == 100
        assert client.top_p == 0.9

    def test_stores_engine_kwargs(self):
        """VLLMClient stores engine kwargs for vLLM."""
        client = VLLMClient(
            model="m",
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=8192,
        )
        assert client.engine_kwargs["tensor_parallel_size"] == 4
        assert client.engine_kwargs["dtype"] == "bfloat16"
        assert client.engine_kwargs["max_model_len"] == 8192

    def test_not_initialized_on_creation(self):
        """vLLM should NOT be imported until generate() is called."""
        client = VLLMClient(model="test/model")
        assert client._llm is None
        assert client._sampling_params is None

    def test_default_sampling_params(self):
        """VLLMClient has sensible default sampling parameters."""
        client = VLLMClient(model="m")
        assert client.temperature == 0.7
        assert client.max_tokens == 512
        assert client.top_p == 1.0


class TestVLLMClientLazyInit:
    """Test lazy initialization behavior (mocked)."""

    def test_ensure_initialized_imports_vllm(self, mocker):
        """_ensure_initialized imports and creates vLLM objects."""
        mock_llm = mocker.patch("vllm.LLM")
        mock_params = mocker.patch("vllm.SamplingParams")

        client = VLLMClient(model="test/model", temperature=0.7)
        client._ensure_initialized()

        mock_llm.assert_called_once_with(model="test/model")
        mock_params.assert_called_once_with(temperature=0.7, max_tokens=512, top_p=1.0)

    def test_passes_engine_kwargs_to_llm(self, mocker):
        """Engine kwargs are passed to LLM constructor."""
        mock_llm = mocker.patch("vllm.LLM")
        mocker.patch("vllm.SamplingParams")

        client = VLLMClient(
            model="m",
            tensor_parallel_size=2,
            dtype="float16",
        )
        client._ensure_initialized()

        mock_llm.assert_called_once_with(
            model="m",
            tensor_parallel_size=2,
            dtype="float16",
        )

    def test_only_initializes_once(self, mocker):
        """Multiple calls to _ensure_initialized only initialize once."""
        mock_llm = mocker.patch("vllm.LLM")
        mocker.patch("vllm.SamplingParams")

        client = VLLMClient(model="m")
        client._ensure_initialized()
        client._ensure_initialized()
        client._ensure_initialized()

        assert mock_llm.call_count == 1


class TestVLLMClientGenerate:
    """Test generate method (mocked)."""

    def test_calls_chat_with_conversations(self, mocker):
        """generate() calls chat() with conversations extracted from prompts."""
        mock_llm_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="response")]
        mock_llm_instance.chat.return_value = [mock_output]

        mocker.patch("vllm.LLM", return_value=mock_llm_instance)
        mocker.patch("vllm.SamplingParams")

        client = VLLMClient(model="m")
        prompts = [PromptItem(id="1", messages=[{"role": "user", "content": "hi"}])]

        results = client.generate(prompts)

        mock_llm_instance.chat.assert_called_once()
        assert results == ["response"]

    def test_extracts_messages_from_prompt_items(self, mocker):
        """generate() extracts messages from PromptItem objects."""
        mock_llm_instance = MagicMock()
        mock_output1 = MagicMock()
        mock_output1.outputs = [MagicMock(text="reply1")]
        mock_output2 = MagicMock()
        mock_output2.outputs = [MagicMock(text="reply2")]
        mock_llm_instance.chat.return_value = [mock_output1, mock_output2]

        mocker.patch("vllm.LLM", return_value=mock_llm_instance)
        mocker.patch("vllm.SamplingParams")

        client = VLLMClient(model="m")
        prompts = [
            PromptItem(id="1", messages=[{"role": "user", "content": "q1"}]),
            PromptItem(id="2", messages=[{"role": "user", "content": "q2"}]),
        ]

        results = client.generate(prompts)

        # Verify conversations were passed correctly
        call_args = mock_llm_instance.chat.call_args
        conversations = call_args[0][0]
        assert len(conversations) == 2
        assert conversations[0] == [{"role": "user", "content": "q1"}]
        assert conversations[1] == [{"role": "user", "content": "q2"}]

        assert results == ["reply1", "reply2"]

    def test_returns_list_of_strings(self, mocker):
        """generate() returns a list of completion strings."""
        mock_llm_instance = MagicMock()
        mock_outputs = []
        for text in ["first", "second", "third"]:
            mock_out = MagicMock()
            mock_out.outputs = [MagicMock(text=text)]
            mock_outputs.append(mock_out)
        mock_llm_instance.chat.return_value = mock_outputs

        mocker.patch("vllm.LLM", return_value=mock_llm_instance)
        mocker.patch("vllm.SamplingParams")

        client = VLLMClient(model="m")
        prompts = [
            PromptItem(id=str(i), messages=[{"role": "user", "content": f"q{i}"}])
            for i in range(3)
        ]

        results = client.generate(prompts)

        assert results == ["first", "second", "third"]
        assert all(isinstance(r, str) for r in results)
