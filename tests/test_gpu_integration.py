"""GPU integration tests that require actual GPU hardware.

Run with: pytest tests/test_gpu_integration.py -m gpu
Skip with: pytest tests/ -m "not gpu"
"""

import json

import pytest

from src.data_gen import Orchestrator, PromptItem, TemplatedPromptSource, VLLMClient

# Check for GPU availability
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
class TestVLLMOnGPU:
    """Integration tests that run on actual GPU."""

    @pytest.fixture
    def small_model_client(self):
        """Create a VLLMClient with small model for V100 testing."""
        return VLLMClient(
            model="Qwen/Qwen2.5-Math-1.5B-Instruct",
            tensor_parallel_size=2,
            dtype="float16",
            max_model_len=2048,
            temperature=0.7,
            max_tokens=256,
        )

    def test_loads_small_model(self, small_model_client):
        """VLLMClient can load 1.5B model on V100."""
        small_model_client._ensure_initialized()
        assert small_model_client._llm is not None
        assert small_model_client._sampling_params is not None

    def test_generates_non_empty_output(self, small_model_client):
        """Generate returns non-empty completions."""
        prompt = PromptItem(
            id="test",
            messages=[
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": "What is 2+2? Answer briefly."},
            ],
        )
        results = small_model_client.generate([prompt])

        assert len(results) == 1
        assert len(results[0]) > 0
        assert isinstance(results[0], str)

    def test_generates_multiple_prompts(self, small_model_client):
        """Generate handles multiple prompts in a batch."""
        prompts = [
            PromptItem(
                id=f"test_{i}",
                messages=[
                    {"role": "user", "content": f"What is {i}+{i}? Answer with just the number."},
                ],
            )
            for i in range(3)
        ]
        results = small_model_client.generate(prompts)

        assert len(results) == 3
        assert all(len(r) > 0 for r in results)
        assert all(isinstance(r, str) for r in results)


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
class TestEndToEndPipeline:
    """End-to-end pipeline tests with real vLLM."""

    def test_orchestrator_with_vllm(self, tmp_path):
        """Full pipeline with Orchestrator and VLLMClient."""
        output_file = tmp_path / "test_output.jsonl"

        client = VLLMClient(
            model="Qwen/Qwen2.5-Math-1.5B-Instruct",
            tensor_parallel_size=2,
            dtype="float16",
            max_model_len=2048,
            temperature=0.7,
            max_tokens=256,
        )

        orchestrator = Orchestrator(
            client=client,
            batch_size=2,
            output_path=output_file,
            resume=False,
            generator_info={"model": "test-model"},
        )

        prompts = [
            PromptItem(
                id=f"e2e_test_{i}",
                messages=[{"role": "user", "content": f"What is {i}*2?"}],
            )
            for i in range(3)
        ]

        results = list(orchestrator.run(iter(prompts)))

        # Verify results
        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "raw_completion" in r
            assert len(r["raw_completion"]) > 0

        # Verify JSONL output
        assert output_file.exists()
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

        for line in lines:
            record = json.loads(line)
            assert "schema_version" in record
            assert "raw_completion" in record

    def test_templated_prompt_source_with_vllm(self, tmp_path):
        """TemplatedPromptSource works with VLLMClient."""
        output_file = tmp_path / "templated_output.jsonl"

        prompt_source = TemplatedPromptSource(
            template="What is {num1} + {num2}? Answer briefly.",
            system_prefix="You are a math tutor.",
            format_system="Give a short answer.",
            diversity_config={
                "num1": {"choice": [1, 2, 3]},
                "num2": {"choice": [10, 20]},
            },
            n=2,
            seed=42,
        )

        client = VLLMClient(
            model="Qwen/Qwen2.5-Math-1.5B-Instruct",
            tensor_parallel_size=2,
            dtype="float16",
            max_model_len=2048,
            temperature=0.7,
            max_tokens=128,
        )

        orchestrator = Orchestrator(
            client=client,
            batch_size=2,
            output_path=output_file,
            resume=False,
        )

        results = list(orchestrator.run(prompt_source))

        assert len(results) == 2
        for r in results:
            assert "raw_completion" in r
            assert len(r["raw_completion"]) > 0

        # Verify output file
        assert output_file.exists()
