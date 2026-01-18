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


class TestDiversitySampling:
    """Tests for TemplatedPromptSource diversity sampling."""

    def test_sample_deterministic_with_seed(self):
        """Same seed produces same sequence of samples."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        diversity_config = {
            "primary_technique": {
                "choice": ["AM-GM", "Cauchy-Schwarz", "Jensen"],
                "weights": [0.5, 0.3, 0.2],
            },
            "num_vars": {
                "choice": [2, 3, 4],
                "weights": [0.4, 0.4, 0.2],
            },
        }

        # Create first instance and sample
        source1 = TemplatedPromptSource(
            template="test",
            system_prefix="test",
            format_system="test",
            diversity_config=diversity_config,
            n=10,
            seed=42,
        )
        samples1 = [source1._sample_attributes() for _ in range(5)]

        # Create second instance with same seed
        source2 = TemplatedPromptSource(
            template="test",
            system_prefix="test",
            format_system="test",
            diversity_config=diversity_config,
            n=10,
            seed=42,
        )
        samples2 = [source2._sample_attributes() for _ in range(5)]

        assert samples1 == samples2

    def test_sample_all_attributes(self):
        """All keys in diversity config appear in sampled output."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        diversity_config = {
            "primary_technique": {"choice": ["AM-GM"], "weights": [1.0]},
            "secondary_technique": {"choice": ["Cauchy-Schwarz", None], "weights": [0.5, 0.5]},
            "num_vars": {"choice": [2, 3, 4]},
            "difficulty": {"choice": ["easy", "medium", "hard"]},
        }

        source = TemplatedPromptSource(
            template="test",
            system_prefix="test",
            format_system="test",
            diversity_config=diversity_config,
            n=10,
            seed=0,
        )

        sample = source._sample_attributes()

        assert set(sample.keys()) == {"primary_technique", "secondary_technique", "num_vars", "difficulty"}

    def test_sample_respects_choices(self):
        """Sampled values are always from the choice list."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        diversity_config = {
            "technique": {"choice": ["AM-GM", "Cauchy-Schwarz", "Jensen"]},
            "num_vars": {"choice": [2, 3, 4, 5]},
        }

        source = TemplatedPromptSource(
            template="test",
            system_prefix="test",
            format_system="test",
            diversity_config=diversity_config,
            n=10,
            seed=123,
        )

        for _ in range(100):
            sample = source._sample_attributes()
            assert sample["technique"] in ["AM-GM", "Cauchy-Schwarz", "Jensen"]
            assert sample["num_vars"] in [2, 3, 4, 5]

    def test_sample_handles_null_choice(self):
        """null in choices (like secondary_technique) works correctly."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        diversity_config = {
            "secondary_technique": {
                "choice": ["Cauchy-Schwarz", None],
                "weights": [0.5, 0.5],
            },
        }

        source = TemplatedPromptSource(
            template="test",
            system_prefix="test",
            format_system="test",
            diversity_config=diversity_config,
            n=10,
            seed=0,
        )

        # Sample until we get None
        found_none = False
        found_value = False
        for _ in range(100):
            sample = source._sample_attributes()
            if sample["secondary_technique"] is None:
                found_none = True
            else:
                found_value = True
            if found_none and found_value:
                break

        assert found_none, "None should be a valid sampled value"
        assert found_value, "Non-None values should also be sampled"

    def test_sample_weights_affect_distribution(self):
        """Higher weight = more frequent (statistical test over many samples)."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        diversity_config = {
            "technique": {
                "choice": ["heavy", "light"],
                "weights": [0.99, 0.01],
            },
        }

        source = TemplatedPromptSource(
            template="test",
            system_prefix="test",
            format_system="test",
            diversity_config=diversity_config,
            n=10,
            seed=42,
        )

        heavy_count = 0
        total = 1000
        for _ in range(total):
            sample = source._sample_attributes()
            if sample["technique"] == "heavy":
                heavy_count += 1

        # With 0.99 weight, we expect >90% to be "heavy"
        assert heavy_count > 900, f"Expected >900 'heavy' samples but got {heavy_count}"


class TestTemplateFilling:
    """Tests for TemplatedPromptSource template filling."""

    def test_fill_template_basic(self):
        """Basic template filling with a single placeholder."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="Generate a problem using {technique}.",
            system_prefix="",
            format_system="",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._fill_template({"technique": "AM-GM"})
        assert result == "Generate a problem using AM-GM."

    def test_fill_template_all_attributes(self):
        """Template filling with multiple attributes."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="Use {technique} with {num_vars} variables at {difficulty} difficulty.",
            system_prefix="",
            format_system="",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._fill_template({
            "technique": "Cauchy-Schwarz",
            "num_vars": 3,
            "difficulty": "medium",
        })
        assert result == "Use Cauchy-Schwarz with 3 variables at medium difficulty."

    def test_fill_template_null_renders_as_none(self):
        """None values render as 'None' string in template."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="Primary: {primary}, Secondary: {secondary}",
            system_prefix="",
            format_system="",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._fill_template({"primary": "AM-GM", "secondary": None})
        assert result == "Primary: AM-GM, Secondary: None"

    def test_fill_template_preserves_formatting(self):
        """Template filling preserves newlines and special characters."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="Step 1: {step1}\n\nStep 2: {step2}",
            system_prefix="",
            format_system="",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._fill_template({"step1": "Initialize", "step2": "Process"})
        assert result == "Step 1: Initialize\n\nStep 2: Process"

    def test_fill_template_integer_values(self):
        """Integer values are converted to strings in template."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="Use {num_vars} variables.",
            system_prefix="",
            format_system="",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._fill_template({"num_vars": 4})
        assert result == "Use 4 variables."


class TestSystemComposition:
    """Tests for TemplatedPromptSource system message composition."""

    def test_compose_system_message(self):
        """Basic system message composition with prefix and format."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="",
            system_prefix="You are a math olympiad expert.",
            format_system="Respond in JSON format.",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._compose_system_message()
        assert result == "You are a math olympiad expert.\n\nRespond in JSON format."

    def test_compose_with_newline_separation(self):
        """Parts are separated by double newlines."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="",
            system_prefix="Part A",
            format_system="Part B",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._compose_system_message()
        assert "\n\n" in result
        assert result.count("\n\n") == 1

    def test_compose_strips_trailing_whitespace(self):
        """Trailing whitespace is stripped from both parts."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="",
            system_prefix="Prefix with trailing space   ",
            format_system="Format with trailing newlines\n\n",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._compose_system_message()
        assert result == "Prefix with trailing space\n\nFormat with trailing newlines"

    def test_compose_preserves_internal_formatting(self):
        """Internal newlines and formatting are preserved."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="",
            system_prefix="Line 1\nLine 2\nLine 3",
            format_system="Format line 1\nFormat line 2",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._compose_system_message()
        assert "Line 1\nLine 2\nLine 3" in result
        assert "Format line 1\nFormat line 2" in result

    def test_compose_empty_system_prefix(self):
        """Empty system_prefix results in only format_system."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="",
            system_prefix="",
            format_system="Format only",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._compose_system_message()
        assert result == "Format only"

    def test_compose_empty_format_system(self):
        """Empty format_system results in only system_prefix."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        source = TemplatedPromptSource(
            template="",
            system_prefix="Prefix only",
            format_system="",
            diversity_config={},
            n=1,
            seed=0,
        )

        result = source._compose_system_message()
        assert result == "Prefix only"


class TestTemplatedPromptSourceIntegration:
    """Integration tests for TemplatedPromptSource __iter__ method."""

    @pytest.fixture
    def basic_source(self):
        """Create a basic TemplatedPromptSource for testing."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        return TemplatedPromptSource(
            template="Generate a {difficulty} problem using {technique}.",
            system_prefix="You are a math expert.",
            format_system="Respond in JSON.",
            diversity_config={
                "technique": {"choice": ["AM-GM", "Cauchy-Schwarz"], "weights": [0.5, 0.5]},
                "difficulty": {"choice": ["easy", "medium", "hard"]},
            },
            n=5,
            seed=42,
        )

    def test_yields_correct_count(self, basic_source):
        """Source yields exactly n items."""
        items = list(basic_source)
        assert len(items) == 5

    def test_prompt_item_structure(self, basic_source):
        """Each yielded item is a PromptItem with required attributes."""
        from src.data_gen.models import PromptItem

        items = list(basic_source)
        for item in items:
            assert isinstance(item, PromptItem)
            assert hasattr(item, "id")
            assert hasattr(item, "messages")
            assert hasattr(item, "meta")

    def test_prompt_ids_are_unique(self, basic_source):
        """All prompt IDs are unique."""
        items = list(basic_source)
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids))

    def test_prompt_id_format(self, basic_source):
        """Prompt IDs follow the format prompt_{seed}_{index:06d}."""
        items = list(basic_source)
        assert items[0].id == "prompt_42_000000"
        assert items[1].id == "prompt_42_000001"
        assert items[4].id == "prompt_42_000004"

    def test_messages_have_system_and_user(self, basic_source):
        """Each item has both system and user messages."""
        items = list(basic_source)
        for item in items:
            assert len(item.messages) == 2
            assert item.messages[0]["role"] == "system"
            assert item.messages[1]["role"] == "user"

    def test_user_message_is_filled_template(self, basic_source):
        """User message content is the filled template."""
        items = list(basic_source)
        for item in items:
            user_content = item.messages[1]["content"]
            # Should contain the template structure
            assert "Generate a" in user_content
            assert "problem using" in user_content
            # Should have filled in values
            assert "{difficulty}" not in user_content
            assert "{technique}" not in user_content

    def test_system_message_is_composed(self, basic_source):
        """System message is composed from prefix and format."""
        items = list(basic_source)
        for item in items:
            system_content = item.messages[0]["content"]
            assert "You are a math expert." in system_content
            assert "Respond in JSON." in system_content

    def test_meta_contains_sampled_attributes(self, basic_source):
        """Meta contains the sampled attribute values."""
        items = list(basic_source)
        for item in items:
            assert "technique" in item.meta
            assert "difficulty" in item.meta
            assert item.meta["technique"] in ["AM-GM", "Cauchy-Schwarz"]
            assert item.meta["difficulty"] in ["easy", "medium", "hard"]

    def test_meta_contains_prompt_index(self, basic_source):
        """Meta contains the index of the prompt."""
        items = list(basic_source)
        for i, item in enumerate(items):
            assert item.meta["index"] == i

    def test_meta_contains_seed(self, basic_source):
        """Meta contains the seed used for generation."""
        items = list(basic_source)
        for item in items:
            assert item.meta["seed"] == 42

    def test_deterministic_with_seed(self):
        """Same seed produces identical sequences."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        config = {
            "technique": {"choice": ["A", "B", "C"], "weights": [0.33, 0.33, 0.34]},
        }

        source1 = TemplatedPromptSource(
            template="{technique}",
            system_prefix="",
            format_system="",
            diversity_config=config,
            n=10,
            seed=123,
        )

        source2 = TemplatedPromptSource(
            template="{technique}",
            system_prefix="",
            format_system="",
            diversity_config=config,
            n=10,
            seed=123,
        )

        items1 = list(source1)
        items2 = list(source2)

        for i1, i2 in zip(items1, items2):
            assert i1.id == i2.id
            assert i1.messages == i2.messages
            assert i1.meta == i2.meta

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds produce different sequences."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        config = {
            "technique": {"choice": ["A", "B", "C", "D", "E"]},
        }

        source1 = TemplatedPromptSource(
            template="{technique}",
            system_prefix="",
            format_system="",
            diversity_config=config,
            n=20,
            seed=1,
        )

        source2 = TemplatedPromptSource(
            template="{technique}",
            system_prefix="",
            format_system="",
            diversity_config=config,
            n=20,
            seed=2,
        )

        items1 = list(source1)
        items2 = list(source2)

        # At least some items should differ
        metas1 = [item.meta["technique"] for item in items1]
        metas2 = [item.meta["technique"] for item in items2]
        assert metas1 != metas2

    def test_iteration_resets_rng_state(self):
        """Multiple iterations over same source produce identical results."""
        from src.data_gen.prompt_source import TemplatedPromptSource

        config = {
            "technique": {"choice": ["A", "B", "C"]},
        }

        source = TemplatedPromptSource(
            template="{technique}",
            system_prefix="",
            format_system="",
            diversity_config=config,
            n=10,
            seed=42,
        )

        # First iteration
        items1 = list(source)

        # Second iteration - should reset RNG and produce same results
        items2 = list(source)

        for i1, i2 in zip(items1, items2):
            assert i1.id == i2.id
            assert i1.messages == i2.messages
            assert i1.meta == i2.meta


class TestTemplatedSourceEndToEnd:
    """End-to-end tests for TemplatedPromptSource with full pipeline."""

    def test_full_pipeline_with_templated_source(self, temp_jsonl):
        """Full pipeline: TemplatedPromptSource -> FakeLLMClient -> Orchestrator -> JSONL."""
        from src.data_gen.client import FakeLLMClient
        from src.data_gen.orchestrator import Orchestrator
        from src.data_gen.prompt_source import TemplatedPromptSource

        # Create a templated source
        source = TemplatedPromptSource(
            template="Create a {difficulty} inequality problem using {technique}.",
            system_prefix="You are a math olympiad problem generator.",
            format_system="Output valid JSON only.",
            diversity_config={
                "technique": {
                    "choice": ["AM-GM", "Cauchy-Schwarz", "Jensen"],
                    "weights": [0.5, 0.3, 0.2],
                },
                "difficulty": {
                    "choice": ["easy", "medium", "hard"],
                },
            },
            n=3,
            seed=42,
        )

        # Create client and orchestrator
        client = FakeLLMClient(response_template="Generated problem for {id}")
        orchestrator = Orchestrator(
            client=client,
            batch_size=2,
            output_path=temp_jsonl,
            resume=False,
        )

        # Run the pipeline
        results = list(orchestrator.run(source))

        # Verify results count
        assert len(results) == 3

        # Read and verify JSONL output
        import json
        records = []
        with open(temp_jsonl) as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 3

        # Verify structure
        for record in records:
            assert record["schema_version"] == 1
            assert record["id"].startswith("prompt_42_")
            assert len(record["messages"]) == 2
            assert record["messages"][0]["role"] == "system"
            assert record["messages"][1]["role"] == "user"
            assert "technique" in record["meta"]
            assert "difficulty" in record["meta"]
            assert "index" in record["meta"]
            assert "seed" in record["meta"]
            assert "Generated problem" in record["raw_completion"]

        # Verify IDs are unique and correctly formatted
        ids = [r["id"] for r in records]
        assert ids == ["prompt_42_000000", "prompt_42_000001", "prompt_42_000002"]


class TestVLLMClientConfig:
    """Tests for VLLMClient configuration and parameter handling."""

    def test_accepts_engine_params(self):
        """tensor_parallel_size, dtype, max_model_len are accepted as engine kwargs."""
        from src.data_gen.client import VLLMClient

        client = VLLMClient(
            model="test-model",
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=8192,
        )

        assert client.model == "test-model"
        assert client.engine_kwargs["tensor_parallel_size"] == 4
        assert client.engine_kwargs["dtype"] == "bfloat16"
        assert client.engine_kwargs["max_model_len"] == 8192

    def test_accepts_sampling_params(self):
        """temperature, top_p, max_tokens from sampling config."""
        from src.data_gen.client import VLLMClient

        client = VLLMClient(
            model="test-model",
            temperature=0.75,
            top_p=0.95,
            max_tokens=1600,
        )

        assert client.temperature == 0.75
        assert client.top_p == 0.95
        assert client.max_tokens == 1600

    def test_sampling_params_used_in_generate(self):
        """SamplingParams object has correct values (mock vllm)."""
        from unittest.mock import MagicMock, patch

        from src.data_gen.client import VLLMClient
        from src.data_gen.models import PromptItem

        client = VLLMClient(
            model="test-model",
            temperature=0.75,
            top_p=0.95,
            max_tokens=1600,
            tensor_parallel_size=4,
        )

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
            # Force re-initialization
            client._llm = None
            client._sampling_params = None

            prompts = [PromptItem(id="test", messages=[{"role": "user", "content": "Hello"}])]
            client.generate(prompts)

            # Verify LLM was called with engine params
            mock_llm_class.assert_called_once_with(model="test-model", tensor_parallel_size=4)

            # Verify SamplingParams was called with correct values
            mock_sampling_params_class.assert_called_once_with(
                temperature=0.75,
                max_tokens=1600,
                top_p=0.95,
            )

    def test_backwards_compatible(self):
        """Old-style flat params still work for existing code."""
        from src.data_gen.client import VLLMClient

        # This is how the client was created before
        client = VLLMClient(
            model="test-model",
            temperature=0.7,
            max_tokens=512,
        )

        assert client.model == "test-model"
        assert client.temperature == 0.7
        assert client.max_tokens == 512
        assert client.top_p == 1.0  # Default value
        assert client.engine_kwargs == {}

    def test_default_values(self):
        """Default values are set correctly."""
        from src.data_gen.client import VLLMClient

        client = VLLMClient(model="test-model")

        assert client.temperature == 0.7
        assert client.max_tokens == 512
        assert client.top_p == 1.0
        assert client.engine_kwargs == {}
