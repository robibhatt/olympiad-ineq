"""LLM client interfaces and implementations."""

from abc import ABC, abstractmethod

from src.data_gen.models import PromptItem


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompts: list[PromptItem]) -> list[str]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of PromptItem instances.

        Returns:
            List of completion strings, one per prompt.
        """
        pass


class FakeLLMClient(LLMClient):
    """A fake LLM client for testing.

    Attributes:
        response_template: Template string for responses. Can contain {id}.
    """

    def __init__(self, response_template: str = "Fake response for {id}"):
        """Initialize with a response template.

        Args:
            response_template: Template string with optional {id} placeholder.
        """
        self.response_template = response_template

    def generate(self, prompts: list[PromptItem]) -> list[str]:
        """Generate fake completions based on the template."""
        return [self.response_template.format(id=p.id) for p in prompts]


class VLLMClient(LLMClient):
    """A vLLM-based LLM client with lazy import.

    The vLLM library is only imported when generate() is first called,
    allowing tests to run without vLLM installed.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ):
        """Initialize the client (does not import vLLM yet).

        Args:
            model: HuggingFace model name or path.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional arguments passed to vLLM.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._llm = None
        self._sampling_params = None

    def _ensure_initialized(self):
        """Lazily initialize vLLM on first use."""
        if self._llm is None:
            from vllm import LLM, SamplingParams

            self._llm = LLM(model=self.model, **self.kwargs)
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def generate(self, prompts: list[PromptItem]) -> list[str]:
        """Generate completions using vLLM."""
        self._ensure_initialized()

        # Convert messages to text prompts for vLLM
        text_prompts = []
        for prompt in prompts:
            # Simple concatenation of messages
            text = ""
            for msg in prompt.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"<|{role}|>\n{content}\n"
            text += "<|assistant|>\n"
            text_prompts.append(text)

        outputs = self._llm.generate(text_prompts, self._sampling_params)
        return [output.outputs[0].text for output in outputs]
