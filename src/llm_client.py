"""LLM access: an OpenAI-compatible client pointed at Groq, plus retry and a mock for offline work."""
import time
from types import SimpleNamespace

from src.config import GROQ_API_KEY, GROQ_BASE_URL, MOCK_LLM

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_NAMES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}


def is_retryable(exc: Exception) -> bool:
    """Rate limits, server errors, network errors and malformed tool calls are worth retrying."""
    if getattr(exc, "status_code", None) in RETRYABLE_STATUS:
        return True
    if type(exc).__name__ in RETRYABLE_ERROR_NAMES:
        return True
    text = str(exc).lower()
    return "tool_use_failed" in text or "failed to call a function" in text


def chat_with_retry(client, *, max_retries=3, backoff=1.0, sleep_fn=time.sleep, **kwargs):
    """Call the chat completions API, retrying transient failures with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt == max_retries or not is_retryable(exc):
                raise
            sleep_fn(backoff * (2 ** attempt))


class MockClient:
    """Offline stand-in for the LLM: calls describe_dataset once, then returns a canned reply."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, messages, **kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            call = SimpleNamespace(id="mock_1", function=SimpleNamespace(name="describe_dataset", arguments="{}"))
            message = SimpleNamespace(content=None, tool_calls=[call])
        else:
            message = SimpleNamespace(
                content="[mock mode] Set MOCK_LLM=false and add a Groq API key to get real answers.",
                tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def get_client():
    if MOCK_LLM:
        return MockClient()
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API is not set. Add it to .env or set MOCK_LLM=true for offline mode.")
    from openai import OpenAI  # imported lazily so the core package works without the SDK
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
