"""Provider-neutral chat gateway for common model APIs.

The graph compiler never depends on this module. It is an optional adapter layer
for agents that want to send compiled context to a model without adopting a
provider SDK. The transport uses only the Python standard library.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """Raised for invalid provider configuration or a provider API failure."""


def load_env_file(path: "os.PathLike[str] | str" = ".env") -> List[str]:
    """Populate `os.environ` from a `KEY=value` file, returning the names set.

    Benchmark runs are launched as fresh subprocesses, so an `export` in an
    interactive shell does not reach them. A gitignored `.env` beside the
    repository is the one place a credential can live and still be visible to every
    run. Existing environment variables always win, and only names are ever
    returned — never values, so a caller cannot log the secret by accident.
    """
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        return []
    loaded: List[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            name = name.strip()
            if name.startswith("export "):
                name = name[len("export "):].strip()
            value = value.strip().strip('"').strip("'")
            if not name or name in os.environ:
                continue
            os.environ[name] = value
            loaded.append(name)
    return loaded


@dataclass(frozen=True)
class ProviderConfig:
    """Normalized model endpoint and authentication settings."""

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: str = "2023-06-01"
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "ProviderConfig":
        env = dict(os.environ if environ is None else environ)
        provider = env.get("AI_PROVIDER", "openai-compatible").strip().lower()
        key_names = {
            "openai": "OPENAI_API_KEY",
            "openai-compatible": "AI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "ollama": "OLLAMA_API_KEY",
        }
        key_name = key_names.get(provider, "AI_API_KEY")
        api_key = env.get(key_name) or env.get("AI_API_KEY")
        defaults = {
            "openai": ("gpt-4o-mini", "https://api.openai.com/v1/chat/completions"),
            "openai-compatible": ("gpt-4o-mini", "http://localhost:8000/v1/chat/completions"),
            "deepseek": ("deepseek-chat", "https://api.deepseek.com/chat/completions"),
            "openrouter": ("openai/gpt-4o-mini", "https://openrouter.ai/api/v1/chat/completions"),
            "groq": ("llama-3.3-70b-versatile", "https://api.groq.com/openai/v1/chat/completions"),
            "anthropic": ("claude-3-5-sonnet-latest", "https://api.anthropic.com/v1/messages"),
            "gemini": ("gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/models"),
            "ollama": ("llama3.2", "http://localhost:11434/v1/chat/completions"),
        }
        if provider not in defaults:
            raise ProviderError("unsupported AI_PROVIDER {!r}".format(provider))
        model, base_url = defaults[provider]
        return cls(
            provider=provider,
            model=env.get("AI_MODEL", model),
            api_key=api_key,
            base_url=env.get("AI_BASE_URL", base_url),
            api_version=env.get("ANTHROPIC_VERSION", "2023-06-01"),
            timeout_seconds=float(env.get("AI_TIMEOUT_SECONDS", "120")),
        )


@dataclass(frozen=True)
class Usage:
    """Provider-independent token accounting."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None


@dataclass(frozen=True)
class ChatResponse:
    """Normalized completion response."""

    text: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    raw: Mapping[str, Any] = field(default_factory=dict)


JsonTransport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


def _default_transport(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise ProviderError("model request failed: {}".format(error)) from error


class ModelClient:
    """Send normalized chat requests through a configured provider adapter."""

    def __init__(self, config: ProviderConfig, transport: Optional[JsonTransport] = None):
        self.config = config
        self._transport = transport or _default_transport

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        prepared = list(messages)
        if system:
            prepared.insert(0, {"role": "system", "content": system})
        if self.config.provider == "anthropic":
            return self._anthropic(prepared, max_tokens, temperature)
        if self.config.provider == "gemini":
            return self._gemini(prepared, max_tokens, temperature)
        return self._openai_compatible(prepared, max_tokens, temperature)

    def _openai_compatible(self, messages: Sequence[Mapping[str, Any]], max_tokens: Optional[int], temperature: Optional[float]) -> ChatResponse:
        payload: Dict[str, Any] = {"model": self.config.model, "messages": list(messages), "stream": False}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"Authorization": "Bearer {}".format(self.config.api_key)} if self.config.api_key else {}
        if self.config.provider == "openrouter":
            headers.update({"HTTP-Referer": "graph-context-compiler", "X-Title": "Graph Context Compiler"})
        data = self._transport(self.config.base_url or "", headers, payload, self.config.timeout_seconds)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content", "")
        if isinstance(text, list):
            text = "".join(str(item.get("text", "")) for item in text if isinstance(item, dict))
        return ChatResponse(
            text=str(text),
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            usage=_usage_from_openai(data.get("usage") or {}),
            raw=data,
        )

    def _anthropic(self, messages: Sequence[Mapping[str, Any]], max_tokens: Optional[int], temperature: Optional[float]) -> ChatResponse:
        system_parts = [str(message["content"]) for message in messages if message.get("role") == "system"]
        non_system = [message for message in messages if message.get("role") != "system"]
        payload: Dict[str, Any] = {"model": self.config.model, "messages": non_system, "max_tokens": max_tokens or 1024}
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"x-api-key": self.config.api_key or "", "anthropic-version": self.config.api_version}
        data = self._transport(self.config.base_url or "", headers, payload, self.config.timeout_seconds)
        text = "".join(str(block.get("text", "")) for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage") or {}
        return ChatResponse(text=text, model=data.get("model"), finish_reason=data.get("stop_reason"), usage=Usage(usage.get("input_tokens"), usage.get("output_tokens")), raw=data)

    def _gemini(self, messages: Sequence[Mapping[str, Any]], max_tokens: Optional[int], temperature: Optional[float]) -> ChatResponse:
        contents: List[Dict[str, Any]] = []
        for message in messages:
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": str(message.get("content", ""))}]})
        payload: Dict[str, Any] = {"contents": contents}
        generation: Dict[str, Any] = {}
        if max_tokens is not None:
            generation["maxOutputTokens"] = max_tokens
        if temperature is not None:
            generation["temperature"] = temperature
        if generation:
            payload["generationConfig"] = generation
        base = (self.config.base_url or "").rstrip("/")
        url = "{}/{:s}:generateContent?{}".format(base, self.config.model, urlencode({"key": self.config.api_key or ""}))
        data = self._transport(url, {}, payload, self.config.timeout_seconds)
        candidates = data.get("candidates") or [{}]
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        metadata = data.get("usageMetadata") or {}
        return ChatResponse(text=text, model=self.config.model, finish_reason=candidates[0].get("finishReason"), usage=Usage(metadata.get("promptTokenCount"), metadata.get("candidatesTokenCount"), metadata.get("cachedContentTokenCount")), raw=data)


def _usage_from_openai(usage: Mapping[str, Any]) -> Usage:
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
    prompt = usage.get("prompt_tokens")
    input_tokens = prompt - cached if isinstance(prompt, int) and isinstance(cached, int) else prompt
    return Usage(input_tokens, usage.get("completion_tokens"), cached, usage.get("prompt_cache_write_tokens"))
