"""Model-provider abstraction for CASE-Bench.

A *candidate* model (the one being benchmarked) is addressed by a spec string of
the form ``provider:model_id``, e.g.::

    anthropic:claude-haiku-4-5
    anthropic:claude-sonnet-4-6
    openrouter:openai/gpt-4o-mini
    openrouter:meta-llama/llama-3.1-70b-instruct

``generate()`` returns the model's raw text completion. Generation is kept
deliberately provider-neutral (plain text in, plain text out, JSON requested via
the prompt) so that every candidate is asked for ideas in exactly the same way
and the comparison stays fair. The Anthropic-specific structured-output path is
reserved for the judge (see ``judge.py``), which is held fixed across all runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic


class ProviderError(RuntimeError):
    """Raised when a provider is misconfigured (e.g. missing API key)."""


@dataclass
class GenerationResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    temperature: float | None = None


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``provider:model_id`` into ``(provider, model_id)``.

    A bare model id with no provider prefix defaults to ``anthropic``.
    """
    if ":" in spec:
        provider, model_id = spec.split(":", 1)
    else:
        provider, model_id = "anthropic", spec
    return provider.strip().lower(), model_id.strip()


# --- Anthropic -------------------------------------------------------------

_anthropic_client: anthropic.Anthropic | None = None


def _anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Export it before running, e.g. "
                "`export ANTHROPIC_API_KEY=sk-ant-...` (get a key at console.anthropic.com)."
            )
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _generate_anthropic(
    model_id: str, system: str, user: str, max_tokens: int, temperature: float | None
) -> GenerationResult:
    kwargs = dict(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    used_temp = None
    if temperature is not None and not model_id.startswith(
        ("claude-opus-4-7", "claude-opus-4-8")
    ):
        kwargs["temperature"] = temperature
        used_temp = temperature
    resp = _anthropic().messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if b.type == "text")
    return GenerationResult(
        text=text,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        temperature=used_temp,
    )


# --- OpenRouter (implemented; requires OPENROUTER_API_KEY) -----------------


def _generate_openrouter(
    model_id: str, system: str, user: str, max_tokens: int, temperature: float | None
) -> GenerationResult:
    # Imported lazily so the package has no hard dependency on `requests`
    # unless an OpenRouter model is actually requested.
    try:
        import requests
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ProviderError(
            "The OpenRouter provider needs the `requests` package: pip install requests"
        ) from e

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ProviderError(
            "OPENROUTER_API_KEY is not set. Add it to your environment to "
            "benchmark OpenRouter-hosted models (gpt-*-mini, 7B/70B open models, ...)."
        )

    payload = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    return GenerationResult(
        text=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        temperature=temperature,
    )


_PROVIDERS = {
    "anthropic": _generate_anthropic,
    "openrouter": _generate_openrouter,
}


def generate(
    spec: str,
    system: str,
    user: str,
    max_tokens: int = 4000,
    temperature: float | None = None,
) -> GenerationResult:
    """Generate a completion from the candidate model named by ``spec``.

    ``temperature`` is silently dropped for models that don't accept it (e.g.
    Opus 4.7+), so callers can request sampling diversity uniformly.
    """
    provider, model_id = parse_spec(spec)
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ProviderError(
            f"Unknown provider {provider!r}. Supported: {', '.join(sorted(_PROVIDERS))}."
        )
    return fn(model_id, system, user, max_tokens, temperature)
