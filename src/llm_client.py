"""Thin LLM client around the Anthropic Messages API.

Two modes:
* **Online** — uses the `anthropic` SDK when ``ANTHROPIC_API_KEY`` is set.
* **Offline** — a deterministic, template-based generator (see diagnosis.py)
  is used when no key is present, so the whole POC runs end-to-end for grading
  and never hard-fails during a demo. Because the offline path only restates
  detector facts, it is also a useful illustration of the hallucination-control
  principle: prose is derived strictly from verified numbers.

Switching providers (OpenAI, Gemini, local) means editing only this file.
"""
from __future__ import annotations

from . import config


def is_online() -> bool:
    """True when a real LLM call can be made."""
    if not config.ANTHROPIC_API_KEY:
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def complete(system: str, messages: list[dict],
             model: str | None = None,
             max_tokens: int | None = None,
             temperature: float | None = None) -> str:
    """Call the Messages API and return the concatenated text content.

    `messages` is a list of {"role": "user"|"assistant", "content": str}.
    Raises RuntimeError if called while offline (callers should check is_online).
    """
    if not is_online():
        raise RuntimeError("LLM offline: no ANTHROPIC_API_KEY / anthropic SDK.")

    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model or config.DEFAULT_MODEL,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()
