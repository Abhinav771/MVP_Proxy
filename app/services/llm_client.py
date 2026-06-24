"""
llm_client.py — Universal LLM generation helper
================================================
Works with ANY provider that exposes an OpenAI-compatible API:

    Provider   | LLM_BASE_URL                          | LLM_MODEL example
    -----------|---------------------------------------|----------------------------
    Groq       | https://api.groq.com/openai/v1        | llama-3.3-70b-versatile
    OpenAI     | https://api.openai.com/v1             | gpt-4o
    Mistral    | https://api.mistral.ai/v1             | mistral-large-latest
    Together   | https://api.together.xyz/v1           | meta-llama/Llama-3-70b-chat-hf
    Ollama     | http://localhost:11434/v1             | llama3
    Perplexity | https://api.perplexity.ai             | llama-3.1-sonar-large-128k-online
    Anyscale   | https://api.endpoints.anyscale.com/v1 | meta-llama/Llama-2-70b-chat-hf

To switch provider: just update LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in your .env.
No code changes needed.
"""

import asyncio
from typing import AsyncGenerator, Generator

from openai import AsyncOpenAI, OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError


# ── Error handler (shared) ────────────────────────────────────────────────────

def _handle_openai_error(e: Exception) -> str:
    """Convert common OpenAI SDK errors into human-readable strings."""
    if isinstance(e, APITimeoutError):
        return "Error: Request timed out. Try again or increase timeout."
    if isinstance(e, APIStatusError):
        messages = {
            401: "Error: Invalid API key. Check LLM_API_KEY in .env.",
            429: "Error: Rate limit hit on provider side. Please wait.",
            500: "Error: LLM provider internal server error.",
            503: "Error: LLM provider is temporarily unavailable.",
        }
        return messages.get(e.status_code, f"Error: HTTP {e.status_code} — {e.message}")
    if isinstance(e, APIConnectionError):
        return "Error: Could not connect to LLM provider. Check LLM_BASE_URL."
    return f"Unexpected error: {str(e)}"


# ── 1. Synchronous streaming  ─────────────────────────────────────────────────

def stream_llm(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: int = 30,
) -> Generator[str, None, None]:
    """
    Stream tokens from any OpenAI-compatible LLM provider (synchronous).

    Args:
        base_url:      Provider's OpenAI-compatible endpoint.
        api_key:       API key for the provider.
        model_name:    Model identifier string.
        prompt:        User message / prompt.
        system_prompt: Optional system instruction prepended to the conversation.
        temperature:   Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens:    Max tokens to generate. None = provider default.
        timeout:       Request timeout in seconds.

    Yields:
        str: Token chunks as they arrive. Yields an error string on failure.

    Example:
        for token in stream_llm(base_url, api_key, model, "Tell me a joke"):
            print(token, end="", flush=True)
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except Exception as e:
        yield _handle_openai_error(e)


# ── 2. Asynchronous streaming  ────────────────────────────────────────────────

async def async_stream_llm(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: int = 30,
    collector: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from any OpenAI-compatible LLM provider (async).

    Same interface as stream_llm() but returns an async generator, suitable
    for use inside FastAPI async endpoints and asyncio coroutines.

    Args:
        base_url:      Provider's OpenAI-compatible endpoint.
        api_key:       API key for the provider.
        model_name:    Model identifier string.
        prompt:        User message / prompt.
        system_prompt: Optional system instruction prepended to the conversation.
        temperature:   Sampling temperature.
        max_tokens:    Max tokens to generate. None = provider default.
        timeout:       Request timeout in seconds.
        collector:     Optional single-element list. If provided, the complete
                       response text is appended to it once the stream finishes,
                       so the caller can access it without a second API call.
                       Pass an empty list: collector = []
                       Read after iteration: full_text = collector[0]

    Yields:
        str: Token chunks as they arrive. Yields an error string on failure.

    Examples:
        # Streaming only
        async for token in async_stream_llm(base_url, api_key, model, "Hello"):
            print(token, end="", flush=True)

        # Streaming + capture full response
        collector = []
        async for token in async_stream_llm(..., collector=collector):
            print(token, end="", flush=True)
        full_text = collector[0]  # available after loop ends
    """
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        complete_response = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content  # may be None on some chunks
            if token:                                # guard against None
                complete_response += token
                yield token

        # Expose the full response to the caller without a second API call
        if collector is not None:
            collector.append(complete_response)

    except Exception as e:
        yield _handle_openai_error(e)


# ── 3. Non-streaming (full response at once) ──────────────────────────────────

async def generate_llm(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: int = 30,
) -> str:
    """
    Generate a complete response from any OpenAI-compatible LLM (non-streaming).

    Waits for the full response before returning. Useful when you need the
    complete text (e.g., to store in cache, run post-processing, etc.).

    Args:
        base_url:      Provider's OpenAI-compatible endpoint.
        api_key:       API key for the provider.
        model_name:    Model identifier string.
        prompt:        User message / prompt.
        system_prompt: Optional system instruction.
        temperature:   Sampling temperature.
        max_tokens:    Max tokens to generate. None = provider default.
        timeout:       Request timeout in seconds.

    Returns:
        str: Full response text, or an error string on failure.

    Example:
        reply = await generate_llm(base_url, api_key, model, "Summarize this...")
        print(reply)
    """
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        return _handle_openai_error(e)
