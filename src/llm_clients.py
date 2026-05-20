"""
CDCE Compression Harness — LLM Clients

Unified interface for Anthropic, OpenAI, and Google models.
All return the same response format for consistent metric extraction.
"""
import time
import asyncio
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Standardized response from any LLM."""
    model: str
    provider: str
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    task_id: str
    budget: int
    timestamp: float = field(default_factory=time.time)


def call_anthropic(model: str, prompt: str, max_tokens: int, api_key: str) -> LLMResponse:
    """Call Anthropic's API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    start = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        system=(
            "You are solving a task under strict token constraints. "
            "Be maximally compressed and efficient. Show your reasoning, "
            "but use the fewest possible distinct operations and steps."
        ),
    )

    latency = (time.time() - start) * 1000
    content = response.content[0].text if response.content else ""

    return LLMResponse(
        model=model,
        provider="anthropic",
        content=content,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency,
        task_id="",
        budget=max_tokens,
    )


def call_openai(model: str, prompt: str, max_tokens: int, api_key: str) -> LLMResponse:
    """Call OpenAI's API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are solving a task under strict token constraints. "
                    "Be maximally compressed and efficient. Show your reasoning, "
                    "but use the fewest possible distinct operations and steps."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    latency = (time.time() - start) * 1000
    content = response.choices[0].message.content or ""
    usage = response.usage

    return LLMResponse(
        model=model,
        provider="openai",
        content=content,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency,
        task_id="",
        budget=max_tokens,
    )


def call_google(model: str, prompt: str, max_tokens: int, api_key: str) -> LLMResponse:
    """Call Google's Gemini API."""
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=api_key)
    start = time.time()

    full_prompt = (
        "You are solving a task under strict token constraints. "
        "Be maximally compressed and efficient. Show your reasoning, "
        "but use the fewest possible distinct operations and steps.\n\n"
        + prompt
    )

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=GenerateContentConfig(max_output_tokens=max_tokens),
    )

    latency = (time.time() - start) * 1000
    content = response.text or ""

    # Gemini usage metadata
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    return LLMResponse(
        model=model,
        provider="google",
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency,
        task_id="",
        budget=max_tokens,
    )


# Dispatch table
PROVIDERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "google": call_google,
}

API_KEYS = {}


def init_keys():
    """Load API keys from config."""
    from src.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
    global API_KEYS
    API_KEYS = {
        "anthropic": ANTHROPIC_API_KEY,
        "openai": OPENAI_API_KEY,
        "google": GOOGLE_API_KEY,
    }


def call_model(
    provider: str, model: str, prompt: str, max_tokens: int, task_id: str
) -> LLMResponse:
    """Universal model caller."""
    if not API_KEYS:
        init_keys()

    caller = PROVIDERS[provider]
    api_key = API_KEYS[provider]

    try:
        resp = caller(model, prompt, max_tokens, api_key)
        resp.task_id = task_id
        return resp
    except Exception as e:
        # Return error response rather than crashing
        return LLMResponse(
            model=model,
            provider=provider,
            content=f"[ERROR: {str(e)}]",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            task_id=task_id,
            budget=max_tokens,
        )


def call_models_parallel(
    models: dict, prompt: str, max_tokens: int, task_id: str
) -> list[LLMResponse]:
    """
    Call multiple models with the same prompt.
    Uses threading for true parallelism across API calls.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    responses = []

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                call_model,
                cfg["provider"],
                cfg["model"],
                prompt,
                max_tokens,
                task_id,
            ): name
            for name, cfg in models.items()
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                resp = future.result()
                responses.append(resp)
            except Exception as e:
                print(f"  [FAIL] {model_name}: {e}")

    return responses
