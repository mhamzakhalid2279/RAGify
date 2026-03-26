"""
LLM Client
───────────
Uses HuggingFace Inference API (free, no cost).
Primary model  : HuggingFaceH4/zephyr-7b-beta
Fallback models: mistralai/Mistral-7B-Instruct-v0.3
                 microsoft/Phi-3-mini-4k-instruct
                 tiiuae/falcon-7b-instruct

Fixed for huggingface_hub InferenceClient chat_completion API:
  - Uses max_tokens (not max_new_tokens)
  - Removes do_sample, repetition_penalty (not valid for chat endpoint)
  - Auto-fallback chain on rate-limit / server errors
  - Token streaming via yield
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, List

from huggingface_hub import InferenceClient

# ──────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────

# PRIMARY_MODEL = "HuggingFaceH4/zephyr-7b-beta"
# FALLBACK_MODELS = [
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "microsoft/Phi-3-mini-4k-instruct",
#     "tiiuae/falcon-7b-instruct",
# ]

# NEW — verified working on HF free tier (2025)
PRIMARY_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FALLBACK_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",          # keep as fallback attempt
    "meta-llama/Llama-3.2-3B-Instruct",       # fast & free
    "microsoft/Phi-3.5-mini-instruct",         # excellent small model
    "Qwen/Qwen2.5-7B-Instruct",               # very capable, free tier
]

# Valid params for InferenceClient.chat_completion:
# max_tokens, temperature, top_p, stop, seed
DEFAULT_PARAMS: Dict[str, Any] = {
    "max_tokens": 1024,
    "temperature": 0.3,
    "top_p": 0.92,
}

# FALLBACK_TRIGGERS = {"rate limit", "503", "500", "429", "overloaded", "unavailable", "quota"}
# OLD

# NEW — add model_not_supported
FALLBACK_TRIGGERS = {
    "rate limit", "503", "500", "429", "overloaded", 
    "unavailable", "quota", "model_not_supported", 
    "not supported", "bad request"
}

def _should_fallback(error_msg: str) -> bool:
    msg = error_msg.lower()
    return any(t in msg for t in FALLBACK_TRIGGERS)


def _make_client(model: str) -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    return InferenceClient(model=model, token=token, timeout=90)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def stream_response(
    messages: List[Dict[str, str]],
    system_prompt: str,
    params: Dict[str, Any] | None = None,
    max_retries: int = 2,
) -> Generator[str, None, None]:
    """
    Stream assistant tokens.
    Falls back through model list on rate-limit/server errors.

    Yields partial text strings.
    """
    gen_params = {**DEFAULT_PARAMS, **(params or {})}
    # Safety: strip any invalid keys that may have been passed
    valid_keys = {"max_tokens", "temperature", "top_p", "stop", "seed"}
    gen_params = {k: v for k, v in gen_params.items() if k in valid_keys}

    full_messages = [{"role": "system", "content": system_prompt}] + messages
    model_order = [PRIMARY_MODEL] + FALLBACK_MODELS
    last_error = ""

    for model in model_order:
        client = _make_client(model)
        for attempt in range(max_retries):
            try:
                stream = client.chat_completion(
                    messages=full_messages,
                    stream=True,
                    **gen_params,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return  # success
            except Exception as e:
                last_error = str(e)
                if _should_fallback(last_error):
                    time.sleep(1.5 * (attempt + 1))
                    break  # try next model
                else:
                    yield f"\n\n[⚠ Error: {last_error}]"
                    return

    yield f"\n\n[⚠ All models unavailable. Last error: {last_error}]"


def generate_sync(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 512,
) -> str:
    """Non-streaming single call. Returns full response string."""
    result_parts = []
    for tok in stream_response(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system_prompt,
        params={"max_tokens": max_tokens, "temperature": 0.5},
    ):
        result_parts.append(tok)
    return "".join(result_parts)