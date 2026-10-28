"""Interface for local Ollama LLM queries, prompt formatting, JSON extraction, and retries."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional
import requests

from backend.config import OLLAMA_URL, MODEL, USE_LLM


def clean_json_text(text: str) -> str:
    """Extract raw JSON text from model output, stripping markdown code fences if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def query_ollama(
    prompt: str,
    system: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    timeout: float = 45.0,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query local Ollama instance and parse the JSON response.

    Args:
        prompt: The input prompt to the model.
        system: Optional system instruction prompt.
        json_schema: Optional JSON schema dict for structured output.
        timeout: Request timeout in seconds.
        model: Model name override (defaults to config.MODEL).
        options: Optional runtime parameters (e.g. {'temperature': 0.1}).

    Returns:
        Parsed JSON dictionary response.
    """
    if not USE_LLM:
        raise RuntimeError("USE_LLM is set to False in config")

    target_model = model or MODEL
    payload: Dict[str, Any] = {
        "model": target_model,
        "prompt": prompt,
        "stream": False,
        "format": json_schema if json_schema is not None else "json",
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options

    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    raw_response = response.json().get("response", "")
    cleaned = clean_json_text(raw_response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM JSON output: {raw_response[:200]}") from exc
