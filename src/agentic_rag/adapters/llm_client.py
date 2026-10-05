"""Lightweight, dependency-free LLM Client Provider for EvidenceFirst RAG.

Supports Google Gemini (default: gemini-1.5-flash) and OpenAI (default: gpt-4o-mini)
using standard HTTP requests (via urllib.request or httpx), requiring zero extra SDKs.
Credentials are automatically discovered from environment variables or Streamlit secrets.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping


def _safe_get_streamlit_secret(key: str) -> str | None:
    """Safely retrieve a key from Streamlit secrets without raising exceptions.

    Guarantees no StreamlitSecretNotFoundError, FileNotFoundError, or KeyError escapes
    if secrets.toml is absent at user home (~/.streamlit/secrets.toml), project root,
    or app directory.
    """
    try:
        import streamlit as st

        # Accessing st.secrets or calling .get() / [] raises StreamlitSecretNotFoundError
        # if no secrets.toml exists. We wrap the access and call defensively.
        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return None
        val = secrets.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    except BaseException:
        # Silently catch StreamlitSecretNotFoundError, FileNotFoundError, KeyError, etc.
        return None
    return None


def get_llm_credentials() -> tuple[str | None, str | None, str | None]:
    """Detect available credentials and return (provider, api_key, model_name).

    Returns (None, None, None) safely if no environment variables and no Streamlit
    secrets exist, never raising any exceptions.
    """
    # 1. Gemini check (primary)
    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or _safe_get_streamlit_secret("GEMINI_API_KEY")
        or _safe_get_streamlit_secret("GOOGLE_API_KEY")
    )
    if gemini_key and str(gemini_key).strip():
        model = (
            os.environ.get("GEMINI_MODEL")
            or _safe_get_streamlit_secret("GEMINI_MODEL")
            or "gemini-1.5-flash"
        )
        return "gemini", str(gemini_key).strip(), str(model).strip()

    # 2. OpenAI check
    openai_key = (
        os.environ.get("OPENAI_API_KEY")
        or _safe_get_streamlit_secret("OPENAI_API_KEY")
    )
    if openai_key and str(openai_key).strip():
        model = (
            os.environ.get("OPENAI_MODEL")
            or _safe_get_streamlit_secret("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        return "openai", str(openai_key).strip(), str(model).strip()

    return None, None, None


class GenericRestLLMClient:
    """Invokes Gemini or OpenAI REST endpoints without requiring third-party SDKs."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        """Execute structured JSON completion."""
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def complete(self, prompt: str) -> str:
        """Direct completion method alias."""
        return self(prompt)

    def generate(self, prompt: str) -> str:
        """Direct generation method alias."""
        return self(prompt)

    def complete_json(
        self,
        task: str = "",
        prompt: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        """Structured completion method alias."""
        return self(prompt)


    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
        candidates = res_data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {res_data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"Gemini API returned empty parts: {res_data}")
        return str(parts[0].get("text", ""))

    def _call_openai(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
        choices = res_data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenAI API returned no choices: {res_data}")
        return str(choices[0].get("message", {}).get("content", ""))


def create_llm_client() -> tuple[GenericRestLLMClient | None, str]:
    """Create an LLM client if credentials exist, otherwise return None and status string.

    Returns:
        (client, description): Client instance or None, plus a human-readable status string.
    """
    provider, api_key, model = get_llm_credentials()
    if not provider or not api_key:
        return None, "Deterministic fallback (No API key detected)"

    client = GenericRestLLMClient(provider=provider, api_key=api_key, model=model or "default")
    provider_name = "OpenAI" if provider == "openai" else "Gemini"
    return client, f"LLM AutoRater ({provider_name} - {model})"
