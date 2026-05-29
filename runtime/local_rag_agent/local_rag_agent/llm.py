from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        api_key = os.getenv("LOCAL_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LOCAL_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = os.getenv("LOCAL_AGENT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        if not api_key:
            return None
        return cls(api_key=api_key, base_url=base_url, model=model)

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0.2}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = _redact_secret(error.read().decode("utf-8", errors="replace"), self.api_key)
            raise RuntimeError(f"Model request failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Model request failed: {error.reason}") from error
        return data["choices"][0]["message"]["content"]


def _redact_secret(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text
