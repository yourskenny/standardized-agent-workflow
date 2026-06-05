from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import Settings
from ..models import ModelDefinition, load_models


@dataclass(frozen=True)
class ModelResolution:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    fallback: str
    credential_status: str
    client: object | None = None

    def trace_metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "fallback": self.fallback,
            "credential_status": self.credential_status,
        }


class CredentialResolver:
    def get(self, env_name: str) -> str:
        return os.getenv(env_name, "")


class ModelProviderResolver:
    def __init__(
        self,
        settings: Settings,
        credentials: CredentialResolver | None = None,
    ):
        self.settings = settings
        self.credentials = credentials or CredentialResolver()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        credentials: CredentialResolver | None = None,
    ) -> "ModelProviderResolver":
        return cls(settings, credentials=credentials)

    def resolve(self, model_client: object | None = None) -> ModelResolution:
        definition = self._definition()
        if definition.provider == "extractive":
            return ModelResolution(
                provider="extractive",
                model=definition.model,
                base_url=definition.base_url,
                api_key_env="",
                fallback=definition.fallback,
                credential_status="not_required",
                client=None,
            )

        api_key = self.credentials.get(definition.api_key_env)
        credential_status = "present" if api_key else "missing"
        return ModelResolution(
            provider=definition.provider,
            model=definition.model,
            base_url=definition.base_url,
            api_key_env=definition.api_key_env,
            fallback=definition.fallback,
            credential_status=credential_status,
            client=model_client,
        )

    def _definition(self) -> ModelDefinition:
        models = load_models(self.settings.model_config_path)
        if models:
            return models[0]
        if self.settings.generation_provider == "extractive":
            return ModelDefinition(
                id="legacy_extractive",
                provider="extractive",
                model="",
                base_url="",
                api_key_env="",
                fallback="extractive",
            )
        return ModelDefinition(
            id="legacy_openai_compatible",
            provider=self.settings.generation_provider,
            model=os.getenv("LOCAL_AGENT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
            base_url=os.getenv("LOCAL_AGENT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1",
            api_key_env="LOCAL_AGENT_API_KEY"
            if os.getenv("LOCAL_AGENT_API_KEY") or not os.getenv("OPENAI_API_KEY")
            else "OPENAI_API_KEY",
            fallback=self.settings.generation_fallback,
        )
