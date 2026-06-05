from __future__ import annotations

from ...config import Settings
from ...types import AgentRequest
from ...validator import validate_project_config, validate_project_contract
from ...services.runtime_service import RuntimeService


def validate_payload(settings: Settings) -> dict[str, object]:
    if settings.config_path is not None:
        return validate_project_config(settings.project_root, settings.config_path).to_dict()
    return validate_project_contract(settings).to_dict()


def chat_payload(
    settings: Settings,
    message: str,
    history: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    request = AgentRequest(
        message=message,
        history=history or [],
        metadata=metadata or {},
    )
    return RuntimeService(settings).run(request).to_dict()
