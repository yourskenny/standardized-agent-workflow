from __future__ import annotations

from ..config import Settings
from ..runtime import AgentRuntime
from ..types import AgentRequest, AgentResponse


class RuntimeService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, request: AgentRequest) -> AgentResponse:
        return AgentRuntime(self.settings).run(request)
