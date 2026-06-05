from __future__ import annotations

from ..config import Settings
from ..validator import ValidationResult, validate_project_config, validate_project_contract


class ValidationService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self) -> ValidationResult:
        if self.settings.config_path is not None:
            return validate_project_config(self.settings.project_root, self.settings.config_path)
        return validate_project_contract(self.settings)
