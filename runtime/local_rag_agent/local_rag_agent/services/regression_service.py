from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..regression import run_regression, summarize_regression_report


class RegressionService:
    def run(
        self,
        questions_path: Path,
        output_path: Path,
        callback: Callable[[str], dict[str, object]],
    ) -> int:
        return run_regression(questions_path, output_path, callback)

    def summarize(self, report_path: Path) -> dict[str, object]:
        return summarize_regression_report(report_path)
