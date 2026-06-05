from __future__ import annotations


def error_payload(code: str, message: str) -> dict[str, object]:
    return {"error": {"code": code, "message": message}}
