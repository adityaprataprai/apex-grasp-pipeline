from dataclasses import dataclass
from typing import Any

@dataclass
class PipelineFailure(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": "failure", "code": self.code, "message": self.message, "details": self.details or {}}
