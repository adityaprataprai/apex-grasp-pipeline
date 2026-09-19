from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .errors import PipelineFailure


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise PipelineFailure("CONFIG_NOT_FOUND", f"Configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise PipelineFailure("CONFIG_INVALID", "Top-level configuration must be a mapping")
    return data


def require(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise PipelineFailure("CONFIG_REQUIRED", f"Missing required configuration: {dotted}")
        cur = cur[part]
    if cur is None:
        raise PipelineFailure("CONFIG_REQUIRED", f"Required configuration is unset: {dotted}")
    return cur


def require_many(cfg: dict[str, Any], names: list[str]) -> None:
    for n in names:
        require(cfg, n)
