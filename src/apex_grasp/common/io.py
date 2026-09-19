from __future__ import annotations
from pathlib import Path
import json
from typing import Any
import numpy as np
from PIL import Image


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: out.append(json.loads(line))
                except json.JSONDecodeError as e: raise ValueError(f"Invalid JSONL at line {i}: {e}")
    return out

def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))

def load_depth(path: str | Path) -> np.ndarray:
    p=Path(path)
    if p.suffix.lower()==".npy": return np.load(p)
    return np.asarray(Image.open(p))

def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0
