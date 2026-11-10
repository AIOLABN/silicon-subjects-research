from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(*parts: Any, length: int = 16) -> str:
    return sha256_text(canonical_json(parts))[:length]


def derive_seed(*parts: Any) -> int:
    return int(sha256_text(canonical_json(parts))[:8], 16)
