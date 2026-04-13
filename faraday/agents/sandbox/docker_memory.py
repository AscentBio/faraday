"""Docker `--memory` string normalization (no heavy imports).

Each code-exec container defaults to ``512m`` (Docker memory limit syntax; ~512 MB).
Unset or too-small values normalize to that default.
"""

from __future__ import annotations

import re
from typing import Optional

# Default per-container limit: 512 MB class (Docker `m` = 1024² bytes).
_DOCKER_MEMORY_FLOOR_BYTES = 512 * 1024 * 1024
DOCKER_MEMORY_DEFAULT = "512m"


def _parse_docker_memory_bytes(value: str) -> Optional[int]:
    """Parse Docker-style memory strings (e.g. 512m, 1g, raw bytes). Returns None if unknown."""
    s = value.strip().lower().replace(" ", "")
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = re.match(r"^(\d+)([kmg])b?$", s)
    if not m:
        return None
    n = int(m.group(1))
    suf = m.group(2)
    mult = {"k": 1024, "m": 1024**2, "g": 1024**3}
    return n * mult[suf]


def normalize_docker_memory(value: Optional[str]) -> str:
    """Default each container to ``512m``; bump parsed limits below that to the same default."""
    raw = (value or "").strip()
    if not raw:
        return DOCKER_MEMORY_DEFAULT
    parsed = _parse_docker_memory_bytes(raw)
    if parsed is not None and parsed < _DOCKER_MEMORY_FLOOR_BYTES:
        return DOCKER_MEMORY_DEFAULT
    return raw
