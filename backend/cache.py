"""Disk and in-memory cache for message feature extractions and simulation run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import CACHE_DIR

# In-memory LRU / dictionary cache
_MEMORY_CACHE: Dict[str, Any] = {}


def hash_key(text: str) -> str:
    """Compute deterministic SHA-256 hash string for arbitrary text."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def get_cache(namespace: str, key: str) -> Optional[Any]:
    """Retrieve item from memory or disk cache."""
    composite_key = f"{namespace}:{key}"
    if composite_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[composite_key]

    disk_file = CACHE_DIR / namespace / f"{key}.json"
    if disk_file.exists():
        try:
            with open(disk_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            _MEMORY_CACHE[composite_key] = data
            return data
        except Exception:
            return None
    return None


def set_cache(namespace: str, key: str, value: Any) -> None:
    """Store item in memory and disk cache."""
    composite_key = f"{namespace}:{key}"
    _MEMORY_CACHE[composite_key] = value

    try:
        ns_dir = CACHE_DIR / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        disk_file = ns_dir / f"{key}.json"
        with open(disk_file, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
    except Exception:
        pass


def clear_cache(namespace: Optional[str] = None) -> None:
    """Clear in-memory and disk cache."""
    global _MEMORY_CACHE
    if namespace:
        _MEMORY_CACHE = {k: v for k, v in _MEMORY_CACHE.items() if not k.startswith(f"{namespace}:")}
        ns_dir = CACHE_DIR / namespace
        if ns_dir.exists():
            for f in ns_dir.glob("*.json"):
                f.unlink(missing_ok=True)
    else:
        _MEMORY_CACHE.clear()
