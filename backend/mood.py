"""
mood.py  -  Persona Community Mood & Macro Climate State
========================================================
Tracks and persists community-level macro psychographic state:
- anxiety (0.0 to 1.0)
- trust_in_health_system (0.0 to 1.0)
- rumour_level (0.0 to 1.0)
- updated_at (ISO-8601 string)

Applies exponential temporal decay (half-life of 5 days):
effective_trait = base + shift * sensitivity * exp(-days/5)
Aggregates only; no per-person tracking anywhere.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

from backend.config import BASE_DIR, MOOD_FILE

logger = logging.getLogger("persona.mood")

# Fallback root path for mood.json
ROOT_MOOD_FILE = BASE_DIR / "mood.json"


class MoodState(BaseModel):
    """Aggregate community mood and macro climate state.
    
    All intensity fields are bounded in [0.0, 1.0].
    Contains no individual personally identifiable or per-person data.
    """

    anxiety: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Community macro anxiety index (0-1)",
    )
    trust_in_health_system: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Institutional trust in healthcare and public governance (0-1)",
    )
    rumour_level: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ambient misinformation and rumour saturation (0-1)",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of last aggregate mood update",
    )


def load_mood() -> MoodState:
    """Load the current community MoodState from mood.json, creating defaults if missing."""
    target_file = MOOD_FILE if MOOD_FILE.exists() else (ROOT_MOOD_FILE if ROOT_MOOD_FILE.exists() else None)
    
    if target_file and target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return MoodState.model_validate(data)
        except Exception as e:
            logger.warning("Could not read %s, initializing default MoodState: %s", target_file, e)

    # Initialize neutral defaults (0.5 across dimensions)
    default_state = MoodState()
    save_mood(default_state)
    return default_state


def save_mood(mood: MoodState) -> MoodState:
    """Persist MoodState to mood.json (in both data/ and root for consistency)."""
    data = mood.model_dump()
    
    # Save to data/mood.json
    try:
        MOOD_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MOOD_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error("Failed to write %s: %s", MOOD_FILE, e)

    # Sync to root mood.json
    try:
        with open(ROOT_MOOD_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error("Failed to write %s: %s", ROOT_MOOD_FILE, e)

    return mood


def get_days_since_update(mood: MoodState, now: Optional[datetime] = None) -> float:
    """Calculate the number of elapsed days since updated_at for temporal exponential decay."""
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        updated_dt = datetime.fromisoformat(mood.updated_at.replace("Z", "+00:00"))
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
        elapsed_seconds = (now - updated_dt).total_seconds()
        return max(0.0, elapsed_seconds / 86400.0)
    except Exception:
        return 0.0


def compute_effective_trait(
    base: float,
    trait_name: str,
    sensitivity: float,
    mood: MoodState,
    days: Optional[float] = None,
) -> float:
    """Apply mood shift with per-archetype sensitivity and temporal decay.
    
    Formula:
        effective_trait = base + shift * sensitivity * exp(-days/5)
    
    Bounded to [0.0, 1.0].
    """
    if days is None:
        days = get_days_since_update(mood)

    # Determine directional shift by psychographic trait
    shift = 0.0
    if trait_name == "emotional_reactivity":
        shift = mood.anxiety - 0.5
    elif trait_name == "time_pressure":
        shift = 0.5 * (mood.anxiety - 0.5)
    elif trait_name == "trust_in_source":
        shift = mood.trust_in_health_system - 0.5
    elif trait_name == "social_sharing":
        shift = mood.rumour_level - 0.5
    elif trait_name == "skepticism":
        # Higher rumours erode baseline skepticism or heighten credulity
        shift = -(mood.rumour_level - 0.5)

    decay = math.exp(-days / 5.0)
    effective_val = base + (shift * sensitivity * decay)
    return max(0.0, min(1.0, effective_val))
