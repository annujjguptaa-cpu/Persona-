"""Configuration settings for Persona: Synthetic Audience Message Spread Simulator."""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PRESETS_DIR = DATA_DIR / "presets"

# Ollama / Local LLM Settings
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL: str = os.getenv("MODEL", "qwen2.5vl:7b")
USE_LLM: bool = os.getenv("USE_LLM", "true").lower() in ("true", "1", "yes")

# Simulation Hyperparameters
N_PEOPLE: int = int(os.getenv("N_PEOPLE", "120"))
SEED: int = int(os.getenv("SEED", "42"))
MAX_WAVES: int = int(os.getenv("MAX_WAVES", "6"))

# Verdict Thresholds (Tier 0 -> Tier 3)
# Tier 0 (Contained/Ignored):   reach < 0.10 and R0 < 0.70
# Tier 1 (Niche hit/Niche):     reach >= 0.10 or R0 >= 0.70
# Tier 2 (Wide spread/Strong):  reach >= 0.30 or R0 >= 1.00
# Tier 3 (Outbreak/Breakout):   reach >= 0.55 or R0 >= 1.40
DEFAULT_REACH_THRESHOLDS: list[float] = [0.10, 0.30, 0.55]
DEFAULT_R0_THRESHOLDS: list[float] = [0.70, 1.00, 1.40]

