"""Load profession profiles from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_cv.models import ProfessionProfile


def load_profile(path: str | Path) -> ProfessionProfile:
    """Load a profession profile from a YAML file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ProfessionProfile(**data)


def list_profiles(profiles_dir: str | Path = "profiles") -> list[Path]:
    """List all available profession profile files."""
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.exists():
        return []
    return sorted(profiles_dir.glob("*.yaml")) + sorted(profiles_dir.glob("*.yml"))
