"""Small reusable utilities shared across application modules."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
