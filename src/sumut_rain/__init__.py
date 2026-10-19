"""Paket pendukung proyek ekstrem curah hujan Sumatera Utara."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Baca config.yaml di root proyek."""
    path = path or PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Ubah path relatif di config menjadi path absolut dan buat foldernya."""
    p = PROJECT_ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    return p
