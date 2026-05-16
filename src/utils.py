from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_project_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)


def root_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def update_metrics(model_name: str, metrics: dict[str, Any], path: Path) -> None:
    current = load_metrics(path)
    current[model_name] = metrics
    save_json(current, path)
