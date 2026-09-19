"""
Shared helpers: seeding, logging, timing, and the experiment manifest.

The manifest is what makes a run auditable -- it records the seed, split sizes,
library versions, per-model timings and chosen hyperparameters for every run.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

import config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int = config.RANDOM_STATE) -> None:
    """Seed Python, NumPy and TensorFlow (if loaded) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass  # TF not installed / not needed for this entry point


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_STEP = {"i": 0, "total": 0}


def start_run(total_steps: int) -> None:
    _STEP["i"] = 0
    _STEP["total"] = total_steps


def step(message: str) -> None:
    """Print a numbered progress line, e.g. '[3/12] Training Random Forest...'"""
    _STEP["i"] += 1
    print(f"[{_STEP['i']}/{_STEP['total']}] {message}", flush=True)


def info(message: str) -> None:
    print(f"    {message}", flush=True)


def warn(message: str) -> None:
    print(f"    !! {message}", flush=True)


@contextmanager
def timer(label: str = ""):
    """Context manager yielding a one-element list that receives elapsed seconds."""
    holder = [0.0]
    t0 = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - t0
        if label:
            info(f"{label} took {holder[0]:.1f}s")


# ---------------------------------------------------------------------------
# Experiment manifest
# ---------------------------------------------------------------------------
class Manifest:
    """Accumulates run metadata and writes outputs/experiment_manifest.json."""

    def __init__(self, mode: str, seed: int = config.RANDOM_STATE):
        self.data: dict = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "seed": seed,
            "python": sys.version.split()[0],
            "platform": f"{platform.system()} {platform.machine()}",
            "library_versions": self._versions(),
            "budgets": config.MODE_BUDGETS[mode],
            "split": {},
            "models": {},
            "failures": {},
            "notes": [],
        }

    @staticmethod
    def _versions() -> dict:
        out = {}
        for name in ("numpy", "pandas", "sklearn", "xgboost", "tensorflow", "shap"):
            try:
                mod = __import__(name)
                out[name] = getattr(mod, "__version__", "unknown")
            except Exception:
                out[name] = "not installed"
        return out

    def record_split(self, **kwargs) -> None:
        self.data["split"].update(kwargs)

    def record_model(self, name: str, **kwargs) -> None:
        self.data["models"].setdefault(name, {}).update(kwargs)

    def record_failure(self, name: str, reason: str, traceback_text: str = "") -> None:
        """A failed model is recorded, never silently skipped."""
        self.data["failures"][name] = {
            "reason": reason,
            "traceback": traceback_text[-2000:],
        }

    def note(self, text: str) -> None:
        self.data["notes"].append(text)

    def save(self, path: Path = config.MANIFEST_JSON) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2, default=str))
        return path
