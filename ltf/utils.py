"""Shared helpers: paths, logging, RNG, device selection."""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .config import CACHE, OUTPUT

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def get_logger(name: str = "ltf") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG = get_logger()


def ensure_dirs() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)


def cache_path(name: str) -> Path:
    ensure_dirs()
    return CACHE / name


def output_path(name: str) -> Path:
    ensure_dirs()
    p = OUTPUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    log = logger or LOG
    t0 = time.perf_counter()
    log.info("%s ...", label)
    yield
    log.info("%s done in %.1fs", label, time.perf_counter() - t0)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def resolve_device(requested: str | None = None) -> str:
    """Pick a torch device without requiring torch to be importable.

    Written so the code runs on CPU today and picks up a GPU later with no
    edits: set `LTF_DEVICE`, pass `requested`, or just let it auto-detect.
    """
    if requested:
        return requested
    env = os.environ.get("LTF_DEVICE")
    if env:
        return env
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or broadcastable arrays."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def describe_array(a: np.ndarray, name: str = "") -> str:
    a = np.asarray(a, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return f"{name}: all non-finite ({a.size} entries)"
    return (f"{name}: n={a.size:,} finite={finite.size:,} "
            f"min={finite.min():.3f} p25={np.percentile(finite,25):.3f} "
            f"med={np.median(finite):.3f} p75={np.percentile(finite,75):.3f} "
            f"max={finite.max():.3f} mean={finite.mean():.3f}")
