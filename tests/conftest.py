"""Shared fixtures: default config and a session-wide LPIPS instance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BenchmarkConfig, load_config  # noqa: E402
from src.metrics import LPIPSMetric  # noqa: E402


@pytest.fixture(scope="session")
def default_config() -> BenchmarkConfig:
    """The shipped configs/default.yaml, with a smaller canvas for speed."""
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    cfg.preprocessing.canvas_size = 256
    cfg.validate()
    return cfg


@pytest.fixture(scope="session")
def lpips_metric(default_config: BenchmarkConfig) -> LPIPSMetric:
    """One LPIPS model for the whole session (loading weights is slow)."""
    return LPIPSMetric(default_config.metrics.lpips)
