"""Configuration loading and validation.

The configuration is a plain nested dictionary loaded from YAML and validated
into typed dataclasses. Validation is strict: unknown keys, bad enum values
and invalid weights raise :class:`ConfigError` with a descriptive message.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

logger = logging.getLogger(__name__)

WEIGHT_TOLERANCE = 1e-6
METRIC_NAMES: tuple[str, ...] = ("lpips", "ssim", "silhouette", "edge")
CROP_MODES: tuple[str, ...] = ("none", "center_crop", "foreground_bbox")
MASK_MODES: tuple[str, ...] = ("alpha", "auto", "background")
RESAMPLE_MODES: tuple[str, ...] = ("lanczos", "bicubic", "bilinear")
ALIGNMENT_METHODS: tuple[str, ...] = ("none", "centroid", "phase_correlation")
LPIPS_NETS: tuple[str, ...] = ("alex", "vgg", "squeeze")
DEVICES: tuple[str, ...] = ("auto", "cuda", "cpu")
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(ValueError):
    """Raised when the configuration is invalid."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class InputConfig:
    """Options controlling how input folders are scanned and paired."""

    extensions: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    skip_unmatched: bool = False

    def validate(self) -> None:
        if not self.extensions:
            raise ConfigError("input.extensions must not be empty")
        self.extensions = [e.lower() if e.startswith(".") else "." + e.lower() for e in self.extensions]
        if not isinstance(self.skip_unmatched, bool):
            raise ConfigError("input.skip_unmatched must be a boolean")


@dataclass
class PreprocessingConfig:
    """Options applied identically to reference and candidate images."""

    canvas_size: int = 512
    background_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    crop_mode: str = "none"
    foreground_padding: float = 0.10
    mask_mode: str = "auto"
    alpha_threshold: int = 16
    mask_background_threshold: int = 12
    mask_min_foreground_fraction: float = 0.001
    mask_max_foreground_fraction: float = 0.98
    resample: str = "lanczos"
    alignment: str = "phase_correlation"
    alignment_max_shift: float = 0.5

    def validate(self) -> None:
        if self.alignment not in ALIGNMENT_METHODS:
            raise ConfigError(f"preprocessing.alignment must be one of {ALIGNMENT_METHODS}, got {self.alignment!r}")
        if not (0.0 < float(self.alignment_max_shift) <= 1.0):
            raise ConfigError("preprocessing.alignment_max_shift must be in (0, 1]")
        if not isinstance(self.canvas_size, int) or self.canvas_size < 16:
            raise ConfigError("preprocessing.canvas_size must be an integer >= 16")
        if (
            not isinstance(self.background_color, (list, tuple))
            or len(self.background_color) != 3
            or any((not isinstance(c, int)) or c < 0 or c > 255 for c in self.background_color)
        ):
            raise ConfigError("preprocessing.background_color must be three integers in [0, 255]")
        self.background_color = [int(c) for c in self.background_color]
        if self.crop_mode not in CROP_MODES:
            raise ConfigError(f"preprocessing.crop_mode must be one of {CROP_MODES}, got {self.crop_mode!r}")
        if not (0.0 <= float(self.foreground_padding) < 1.0):
            raise ConfigError("preprocessing.foreground_padding must be in [0, 1)")
        if self.mask_mode not in MASK_MODES:
            raise ConfigError(f"preprocessing.mask_mode must be one of {MASK_MODES}, got {self.mask_mode!r}")
        if not (0 <= int(self.alpha_threshold) <= 255):
            raise ConfigError("preprocessing.alpha_threshold must be in [0, 255]")
        if not (0 <= int(self.mask_background_threshold) <= 255):
            raise ConfigError("preprocessing.mask_background_threshold must be in [0, 255]")
        if not (0.0 <= float(self.mask_min_foreground_fraction) < 1.0):
            raise ConfigError("preprocessing.mask_min_foreground_fraction must be in [0, 1)")
        if not (float(self.mask_min_foreground_fraction) < float(self.mask_max_foreground_fraction) <= 1.0):
            raise ConfigError("preprocessing.mask_max_foreground_fraction must be in (min_fraction, 1]")
        if self.resample not in RESAMPLE_MODES:
            raise ConfigError(f"preprocessing.resample must be one of {RESAMPLE_MODES}, got {self.resample!r}")


@dataclass
class SSIMConfig:
    gaussian_weights: bool = True
    sigma: float = 1.5
    win_size: int = 7

    def validate(self) -> None:
        if float(self.sigma) <= 0:
            raise ConfigError("metrics.ssim.sigma must be > 0")
        if int(self.win_size) < 3 or int(self.win_size) % 2 == 0:
            raise ConfigError("metrics.ssim.win_size must be an odd integer >= 3")


@dataclass
class LPIPSConfig:
    net: str = "alex"
    device: str = "auto"

    def validate(self) -> None:
        if self.net not in LPIPS_NETS:
            raise ConfigError(f"metrics.lpips.net must be one of {LPIPS_NETS}, got {self.net!r}")
        if self.device not in DEVICES:
            raise ConfigError(f"metrics.lpips.device must be one of {DEVICES}, got {self.device!r}")


@dataclass
class EdgeConfig:
    canny_low: int = 100
    canny_high: int = 200
    blur_kernel: int = 3
    max_distance: float = 20.0
    min_edge_fraction: float = 0.0005

    def validate(self) -> None:
        if not (0 <= int(self.canny_low) <= int(self.canny_high) <= 255 * 4):
            raise ConfigError("metrics.edge: require 0 <= canny_low <= canny_high")
        if int(self.blur_kernel) < 0 or (int(self.blur_kernel) > 0 and int(self.blur_kernel) % 2 == 0):
            raise ConfigError("metrics.edge.blur_kernel must be 0 or an odd integer")
        if float(self.max_distance) <= 0:
            raise ConfigError("metrics.edge.max_distance must be > 0")
        if not (0.0 <= float(self.min_edge_fraction) < 1.0):
            raise ConfigError("metrics.edge.min_edge_fraction must be in [0, 1)")


@dataclass
class MetricsConfig:
    ssim: SSIMConfig = field(default_factory=SSIMConfig)
    lpips: LPIPSConfig = field(default_factory=LPIPSConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)

    def validate(self) -> None:
        self.ssim.validate()
        self.lpips.validate()
        self.edge.validate()


@dataclass
class OutputConfig:
    save_preprocessed: bool = True
    save_comparisons: bool = True
    report_pairs_per_page: int = 6
    log_level: str = "INFO"
    group_separator: str = "_"

    def validate(self) -> None:
        if int(self.report_pairs_per_page) < 1:
            raise ConfigError("output.report_pairs_per_page must be >= 1")
        if self.group_separator is None:
            self.group_separator = ""
        if not isinstance(self.group_separator, str):
            raise ConfigError("output.group_separator must be a string (empty string disables grouping)")
        self.log_level = str(self.log_level).upper()
        if self.log_level not in LOG_LEVELS:
            raise ConfigError(f"output.log_level must be one of {LOG_LEVELS}")


@dataclass
class BenchmarkConfig:
    """Top-level validated configuration."""

    input: InputConfig = field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    weights: dict[str, float] = field(
        default_factory=lambda: {"lpips": 0.40, "ssim": 0.30, "silhouette": 0.20, "edge": 0.10}
    )
    # Per-metric score (0-100) that unrelated objects reach "for free"; scores
    # are rescaled so that the floor maps to 0 and 100 stays 100. All zeros
    # (the default) leaves scores unchanged. Missing metrics default to 0.
    score_floors: dict[str, float] = field(default_factory=dict)
    # Exponent applied after the floor rescaling: calibrated = 100 * x ** gamma
    # with x in [0, 1]. 1.0 = linear; < 1 lifts mid-range scores (a rough but
    # recognisable replica) while 0 stays 0 and 100 stays 100.
    score_gamma: float = 1.0
    # How the view scores of one object are combined: power mean with this
    # exponent. 1.0 = arithmetic mean; smaller values lean towards the WEAKEST
    # view, because a real replica matches in every view while unrelated
    # objects often coincide in one (two round blobs seen from the top).
    view_power: float = 1.0
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.input.validate()
        self.preprocessing.validate()
        self.metrics.validate()
        self.weights = validate_weights(self.weights)
        self.score_floors = validate_score_floors(self.score_floors)
        if isinstance(self.score_gamma, bool) or not isinstance(self.score_gamma, (int, float)):
            raise ConfigError(f"score_gamma must be a number, got {self.score_gamma!r}")
        self.score_gamma = float(self.score_gamma)
        if not 0.1 <= self.score_gamma <= 5.0:
            raise ConfigError(f"score_gamma must be in [0.1, 5], got {self.score_gamma}")
        if isinstance(self.view_power, bool) or not isinstance(self.view_power, (int, float)):
            raise ConfigError(f"view_power must be a number, got {self.view_power!r}")
        self.view_power = float(self.view_power)
        if not 0.05 <= self.view_power <= 1.0:
            raise ConfigError(f"view_power must be in [0.05, 1], got {self.view_power}")
        self.output.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the configuration."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def validate_score_floors(floors: Mapping[str, Any] | None) -> dict[str, float]:
    """Validate ``score_floors``: known metric names, each floor in ``[0, 100)``."""
    floors = floors or {}
    if not isinstance(floors, Mapping):
        raise ConfigError("score_floors must be a mapping of metric name -> floor score")
    unknown = set(floors) - set(METRIC_NAMES)
    if unknown:
        raise ConfigError(f"score_floors: unknown metric(s) {sorted(unknown)}; expected {list(METRIC_NAMES)}")
    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        value = floors.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"score_floors.{name} must be a number, got {value!r}")
        value = float(value)
        if not 0.0 <= value < 100.0:
            raise ConfigError(f"score_floors.{name} must be in [0, 100), got {value}")
        out[name] = value
    return out


def validate_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    """Validate metric weights.

    Rules: every key must be a known metric, every known metric must be
    present, every value must be a finite number >= 0 and the values must sum
    to 1.0 within :data:`WEIGHT_TOLERANCE`.

    Returns:
        A new ``dict`` with float values in canonical metric order.

    Raises:
        ConfigError: if any rule is violated.
    """
    if not isinstance(weights, Mapping):
        raise ConfigError("weights must be a mapping of metric name -> weight")
    unknown = set(weights) - set(METRIC_NAMES)
    if unknown:
        raise ConfigError(f"weights: unknown metric(s) {sorted(unknown)}; expected {list(METRIC_NAMES)}")
    missing = set(METRIC_NAMES) - set(weights)
    if missing:
        raise ConfigError(f"weights: missing metric(s) {sorted(missing)}")
    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        value = weights[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"weights.{name} must be a number, got {value!r}")
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ConfigError(f"weights.{name} must be finite")
        if value < 0:
            raise ConfigError(f"weights.{name} must be >= 0, got {value}")
        out[name] = value
    total = sum(out.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ConfigError(f"weights must sum to 1.0 (tolerance {WEIGHT_TOLERANCE}), got {total:.6f}")
    return out


def _build_dataclass(cls: type, data: Mapping[str, Any], path: str) -> Any:
    """Recursively build a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must be a mapping, got {type(data).__name__}")
    allowed = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    kwargs: dict[str, Any] = {}
    for name, f in allowed.items():
        if name not in data:
            continue
        value = data[name]
        ftype = f.type if not isinstance(f.type, str) else None
        nested = _NESTED.get((cls, name))
        if nested is not None:
            kwargs[name] = _build_dataclass(nested, value, f"{path}.{name}")
        else:
            kwargs[name] = value
        del ftype
    return cls(**kwargs)


# Map (owner dataclass, field name) -> nested dataclass type. Explicit so we do
# not depend on evaluating string annotations.
_NESTED: dict[tuple[type, str], type] = {
    (BenchmarkConfig, "input"): InputConfig,
    (BenchmarkConfig, "preprocessing"): PreprocessingConfig,
    (BenchmarkConfig, "metrics"): MetricsConfig,
    (BenchmarkConfig, "output"): OutputConfig,
    (MetricsConfig, "ssim"): SSIMConfig,
    (MetricsConfig, "lpips"): LPIPSConfig,
    (MetricsConfig, "edge"): EdgeConfig,
}


def config_from_dict(data: Mapping[str, Any] | None) -> BenchmarkConfig:
    """Build and validate a :class:`BenchmarkConfig` from a nested mapping."""
    data = dict(data or {})
    cfg = _build_dataclass(BenchmarkConfig, data, "config")
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None) -> BenchmarkConfig:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML file. ``None`` returns the built-in defaults.

    Raises:
        ConfigError: if the file is missing, not valid YAML or fails validation.
    """
    if path is None:
        cfg = BenchmarkConfig()
        cfg.validate()
        return cfg
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {path}: {exc}") from exc
    if data is None:
        data = {}
    logger.debug("Loaded config from %s", path)
    return config_from_dict(data)
