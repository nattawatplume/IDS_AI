"""Settings: YAML -> deep-merged overrides -> validated pydantic models."""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_SPLITS = REPO_ROOT / "configs" / "splits.yaml"

DateRange = tuple[date, date]


class PathsCfg(BaseModel):
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    app_db: Path = Path("data/app.duckdb")
    research_db: Path = Path("data/research.duckdb")
    report_json: Path = Path("data/report.json")


class FeatureCfg(BaseModel):
    windows_s: list[int] = [10, 60, 300, 1800]
    stride_s: int = Field(default=10, gt=0)
    baseline_s: int = 86400
    baseline_sample_s: int = Field(default=60, gt=0)
    baseline_refresh_s: int = 300
    min_baseline_samples: int = 360
    quote_unit: dict[str, float] = {"BTC": 1e-4, "USDT": 1.0}

    @model_validator(mode="after")
    def _check_grid(self) -> "FeatureCfg":
        for w in self.windows_s:
            if w <= 0 or w % self.stride_s:
                raise ValueError(f"stride_s={self.stride_s} must divide window {w}")
        if self.baseline_sample_s % self.stride_s:
            raise ValueError(
                f"stride_s={self.stride_s} must divide baseline_sample_s={self.baseline_sample_s}"
            )
        if self.baseline_refresh_s % self.baseline_sample_s:
            raise ValueError(
                f"baseline_sample_s={self.baseline_sample_s} must divide "
                f"baseline_refresh_s={self.baseline_refresh_s}"
            )
        return self


class ChiefCfg(BaseModel):
    budget_per_symbol_day: float = 2.0
    suppress_for_s: int = 300
    agreement_p: float = 0.01
    attribution_min: float = 3.0


class EvalCfg(BaseModel):
    before_s: int = 120
    after_s: int = 300
    seeds: list[int] = [0, 1, 2, 3, 4]
    split_seed: int = 0
    n_boot: int = 1000
    min_effect: float = 0.02
    budgets_per_symbol_day: list[float] = [0.1, 0.5, 2, 10]
    hard_negative_k: int = 20


class SplitsCfg(BaseModel):
    train: DateRange
    accept: DateRange
    cal: DateRange
    test: DateRange
    normal_days_per_symbol: int = 4
    min_gap_days: int = 2
    label_offset_s: dict[str, int] = {"sapienza": 0, "hu": 0}

    @model_validator(mode="after")
    def _check_order(self) -> "SplitsCfg":
        for name in ("train", "cal", "accept", "test"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} start {lo} is after its end {hi}")
        ordered = (
            self.train[1] < self.cal[0]
            and self.cal[1] < self.accept[0]
            and self.accept[1] < self.test[0]
        )
        if not ordered:
            raise ValueError("splits must be strictly ordered: train < cal < accept < test")
        return self


class AECfg(BaseModel):
    hidden: int = 64
    bottleneck: int = 16
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    patience: int = 5
    clip: float = 10.0
    train_stride_s: int = 60


class LiveCfg(BaseModel):
    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT"]
    ws_url: str = "wss://data-stream.binance.vision/stream"
    budget_per_hour: float = 2.0
    universe_size: int = 5
    train_days: int = 7
    bar_grace_s: int = 2


class Settings(BaseModel):
    paths: PathsCfg = PathsCfg()
    features: FeatureCfg = FeatureCfg()
    chief: ChiefCfg = ChiefCfg()
    eval: EvalCfg = EvalCfg()
    splits: SplitsCfg
    ae: AECfg = AECfg()
    live: LiveCfg = LiveCfg()
    detectors: list[str] = ["rules", "stat.robust_z", "stat.cusum", "ai.autoencoder_trade"]
    mode: Literal["replay", "live"] = "replay"
    port: int = 8800


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(path: str | Path | None = None, overrides: dict | None = None) -> Settings:
    """Load settings from YAML (default: configs/default.yaml at the repo root).

    The split dates come from `splits.yaml` next to the config file, falling back to the repo's
    `configs/splits.yaml`. `overrides` is deep-merged on top before validation.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    raw = _read_yaml(cfg_path)
    if "splits" not in raw:
        sibling = cfg_path.parent / "splits.yaml"
        raw["splits"] = _read_yaml(sibling if sibling.exists() else DEFAULT_SPLITS)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Settings.model_validate(raw)


def config_hash(s: Settings) -> str:
    """12-hex sha256 of the canonical JSON dump of the settings."""
    blob = json.dumps(s.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def file_sha(path: str | Path) -> str:
    """12-hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def quote_unit_for(symbol: str, cfg: FeatureCfg) -> float:
    """Quote-volume unit chosen by quote-asset suffix; KeyError if no suffix matches."""
    for suffix, unit in cfg.quote_unit.items():
        if symbol.endswith(suffix):
            return unit
    raise KeyError(f"no quote_unit configured for symbol {symbol!r}")
