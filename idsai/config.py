"""Settings: YAML -> deep-merged overrides -> validated pydantic models."""

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_SPLITS = REPO_ROOT / "configs" / "splits.yaml"

DateRange = tuple[date, date]

AiMode = Literal["fuse", "exclude", "mondrian", "shift"]
AI_MODES = ("fuse", "exclude", "mondrian", "shift")


def check_ai_fusion(mode: str, shift: float, regime_feature: str | None) -> None:
    """The one set of rules for how the `ai` layer enters the fusion (`idsai.ai.chief.AiFusion`
    and `ChiefCfg.ai_*` both call it); raises ValueError."""
    if mode not in AI_MODES:
        raise ValueError(f"unknown ai mode {mode!r}; valid: {AI_MODES}")
    if not (math.isfinite(shift) and shift >= 0):
        raise ValueError(f"ai shift must be finite and >= 0, got {shift}")
    if mode != "shift" and shift != 0:
        raise ValueError(f"ai shift {shift} needs mode 'shift', got {mode!r}")
    if (mode == "mondrian") != (regime_feature is not None):
        raise ValueError("ai mode 'mondrian' needs a regime_feature (and only it takes one)")


class _Cfg(BaseModel):
    """Base for config models: unknown keys (typos) are errors."""

    model_config = ConfigDict(extra="forbid")


class PathsCfg(_Cfg):
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    app_db: Path = Path("data/app.duckdb")
    research_db: Path = Path("data/research.duckdb")
    report_json: Path = Path("data/report.json")

    @field_serializer("*")
    def _posix(self, v: Path) -> str:
        # platform-independent serialization (keeps config_hash identical on Windows/Linux)
        return v.as_posix()


class FeatureCfg(_Cfg):
    windows_s: list[int] = [10, 60, 300, 1800]
    stride_s: int = Field(default=10, gt=0)
    baseline_s: int = 86400
    baseline_sample_s: int = Field(default=60, gt=0)
    baseline_refresh_s: int = 300
    min_baseline_samples: int = 360
    quote_unit: dict[str, float] = {"BTC": 1e-4, "USDT": 1.0}
    # M4: sell-side features (`sell_rush`, `drop`, `cusum_sell`) and the sell-side signatures
    # (`configs/signatures_sell.yaml`). Off by default; off leaves every output bit-identical.
    sell_side: bool = False

    @model_serializer(mode="wrap")
    def _drop_default_sell_side(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # off = absent from every dump, so config_hash, the feature-cache key and every stored
        # config stay byte-identical to the ones made before the flag existed
        out = handler(self)
        if not self.sell_side:
            out.pop("sell_side", None)
        return out

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


_AI_DEFAULTS = {"ai_mode": "fuse", "ai_shift": 0.0, "ai_regime_feature": None}
STEALTH_SID = 100102  # M8: the stealth pump rule that `chief.stealth` moves or gates
StealthMode = Literal["attr", "act"]


class ChiefCfg(_Cfg):
    budget_per_symbol_day: float = 2.0
    suppress_for_s: int = 300
    agreement_p: float = 0.01
    attribution_min: float = 3.0
    # how the production Chief fuses the `ai` layer (`idsai.ai.chief.AiFusion`); live AI layer
    # since M2: `ai_mode: shift`, `ai_shift: 1.0` with `ai.iforest` (docs/REPORT.md section 11)
    ai_mode: AiMode = "fuse"
    ai_shift: float = 0.0
    ai_regime_feature: str | None = None
    # M3: naming-only signatures (`configs/signatures_attr.yaml`) name an episode that no F-rule
    # or cusum row named; they never enter F, p, agreement or tau. Off leaves every output
    # bit-identical and is absent from every dump (like `FeatureCfg.sell_side`).
    attr_rules: bool = False
    # M5: pump-rule ids whose rev 2 (`configs/signatures_r2.yaml`) replaces rev 1 in the rule
    # layer (e.g. [100101, 100102]); rev 2 keeps the detector name `rule.<sid>`. Empty (default)
    # leaves every output bit-identical and is absent from every dump (like `attr_rules`).
    rule_rev2: list[int] = []
    # M8: the stealth pump rule 100102. "attr": it leaves the F rule layer and becomes a
    # naming-only rule (`attr.100102`, appended after `signatures_attr.yaml`; needs `attr_rules`).
    # "act": its rev 3 (`configs/signatures_r3.yaml`: rev 1 + `z_n_trades_60 >= 3`) replaces rev 1
    # under the same name `rule.100102`. None (default) leaves every output bit-identical and is
    # absent from every dump (like `rule_rev2`).
    stealth: StealthMode | None = None

    @field_validator("rule_rev2", mode="after")
    @classmethod
    def _sort_rule_rev2(cls, v: list[int]) -> list[int]:
        # a set of rules, not a sequence: sorted so the dump and config_hash do not depend on the
        # order it was written in (repeats are kept here and rejected in `_check_ai`)
        return sorted(v)

    @model_validator(mode="after")
    def _check_ai(self) -> "ChiefCfg":
        check_ai_fusion(self.ai_mode, self.ai_shift, self.ai_regime_feature)
        if len(set(self.rule_rev2)) != len(self.rule_rev2):
            raise ValueError(f"chief.rule_rev2 lists a rule twice: {self.rule_rev2}")
        if self.stealth is not None and STEALTH_SID in self.rule_rev2:
            raise ValueError(
                f"chief.stealth {self.stealth!r} and chief.rule_rev2 both change rule "
                f"{STEALTH_SID}: use one"
            )
        if self.stealth == "attr" and not self.attr_rules:
            raise ValueError(
                "chief.stealth 'attr' makes rule 100102 a naming rule, which needs "
                "chief.attr_rules: true"
            )
        return self

    @model_serializer(mode="wrap")
    def _drop_default_ai(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # all three at their defaults = absent from every dump (like `FeatureCfg.sell_side`), so
        # config_hash and every stored config stay byte-identical to the ones made before them
        out = handler(self)
        if all(getattr(self, k) == v for k, v in _AI_DEFAULTS.items()):
            for k in _AI_DEFAULTS:
                out.pop(k, None)
        if not self.attr_rules:
            out.pop("attr_rules", None)
        if not self.rule_rev2:
            out.pop("rule_rev2", None)
        if self.stealth is None:
            out.pop("stealth", None)
        return out


class EvalCfg(_Cfg):
    before_s: int = 120
    after_s: int = 300
    seeds: list[int] = [0, 1, 2, 3, 4]
    split_seed: int = 0
    n_boot: int = 1000
    min_effect: float = 0.02
    budgets_per_symbol_day: list[float] = [0.1, 0.5, 2, 10]
    hard_negative_k: int = 20


class SplitsCfg(_Cfg):
    train: DateRange
    accept: DateRange
    cal: DateRange
    test: DateRange
    normal_days_per_symbol: int = 4
    min_gap_days: int = 2
    label_offset_s: dict[str, int] = {"sapienza": 0, "hu": 0}
    # retrain_drift training ranges (inside train); None = first / last two thirds of train
    drift_v1: DateRange | None = None
    drift_v2: DateRange | None = None

    @model_validator(mode="after")
    def _check_order(self) -> "SplitsCfg":
        for name in ("train", "cal", "accept", "test", "drift_v1", "drift_v2"):
            rng = getattr(self, name)
            if rng is None:
                continue
            lo, hi = rng
            if lo > hi:
                raise ValueError(f"{name} start {lo} is after its end {hi}")
        for name in ("drift_v1", "drift_v2"):
            rng = getattr(self, name)
            if rng is not None and not (self.train[0] <= rng[0] and rng[1] <= self.train[1]):
                raise ValueError(f"{name} {rng} must lie inside train {self.train}")
        ordered = (
            self.train[1] < self.cal[0]
            and self.cal[1] < self.accept[0]
            and self.accept[1] < self.test[0]
        )
        if not ordered:
            raise ValueError("splits must be strictly ordered: train < cal < accept < test")
        return self


class AECfg(_Cfg):
    hidden: int = 64
    bottleneck: int = 16
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    patience: int = 5
    clip: float = 10.0
    train_stride_s: int = 60


class LiveCfg(_Cfg):
    symbols: list[str] = ["BTCUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT"]
    ws_url: str = "wss://data-stream.binance.vision/stream"
    budget_per_hour: float = 2.0
    universe_size: int = 4
    train_days: int = 7
    bar_grace_s: int = 2


class AuthCfg(_Cfg):
    enabled: bool = False
    cookie_secure: bool = False
    remember_days: int = Field(default=30, gt=0)
    session_hours: int = Field(default=12, gt=0)


class Settings(_Cfg):
    paths: PathsCfg = PathsCfg()
    features: FeatureCfg = FeatureCfg()
    chief: ChiefCfg = ChiefCfg()
    eval: EvalCfg = EvalCfg()
    splits: SplitsCfg
    ae: AECfg = AECfg()
    live: LiveCfg = LiveCfg()
    auth: AuthCfg = AuthCfg()
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
    """12-hex sha256 of the canonical JSON dump of the settings, minus `auth`: sign-in changes no
    model, feature or detection output, so it must not invalidate packs, bundles or run records."""
    blob = json.dumps(
        s.model_dump(mode="json", exclude={"auth"}), sort_keys=True, separators=(",", ":")
    )
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
