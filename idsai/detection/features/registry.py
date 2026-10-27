"""Feature registry: names, units, Thai descriptions, z-score transform and floor per feature.

Window features are named `<base>_<W>` for every window `W` in `FeatureCfg.windows_s`; each one
also gets a robust z-score column `z_<base>_<W>` (see `kernel.Baseline`): median-centred, with
`scale = max(IQR / 1.349, tail, floor)`. For log-transformed features (`transform` log1p /
log1p_quote) the scale is split: above the median `tail = (q99 - med) / 2.326`, below it
`tail = (med - q01) / 2.326`; ret / jump / ratio features use one symmetric
`tail = (q99.5 - q0.5) / (2 * 2.576)`. Floors: 0.25 (log), max(1e-4, tick size) (ret), 0.05 (ratio).
State features (`cusum_*`, `kk_*`) are sequential and are not z-scored.
"""

import re
from dataclasses import dataclass
from typing import Literal

from idsai.config import FeatureCfg

Transform = Literal["none", "log1p", "log1p_quote"]
Floor = Literal["log", "ret", "ratio"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    desc_th: str
    transform: Transform
    floor: Floor
    zscored: bool = True  # False for state features (no `z_` column)


_SPECS = [
    FeatureSpec("n_trades", "trades", "จำนวนรายการซื้อขายในหน้าต่างเวลา", "log1p", "log"),
    FeatureSpec("quote_vol", "quote", "มูลค่าซื้อขายรวม (หน่วยสกุลเงินอ้างอิง)", "log1p_quote", "log"),
    FeatureSpec(
        "buy_quote", "quote", "มูลค่าซื้อขายที่ฝั่งผู้ซื้อเป็นผู้เข้าตลาด (taker buy)", "log1p_quote", "log"
    ),
    FeatureSpec(
        "rush", "orders", "จำนวน rush order (fill ฝั่งซื้อ ≥ 2 รายการในมิลลิวินาทีเดียวกัน)", "log1p", "log"
    ),
    FeatureSpec(
        "pingpong",
        "pairs",
        "คู่รายการติดกันที่สลับฝั่งซื้อ-ขาย ราคาและขนาดเท่ากัน ห่างกันไม่เกิน 50 มิลลิวินาที "
        "(นับคู่ที่ซ้อนกัน เช่น ซื้อ/ขาย/ซื้อ = 2)",
        "log1p",
        "log",
    ),
    FeatureSpec("round_frac", "ratio", "สัดส่วนรายการที่ขนาดเป็นเลขกลม", "none", "ratio"),
    FeatureSpec("buy_imb", "ratio", "ความไม่สมดุลของมูลค่าซื้อ-ขาย (−1 ถึง 1)", "none", "ratio"),
    FeatureSpec("ret", "log-return", "ผลตอบแทนแบบลอการิทึมของราคาปิดตลอดหน้าต่าง", "none", "ret"),
    FeatureSpec("jump", "relative", "ราคาสูงสุดในหน้าต่างเทียบราคาปิดก่อนหน้าต่าง (−1)", "none", "ret"),
    FeatureSpec(
        "cusum_ret",
        "sigma",
        "ค่าสะสม CUSUM ด้านบวกของผลตอบแทน 10 วินาทีที่ปรับมาตรฐานแล้ว",
        "none",
        "ratio",
        zscored=False,
    ),
    FeatureSpec(
        "cusum_imb",
        "sigma",
        "ค่าสะสม CUSUM ด้านบวกของความไม่สมดุลซื้อ-ขาย 10 วินาที",
        "none",
        "ratio",
        zscored=False,
    ),
    FeatureSpec(
        "kk_vol_ratio",
        "ratio",
        "มูลค่าซื้อขาย 30 นาทีล่าสุดเทียบค่าเฉลี่ย 24 ช่วง 30 นาทีก่อนหน้า",
        "none",
        "ratio",
        zscored=False,
    ),
    FeatureSpec(
        "kk_price_ratio",
        "ratio",
        "ราคาปิดล่าสุดเทียบค่าเฉลี่ยราคาปิด ณ 24 จุดทุก 30 นาทีก่อนหน้า",
        "none",
        "ratio",
        zscored=False,
    ),
]

REGISTRY: dict[str, FeatureSpec] = {s.name: s for s in _SPECS}

# deterministic orders (the kernel packs its row vector in exactly this order)
WINDOW_BASES: tuple[str, ...] = tuple(s.name for s in _SPECS if s.zscored)
STATE_FEATURES: tuple[str, ...] = tuple(s.name for s in _SPECS if not s.zscored)


def raw_columns(cfg: FeatureCfg) -> list[str]:
    return [f"{base}_{w}" for base in WINDOW_BASES for w in cfg.windows_s]


def feature_columns(cfg: FeatureCfg) -> list[str]:
    """All window-row feature columns: raw, then `z_<raw>`, then state features."""
    raw = raw_columns(cfg)
    return [*raw, *(f"z_{c}" for c in raw), *STATE_FEATURES]


def model_features(cfg: FeatureCfg) -> list[str]:
    """Model inputs: all `z_*`, then raw `buy_imb_*`, `round_frac_*`, then `cusum_*`."""
    cols = feature_columns(cfg)
    z = [c for c in cols if c.startswith("z_")]
    ratios = [f"{b}_{w}" for b in ("buy_imb", "round_frac") for w in cfg.windows_s]
    cusum = [c for c in STATE_FEATURES if c.startswith("cusum_")]
    return [*z, *ratios, *cusum]


_COL_RE = re.compile(r"^(?:z_)?(?P<base>[a-z_]+?)_(?P<w>\d+)$")


def spec_for(column: str) -> FeatureSpec:
    """Spec of a column (`quote_vol_60`, `z_quote_vol_60`, `cusum_ret`); KeyError if unknown."""
    if column in REGISTRY:
        return REGISTRY[column]
    m = _COL_RE.match(column)
    if m and m["base"] in WINDOW_BASES:
        return REGISTRY[m["base"]]
    raise KeyError(f"unknown feature column {column!r}")
