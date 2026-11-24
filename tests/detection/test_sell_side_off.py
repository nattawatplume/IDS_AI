"""M4 hard constraint: with `features.sell_side` off every existing output is bit-identical.

The golden digests below were computed from the code before M4 (commit fdf7eb4) on the same
synthetic stream; any change to the flag-off feature frame, rule scores, alerts, Chief state,
config hashes or the signatures sha breaks this test.

The float-derived digests (windows, scores, Chief state, alerts) depend on the platform's libm/BLAS,
so they are pinned per `sys.platform` (Windows: local run; Linux: GitHub CI on the same commit).

`configs/default.yaml` has since moved to the live AI layer (`ai.iforest`, `chief.ai_mode: shift`);
`PRE_LIVE_AI` puts the old values back, so the digests (default config hash included) also prove
that `ChiefCfg.ai_*` at their defaults leave the dump byte-identical.
"""

import hashlib
import json
import sys
from datetime import date

import numpy as np
import polars as pl

from idsai.ai.chief import Chief
from idsai.config import REPO_ROOT, config_hash, file_sha, load_settings
from idsai.contracts.frames import TRADES
from idsai.detection.features.bars import trades_to_bars
from idsai.detection.features.registry import feature_columns, model_features
from idsai.detection.features.windows import compute_windows
from idsai.detection.registry import build_detectors
from idsai.detection.rules import SIGNATURES_PATH
from idsai.pipeline.batch import score_windows
from tests.synth import day_us, make_trades

S = 1_000_000
SYM = "TESTBTC"
D = date(2025, 3, 1)
T0 = day_us(D)
UNIT = 1e-4

OLD_BARS = [
    "symbol", "t_us", "n_trades", "buy_quote", "sell_quote", "quote_vol", "open", "high",
    "low", "close", "rush_orders", "max_trade_quote", "pingpong", "n_round",
]  # fmt: skip

PRE_LIVE_AI = {
    "detectors": ["rules", "stat.robust_z", "stat.cusum", "ai.autoencoder_trade"],
    "chief": {"ai_mode": "fuse", "ai_shift": 0.0},
    "features": {"sell_side": False},  # default.yaml turns it on since the live sell-side change
}

GOLDEN = {
    "model_features": "6b99d52943321bce",
    "bars": "9da72588247ad466",
    "config_default": "6da45ffe7efd",
    "config_prev_live": "53e3b662ab96",
    "signatures": "0fe49a5924d7",
}
GOLDEN_FLOAT = {  # keyed by sys.platform
    "win32": {
        "windows": "5912df6351ffc711",
        "scores": "67eae2b9b0864784",
        "alerts": "4932795f4d9dd39d",
        "chief_state": "51bc97774c37b285",
    },
    "linux": {
        "windows": "0bb20f1faacae56a",
        "scores": "b73c81dc15b9e634",
        "alerts": "de492db83f63a5ea",
        "chief_state": "72a4cb0909e8bac6",
    },
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _frame_sha(df: pl.DataFrame) -> str:
    h = hashlib.sha256()
    for c in df.columns:
        h.update(c.encode())
        s = df[c]
        if s.dtype == pl.String:
            h.update("\x00".join(s.to_list()).encode())
        else:
            h.update(np.ascontiguousarray(s.to_numpy()).tobytes())
    return h.hexdigest()[:16]


def _trades() -> pl.DataFrame:
    """2 days of sparse trades with buy rush clusters, a rise and a crash."""
    base = make_trades(D, 2, symbol=SYM, seed=41, rate=0.3)
    rng = np.random.default_rng(42)
    n = 300
    ms = rng.integers(0, 2 * 86400 * 1000 - 1, n) * 1000 + T0
    side = np.where(rng.random(n) < 0.5, "buy", "sell")
    k = 3
    ts = np.repeat(ms, k) + np.tile([0, 100, 200], n)
    price = np.full(n * k, 5e-4)
    qty = np.repeat(rng.integers(1, 9, n) * 10.0, k)
    rush = pl.DataFrame(
        {
            "ts_us": ts.astype(np.int64),
            "recv_us": ts.astype(np.int64),
            "symbol": [SYM] * (n * k),
            "price": price,
            "qty": qty,
            "quote_qty": price * qty,
            "taker_side": np.repeat(side, k).tolist(),
            "trade_id": np.arange(10_000_000, 10_000_000 + n * k, dtype=np.int64),
            "n_fills": np.ones(n * k, dtype=np.int32),
            "source": ["archive"] * (n * k),
        },
        schema=TRADES,
    )
    return pl.concat([base, rush]).sort(["ts_us", "trade_id"])


def _digests() -> dict[str, str]:
    cfg = load_settings(overrides=PRE_LIVE_AI)
    fcfg = cfg.features
    out = {"model_features": _sha(json.dumps(model_features(fcfg)).encode())}
    trades = _trades()
    bars = trades_to_bars(trades, SYM, T0, T0 + 2 * 86400 * S)
    out["bars"] = _frame_sha(bars.select(OLD_BARS))
    win = compute_windows(bars, fcfg, UNIT)
    out["windows"] = _frame_sha(win.select("symbol", "t_end_us", *feature_columns(fcfg)))
    dets = build_detectors(["rules", "stat.robust_z", "stat.cusum"], cfg, 0)
    scored = score_windows(win, dets)
    out["scores"] = _frame_sha(
        scored.select([c for c in scored.columns if c.startswith("score__")])
    )
    day2 = T0 + 86400 * S
    cal = scored.filter(pl.col("t_end_us") < day2)
    ev = scored.filter(pl.col("t_end_us") >= day2)
    names = [f"score__{d.name}" for d in dets]
    chief = Chief(dets, cfg.chief, config_hash(cfg), {})
    chief.calibrate(cal.select("symbol", "t_end_us", *names))
    chief.fit_threshold(chief.fuse(cal.select("symbol", "t_end_us", *names)), 10.0)
    st = chief.state_dict()
    st_json = json.dumps(
        {k: ({n: a.tolist() for n, a in v.items()} if k == "cal" else v) for k, v in st.items()},
        sort_keys=True,
        default=str,
    )
    out["chief_state"] = _sha(st_json.encode())
    alerts = chief.episodes(chief.fuse(ev), provenance="replay")
    out["alerts"] = _sha("\n".join(a.model_dump_json() for a in alerts).encode())
    out["config_default"] = config_hash(cfg)
    out["config_prev_live"] = config_hash(load_settings(REPO_ROOT / "configs/eval_prev_live.yaml"))
    out["signatures"] = file_sha(SIGNATURES_PATH)
    return out


def test_flag_off_outputs_are_bit_identical_to_before_m4():
    if sys.platform not in GOLDEN_FLOAT:
        raise AssertionError(
            f"no golden float digests for platform {sys.platform!r}; run this file as a script "
            "on the pre-M4 commit (fdf7eb4) there and add them to GOLDEN_FLOAT"
        )
    got = _digests()
    assert got == {**GOLDEN, **GOLDEN_FLOAT[sys.platform]}


if __name__ == "__main__":  # prints the digests (used once to freeze GOLDEN)
    print(json.dumps(_digests(), indent=4))
