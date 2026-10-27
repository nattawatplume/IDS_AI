"""Batch driver: 1 s bars -> window feature rows, via the shared `kernel.Kernel`."""

import numpy as np
import polars as pl

from idsai.config import FeatureCfg
from idsai.detection.features.kernel import BarArrays, Kernel
from idsai.detection.features.registry import feature_columns


def windows_schema(cfg: FeatureCfg) -> pl.Schema:
    return pl.Schema(
        {
            "symbol": pl.String,
            "t_end_us": pl.Int64,
            **dict.fromkeys(feature_columns(cfg), pl.Float64),
        }
    )


def compute_windows(bars: pl.DataFrame, cfg: FeatureCfg, quote_unit: float) -> pl.DataFrame:
    """Window rows for one symbol's contiguous 1 s bars (`bars.trades_to_bars` output).

    One row per `t_end_us` multiple of `stride_s` with
    `bars.t_us.min() + max(windows_s) <= t_end_us <= bars.t_us.max() + 1 s`; columns
    `symbol, t_end_us, *feature_columns(cfg)`. Each row uses only bars with `t_us < t_end_us`.
    """
    schema = windows_schema(cfg)
    if bars.height == 0:
        return pl.DataFrame(schema=schema)
    symbols = bars["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise ValueError(f"compute_windows needs exactly one symbol, got {symbols}")
    kernel = Kernel(cfg, quote_unit)
    t_end, values = kernel.extend(BarArrays.from_frame(bars.sort("t_us")))
    cols = feature_columns(cfg)
    feats = pl.DataFrame(
        {c: values[:, j] for j, c in enumerate(cols)} if len(t_end) else None,
        schema=dict.fromkeys(cols, pl.Float64),
    )
    return pl.concat(
        [
            pl.DataFrame(
                {"symbol": [symbols[0]] * len(t_end), "t_end_us": np.asarray(t_end, np.int64)},
                schema={"symbol": pl.String, "t_end_us": pl.Int64},
            ),
            feats,
        ],
        how="horizontal",
    ).select(schema.names())
