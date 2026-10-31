"""Stream driver: 1 s bars -> window feature rows, via the shared `kernel.Kernel`.

Bars pushed one at a time go through the same `Kernel.extend` as `windows.compute_windows`, so the
rows are bit-identical to batch when both start at the same first bar.
"""

from typing import Any

from idsai.config import FeatureCfg
from idsai.detection.features.kernel import Baseline, Kernel


class StreamingWindows:
    """Window rows for one symbol, one contiguous 1 s bar at a time."""

    def __init__(
        self, symbol: str, cfg: FeatureCfg, quote_unit: float, *, baseline: Baseline | None = None
    ):
        self.symbol = symbol
        self.kernel = Kernel(cfg, quote_unit, baseline=baseline)

    def push_bar(self, bar: dict[str, Any]) -> dict[str, Any] | None:
        """The row `{symbol, t_end_us, *feature_columns}` ending at this bar's close, if one is due.

        Raises ValueError for another symbol's bar or a bar that is not the next second.
        """
        if bar["symbol"] != self.symbol:
            raise ValueError(f"bar of {bar['symbol']!r} pushed to windows of {self.symbol!r}")
        out = self.kernel.push_bar(
            t_us=bar["t_us"],
            n_trades=bar["n_trades"],
            buy_quote=bar["buy_quote"],
            sell_quote=bar["sell_quote"],
            quote_vol=bar["quote_vol"],
            high=bar["high"],
            close=bar["close"],
            rush_orders=bar["rush_orders"],
            pingpong=bar["pingpong"],
            n_round=bar["n_round"],
        )
        if out is None:
            return None
        t_end_us, values = out
        return {"symbol": self.symbol, "t_end_us": t_end_us, **self.kernel.row_dict(values)}
