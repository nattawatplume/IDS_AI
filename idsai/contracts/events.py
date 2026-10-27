"""Event models that flow on the bus."""

from typing import Literal

from pydantic import BaseModel


class Trade(BaseModel):
    """One aggregated trade row; mirrors `frames.TRADES` column for column."""

    ts_us: int
    recv_us: int
    symbol: str
    price: float
    qty: float
    quote_qty: float
    taker_side: Literal["buy", "sell"]
    trade_id: int
    n_fills: int
    source: Literal["archive", "live", "sim"]
