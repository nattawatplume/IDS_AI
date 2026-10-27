"""Polars schemas for the two core tabular contracts: TRADES and LABELS."""

import polars as pl

TRADES: pl.Schema = pl.Schema(
    {
        "ts_us": pl.Int64,
        "recv_us": pl.Int64,
        "symbol": pl.String,
        "price": pl.Float64,
        "qty": pl.Float64,
        "quote_qty": pl.Float64,
        "taker_side": pl.String,
        "trade_id": pl.Int64,
        "n_fills": pl.Int32,
        "source": pl.String,
    }
)

LABELS: pl.Schema = pl.Schema(
    {
        "label_id": pl.String,
        "source": pl.String,
        "symbol": pl.String,
        "t_start_us": pl.Int64,
        "t_end_us": pl.Int64,
        "technique_id": pl.String,
        "family": pl.String,
        "scenario_seed": pl.Int64,  # nullable: only simulated labels carry a seed
        "provenance": pl.String,
    }
)


class SchemaError(ValueError):
    """A frame does not satisfy the expected schema."""


def empty(schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def conform(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Return `df` with exactly the schema's columns, in order, cast to its dtypes.

    Extra columns are dropped. A missing column raises SchemaError.
    """
    missing = [c for c in schema.names() if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    return df.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
