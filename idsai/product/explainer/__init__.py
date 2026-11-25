"""Thai explainer: deterministic template text for an Alert plus a number validator."""

from idsai.product.explainer.history import HISTORY_SOURCE_TH, HISTORY_TH, history_line
from idsai.product.explainer.template import (
    BANNED_TH,
    Explanation,
    allowed_numbers,
    explain,
    numbers_in,
    validate_numbers,
)

__all__ = [
    "BANNED_TH",
    "Explanation",
    "HISTORY_SOURCE_TH",
    "HISTORY_TH",
    "allowed_numbers",
    "explain",
    "history_line",
    "numbers_in",
    "validate_numbers",
]
