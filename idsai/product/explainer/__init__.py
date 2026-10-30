"""Thai explainer: deterministic template text for an Alert plus a number validator."""

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
    "allowed_numbers",
    "explain",
    "numbers_in",
    "validate_numbers",
]
