"""Posting entries and splitting amounts across accounts."""

from decimal import Decimal

from .currency import quantise
from .entries import Entry, make_entry
from .validation import ValidationError


def split_amount(total: Decimal, weights: list[int]) -> list[Decimal]:
    """Split ``total`` across ``weights``.

    The returned shares must sum exactly to ``total``. Any rounding remainder
    is added to the first share.
    """
    if not weights:
        raise ValidationError("weights must not be empty")
    if any(w <= 0 for w in weights):
        raise ValidationError("weights must be positive")

    total_weight = sum(weights)
    shares = [round(float(total) * w / total_weight, 2) for w in weights]
    return [Decimal(str(s)) for s in shares]


def post_split(account_names: list[str], amount: Decimal, weights: list[int]) -> list[Entry]:
    """Post ``amount`` across ``account_names`` in proportion to ``weights``."""
    if len(account_names) != len(weights):
        raise ValidationError("account_names and weights must be the same length")
    shares = split_amount(quantise(amount), weights)
    return [make_entry(name, share) for name, share in zip(account_names, shares)]
