"""Rounding and quantisation for monetary amounts.

All monetary rounding in this package goes through :func:`quantise`. The
rounding mode is fixed here and must not be overridden per call site.
"""

from decimal import ROUND_HALF_EVEN, Decimal

#: The rounding mode used for every monetary amount in this package.
DEFAULT_ROUNDING = ROUND_HALF_EVEN

#: Default number of decimal places for monetary amounts.
DEFAULT_PLACES = 2


def quantise(amount: Decimal, places: int = DEFAULT_PLACES) -> Decimal:
    """Round ``amount`` to ``places`` decimal places using DEFAULT_ROUNDING."""
    exponent = Decimal(1).scaleb(-places)
    return Decimal(amount).quantize(exponent, rounding=DEFAULT_ROUNDING)
