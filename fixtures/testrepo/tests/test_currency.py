from decimal import Decimal

from ledger.currency import DEFAULT_ROUNDING, quantise


def test_quantise_two_places():
    assert quantise(Decimal("1.005")) == Decimal("1.00")


def test_rounding_mode_is_half_even():
    assert DEFAULT_ROUNDING == "ROUND_HALF_EVEN"
