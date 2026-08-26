from decimal import Decimal

from ledger.entries import make_entry
from ledger.reporting.balance import closing_balance, running_balance


def test_running_balance():
    entries = [make_entry("a", Decimal("5.00")), make_entry("a", Decimal("2.50"))]
    assert running_balance(entries) == [Decimal("5.00"), Decimal("7.50")]


def test_closing_balance_empty():
    assert closing_balance([]) == Decimal("0.00")
