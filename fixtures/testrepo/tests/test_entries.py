from decimal import Decimal

import pytest

from ledger.entries import make_entry, total
from ledger.validation import ValidationError


def test_make_entry_strips_account():
    entry = make_entry("  cash  ", Decimal("10.00"))
    assert entry.account == "cash"


def test_make_entry_rejects_empty_account():
    with pytest.raises(ValidationError):
        make_entry("", Decimal("10.00"))


def test_total():
    entries = [make_entry("a", Decimal("1.10")), make_entry("b", Decimal("2.20"))]
    assert total(entries) == Decimal("3.30")
