from decimal import Decimal

import pytest

from ledger.posting import post_split, split_amount
from ledger.validation import ValidationError


def test_split_two_way_exact():
    assert split_amount(Decimal("100.00"), [1, 1]) == [Decimal("50.00"), Decimal("50.00")]


def test_split_posting_balances():
    shares = split_amount(Decimal("100.00"), [1, 1, 1])
    assert sum(shares) == Decimal("100.00")


def test_split_rejects_empty_weights():
    with pytest.raises(ValidationError):
        split_amount(Decimal("10.00"), [])


def test_post_split_length_mismatch():
    with pytest.raises(ValidationError):
        post_split(["a"], Decimal("10.00"), [1, 1])
