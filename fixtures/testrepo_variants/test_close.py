from decimal import Decimal

import pytest

from ledger.accounts import Account


def test_close_returns_final_balance():
    account = Account("cash")
    account.credit(Decimal("12.50"))
    assert account.close() == Decimal("12.50")
    assert account.closed is True


def test_close_twice_is_an_error():
    account = Account("cash")
    account.close()
    with pytest.raises(ValueError):
        account.close()
