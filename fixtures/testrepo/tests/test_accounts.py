from decimal import Decimal

from ledger.accounts import Account


def test_credit_and_debit():
    account = Account("cash")
    account.credit(Decimal("10.00"))
    account.debit(Decimal("2.50"))
    assert account.balance == Decimal("7.50")


def test_new_account_is_open():
    assert Account("cash").closed is False
