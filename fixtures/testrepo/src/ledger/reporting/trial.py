"""Trial balance across accounts."""

from decimal import Decimal

from ..accounts import Account
from ..currency import quantise


def trial_balance(accounts: list[Account]) -> dict[str, Decimal]:
    balances = {account.name: account.balance for account in accounts}
    balances["__total__"] = quantise(sum(balances.values(), Decimal("0")))
    return balances
