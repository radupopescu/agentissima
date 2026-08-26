"""Individual ledger entries."""

from dataclasses import dataclass
from decimal import Decimal

from .currency import quantise
from .validation import ValidationError, describe


@dataclass(frozen=True)
class Entry:
    account: str
    amount: Decimal
    memo: str = ""


def make_entry(account: str, amount: Decimal, memo: str = "") -> Entry:
    """Build a validated :class:`Entry`."""
    if not account or not account.strip():
        raise ValidationError("account must be a non-empty name")
    if not isinstance(amount, Decimal):
        raise ValidationError("amount must be a Decimal, " + describe("amount", amount))
    return Entry(account=account.strip(), amount=quantise(amount), memo=memo)


def total(entries: list[Entry]) -> Decimal:
    """Sum the amounts of ``entries``."""
    return quantise(sum((e.amount for e in entries), Decimal("0")))
