"""Running balances over a sequence of entries."""

from decimal import Decimal

from ..currency import quantise
from ..entries import Entry


def running_balance(entries: list[Entry], opening: Decimal = Decimal("0.00")) -> list[Decimal]:
    """Return the balance after each entry, starting from ``opening``."""
    balance = quantise(opening)
    out = []
    for entry in entries:
        balance = quantise(balance + entry.amount)
        out.append(balance)
    return out


def closing_balance(entries: list[Entry], opening: Decimal = Decimal("0.00")) -> Decimal:
    """Return the balance after all ``entries``."""
    balances = running_balance(entries, opening)
    return balances[-1] if balances else quantise(opening)
