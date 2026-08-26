"""Account objects and balances."""

from decimal import Decimal

from .currency import quantise


class Account:
    """A named account with a running balance."""

    def __init__(self, name: str, balance: Decimal = Decimal("0.00")) -> None:
        self.name = name
        self.balance = quantise(balance)
        self.closed = False

    def debit(self, amount: Decimal) -> Decimal:
        """Decrease the balance by ``amount`` and return the new balance."""
        self.balance = quantise(self.balance - amount)
        return self.balance

    def credit(self, amount: Decimal) -> Decimal:
        """Increase the balance by ``amount`` and return the new balance."""
        self.balance = quantise(self.balance + amount)
        return self.balance

    def close(self) -> Decimal:
        """Close the account and return its final balance.

        Closing an account that is already closed is an error.
        """
        raise NotImplementedError("Account.close is not implemented yet")

    def __repr__(self) -> str:
        return f"Account(name={self.name!r}, balance={self.balance})"
