"""Validation errors shared across the package."""


class ValidationError(ValueError):
    """Raised when input does not satisfy a ledger invariant."""


def describe(field: str, value: object) -> str:
    """Return a human-readable description of a rejected field value."""
    return f"{field}={value!r}"
