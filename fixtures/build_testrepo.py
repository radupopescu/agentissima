"""Generate the coding fixture (`testrepo/`) and its expected values.

Per benchmark.md §6 the generator emits both the fixture and
`fixtures/expected/T*.json`. Expected values that describe the source (which
modules raise ValidationError, how many test files exist) are *derived by
scanning the generated tree*, not hand-written, so they cannot drift.

The base repo has exactly one failing test: test_split_posting_balances.
`tests/test_close.py` is not part of the base tree — it is added by the T09
fixture variant (see harness/tasks/repo.py), so that T03's "the whole suite
passes" assertion is exact.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

SEED = 20260202

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "testrepo"
EXPECTED = ROOT / "expected"

RUNBOOK_TOKEN = "OPS-4417"  # buried in docs/operations.md (T08)
REFERENCE_TOKEN = "LG-9032"  # T10
EXPORT_DECIMAL_PLACES = 4  # config/defaults.yaml, end of the 3-hop chain

# --- source files -----------------------------------------------------------

FILES: dict[str, str] = {}

FILES["pyproject.toml"] = """[project]
name = "ledger"
version = "0.3.1"
requires-python = ">=3.11"
dependencies = ["pyyaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
"""

FILES["README.md"] = """# ledger

A small double-entry bookkeeping library.

- `src/ledger/accounts.py` — account objects and balances
- `src/ledger/entries.py` — individual ledger entries
- `src/ledger/posting.py` — posting and splitting amounts across accounts
- `src/ledger/currency.py` — rounding and quantisation
- `src/ledger/reporting/` — balance, trial balance and CSV export
- `src/ledger/storage/` — in-memory and file-backed stores

Run the tests with `pytest`.
"""

FILES["AGENTS.md"] = """# Repository conventions

Anyone working in this repository — human or automated — must:

- Add an entry to `docs/changelog.md` describing every change made.
- Record any investigation in `docs/` before changing code.
- Use American English spelling in comments, docstrings and commit messages.
"""

FILES["src/ledger/__init__.py"] = '''"""A small double-entry bookkeeping library."""

__version__ = "0.3.1"
'''

FILES["src/ledger/validation.py"] = '''"""Validation errors shared across the package."""


class ValidationError(ValueError):
    """Raised when input does not satisfy a ledger invariant."""


def describe(field: str, value: object) -> str:
    """Return a human-readable description of a rejected field value."""
    return f"{field}={value!r}"
'''

FILES["src/ledger/currency.py"] = '''"""Rounding and quantisation for monetary amounts.

All monetary rounding in this package goes through :func:`quantise`. The
rounding mode is fixed here and must not be overridden per call site.
"""

from decimal import ROUND_HALF_EVEN, Decimal

#: The rounding mode used for every monetary amount in this package.
DEFAULT_ROUNDING = ROUND_HALF_EVEN

#: Default number of decimal places for monetary amounts.
DEFAULT_PLACES = 2


def quantise(amount: Decimal, places: int = DEFAULT_PLACES) -> Decimal:
    """Round ``amount`` to ``places`` decimal places using DEFAULT_ROUNDING."""
    exponent = Decimal(1).scaleb(-places)
    return Decimal(amount).quantize(exponent, rounding=DEFAULT_ROUNDING)
'''

FILES["src/ledger/entries.py"] = '''"""Individual ledger entries."""

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
'''

# The planted fault: float arithmetic and per-share rounding, so shares do not
# sum back to the total. The docstring states the intended contract, so the fix
# is discoverable from the file itself.
FILES["src/ledger/posting.py"] = '''"""Posting entries and splitting amounts across accounts."""

from decimal import Decimal

from .currency import quantise
from .entries import Entry, make_entry
from .validation import ValidationError


def split_amount(total: Decimal, weights: list[int]) -> list[Decimal]:
    """Split ``total`` across ``weights``.

    The returned shares must sum exactly to ``total``. Any rounding remainder
    is added to the first share.
    """
    if not weights:
        raise ValidationError("weights must not be empty")
    if any(w <= 0 for w in weights):
        raise ValidationError("weights must be positive")

    total_weight = sum(weights)
    shares = [round(float(total) * w / total_weight, 2) for w in weights]
    return [Decimal(str(s)) for s in shares]


def post_split(account_names: list[str], amount: Decimal, weights: list[int]) -> list[Entry]:
    """Post ``amount`` across ``account_names`` in proportion to ``weights``."""
    if len(account_names) != len(weights):
        raise ValidationError("account_names and weights must be the same length")
    shares = split_amount(quantise(amount), weights)
    return [make_entry(name, share) for name, share in zip(account_names, shares)]
'''

FILES["src/ledger/accounts.py"] = '''"""Account objects and balances."""

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
'''

FILES["src/ledger/reporting/__init__.py"] = '''"""Reporting helpers."""
'''

FILES["src/ledger/reporting/balance.py"] = '''"""Running balances over a sequence of entries."""

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
'''

FILES["src/ledger/reporting/trial.py"] = '''"""Trial balance across accounts."""

from decimal import Decimal

from ..accounts import Account
from ..currency import quantise


def trial_balance(accounts: list[Account]) -> dict[str, Decimal]:
    balances = {account.name: account.balance for account in accounts}
    balances["__total__"] = quantise(sum(balances.values(), Decimal("0")))
    return balances
'''

FILES["src/ledger/reporting/export_csv.py"] = '''"""CSV export of ledger entries."""

import csv
from pathlib import Path

from ..config.settings import export_decimal_places
from ..entries import Entry
from ..validation import ValidationError


def export_entries(entries: list[Entry], path: Path) -> int:
    """Write ``entries`` to ``path`` as CSV. Returns the number of rows written."""
    if not entries:
        raise ValidationError("refusing to export an empty entry list")

    places = export_decimal_places()
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\\n")
        writer.writerow(["account", "amount", "memo"])
        for entry in entries:
            writer.writerow([entry.account, f"{entry.amount:.{places}f}", entry.memo])
    return len(entries)
'''

FILES["src/ledger/storage/__init__.py"] = '''"""Entry storage backends."""
'''

FILES["src/ledger/storage/memory_store.py"] = '''"""In-memory entry store."""

from ..entries import Entry


class MemoryStore:
    def __init__(self) -> None:
        self._entries: list[Entry] = []

    def add(self, entry: Entry) -> None:
        self._entries.append(entry)

    def all(self) -> list[Entry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
'''

FILES["src/ledger/storage/file_store.py"] = '''"""File-backed entry store."""

import json
from decimal import Decimal
from pathlib import Path

from ..entries import Entry
from ..validation import ValidationError


class FileStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, entries: list[Entry]) -> None:
        if self.path.is_dir():
            raise ValidationError("store path must be a file, not a directory")
        payload = [
            {"account": e.account, "amount": str(e.amount), "memo": e.memo} for e in entries
        ]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> list[Entry]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            Entry(account=row["account"], amount=Decimal(row["amount"]), memo=row["memo"])
            for row in payload
        ]
'''

FILES["src/ledger/storage/migrations.py"] = '''"""Schema migrations for the file store."""

SCHEMA_VERSION = 3


def migrate(payload: list[dict], from_version: int) -> list[dict]:
    """Bring ``payload`` up to :data:`SCHEMA_VERSION`."""
    if from_version < 2:
        for row in payload:
            row.setdefault("memo", "")
    if from_version < 3:
        for row in payload:
            row["amount"] = str(row["amount"])
    return payload
'''

FILES["src/ledger/config/__init__.py"] = '''"""Configuration loading."""
'''

FILES["src/ledger/config/settings.py"] = '''"""Settings loaded from defaults.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


@lru_cache(maxsize=1)
def load_defaults() -> dict:
    """Load and cache the contents of defaults.yaml."""
    return yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))


def export_decimal_places() -> int:
    """Number of decimal places used by the CSV export."""
    return int(load_defaults()["export"]["decimal_places"])


def store_format() -> str:
    return str(load_defaults()["storage"]["format"])
'''

FILES["src/ledger/config/defaults.yaml"] = f"""# Package defaults.
export:
  decimal_places: {EXPORT_DECIMAL_PLACES}
  include_memo: true

storage:
  format: json
  schema_version: 3
"""

FILES["src/ledger/cli.py"] = '''"""Minimal command line entry point."""

import argparse
from decimal import Decimal

from .posting import post_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ledger")
    parser.add_argument("--amount", required=True)
    parser.add_argument("--accounts", required=True, help="comma-separated account names")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.accounts.split(",")]
    entries = post_split(names, Decimal(args.amount), [1] * len(names))
    for entry in entries:
        print(f"{entry.account}\\t{entry.amount}")
    return 0
'''

# --- tests (base tree; test_close.py is added by the T09 variant) -----------

FILES["tests/test_currency.py"] = '''from decimal import Decimal

from ledger.currency import DEFAULT_ROUNDING, quantise


def test_quantise_two_places():
    assert quantise(Decimal("1.005")) == Decimal("1.00")


def test_rounding_mode_is_half_even():
    assert DEFAULT_ROUNDING == "ROUND_HALF_EVEN"
'''

FILES["tests/test_entries.py"] = '''from decimal import Decimal

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
'''

FILES["tests/test_accounts.py"] = '''from decimal import Decimal

from ledger.accounts import Account


def test_credit_and_debit():
    account = Account("cash")
    account.credit(Decimal("10.00"))
    account.debit(Decimal("2.50"))
    assert account.balance == Decimal("7.50")


def test_new_account_is_open():
    assert Account("cash").closed is False
'''

FILES["tests/test_balance.py"] = '''from decimal import Decimal

from ledger.entries import make_entry
from ledger.reporting.balance import closing_balance, running_balance


def test_running_balance():
    entries = [make_entry("a", Decimal("5.00")), make_entry("a", Decimal("2.50"))]
    assert running_balance(entries) == [Decimal("5.00"), Decimal("7.50")]


def test_closing_balance_empty():
    assert closing_balance([]) == Decimal("0.00")
'''

FILES["tests/test_posting.py"] = '''from decimal import Decimal

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
'''

# The T09 variant file, written by the task rather than the base generator.
TEST_CLOSE = '''from decimal import Decimal

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
'''

# --- documentation ----------------------------------------------------------

FILES["docs/architecture.md"] = """# Architecture

The package is layered:

1. `currency` — rounding primitives. Depends on nothing else.
2. `validation` — the shared error type.
3. `entries` — validated entry construction.
4. `posting` — splitting and posting across accounts.
5. `reporting` — read-only views over entries and accounts.
6. `storage` — persistence backends.
7. `config` — defaults loaded from `defaults.yaml`.

Layers may only import from layers above them in this list.

Monetary rounding is centralised in `currency.quantise`; no other module may
choose its own rounding mode.
"""

FILES["docs/changelog.md"] = """# Changelog

## 0.3.1
- Added the file-backed store.

## 0.3.0
- Introduced the trial balance report.

## 0.2.0
- Centralised rounding in `currency.quantise`.
"""


def build_operations_doc(rng: random.Random) -> str:
    """Operations runbook. The token sits well past the 4000-character
    truncation limit (§4.6), so it is reachable only by searching."""
    paragraphs = [
        "Routine checks are performed at the start of each working day and the "
        "results recorded in the shared operations log.",
        "Where a check fails, the on-call engineer raises a ticket before "
        "attempting any remediation.",
        "Backups of the file store are verified weekly by restoring into a "
        "scratch directory and comparing checksums.",
        "Schema migrations are applied in a maintenance window and are always "
        "reversible within the same release.",
        "Alert thresholds are reviewed quarterly; changes require sign-off from "
        "the service owner.",
        "Access to the production store is granted for a fixed period and "
        "revoked automatically on expiry.",
        "Incident reviews are held within five working days and produce a short "
        "written summary.",
    ]
    sections = [
        "Daily checks",
        "Backups",
        "Migrations",
        "Alerting",
        "Access",
        "Runbook references",
        "Incidents",
        "Escalation",
    ]

    out = ["# Operations", ""]
    for section in sections:
        out += [f"## {section}", ""]
        for _ in range(rng.randrange(9, 13)):
            out.append(rng.choice(paragraphs))
            out.append("")
        if section == "Runbook references":
            out.append(
                f"The runbook reference for a failed export is {RUNBOOK_TOKEN}; "
                "quote it when raising a ticket."
            )
            out.append("")
    return "\n".join(out)


# --- writing ----------------------------------------------------------------


def main() -> None:
    rng = random.Random(SEED)

    if OUT.exists():
        shutil.rmtree(OUT)

    files = dict(FILES)
    files["docs/operations.md"] = build_operations_doc(rng)

    for relative, content in files.items():
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    (ROOT / "testrepo_variants").mkdir(exist_ok=True)
    (ROOT / "testrepo_variants" / "test_close.py").write_text(TEST_CLOSE, encoding="utf-8")

    _write_expected(files)


def _write_expected(files: dict[str, str]) -> None:
    # Derived by scanning the generated source, never hand-listed.
    raisers = sorted(
        Path(relative).name
        for relative, content in files.items()
        if relative.startswith("src/") and "raise ValidationError" in content
    )
    test_files = sorted(
        Path(relative).name for relative in files if relative.startswith("tests/")
    )

    expected = {
        "T01": {"file": "currency.py", "rounding": "ROUND_HALF_EVEN"},
        "T02": {"file": "posting.py", "test": "test_split_posting_balances"},
        "T03": {"allowed_changes": ["src/ledger/posting.py"]},
        "T04": {"wrong_path": "src/ledger/reporting/balances.py",
                "real_path": "src/ledger/reporting/balance.py",
                "function": "running_balance"},
        "T05": {"files": raisers},
        "T06": {"decimal_places": EXPORT_DECIMAL_PLACES, "source": "defaults.yaml"},
        "T07": {"function": "trial_balance", "file": "src/ledger/reporting/trial.py"},
        "T08": {"token": RUNBOOK_TOKEN},
        "T09": {"variant_file": "tests/test_close.py"},
        "T10": {"token": REFERENCE_TOKEN, "count": len(test_files)},
        "_meta_repo": {"seed": SEED, "test_files": test_files},
    }

    for key, value in expected.items():
        (EXPECTED / f"{key}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(f"testrepo: {sum(1 for p in OUT.rglob('*') if p.is_file())} files")
    print(f"  T05 ValidationError raisers = {', '.join(raisers)}")
    print(f"  T10 base test files         = {len(test_files)}")


if __name__ == "__main__":
    main()
