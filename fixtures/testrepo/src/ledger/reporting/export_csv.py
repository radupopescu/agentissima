"""CSV export of ledger entries."""

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
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["account", "amount", "memo"])
        for entry in entries:
            writer.writerow([entry.account, f"{entry.amount:.{places}f}", entry.memo])
    return len(entries)
