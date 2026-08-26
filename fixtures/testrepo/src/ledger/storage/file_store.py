"""File-backed entry store."""

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
