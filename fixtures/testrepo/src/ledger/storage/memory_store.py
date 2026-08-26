"""In-memory entry store."""

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
