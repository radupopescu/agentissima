"""Schema migrations for the file store."""

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
