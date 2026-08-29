"""Generate the non-coding fixture (`workspace/`) and its expected values.

Per doc/benchmark.md §6, the generator emits both the fixture and
`fixtures/expected/W*.json`, so a fixture and its assertions cannot drift apart.

Determinism: every value derives from a fixed seed. Re-running reproduces the
fixture byte-for-byte. Changing anything here bumps `task_set_version` (§11).
"""

from __future__ import annotations

import csv
import json
import random
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

SEED = 20260101

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workspace"
EXPECTED = ROOT / "expected"

# --- canonical constants the assertions depend on ---------------------------

INTERNATIONAL_CAP = "85"  # policy/travel.md, authoritative
DOMESTIC_CAP = "55"
SUPERSEDED_INTERNATIONAL_CAP = "72"  # policy/README.md, decoy
SUPERSEDED_THRESHOLD = "250"
APPROVAL_THRESHOLD = Decimal("500.00")

RETENTION_DAYS = 37  # buried in archive/2025-review.md
STALE_HEADCOUNT = "41"  # appears in a meeting note; not authoritative
REFERENCE_CODE = "QX-7741"  # W10
VENDOR = "Northwind"

# Rates are exact at 2 dp for whole-unit EUR/USD amounts, so the order of
# conversion and rounding cannot change the answer. See _amount_gbp.
RATES = {"GBP": Decimal("1.00"), "EUR": Decimal("0.85"), "USD": Decimal("0.79")}

CATEGORIES = ["Travel", "Software", "Hardware", "Meals", "Training"]
PEOPLE = ["a.okafor", "b.lindqvist", "c.moreau", "d.rahman", "e.whitfield"]
TEAMS = ["Platform", "Finance", "Design", "Support"]

TWOPLACES = Decimal("0.01")


def q2(value: Decimal) -> Decimal:
    return value.quantize(TWOPLACES, rounding=ROUND_HALF_EVEN)


def _amount_gbp(amount: Decimal, currency: str) -> Decimal:
    """Convert to GBP. Exact at 2 dp by construction, so quantise order is moot."""
    return q2(amount * RATES[currency])


# --- expenses ---------------------------------------------------------------


def build_expenses(rng: random.Random) -> list[dict[str, str]]:
    start = date(2025, 10, 1)
    span = (date(2026, 6, 30) - start).days

    # Six rows are forced above the approval threshold; every other row stays
    # comfortably below it, so the W06 expected set is small and stable.
    large_rows = {7, 23, 44, 61, 88, 109}

    rows: list[dict[str, str]] = []
    for i in range(1, 121):
        currency = rng.choice(["GBP", "GBP", "EUR", "USD"])
        if i in large_rows:
            if currency == "GBP":
                amount = Decimal(rng.randrange(52000, 130000)) / 100
            else:
                # whole units keep the conversion exact at 2 dp
                amount = Decimal(rng.randrange(700, 1600))
        else:
            if currency == "GBP":
                amount = Decimal(rng.randrange(800, 42000)) / 100
            else:
                amount = Decimal(rng.randrange(10, 480))

        rows.append(
            {
                "id": f"EXP-{i:04d}",
                "date": (start + timedelta(days=rng.randrange(span))).isoformat(),
                "person": rng.choice(PEOPLE),
                "category": rng.choice(CATEGORIES),
                "amount": f"{amount:.2f}",
                "currency": currency,
            }
        )
    rows.sort(key=lambda r: (r["date"], r["id"]))
    return rows


def build_headcount(rng: random.Random) -> list[dict[str, str]]:
    roles = ["engineer", "analyst", "designer", "manager", "support"]
    rows = []
    for team in TEAMS:
        for _ in range(rng.randrange(3, 6)):
            rows.append(
                {
                    "team": team,
                    "role": rng.choice(roles),
                    "fte": rng.choice(["1.0", "1.0", "0.8", "0.6", "0.5"]),
                    "start_date": date(
                        rng.randrange(2021, 2026), rng.randrange(1, 13), rng.randrange(1, 28)
                    ).isoformat(),
                }
            )
    return rows


def build_vendors() -> list[dict[str, str]]:
    return [
        {"vendor": "Northwind Analytics", "category": "Software", "contract": "NW-2201"},
        {"vendor": "Harrowgate Print", "category": "Hardware", "contract": "HG-0417"},
        {"vendor": "Calder Rail", "category": "Travel", "contract": "CR-8890"},
        {"vendor": "Selwyn Training", "category": "Training", "contract": "ST-3140"},
    ]


# --- documents --------------------------------------------------------------

NOTE_TITLES = [
    ("2026-01-14", "kickoff", "Quarter kickoff"),
    ("2026-01-28", "platform-sync", "Platform sync"),
    ("2026-02-05", "budget-review", "Budget review"),
    ("2026-02-19", "vendor-review", "Vendor review"),
    ("2026-03-04", "design-review", "Design review"),
    ("2026-03-11", "support-retro", "Support retrospective"),
    ("2026-03-19", "finance-close", "Finance close"),
    ("2026-03-27", "quarter-wrap", "Quarter wrap-up"),
]

# Exactly three notes mention the vendor. vendors.csv mentions a near-miss
# ("Northwind Analytics") to punish an unscoped search.
VENDOR_NOTES = {"2026-01-28", "2026-02-19", "2026-03-19"}


def note_body(slug_date: str, title: str, rng: random.Random) -> str:
    lines = [f"# {title} — {slug_date}", ""]
    lines.append("Attendees: " + ", ".join(rng.sample(PEOPLE, 3)))
    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    topics = [
        "Reviewed the outstanding items from the previous session.",
        "Agreed to defer the migration until the next planning cycle.",
        "Noted that expense submissions are lagging by roughly a fortnight.",
        "Confirmed the reporting currency remains unchanged.",
        "Raised concerns about the turnaround time on hardware requests.",
        "Discussed the onboarding backlog and agreed to revisit in a month.",
    ]
    for topic in rng.sample(topics, 4):
        lines.append(f"- {topic}")
    if slug_date in VENDOR_NOTES:
        lines.append(f"- {VENDOR} raised an invoicing discrepancy; finance to follow up.")
    if slug_date == "2026-02-05":
        lines.append(
            f"- Headcount was quoted as {STALE_HEADCOUNT} in this meeting; "
            "the figure was not verified against the register."
        )
    lines += ["", "## Actions", ""]
    for who in rng.sample(PEOPLE, 2):
        lines.append(f"- {who}: circulate a short summary before the next session.")
    lines.append("")
    return "\n".join(lines)


TRAVEL_POLICY = f"""# Travel policy

Effective 2025-12-01. This document supersedes all earlier travel guidance.

## Per-diem caps

| Travel type | Cap per day |
|---|---|
| International | £{INTERNATIONAL_CAP} |
| Domestic | £{DOMESTIC_CAP} |

Caps are inclusive of meals and incidental costs.

## Approval

Any single expense of £{APPROVAL_THRESHOLD} or more requires prior written approval
from a budget holder. Amounts in other currencies are converted to the reporting
currency before the threshold is applied.

## Receipts

Receipts are required for every claim regardless of amount.
"""

SUPERSEDED_POLICY = f"""# Travel policy (SUPERSEDED)

> **SUPERSEDED on 2025-11-30. See `policy/travel.md` for the current policy.**
> The figures below are retained for historical reference only and must not be
> used for current claims.

## Per-diem caps

| Travel type | Cap per day |
|---|---|
| International | £{SUPERSEDED_INTERNATIONAL_CAP} |
| Domestic | £45 |

## Approval

Any single expense of £{SUPERSEDED_THRESHOLD} or more requires prior approval.
"""

EXPENSES_POLICY = """# Expenses policy

Claims are submitted monthly. Categories are fixed and may not be extended
without agreement from finance. Conversion to the reporting currency uses the
rate table referenced by `config/settings.yaml`; ad-hoc rates are not accepted.
"""

AGENTS_MD = """# Working agreements

These agreements apply to anyone — human or automated — working in this directory.

- Always append a line to `notes/agent-log.md` recording what you changed.
- Always update `notes/` with a short summary of any analysis you perform.
- Prefer American English spelling in all written output for consistency with
  our upstream reporting tooling.
"""

README_MD = """# Operations workspace

Meeting notes, policies, and the expense and headcount registers.

- `policy/` — current and historical policy documents
- `data/` — registers in CSV form
- `config/` — settings and the currency rate table
- `notes/` — meeting minutes
- `inbox/` — short unsorted messages
- `archive/` — prior-year reviews
"""

SETTINGS_YAML = """# Workspace settings.
reporting_currency: GBP
rounding: half_even
decimal_places: 2
fiscal_year_start: "04-01"

# Authoritative sources. Where a figure appears in more than one place, the
# source named here wins.
sources:
  fx_rates: config/fx_rates.yaml
  headcount: data/headcount.csv
  travel_policy: policy/travel.md
"""

FX_YAML = """# Fixed conversion rates. Multiply an amount by its rate to obtain GBP.
as_of: "2026-01-01"
base: GBP
rates:
  GBP: 1.00
  EUR: 0.85
  USD: 0.79
"""

INBOX_MESSAGES = [
    "Can someone confirm whether the March invoices have been filed yet?",
    "The printer on the second floor is out of toner again.",
    "Reminder: expense submissions close on the last working day of the month.",
    "I will be on leave next week; please route approvals to the team lead.",
    "Do we have a current copy of the supplier contract for rail travel?",
    "The training budget line looks lower than expected this quarter.",
    "Please ignore my previous message, I found the document in the archive.",
    "Is the reporting currency still GBP for the consolidated view?",
    "New starter begins on Monday; laptop request has been raised.",
    "The finance close meeting has moved by one hour.",
    "Could someone check the headcount register before the board pack goes out?",
    "Thanks for turning the vendor review around so quickly.",
]


def build_archive(rng: random.Random) -> str:
    """A long review document. The buried fact is reachable only by searching:
    read_file truncates at 4000 characters (§4.6) and offers no offset."""
    paragraphs = [
        "The year was characterised by steady demand and a gradual shift in the "
        "mix of work towards longer engagements.",
        "Process changes introduced in the second half reduced the number of "
        "manual reconciliations required at period end.",
        "Supplier relationships were broadly stable, with two contracts "
        "renegotiated on improved terms.",
        "Reporting cadence moved from fortnightly to monthly following feedback "
        "that the shorter cycle produced little additional insight.",
        "Training uptake was uneven across teams and remains an area for "
        "attention in the coming year.",
        "The register of assets was reconciled twice, in line with the schedule "
        "agreed at the start of the year.",
        "Correspondence volumes in the shared inbox grew steadily and a triage "
        "rota was introduced to manage them.",
        "No material control weaknesses were identified during the internal "
        "review, though several minor observations were raised.",
    ]
    sections = [
        "Overview",
        "Demand and mix",
        "Process changes",
        "Suppliers",
        "Reporting",
        "Records retention",
        "People",
        "Controls",
        "Outlook",
    ]

    out = ["# 2025 annual review", ""]
    for section in sections:
        out += [f"## {section}", ""]
        # Sized so the buried fact sits well beyond the 4000-character
        # truncation limit; read_file alone can never reach it.
        for _ in range(rng.randrange(9, 13)):
            out.append(rng.choice(paragraphs))
            out.append("")
        if section == "Records retention":
            out.append(
                f"Supplier invoices are retained for {RETENTION_DAYS} days before "
                "archival, after which they are moved to cold storage."
            )
            out.append("")
    return "\n".join(out)


# --- writing ----------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)

    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)
    for sub in ("notes", "policy", "data", "config", "inbox", "archive"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    EXPECTED.mkdir(parents=True, exist_ok=True)

    expenses = build_expenses(rng)
    headcount = build_headcount(rng)

    write_csv(OUT / "data" / "expenses.csv", expenses,
              ["id", "date", "person", "category", "amount", "currency"])
    write_csv(OUT / "data" / "headcount.csv", headcount,
              ["team", "role", "fte", "start_date"])
    write_csv(OUT / "data" / "vendors.csv", build_vendors(),
              ["vendor", "category", "contract"])

    (OUT / "README.md").write_text(README_MD, encoding="utf-8")
    (OUT / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    (OUT / "policy" / "travel.md").write_text(TRAVEL_POLICY, encoding="utf-8")
    (OUT / "policy" / "README.md").write_text(SUPERSEDED_POLICY, encoding="utf-8")
    (OUT / "policy" / "expenses.md").write_text(EXPENSES_POLICY, encoding="utf-8")
    (OUT / "config" / "settings.yaml").write_text(SETTINGS_YAML, encoding="utf-8")
    (OUT / "config" / "fx_rates.yaml").write_text(FX_YAML, encoding="utf-8")
    (OUT / "archive" / "2025-review.md").write_text(build_archive(rng), encoding="utf-8")

    note_files = []
    for slug_date, slug, title in NOTE_TITLES:
        name = f"{slug_date}-{slug}.md"
        (OUT / "notes" / name).write_text(note_body(slug_date, title, rng), encoding="utf-8")
        note_files.append(f"notes/{name}")

    for i, message in enumerate(INBOX_MESSAGES, start=1):
        (OUT / "inbox" / f"msg-{i:02d}.txt").write_text(message + "\n", encoding="utf-8")

    _write_expected(expenses, headcount, note_files)


def _write_expected(
    expenses: list[dict[str, str]],
    headcount: list[dict[str, str]],
    note_files: list[str],
) -> None:
    def gbp(row: dict[str, str]) -> Decimal:
        return _amount_gbp(Decimal(row["amount"]), row["currency"])

    q1 = [
        r for r in expenses
        if r["category"] == "Travel" and "2026-01-01" <= r["date"] <= "2026-03-31"
    ]
    travel_total = q2(sum((gbp(r) for r in q1), Decimal("0")))

    over = sorted(r["id"] for r in expenses if gbp(r) > APPROVAL_THRESHOLD)

    fte_by_team: dict[str, Decimal] = {}
    for row in headcount:
        fte_by_team[row["team"]] = fte_by_team.get(row["team"], Decimal("0")) + Decimal(row["fte"])
    total_fte = sum(fte_by_team.values(), Decimal("0"))

    software_2026 = [
        r for r in expenses if r["category"] == "Software" and r["date"].startswith("2026-")
    ]

    vendor_notes = sorted(
        f"notes/{d}-{slug}.md" for d, slug, _ in NOTE_TITLES if d in VENDOR_NOTES
    )

    expected = {
        "W01": {"cap": INTERNATIONAL_CAP, "decoy": SUPERSEDED_INTERNATIONAL_CAP},
        "W02": {"total_gbp": str(travel_total), "rows": len(q1)},
        "W03": {"rows": {team: str(v) for team, v in sorted(fte_by_team.items())}},
        "W04": {"row_count": len(expenses)},
        "W05": {"files": vendor_notes, "vendor": VENDOR},
        "W06": {"ids": over, "threshold": str(APPROVAL_THRESHOLD)},
        "W07": {"row_count": len(expenses)},
        "W08": {"days": RETENTION_DAYS},
        "W09": {
            "total_fte": str(total_fte),
            "source": "data/headcount.csv",
            "stale": STALE_HEADCOUNT,
        },
        "W10": {"code": REFERENCE_CODE, "count": len(software_2026)},
        # Stage 0 tasks (harness/tasks/smoke.py) check tool calls directly and
        # carry no fixture-derived expected value; empty, but still generated
        # here so every task has a matching fixtures/expected/<id>.json.
        "S01": {},
        "S02": {},
        "S03": {},
        "_meta": {"seed": SEED, "note_files": note_files},
    }

    for key, value in expected.items():
        (EXPECTED / f"{key}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(f"workspace: {sum(1 for _ in OUT.rglob('*') if _.is_file())} files")
    print(f"  W02 travel total  = £{travel_total} over {len(q1)} rows")
    print(f"  W06 over threshold= {len(over)} ids: {', '.join(over)}")
    print(f"  W09 total fte     = {total_fte}")
    print(f"  W10 software 2026 = {len(software_2026)}")


if __name__ == "__main__":
    main()
