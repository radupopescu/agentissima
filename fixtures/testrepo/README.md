# ledger

A small double-entry bookkeeping library.

- `src/ledger/accounts.py` — account objects and balances
- `src/ledger/entries.py` — individual ledger entries
- `src/ledger/posting.py` — posting and splitting amounts across accounts
- `src/ledger/currency.py` — rounding and quantisation
- `src/ledger/reporting/` — balance, trial balance and CSV export
- `src/ledger/storage/` — in-memory and file-backed stores

Run the tests with `pytest`.
