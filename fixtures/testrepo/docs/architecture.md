# Architecture

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
