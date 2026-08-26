"""Minimal command line entry point."""

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
        print(f"{entry.account}\t{entry.amount}")
    return 0
