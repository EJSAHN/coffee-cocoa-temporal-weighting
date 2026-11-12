#!/usr/bin/env python3
"""Export exact linear-contrast diagnostics from a CSV or gzipped CSV file."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

# Support a directly invoked local script under Python's isolated mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_diagnostics import DiagnosticInputError, diagnose, write_report


def read_records(path, columns):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise DiagnosticInputError("CSV contains duplicate column names")
        for key in ("unit", "year", "weight"):
            if columns[key] not in fields:
                raise DiagnosticInputError(f"Required CSV column is missing: {columns[key]}")
        for key in ("contrast", "group", "outcome"):
            if columns[key] != key and columns[key] not in fields:
                raise DiagnosticInputError(f"Requested CSV column is missing: {columns[key]}")
        for row_number, raw in enumerate(reader, 2):
            if None in raw:
                raise DiagnosticInputError(f"CSV line {row_number}: too many fields")
            yield {key: raw[column] for key, column in columns.items() if column in fields}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV or CSV.gz file")
    parser.add_argument("--output", required=True, type=Path, help="Output directory; existing report files are never overwritten")
    parser.add_argument("--origin", type=int, default=2006, help="Origin for centered time bases (default: 2006)")
    for name in ("unit", "year", "weight", "contrast", "group", "outcome"):
        parser.add_argument("--" + name + "-column", default=name)
    args = parser.parse_args(argv)
    columns = {name: getattr(args, name + "_column") for name in ("unit", "year", "weight", "contrast", "group", "outcome")}
    try:
        result = diagnose(read_records(args.input, columns), origin=args.origin)
        files = write_report(args.output, result)
    except (DiagnosticInputError, OSError, csv.Error, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps({"status": "COMPLETE_WEIGHT_DIAGNOSTICS", "contrasts": len(result["contrast_summary"]),
                      "files": files}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
