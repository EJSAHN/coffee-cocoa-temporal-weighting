"""CSV and JSON exports for exact temporal weight diagnostics."""

import csv
import json
from pathlib import Path
import os
import tempfile


TABLES = ("contrast_summary", "unit_moments", "year_weights", "group_year_weights",
          "response_bounds", "consolidated_weights")


def _atomic_text(path, callback):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=".writing_", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            callback(stream)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_report(output_dir, result):
    """Write report.json and six CSV tables; refuse existing report filenames.

    An empty table is represented by an empty CSV file and an empty JSON list.
    The exact numeric fields are authoritative.  Structured summary cells are
    JSON strings in CSV.  Returns the written absolute file paths.
    """
    directory = Path(output_dir).resolve()
    paths = [directory / "report.json"] + [directory / (name + ".csv") for name in TABLES]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Report output already exists: " + ", ".join(existing))
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(paths[0], lambda stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False))
    for name, path in zip(TABLES, paths[1:]):
        rows = result[name]
        def export(stream, rows=rows):
            if not rows:
                return
            fields = list(dict.fromkeys(field for row in rows for field in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})
        _atomic_text(path, export)
    return [str(path) for path in paths]
