"""Portable table and provenance utilities."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_table(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError('Missing or duplicate columns: '+str(path))
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError('Malformed table: '+str(path))
    return rows


def write_table(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ['status']
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(directory, filename='output_manifest.json'):
    directory = Path(directory)
    hashes = {p.relative_to(directory).as_posix(): sha256(p)
              for p in sorted(directory.rglob('*'))
              if p.is_file() and p.relative_to(directory).as_posix() != filename
              and '__pycache__' not in p.parts}
    write_json(directory/filename, {'sha256': hashes})
    return hashes
