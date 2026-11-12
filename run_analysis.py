#!/usr/bin/env python3
"""Reproduce the numerical analyses, independent-design benchmark and S1 tables."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path,
                        help='Portable prepared inputs or manifested raw sources')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='New directory, or a matching completed/interrupted workflow to resume')
    parser.add_argument('--smoke', action='store_true',
                        help='Development-only: original 2 replicates; expansion 2 designs x 4 replicates')
    args = parser.parse_args(argv)
    try:
        if sys.version_info < (3, 10):
            raise RuntimeError('Python 3.10 or later is required')
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[name] = '1'
        from coffee_cocoa_core.workflow_v2 import run
        summary = run(args.input_dir, args.output_dir, smoke=args.smoke,
                      log_fn=lambda message: print(message, flush=True))
        from publication.export_v2 import export_all
        export_all(args.output_dir, args.output_dir / 'publication',
                   input_dir=Path(summary['prepared_input_dir']), figures=False)
        print(json.dumps(dict(status=summary['status'], development_run=args.smoke,
                              numerical_output=str(args.output_dir.resolve()),
                              supplementary_output=str((args.output_dir / 'publication').resolve())),
                         ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print('INCOMPLETE: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
