#!/usr/bin/env python3
"""The downstream suite (20 tasks of arXiv:2604.21106 plus 6 carried over from CORE) at every Jacobi pass.

One model at a time, sharded over the GPUs, then merged.

  python scripts/eval_bench.py --models <run>:<kind>[,...] --shape d768 --passes 3 --gpus 8

The bundle is the one `scripts/prepare_bench.py` writes; point `BENCH_BUNDLE` at it (see README.md).
"""
from __future__ import annotations

import os

import _bootstrap  # noqa: F401

from alf import cli


def main() -> int:
    ap = cli.parser(__doc__)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max-per-task", type=int, default=-1, help="-1 = the suite's own cap; a smaller K for a smoke run")
    ap.add_argument("--bundle", default=None, help="default: $BENCH_BUNDLE")
    ap.add_argument("--decode", default="fused", help="the decoding mode the answers are scored under")
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: the whole suite)")
    args = ap.parse_args()
    bundle = args.bundle or os.environ.get("BENCH_BUNDLE", "")
    if not bundle and not args.list_kinds:
        raise SystemExit("set BENCH_BUNDLE (or pass --bundle) to a bundle from scripts/prepare_bench.py; see README.md")

    def extra(kind):
        out = ["--passes", cli.passes_for(kind, str(args.passes)), "--bundle", bundle,
               "--max-per-task", str(args.max_per_task), "--decode", args.decode]
        return out + (["--tasks", args.tasks] if args.tasks else [])

    return cli.sharded(args, "bench", "alf.eval.downstream", extra)


if __name__ == "__main__":
    raise SystemExit(main())
