#!/usr/bin/env python3
"""Held-out loss and bits per byte per Jacobi pass count, one model per GPU.

  python scripts/eval_heldout.py --models <run>:<kind>[,...] --shape d768 --passes 1,2,3,4
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from alf import cli


def main() -> int:
    ap = cli.parser(__doc__)
    ap.add_argument("--passes", default="1,2,3,4", help="comma-separated pass counts")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--eval-batch", type=int, default=64)
    args = ap.parse_args()
    data = args.data_root or f"{cli.data_root()}/dolma3"
    evals = args.eval_root or f"{data}_validation"

    def extra(kind):
        return ["--passes", cli.passes_for(kind, args.passes), "--data-root", data,
                "--eval-root", evals, "--eval-batch", str(args.eval_batch)]

    return cli.one_per_gpu(args, "heldout", "alf.eval.heldout", extra)


if __name__ == "__main__":
    raise SystemExit(main())
