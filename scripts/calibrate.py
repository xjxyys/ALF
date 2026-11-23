#!/usr/bin/env python3
"""Calibrate the entry biases of the one-softmax ablation at the branch point.

  python scripts/calibrate.py --init-from <base run> --init-step N --shape d768 --data-root <root> --out biases.json

`scripts/train.py --kind attnres-alf-joint` then takes `--feedback-entry-bias biases.json`: that branch adds a
parameter its base does not have, and starting it at the share the base's dormant logits imply keeps the switch from
disturbing the model. `python scripts/calibrate.py --help` lists the rest.
"""
from __future__ import annotations

import os
import sys

import _bootstrap  # noqa: F401

from alf import cli
from alf.calibrate import main


def run() -> None:
    # --init-from is a run name here, as it is for scripts/train.py; the library takes a directory.
    argv = sys.argv[1:]
    if "--init-from" in argv:
        i = argv.index("--init-from")
        if i + 1 >= len(argv):
            raise SystemExit("--init-from needs a value")
        name = argv[i + 1]
        if os.path.sep not in name:
            root = os.environ.get("RUN_ROOT") or cli.run_root()
            argv[i + 1] = os.path.join(root, name)
        sys.argv = [sys.argv[0]] + argv
    main()


if __name__ == "__main__":
    run()
