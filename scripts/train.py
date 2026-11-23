#!/usr/bin/env python3
"""Train one configuration: a backbone on its own, or a feedback variant branched from a backbone's checkpoint.

  python scripts/train.py --run-name moe_1p6b_plain --variant plain --stop-step 8565 \
      --shape d768 --data dolma --data-root <root> --eval-root <root>_validation

  python scripts/train.py --run-name moe_1p6b_alf --kind standard-alf \
      --init-from moe_1p6b_plain --init-step 8565 --passes 2 --eval-passes 3 \
      --embed-inject plain:0.01 --stop-step 11420 --shape d768 --data dolma --data-root <root>

`--kind` expands into the arguments that build that model (`--list-kinds` prints the table); they can also be given
directly. `--run-name` names a directory under the run root, and `--init-from` takes the run name of the branch
source. For several nodes, run the same command on each with `--num-processes`, `--process-id` and `--coordinator`.
Everything else goes to the trainer; `python scripts/train.py --help` lists it.
"""
from __future__ import annotations

import os
import sys

import _bootstrap  # noqa: F401

from alf import cli, kinds

RANK_FLAGS = (("--inject-rank", "inject_rank"), ("--proj-rank", "proj_rank"),
              ("--src-block", "src_block"), ("--src-rank", "src_rank"))


def _take(argv: list[str], flag: str) -> str | None:
    """Remove `flag VALUE` from argv and return VALUE (None when the flag is absent)."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} needs a value")
    value = argv[i + 1]
    del argv[i:i + 2]
    return value


WRAPPER_HELP = """This entry point adds a few arguments of its own and hands the rest to the trainer:

  --kind NAME          the model to build, from the table of the paper (--list-kinds prints it); expands into the
                       --variant / --feedback-* arguments listed below, which can also be given directly
  --list-kinds         print that table and exit
  --run-name NAME      a directory NAME under the run root, in place of --run-dir and --run-id
  --run-root DIR       where runs live (default: $RUN_ROOT, else <OUTPUT_ROOT>/runs)
  --inject-rank N      with --kind: the ranks the kind needs (--proj-rank, --src-block, --src-rank likewise)

  --init-from NAME     the branch source as a run name, resolved under the run root

The trainer's own arguments follow.
"""


def main() -> None:
    argv = sys.argv[1:]
    if "--list-kinds" in argv:
        print(kinds.table())
        raise SystemExit(0)
    if "--help" in argv or "-h" in argv:
        print(WRAPPER_HELP)

    # --kind is shorthand for a model's arguments; the ranks go to the table, not to the trainer.
    kind = _take(argv, "--kind")
    ranks = {}
    for flag, key in RANK_FLAGS:
        value = _take(argv, flag)
        if value is not None:
            ranks[key] = int(value)
    if kind:
        argv = kinds.model_args(kind, **ranks) + argv
    elif ranks:
        # Without --kind the trainer takes the ranks itself, under its own names.
        for flag, key in RANK_FLAGS:
            if key in ranks:
                argv += [flag.replace("--inject-rank", "--feedback-inject-rank")
                              .replace("--proj-rank", "--feedback-proj-rank"), str(ranks[key])]

    # A run is named, not addressed by path: the trainer wants a directory and an id, and a branch source the same.
    root = _take(argv, "--run-root") or cli.run_root()
    name = _take(argv, "--run-name")
    if name:
        if "--run-dir" in argv or "--run-id" in argv:
            raise SystemExit("--run-name replaces --run-dir and --run-id; give one or the other")
        argv += ["--run-dir", os.path.join(root, name), "--run-id", name]
    init_from = _take(argv, "--init-from")
    if init_from:
        argv += ["--init-from", init_from if os.path.sep in init_from else os.path.join(root, init_from)]

    sys.argv = [sys.argv[0]] + argv

    from alf.trainer import main as train_main

    train_main()


if __name__ == "__main__":
    main()
