#!/usr/bin/env python3
"""Build the downstream bundle: the 20 tasks of arXiv:2604.21106 (v3) plus 6 further CORE tasks.

The suite is `alf/bench/suite.json` -- 26 tasks with their axes, few-shot counts and expected question counts. The
task formats and the record transforms are the author's, vendored from `kschwethelm/looped-lm-scaling` (MIT) at
`ad7d8f3`: the CORE-sourced tasks are copied out of the published CORE archive, the HuggingFace-sourced ones are
pinned to the dataset revision they were downloaded at, and the variable-assignment and induction-head tasks are
generated offline from fixed seeds. `alf.bench.bundle` does the work and documents each source.

The 6 CORE tasks beyond the paper's 20 are hellaswag (10-shot and 0-shot), jeopardy, arc_challenge, piqa and squad,
set up as `core.yaml` has them. The three multiple-choice tasks whose choices are bare letters (commonsense_qa,
agi_eval_lsat_ar, bigbench_language_identification) and boolq, whose majority class dominates, are left out.

  python -m alf.bench.prepare --dest DIR [--data-root DIR] [--archive ZIP] [--pin BUNDLE] [--dry-run]

Writes `<dest>/{manifest.json, suite.json, records/<task>.jsonl}` and never writes over an existing bundle. The
HuggingFace-sourced tasks need network access; `--dry-run` needs none and checks everything that can be checked
offline, including that the synthetic and CORE records hash to what `--pin` says they should.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from alf.bench.bundle import default_suite, prepare_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT") or None,
                    help="where the CORE archive is cached (default: $DATA_ROOT, else the temporary directory)")
    ap.add_argument("--archive", default=os.environ.get("CORE_ARCHIVE") or None,
                    help="a local copy of the CORE archive; its SHA-256 is checked either way")
    ap.add_argument("--pin", default=None,
                    help="an earlier bundle to reproduce: its dataset revisions are reused and every task whose "
                         "bytes differ from its manifest is reported")
    ap.add_argument("--offline", action="store_true", help="refuse anything that would reach the network")
    ap.add_argument("--dry-run", action="store_true", help="resolve and check the sources, write nothing")
    ap.add_argument("--allow-count-mismatch", action="store_true")
    ap.add_argument("--report", default=None, help="write the manifest, or the dry-run plan, to this JSON file too")
    args = ap.parse_args()

    out = prepare_bundle(args.dest, default_suite(), pin=args.pin, data_root=args.data_root, archive=args.archive,
                         online=not args.offline, allow_count_mismatch=args.allow_count_mismatch,
                         dry_run=args.dry_run)
    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("dry run: nothing written" if args.dry_run else f"bundle {args.dest}")


if __name__ == "__main__":
    main()
