#!/usr/bin/env python3
"""Build the downstream suite's bundle from its pinned public sources.

  python scripts/prepare_bench.py --dest ${DATA_ROOT}/bench

Needs network access: the tasks come from Hugging Face datasets at pinned revisions, from the public CORE archive,
and from offline generators with fixed seeds. The bundle records the revision and the SHA-256 of every task, so a
rebuild can be checked against an existing one.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from alf.bench.prepare import main

if __name__ == "__main__":
    main()
