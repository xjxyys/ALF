#!/usr/bin/env python3
"""Tokenize the training corpus, and the held-out sources, into levanter caches.

  python scripts/prepare_data.py --src <parquet shards> --out ${DATA_ROOT}/dolma3 --shards 0-255
  python scripts/prepare_data.py --check --out ${DATA_ROOT}/dolma3

`--check` writes nothing: it reports how many units of the range are complete and exits non-zero if any are missing,
which is the pre-flight before a training run. `python scripts/prepare_data.py --help` lists the rest.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from alf.tokenize_dolma import main

if __name__ == "__main__":
    main()
