#!/usr/bin/env python3
"""The pre-registered readout of the downstream suite over several models.

  python scripts/read_bench.py --model standard=OUT/bench/<run>_s<step> ... --main alf --out readout.json

See `python scripts/read_bench.py --help` for the screen, the seed averaging and the noise floor.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from alf.bench.readout import main

if __name__ == "__main__":
    main()
