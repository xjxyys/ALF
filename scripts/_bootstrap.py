"""Read `.env` and put the Marin checkout on `sys.path`. Imported first by every entry point in this directory."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alf.cli import load_env  # noqa: E402
from alf.marin.bootstrap import bootstrap  # noqa: E402

load_env()
bootstrap()
