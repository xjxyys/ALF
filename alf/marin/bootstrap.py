"""Make Marin importable and keep its caches out of its checkout.

The model, the train step and the recipe come from a checkout of `marin-community/marin` at the pinned commit.
Marin's own environment installs that checkout, so it is usually importable already; if it is not, `MARIN_REPO`
points at it and this module puts it on `sys.path`. Every entry point in `scripts/` calls `bootstrap()` before
importing anything that reaches into Marin or levanter, and `alf.launch` passes the same path on to the per-GPU
child processes, which inherit the environment rather than the parent's `sys.path`.

Environment (see .env.example):
  MARIN_REPO            a checkout of marin-community/marin at the pinned commit; needed only when
                        Marin is not installed in the environment (its own environment installs it)
  MARIN_TOKENIZER_DIR   optional: a local copy of marin-community/marin-tokenizer, used offline
  ALF_CACHE_DIR         optional: where the bytecode and JAX compilation caches go (default ~/.cache/alf)
"""
from __future__ import annotations

import importlib.util
import os
import sys

MARIN_COMMIT = "8577d258214dd0dcb992c04458ee4eb10496d4e5"


def marin_installed() -> bool:
    """Whether Marin is already importable, which its own environment arranges by installing it."""
    return importlib.util.find_spec("experiments") is not None


def marin_repo() -> str | None:
    """The Marin checkout to put on `sys.path`, or None when Marin is already importable."""
    repo = os.environ.get("MARIN_REPO", "")
    if not repo:
        if marin_installed():
            return None
        raise SystemExit(
            "Marin is not importable: set MARIN_REPO to a checkout of marin-community/marin at "
            + MARIN_COMMIT + ", or install it in this environment (see README.md)"
        )
    if not os.path.isdir(repo):
        raise SystemExit(f"MARIN_REPO={repo} is not a directory")
    return os.path.abspath(repo)


def bootstrap(*, offline: bool = True) -> str | None:
    """Put the Marin checkout on `sys.path` if it is not installed, point the caches outside it, return its path."""
    repo = marin_repo()
    if repo is not None and repo not in sys.path:
        sys.path.insert(0, repo)

    cache = os.environ.get("ALF_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "alf")
    os.environ.setdefault("PYTHONPYCACHEPREFIX", os.path.join(cache, "pycache"))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(cache, "jax"))
    if offline:
        # The tokenizer is a local copy; nothing else is fetched at run time.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return repo
