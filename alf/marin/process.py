"""Two process-level helpers the trainer needs on a multi-process JAX job.

They are independent of the model and of this method; they are here because a JAX trainer on a cluster needs them.
"""
from __future__ import annotations

import logging
import os
import sys
import time


def skip_core_dumps() -> None:
    """Make the kernel skip this process's core dump.

    Where the node pipes core dumps to a collector, dumping a crashed trainer's tens of gigabytes stalls its
    thousands of threads for minutes, long enough for the cluster to kill the container. For a piped
    ``core_pattern`` the kernel ignores ``RLIMIT_CORE`` except for the value 1, which makes it skip the dump.
    """
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_CORE)
    if hard == resource.RLIM_INFINITY or hard >= 1:
        resource.setrlimit(resource.RLIMIT_CORE, (1, hard))


def exit_after_success(num_processes: int, process_id: int) -> None:
    """Leave without interpreter teardown once training returned.

    XLA and JAX's distributed client can abort in their destructors (a non-zero exit after a completed, checkpointed
    phase; with several processes the coordination service's task watcher throws when a peer's service goes away
    first). All processes meet first and the coordinator leaves last.
    """
    if num_processes > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("alf_train_exit")
        if process_id == 0:
            time.sleep(10)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
