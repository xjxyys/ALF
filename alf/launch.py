"""Running the evaluations: one model per GPU, or one model at a time sharded over the GPUs.

The research code drove these from shell launchers inside a terminal multiplexer, with sentinels and site paths; the
parts that are not site plumbing are here: find a run's newest complete checkpoint, start one process per GPU, skip
work whose output is already there, and report which processes produced nothing.

Every job writes a JSON file. A job whose file exists is not run again, so an interrupted evaluation is resumed by
repeating the same command.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


POLL_SECONDS = 60.0  # how often a wait prints how many jobs are done


def gpu_count() -> int:
    """The number of visible GPUs (CUDA_VISIBLE_DEVICES wins, else nvidia-smi)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([d for d in visible.split(",") if d.strip() != ""])
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return len([ln for ln in out.splitlines() if ln.strip()])


def complete_steps(run_dir: str | os.PathLike) -> list[int]:
    """The steps under ``<run_dir>/checkpoints`` whose write finished (they have a metadata.json)."""
    root = Path(run_dir) / "checkpoints"
    return sorted(int(p.name.split("-", 1)[1]) for p in root.glob("step-*")
                  if re.fullmatch(r"step-\d+", p.name) and (p / "metadata.json").exists())


def pick_step(run_dir: str | os.PathLike, step: int | None = None) -> int:
    steps = complete_steps(run_dir)
    if step is not None:
        if step not in steps:
            raise SystemExit(f"{run_dir}: no complete checkpoint at step {step} (have: {steps or 'none'})")
        return step
    if not steps:
        raise SystemExit(f"{run_dir}: no complete checkpoint under checkpoints/")
    return steps[-1]


@dataclasses.dataclass
class Job:
    """One process: its label, its command, the GPU it gets, and the file that says it is done."""

    label: str
    argv: list[str]
    gpu: int
    out: Path
    log: Path


def run_jobs(jobs: list[Job]) -> list[str]:
    """Start every job that has no output yet, wait, and return the labels that produced none."""
    todo = []
    for job in jobs:
        if job.out.exists() and job.out.stat().st_size > 0:
            print(f"[alf] {job.label}: {job.out} exists, skipping", flush=True)
            continue
        todo.append(job)
    if not todo:
        return []

    procs = []
    for job in todo:
        job.out.parent.mkdir(parents=True, exist_ok=True)
        job.log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job.gpu), PYTHONUNBUFFERED="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONPATH=child_path())
        print(f"[alf] {job.label}: start on GPU {job.gpu}, log {job.log}", flush=True)
        with open(job.log, "ab") as log:
            procs.append((job, subprocess.Popen(job.argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True)))

    t0 = time.time()
    try:
        while any(p.poll() is None for _, p in procs):
            done = sum(1 for job, _ in procs if job.out.exists())
            print(f"[alf] {done}/{len(procs)} done, {time.time() - t0:.0f}s", flush=True)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        for _, p in procs:
            p.terminate()
        raise
    for _, p in procs:
        p.wait()

    failed = []
    for job, _ in procs:
        if job.out.exists() and job.out.stat().st_size > 0:
            continue
        tail = ""
        if job.log.exists():
            lines = job.log.read_text(errors="replace").splitlines()
            hits = [ln for ln in lines if re.search(r"(Error|Exception)(: |$)|Fatal|Check failed", ln)]
            tail = (hits or lines or ["empty log"])[-1][:300]
        print(f"[alf] {job.label}: no output; last error line: {tail}", flush=True)
        failed.append(job.label)
    return failed


def write_summary(path: str | os.PathLike, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[alf] summary {path}", flush=True)


def child_path() -> str:
    """PYTHONPATH for a child process: this package and, when Marin is a plain checkout, Marin too.

    A child inherits the environment, not the parent's `sys.path`, so neither an `alf` that was never installed nor
    a Marin that `bootstrap()` only put on the parent's path would be importable without this.
    """
    from alf.marin.bootstrap import marin_repo

    parts = [str(Path(__file__).resolve().parents[1])]
    repo = marin_repo()
    if repo:
        parts.append(repo)
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(parts)


def entry(module: str) -> list[str]:
    """The command that runs one of this package's module entry points in this interpreter."""
    return [sys.executable, "-m", module]
