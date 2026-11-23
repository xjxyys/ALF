"""Shared command-line plumbing for the entry points in `scripts/`.

Each entry point is thin: read `.env`, put the Marin checkout on `sys.path`, turn `<run>:<kind>` into the arguments
that rebuild each model, and start one process per GPU. The work itself is in `alf.eval.*`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from alf import kinds
from alf.launch import Job, entry, gpu_count, pick_step

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: str | os.PathLike | None = None) -> None:
    """Read `.env` (KEY=VALUE, `#` comments). Values already in the environment win, as in a shell."""
    env = Path(path) if path else REPO_ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def data_root() -> str:
    return os.environ.get("DATA_ROOT") or str(REPO_ROOT / "data")


def output_root() -> str:
    return os.environ.get("OUTPUT_ROOT") or str(REPO_ROOT / "output")


def run_root() -> str:
    return os.environ.get("RUN_ROOT") or os.path.join(output_root(), "runs")


def parser(description: str) -> argparse.ArgumentParser:
    """An evaluation entry point's arguments: which models, which rung, where the data and the outputs are."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="<run>:<kind>[,<run>:<kind>...]; --list-kinds prints the kinds")
    ap.add_argument("--list-kinds", action="store_true", help="print the models of the paper and exit")
    ap.add_argument("--shape", default="d768", choices=("d768", "d896", "d1024"))
    ap.add_argument("--step", type=int, default=None, help="default: each run's newest complete checkpoint")
    ap.add_argument("--run-root", default=None, help=f"training run directories (default: {run_root()})")
    ap.add_argument("--out-dir", default=None, help="where the per-model JSON goes (default: <OUTPUT_ROOT>/<task>)")
    ap.add_argument("--gpus", type=int, default=None, help="default: every visible GPU")
    ap.add_argument("--inject-rank", type=int, default=64, help="must match the run's training arguments")
    ap.add_argument("--proj-rank", type=int, default=64, help="must match the run's training arguments")
    ap.add_argument("--src-block", type=int, default=None, help="the single-source route's source block")
    ap.add_argument("--src-rank", type=int, default=64)
    return ap


def resolved(args, task: str) -> tuple[list[tuple[str, str]], Path, Path]:
    """The models, the output directory and the log directory; exits on `--list-kinds` or a missing `--models`."""
    if args.list_kinds:
        print(kinds.table())
        raise SystemExit(0)
    if not args.models:
        raise SystemExit("--models is required (or --list-kinds)")
    models = kinds.parse_models(args.models)
    out_dir = Path(args.out_dir or os.path.join(output_root(), task))
    return models, out_dir, out_dir / "logs"


def model_args(args, kind: str) -> list[str]:
    return kinds.model_args(kind, inject_rank=args.inject_rank, proj_rank=args.proj_rank,
                            src_block=args.src_block, src_rank=args.src_rank)


def passes_for(kind: str, passes: str) -> str:
    """A backbone without feedback is always one pass, whatever `--passes` asks for."""
    return passes if kinds.resolve(kind).passes else "1"


def one_per_gpu(args, task: str, module: str, extra, *, out_suffix: str = ".json") -> int:
    """Run one model per GPU: `extra(kind)` gives the task's own arguments.

    Returns a process exit status: 0 when every model produced its output.
    """
    from alf.launch import run_jobs, write_summary

    models, out_dir, log_dir = resolved(args, task)
    root = args.run_root or run_root()
    have = args.gpus or gpu_count()
    if have < len(models):
        raise SystemExit(f"{len(models)} models need as many GPUs, found {have}")

    jobs, steps = [], {}
    for i, (run, kind) in enumerate(models):
        run_dir = os.path.join(root, run)
        step = pick_step(run_dir, args.step)
        steps[run] = step
        name = f"{run}_s{step}{out_suffix}"
        out = out_dir / name
        jobs.append(Job(label=f"{run}:{kind}", gpu=i, out=out, log=log_dir / f"{run}_s{step}.log",
                        argv=entry(module) + ["--run-dir", run_dir, "--step", str(step),
                                              *model_args(args, kind), *extra(kind),
                                              "--shape", args.shape, "--out", str(out)]))

    failed = run_jobs(jobs)
    write_summary(out_dir / "summary.json", {
        "task": task, "shape": args.shape,
        "models": {run: {"kind": kind, "step": steps[run]} for run, kind in models},
        "failed": failed,
    })
    if failed:
        print(f"[alf] {len(failed)} model(s) produced no output: {', '.join(failed)}")
        return 1
    return 0


def sharded(args, task: str, module: str, extra) -> int:
    """Run the models in turn, each sharded over the GPUs, then merge each model's shards."""
    import subprocess

    from alf.launch import run_jobs, write_summary

    models, out_dir, log_dir = resolved(args, task)
    root = args.run_root or run_root()
    n = args.gpus or gpu_count()
    if n < 1:
        raise SystemExit("no GPUs found")

    failed, steps = [], {}
    for run, kind in models:
        run_dir = os.path.join(root, run)
        step = pick_step(run_dir, args.step)
        steps[run] = step
        model_out = out_dir / f"{run}_s{step}"
        common = entry(module) + ["--run-dir", run_dir, "--step", str(step), *model_args(args, kind),
                                  *extra(kind), "--shape", args.shape, "--out", str(model_out)]
        jobs = [Job(label=f"{run}:{kind} shard {i}", gpu=i, out=model_out / f"shard_{i}.json",
                    log=log_dir / f"{run}_s{step}_shard_{i}.log",
                    argv=common + ["--shard", str(i), "--num-shards", str(n)]) for i in range(n)]
        missing = run_jobs(jobs)
        if missing:
            failed.append(run)
            continue
        merge = common + ["--merge", "--num-shards", str(n)]
        print(f"[alf] {run}: merging {n} shards", flush=True)
        if subprocess.run(merge).returncode != 0:
            failed.append(run)

    write_summary(out_dir / "summary.json", {
        "task": task, "shape": args.shape,
        "models": {run: {"kind": kind, "step": steps.get(run)} for run, kind in models},
        "failed": failed,
    })
    if failed:
        print(f"[alf] {len(failed)} model(s) failed: {', '.join(failed)}")
        return 1
    return 0
