#!/usr/bin/env python3
"""A CPU self-test of the downstream bench, with no model: a real bundle, the Marin tokenizer and a fake forward
whose loss depends only on the tokens of its own row. It checks the packing, the per-question records, the skipped
multiple-choice questions and the merge of `alf.eval.downstream`, and then the whole chain down to `alf.bench.readout`.

  python tests/test_bench.py [--bundle B]      (the same environment as alf.eval.downstream)

Rows never interact in a packed batch, so a row's loss under the fake forward can be predicted in closed form; the
test asserts the recorded NLL and the verdict of every question against that prediction, for every pass. The readout
then runs over four fake models, two of which are seeds of one model: the averaged model scores lower by
construction, so its paired difference against the others must be negative, and the two seeds must differ by less.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

# Read .env and put the Marin checkout on sys.path, so this file can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alf.cli import load_env  # noqa: E402
from alf.marin.bootstrap import bootstrap  # noqa: E402

load_env()
bootstrap()
from argparse import Namespace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from alf.eval import downstream as be  # noqa: E402

PASSES = 3
STEP = 11420
# label -> (scale of the fake loss, passes recorded). The two seeds of `alf` score lowest; `attnres` has no feedback
# and so only one pass. `attnres-early` stands in for an earlier checkpoint of the reference model.
MODELS = {"attnres": (1.0, 1), "fbt": (0.99, PASSES), "alf-s0": (0.98, PASSES), "alf-s1": (0.981, PASSES)}
TRAJ = {"attnres-early": (1.2, 1)}


def fake_score(scale: float = 1.0):
    def score_fn(padded, prompt=None):  # the prompt lengths only matter to the decoding modes
        tok = padded.astype(np.int64)
        nxt = np.roll(tok, -1, axis=1)
        base = ((nxt * 31 + tok * 7) % 97) / 97.0 + 0.01
        losses = np.stack([base * scale * (1 - 0.01 * k) for k in range(PASSES)])
        return losses, np.stack([nxt] * PASSES)
    return score_fn


def expected(tokenizer, indexed, index, task, delimiter, scale=1.0):
    built = be.build_example(tokenizer, indexed, index, task, delimiter, 4096)
    if built is None:
        return None
    tokens, starts, ends, gold = built
    out = []
    for k in range(PASSES):
        sums = []
        for r, seq in enumerate(tokens):
            s, e = starts[r], ends[r]
            row = sum(((seq[t + 1] * 31 + seq[t] * 7) % 97) / 97.0 + 0.01 for t in range(s - 1, e - 1))
            sums.append(row * scale * (1 - 0.01 * k))
        if task["task_type"] == "multiple_choice":
            means = [x / (ends[r] - starts[r]) for r, x in enumerate(sums)]
            ok = means.index(min(means)) == gold
        else:
            ok = True
        out.append((sums[gold], ok))
    return out


def write_model(out: Path, tasks: list[dict], records: dict, manifest: dict, *, tokenizer, bos: int, scale: float,
                passes: int, run_dir: Path, bundle: Path) -> None:
    """One model's bench output: two shards of per-question records, then the merge that reads them back."""
    for shard in range(2):
        sd = out / f"shard_{shard}"
        sd.mkdir(parents=True)
        for task in tasks:
            indexed = be.task_pool(records[task["name"]], 24)
            recs, _ = be.score_task(task, indexed, manifest["tasks"][task["name"]]["continuation_delimiter"],
                                    tokenizer=tokenizer, bos=bos, max_len=4096, passes=PASSES,
                                    score_fn=fake_score(scale), shard=shard, num_shards=2)
            for rec in recs.values():
                if "nll" in rec:
                    rec["nll"], rec["ok"] = rec["nll"][:passes], rec["ok"][:passes]
                    if "cand" in rec:
                        rec["cand"] = rec["cand"][:passes]
            with open(sd / f"{task['name']}.jsonl", "w") as stream:
                for index in sorted(recs):
                    stream.write(json.dumps(recs[index]) + "\n")
        json.dump({"passes": passes, "step": STEP, "variant": "x", "max_per_task": 24, "seconds": 1.0},
                  open(out / f"shard_{shard}.json", "w"))
    be.merge(Namespace(run_dir=str(run_dir), step=STEP, bundle=str(bundle), out=str(out), num_shards=2))
    assert (out / "bench.json").exists() and (out / "done.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=be.BUNDLE)
    args = ap.parse_args()
    from levanter.tokenizers import load_tokenizer

    from alf.eval.downstream import MarinBatchTokenizer

    suite, manifest, records = be.load_bundle(args.bundle)
    tokenizer = MarinBatchTokenizer(load_tokenizer(
        os.environ.get("MARIN_TOKENIZER_DIR", "marin-community/marin-tokenizer")))
    bos = tokenizer.get_bos_token_id()
    tasks = {t["name"]: t for t in suite["tasks"]}
    cases = [("triviaqa", 24, 0, 2), ("piqa", -1, 2, 8), ("drop", 24, 1, 2), ("hellaswag", 24, 0, 2)]
    for name, limit, shard, nshards in cases:
        task = tasks[name]
        delim = manifest["tasks"][name]["continuation_delimiter"]
        indexed = be.task_pool(records[name], limit)
        recs, check = be.score_task(task, indexed, delim, tokenizer=tokenizer, bos=bos, max_len=4096,
                                    passes=PASSES, score_fn=fake_score(), shard=shard, num_shards=nshards,
                                    selfcheck=True)
        assert check == 0.0, (name, check)
        assert sorted(recs) == list(range(shard, len(indexed), nshards)), name
        skipped = 0
        for index, rec in recs.items():
            exp = expected(tokenizer, indexed, index, task, delim)
            if exp is None:
                assert rec.get("skip") == "choice_is_prefix", (name, index)
                skipped += 1
                continue
            for k in range(PASSES):
                assert math.isclose(rec["nll"][k], exp[k][0], rel_tol=1e-6, abs_tol=1e-5), \
                    (name, index, k, rec["nll"][k], exp[k][0])
                assert rec["ok"][k] == exp[k][1], (name, index, k)
        print(f"[ok] {name}: shard {shard}/{nshards} {len(recs)} items, skipped {skipped}, packed == single",
              flush=True)
        if name == "piqa":
            assert recs[194].get("skip") == "choice_is_prefix", "piqa pool item 194 should be skipped"

    tmp = Path(tempfile.mkdtemp(prefix="bench_test_"))
    try:
        mini = dict(suite, tasks=[tasks["triviaqa"], tasks["hellaswag"]])
        bundle = tmp / "bundle"
        bundle.mkdir()
        (bundle / "suite.json").write_text(json.dumps(mini))
        # The merge resolves --step against the run's checkpoints; nothing is read out of them here.
        run_dir = tmp / "run"
        (run_dir / "checkpoints" / f"step-{STEP}").mkdir(parents=True)
        (run_dir / "checkpoints" / f"step-{STEP}" / "metadata.json").write_text("{}")
        for label, (scale, passes) in list(MODELS.items()) + list(TRAJ.items()):
            write_model(tmp / "exps" / label, mini["tasks"], records, manifest, tokenizer=tokenizer, bos=bos,
                        scale=scale, passes=passes, run_dir=run_dir, bundle=bundle)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                                             env.get("PYTHONPATH", "")])
        cmd = [sys.executable, "-m", "alf.bench.readout", "--out", str(tmp / "readout.json"), "--boot", "200",
               "--average", "alf=alf-s0,alf-s1", "--null", "alf-s0,alf-s1", "--reference", "attnres",
               "--traj", f"attnres-early={tmp / 'exps' / 'attnres-early'}", "--main", "alf",
               "--compare", "attnres,fbt"]
        for label in MODELS:
            cmd += ["--model", f"{label}={tmp / 'exps' / label}"]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(res.stdout[-2500:])
        if res.returncode:
            print(res.stderr[-3000:])
            raise SystemExit("readout failed")
        summary = json.load(open(tmp / "readout.json"))
        delta = summary["deltas"]["alf - attnres"]["all_tasks"]
        null = summary["nulls"]["alf-s0 - alf-s1"]["all_tasks"]
        assert delta["dnll"] < 0, delta
        assert summary["deltas"]["alf - fbt"]["all_tasks"]["dnll"] < 0, summary["deltas"]["alf - fbt"]
        assert abs(null["dnll"]) < abs(delta["dnll"]), (null, delta)
        print("[ok] merge + readout on fake models")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL OK")


if __name__ == "__main__":
    main()
