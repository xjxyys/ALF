#!/usr/bin/env python3
"""The pre-registered readout of the downstream suite: per-task loss and accuracy, paired differences, noise floor.

Reads the per-item records that `alf.eval.downstream` writes (one directory per model, holding `bench.json` and
`shard_*/<task>.jsonl`) and reports:

* **Loss.** The token-normalized negative log-likelihood of the correct answer, macro-averaged over the 26 tasks of
  the suite, and over the five axes of its 20-task core (the 6 tasks carried over from CORE sit on their own axis and
  are reported separately). No task is screened out of the loss.
* **Accuracy.** Averaged only over tasks where accuracy carries signal, which is decided on one reference model
  alone, never on the models being compared: its final accuracy must stand at least `--screen-se` binomial standard
  errors above both its own accuracy at an earlier checkpoint (`--traj`) and the chance level of the task (0 for
  language modelling, 1 / number of options otherwise).
* **Paired differences.** `--main` minus each comparison model, per task and per axis, with a 95% interval from a
  bootstrap over items within each task, then averaged over tasks.
* **A noise floor.** The same paired difference between two seeds of the same model (`--null`), which bounds what a
  difference of this size can mean.

A model given through `--average` is the per-item mean over several seeds, taken before any pairing.

  python -m alf.bench.readout \\
      --model standard=OUT/bench/moe_1p6b_plain_s11420 \\
      --model attnres=OUT/bench/moe_1p6b_attnres_s11420 \\
      --model fbt=OUT/bench/moe_1p6b_fbt_s11420 \\
      --model alf-s0=OUT/bench/moe_1p6b_alf_s0_s11420 --model alf-s1=OUT/bench/moe_1p6b_alf_s1_s11420 \\
      --average alf=alf-s0,alf-s1 --null alf-s0,alf-s1 \\
      --reference attnres --traj attnres=OUT/bench/moe_1p6b_attnres_s2000 \\
      --main alf --out OUT/bench/readout.json

Every model must have been evaluated on the same items of the same tasks; the readout refuses to go on otherwise,
since the differences are paired per item.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np


def load_model(path: Path) -> dict | None:
    bench = path / "bench.json"
    if not bench.exists():
        return None
    meta = json.load(open(bench))
    tasks = {}
    for name in meta["tasks"]:
        rows = []
        for f in sorted(glob.glob(str(path / "shard_*" / f"{name}.jsonl"))):
            with open(f) as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
        rows = sorted((r for r in rows if "skip" not in r), key=lambda r: r["i"])
        tasks[name] = {"i": np.asarray([r["i"] for r in rows]),
                       "nll": np.asarray([r["nll"] for r in rows], np.float64).T,
                       "ok": np.asarray([r["ok"] for r in rows], np.float64).T,
                       "t": np.asarray([r["t"] for r in rows], np.float64),
                       "choices": len(rows[0]["cand"][0]) if "cand" in rows[0] else 0}
    return {"meta": meta, "tasks": tasks}


def item_scores(model: dict, name: str, which: str = "last") -> tuple[np.ndarray, np.ndarray]:
    task = model["tasks"][name]
    k = -1 if which == "last" else 0
    return task["nll"][k], task["ok"][k]


def averaged(models: dict, labels: tuple, name: str, which: str = "last"):
    pairs = [item_scores(models[lab], name, which) for lab in labels]
    return np.mean([p[0] for p in pairs], 0), np.mean([p[1] for p in pairs], 0)


def scores(models: dict, groups: dict, label: str, name: str, which: str = "last"):
    if label in groups:
        return averaged(models, groups[label], name, which)
    return item_scores(models[label], name, which)


def task_loss(nll, t):
    return float(nll.sum() / t.sum())


def paired_delta(models, groups, a, b, names, t_of, boot, rng, which_a="last", which_b="last"):
    """Task-macro change in loss and in accuracy (a - b) over `names`, with a per-task item bootstrap."""
    point_l, point_a = [], []
    rep_l = np.zeros(boot)
    rep_a = np.zeros(boot)
    for name in names:
        na, oa = scores(models, groups, a, name, which_a)
        nb, ob = scores(models, groups, b, name, which_b)
        t = t_of[name]
        dn, do = na - nb, oa - ob
        point_l.append(dn.sum() / t.sum())
        point_a.append(do.mean())
        n = len(t)
        idx = rng.integers(0, n, size=(boot, n))
        rep_l += dn[idx].sum(1) / t[idx].sum(1)
        rep_a += do[idx].mean(1)
    k = len(names)
    rep_l /= k
    rep_a /= k

    def ci(r):
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

    return {"dnll": float(np.mean(point_l)), "dnll_ci": ci(rep_l), "dacc": float(np.mean(point_a)),
            "dacc_ci": ci(rep_a)}


def _pairs(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--{what} takes LABEL=VALUE, got {item!r}")
        label, value = item.split("=", 1)
        out[label.strip()] = value.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", metavar="LABEL=DIR",
                    help="a model's bench output directory; repeat for every model")
    ap.add_argument("--average", action="append", metavar="LABEL=L1,L2",
                    help="a label standing for the per-item mean of these models (seeds of one model)")
    ap.add_argument("--null", action="append", metavar="L1,L2",
                    help="two seeds of the same model, reported as the noise floor; repeatable")
    ap.add_argument("--reference", default=None,
                    help="the model the accuracy screen and the item-identity check use (default: the first --model)")
    ap.add_argument("--traj", action="append", metavar="LABEL=DIR",
                    help="the reference model at an earlier checkpoint, for the accuracy screen")
    ap.add_argument("--main", default=None, help="the model whose paired differences are reported")
    ap.add_argument("--compare", default=None,
                    help="comma-separated models to compare --main against (default: every other model)")
    ap.add_argument("--screen-se", type=float, default=5.0, help="standard errors the accuracy screen asks for")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wanted = _pairs(args.model, "model")
    if not wanted:
        raise SystemExit("at least one --model LABEL=DIR is required")
    groups = {label: tuple(p.strip() for p in parts.split(",")) for label, parts in _pairs(args.average, "average").items()}

    models = {}
    for label, path in wanted.items():
        m = load_model(Path(path))
        if m is None:
            raise SystemExit(f"{label}: no bench.json under {path}")
        models[label] = m

    traj = {}
    for label, path in _pairs(args.traj, "traj").items():
        m = load_model(Path(path))
        if m is None:
            raise SystemExit(f"--traj {label}: no bench.json under {path}")
        traj[label] = m

    ref_label = args.reference or next(iter(wanted))
    if ref_label not in models:
        raise SystemExit(f"--reference {ref_label} is not one of the models")
    ref = models[ref_label]

    suite_tasks = list(ref["meta"]["tasks"])
    axis = {n: ref["meta"]["tasks"][n]["axis"] for n in suite_tasks}
    core = [n for n in suite_tasks if axis[n] != "core_extra"]
    axes = list(dict.fromkeys(axis[n] for n in core))
    t_of = {n: ref["tasks"][n]["t"] for n in suite_tasks}

    # The differences are paired item by item, so every model has to have scored the same items.
    for label, m in list(models.items()) + list(traj.items()):
        for n in suite_tasks:
            if not (np.array_equal(m["tasks"][n]["i"], ref["tasks"][n]["i"])
                    and np.array_equal(m["tasks"][n]["t"], t_of[n])):
                raise SystemExit(f"{label} {n}: items or answer token counts differ from {ref_label}")

    # The accuracy screen, decided on the reference model alone.
    screen, early = {}, next(iter(traj.values()), None)
    base = ref["tasks"]
    for n in suite_tasks:
        acc = float(base[n]["ok"][-1].mean())
        se = math.sqrt(max(acc * (1 - acc), 1e-12) / len(base[n]["t"]))
        chance = 1.0 / base[n]["choices"] if base[n]["choices"] else 0.0
        a0 = float(early["tasks"][n]["ok"][-1].mean()) if early else None
        screen[n] = {"acc_final": acc, "acc_early": a0, "se": se, "chance": chance,
                     "keep": a0 is not None and acc - a0 >= args.screen_se * se
                     and acc - chance >= args.screen_se * se}
    kept = [n for n in suite_tasks if screen[n]["keep"]]

    present = [lab for lab in wanted if lab not in {p for parts in groups.values() for p in parts}]
    present += [l for l, parts in groups.items() if all(p in models for p in parts)]

    rng = np.random.default_rng(0)
    summary = {"reference": ref_label, "screen_se": args.screen_se, "variants": {}, "deltas": {}, "nulls": {},
               "screen": screen, "kept": kept, "per_task": {}, "axes": {}}
    for label in present:
        row = {}
        for which in ("last", "first"):
            per = {n: (task_loss(scores(models, groups, label, n, which)[0], t_of[n]),
                       float(scores(models, groups, label, n, which)[1].mean())) for n in suite_tasks}
            ax_l = {a: float(np.mean([per[n][0] for n in core if axis[n] == a])) for a in axes}
            ax_a = {a: float(np.mean([per[n][1] for n in core if axis[n] == a])) for a in axes}
            row[which] = {"task_macro_nll": float(np.mean([per[n][0] for n in suite_tasks])),
                          "axis_macro_nll": float(np.mean(list(ax_l.values()))),
                          "axis_macro_acc": float(np.mean(list(ax_a.values()))),
                          "screened_acc": float(np.mean([per[n][1] for n in kept])) if kept else None,
                          "axes_nll": ax_l, "axes_acc": ax_a, "tasks": per}
        summary["variants"][label] = row

    main_label = args.main
    if main_label and main_label not in present:
        raise SystemExit(f"--main {main_label!r} is not one of the models: {', '.join(present)}")
    if main_label:
        others = ([c.strip() for c in args.compare.split(",")] if args.compare
                  else [lab for lab in present if lab != main_label])
        for other in others:
            if other not in present:
                raise SystemExit(f"--compare {other} is not one of the models")
            d_axes = {a: paired_delta(models, groups, main_label, other, [n for n in core if axis[n] == a], t_of,
                                      args.boot, rng) for a in axes}
            extra = [n for n in suite_tasks if axis[n] == "core_extra"]
            if extra:
                d_axes["core_extra"] = paired_delta(models, groups, main_label, other, extra, t_of, args.boot, rng)
            summary["deltas"][f"{main_label} - {other}"] = {
                "all_tasks": paired_delta(models, groups, main_label, other, suite_tasks, t_of, args.boot, rng),
                "kept_tasks": paired_delta(models, groups, main_label, other, kept, t_of, args.boot, rng) if kept else None,
                "axes": d_axes}

    for item in args.null or []:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"--null takes two labels, got {item!r}")
        a, b = parts
        missing = [x for x in (a, b) if x not in models]
        if missing:
            raise SystemExit(f"--null {item!r}: {', '.join(missing)} is not a --model label (an --average label does "
                             f"not work here, the noise floor is between two raw models): {', '.join(models)}")
        summary["nulls"][f"{a} - {b}"] = {
            "all_tasks": paired_delta(models, groups, a, b, suite_tasks, t_of, args.boot, rng),
            "kept_tasks": paired_delta(models, groups, a, b, kept, t_of, args.boot, rng) if kept else None}

    for n in suite_tasks:
        summary["per_task"][n] = {lab: {"nll": summary["variants"][lab]["last"]["tasks"][n][0],
                                        "acc": summary["variants"][lab]["last"]["tasks"][n][1]} for lab in present}
    if traj:
        summary["trajectory"] = {label: {n: {"nll": task_loss(m["tasks"][n]["nll"][-1], t_of[n]),
                                             "acc": float(m["tasks"][n]["ok"][-1].mean())} for n in suite_tasks}
                                 for label, m in traj.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)

    print("## Main table (a model with feedback is read at its last pass; averaged labels are per-item seed means)\n")
    print(f"| model | {len(suite_tasks)}-task NLL (lower) | axis-macro NLL (lower) | axis-macro accuracy (higher) "
          f"| screened accuracy, {len(kept)} tasks (higher) |")
    print("|---|---|---|---|---|")
    for label in present:
        r = summary["variants"][label]["last"]
        sa = f"{r['screened_acc']:.4f}" if r["screened_acc"] is not None else "n/a"
        print(f"| {label} | {r['task_macro_nll']:.4f} | {r['axis_macro_nll']:.4f} | {r['axis_macro_acc']:.4f} | {sa} |")

    def fmt(d, k):
        return f"{d[k]:+.4f} [{d[k + '_ci'][0]:+.4f}, {d[k + '_ci'][1]:+.4f}]"

    if summary["deltas"] or summary["nulls"]:
        print("\n## Differences (95% interval, items resampled within each task)\n")
        print(f"| comparison | {len(suite_tasks)}-task change in NLL | screened change in accuracy |")
        print("|---|---|---|")
        for key, d in list(summary["deltas"].items()) + list(summary["nulls"].items()):
            kd = fmt(d["kept_tasks"], "dacc") if d.get("kept_tasks") else "n/a"
            print(f"| {key} | {fmt(d['all_tasks'], 'dnll')} | {kd} |")
    if summary["deltas"]:
        print("\n## Per axis: change in NLL / in accuracy\n")
        for key, d in summary["deltas"].items():
            print(f"- {key}: " + "; ".join(f"{a} {v['dnll']:+.4f} / {v['dacc']:+.4f}" for a, v in d["axes"].items()))

    print(f"\n## Accuracy screen on {ref_label} (final minus the earlier checkpoint, and minus chance, "
          f"each at least {args.screen_se:g} standard errors)\n")
    print("kept: " + (", ".join(kept) or "none"))
    dropped = [n for n in suite_tasks if not screen[n]["keep"]]
    print("dropped: " + (", ".join(
        f"{n} ({'n/a' if screen[n]['acc_early'] is None else round(screen[n]['acc_early'], 3)}"
        f" -> {screen[n]['acc_final']:.3f})" for n in dropped) or "none"))

    print("\n## Per task (NLL / accuracy)\n")
    print("| task | axis | " + " | ".join(present) + " |")
    print("|---|---|" + "---|" * len(present))
    for n in suite_tasks:
        print(f"| {n} | {axis[n]} | " + " | ".join(
            f"{summary['per_task'][n][lab]['nll']:.3f} / {summary['per_task'][n][lab]['acc']:.3f}"
            for lab in present) + " |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
