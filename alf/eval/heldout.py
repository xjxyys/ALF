#!/usr/bin/env python3
"""Held-out loss / bits per byte of a Marin MoE checkpoint at chosen Jacobi pass counts (JAX).

This is the paper's main held-out number (Section 4.2). The training run's own evaluator
(experiments.grug.moe_hero_ep.train.build_tagged_evaluator over the Dolma validation sources of
alf.marin.recipe.dolma_data_config, compute_bpb) scores the `params` of checkpoints/step-N once per pass count; the
pass count is set through the evaluator's model_transform (AttnResTransformer.eval_passes). The training log's final
eval used the run's --eval-passes, so that row reproduces it (a check of the setup); the other rows give pass 1 (the
backbone with the feedback path off) and the train-inference gap (pass 2, the regime training optimizes, against the
converged passes).

Arguments of one model; scripts/eval_heldout.py passes them and puts the Marin checkout on sys.path:

  --run-dir OUTPUT/RUN [--step N] --variant plain|attnres|feedback|src|stdres [--feedback-share pool|inject]
      [--feedback-query shared] [--feedback-rank 0] [--feedback-inject-rank 64] --passes 1,2,3,4 --data-root DIR
      --eval-root DIR [--eval-batch 64] [--shape d768] --out FILE.json

--shape names the model size: d768 = MoE-1.6B, d896 = MoE-2.6B, d1024 = MoE-4.0B. Uses every visible GPU as the data
axis of the mesh.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np


def complete_steps(run_dir: str) -> list[int]:
    root = Path(run_dir) / "checkpoints"
    return sorted(int(p.name.split("-", 1)[1]) for p in root.glob("step-*") if (p / "metadata.json").exists())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--variant", choices=("plain", "attnres", "feedback", "src", "stdres"), required=True)
    ap.add_argument("--feedback-share", default="pool", choices=("pool", "joint", "inject", "fbt"))
    ap.add_argument("--src-block", type=int, default=-1)
    ap.add_argument("--src-rank", type=int, default=64)
    ap.add_argument("--feedback-query", default="shared")
    ap.add_argument("--feedback-rank", type=int, default=0)
    ap.add_argument("--feedback-inject-rank", type=int, default=0)
    ap.add_argument("--feedback-proj", default="none", choices=("none", "source", "slot"))
    ap.add_argument("--feedback-proj-rank", type=int, default=0)
    ap.add_argument("--feedback-depth", default="whole", help="whole: deeper slots; top: last layer; "
                    "win<k>: nearest k slots, equal weights, no softmax (stdres pool/uniform)")
    ap.add_argument("--feedback-replaces-input", default="residual", choices=("residual", "input"))
    ap.add_argument("--feedback-source", default="outputs", choices=("outputs", "top"))
    ap.add_argument("--passes", default="1", help="comma-separated pass counts (feedback); plain / attnres use 1")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--max-eval-batches", type=int, default=None, help="per held-out source (quick checks only)")
    ap.add_argument("--shape", default="d768", help="model size: d768 = MoE-1.6B, d896 = MoE-2.6B, d1024 = MoE-4.0B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    passes = [int(p) for p in args.passes.split(",")] if args.variant in ("feedback", "src", "stdres") else [1]

    import jax
    import jmp
    from jax.sharding import AxisType, Mesh, set_mesh

    from alf import model as am
    import experiments.grug.moe_hero_ep.model as gm
    from experiments.grug.moe_hero_ep import train as grug_train
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.compat import register_gpu_kinds_for_pallas
    from levanter.checkpoint import load_checkpoint
    from levanter.tokenizers import load_tokenizer
    from alf.marin.recipe import dolma_data_config, model_config

    from alf import rungs as mid_rung

    mid_rung.register_mid_shapes()
    register_gpu_kinds_for_pallas()
    steps = complete_steps(args.run_dir)
    step = args.step if args.step is not None else (steps[-1] if steps else None)
    if step is None or step not in steps:
        raise SystemExit(f"no complete checkpoint{'' if args.step is None else f' step-{args.step}'} under {args.run_dir}")
    cfg = model_config(shape=args.shape)
    data_config, _ = dolma_data_config(args.data_root, seq_len=cfg.max_seq_len, eval_root=args.eval_root)
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices).reshape(1, len(devices), 1, 1, 1), ("replica_dcn", "data", "context", "expert", "model"),
                axis_types=(AxisType.Explicit,) * 5)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    eval_cfg = grug_train.GrugEvalConfig(eval_batch_size=args.eval_batch, steps_per_eval=1, eval_current=True,
                                         eval_ema=False, compute_bpb=True,
                                         max_eval_batches=args.max_eval_batches)
    bos = int(load_tokenizer(os.environ.get("MARIN_TOKENIZER_DIR", "marin-community/marin-tokenizer")).bos_token_id)
    fbt_input_norm = am.run_fbt_input_norm(args.run_dir, "fbt" if args.feedback_replaces_input == "input" else
                                           args.feedback_share if args.variant in ("feedback", "stdres") else "none")
    t0 = time.time()
    rows = {}
    with set_mesh(mesh):
        key = jax.random.key(0)
        base = gm.Transformer.init(cfg, key=key)
        if args.variant == "plain":
            model = base
        elif args.variant == "src":
            model = am.SrcTransformer.from_base(base, key=key, src_block=args.src_block, src_rank=args.src_rank,
                                                passes=max(passes), eval_passes=max(passes), jitter=0.0, bos_id=bos)
        elif args.variant == "stdres":
            model = am.StdResTransformer.from_base(base, key=key, share=args.feedback_share, query=args.feedback_query,
                                                   proj=args.feedback_proj, proj_rank=args.feedback_proj_rank,
                                                   passes=max(passes), eval_passes=max(passes), jitter=0.0, bos_id=bos,
                                                   fbt_input_norm=fbt_input_norm, depth=args.feedback_depth,
                                                   inject=args.feedback_replaces_input, source=args.feedback_source,
                                                   embed_inject=am.read_embed_inject(args.run_dir))
        else:
            model = am.AttnResTransformer.from_base(
                base, key=key, share=args.feedback_share, query=args.feedback_query, rank=args.feedback_rank,
                passes=max(passes), eval_passes=max(passes),
                jitter=0.0, bos_id=bos, inject_rank=args.feedback_inject_rank, proj=args.feedback_proj,
                proj_rank=args.feedback_proj_rank, depth=args.feedback_depth, fbt_input_norm=fbt_input_norm,
                embed_inject=am.read_embed_inject(args.run_dir))
        ckpt = f"{args.run_dir}/checkpoints/step-{step}"
        model = load_checkpoint({"params": model}, ckpt, axis_mapping=None, mesh=mesh, allow_partial=False)["params"]
        print(f"[heldout] loaded {ckpt} variant={args.variant} passes={passes} fbt_input_norm={fbt_input_norm} "
              f"devices={len(devices)} ({time.time() - t0:.0f}s)", flush=True)
        for p in passes:
            transform = (None if args.variant not in ("feedback", "src", "stdres")
                         else (lambda m, p=p: dataclasses.replace(m, eval_passes=p)))
            evaluator = grug_train.build_tagged_evaluator(data_config=data_config, max_seq_len=cfg.max_seq_len, mesh=mesh,
                                                          eval_cfg=eval_cfg, mp=mp, model_transform=transform)
            t1 = time.time()
            res = evaluator.evaluate(model)
            rows[str(p)] = {"macro_loss": float(res.macro_avg_loss), "macro_bpb": float(res.macro_bpb),
                            "loss": float(res.micro_avg_loss), "bpb": float(res.micro_bpb) if res.micro_bpb is not None else None,
                            "tag_macro_loss": {k: float(v) for k, v in res.tag_macro_losses.items()},
                            "seconds": time.time() - t1}
            print(f"[heldout] passes={p}: macro_loss {res.macro_avg_loss:.5f} macro_bpb {res.macro_bpb:.5f} "
                  f"loss {res.micro_avg_loss:.5f} ({time.time() - t1:.0f}s)", flush=True)
    out = {"run_dir": args.run_dir, "step": step, "variant": args.variant, "feedback_share": args.feedback_share,
           "feedback_inject_rank": args.feedback_inject_rank, "feedback_proj": args.feedback_proj,
           "feedback_proj_rank": args.feedback_proj_rank, "feedback_depth": args.feedback_depth,
           "feedback_replaces_input": args.feedback_replaces_input, "feedback_source": args.feedback_source,
           "src_block": args.src_block, "src_rank": args.src_rank,
           "fbt_input_norm": fbt_input_norm, "eval_root": args.eval_root, "eval_batch": args.eval_batch,
           "max_eval_batches": args.max_eval_batches,
           "by_passes": rows, "seconds": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[heldout] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
