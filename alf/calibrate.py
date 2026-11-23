#!/usr/bin/env python3
"""Entry biases of a one-softmax (--feedback-share joint) feedback branch, calibrated at the branch point (JAX).

With the previous position's slots at zero weight the second pass equals the first, so at every mixing point the score
gap LSE(previous slots) - LSE(same-token slots), taken over the positions that have feedback, fixes the entry bias b_j
that gives the previous slots `target` of the softmax weight on average (``alf.model.calibrate_joint_bias``; FBT,
arXiv:2608.08888, calibrates the same way on the first two-pass batch). Here the model is the base checkpoint at the
branch step (restored into the joint template; the source maps start at the identity, so they do not move the gap) and
the batch is the branch step's training batch (same data config, data seed and batch schedule as ``alf.trainer``; its
first --rows rows). Writes {"entry_bias": [2L floats], ...}; ``alf.trainer --feedback-entry-bias FILE`` then starts
the branch from them.

  python scripts/calibrate.py --init-from DIR --init-step N --shape d768 --data-root ROOT --data-seed 0 \
      [--feedback-inject-rank 64] [--feedback-proj source --feedback-proj-rank 64] [--rows 128] --out FILE.json

One process, every visible GPU (the training mesh of a one-node fsdp run).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def joint_share(am, model, tokens, mask):
    """Mean weight of the previous slots over mixing points 0..2L-1 at feedback positions, and per mixing point."""
    import jax
    import jax.numpy as jnp

    cfg = model.config
    n_mix, eps_d = 2 * cfg.num_layers, cfg.hidden_dim * am.ATTN_RES_EPS
    ctx = model._context(tokens, mask)
    x0 = model._embed(tokens)
    _, slots, _ = model._trunk(x0, ctx, None)
    q = model.attnres_query.weight.astype(x0.dtype)
    scores = [am.slot_scores(s, q, eps_d) for s in [x0, *slots]]
    _, valid, ps = model.feedback_inputs(slots, ctx, train=False)
    validf = valid.astype(jnp.float32)
    shares = []
    for j in range(n_mix):
        same = jnp.stack([sc[..., j] for sc in scores[: j + 1]], -1)
        prev = ps[:, :, j:, j] + model.frontier_entry_bias.weight[j]
        w = jax.nn.softmax(jnp.concatenate([same, prev], -1), -1)
        shares.append(jnp.sum(jnp.sum(w[..., j + 1:], -1) * validf) / jnp.sum(validf))
    return jnp.stack(shares)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-from", required=True, help="base run directory")
    ap.add_argument("--init-step", type=int, required=True)
    ap.add_argument("--shape", choices=("d768", "d896", "d1024"), default="d768",
                    help="model size: d768 = MoE-1.6B, d896 = MoE-2.6B, d1024 = MoE-4.0B")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--flavor", choices=("fsdp", "ep8"), default="fsdp")
    ap.add_argument("--batch", type=int, default=None, help="default: the rung batch (as alf.trainer uses it)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--feedback-inject-rank", type=int, default=0)
    ap.add_argument("--feedback-proj", choices=("none", "source", "slot"), default="none")
    ap.add_argument("--feedback-proj-rank", type=int, default=0)
    ap.add_argument("--feedback-init-share", type=float, default=0.01, help="target mean share of the previous slots")
    ap.add_argument("--rows", type=int, default=128, help="rows of the branch step's batch used for the calibration")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import equinox as eqx
    import jax
    import jmp
    from jax.sharding import NamedSharding, set_mesh
    from jax.sharding import PartitionSpec as P
    from levanter.checkpoint import load_checkpoint
    from levanter.data.loader import DataLoader
    from levanter.grug.sharding import compact_grug_mesh
    from levanter.schedule import BatchSchedule
    from levanter.tokenizers import load_tokenizer

    from alf import model as am
    import experiments.grug.moe_hero_ep.model as gm
    from experiments.grug.moe_hero_ep import train as grug_train
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from alf import rungs
    from alf.marin.compat import register_gpu_kinds_for_pallas
    from alf.marin.recipe import dolma_data_config, ladder_schedule, model_config

    register_gpu_kinds_for_pallas()
    rungs.register_mid_shapes()  # d896 is not one of Marin's own shapes
    t0 = time.time()
    ckpt = f"{args.init_from}/checkpoints/step-{args.init_step}"
    if not os.path.isfile(os.path.join(ckpt, "metadata.json")):
        raise SystemExit(f"no complete checkpoint {ckpt}")
    cfg = model_config(shape=args.shape, layers=args.layers, flavor=args.flavor)
    # The batch has to be the one training uses, or the biases are calibrated on a different batch. For Marin's own
    # rungs that is ladder_schedule; for the interpolated rung alf.rungs.schedule returns MID_BATCH, as the trainer
    # takes it (only the step count there depends on the active parameters, which calibration does not need).
    rung_batch = ladder_schedule(cfg, args.shape)[0] if args.shape in ("d768", "d1024") else rungs.MID_BATCH
    batch_size = args.batch or rung_batch
    if args.rows % jax.device_count() or args.rows > batch_size:
        raise SystemExit(f"--rows {args.rows} must divide over {jax.device_count()} devices and fit the batch {batch_size}")
    data_config, _ = dolma_data_config(args.data_root, seq_len=cfg.max_seq_len, eval_root=None)
    data_key, model_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    if args.data_seed is not None:
        data_key = jax.random.PRNGKey(args.data_seed)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[grug_train.MasterParamMode.DEVICE])
    tokenizer_dir = os.environ.get("MARIN_TOKENIZER_DIR", "marin-community/marin-tokenizer")
    bos = load_tokenizer(tokenizer_dir).bos_token_id
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=None, context_axis_size=1)
    with set_mesh(mesh):
        base = gm.Transformer.init(cfg, key=model_key)
        template = am.AttnResTransformer.from_base(
            base, key=model_key, share="joint", query="shared", passes=2, eval_passes=2, jitter=0.0,
            bos_id=-1 if bos is None else int(bos), inject_rank=args.feedback_inject_rank, proj=args.feedback_proj,
            proj_rank=args.feedback_proj_rank, init_share=args.feedback_init_share)
        model = load_checkpoint({"params": template}, ckpt, axis_mapping=None, mesh=mesh, allow_partial=True)["params"]
        print(f"[calib] restored {ckpt} into the joint template ({time.time() - t0:.0f}s)", flush=True)

        schedule = BatchSchedule(batch_size)
        dataset = grug_train.build_train_dataset(data_config, max_seq_len=cfg.max_seq_len, batch_schedule=schedule,
                                                 key=data_key)
        # build_train_loader's loader without its 512-batch prefetch (one batch is read here)
        loader = DataLoader(dataset, schedule.schedule, max_buffered_batches=0, mesh=mesh,
                            axis_resources={"__BATCH__": grug_train._BATCH_AXES}, fetch_batch_size=1,
                            batch_axis_name="__BATCH__", allow_nondivisible_batch_size=False)
        batch = next(iter(loader.iter_from_step(args.init_step)))

        def head(x):
            if not isinstance(x, jax.Array) or x.ndim == 0:
                return x
            host = np.asarray(jax.device_get(x))[: args.rows]
            return jax.device_put(host, NamedSharding(mesh, P(grug_train._BATCH_AXES, *([None] * (host.ndim - 1)))))

        small = jax.tree.map(head, batch)
        seg = small.attn_mask.segment_ids
        print(f"[calib] step {args.init_step} batch: {batch.tokens.shape} -> first {args.rows} rows; segment ids "
              f"{'present' if seg is not None else 'absent'} ({time.time() - t0:.0f}s)", flush=True)

        compute = mp.cast_to_compute(model)
        bias = eqx.filter_jit(lambda m, t, k: am.calibrate_joint_bias(m, t, k, target=args.feedback_init_share))(
            compute, small.tokens, small.attn_mask)
        bias = np.asarray(jax.device_get(bias), np.float64)
        calibrated = eqx.tree_at(lambda m: m.frontier_entry_bias.weight, model,
                                 jax.device_put(bias.astype(np.float32), NamedSharding(mesh, P(None))))
        shares = eqx.filter_jit(lambda m, t, k: joint_share(am, m, t, k))(mp.cast_to_compute(calibrated),
                                                                            small.tokens, small.attn_mask)
        shares = np.asarray(jax.device_get(shares), np.float64)
    mean_share = float(shares.mean())
    print(f"[calib] entry biases {np.round(bias, 3).tolist()}", flush=True)
    print(f"[calib] previous-slot share per mixing point {np.round(shares, 5).tolist()} mean {mean_share:.5f} "
          f"(target {args.feedback_init_share}) ({time.time() - t0:.0f}s)", flush=True)
    if not np.all(np.isfinite(bias)) or abs(mean_share - args.feedback_init_share) > 0.1 * args.feedback_init_share:
        raise SystemExit(f"calibration failed: mean share {mean_share} vs target {args.feedback_init_share}")
    out = {"entry_bias": bias.tolist(), "target": args.feedback_init_share, "share_per_mixing_point": shares.tolist(),
           "mean_share": mean_share, "base_checkpoint": ckpt, "step": args.init_step, "rows": args.rows,
           "batch": batch_size, "seq_len": cfg.max_seq_len, "seed": args.seed, "data_seed": args.data_seed,
           "data_root": args.data_root, "feedback_inject_rank": args.feedback_inject_rank,
           "feedback_proj": args.feedback_proj, "feedback_proj_rank": args.feedback_proj_rank,
           "seconds": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{args.out}.tmp"
    Path(tmp).write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, args.out)
    print(f"[calib] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
