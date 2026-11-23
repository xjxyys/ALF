#!/usr/bin/env python3
"""Training entry point for the language-model runs of the paper: Attention Latent Feedback (ALF, arXiv:2611.31154).

Backbone, data, optimizer, checkpointing and evaluation are the Marin grug MoE hero scaling ladder
(``marin-community/marin`` at ``8577d258214dd0dcb992c04458ee4eb10496d4e5``): one JAX process per node, 1-8 nodes, the
rung's batch and step count from ``alf.marin.recipe``, the middle rung from ``alf.rungs``. The rungs are d768
(MoE-1.6B), d896 (MoE-2.6B) and d1024 (MoE-4.0B) active parameters. Only the model is swapped (``alf.model``):

  --variant plain     the standard pre-norm residual backbone, as the recipe trains it (the one-pass baseline);
  --variant attnres   the attention residual backbone (every sublayer input is a learned softmax over the sublayer
                  outputs below it; Full Attention Residuals, arXiv:2603.15031), one pass. --feedback-share /
                  --feedback-query / --feedback-rank / --feedback-inject-rank create the feedback parameters dormant
                  (they are zero-initialized), so a feedback run can branch from any of its checkpoints; keep
                  --feedback-inject-rank the same on the base and on every branch, as it fixes the parameter tree
                  whatever the share;
  --variant feedback  the attention residual backbone + ALF's sublayer feedback path: every sublayer applies a depth
                  softmax over the previous position's outputs at or above its own depth and mixes the result into its
                  backbone input. --passes K training passes (loss = pass 1 + the mean of the others), --eval-passes
                  for held-out evaluation. --feedback-share fbt instead fuses the previous position's top state into
                  the token input, the layer-0 gate product of the Full-bandwidth Transformer (FBT, arXiv:2608.08888);
  --variant stdres    the standard residual backbone + the same sublayer feedback path (--feedback-share pool: every
                  sublayer mixes the depth softmax over the previous position's deeper outputs into its input,
                  rescaled to the RMS of the residual stream; --feedback-query own, --feedback-proj slot
                  --feedback-proj-rank r), or FBT's layer-0 fusion of the previous top state (--feedback-share fbt).
                  This is the backbone the paper reports ALF on, and --embed-inject adds ALF's second pathway, the
                  input injection path. --passes K as for feedback; branches from a --variant plain checkpoint with
                  --allow-partial-checkpoint; --prefix-mixin and --jitter 0.02 reproduce FBT's own training recipe;
  --variant src       the standard residual backbone + FBT's single fixed-depth source route: the previous position's
                  stream after block --src-block injected into every block up to it. --passes K as for feedback.
  --fbt-input-norm unit|embed|match (--feedback-share fbt, on either backbone): the fused input at RMS 1, through
                  Marin's embedding norms, or at the RMS of the token input it replaces (see the ``alf.model``
                  docstring); written to <run-dir>/fbt_input_norm.txt on the first start, a resume with another value
                  is refused, and the evaluation scripts rebuild the model from that file.

These flags rarely need to be spelled out by hand: ``alf.kinds`` holds the "kind -> arguments" table that names every
model of the paper, and the evaluation scripts take a model as ``<run>:<kind>`` and build the arguments from it.

Branching. The feedback passes are switched on partway through a one-pass run, so a run usually starts from another
run's checkpoint: with no complete checkpoint of its own, --init-from DIR --init-step N copies DIR/checkpoints/step-N
into this run's checkpoints/ and the trainer resumes from it (same schedule, same data position). For
--feedback-share joint the entry biases are calibrated before the branch trains (``alf.calibrate``: the base
checkpoint's model on the branch step's training batch, ``alf.model.calibrate_joint_bias``, as FBT calibrates on the
first two-pass batch) and come in through --feedback-entry-bias FILE: they initialize the frontier_entry_bias leaf,
which the base does not have, so the partial restore keeps them.
A branch may add parameters its base does not carry (the paper's final form, --feedback-share pool --feedback-query
own --feedback-proj source --feedback-proj-rank r, from a base built with the shared query and no source maps):
--allow-partial-checkpoint then restores every leaf the checkpoint has and takes the missing ones (the second query at
zero, the source maps with a zero up projection or a zero full map, their Adam states) from the initialization.

  python scripts/train.py --run-dir DIR --run-id ID --shape d768 --data dolma --data-root ROOT --stop-step N \
      --variant attnres|feedback [--passes 2 --eval-passes 4] [--feedback-share pool --feedback-query shared
      --feedback-rank 0 --feedback-init-share 0.01 --jitter 0.02] [--init-from DIR --init-step N] [recipe knobs ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import timedelta


def _branch_copy(init_from: str, init_step: int, run_dir: str) -> None:
    ckpt = os.path.join(run_dir, "checkpoints")
    have = [d for d in os.listdir(ckpt)] if os.path.isdir(ckpt) else []
    if any(d.startswith("step-") for d in have):
        print(f"[train] {ckpt} already holds {sorted(have)}; not copying the branch source", flush=True)
        return
    src = os.path.join(init_from, "checkpoints", f"step-{init_step}")
    if not os.path.isfile(os.path.join(src, "metadata.json")):
        raise SystemExit(f"branch source {src} has no complete checkpoint (metadata.json missing)")
    os.makedirs(ckpt, exist_ok=True)
    tmp = os.path.join(ckpt, f".copy-step-{init_step}")
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    shutil.copytree(src, tmp)
    os.replace(tmp, os.path.join(ckpt, f"step-{init_step}"))
    print(f"[train] branched from {src} ({time.time() - t0:.0f}s copy)", flush=True)


def main() -> None:
    from alf.marin.process import exit_after_success, skip_core_dumps

    skip_core_dumps()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--shape", choices=("d768", "d896", "d1024"), default="d768",
                    help="d896: the middle rung of the paper's MoE table (alf.rungs), batch 1440 and 791 tokens per "
                         "active parameter unless --batch / --schedule-steps say otherwise")
    ap.add_argument("--flavor", choices=("fsdp", "ep8"), default="fsdp")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--schedule-steps", type=int, default=None)
    ap.add_argument("--stop-step", type=int, required=True)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--save-minutes", type=float, default=30.0)
    ap.add_argument("--data", choices=("synthetic", "dolma"), default="synthetic")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--expert-axis", type=int, default=8)
    ap.add_argument("--coordinator", default=None)
    ap.add_argument("--num-processes", type=int, default=1)
    ap.add_argument("--process-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--microbatches", type=int, default=1)
    ap.add_argument("--variant", choices=("plain", "attnres", "feedback", "src", "stdres"), required=True,
                    help="src: the standard residual backbone + one fixed source of the previous position "
                         "(--src-block); stdres: the standard residual backbone + the sublayer feedback path "
                         "(--feedback-share pool) or FBT's fusion (fbt)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--eval-passes", type=int, default=None, help="default: --passes (feedback / src), 1 (attnres)")
    ap.add_argument("--feedback-share", choices=("joint", "pool", "inject", "fbt"), default="pool",
                    help="fbt: no previous slot, FBT's layer-0 gate_product fusion of the previous top state")
    ap.add_argument("--src-block", type=int, default=-1,
                    help="variant src: source block s (its stream feeds blocks <= s)")
    ap.add_argument("--src-rank", type=int, default=64, help="variant src: rank of each target block's map")
    ap.add_argument("--src-alpha", type=float, default=1.0, help="variant src: scale of the injected delta (FBT alpha)")
    ap.add_argument("--feedback-query", choices=("shared", "own", "uniform"), default="shared")
    ap.add_argument("--feedback-rank", type=int, default=0)
    ap.add_argument("--feedback-inject-rank", type=int, default=0, help="rank of each layer's injected-source map")
    ap.add_argument("--feedback-init-share", type=float, default=0.01)
    ap.add_argument("--feedback-proj", choices=("none", "source", "slot"), default="none",
                    help="source: M_i v = v + Delta_i v on the previous token's two slots of layer i (FBT, "
                         "arXiv:2608.08888)")
    ap.add_argument("--feedback-proj-rank", type=int, default=0, help="rank of each Delta_i (0 = full D x D)")
    ap.add_argument("--feedback-depth", default="whole",
                    help="whole: each mixing point reads the previous position's deeper outputs; top: only its last "
                         "layer (ablation); win<k>: the fixed nearest window, an equal-weight average of the k "
                         "outputs from its own depth upward, with no depth softmax")
    ap.add_argument("--feedback-replaces-input", choices=("residual", "input"), default="residual",
                    help="variant stdres, injection ablation: input = sublayer 0's feedback read goes through FBT's "
                         "gate product in place of the token input (FBT's injection with ALF's read)")
    ap.add_argument("--feedback-source", choices=("outputs", "top"), default="outputs",
                    help="variant stdres, injection ablation: top = every sublayer reads only the previous position's "
                         "final state through its own map, mixed in as ALF mixes (ALF's injection with FBT's read)")
    ap.add_argument("--embed-inject", default="none",
                    help="variant stdres with --feedback-share pool: the input injection path. Passes 2..K also add "
                         "the previous position's top state, RMS-normalized and through a zero-initialized D x D map, "
                         "to the token input. plain:<c> is the form the paper reports (Recycler's adapter as it is, "
                         "x0 + c * RMSNorm(h) W_E, c the learning-rate multiplier of W_E; see github.com/ifm-ai/LoRM) "
                         "and plainonly:<c> is the same without the sublayer feedback path; match (at the RMS of the "
                         "embedded token), embed (to the raw embedding before Marin's embedding norms) and only (the "
                         "match injection without the sublayer feedback path) are retained for checkpoints trained "
                         "before the final form. Recorded in the run directory")
    ap.add_argument("--allow-partial-checkpoint", action="store_true",
                    help="branches whose base checkpoint lacks the new parameters: restore the leaves it has, "
                         "initialize the missing ones (see Branching in the module docstring)")
    ap.add_argument("--feedback-entry-bias", default=None,
                    help="JSON from alf.calibrate: initial entry biases of a joint branch")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--prefix-mixin", action="store_true",
                    help="variant stdres: FBT's --feedback-prefix-mixin (each feedback pass, every document keeps an "
                         "independently sampled plain prefix; training only)")
    ap.add_argument("--fbt-input-norm", choices=("unit", "embed", "match"), default="unit",
                    help="share fbt: scale of the fused input (unit: RMS 1, the paper's setting; embed: through "
                         "Marin's embedding norms; match: the RMS of the token input it replaces, retained for "
                         "checkpoints trained before the final form); recorded in the run directory")
    ap.add_argument("--init-from", default=None, help="branch source run directory")
    ap.add_argument("--init-step", type=int, default=None)
    args = ap.parse_args()
    # Keep the command line in sync with the model's parse_depth / window contract.
    depth = args.feedback_depth
    window_depth = depth.startswith("win") and depth[3:].isdigit() and int(depth[3:]) >= 1
    if depth not in ("whole", "top") and not window_depth:
        ap.error("--feedback-depth must be whole, top, or win<k> with k >= 1")
    if window_depth and (args.variant != "stdres" or args.feedback_share != "pool" or args.feedback_query != "uniform"):
        ap.error("--feedback-depth win<k> requires --variant stdres --feedback-share pool --feedback-query uniform")
    if (args.init_from is None) != (args.init_step is None):
        raise SystemExit("--init-from and --init-step go together")
    if args.variant == "stdres" and (args.feedback_share not in ("pool", "fbt") or args.feedback_rank or args.feedback_inject_rank
                                 or (depth != "whole" and args.feedback_share != "pool")):
        raise SystemExit("--variant stdres takes --feedback-share pool or fbt with --feedback-rank 0, "
                         "--feedback-inject-rank 0 and --feedback-depth whole (top only with --feedback-share pool)")
    if args.prefix_mixin and args.variant != "stdres":
        raise SystemExit("--prefix-mixin goes with --variant stdres")
    if (args.feedback_replaces_input, args.feedback_source) != ("residual", "outputs") and args.variant != "stdres":
        raise SystemExit("--feedback-replaces-input input / --feedback-source top go with --variant stdres")
    if args.embed_inject != "none" and not (args.variant in ("stdres", "feedback") and args.feedback_share == "pool"):
        raise SystemExit("--embed-inject goes with --feedback-share pool on --variant stdres or feedback")
    if (args.embed_inject != "none" and args.variant == "feedback"
            and args.embed_inject.partition("@")[0].partition(":")[0] not in ("plain", "plainonly")):
        raise SystemExit("on --variant feedback the injection is plain:<c> or plainonly:<c>")
    fbt_run = (args.variant in ("feedback", "stdres") and args.feedback_share == "fbt") or args.feedback_replaces_input == "input"
    if args.fbt_input_norm != "unit" and not fbt_run:  # fbt_run includes --feedback-replaces-input input
        raise SystemExit("--fbt-input-norm embed / match goes with --feedback-share fbt (variant feedback or stdres)")
    entry_bias = None
    if args.feedback_entry_bias:
        if args.feedback_share != "joint":
            raise SystemExit("--feedback-entry-bias goes with --feedback-share joint")
        import json

        with open(args.feedback_entry_bias) as f:
            entry_bias = [float(b) for b in json.load(f)["entry_bias"]]
    if args.init_from and args.variant == "feedback" and args.feedback_share == "joint":
        if entry_bias is None:
            raise SystemExit("a joint branch needs --feedback-entry-bias (produced by alf.calibrate)")
        if not args.allow_partial_checkpoint:
            raise SystemExit("a joint branch from a pool / AttnRes base needs --allow-partial-checkpoint (new leaf)")
    if args.allow_partial_checkpoint:
        # The new parameters are zero-initialized by design, so a parameter tree that does not match the checkpoint
        # is not an error: the feedback pathway loads as zeros and the run trains the one-pass backbone while
        # reporting a plausible loss. The first step's pass-1 loss is the check -- it should match the base run's
        # last step.
        print("[train] partial restore on: leaves missing from the checkpoint start at their initialization. "
              "If the arguments do not match the run this branches from, the feedback parameters load as zeros "
              "and this trains the backbone alone; compare the first step's pass-1 loss with the base's last step.",
              flush=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stdout)

    import jax
    import jmp
    from fray.cluster import ResourceConfig
    from levanter.checkpoint import CheckpointerConfig
    from levanter.data.text.datasets import LmDataConfig
    from levanter.distributed import DistributedConfig
    from levanter.tokenizers import load_tokenizer
    from levanter.tracker.json_logger import JsonLoggerConfig
    from levanter.trainer import TrainerConfig

    from alf import model as attnres_model
    from experiments.grug.moe_hero_ep import train as grug_train
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from alf.marin.compat import patch_checkpoint_write_copies_source, patch_single_device_newton_schulz, register_gpu_kinds_for_pallas
    from alf.marin.recipe import GC_INTERVAL, Z_LOSS_WEIGHT, dolma_data_config, ladder_schedule, model_config, optimizer_config

    from alf import rungs as mid_rung

    mid_rung.register_mid_shapes()
    model = model_config(shape=args.shape, layers=args.layers, flavor=args.flavor)
    if args.shape in mid_rung.MID_SHAPES:
        from experiments.grug.moe_hero_ep.small_scale_abl_launch import _active_params

        rung_batch, rung_steps = mid_rung.schedule(_active_params(model), seq_len=model.max_seq_len)
    else:
        rung_batch, rung_steps = ladder_schedule(model, args.shape)
    args.batch = args.batch or rung_batch
    args.schedule_steps = args.schedule_steps or rung_steps
    optimizer = optimizer_config(model, num_train_steps=args.schedule_steps, batch_size=args.batch)
    tokenizer_dir = os.environ.get("MARIN_TOKENIZER_DIR", "marin-community/marin-tokenizer")
    if args.data == "dolma":
        if not args.data_root:
            raise SystemExit("--data dolma needs --data-root")
        data_config, manifest = dolma_data_config(args.data_root, seq_len=model.max_seq_len, eval_root=args.eval_root)
        needed = args.stop_step * args.batch * model.max_seq_len
        print(f"[train] data {args.data_root}: {len(manifest['units'])} caches, {manifest['total_tokens']} tokens; run to "
              f"step {args.stop_step} reads {needed}", flush=True)
        if needed > manifest["total_tokens"]:
            raise SystemExit(f"data too small: {manifest['total_tokens']} tokens < {needed} needed by step {args.stop_step}")
        data_mode = grug_train.TrainingDataMode.MIXTURE
    else:
        if args.eval_root:
            raise SystemExit("--eval-root needs --data dolma")
        data_config = LmDataConfig(tokenizer=tokenizer_dir, components={})
        data_mode = grug_train.TrainingDataMode.SYNTHETIC
    eval_config = None
    if args.eval_root:
        eval_config = grug_train.GrugEvalConfig(eval_batch_size=args.eval_batch, steps_per_eval=args.eval_every,
                                                eval_current=True, eval_ema=False, compute_bpb=True)
    master_mode = grug_train.MasterParamMode.DEVICE
    distributed = DistributedConfig()
    if args.num_processes > 1:
        if not args.coordinator:
            raise SystemExit("--coordinator is required with --num-processes > 1")
        distributed = DistributedConfig(coordinator_address=args.coordinator, num_processes=args.num_processes,
                                        process_id=args.process_id)
    if args.init_from and args.process_id == 0:
        _branch_copy(args.init_from, args.init_step, args.run_dir)
    if fbt_run and args.process_id == 0:
        try:
            attnres_model.record_fbt_input_norm(args.run_dir, args.fbt_input_norm)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.embed_inject != "none" and args.process_id == 0:
        try:
            attnres_model.record_embed_inject(args.run_dir, args.embed_inject)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.num_processes > 1 and args.init_from:
        raise SystemExit("multi-process branch copy is not wired (run the first phase on one node)")
    ckpt = os.path.join(args.run_dir, "checkpoints")
    trainer = TrainerConfig(
        id=args.run_id, seed=args.seed, train_batch_size=args.batch, num_train_steps=args.schedule_steps,
        mp=jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[master_mode]), tracker=JsonLoggerConfig(),
        use_explicit_mesh_axes=True, require_accelerator=True, allow_nondivisible_batch_size=False,
        log_dir=os.path.join(args.run_dir, "logs"), distributed=distributed,
        checkpointer=CheckpointerConfig(
            base_path=ckpt, temporary_base_path=os.path.join(args.run_dir, "checkpoints_tmp"),
            save_interval=timedelta(minutes=args.save_minutes),
            keep=[{"every": args.save_every}] if args.save_every else None, append_run_id_to_base_path=False,
            delete_old_temp_checkpoints=True, keep_last_temporary_checkpoints=1,
        ),
        allow_partial_checkpoint=args.allow_partial_checkpoint,
    )
    grug_trainer = grug_train.GrugTrainerConfig(
        trainer=trainer, log_every=1, gc_interval=GC_INTERVAL, z_loss_weight=Z_LOSS_WEIGHT, offload_opt_state=False,
        master_param_mode=master_mode, training_data_mode=data_mode, watch_mode=grug_train.WatchMode.INLINE,
        save_checkpoints=True, expert_axis_size=args.expert_axis if args.flavor == "ep8" else 1, replica_axis_size=None,
        data_seed=args.data_seed,
    )
    # The resource spec follows the node rather than being fixed at the eight-GPU nodes the paper's runs used.
    config = grug_train.GrugRunConfig(model=model, data=data_config,
                                      resources=ResourceConfig.with_gpu("H100", count=jax.local_device_count()),
                                      optimizer=optimizer, trainer=grug_trainer, eval=eval_config,
                                      stop_after_steps=args.stop_step)
    if args.variant == "src":
        bos_id = load_tokenizer(tokenizer_dir).bos_token_id
        eval_passes = args.eval_passes or args.passes
        attnres_model.install_src(passes=args.passes, eval_passes=eval_passes, src_block=args.src_block,
                                  src_rank=args.src_rank, src_alpha=args.src_alpha, jitter=args.jitter,
                                  bos_id=-1 if bos_id is None else int(bos_id))
        print(f"[train] src variant passes={args.passes} eval_passes={eval_passes} src_block={args.src_block} "
              f"src_rank={args.src_rank} src_alpha={args.src_alpha} jitter={args.jitter} bos_id={bos_id} "
              f"allow_partial={args.allow_partial_checkpoint}", flush=True)
    elif args.variant == "stdres":
        bos_id = load_tokenizer(tokenizer_dir).bos_token_id
        eval_passes = args.eval_passes or args.passes
        attnres_model.install_stdres(passes=args.passes, eval_passes=eval_passes, share=args.feedback_share,
                                     query=args.feedback_query, init_share=args.feedback_init_share,
                                     proj=args.feedback_proj, proj_rank=args.feedback_proj_rank, jitter=args.jitter,
                                     bos_id=-1 if bos_id is None else int(bos_id), prefix_mixin=args.prefix_mixin,
                                     fbt_input_norm=args.fbt_input_norm, depth=args.feedback_depth,
                                     inject=args.feedback_replaces_input, source=args.feedback_source,
                                     embed_inject=args.embed_inject)
        print(f"[train] stdres variant (standard residual backbone) passes={args.passes} eval_passes={eval_passes} "
              f"share={args.feedback_share} query={args.feedback_query} init_share={args.feedback_init_share} "
              f"proj={args.feedback_proj} proj_rank={args.feedback_proj_rank} jitter={args.jitter} "
              f"prefix_mixin={args.prefix_mixin} fbt_input_norm={args.fbt_input_norm} depth={args.feedback_depth} "
              f"inject={args.feedback_replaces_input} source={args.feedback_source} embed_inject={args.embed_inject} "
              f"bos_id={bos_id} "
              f"allow_partial={args.allow_partial_checkpoint}", flush=True)
    elif args.variant != "plain":
        bos_id = load_tokenizer(tokenizer_dir).bos_token_id
        passes = args.passes if args.variant == "feedback" else 1
        eval_passes = args.eval_passes or passes
        attnres_model.install(passes=passes, eval_passes=eval_passes, share=args.feedback_share,
                              query=args.feedback_query, rank=args.feedback_rank, init_share=args.feedback_init_share,
                              jitter=args.jitter, bos_id=-1 if bos_id is None else int(bos_id),
                              inject_rank=args.feedback_inject_rank, proj=args.feedback_proj,
                              proj_rank=args.feedback_proj_rank, entry_bias=entry_bias, depth=args.feedback_depth,
                              fbt_input_norm=args.fbt_input_norm)
        print(f"[train] attnres variant={args.variant} passes={passes} eval_passes={eval_passes} share={args.feedback_share} "
              f"fbt_input_norm={args.fbt_input_norm} "
              f"query={args.feedback_query} rank={args.feedback_rank} inject_rank={args.feedback_inject_rank} "
              f"init_share={args.feedback_init_share} jitter={args.jitter} bos_id={bos_id} proj={args.feedback_proj} "
              f"proj_rank={args.feedback_proj_rank} depth={args.feedback_depth} allow_partial={args.allow_partial_checkpoint}"
              + ("" if entry_bias is None else f" entry_bias={['%.3f' % b for b in entry_bias]}"), flush=True)
    if args.microbatches > 1:
        from alf.marin import microbatch

        microbatch.install(args.microbatches)
    print(f"[train] run={args.run_id} variant={args.variant} shape={args.shape} layers={model.num_layers} batch={args.batch} "
          f"schedule={args.schedule_steps} stop={args.stop_step} save_every={args.save_every} data={args.data} "
          f"seed={args.seed} data_seed={args.data_seed} init={args.init_from}@{args.init_step} "
          f"process={args.process_id}/{args.num_processes}", flush=True)
    grug_train._apply_hero_ep_runtime_defaults(inline_watch_enabled=trainer.watch.is_enabled,
                                               moe_implementation=model.moe_implementation,
                                               remat_mode=model.remat_mode, processes_per_task=1)
    register_gpu_kinds_for_pallas()
    patch_single_device_newton_schulz()
    patch_checkpoint_write_copies_source()
    t0 = time.time()
    grug_train._run_grug_local(config)
    print(f"[train] done in {time.time() - t0:.1f}s stop_step={args.stop_step}", flush=True)
    exit_after_success(args.num_processes, args.process_id)


if __name__ == "__main__":
    main()
