"""The Marin grug MoE hero scaling-ladder recipe (d768 = MoE-1.6B, d1024 = MoE-4.0B; d896 = MoE-2.6B is
interpolated by ``alf.rungs``) as the paper's runs use it
(``marin-community/marin`` at ``8577d258214dd0dcb992c04458ee4eb10496d4e5``).

Model, optimizer and trainer knobs follow experiments/grug/moe_hero_ep/launch_scaling_ladder.py:
`_ladder_model(size)` (= `_small_model(SMALL_SHAPES[size], ...)`, 384 experts top-8, QB histogram estimator), the
rung's batch (1024 sequences per rack) and step count (791 tokens per active parameter), `MoeHeuristic` at that
full-run budget, gate/router weight decay 0.02, z-loss 1e-4, fp32 parameters with bf16 compute, inline watch,
coordinated GC every 100 steps.

Execution-layer choices for the sm_90 (Hopper) GPUs these runs used, none of which changes the model math:
- MoE transport: the ragged all-to-all of Marin's own target hardware needs its patched PJRT; here the transport is
  either `sonic` (FSDP, every expert on every device, dropless) or `fixed_pooled_wave_all_to_all` (expert parallel
  inside a node, as Marin's sm_90 ladder runs).
- `use_syrk=False`: QuACK's symmetric GEMM for the expert-stack Newton-Schulz needs sm_100 (Marin's own sm_90 targets
  in small_scale_abl_launch set the same).
- Optimizer state stays on device (it fits HBM at these widths; Marin's small-scale runs also keep it on device).
"""
from __future__ import annotations

import dataclasses
import json
import os

LADDER_RACKS = {"d768": 1, "d1024": 2}
SEQUENCES_PER_RACK = 1024
TOKENS_PER_ACTIVE_PARAM = 791
FULL_RUN_STEPS = 15276
FULL_RUN_BATCH = 2048
GATE_ROUTER_WEIGHT_DECAY = 0.02
Z_LOSS_WEIGHT = 1e-4
GC_INTERVAL = 100
FLAVOR_MOE = {"fsdp": "sonic", "ep8": "fixed_pooled_wave_all_to_all"}
SHUFFLE_BLOCK_TOKENS = 2**20


def model_config(*, shape: str = "d1024", layers: int | None = None, flavor: str = "fsdp"):
    attention, experts = "gpu_fa4_cute", 384
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_QB_HIST_BINS
    from experiments.grug.moe_hero_ep.small_scale_abl_launch import _EP_CAPACITY_FACTOR, SEQ_LEN, SMALL_SHAPES, _small_model

    moe = FLAVOR_MOE[flavor]
    pooled = moe == "fixed_pooled_wave_all_to_all"
    small = SMALL_SHAPES[shape]
    if layers is not None:
        small = dataclasses.replace(small, num_layers=layers)
    return _small_model(
        small,
        _EP_CAPACITY_FACTOR,
        attention_implementation=attention,
        moe_implementation=moe,
        expert_chunks=1,
        seq_len=SEQ_LEN,
        num_experts=experts,
        num_experts_per_token=8,
        intermediate_dim=None,
        latent_dim=None,
        pooled_transport_capacity_factor=_EP_CAPACITY_FACTOR if pooled else None,
        num_expert_waves=3 if pooled else 1,
        qb_use_histogram=True,
        qb_hist_bins=HERO_QB_HIST_BINS,
    )


def ladder_schedule(model, shape: str) -> tuple[int, int]:
    """The rung's (batch, steps) as launch_scaling_ladder.build_ladder_run derives them at 4K context."""
    from experiments.grug.moe_hero_ep.small_scale_abl_launch import _active_params

    batch = SEQUENCES_PER_RACK * LADDER_RACKS[shape]
    steps = max(1, round(TOKENS_PER_ACTIVE_PARAM * _active_params(model) / (batch * model.max_seq_len)))
    return batch, steps


def optimizer_config(model, *, num_train_steps: int = FULL_RUN_STEPS, batch_size: int = FULL_RUN_BATCH):
    from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic

    opt = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_train_steps, batch_size=batch_size, hidden_dim=model.hidden_dim, seq_len=model.max_seq_len
    )
    return dataclasses.replace(opt, gate_router_weight_decay=GATE_ROUTER_WEIGHT_DECAY, use_syrk=False)


def dolma_data_config(root: str, *, seq_len: int, eval_root: str | None = None):
    """One component concatenating the caches ``alf.tokenize_dolma`` wrote under ``root`` (shard / part order),
    with the Marin recipe's fixed-token block shuffle; returns (LmDataConfig, manifest). ``eval_root`` is a directory
    of held-out caches, one per source, with a manifest listing them; it adds one zero-weight eval component per
    held-out source, val_<source>."""
    from levanter.data.text.datasets import BlockShuffleConfig, ConcatDatasetComponent, DatasetComponent, LmDataConfig
    from levanter.data.text.formats import TextLmDatasetFormat

    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    eval_components = {}
    if eval_root is not None:
        with open(os.path.join(eval_root, "manifest.json")) as f:
            eval_manifest = json.load(f)
        eval_components = {
            f"val_{s['name']}": DatasetComponent(
                source=None, cache_dir=os.path.join(eval_root, s["name"]), format=TextLmDatasetFormat(), tags=["heldout"]
            )
            for s in eval_manifest["sources"]
        }
    children = {
        unit["name"]: DatasetComponent(
            source=None, cache_dir=os.path.join(root, unit["name"]), format=TextLmDatasetFormat(), flat_cache=True
        )
        for unit in manifest["units"]
    }
    if SHUFFLE_BLOCK_TOKENS % seq_len:
        raise ValueError(f"seq_len {seq_len} must divide {SHUFFLE_BLOCK_TOKENS} shuffle-block tokens")
    config = LmDataConfig(
        tokenizer=os.environ.get("MARIN_TOKENIZER_DIR", "marin-community/marin-tokenizer"),
        cache_dir=None,
        components={"dolma3_hq": ConcatDatasetComponent(children=children), **eval_components},
        train_weights={"dolma3_hq": 1.0, **{name: 0.0 for name in eval_components}},
        auto_build_caches=False,
        shuffle=BlockShuffleConfig(io_block_size=SHUFFLE_BLOCK_TOKENS // seq_len, window_blocks=512, perm_type="feistel"),
    )
    return config, manifest
