#!/usr/bin/env python3
"""Correctness checks for the ALF model (alf/model.py): attention latent feedback on the attention residual
(AttnRes, arXiv:2603.15031) and the standard residual trunk of Marin's MoE backbone. Every model path is compared
against a reference implementation written out independently from the paper's formulas, so a mistake in either one
shows up as a mismatch. One device; the d768 cases need a GPU with the real kernels.

tiny (fp32, reference attention, dropless `scatter` MoE, the features of the d768 / MoE-1.6B recipe on; new parameters
randomized so that every path carries signal):
  1. one pass == an independent per-sublayer reference written from the formulas (hidden, router metrics, loss);
  2. two passes, every feedback variant (joint; pool x shared / own / uniform; rank 0 / 4) == the reference built from
     the first pass's slots shifted by one position and zeroed at document starts;
  3. zero share (joint bias / pool logit -> -1e4) reproduces the AttnRes trunk;
  4. packed documents: changing one token leaves earlier positions, the other document and the other row unchanged;
  5. K = T Jacobi passes == the exact token-by-token recursion (built position by position from exact states);
  6. gradients: one pass leaves every feedback parameter at exactly zero gradient (dormant); two passes give every new
     parameter a non-zero first-step gradient; the training loss = pass-1 CE + mean of later passes' CE;
  7. optimizer mask: every new parameter goes to Adam (a MuonH zero matrix would never move);
  8. install(): Transformer.init builds the model with Marin's weights for the same key; assignment accounting and FLOPs
     count passes x layers; calibration hits the target share.
d768 (2 layers, bf16 compute as in training, FA4 attention, sonic MoE): finite loss and gradients for one pass and for
two passes (joint and pool), and causality on the real kernels.
Group 14 (variant stdres: Marin's standard residual trunk + the sublayer feedback path, or the fusion of the
Full-bandwidth Transformer, FBT, arXiv:2608.08888) repeats 1-8 for that trunk (reference, Jacobi == exact, zero share,
gradients, branch from a plain Marin checkpoint, install) and adds a d768 case (pass 1 == Marin's own loss on the real
kernels). Group 15 checks FBT's training recipe on that variant: the prefix mixin mask against FBT's
build_feedback_mask, passes reading only where their mask says, evaluation unaffected, the jitter against a reference
with the same noise, gradients, install; plus d768 cases on the real kernels. Group 16 checks the decoding modes
(run_passes context / context_passes) on both trunks with the feedback path and FBT's fusion: K = T Jacobi passes == a
prompt through c passes (c = 1: SOFT) kept, then exact token-by-token decoding; an empty prompt or c >= K is every
position reading; causality; pass_losses; refusal in training; plus d768 cases on the real kernels. Group 17 checks
fbt_input_norm (unit / embed / match) of FBT's fused input on both trunks: pass 2 == the reference, the fused input's
scale (RMS 1 / the embedding norms / rms(x0) per position), x0 kept at document starts, causality, Jacobi == exact,
gradients, validation, branching from a plain checkpoint, the run-directory record; plus d768 cases on the real
kernels.
Group 23 (--only-embed-plain-attnres) checks embed_inject plain:<c> / plainonly:<c> on the AttnRes trunk (variant
feedback, share pool): the final mixture _trunk hands out, two passes == a reference that reads it, the zero start, the
constant folding into the weights, gradients and the optimizer group, causality, Jacobi == exact, the decoding modes,
the rejections, the branch from an AttnRes base; plus d768 cases.

Group 24 (--only-window) checks --feedback-depth win<k>, the fixed nearest window of the paper's Section 6.2: every
mixing point's read against a written-out equal-weight mean, a window as wide as the depth against the whole depth, a
zero share against Marin's own transformer, Jacobi == exact, and the refusal of a learned query. Because win<k> and
whole build the same parameter tree, a wrong window cannot raise; the check that win4 and whole disagree on the same
weights is what would catch a model trained or scored under the other depth.

  python tests/test_model.py [--skip-d768]
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys

# Read .env and put the Marin checkout on sys.path, so this file can be run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alf.cli import load_env  # noqa: E402
from alf.marin.bootstrap import bootstrap  # noqa: E402

load_env()
bootstrap()

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, Mesh, set_mesh  # noqa: E402

from alf import model as am  # noqa: E402
import experiments.grug.moe_hero_ep.model as gm  # noqa: E402

FAILED = []
BOS = 1
START = 1
ONLY_INJECT = False


def check(ok: bool, msg: str) -> None:
    print(f"[{'ok  ' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok:
        FAILED.append(msg)


def max_diff(a, b) -> float:
    diffs = jax.tree.leaves(jax.tree.map(
        lambda x, y: float(jnp.max(jnp.abs(jnp.asarray(x, jnp.float32) - jnp.asarray(y, jnp.float32)))), a, b))
    return max(diffs) if diffs else 0.0


def max_abs(a) -> float:
    return max(float(jnp.max(jnp.abs(jnp.asarray(x, jnp.float32)))) for x in jax.tree.leaves(a))


def explicit_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1), ("replica_dcn", "data", "context", "expert", "model"),
                axis_types=(AxisType.Explicit,) * 5)


def tiny_config() -> gm.GrugModelConfig:
    return gm.GrugModelConfig(
        vocab_size=128, hidden_dim=32, intermediate_dim=16, shared_expert_intermediate_dim=16, num_shared_experts=2,
        num_experts=8, num_experts_per_token=2, latent_dim=16, qb_estimator=gm.QbEstimator.HIST, qb_hist_bins=64,
        num_layers=4, num_heads=4, num_kv_heads=2, local_kv_heads=1, global_kv_heads=2, head_dim=8, max_seq_len=16,
        sliding_window=4, global_every=2, capacity_factor=1.0, initializer_std=0.5 / math.sqrt(32), qk_mult=1.3,
        sconv=True, sconv_kernel=4, sconv_sites=("k", "attn", "mlp"), rope_fused=True, remat_mode="recompute_all",
        attention_implementation="reference", moe_implementation="scatter", report_capacity_overflow=True,
    )


def packed_batch(seq_len: int, split: int, vocab: int, key):
    """Row 0: documents [0, split) and [split, S) (BOS at both starts); row 1: one document."""
    tokens = jax.random.randint(key, (2, seq_len), 2, vocab)
    tokens = tokens.at[:, 0].set(BOS).at[0, split].set(BOS)
    seg = jnp.stack([jnp.where(jnp.arange(seq_len) < split, 0, 1), jnp.zeros(seq_len, jnp.int32)]).astype(jnp.int32)
    return tokens, gm.AttentionMask.causal().with_segment_ids(seg)


def variant(base, key, *, share="none", query="shared", rank=0, passes=1, eval_passes=None, jitter=0.0, randomize=True,
            inject_rank=0, proj="none", proj_rank=0, entry_bias=None, depth="whole", fbt_input_norm="unit",
            embed_inject="none"):
    model = am.AttnResTransformer.from_base(base, key=key, share=share, query=query, rank=rank, passes=passes,
                                            eval_passes=eval_passes or passes, jitter=jitter, bos_id=BOS,
                                            inject_rank=inject_rank, proj=proj, proj_rank=proj_rank,
                                            entry_bias=entry_bias, depth=depth, fbt_input_norm=fbt_input_norm,
                                            embed_inject=embed_inject)
    if not randomize:
        return model
    ks = jax.random.split(jax.random.fold_in(key, 7), 5)
    d = base.config.hidden_dim
    model = eqx.tree_at(lambda m: m.attnres_query.weight, model,
                        2.0 * jax.random.normal(ks[0], model.attnres_query.weight.shape) / math.sqrt(d))
    if model.frontier_logit is not None:
        model = eqx.tree_at(lambda m: m.frontier_logit.weight, model,
                            jax.random.uniform(ks[1], model.frontier_logit.weight.shape,
                                               minval=-2.0, maxval=0.0))
    if model.frontier_entry_bias is not None:
        model = eqx.tree_at(lambda m: m.frontier_entry_bias.weight, model,
                            jax.random.uniform(ks[1], model.frontier_entry_bias.weight.shape, minval=-2.0, maxval=0.0))
    if model.frontier_query is not None:
        model = eqx.tree_at(lambda m: m.frontier_query.weight, model,
                            2.0 * jax.random.normal(ks[2], model.frontier_query.weight.shape) / math.sqrt(d))
    if model.frontier_up is not None:
        model = eqx.tree_at(lambda m: m.frontier_up.weight, model,
                            jax.random.normal(ks[3], model.frontier_up.weight.shape) / math.sqrt(rank))
    if model.frontier_inject_up is not None:
        model = eqx.tree_at(lambda m: m.frontier_inject_up.weight, model,
                            jax.random.normal(ks[4], model.frontier_inject_up.weight.shape) / math.sqrt(inject_rank))
    kp = jax.random.fold_in(key, 11)
    if model.frontier_proj_up is not None:
        model = eqx.tree_at(lambda m: m.frontier_proj_up.weight, model,
                            jax.random.normal(kp, model.frontier_proj_up.weight.shape) / math.sqrt(proj_rank))
    if model.frontier_proj_full is not None:
        model = eqx.tree_at(lambda m: m.frontier_proj_full.weight, model,
                            0.5 * jax.random.normal(kp, model.frontier_proj_full.weight.shape) / math.sqrt(d))
    return model


# ---------------------------------------------------------------------------------------------------- reference (fp32)
def ref_scores(v, q, eps_d):
    v32 = v.astype(jnp.float32)
    return jnp.einsum("...d,jd->...j", v32, q.astype(jnp.float32)) / jnp.sqrt(jnp.sum(v32 * v32, -1, keepdims=True) + eps_d)


def ref_masks(model, tokens, mask):
    cfg = model.config
    q_seg = gm._batch_reshard(mask.segment_ids[0])
    seg = (q_seg, q_seg)
    short_mask = gm.AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=seg)
    long_mask = gm.AttentionMask(is_causal=True, sliding_window=None, segment_ids=seg)
    b, s = tokens.shape
    long_lb, valid = gm.fa4_cute_segment_bounds(long_mask, batch_size=b, seq_len=s, sliding_window=None)
    short_lb, _ = gm.fa4_cute_segment_bounds(short_mask, batch_size=b, seq_len=s, sliding_window=cfg.sliding_window)
    with jax.ensure_compile_time_eval():
        schedule = [bool(x) for x in np.asarray(gm._long_layer_schedule(cfg.num_layers, cfg.global_every))]
    layer_masks = [long_mask.with_fa4_bounds(long_lb if lg else short_lb, valid) for lg in schedule]
    starts = (tokens == BOS) | (jnp.arange(s)[None, :] == 0) | jnp.pad(q_seg[:, 1:] != q_seg[:, :-1], ((0, 0), (1, 0)))
    return layer_masks, schedule, ~starts


def ref_trunk(model, tokens, mask, prev=None, x0=None):
    """prev: list of 2L previous-token slot arrays (already shifted / zeroed / mapped) and the feedback-position mask;
    x0: slot 0 in place of the embedded token (share "fbt")."""
    cfg = model.config
    n_mix, eps_d = 2 * cfg.num_layers, cfg.hidden_dim * am.ATTN_RES_EPS
    layer_masks, schedule, _ = ref_masks(model, tokens, mask)
    blocks = model.stacked_blocks.unstacked()
    q = model.attnres_query.weight
    slots = [model.embed_gated_norm(model.embed_norm(gm._embedding_gather(model.token_embed, tokens))) if x0 is None else x0]

    def softmax_mix(logits, values):
        w = jax.nn.softmax(jnp.stack(logits, -1), axis=-1)
        return sum(w[..., s:s + 1] * v for s, v in enumerate(values))

    inject = model.feedback_share == "inject"

    def pool_ref(j):
        cand = prev[0][j if model.feedback_depth == "whole" else max(j, n_mix - 2):]
        if model.feedback_query_kind == "uniform":
            return sum(cand) / len(cand)
        fq = q[:n_mix] if model.feedback_query_kind == "shared" else model.frontier_query.weight
        return softmax_mix([ref_scores(c, fq, eps_d)[..., j] for c in cand], cand)

    def inject_ref(i):
        if prev is None:
            return jnp.zeros_like(slots[0])
        n = pool_ref(2 * i)
        normed = n / jnp.sqrt(jnp.mean(n * n, -1, keepdims=True) + am.ATTN_RES_EPS)
        return (normed @ model.frontier_inject_down.weight[i]) @ model.frontier_inject_up.weight[i] * prev[1][..., None]

    def mix_ref(j):
        logits = [ref_scores(v, q, eps_d)[..., j] for v in slots]
        if prev is None or j >= n_mix or inject:
            return softmax_mix(logits, slots)
        cand, valid = prev[0][j:], prev[1]
        if model.feedback_share == "joint":
            b = model.frontier_entry_bias.weight[j]
            plog = [jnp.where(valid, ref_scores(c, q[:n_mix], eps_d)[..., j] + b, -jnp.inf) for c in cand]
            return softmax_mix(logits + plog, slots + cand)
        m = softmax_mix(logits, slots)
        rho = (jax.nn.sigmoid(model.frontier_logit.weight[j]) * valid)[..., None]
        return (1.0 - rho) * m + rho * pool_ref(j)

    stats, outs = [], []
    for i, (block, lg) in enumerate(zip(blocks, schedule)):
        flag = jnp.asarray(lg)
        lm = layer_masks[i]
        seg = lm.segment_ids[0]
        if inject:
            slots.append(inject_ref(i))
        h = mix_ref(2 * i)
        a = block.attn(block.attn_gated_norm(block.rms_attn(h)), lm, disable_rope=flag, is_global=flag)
        slots.append(block.sconv_attn(a, seg) if block.sconv_attn is not None else a)
        outs.append(slots[-1])
        h = mix_ref(2 * i + 1)
        mlp_in = block.mlp_gated_norm(block.rms_mlp(h))
        tv = gm.token_validity_from_attention_mask(lm, batch_size=h.shape[0], sequence_length=h.shape[1])
        o, st = block.mlp(mlp_in, tv)
        for se in block.shared or ():
            o = o + se(mlp_in, activation=gm.ActivationFunctionEnum.silu)
        slots.append(block.sconv_mlp(o, seg) if block.sconv_mlp is not None else o)
        outs.append(slots[-1])
        stats.append(st)
    out = model.final_gated_norm(model.final_norm(mix_ref(n_mix)))
    return out, outs, stats


def ref_prev(model, slots, fb_valid):
    """Shift each slot right by one position, zero positions without feedback, apply the shared low-rank map, then
    (proj "source") M_i p = p + Delta_i p for the two slots of layer i, (proj "slot") M_k p = p + Delta_k p per slot."""
    out = []
    for k, v in enumerate(slots):
        p = jnp.pad(v[:, :-1], ((0, 0), (1, 0), (0, 0))) * fb_valid[..., None]
        if model.frontier_down is not None:
            p = (p @ model.frontier_down.weight) @ model.frontier_up.weight
        m_idx = k if model.feedback_proj == "slot" else k // 2
        if model.frontier_proj_full is not None:
            p = p + p @ model.frontier_proj_full.weight[m_idx]
        elif model.frontier_proj_up is not None:
            p = p + (p @ model.frontier_proj_down.weight[m_idx]) @ model.frontier_proj_up.weight[m_idx]
        out.append(p)
    return out, fb_valid


def ref_passes(model, tokens, mask, n):
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, slots, _ = ref_trunk(model, tokens, mask)
    outs = [out]
    for _ in range(n - 1):
        out, slots, _ = ref_trunk(model, tokens, mask, prev=ref_prev(model, slots, fb_valid))
        outs.append(out)
    return outs


def ref_exact(model, tokens, mask):
    """Token-by-token recursion: position t reads the exact slots of t - 1. After step t positions 0..t are exact."""
    s = tokens.shape[1]
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, slots, _ = ref_trunk(model, tokens, mask)
    exact = [jnp.zeros_like(v) for v in slots]
    out_exact = jnp.zeros_like(out)
    for t in range(s):
        if t:
            out, slots, _ = ref_trunk(model, tokens, mask, prev=ref_prev(model, exact, fb_valid))
        exact = [e.at[:, t].set(v[:, t]) for e, v in zip(exact, slots)]
        out_exact = out_exact.at[:, t].set(out[:, t])
    return out_exact


def ref_norm(x):
    return x / jnp.sqrt(jnp.mean(x * x, -1, keepdims=True) + am.ATTN_RES_EPS)


def ref_fbt_scaled(model, fused, x0):
    """fbt_input_norm on the RMS-1 fused input: unit as is, embed through the embedding norms, match at rms(x0)."""
    if model.fbt_input_norm == "embed":
        return model.embed_gated_norm(model.embed_norm(fused))
    if model.fbt_input_norm == "match":
        return fused * jnp.sqrt(jnp.mean(x0 * x0, -1, keepdims=True) + am.ATTN_RES_EPS)
    return fused


def ref_fbt_input(model, tokens, top, fb_valid):
    """FBT gate_product on the previous position's top state: norm(W_s h_{t-1} * sigmoid(W_g norm(e_t))), then
    fbt_input_norm."""
    h = jnp.pad(top[:, :-1], ((0, 0), (1, 0), (0, 0)))
    e = gm._embedding_gather(model.token_embed, tokens)
    x0 = model.embed_gated_norm(model.embed_norm(e))
    fused = ref_norm((h @ model.fbt_state_proj.weight) * jax.nn.sigmoid(ref_norm(e) @ model.fbt_token_gate.weight))
    return jnp.where(fb_valid[..., None], ref_fbt_scaled(model, fused, x0), x0)


def ref_fbt_passes(model, tokens, mask, n):
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, _, _ = ref_trunk(model, tokens, mask)
    outs = [out]
    for _ in range(n - 1):
        out, _, _ = ref_trunk(model, tokens, mask, x0=ref_fbt_input(model, tokens, out, fb_valid))
        outs.append(out)
    return outs


def ref_fbt_exact(model, tokens, mask):
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, _, _ = ref_trunk(model, tokens, mask)
    top = jnp.zeros_like(out)
    for t in range(tokens.shape[1]):
        if t:
            out, _, _ = ref_trunk(model, tokens, mask, x0=ref_fbt_input(model, tokens, top, fb_valid))
        top = top.at[:, t].set(out[:, t])
    return top


def ref_src_trunk(model, tokens, mask, prev=None):
    """Marin's residual block written out (as Block.__call__), with the src delta before every block <= s."""
    layer_masks, schedule, _ = ref_masks(model, tokens, mask)
    x = model.embed_gated_norm(model.embed_norm(gm._embedding_gather(model.token_embed, tokens)))
    source = None
    for i, (block, lg) in enumerate(zip(model.stacked_blocks.unstacked(), schedule)):
        flag = jnp.asarray(lg)
        lm = layer_masks[i]
        seg = lm.segment_ids[0]
        if prev is not None and i <= model.src_block:
            x = x + model.src_alpha * ((prev @ model.src_down.weight[i]) @ model.src_up.weight[i])
        a = block.attn(block.attn_gated_norm(block.rms_attn(x)), lm, disable_rope=flag, is_global=flag)
        x = x + (block.sconv_attn(a, seg) if block.sconv_attn is not None else a)
        mlp_in = block.mlp_gated_norm(block.rms_mlp(x))
        tv = gm.token_validity_from_attention_mask(lm, batch_size=x.shape[0], sequence_length=x.shape[1])
        o, _ = block.mlp(mlp_in, tv)
        for se in block.shared or ():
            o = o + se(mlp_in, activation=gm.ActivationFunctionEnum.silu)
        x = x + (block.sconv_mlp(o, seg) if block.sconv_mlp is not None else o)
        if i == model.src_block:
            source = x
    return model.final_gated_norm(model.final_norm(x)), source


def ref_src_prev(source, fb_valid):
    return jnp.pad(ref_norm(source)[:, :-1], ((0, 0), (1, 0), (0, 0))) * fb_valid[..., None]


def ref_src_passes(model, tokens, mask, n):
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, source = ref_src_trunk(model, tokens, mask)
    outs = [out]
    for _ in range(n - 1):
        out, source = ref_src_trunk(model, tokens, mask, prev=ref_src_prev(source, fb_valid))
        outs.append(out)
    return outs


def ref_src_exact(model, tokens, mask):
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, source = ref_src_trunk(model, tokens, mask)
    exact_src, exact_out = jnp.zeros_like(source), jnp.zeros_like(out)
    for t in range(tokens.shape[1]):
        if t:
            out, source = ref_src_trunk(model, tokens, mask, prev=ref_src_prev(exact_src, fb_valid))
        exact_src = exact_src.at[:, t].set(source[:, t])
        exact_out = exact_out.at[:, t].set(out[:, t])
    return exact_out


def src_variant(base, key, *, src_block, passes=1, rank=4, randomize=True):
    model = am.SrcTransformer.from_base(base, key=key, src_block=src_block, src_rank=rank, passes=passes,
                                        eval_passes=passes, jitter=0.0, bos_id=BOS)
    if randomize:
        model = eqx.tree_at(lambda m: m.src_up.weight, model,
                            jax.random.normal(jax.random.fold_in(key, 13), model.src_up.weight.shape) / math.sqrt(rank))
    return model


def ref_stdres_trunk(model, tokens, mask, prev=None, x0=None):
    """Marin's residual block written out (as Block.__call__); with a previous pass (prev as ref_prev builds it) sublayer
    j reads (1 - rho_j) x + rho_j c n_j, n_j the previous outputs j..2L-1 under its own query (or their mean),
    c = rms(x) / rms(n_j); x0: the start of the stream in place of the embedded token (share "fbt")."""
    cfg = model.config
    eps_d = cfg.hidden_dim * am.ATTN_RES_EPS
    layer_masks, schedule, _ = ref_masks(model, tokens, mask)
    x = model.embed_gated_norm(model.embed_norm(gm._embedding_gather(model.token_embed, tokens))) if x0 is None else x0

    def read(j, x):
        if prev is None:
            return x
        mode, win = am.parse_depth(model.feedback_depth)
        if mode == "win":  # the fixed nearest window: outputs j .. j + k - 1, nothing above
            cand, valid = prev[0][j:j + win], prev[1]
        else:
            s0 = j if mode == "whole" else max(j, 2 * cfg.num_layers - 2)
            cand, valid = prev[0][s0:], prev[1]
        if getattr(model, "feedback_source", "outputs") == "top":
            cand = prev[0][j:j + 1]  # source top: sublayer j reads the previous final state through its own map M_j
        if model.feedback_query_kind == "uniform":
            n = sum(cand) / len(cand)
        else:
            w = jax.nn.softmax(jnp.stack([ref_scores(c, model.frontier_query.weight, eps_d)[..., j] for c in cand], -1), -1)
            n = sum(w[..., s:s + 1] * c for s, c in enumerate(cand))
        rho = (jax.nn.sigmoid(model.frontier_logit.weight[j]) * valid)[..., None]
        c = jnp.sqrt((jnp.sum(x * x, -1, keepdims=True) + eps_d) / (jnp.sum(n * n, -1, keepdims=True) + eps_d))
        return (1.0 - rho) * x + rho * c * n

    outs = []
    for i, (block, lg) in enumerate(zip(model.stacked_blocks.unstacked(), schedule)):
        flag = jnp.asarray(lg)
        lm = layer_masks[i]
        seg = lm.segment_ids[0]
        h = read(2 * i, x)
        a = block.attn(block.attn_gated_norm(block.rms_attn(h)), lm, disable_rope=flag, is_global=flag)
        a = block.sconv_attn(a, seg) if block.sconv_attn is not None else a
        x = x + a
        h = read(2 * i + 1, x)
        mlp_in = block.mlp_gated_norm(block.rms_mlp(h))
        tv = gm.token_validity_from_attention_mask(lm, batch_size=x.shape[0], sequence_length=x.shape[1])
        o, _ = block.mlp(mlp_in, tv)
        for se in block.shared or ():
            o = o + se(mlp_in, activation=gm.ActivationFunctionEnum.silu)
        o = block.sconv_mlp(o, seg) if block.sconv_mlp is not None else o
        x = x + o
        outs += [a, o]
    return model.final_gated_norm(model.final_norm(x)), outs


def ref_feedback_input(model, tokens, prev):
    """inject "input": sublayer 0's read (softmax of its query over every mapped previous output) through FBT's gate
    product in place of the token input."""
    cand, valid = prev
    eps_d = model.config.hidden_dim * am.ATTN_RES_EPS
    w = jax.nn.softmax(jnp.stack([ref_scores(c, model.frontier_query.weight, eps_d)[..., 0] for c in cand], -1), -1)
    n = sum(w[..., s:s + 1] * c for s, c in enumerate(cand))
    e = gm._embedding_gather(model.token_embed, tokens)
    x0 = model.embed_gated_norm(model.embed_norm(e))
    fused = ref_norm((n @ model.fbt_state_proj.weight) * jax.nn.sigmoid(ref_norm(e) @ model.fbt_token_gate.weight))
    return jnp.where(valid[..., None], ref_fbt_scaled(model, fused, x0), x0)


def ref_embed_inject(model, tokens, top, fb_valid):
    """embed_inject: u_t = norm(h_{t-1}) W_E / sqrt(D); "match" x0_t + rms(x0_t) u_t, "embed" EmbedNorms(e_t + rms(e_t) u_t) where
    position t has feedback, x0 elsewhere."""
    def rms(x):
        return jnp.sqrt(jnp.mean(x * x, -1, keepdims=True) + am.ATTN_RES_EPS)

    h = jnp.pad(top[:, :-1], ((0, 0), (1, 0), (0, 0)))
    e = gm._embedding_gather(model.token_embed, tokens)
    x0 = model.embed_gated_norm(model.embed_norm(e))
    if model.embed_inject in ("plain", "plainonly"):
        return jnp.where(fb_valid[..., None], x0 + model.embed_inject_scale * (ref_norm(h) @ model.embed_inject_proj.weight), x0)
    u = ref_norm(h) @ model.embed_inject_proj.weight / math.sqrt(h.shape[-1])
    if model.embed_inject == "embed":
        mixed = model.embed_gated_norm(model.embed_norm(e + rms(e) * u))
    else:
        mixed = x0 + rms(x0) * u
    return jnp.where(fb_valid[..., None], mixed, x0)


def ref_raw_top(model, tokens, outs, x0=None):
    """The residual stream after the last block: the start of the stream plus every sublayer output, in order."""
    x = model.embed_gated_norm(model.embed_norm(gm._embedding_gather(model.token_embed, tokens))) if x0 is None else x0
    for o in outs:
        x = x + o
    return x


def ref_stdres_next(model, tokens, fb_valid, top, outs, raw=None):
    """What the next pass reads: the previous outputs (share "pool") or the fused input (share "fbt"); the injection
    ablation: the sublayer feedback read through FBT's gate product (inject "input") or the previous final state
    in every slot (source "top")."""
    if model.feedback_share == "fbt":
        return {"x0": ref_fbt_input(model, tokens, top, fb_valid)}
    if getattr(model, "feedback_replaces_input", "residual") == "input":
        return {"x0": ref_feedback_input(model, tokens, ref_prev(model, outs, fb_valid))}
    if getattr(model, "feedback_source", "outputs") == "top":
        return {"prev": ref_prev(model, [top] * len(outs), fb_valid)}
    if getattr(model, "embed_inject", "none") in ("plain", "plainonly"):
        top = raw  # LoRM's top: the stream before the final norms
    if getattr(model, "embed_inject", "none") in ("only", "plainonly"):
        return {"x0": ref_embed_inject(model, tokens, top, fb_valid)}
    if getattr(model, "embed_inject", "none") != "none":
        return {"prev": ref_prev(model, outs, fb_valid), "x0": ref_embed_inject(model, tokens, top, fb_valid)}
    return {"prev": ref_prev(model, outs, fb_valid)}


def ref_stdres_passes(model, tokens, mask, n):
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, outs = ref_stdres_trunk(model, tokens, mask)
    raw = ref_raw_top(model, tokens, outs)
    res = [out]
    for _ in range(n - 1):
        nxt = ref_stdres_next(model, tokens, fb_valid, out, outs, raw=raw)
        out, outs = ref_stdres_trunk(model, tokens, mask, **nxt)
        raw = ref_raw_top(model, tokens, outs, nxt.get("x0"))
        res.append(out)
    return res


def ref_stdres_exact(model, tokens, mask):
    """Token-by-token recursion: position t reads the exact outputs (pool) or top state (fbt) of t - 1."""
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, outs = ref_stdres_trunk(model, tokens, mask)
    raw = ref_raw_top(model, tokens, outs)
    exact_outs, exact_top, exact_raw = [jnp.zeros_like(v) for v in outs], jnp.zeros_like(out), jnp.zeros_like(out)
    for t in range(tokens.shape[1]):
        if t:
            nxt = ref_stdres_next(model, tokens, fb_valid, exact_top, exact_outs, raw=exact_raw)
            out, outs = ref_stdres_trunk(model, tokens, mask, **nxt)
            raw = ref_raw_top(model, tokens, outs, nxt.get("x0"))
        exact_outs = [e.at[:, t].set(v[:, t]) for e, v in zip(exact_outs, outs)]
        exact_top = exact_top.at[:, t].set(out[:, t])
        exact_raw = exact_raw.at[:, t].set(raw[:, t])
    return exact_top


def ref_final_mix(model, tokens, outs, x0=None):
    """The AttnRes trunk's state after the last block and before the final norms, from the formula: the mixture of
    mixing point 2L, sum_s softmax_s(w_2L . v_s / sqrt(|v_s|^2 + D eps)) v_s over slot 0 (the embedded token, or x0 in
    its place) and the 2L sublayer outputs."""
    cfg = model.config
    n_mix, eps_d = 2 * cfg.num_layers, cfg.hidden_dim * am.ATTN_RES_EPS
    first = model.embed_gated_norm(model.embed_norm(gm._embedding_gather(model.token_embed, tokens))) if x0 is None else x0
    slots = [first, *outs]
    w = jax.nn.softmax(jnp.stack([ref_scores(v, model.attnres_query.weight, eps_d)[..., n_mix] for v in slots], -1), axis=-1)
    return sum(w[..., s:s + 1] * v for s, v in enumerate(slots))


def ref_attnres_embed_next(model, tokens, fb_valid, outs, mixed):
    """What the next pass of the AttnRes trunk reads under embed_inject plain / plainonly: slot 0 = x0_t + c *
    norm(h_{t-1}) W_E (ref_embed_inject on the previous final mixture) and, for plain, the previous slots."""
    nxt = {"x0": ref_embed_inject(model, tokens, mixed, fb_valid)}
    if model.embed_inject == "plain":
        nxt["prev"] = ref_prev(model, outs, fb_valid)
    return nxt


def ref_attnres_embed_passes(model, tokens, mask, n):
    _, _, fb_valid = ref_masks(model, tokens, mask)
    out, outs, _ = ref_trunk(model, tokens, mask)
    mixed = ref_final_mix(model, tokens, outs)
    res = [out]
    for _ in range(n - 1):
        nxt = ref_attnres_embed_next(model, tokens, fb_valid, outs, mixed)
        out, outs, _ = ref_trunk(model, tokens, mask, **nxt)
        mixed = ref_final_mix(model, tokens, outs, nxt["x0"])
        res.append(out)
    return res


def ref_attnres_embed_exact(model, tokens, mask, context=None, c=1):
    """Token-by-token recursion on the AttnRes trunk with embed_inject plain / plainonly: position t reads the exact
    slots and the exact final mixture of t - 1. With a prompt (context [B, T] bool, as ref_prefill_exact): the prompt
    goes through c Jacobi passes and keeps what it read in pass c (its slot 0 and its previous slots); c = 1: the
    prompt reads nothing."""
    _, _, fbv = ref_masks(model, tokens, mask)

    def step(reads):
        out, outs, _ = ref_trunk(model, tokens, mask, **reads)
        return out, outs, ref_final_mix(model, tokens, outs, reads.get("x0"))

    out, outs, mixed = step({})
    hist = [(outs, mixed)]
    for _ in range(c - 1):
        out, outs, mixed = step(ref_attnres_embed_next(model, tokens, fbv, outs, mixed))
        hist.append((outs, mixed))
    held = None if c == 1 else ref_attnres_embed_next(model, tokens, fbv, *hist[c - 2])
    read_valid = fbv if context is None or c > 1 else fbv & ~context
    exact_top, exact_outs, exact_mixed = jnp.zeros_like(out), [jnp.zeros_like(v) for v in outs], jnp.zeros_like(mixed)
    for t in range(tokens.shape[1]):
        if t:
            fresh = ref_attnres_embed_next(model, tokens, read_valid, exact_outs, exact_mixed)
            out, outs, mixed = step(fresh if held is None else pick_reads(context, held, fresh))
        elif held is not None:
            out, outs, mixed = step(held)
        exact_top = exact_top.at[:, t].set(out[:, t])
        exact_outs = [e.at[:, t].set(v[:, t]) for e, v in zip(exact_outs, outs)]
        exact_mixed = exact_mixed.at[:, t].set(mixed[:, t])
    return exact_top


def stdres_variant(base, key, *, share="pool", query="own", passes=1, proj="slot", proj_rank=4, randomize=True,
                   fbt_input_norm="unit", depth="whole", inject="residual", source="outputs", embed_inject="none",
                   embed_scale=0.5):
    model = am.StdResTransformer.from_base(base, key=key, share=share, query=query, passes=passes, eval_passes=passes,
                                           jitter=0.0, bos_id=BOS, proj=proj, proj_rank=proj_rank,
                                           fbt_input_norm=fbt_input_norm, depth=depth, inject=inject, source=source,
                                           embed_inject=embed_inject)
    if not randomize:
        return model
    ks = jax.random.split(jax.random.fold_in(key, 17), 3)
    d = base.config.hidden_dim
    if model.embed_inject_proj is not None:
        model = eqx.tree_at(lambda m: m.embed_inject_proj.weight, model,
                            embed_scale * jax.random.normal(jax.random.fold_in(key, 23), (d, d)))
    if model.frontier_logit is not None:
        model = eqx.tree_at(lambda m: m.frontier_logit.weight, model,
                            jax.random.uniform(ks[0], model.frontier_logit.weight.shape, minval=-2.0, maxval=0.0))
    if model.frontier_query is not None:
        model = eqx.tree_at(lambda m: m.frontier_query.weight, model,
                            2.0 * jax.random.normal(ks[1], model.frontier_query.weight.shape) / math.sqrt(d))
    if model.frontier_proj_up is not None:
        model = eqx.tree_at(lambda m: m.frontier_proj_up.weight, model,
                            jax.random.normal(ks[2], model.frontier_proj_up.weight.shape) / math.sqrt(proj_rank))
    if model.frontier_proj_full is not None:
        model = eqx.tree_at(lambda m: m.frontier_proj_full.weight, model,
                            0.5 * jax.random.normal(ks[2], model.frontier_proj_full.weight.shape) / math.sqrt(d))
    return model


def causality(fwd, tokens, mask, *, pos: int, split: int, label: str, tol: float) -> None:
    base = np.asarray(fwd(tokens, mask), np.float32)
    bumped = np.asarray(fwd(tokens.at[0, pos].set((tokens[0, pos] + 1) % 128 + 2), mask), np.float32)
    delta = np.abs(bumped - base).max(axis=-1)
    same = max(delta[0, :split].max(), delta[0, split:pos].max() if pos > split else 0.0, delta[1].max())
    moved = delta[0, pos:].max()
    check(same <= tol and moved > 100 * max(tol, 1e-6),
          f"{label}: token {pos} changed -> other document / earlier positions / other row max |d|={same:.3g} "
          f"(tol {tol:g}), same and later positions max |d|={moved:.3g}")


def tiny_checks() -> None:
    # fp32 matmuls at full precision: with the default, a GPU may run separately compiled instances of the same matmul
    # with different TF32 algorithms, which alone moves the outputs by ~1e-4 and would swamp the tolerances below.
    with jax.default_matmul_precision("highest"):
        _tiny_checks()


def _tiny_checks() -> None:
    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261106), 2)
    split, seq = 7, cfg.max_seq_len
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)

        if START <= 4:
            # 1. one pass == reference
            model = variant(base, key_model)
            hidden, metrics = eqx.filter_jit(lambda m, t: m(t, mask=mask))(model, tokens)
            ref_h, _, ref_stats = eqx.filter_jit(lambda m, t: ref_trunk(m, t, mask))(model, tokens)
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs, 0), *ref_stats)
            ref_metrics = eqx.filter_jit(lambda m, s: m.router_metrics([s]))(model, stacked)
            d_h, d_m = max_diff(hidden, ref_h), max(max_diff(metrics[k], ref_metrics[k]) for k in ref_metrics)
            base_h, base_m = eqx.filter_jit(lambda m, t: m(t, mask=mask))(base, tokens)
            check(d_h <= 1e-5 and d_m <= 1e-5 and set(metrics) == set(base_m) and max_diff(hidden, base_h) > 1e-3,
                  f"tiny 1. one pass == reference: hidden max |d|={d_h:.3g}, router metrics max |d|={d_m:.3g}; "
                  f"differs from Marin's residual trunk by {max_diff(hidden, base_h):.3g}; same metric keys")

            # 2. two passes, every variant
            variants = [("joint", "shared", 0), ("joint", "shared", 4), ("pool", "shared", 0), ("pool", "own", 0),
                        ("pool", "uniform", 0), ("pool", "own", 4)]
            for share, query, rank in variants:
                model = variant(base, key_model, share=share, query=query, rank=rank, passes=2)
                outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
                refs = eqx.filter_jit(lambda m, t: ref_passes(m, t, mask, 2))(model, tokens)
                d1, d2 = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1])
                moved = max_diff(outs[1], outs[0])
                check(d1 <= 1e-5 and d2 <= 1e-5 and moved > 1e-3,
                      f"tiny 2. {share}/{query}/rank {rank}: pass 1 max |d|={d1:.3g}, pass 2 max |d|={d2:.3g} vs reference; "
                      f"pass 2 moved {moved:.3g} from pass 1")

            # 3. zero share
            for share in ("joint", "pool"):
                model = variant(base, key_model, share=share, passes=2)
                leaf = (lambda m: m.frontier_entry_bias.weight) if share == "joint" else (lambda m: m.frontier_logit.weight)
                model = eqx.tree_at(leaf, model, jnp.full_like(leaf(model), -1e4))
                outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
                check(max_diff(outs[0], outs[1]) <= 1e-6, f"tiny 3. {share}: zero share -> pass 2 == pass 1 (AttnRes alone), "
                      f"max |d|={max_diff(outs[0], outs[1]):.3g}")

            # 4. causality (two and three passes)
            for share, query in (("joint", "shared"), ("pool", "own")):
                model = variant(base, key_model, share=share, query=query, rank=4, passes=3)
                fwd = eqx.filter_jit(lambda t, m, model=model: model.run_passes(t, m, 3, train=False)[0][-1])
                causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 4. {share}/{query} 3 passes", tol=0.0)
                causality(fwd, tokens, mask, pos=split, split=split, label=f"tiny 4. {share}/{query} 3 passes, doc start", tol=0.0)

        if START <= 5:
            # 5. K = T Jacobi == exact recursion (8 tokens: two documents of 4)
            short = 8
            tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
            for share, query in (("joint", "shared"), ("pool", "own")):
                model = variant(base, key_model, share=share, query=query, rank=4, passes=short)
                jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model, tok8)
                exact = eqx.filter_jit(lambda m, t: ref_exact(m, t, mask8))(model, tok8)
                two = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 2, train=False)[0][-1])(model, tok8)
                check(max_diff(jac, exact) <= 1e-5 and max_diff(two, exact) > 1e-4,
                      f"tiny 5. {share}/{query}: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g}); "
                      f"2 passes differ by {max_diff(two, exact):.3g}")

        # 6. gradients and the training loss
        def loss_of(m):
            return m.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                     return_router_metrics=True)

        for share, query, rank in (("joint", "shared", 0), ("pool", "own", 0), ("pool", "own", 4)):
            dormant = variant(base, key_model, share=share, query=query, rank=rank, passes=1, randomize=False)
            (_, _), g1 = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(dormant)
            feedback_g = [x for x in (g1.frontier_logit, g1.frontier_entry_bias, g1.frontier_query, g1.frontier_down,
                                      g1.frontier_up) if x is not None]
            zero = all(float(jnp.max(jnp.abs(x.weight))) == 0.0 for x in feedback_g)
            fresh = variant(base, key_model, share=share, query=query, rank=rank, passes=2, randomize=False)
            (loss, met), g2 = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(fresh)
            # with a zero-initialized up projection the mapped previous slots are zero, so frontier_down and an own
            # query get no first-step gradient by construction; every other new parameter must have one
            blind = ("frontier_down", "frontier_query") if rank else ()
            new_g = {n: float(jnp.max(jnp.abs(getattr(g2, n).weight))) for n in am.NEW_PARAM_FIELDS
                     if getattr(g2, n, None) is not None and n not in blind}
            blind_g = {n: float(jnp.max(jnp.abs(getattr(g2, n).weight))) for n in blind if getattr(g2, n) is not None}
            zero = zero and all(v == 0.0 for v in blind_g.values())
            ce1, ce2 = met["train/cross_entropy_loss_pass1"], met["train/cross_entropy_loss_pass2"]
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
            check(zero and all(v > 0 for v in new_g.values()) and abs(float(loss) - float(ce1 + ce2)) <= 1e-5 and finite,
                  f"tiny 6. {share}/{query}/rank {rank}: one pass -> feedback gradients all zero ({zero}); two passes -> "
                  f"new-parameter max |g| {new_g}; loss {float(loss):.5f} = CE1 {float(ce1):.5f} + CE2 {float(ce2):.5f}; finite={finite}")
        model3 = variant(base, key_model, share="pool", query="own", rank=4, passes=3)
        (loss3, met3) = eqx.filter_jit(loss_of)(model3)
        c = [float(met3[f"train/cross_entropy_loss_pass{k}"]) for k in (1, 2, 3)]
        check(abs(float(loss3) - (c[0] + 0.5 * (c[1] + c[2]))) <= 1e-5,
              f"tiny 6. three passes: loss {float(loss3):.5f} = CE1 + mean(CE2, CE3) = {c[0] + 0.5 * (c[1] + c[2]):.5f}")
        jit_model = variant(base, key_model, share="pool", query="own", rank=4, passes=2, jitter=0.5)
        l_noisy = float(eqx.filter_jit(loss_of)(jit_model)[0])
        l_clean = float(eqx.filter_jit(loss_of)(dataclasses.replace(jit_model, jitter=0.0))[0])
        e_noisy = float(eqx.filter_jit(lambda m: m.next_token_loss(tokens, loss_weight, mask=mask, reduction="none").sum())(jit_model))
        clean_loss = eqx.filter_jit(
            lambda m: m.next_token_loss(tokens, loss_weight, mask=mask, reduction="none").sum())
        e_clean = float(clean_loss(dataclasses.replace(jit_model, jitter=0.0)))
        check(abs(l_noisy - l_clean) > 1e-6 and e_noisy == e_clean,
              f"tiny 6. jitter acts in training only: train loss {l_noisy:.6f} vs {l_clean:.6f} without; eval sums equal")

        # 7. optimizer groups
        from alf.marin.recipe import optimizer_config

        for share, query, rank in (("joint", "shared", 0), ("pool", "own", 4)):
            model = variant(base, key_model, share=share, query=query, rank=rank, passes=2, randomize=False)
            groups = am.optimizer_groups(model, optimizer_config(cfg, num_train_steps=100, batch_size=8))
            check(groups and all(v == "adam" for v in groups.values()),
                  f"tiny 7. {share}/{query}/rank {rank}: new parameters -> {groups}")

        # 8. install() + calibration
        import experiments.grug.moe_hero_ep.train as grug_train

        drop, flops = grug_train._drop_metrics, grug_train._compute_flops
        am.install(passes=2, eval_passes=4, share="joint", query="shared", rank=0, jitter=0.02, bos_id=BOS)
        built = grug_train.Transformer.init(cfg, key=key_model)
        same_backbone = max_diff([getattr(built, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"],
                                 [getattr(base, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"])
        check(isinstance(built, am.AttnResTransformer) and built.passes == 2 and built.eval_passes == 4 and same_backbone == 0.0,
              f"tiny 8. install(): Transformer.init -> AttnResTransformer(passes 2, eval 4), backbone == Marin init (|d|={same_backbone})")
        n = 2 * cfg.num_layers
        per = 2 * seq * cfg.num_experts_per_token
        zeros = np.zeros(n, np.int32)
        out = grug_train._drop_metrics(zeros, zeros, zeros, zeros, np.full(n, per, np.int32), batch_size=2,
                                       sequence_length=seq, top_k=cfg.num_experts_per_token, num_layers=cfg.num_layers)
        f_two = flops(model_config=dataclasses.replace(cfg, num_layers=n))[0]
        check(out["moe/drop_fraction"] == 0.0 and grug_train._compute_flops(model_config=cfg)[0] == f_two,
              "tiny 8. install(): assignment accounting and analytic FLOPs count passes x layers")
        grug_train.Transformer, grug_train._drop_metrics, grug_train._compute_flops = gm.Transformer, drop, flops
        model = variant(base, key_model, share="joint", passes=2)
        bias = eqx.filter_jit(lambda m, t: am.calibrate_joint_bias(m, t, mask, target=0.01))(model, tokens)
        calibrated = eqx.tree_at(lambda m: m.frontier_entry_bias.weight, model, bias)
        share_now = eqx.filter_jit(lambda m, t: joint_share(m, t, mask))(calibrated, tokens)
        check(abs(float(share_now) - 0.01) < 1e-4, f"tiny 8. calibration: mean previous-slot share {float(share_now):.6f} (target 0.01)")
        # 8b. install(entry_bias=...) puts the calibrated biases into the initialization (a joint branch restores
        # everything else from its base, partially: the base has no entry-bias leaf); refused for the pool share
        values = [float(b) for b in np.asarray(bias)]
        am.install(passes=2, eval_passes=2, share="joint", query="shared", rank=0, bos_id=BOS, entry_bias=values)
        built = grug_train.Transformer.init(cfg, key=key_model)
        grug_train.Transformer, grug_train._drop_metrics, grug_train._compute_flops = gm.Transformer, drop, flops
        am._SPEC["entry_bias"] = None
        refused = False
        try:
            am.AttnResTransformer.from_base(base, key=key_model, share="pool", entry_bias=values)
        except ValueError:
            refused = True
        check(built.frontier_logit is None and max_diff(built.frontier_entry_bias.weight, bias) == 0.0 and refused,
              f"tiny 8b. install(entry_bias=...): Transformer.init carries the calibrated biases "
              f"(max |d|={max_diff(built.frontier_entry_bias.weight, bias):.3g}), no pool logits; pool + entry_bias "
              f"refused ({refused})")


def inject_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _inject_checks()


def _inject_checks() -> None:
    """Group 9: share "inject" (one projected source per layer from the pool read, as in the Full-bandwidth
    Transformer, arXiv:2608.08888)."""
    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261106), 2)
    split, seq = 7, cfg.max_seq_len
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        # 9a. two passes == reference (each pool scoring), and a pool run carrying dormant inject maps is unchanged
        for share, query in (("inject", "shared"), ("inject", "own"), ("inject", "uniform"), ("pool", "shared")):
            model = variant(base, key_model, share=share, query=query, passes=2, inject_rank=4)
            outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_passes(m, t, mask, 2))(model, tokens)
            d1, d2, moved = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1]), max_diff(outs[1], outs[0])
            check(d1 <= 1e-5 and d2 <= 1e-5 and moved > 1e-3,
                  f"tiny 9a. {share}/{query} (inject maps rank 4): pass 1 max |d|={d1:.3g}, pass 2 max |d|={d2:.3g} vs "
                  f"reference; pass 2 moved {moved:.3g}")
        # 9b. causality (three passes) and K = T Jacobi == exact recursion
        model = variant(base, key_model, share="inject", passes=3, inject_rank=4)
        fwd = eqx.filter_jit(lambda t, m, model=model: model.run_passes(t, m, 3, train=False)[0][-1])
        causality(fwd, tokens, mask, pos=11, split=split, label="tiny 9b. inject 3 passes", tol=0.0)
        causality(fwd, tokens, mask, pos=split, split=split, label="tiny 9b. inject 3 passes, doc start", tol=0.0)
        short = 8
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        model = variant(base, key_model, share="inject", passes=short, inject_rank=4)
        jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model, tok8)
        exact = eqx.filter_jit(lambda m, t: ref_exact(m, t, mask8))(model, tok8)
        two = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 2, train=False)[0][-1])(model, tok8)
        check(max_diff(jac, exact) <= 1e-5 and max_diff(two, exact) > 1e-4,
              f"tiny 9b. inject: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g}); "
              f"2 passes differ by {max_diff(two, exact):.3g}")

        # 9c. zero maps: pass 2 == pass 1; gradients (dormant in a pool run, first step of an inject run)
        def loss_of(m):
            return m.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                     return_router_metrics=True)

        fresh = variant(base, key_model, share="inject", passes=2, inject_rank=4, randomize=False)
        outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(fresh, tokens)
        (loss, met), g = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(fresh)
        up_g, down_g = max_abs(g.frontier_inject_up.weight), max_abs(g.frontier_inject_down.weight)
        logit_g = max_abs(g.frontier_logit.weight)
        per_layer = [float(jnp.max(jnp.abs(g.frontier_inject_up.weight[i]))) for i in range(cfg.num_layers)]
        check(max_diff(outs[0], outs[1]) <= 1e-6 and min(per_layer) > 0 and down_g == 0.0 and logit_g == 0.0,
              f"tiny 9c. inject at zero maps: pass 2 == pass 1 (max |d|={max_diff(outs[0], outs[1]):.3g}); first-step "
              f"|g| up per layer {['%.3g' % v for v in per_layer]}, down {down_g:.3g}, share logits {logit_g:.3g}")
        pool = variant(base, key_model, share="pool", passes=2, inject_rank=4, randomize=False)
        (_, _), gp = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(pool)
        dormant = max(max_abs(gp.frontier_inject_up.weight), max_abs(gp.frontier_inject_down.weight))
        check(dormant == 0.0 and max_abs(gp.frontier_logit.weight) > 0,
              f"tiny 9c. pool run with inject maps: maps dormant (max |g|={dormant:.3g}), share logits trained")
        # 9d. optimizer groups
        from alf.marin.recipe import optimizer_config

        groups = am.optimizer_groups(fresh, optimizer_config(cfg, num_train_steps=100, batch_size=8))
        check(groups and all(v == "adam" for v in groups.values())
              and any(k.startswith("frontier_inject_up") for k in groups),
              f"tiny 9d. inject: new parameters -> {groups}")


def proj_checks() -> None:
    with jax.default_matmul_precision("highest"):
        for kind in ("source", "slot"):
            _proj_checks(kind)


def _proj_checks(kind: str = "source") -> None:
    """Group 10 (run for kind "source" and "slot"; "slot" = one map per previous slot): proj "source", one map per
    source layer (M_i v = v + Delta_i v on the previous token's two slots of layer i, applied once, as in the
    Full-bandwidth Transformer, arXiv:2608.08888), with two softmaxes (pool + own query) as in the final method."""
    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261106), 2)
    split, seq = 7, cfg.max_seq_len
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
        # 10a. two passes == reference
        for share, query, r in (("pool", "own", 4), ("pool", "own", 0), ("pool", "shared", 4), ("joint", "shared", 4)):
            model = variant(base, key_model, share=share, query=query, passes=2, proj=kind, proj_rank=r)
            outs = run2(model, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_passes(m, t, mask, 2))(model, tokens)
            d1, d2, moved = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1]), max_diff(outs[1], outs[0])
            check(d1 <= 1e-5 and d2 <= 1e-5 and moved > 1e-3,
                  f"tiny 10a. {share}/{query}/{kind} rank {r or 'full'}: pass 1 max |d|={d1:.3g}, pass 2 max |d|={d2:.3g} "
                  f"vs reference; pass 2 moved {moved:.3g}")
        # 10b. zero Delta == the unmapped read; 10c. zero share == AttnRes alone
        for r in (4, 0):
            model = variant(base, key_model, share="pool", query="own", passes=2, proj=kind, proj_rank=r)
            plain = variant(base, key_model, share="pool", query="own", passes=2)
            if r:
                zeroed = eqx.tree_at(lambda m: m.frontier_proj_up.weight, model, jnp.zeros_like(model.frontier_proj_up.weight))
            else:
                zeroed = eqx.tree_at(lambda m: m.frontier_proj_full.weight, model, jnp.zeros_like(model.frontier_proj_full.weight))
            d0 = max_diff(run2(zeroed, tokens), run2(plain, tokens))
            mapped = max_diff(run2(model, tokens)[1], run2(plain, tokens)[1])
            off = eqx.tree_at(lambda m: m.frontier_logit.weight, model, jnp.full_like(model.frontier_logit.weight, -1e4))
            o = run2(off, tokens)
            check(d0 <= 1e-6 and mapped > 1e-3 and max_diff(o[0], o[1]) <= 1e-6,
                  f"tiny 10b/c. pool/own/{kind} rank {r or 'full'}: zero Delta == unmapped read (max |d|={d0:.3g}, the maps "
                  f"move pass 2 by {mapped:.3g}); zero share -> pass 2 == pass 1 (max |d|={max_diff(o[0], o[1]):.3g})")
        # 10d. causality and K = T Jacobi == exact recursion
        model = variant(base, key_model, share="pool", query="own", passes=3, proj=kind, proj_rank=4)
        fwd = eqx.filter_jit(lambda t, m, model=model: model.run_passes(t, m, 3, train=False)[0][-1])
        causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 10d. pool/own/{kind} 3 passes", tol=0.0)
        causality(fwd, tokens, mask, pos=split, split=split, label=f"tiny 10d. pool/own/{kind} 3 passes, doc start", tol=0.0)
        short = 8
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        model = variant(base, key_model, share="pool", query="own", passes=short, proj=kind, proj_rank=4)
        jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model, tok8)
        exact = eqx.filter_jit(lambda m, t: ref_exact(m, t, mask8))(model, tok8)
        two = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 2, train=False)[0][-1])(model, tok8)
        check(max_diff(jac, exact) <= 1e-5 and max_diff(two, exact) > 1e-4,
              f"tiny 10d. pool/own/{kind}: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g}); "
              f"2 passes differ by {max_diff(two, exact):.3g}")

        # 10e. gradients: dormant with one pass; first two-pass step at the initialization (up / full zero)
        def loss_of(m):
            return m.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                     return_router_metrics=True)

        for r in (4, 0):
            names = ("frontier_proj_down", "frontier_proj_up") if r else ("frontier_proj_full",)
            dormant = variant(base, key_model, share="pool", query="own", passes=1, proj=kind, proj_rank=r, randomize=False)
            (_, _), g1 = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(dormant)
            zero = all(max_abs(getattr(g1, n).weight) == 0.0 for n in names + ("frontier_logit", "frontier_query"))
            fresh = variant(base, key_model, share="pool", query="own", passes=2, proj=kind, proj_rank=r, randomize=False)
            (_, _), g2 = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(fresh)
            trained = "frontier_proj_up" if r else "frontier_proj_full"
            per_layer = [float(jnp.max(jnp.abs(getattr(g2, trained).weight[i]))) for i in range(getattr(g2, trained).weight.shape[0])]
            down_g = max_abs(g2.frontier_proj_down.weight) if r else 0.0
            q_g, l_g = max_abs(g2.frontier_query.weight), max_abs(g2.frontier_logit.weight)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
            check(zero and min(per_layer) > 0 and down_g == 0.0 and q_g > 0 and l_g > 0 and finite,
                  f"tiny 10e. pool/own/{kind} rank {r or 'full'}: one pass -> map / share / query gradients zero ({zero}); "
                  f"first two-pass step |g| {trained} per layer {['%.3g' % v for v in per_layer]}, down {down_g:.3g} "
                  f"(zero while up is zero), own query {q_g:.3g}, share logits {l_g:.3g}; finite={finite}")
        # 10f. optimizer groups
        from alf.marin.recipe import optimizer_config

        for r in (4, 0):
            model = variant(base, key_model, share="pool", query="own", passes=2, proj=kind, proj_rank=r, randomize=False)
            groups = am.optimizer_groups(model, optimizer_config(cfg, num_train_steps=100, batch_size=8))
            # The prefix is a parameter leaf path, which keeps its on-disk spelling (see alf.model's frozen names).
            check(groups and all(v == "adam" for v in groups.values()) and any(k.startswith("frontier_proj_") for k in groups),
                  f"tiny 10f. pool/own/{kind} rank {r or 'full'}: new parameters -> {groups}")

        # 10g. branching the final method from an AttnRes base built with pool / shared query (no second query, no
        # source maps): a partial restore fills the missing leaves from the template and keeps every saved leaf
        import tempfile

        from levanter.checkpoint import load_checkpoint, save_checkpoint

        saved = variant(base, key_model, share="pool", query="shared", passes=1, inject_rank=4)
        template = variant(base, key_model, share="pool", query="own", passes=2, inject_rank=4, proj=kind,
                           proj_rank=4, randomize=False)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint({"model": saved}, step=1, checkpoint_path=tmp, is_temporary=False)
            strict_refused = False
            try:
                load_checkpoint({"model": template}, tmp, allow_partial=False)
            except Exception:
                strict_refused = True
            restored = load_checkpoint({"model": template}, tmp, allow_partial=True)["model"]
        kept = max_diff([getattr(restored, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
                        + [restored.attnres_query, restored.frontier_logit, restored.frontier_inject_down],
                        [getattr(saved, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
                        + [saved.attnres_query, saved.frontier_logit, saved.frontier_inject_down])
        filled = max_diff([restored.frontier_query, restored.frontier_proj_down, restored.frontier_proj_up],
                          [template.frontier_query, template.frontier_proj_down, template.frontier_proj_up])
        check(strict_refused and kept == 0.0 and filled == 0.0 and max_abs(restored.frontier_proj_up.weight) == 0.0,
              f"tiny 10g. branch from a pool / shared base: strict restore refused ({strict_refused}); partial restore keeps "
              f"every saved leaf (max |d|={kept:.3g}) and takes the second query and the source maps from the template "
              f"(max |d|={filled:.3g}, up zero)")

        # 10h. the one-softmax ablation branched from the same base: calibrate the entry biases on the restored model,
        # then a template carrying them restores everything else; the base's dormant pool logits are not read
        def joint_template(entry_bias=None):
            return variant(base, key_model, share="joint", query="shared", passes=2, inject_rank=4, proj=kind,
                           proj_rank=4, randomize=False, entry_bias=entry_bias)

        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint({"model": saved}, step=1, checkpoint_path=tmp, is_temporary=False)
            first = load_checkpoint({"model": joint_template()}, tmp, allow_partial=True)["model"]
            bias = eqx.filter_jit(lambda m, t: am.calibrate_joint_bias(m, t, mask, target=0.01))(first, tokens)
            branched = load_checkpoint({"model": joint_template(tuple(float(b) for b in np.asarray(bias)))}, tmp,
                                       allow_partial=True)["model"]
        kept = max_diff([getattr(branched, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
                        + [branched.attnres_query, branched.frontier_inject_down],
                        [getattr(saved, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
                        + [saved.attnres_query, saved.frontier_inject_down])
        bias_kept = max_diff(branched.frontier_entry_bias.weight, bias)
        share_now = float(eqx.filter_jit(lambda m, t: joint_share(m, t, mask))(branched, tokens))
        pass1 = max_diff(run2(branched, tokens)[0],
                         eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 1, train=False)[0][0])(saved, tokens))
        check(kept == 0.0 and bias_kept == 0.0 and branched.frontier_logit is None and abs(share_now - 0.01) < 1e-4
              and max_abs(branched.frontier_proj_up.weight) == 0.0 and pass1 <= 1e-6,
              f"tiny 10h. one softmax branched from a pool / shared base: saved leaves kept (max |d|={kept:.3g}), "
              f"calibrated entry biases kept (max |d|={bias_kept:.3g}), previous-slot share {share_now:.6f} (target "
              f"0.01), source maps from the template (up zero), pass 1 == the base (max |d|={pass1:.3g})")


def depth_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _depth_checks()


def _depth_checks() -> None:
    """Group 13 (the feedback-depth ablation): depth "top" reads only the previous token's last layer (slots 2L-1, 2L)
    instead of all deeper slots, with everything else as in the final method (pool + own query + one map per previous
    slot)."""
    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261110), 2)
    split, seq = 7, cfg.max_seq_len
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
        # 13a. two passes == the reference with the top window; 13b. top differs from whole in pass 2 only
        for r in (4, 0):
            top = variant(base, key_model, share="pool", query="own", passes=2, proj="slot", proj_rank=r, depth="top")
            whole = variant(base, key_model, share="pool", query="own", passes=2, proj="slot", proj_rank=r)
            outs = run2(top, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_passes(m, t, mask, 2))(top, tokens)
            d1, d2 = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1])
            wo = run2(whole, tokens)
            same1, apart2 = max_diff(outs[0], wo[0]), max_diff(outs[1], wo[1])
            check(d1 <= 1e-5 and d2 <= 1e-5 and same1 <= 1e-6 and apart2 > 1e-3,
                  f"tiny 13a/b. pool/own/slot rank {r or 'full'} depth top: pass 1 max |d|={d1:.3g}, pass 2 max |d|={d2:.3g} "
                  f"vs reference; pass 1 == depth whole (max |d|={same1:.3g}), pass 2 differs by {apart2:.3g}")
        # 13c. validation: depth top needs share pool
        try:
            variant(base, key_model, share="joint", query="shared", passes=2, depth="top", randomize=False)
            rejected = False
        except ValueError:
            rejected = True
        check(rejected, "tiny 13c. depth top with share joint is rejected")
        # 13d. causality and K = T Jacobi == exact recursion with the top window
        model = variant(base, key_model, share="pool", query="own", passes=3, proj="slot", proj_rank=4, depth="top")
        fwd = eqx.filter_jit(lambda t, m, model=model: model.run_passes(t, m, 3, train=False)[0][-1])
        causality(fwd, tokens, mask, pos=11, split=split, label="tiny 13d. pool/own/slot depth top 3 passes", tol=0.0)
        causality(fwd, tokens, mask, pos=split, split=split, label="tiny 13d. pool/own/slot depth top 3 passes, doc start", tol=0.0)
        short = 8
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        model = variant(base, key_model, share="pool", query="own", passes=short, proj="slot", proj_rank=4, depth="top")
        jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model, tok8)
        exact = eqx.filter_jit(lambda m, t: ref_exact(m, t, mask8))(model, tok8)
        check(max_diff(jac, exact) <= 1e-5,
              f"tiny 13d. pool/own/slot depth top: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g})")


def baseline_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _baseline_checks()


def _baseline_checks() -> None:
    """Groups 11 / 12: the two d768 comparison variants (share "fbt" on AttnRes, and variant src)."""
    import tempfile

    from levanter.checkpoint import load_checkpoint, save_checkpoint
    from alf.marin.recipe import optimizer_config

    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261106), 2)
    split, seq = 7, cfg.max_seq_len
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        short = 8
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))

        def grads(m, tok=tokens, msk=mask):
            def loss_of(mm):
                return mm.next_token_loss(tok, loss_weight, mask=msk, reduction="mean", logsumexp_weight=1e-4,
                                          return_router_metrics=True)
            (loss, _), g = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(m)
            return loss, g

        # 11. share "fbt" (AttnRes + FBT's layer-0 gate_product fusion)
        model = variant(base, key_model, share="fbt", passes=2, inject_rank=4)
        outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
        refs = eqx.filter_jit(lambda m, t: ref_fbt_passes(m, t, mask, 2))(model, tokens)
        plain = eqx.filter_jit(lambda m, t: ref_trunk(m, t, mask)[0])(model, tokens)
        d1, d2, moved = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1]), max_diff(outs[1], outs[0])
        check(d1 <= 1e-5 and d2 <= 1e-5 and max_diff(outs[0], plain) <= 1e-5 and moved > 1e-3,
              f"tiny 11a. fbt: pass 1 == the AttnRes trunk and the reference (max |d|={d1:.3g}), pass 2 == reference "
              f"(max |d|={d2:.3g}); pass 2 moved {moved:.3g}")
        model3 = variant(base, key_model, share="fbt", passes=3, inject_rank=4)
        fwd = eqx.filter_jit(lambda t, m, model=model3: model.run_passes(t, m, 3, train=False)[0][-1])
        causality(fwd, tokens, mask, pos=11, split=split, label="tiny 11b. fbt 3 passes", tol=0.0)
        causality(fwd, tokens, mask, pos=split, split=split, label="tiny 11b. fbt 3 passes, doc start", tol=0.0)
        model8 = variant(base, key_model, share="fbt", passes=short, inject_rank=4)
        jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
        exact = eqx.filter_jit(lambda m, t: ref_fbt_exact(m, t, mask8))(model8, tok8)
        two = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 2, train=False)[0][-1])(model8, tok8)
        check(max_diff(jac, exact) <= 1e-5 and max_diff(two, exact) > 1e-4,
              f"tiny 11b. fbt: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g}); "
              f"2 passes differ by {max_diff(two, exact):.3g}")
        _, g1 = grads(variant(base, key_model, share="fbt", passes=1, inject_rank=4, randomize=False))
        _, g2 = grads(variant(base, key_model, share="fbt", passes=2, inject_rank=4, randomize=False))
        dormant = max(max_abs(g1.fbt_state_proj.weight), max_abs(g1.fbt_token_gate.weight))
        live = min(max_abs(g2.fbt_state_proj.weight), max_abs(g2.fbt_token_gate.weight))
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
        groups = am.optimizer_groups(variant(base, key_model, share="fbt", passes=2, randomize=False),
                                     optimizer_config(cfg, num_train_steps=100, batch_size=8))
        check(dormant == 0.0 and live > 0 and finite and groups and all(v == "adam" for v in groups.values())
              and any(k.startswith("fbt_") for k in groups),
              f"tiny 11c. fbt: one pass -> fusion gradients zero ({dormant:.3g}); two passes -> min |g| {live:.3g}, "
              f"finite={finite}; new parameters -> {groups}")
        saved = variant(base, key_model, share="pool", query="shared", passes=1, inject_rank=4)
        template = variant(base, key_model, share="fbt", passes=2, inject_rank=4, randomize=False)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint({"model": saved}, step=1, checkpoint_path=tmp, is_temporary=False)
            restored = load_checkpoint({"model": template}, tmp, allow_partial=True)["model"]
        kept = max_diff([getattr(restored, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
                        + [restored.attnres_query, restored.frontier_inject_down],
                        [getattr(saved, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
                        + [saved.attnres_query, saved.frontier_inject_down])
        filled = max_diff([restored.fbt_state_proj, restored.fbt_token_gate], [template.fbt_state_proj, template.fbt_token_gate])
        pass1 = max_diff(eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 1, train=False)[0][0])(restored, tokens),
                         eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 1, train=False)[0][0])(saved, tokens))
        check(kept == 0.0 and filled == 0.0 and restored.frontier_logit is None and pass1 <= 1e-6,
              f"tiny 11d. fbt branched from a pool / shared base: saved leaves kept (max |d|={kept:.3g}), fusion "
              f"matrices from the template (max |d|={filled:.3g}), pass 1 == the base (max |d|={pass1:.3g})")

        # 12. variant src (Marin's residual trunk + one fixed source); source block 1 of 2 feeds blocks 0 and 1
        s = cfg.num_layers - 1
        plain_marin = eqx.filter_jit(lambda m, t: m(t, mask)[0])(base, tokens)
        model = src_variant(base, key_model, src_block=s, passes=2)
        outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
        refs = eqx.filter_jit(lambda m, t: ref_src_passes(m, t, mask, 2))(model, tokens)
        d0 = max_diff(outs[0], plain_marin)
        d1, d2, moved = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1]), max_diff(outs[1], outs[0])
        zero_up = src_variant(base, key_model, src_block=s, passes=2, randomize=False)
        z = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(zero_up, tokens)
        check(d0 <= 1e-5 and d1 <= 1e-5 and d2 <= 1e-5 and moved > 1e-3 and max_diff(z[0], z[1]) <= 1e-6,
              f"tiny 12a. src: pass 1 == Marin's own Transformer (max |d|={d0:.3g}) and the reference ({d1:.3g}), pass 2 "
              f"== reference ({d2:.3g}), moved {moved:.3g}; zero up -> pass 2 == pass 1 ({max_diff(z[0], z[1]):.3g})")
        model3 = src_variant(base, key_model, src_block=s, passes=3)
        fwd = eqx.filter_jit(lambda t, m, model=model3: model.run_passes(t, m, 3, train=False)[0][-1])
        causality(fwd, tokens, mask, pos=11, split=split, label="tiny 12b. src 3 passes", tol=0.0)
        causality(fwd, tokens, mask, pos=split, split=split, label="tiny 12b. src 3 passes, doc start", tol=0.0)
        model8 = src_variant(base, key_model, src_block=s, passes=short)
        jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
        exact = eqx.filter_jit(lambda m, t: ref_src_exact(m, t, mask8))(model8, tok8)
        two = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 2, train=False)[0][-1])(model8, tok8)
        check(max_diff(jac, exact) <= 1e-5 and max_diff(two, exact) > 1e-4,
              f"tiny 12b. src: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g}); "
              f"2 passes differ by {max_diff(two, exact):.3g}")
        _, g1 = grads(src_variant(base, key_model, src_block=s, passes=1, randomize=False))
        _, g2 = grads(src_variant(base, key_model, src_block=s, passes=2, randomize=False))
        dormant = max(max_abs(g1.src_up.weight), max_abs(g1.src_down.weight))
        per_block = [float(jnp.max(jnp.abs(g2.src_up.weight[i]))) for i in range(s + 1)]
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
        groups = am.optimizer_groups(src_variant(base, key_model, src_block=s, passes=2, randomize=False),
                                     optimizer_config(cfg, num_train_steps=100, batch_size=8))
        check(dormant == 0.0 and min(per_block) > 0 and max_abs(g2.src_down.weight) == 0.0 and finite
              and groups and all(v == "adam" for v in groups.values()) and any(k.startswith("src_") for k in groups),
              f"tiny 12c. src: one pass -> map gradients zero ({dormant:.3g}); first two-pass step |g| up per target "
              f"block {['%.3g' % v for v in per_block]}, down {max_abs(g2.src_down.weight):.3g} (zero while up is zero); "
              f"finite={finite}; new parameters -> {groups}")
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint({"model": base}, step=1, checkpoint_path=tmp, is_temporary=False)
            template = src_variant(base, key_model, src_block=s, passes=2, randomize=False)
            restored = load_checkpoint({"model": template}, tmp, allow_partial=True)["model"]
        kept = max_diff([getattr(restored, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"],
                        [getattr(base, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"])
        pass1 = max_diff(eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 1, train=False)[0][0])(restored, tokens),
                         plain_marin)
        check(kept == 0.0 and max_abs(restored.src_up.weight) == 0.0 and pass1 <= 1e-5,
              f"tiny 12d. src branched from a plain Marin checkpoint: saved leaves kept (max |d|={kept:.3g}), maps from "
              f"the template (up zero), pass 1 == Marin's Transformer (max |d|={pass1:.3g})")
        import experiments.grug.moe_hero_ep.train as grug_train

        drop, flops = grug_train._drop_metrics, grug_train._compute_flops
        am.install_src(passes=2, eval_passes=3, src_block=s, src_rank=4, bos_id=BOS)
        built = grug_train.Transformer.init(cfg, key=key_model)
        grug_train.Transformer, grug_train._drop_metrics, grug_train._compute_flops = gm.Transformer, drop, flops
        am._SRC_SPEC.update(src_block=-1, passes=1, eval_passes=1, src_rank=64)
        backbone = max_diff([getattr(built, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"],
                            [getattr(base, f.name) for f in dataclasses.fields(gm.Transformer) if f.name != "config"])
        check(isinstance(built, am.SrcTransformer) and built.passes == 2 and built.eval_passes == 3 and built.src_block == s
              and backbone == 0.0 and built.attnres_query is None,
              f"tiny 12e. install_src(): Transformer.init -> SrcTransformer(passes 2, eval 3, source {s}), backbone == "
              f"Marin init (|d|={backbone}), no AttnRes parameters")


STDRES_CASES = (("pool", "own", "slot", 4), ("pool", "own", "slot", 0), ("pool", "uniform", "none", 0),
                ("fbt", "shared", "none", 0))
STDRES_NEW = ("frontier_logit", "frontier_query", "frontier_proj_down", "frontier_proj_up", "frontier_proj_full",
              "fbt_state_proj", "fbt_token_gate")


def stdres_top_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _stdres_top_checks()


def _stdres_top_checks() -> None:
    """Group 18 (the feedback-depth ablation on the standard residual): variant stdres with depth "top" reads only the
    previous position's last block (its attention and MoE outputs) in the sublayer feedback path, everything else as
    ALF on the standard residual trunk (pool + own query, with or without the maps)."""
    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261113), 2)
    split, seq = 7, cfg.max_seq_len
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
        # 18a. two passes == the reference with the top window; pass 1 == depth whole, pass 2 differs
        for proj, r in (("slot", 4), ("none", 0)):
            top = stdres_variant(base, key_model, passes=2, proj=proj, proj_rank=r, depth="top")
            whole = stdres_variant(base, key_model, passes=2, proj=proj, proj_rank=r)
            outs = run2(top, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(top, tokens)
            d1, d2 = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1])
            wo = run2(whole, tokens)
            same1, apart2 = max_diff(outs[0], wo[0]), max_diff(outs[1], wo[1])
            check(d1 <= 1e-5 and d2 <= 1e-5 and same1 <= 1e-6 and apart2 > 1e-3,
                  f"tiny 18a. stdres pool/own proj {proj} depth top: pass 1 max |d|={d1:.3g}, pass 2 max |d|={d2:.3g} "
                  f"vs reference; pass 1 == depth whole (max |d|={same1:.3g}), pass 2 differs by {apart2:.3g}")
        # 18b. depth whole is unchanged by the option: the reference with the whole window still matches
        whole = stdres_variant(base, key_model, passes=2, proj="slot", proj_rank=4)
        outs = run2(whole, tokens)
        refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(whole, tokens)
        check(max_diff(outs[1], refs[1]) <= 1e-5,
              f"tiny 18b. stdres depth whole still == reference (max |d|={max_diff(outs[1], refs[1]):.3g})")
        # 18c. validation: depth top needs share pool and query own
        for kw in ({"share": "fbt", "query": "shared", "proj": "none", "proj_rank": 0}, {"query": "uniform"},
                   {"depth_bad": True}):
            try:
                if kw.get("depth_bad"):
                    am.StdResTransformer.from_base(base, key=key_model, depth="middle")
                else:
                    stdres_variant(base, key_model, passes=2, depth="top", randomize=False, **kw)
                rejected = False
            except ValueError:
                rejected = True
            check(rejected, f"tiny 18c. stdres depth {'middle' if kw.get('depth_bad') else 'top with ' + str(kw)} is rejected")
        # 18d. causality and K = T Jacobi == exact recursion with the top window
        model = stdres_variant(base, key_model, passes=3, proj="slot", proj_rank=4, depth="top")
        fwd = eqx.filter_jit(lambda t, m, model=model: model.run_passes(t, m, 3, train=False)[0][-1])
        causality(fwd, tokens, mask, pos=11, split=split, label="tiny 18d. stdres depth top 3 passes", tol=0.0)
        causality(fwd, tokens, mask, pos=split, split=split, label="tiny 18d. stdres depth top 3 passes, doc start", tol=0.0)
        short = 8
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        model = stdres_variant(base, key_model, passes=short, proj="slot", proj_rank=4, depth="top")
        jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model, tok8)
        exact = eqx.filter_jit(lambda m, t: ref_stdres_exact(m, t, mask8))(model, tok8)
        check(max_diff(jac, exact) <= 1e-5,
              f"tiny 18d. stdres depth top: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g})")
        # 18e. install_stdres records depth top
        am.install_stdres(passes=2, eval_passes=3, share="pool", query="own", proj="slot", proj_rank=4, bos_id=BOS,
                          depth="top")
        built = am.StdResTransformer.init(cfg, key=key_model)
        check(built.feedback_depth == "top", f"tiny 18e. install_stdres(depth='top') -> feedback_depth={built.feedback_depth}")


def inject_ablation_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _inject_ablation_checks()


def _inject_ablation_checks() -> None:
    """Group 19 (the injection ablation on the standard residual): inject "input" = ALF's sublayer feedback read
    entering through FBT's gate product in place of the token input; source "top" = ALF's per-sublayer residual mix
    reading only the previous final state through each sublayer's own map."""
    from alf.marin.recipe import optimizer_config

    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261114), 2)
    split, seq, short = 7, cfg.max_seq_len, 8
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
        alf_out = run2(stdres_variant(base, key_model, passes=2), tokens)
        cases = [("inject input, norm embed", dict(inject="input", fbt_input_norm="embed")),
                 ("inject input, norm unit", dict(inject="input")),
                 ("source top", dict(source="top"))]
        # 19a. two passes == the reference; pass 1 == ALF's pass 1; pass 2 differs from ALF
        for label, kw in cases:
            model = stdres_variant(base, key_model, passes=2, **kw)
            outs = run2(model, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(model, tokens)
            d1, d2 = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1])
            same1, apart2 = max_diff(outs[0], alf_out[0]), max_diff(outs[1], alf_out[1])
            check(d1 <= 1e-5 and d2 <= 1e-5 and same1 <= 1e-6 and apart2 > 1e-3,
                  f"tiny 19a. stdres {label}: pass 1 max |d|={d1:.3g}, pass 2 max |d|={d2:.3g} vs reference; pass 1 == "
                  f"ALF (max |d|={same1:.3g}), pass 2 differs by {apart2:.3g}")
        # 19b. causality and K = T Jacobi passes == the exact token-by-token recursion
        for label, kw in cases[::2]:
            model = stdres_variant(base, key_model, passes=3, **kw)
            fwd = eqx.filter_jit(lambda t, m, model=model: model.run_passes(t, m, 3, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 19b. stdres {label} 3 passes", tol=0.0)
            causality(fwd, tokens, mask, pos=split, split=split, label=f"tiny 19b. stdres {label} 3 passes, doc start",
                      tol=0.0)
            model = stdres_variant(base, key_model, passes=short, **kw)
            jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model, tok8)
            exact = eqx.filter_jit(lambda m, t: ref_stdres_exact(m, t, mask8))(model, tok8)
            check(max_diff(jac, exact) <= 1e-5,
                  f"tiny 19b. stdres {label}: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g})")

        # 19c. gradients: one pass leaves the new parameters at zero, two passes move the ones each mode uses
        def grads(m):
            def loss_of(mm):
                return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                          return_router_metrics=True)
            (loss, metrics), g = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(m)
            return loss, metrics, g

        for label, kw, used in (("inject input", dict(inject="input", fbt_input_norm="embed"),
                                 ("fbt_state_proj", "fbt_token_gate", "frontier_query", "frontier_proj_up")),
                                ("source top", dict(source="top"), ("frontier_logit", "frontier_proj_up"))):
            _, _, g1 = grads(stdres_variant(base, key_model, passes=1, randomize=False, **kw))
            loss2, metrics2, g2 = grads(stdres_variant(base, key_model, passes=2, randomize=False, **kw))
            dormant = max(max_abs(getattr(g1, f).weight) for f in used)
            live = {f: max_abs(getattr(g2, f).weight) for f in used}
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
            ce = [float(metrics2[f"train/cross_entropy_loss_pass{k}"]) for k in (1, 2)]
            groups = am.optimizer_groups(stdres_variant(base, key_model, passes=2, randomize=False, **kw),
                                         optimizer_config(cfg, num_train_steps=100, batch_size=8))
            check(dormant == 0.0 and min(live.values()) > 0 and finite and abs(float(loss2) - ce[0] - ce[1]) <= 1e-5
                  and all(v == "adam" for v in groups.values()),
                  f"tiny 19c. stdres {label}: one pass -> gradients zero ({dormant:.3g}); two passes -> max |g| "
                  f"{ {k: float('%.3g' % v) for k, v in live.items()} }; finite={finite}; loss = pass-1 + pass-2 CE; "
                  f"new parameters -> Adam ({len(groups)})")
        # 19d. validation and the decoding modes refused
        bad = [dict(inject="input", source="top"), dict(inject="input", query="uniform"),
               dict(source="top", proj="none", proj_rank=0), dict(inject="sideways"),
               dict(inject="input", depth="top"), dict(share="fbt", query="shared", proj="none", proj_rank=0, source="top")]
        for kw in bad:
            try:
                stdres_variant(base, key_model, passes=2, randomize=False, **kw)
                rejected = False
            except ValueError:
                rejected = True
            check(rejected, f"tiny 19d. stdres {kw} is rejected")
        try:
            am.StdResTransformer.from_base(base, key=key_model, passes=2, eval_passes=2, jitter=0.02, bos_id=BOS,
                                           proj="slot", proj_rank=4, inject="input")
            rejected = False
        except ValueError:
            rejected = True
        check(rejected, "tiny 19d. stdres inject input with jitter is rejected")
        model = stdres_variant(base, key_model, passes=2, inject="input")
        try:
            model.run_passes(tokens, mask, 2, train=False, context=jnp.zeros(tokens.shape, bool), context_passes=1)
            rejected = False
        except ValueError:
            rejected = True
        check(rejected, "tiny 19d. stdres inject input refuses the decoding modes (context)")
        # 19e. install_stdres records the mode
        am.install_stdres(passes=2, eval_passes=3, share="pool", query="own", proj="slot", proj_rank=4, bos_id=BOS,
                          inject="input", fbt_input_norm="embed")
        built = am.StdResTransformer.init(cfg, key=key_model)
        check(built.feedback_replaces_input == "input" and built.fbt_state_proj is not None and built.fbt_input_norm == "embed",
              f"tiny 19e. install_stdres(inject='input') -> inject={built.feedback_replaces_input}, fusion built, "
              f"norm={built.fbt_input_norm}")


def embed_inject_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _embed_inject_checks()


def _embed_inject_checks() -> None:
    """Group 20: the sublayer feedback path of every sublayer plus the input injection path, the previous top state
    added to the token input through a zero-initialized D x D map (embed_inject "match" / "embed")."""
    import tempfile

    from alf.marin.recipe import optimizer_config

    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(2026111420), 2)
    split, seq, short = 7, cfg.max_seq_len, 8
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
        alf_out = run2(stdres_variant(base, key_model, passes=2), tokens)
        for mode in ("match", "embed"):
            # 20a. two passes == the reference; pass 1 == ALF's pass 1; pass 2 differs once W_E is not zero
            model = stdres_variant(base, key_model, passes=2, embed_inject=mode)
            outs = run2(model, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(model, tokens)
            d1, d2 = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1])
            same1, apart2 = max_diff(outs[0], alf_out[0]), max_diff(outs[1], alf_out[1])
            check(d1 <= 1e-5 and d2 <= 1e-5 and same1 <= 1e-6 and apart2 > 1e-3,
                  f"tiny 20a. stdres embed_inject {mode}: pass 1 max |d|={d1:.3g}, pass 2 max |d|={d2:.3g} vs reference; "
                  f"pass 1 == ALF (max |d|={same1:.3g}), pass 2 differs by {apart2:.3g}")
            # 20b. W_E = 0 (the initialization): every pass is ALF's, exactly
            zero = eqx.tree_at(lambda m: m.embed_inject_proj.weight, model, jnp.zeros_like(model.embed_inject_proj.weight))
            z = run2(zero, tokens)
            # embed re-runs the embedding norms on e + 0, match adds an exact zero: both must reproduce ALF
            check(max_diff(z[1], alf_out[1]) <= (1e-6 if mode == "embed" else 0.0),
                  f"tiny 20b. stdres embed_inject {mode} with W_E = 0: pass 2 == ALF (max |d|={max_diff(z[1], alf_out[1]):.3g})")
            # 20c. the injection is scale-free in the top state: h -> 7 h (through the final gains) changes nothing it adds
            x0 = model._embed(tokens)
            ctx = model._context(tokens, mask)
            a = model._embed_inject_input(tokens, x0, outs[0], ctx, train=False, pass_idx=0)
            b = model._embed_inject_input(tokens, x0, 7.0 * outs[0], ctx, train=False, pass_idx=0)
            starts = ~ctx["fb_valid"]
            kept = max_diff(jnp.where(starts[..., None], a, 0.0), jnp.where(starts[..., None], x0, 0.0))
            ratio = float(jnp.sqrt(jnp.mean(jnp.square(a - x0)) / jnp.mean(jnp.square(x0))))
            check(max_diff(a, b) <= 2e-5 and kept == 0.0 and int(jnp.sum(starts)) == 3,
                  f"tiny 20c. stdres embed_inject {mode}: 7 x top state -> same input (max |d|={max_diff(a, b):.3g}); the "
                  f"3 document starts keep x0 exactly; rms(input - x0) / rms(x0) = {ratio:.3g} at |W_E| ~ 0.5")
            # 20d. causality and K = T Jacobi passes == the exact token-by-token recursion
            model3 = stdres_variant(base, key_model, passes=3, embed_inject=mode)
            fwd = eqx.filter_jit(lambda t, m, model3=model3: model3.run_passes(t, m, 3, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 20d. stdres embed_inject {mode} 3 passes", tol=0.0)
            causality(fwd, tokens, mask, pos=split, split=split,
                      label=f"tiny 20d. stdres embed_inject {mode} 3 passes, doc start", tol=0.0)
            model8 = stdres_variant(base, key_model, passes=short, embed_inject=mode)
            jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
            exact = eqx.filter_jit(lambda m, t: ref_stdres_exact(m, t, mask8))(model8, tok8)
            check(max_diff(jac, exact) <= 1e-5,
                  f"tiny 20d. stdres embed_inject {mode}: {short} Jacobi passes == exact recursion "
                  f"(max |d|={max_diff(jac, exact):.3g})")

            # 20e. gradients: one pass leaves every new parameter at zero; two passes from the initialization move W_E
            def grads(m):
                def loss_of(mm):
                    return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                              return_router_metrics=True)
                (loss, metrics), g = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(m)
                return loss, metrics, g

            used = ("embed_inject_proj", "frontier_logit", "frontier_proj_up")
            _, _, g1 = grads(stdres_variant(base, key_model, passes=1, randomize=False, embed_inject=mode))
            init2 = stdres_variant(base, key_model, passes=2, randomize=False, embed_inject=mode)
            loss2, metrics2, g2 = grads(init2)
            dormant = max(max_abs(getattr(g1, f).weight) for f in used)
            live = {f: max_abs(getattr(g2, f).weight) for f in used}
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
            ce = [float(metrics2[f"train/cross_entropy_loss_pass{k}"]) for k in (1, 2)]
            groups = am.optimizer_groups(init2, optimizer_config(cfg, num_train_steps=100, batch_size=8))
            check(dormant == 0.0 and min(live.values()) > 0 and finite and abs(float(loss2) - ce[0] - ce[1]) <= 1e-5
                  and groups.get("embed_inject_proj.weight") == "adam" and all(v == "adam" for v in groups.values()),
                  f"tiny 20e. stdres embed_inject {mode}: one pass -> gradients zero ({dormant:.3g}); two passes from the "
                  f"zero map -> max |g| { {k: float('%.3g' % v) for k, v in live.items()} }; finite={finite}; "
                  f"W_E -> {groups.get('embed_inject_proj.weight')} ({len(groups)} new leaves, all Adam)")
            # 20f. decoding modes: FUSED with the prompt in every pass == the plain passes; SOFT differs and is causal
            ctx_mask = jnp.zeros(tokens.shape, bool).at[:, :5].set(True)
            fused = eqx.filter_jit(lambda m, t: m.run_passes(
                t, mask, 3, train=False, context=ctx_mask, context_passes=3)[0][-1])(model3, tokens)
            plain3 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=False)[0][-1])(model3, tokens)
            soft = eqx.filter_jit(lambda m, t: m.run_passes(
                t, mask, 3, train=False, context=ctx_mask, context_passes=1)[0][-1])(model3, tokens)
            p1 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 1, train=False)[0][-1])(model3, tokens)
            hold2 = eqx.filter_jit(lambda m, t: m.run_passes(
                t, mask, 3, train=False, context=ctx_mask, context_passes=2)[0][-1])(model3, tokens)
            p2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0][-1])(model3, tokens)
            check(max_diff(fused, plain3) == 0.0 and max_diff(soft[:, :5], p1[:, :5]) <= 1e-6
                  and max_diff(soft[:, 5:], plain3[:, 5:]) > 1e-4 and max_diff(hold2[:, :5], p2[:, :5]) <= 1e-6,
                  f"tiny 20f. stdres embed_inject {mode} decoding modes: FUSED == plain passes; SOFT prompt == pass 1 "
                  f"(max |d|={max_diff(soft[:, :5], p1[:, :5]):.3g}), its continuation differs from plain "
                  f"({max_diff(soft[:, 5:], plain3[:, 5:]):.3g}); a 2-pass prompt keeps its pass-2 state "
                  f"(max |d|={max_diff(hold2[:, :5], p2[:, :5]):.3g})")
        # 20g. validation, the run-directory record, install_stdres
        bad = [dict(embed_inject="unit"), dict(embed_inject="match", inject="input"),
               dict(embed_inject="match", source="top"),
               dict(embed_inject="embed", share="fbt", query="shared", proj="none", proj_rank=0)]
        for kw in bad:
            try:
                stdres_variant(base, key_model, passes=2, randomize=False, **kw)
                rejected = False
            except ValueError:
                rejected = True
            check(rejected, f"tiny 20g. stdres {kw} is rejected")
        with tempfile.TemporaryDirectory() as tmp:
            none = am.read_embed_inject(tmp)
            am.record_embed_inject(tmp, "match")
            am.record_embed_inject(tmp, "match")
            try:
                am.record_embed_inject(tmp, "embed")
                refused = False
            except ValueError:
                refused = True
            check(none == "none" and am.read_embed_inject(tmp) == "match" and refused,
                  f"tiny 20g. run-directory record: no file -> {none}, written -> {am.read_embed_inject(tmp)}, a resume "
                  f"with another mode refused={refused}")
        am.install_stdres(passes=2, eval_passes=3, share="pool", query="own", proj="slot", proj_rank=4, bos_id=BOS,
                          embed_inject="match")
        built = am.StdResTransformer.init(cfg, key=key_model)
        check(built.embed_inject == "match" and built.embed_inject_proj is not None
              and max_abs(built.embed_inject_proj.weight) == 0.0 and built.frontier_proj_up is not None,
              f"tiny 20g. install_stdres(embed_inject='match') -> {built.embed_inject}, W_E zero-initialized "
              f"{tuple(built.embed_inject_proj.weight.shape)}, per-output maps built")


def embed_only_checks() -> None:
    """Group 21: embed_inject "only", the input injection path without the sublayer feedback path."""
    with jax.default_matmul_precision("highest"):
        cfg = tiny_config()
        key_tok, key_model = jax.random.split(jax.random.key(2026111421), 2)
        split, seq, short = 7, cfg.max_seq_len, 8
        with set_mesh(explicit_mesh()):
            base = gm.Transformer.init(cfg, key=key_model)
            tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
            tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
            loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
            run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
            model = stdres_variant(base, key_model, passes=2, embed_inject="only")
            outs = run2(model, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(model, tokens)
            both = run2(stdres_variant(base, key_model, passes=2, embed_inject="match"), tokens)
            check(max_diff(outs[0], refs[0]) <= 1e-5 and max_diff(outs[1], refs[1]) <= 1e-5
                  and max_diff(outs[1], outs[0]) > 1e-3 and max_diff(outs[1], both[1]) > 1e-3,
                  f"tiny 21a. stdres embed_inject only: two passes == reference (max |d|={max_diff(outs[1], refs[1]):.3g}); "
                  f"pass 2 differs from pass 1 ({max_diff(outs[1], outs[0]):.3g}) and from match with the stream "
                  f"({max_diff(outs[1], both[1]):.3g})")
            zero = eqx.tree_at(lambda m: m.embed_inject_proj.weight, model, jnp.zeros_like(model.embed_inject_proj.weight))
            z = run2(zero, tokens)
            check(max_diff(z[1], z[0]) == 0.0,
                  f"tiny 21b. stdres embed_inject only with W_E = 0: pass 2 == pass 1 exactly (max |d|={max_diff(z[1], z[0]):.3g}): "
                  "no sublayer reads the previous position")
            model8 = stdres_variant(base, key_model, passes=short, embed_inject="only")
            jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
            exact = eqx.filter_jit(lambda m, t: ref_stdres_exact(m, t, mask8))(model8, tok8)
            fwd = eqx.filter_jit(lambda t, m: model.run_passes(t, m, 2, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=11, split=split, label="tiny 21c. stdres embed_inject only 2 passes", tol=0.0)
            check(max_diff(jac, exact) <= 1e-5,
                  f"tiny 21c. stdres embed_inject only: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g})")

            def loss_of(mm):
                return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4)
            init2 = stdres_variant(base, key_model, passes=2, randomize=False, embed_inject="only")
            g = eqx.filter_jit(eqx.filter_grad(loss_of))(init2)
            dormant = max(max_abs(getattr(g, f).weight) for f in ("frontier_logit", "frontier_query", "frontier_proj_up"))
            check(max_abs(g.embed_inject_proj.weight) > 0 and dormant == 0.0
                  and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g)),
                  f"tiny 21d. stdres embed_inject only, two passes from the zero map: W_E max |g| "
                  f"{max_abs(g.embed_inject_proj.weight):.3g}, the stream's parameters stay dormant ({dormant:.3g})")


def embed_plain_checks() -> None:
    """Group 22: embed_inject "plain:<c>", Recycler's adapter as it is, x0 + c RMSNorm(h) W_E."""
    with jax.default_matmul_precision("highest"):
        cfg = tiny_config()
        key_tok, key_model = jax.random.split(jax.random.key(2026111422), 2)
        split, seq, short = 7, cfg.max_seq_len, 8
        d = cfg.hidden_dim
        with set_mesh(explicit_mesh()):
            base = gm.Transformer.init(cfg, key=key_model)
            tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
            tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
            loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
            run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
            alf_out = run2(stdres_variant(base, key_model, passes=2), tokens)
            w = 0.02 * jax.random.normal(jax.random.fold_in(key_model, 23), (d, d))
            for mode in ("plain", "plainonly"):
                model = stdres_variant(base, key_model, passes=2, embed_inject=f"{mode}:0.25")
                model = eqx.tree_at(lambda m: m.embed_inject_proj.weight, model, w)
                outs = run2(model, tokens)
                refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(model, tokens)
                folded = stdres_variant(base, key_model, passes=2, embed_inject=f"{mode}:1")
                folded = eqx.tree_at(lambda m: m.embed_inject_proj.weight, folded, 0.25 * w)
                f = run2(folded, tokens)
                check(model.embed_inject == mode and model.embed_inject_scale == 0.25 and max_diff(outs[0], refs[0]) <= 1e-5
                      and max_diff(outs[1], refs[1]) <= 1e-5 and max_diff(outs[1], f[1]) <= 1e-6 and max_diff(outs[1], outs[0]) > 1e-3,
                      f"tiny 22a. stdres embed_inject {mode}:0.25: two passes == reference (max |d|={max_diff(outs[1], refs[1]):.3g}); "
                      f"(c, W_E) == (1, c W_E), the constant folds into the weights (max |d|={max_diff(outs[1], f[1]):.3g}); "
                      f"pass 2 differs from pass 1 by {max_diff(outs[1], outs[0]):.3g}")
                zero = eqx.tree_at(lambda m: m.embed_inject_proj.weight, model, jnp.zeros_like(w))
                z = run2(zero, tokens)
                target = alf_out[1] if mode == "plain" else z[0]
                check(max_diff(z[1], target) == 0.0,
                      f"tiny 22b. stdres embed_inject {mode} with W_E = 0: pass 2 == {'ALF' if mode == 'plain' else 'pass 1'} exactly")
                model8 = eqx.tree_at(lambda m: m.embed_inject_proj.weight,
                                     stdres_variant(base, key_model, passes=short, embed_inject=f"{mode}:0.25"), w)
                jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
                exact = eqx.filter_jit(lambda m, t: ref_stdres_exact(m, t, mask8))(model8, tok8)
                fwd = eqx.filter_jit(lambda t, m, model=model: model.run_passes(t, m, 2, train=False)[0][-1])
                causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 22c. stdres embed_inject {mode} 2 passes", tol=0.0)
                check(max_diff(jac, exact) <= 1e-5,
                      f"tiny 22c. stdres embed_inject {mode}: {short} Jacobi passes == exact recursion "
                      f"(max |d|={max_diff(jac, exact):.3g})")

                def loss_of(mm):
                    return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4)
                init2 = stdres_variant(base, key_model, passes=2, randomize=False, embed_inject=f"{mode}:0.25")
                init1 = stdres_variant(base, key_model, passes=2, randomize=False, embed_inject=f"{mode}:1")
                g, g1 = eqx.filter_jit(eqx.filter_grad(loss_of))(init2), eqx.filter_jit(eqx.filter_grad(loss_of))(init1)
                ratio = max_abs(g.embed_inject_proj.weight) / max_abs(g1.embed_inject_proj.weight)
                check(max_abs(init2.embed_inject_proj.weight) == 0.0 and abs(ratio - 0.25) <= 1e-4,
                      f"tiny 22d. stdres embed_inject {mode}: zero start; the gradient of W_E under c = 0.25 is {ratio:.4f} "
                      "times that under c = 1 (Adam's step does not see this factor: c is its learning-rate multiplier)")
            for bad in ("plain", "plain:0", "plain:-1", "match:2", "plane:1"):
                try:
                    am.parse_embed_inject(bad)
                    rejected = False
                except ValueError:
                    rejected = True
                check(rejected, f"tiny 22e. embed_inject {bad!r} is rejected")
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                am.record_embed_inject(tmp, "plain:0.003")
                check(am.read_embed_inject(tmp) == "plain:0.003" and am.parse_embed_inject(am.read_embed_inject(tmp)) == ("plain", 0.003),
                      f"tiny 22e. run-directory record keeps the constant: {am.read_embed_inject(tmp)}")


def embed_inject_d768_checks() -> None:
    """d768 (2 layers, bf16 compute, FA4 attention, sonic MoE): stdres pool + embed_inject match / embed."""
    import jmp
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.recipe import model_config

    cfg = model_config(shape="d768", layers=2)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    seq, split = cfg.max_seq_len, 2500
    key_tok, key_model = jax.random.split(jax.random.key(768), 2)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, 128, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32)
        plain = eqx.filter_jit(lambda m: mp.cast_to_compute(m).next_token_loss(
            tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4))(base)
        for mode in ("match", "embed"):
            model = stdres_variant(base, key_model, passes=2, proj_rank=64, randomize=False, embed_inject=mode)

            def loss_fn(m):
                return mp.cast_to_compute(m).next_token_loss(tokens, loss_weight, mask=mask, reduction="mean",
                                                             logsumexp_weight=1e-4, return_router_metrics=True)

            (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
            ce1 = float(metrics["train/cross_entropy_loss_pass1"])
            live = max_abs(grads.embed_inject_proj.weight)
            check(bool(jnp.isfinite(loss)) and finite and live > 0 and abs(ce1 - float(plain)) <= 5e-3,
                  f"d768 stdres embed_inject {mode}: loss {float(loss):.4f}, pass-1 CE {ce1:.5f} vs Marin's Transformer "
                  f"{float(plain):.5f}, gradients finite={finite}, W_E max |g| {live:.3g}")
            rand = stdres_variant(base, key_model, passes=2, proj_rank=64, embed_inject=mode)
            fwd = eqx.filter_jit(lambda t, m, rand=rand: mp.cast_to_compute(rand).run_passes(t, m, 2, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=3000, split=split, label=f"d768 stdres embed_inject {mode} 2 passes", tol=1e-2)


def embed_plain_attnres_checks() -> None:
    """Group 23: embed_inject "plain:<c>" / "plainonly:<c>" on the AttnRes trunk (variant feedback, share pool, own
    query, one map per previous slot): slot 0 of passes 2..K gets c RMSNorm(h) W_E, h the previous position's final
    mixture, the input of the final norms."""
    import tempfile

    import experiments.grug.moe_hero_ep.train as grug_train
    from levanter.checkpoint import load_checkpoint, save_checkpoint
    from alf.marin.recipe import optimizer_config


    with jax.default_matmul_precision("highest"):
        cfg = tiny_config()
        key_tok, key_model = jax.random.split(jax.random.key(2026111523), 2)
        split, seq, short = 7, cfg.max_seq_len, 8
        d, n_mix = cfg.hidden_dim, 2 * cfg.num_layers
        with set_mesh(explicit_mesh()):
            base = gm.Transformer.init(cfg, key=key_model)
            tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
            tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
            loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
            w = 0.1 * jax.random.normal(jax.random.fold_in(key_model, 23), (d, d))
            method = dict(share="pool", query="own", proj="slot", proj_rank=4, inject_rank=4)

            def build(passes, embed_inject="none", *, w_e=None, randomize=True, **kw):
                """The method on this trunk (pool, own query, one map per previous slot; the dormant inject maps an
                AttnRes base carries), with the knob and W_E = w_e."""
                m = variant(base, key_model, passes=passes, randomize=randomize, embed_inject=embed_inject, **{**method, **kw})
                return m if w_e is None else eqx.tree_at(lambda mm: mm.embed_inject_proj.weight, m, w_e)

            run2 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])
            run3 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=False)[0])
            noknob = run3(build(3), tokens)

            def loss_of(mm):
                return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4)

            # 23a. _trunk hands out its final mixture (the input of the final norms) and nothing else changes
            probe = build(2, "plain:0.25", w_e=w)
            ctx0, x00 = probe._context(tokens, mask), probe._embed(tokens)
            three = probe._trunk(x00, ctx0, None)
            four = probe._trunk(x00, ctx0, None, with_top=True)
            _, ref_outs, _ = ref_trunk(probe, tokens, mask)
            d_mix = max_diff(four[3], ref_final_mix(probe, tokens, ref_outs))
            d_norm = max_diff(probe.final_gated_norm(probe.final_norm(four[3])), four[0])
            same = max(max_diff(three[0], four[0]), max_diff(three[1], four[1]), max_diff(three[2], four[2]))
            handed = probe._top_state(x00, four[0], four[1], four[3]) is four[3]
            check(len(three) == 3 and len(four) == 4 and same == 0.0 and d_mix <= 1e-5 and d_norm <= 1e-6 and handed
                  and max_diff(four[3], four[0]) > 1e-3,
                  f"tiny 23a. AttnRes _trunk(with_top=True): the 4th output == the final mixture of the formula (max |d|="
                  f"{d_mix:.3g}), its final norms == the output (max |d|={d_norm:.3g}), _top_state returns that tensor "
                  f"itself ({handed}); the first three outputs == the plain call (max |d|={same:.3g}); the mixture "
                  f"differs from the normalized output by {max_diff(four[3], four[0]):.3g}")

            for mode in ("plain", "plainonly"):
                # 23b. two passes == the reference; (c, W_E) == (1, c W_E)
                model = build(2, f"{mode}:0.25", w_e=w)
                outs = run2(model, tokens)
                refs = eqx.filter_jit(lambda m, t: ref_attnres_embed_passes(m, t, mask, 2))(model, tokens)
                f = run2(build(2, f"{mode}:1", w_e=0.25 * w), tokens)
                apart = max_diff(outs[1], noknob[1] if mode == "plain" else outs[0])
                check(model.embed_inject == mode and model.embed_inject_scale == 0.25 and max_diff(outs[0], refs[0]) <= 1e-5
                      and max_diff(outs[1], refs[1]) <= 1e-5 and max_diff(outs[0], noknob[0]) <= 1e-6
                      and max_diff(outs[1], f[1]) <= 1e-6 and apart > 1e-3,
                      f"tiny 23b. attnres embed_inject {mode}:0.25: pass 1 == the model without the knob (max |d|="
                      f"{max_diff(outs[0], noknob[0]):.3g}), two "
                      f"passes == reference (max |d|={max_diff(outs[1], refs[1]):.3g}); (c, W_E) == (1, c W_E) (max |d|="
                      f"{max_diff(outs[1], f[1]):.3g}); pass 2 differs from "
                      f"{'the pool read alone' if mode == 'plain' else 'pass 1'} by {apart:.3g}")
                # 23c. W_E = 0 (the initialization): the model built without the knob, exactly, in passes 2 and 3
                # (plainonly: pass 3 is compared with pass 1 of the same program, three compiled instances of one trunk;
                # on an H800 the third comes out one float32 ulp away, 1.19e-07, the second not: bound, not exact)
                z = run3(build(3, f"{mode}:0.25"), tokens)
                target = noknob if mode == "plain" else [z[0], z[0], z[0]]
                d2, d3 = max_diff(z[1], target[1]), max_diff(z[2], target[2])
                check(d2 == 0.0 and d3 <= (0.0 if mode == "plain" else 1e-6) and max_diff(z[0], noknob[0]) == 0.0,
                      f"tiny 23c. attnres embed_inject {mode} with W_E = 0: pass 2 == "
                      f"{'the model without the knob' if mode == 'plain' else 'pass 1 (no mixing point reads)'} exactly "
                      f"(max |d|={d2:.3g}), pass 3 {'exactly' if mode == 'plain' else 'within 1e-6'} (max |d|={d3:.3g})")
                # 23d. causality (three passes; the document boundary, the other row)
                model3 = build(3, f"{mode}:0.25", w_e=w)
                fwd = eqx.filter_jit(lambda t, m, model3=model3: model3.run_passes(t, m, 3, train=False)[0][-1])
                causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 23d. attnres embed_inject {mode} 3 passes", tol=0.0)
                causality(fwd, tokens, mask, pos=split, split=split,
                          label=f"tiny 23d. attnres embed_inject {mode} 3 passes, doc start", tol=0.0)
                # 23e. K = T Jacobi passes == the token-by-token recursion
                model8 = build(short, f"{mode}:0.25", w_e=w)
                jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
                exact = eqx.filter_jit(lambda m, t: ref_attnres_embed_exact(m, t, mask8))(model8, tok8)
                two = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 2, train=False)[0][-1])(model8, tok8)
                check(max_diff(jac, exact) <= 1e-5 and max_diff(two, exact) > 1e-4,
                      f"tiny 23e. attnres embed_inject {mode}: {short} Jacobi passes == exact recursion (max |d|="
                      f"{max_diff(jac, exact):.3g}); 2 passes differ by {max_diff(two, exact):.3g}")
                # 23f. gradients: one pass leaves W_E (and the pool parameters) at zero; from the zero start W_E moves,
                # under c = 0.25 by 0.25 times its gradient under c = 1; every new leaf goes to Adam
                read = ("frontier_logit", "frontier_query", "frontier_proj_up")
                g_one = eqx.filter_jit(eqx.filter_grad(loss_of))(build(1, f"{mode}:0.25", w_e=w))
                dormant = max(max_abs(getattr(g_one, n).weight) for n in read + ("embed_inject_proj",))
                init2 = build(2, f"{mode}:0.25", randomize=False)
                g = eqx.filter_jit(eqx.filter_grad(loss_of))(init2)
                g1 = eqx.filter_jit(eqx.filter_grad(loss_of))(build(2, f"{mode}:1", randomize=False))
                ratio = max_abs(g.embed_inject_proj.weight) / max_abs(g1.embed_inject_proj.weight)
                pool_g = {n: max_abs(getattr(g, n).weight) for n in read}
                pool_ok = (min(pool_g["frontier_logit"], pool_g["frontier_proj_up"]) > 0 if mode == "plain"
                           else max(pool_g.values()) == 0.0)
                finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g))
                groups = am.optimizer_groups(init2, optimizer_config(cfg, num_train_steps=100, batch_size=8))
                check(dormant == 0.0 and max_abs(init2.embed_inject_proj.weight) == 0.0 and abs(ratio - 0.25) <= 1e-4
                      and pool_ok and finite and groups.get("embed_inject_proj.weight") == "adam"
                      and all(v == "adam" for v in groups.values()),
                      f"tiny 23f. attnres embed_inject {mode}: one pass -> W_E and the pool parameters at zero gradient "
                      f"({dormant:.3g}); zero start, two passes -> W_E max |g| {max_abs(g.embed_inject_proj.weight):.3g}, "
                      f"{ratio:.4f} times that under c = 1; pool parameters max |g| "
                      f"{ {k: float('%.3g' % v) for k, v in pool_g.items()} }; finite={finite}; W_E -> "
                      f"{groups.get('embed_inject_proj.weight')} ({len(groups)} new leaves, all Adam)")
                # 23g. decoding modes: FUSED with the prompt in every pass == the plain passes; SOFT keeps the prompt at
                # pass 1, a 2-pass prompt at pass 2; K = T Jacobi passes == the prompt through c passes, kept, then
                # exact token-by-token decoding
                ctx = prompt_mask(model3, tokens, mask, 3)
                c3 = ctx[..., None]
                run_ctx = eqx.filter_jit(lambda m, t, c, cp: m.run_passes(t, mask, 3, train=False, context=c, context_passes=cp)[0])
                plain3 = run3(model3, tokens)
                fused, soft, hold2 = (run_ctx(model3, tokens, ctx, cp) for cp in (3, 1, 2))
                d_empty = max(max_diff(run_ctx(model3, tokens, jnp.zeros_like(ctx), cp), plain3) for cp in (1, 2))
                soft_prompt = max(max_diff(jnp.where(c3, o, 0.0), jnp.where(c3, plain3[0], 0.0)) for o in soft)
                hold_prompt = max(max_diff(jnp.where(c3, o, 0.0), jnp.where(c3, plain3[1], 0.0)) for o in hold2[1:])
                moved = min(max_diff(soft[-1], plain3[-1]), max_diff(hold2[-1], plain3[-1]), max_diff(soft[-1], hold2[-1]))
                check(max_diff(fused, plain3) == 0.0 and d_empty <= 1e-6 and soft_prompt <= 1e-6 and hold_prompt <= 1e-6
                      and moved > 1e-5,
                      f"tiny 23g. attnres embed_inject {mode} decoding modes: FUSED with the prompt in every pass == the "
                      f"plain passes exactly; an empty prompt == every position reading ({d_empty:.3g}); SOFT keeps the "
                      f"prompt at pass 1 ({soft_prompt:.3g}), a 2-pass prompt keeps its pass-2 state ({hold_prompt:.3g}); "
                      f"modes differ after the prompt (min {moved:.3g})")
                ctx8 = prompt_mask(model8, tok8, mask8, 3)
                free = eqx.filter_jit(lambda m, t: ref_attnres_embed_exact(m, t, mask8))(model8, tok8)
                for cp in (1, 2):
                    jac = eqx.filter_jit(lambda m, t, c, cp=cp: m.run_passes(t, mask8, short, train=False, context=c,
                                                                             context_passes=cp)[0][-1])(model8, tok8, ctx8)
                    ref = eqx.filter_jit(lambda m, t, c, cp=cp: ref_attnres_embed_exact(m, t, mask8, c, cp))(model8, tok8, ctx8)
                    check(max_diff(jac, ref) <= 1e-5 and max_diff(ref, free) > 1e-5,
                          f"tiny 23g. attnres embed_inject {mode} c = {cp}: {short} Jacobi passes == prompt (3 tokens per "
                          f"document) through {cp} pass{'es' if cp > 1 else ''} then exact token-by-token decoding (max |d|="
                          f"{max_diff(jac, ref):.3g}); differs from decoding with no prompt by {max_diff(ref, free):.3g}")
                fwd_soft = eqx.filter_jit(lambda t, m, model3=model3: model3.run_passes(
                    t, m, 3, train=False, context=prompt_mask(model3, t, m, 3), context_passes=1)[0][-1])
                causality(fwd_soft, tokens, mask, pos=11, split=split, label=f"tiny 23g. attnres embed_inject {mode} SOFT", tol=0.0)
                # 23h. the jitter of the injected state acts in training only
                noisy = build(2, f"{mode}:0.25", w_e=w, jitter=0.5)
                quiet = dataclasses.replace(noisy, jitter=0.0)
                l_noisy, l_quiet = float(eqx.filter_jit(loss_of)(noisy)), float(eqx.filter_jit(loss_of)(quiet))
                e_noisy, e_quiet = run2(noisy, tokens), run2(quiet, tokens)
                check(abs(l_noisy - l_quiet) > 1e-6 and max_diff(e_noisy, e_quiet) == 0.0,
                      f"tiny 23h. attnres embed_inject {mode}, jitter 0.5: train loss {l_noisy:.6f} vs {l_quiet:.6f} "
                      f"without; evaluation passes equal")
            # 23j. the rejections; the default builds no leaf
            bad = [dict(embed_inject="match"), dict(embed_inject="embed"), dict(embed_inject="only"),
                   dict(embed_inject="plain"), dict(embed_inject="plain:0"),
                   dict(embed_inject="plain:0.01", share="joint", query="shared"),
                   dict(embed_inject="plain:0.01", share="inject", query="shared", proj="none", proj_rank=0),
                   dict(embed_inject="plain:0.01", share="fbt", query="shared", proj="none", proj_rank=0),
                   dict(embed_inject="plainonly:0.01", share="fbt", query="shared", proj="none", proj_rank=0),
                   dict(embed_inject="plain:0.01", share="none", query="shared", proj="none", proj_rank=0)]
            for kw in bad:
                n_pass = 2 if kw.get("share") != "none" else 1
                try:
                    build(n_pass, randomize=False, **kw)
                    rejected = False
                except ValueError:
                    rejected = True
                build(n_pass, randomize=False, **{**kw, "embed_inject": "none"})  # the same variant without the knob builds
                check(rejected, f"tiny 23j. attnres {kw} is rejected")
            off, on = build(2, randomize=False), build(2, "plain:0.01", randomize=False)
            check(off.embed_inject == "none" and off.embed_inject_proj is None and on.embed_inject_proj.weight.shape == (d, d)
                  and len(jax.tree.leaves(on)) == len(jax.tree.leaves(off)) + 1,
                  f"tiny 23j. the default builds no leaf ({len(jax.tree.leaves(off))} leaves); plain:<c> adds W_E alone "
                  f"({len(jax.tree.leaves(on))} leaves)")
            # 23k. install(): the knob reaches Transformer.init; a share that does not take it leaves the trainer alone
            drop, flops, spec0 = grug_train._drop_metrics, grug_train._compute_flops, dict(am._SPEC)
            try:
                am.install(passes=2, eval_passes=3, bos_id=BOS, share="joint", query="shared", embed_inject="plain:0.01")
                refused = False
            except ValueError:
                refused = True
            untouched = grug_train.Transformer is gm.Transformer and am._SPEC == spec0
            am.install(passes=2, eval_passes=3, bos_id=BOS, embed_inject="plain:0.01", **method)
            built = grug_train.Transformer.init(cfg, key=key_model)
            grug_train.Transformer, grug_train._drop_metrics, grug_train._compute_flops = gm.Transformer, drop, flops
            am._SPEC.clear()
            am._SPEC.update(spec0)
            check(refused and untouched and isinstance(built, am.AttnResTransformer) and built.embed_inject == "plain"
                  and built.embed_inject_scale == 0.01 and max_abs(built.embed_inject_proj.weight) == 0.0
                  and built.frontier_proj_up is not None and built.passes == 2 and built.eval_passes == 3,
                  f"tiny 23k. install(embed_inject='plain:0.01') -> Transformer.init builds embed_inject {built.embed_inject}, "
                  f"c = {built.embed_inject_scale}, W_E zero {tuple(built.embed_inject_proj.weight.shape)}; with share "
                  f"joint refused ({refused}), trainer and spec untouched ({untouched})")
            # 23l. the branch from a one-pass AttnRes base (pool / shared query, inject maps, no W_E): a strict restore
            # is refused, the partial restore keeps every saved leaf and takes W_E (zero) from the template; the
            # branch then starts exactly where the same branch without the knob starts
            saved = variant(base, key_model, share="pool", query="shared", passes=1, inject_rank=4)
            with tempfile.TemporaryDirectory() as tmp:
                save_checkpoint({"model": saved}, step=1, checkpoint_path=tmp, is_temporary=False)
                strict_refused = False
                try:
                    load_checkpoint({"model": build(2, "plain:0.01", randomize=False)}, tmp, allow_partial=False)
                except Exception:
                    strict_refused = True
                restored = load_checkpoint({"model": build(2, "plain:0.01", randomize=False)}, tmp, allow_partial=True)["model"]
                old_branch = load_checkpoint({"model": build(2, randomize=False)}, tmp, allow_partial=True)["model"]
            names = [f.name for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
            names += ["attnres_query", "frontier_logit", "frontier_inject_down", "frontier_inject_up"]
            kept = max_diff([getattr(restored, n) for n in names], [getattr(saved, n) for n in names])
            a, b = run2(restored, tokens), run2(old_branch, tokens)
            check(saved.embed_inject_proj is None and strict_refused and kept == 0.0
                  and max_abs(restored.embed_inject_proj.weight) == 0.0 and max_diff(a, b) == 0.0,
                  f"tiny 23l. branch from a one-pass AttnRes base without W_E: strict restore refused ({strict_refused}); "
                  f"the partial restore keeps every saved leaf (max |d|={kept:.3g}) and takes W_E = 0 from the template; "
                  f"both passes == the same branch without the knob exactly (max |d|={max_diff(a, b):.3g})")
            with tempfile.TemporaryDirectory() as tmp:
                am.record_embed_inject(tmp, "plain:0.01")
                check(am.read_embed_inject(tmp) == "plain:0.01",
                      f"tiny 23l. run-directory record: {am.read_embed_inject(tmp)}")



def embed_plain_attnres_d768_checks() -> None:
    """d768 (2 layers, bf16 compute, FA4 attention, sonic MoE): the AttnRes trunk, pool / own / slot rank 64 +
    embed_inject plain / plainonly."""
    import jmp
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.recipe import model_config

    cfg = model_config(shape="d768", layers=2)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    seq, split = cfg.max_seq_len, 2500
    key_tok, key_model = jax.random.split(jax.random.key(768), 2)
    method = dict(share="pool", query="own", proj="slot", proj_rank=64, inject_rank=64)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, 128, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32)
        w = 0.1 * jax.random.normal(jax.random.fold_in(key_model, 23), (cfg.hidden_dim, cfg.hidden_dim))

        def loss_fn(m):
            return mp.cast_to_compute(m).next_token_loss(tokens, loss_weight, mask=mask, reduction="mean",
                                                         logsumexp_weight=1e-4, return_router_metrics=True)

        _, old_metrics = eqx.filter_jit(loss_fn)(variant(base, key_model, passes=2, randomize=False, **method))
        old1, old2 = (float(old_metrics[f"train/cross_entropy_loss_pass{k}"]) for k in (1, 2))
        for mode in ("plain", "plainonly"):
            model = variant(base, key_model, passes=2, randomize=False, embed_inject=f"{mode}:0.01", **method)
            (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
            ce1, ce2 = (float(metrics[f"train/cross_entropy_loss_pass{k}"]) for k in (1, 2))
            live = max_abs(grads.embed_inject_proj.weight)
            want2 = old2 if mode == "plain" else ce1
            check(bool(jnp.isfinite(loss)) and finite and live > 0 and abs(ce1 - old1) <= 5e-3 and abs(ce2 - want2) <= 5e-3,
                  f"d768 attnres embed_inject {mode}:0.01 from the zero start: loss {float(loss):.4f}, pass-1 CE {ce1:.5f} "
                  f"vs {old1:.5f} without the knob, pass-2 CE {ce2:.5f} vs "
                  f"{'%.5f without the knob' % old2 if mode == 'plain' else 'its pass 1'}, gradients finite={finite}, "
                  f"W_E max |g| {live:.3g}")
            rand = eqx.tree_at(lambda m: m.embed_inject_proj.weight,
                               variant(base, key_model, passes=2, embed_inject=f"{mode}:0.25", **method), w)
            fwd = eqx.filter_jit(lambda t, m, rand=rand: mp.cast_to_compute(rand).run_passes(t, m, 2, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=3000, split=split, label=f"d768 attnres embed_inject {mode} 2 passes", tol=1e-2)


def stdres_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _stdres_checks()


def _stdres_checks() -> None:
    """Group 14: variant stdres, Marin's standard residual trunk + the sublayer feedback path (share pool) or FBT's
    fusion (share fbt)."""
    import tempfile

    import experiments.grug.moe_hero_ep.train as grug_train
    from levanter.checkpoint import load_checkpoint, save_checkpoint
    from alf.marin.recipe import optimizer_config

    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261111), 2)
    split, seq, short = 7, cfg.max_seq_len, 8
    fields = [f.name for f in dataclasses.fields(gm.Transformer) if f.name != "config"]
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        plain_marin = eqx.filter_jit(lambda m, t: m(t, mask)[0])(base, tokens)

        def grads(m):
            def loss_of(mm):
                return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                          return_router_metrics=True)
            (loss, metrics), g = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(m)
            return loss, metrics, g

        for share, query, proj, r in STDRES_CASES:
            label = f"{share}/{query}/{proj}" + ("" if proj == "none" else f" rank {r or 'full'}")
            kw = dict(share=share, query=query, proj=proj, proj_rank=r)
            model = stdres_variant(base, key_model, passes=2, **kw)
            outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(model, tokens)
            d0 = max_diff(outs[0], plain_marin)
            d1, d2, moved = max_diff(outs[0], refs[0]), max_diff(outs[1], refs[1]), max_diff(outs[1], outs[0])
            check(d0 <= 1e-5 and d1 <= 1e-5 and d2 <= 1e-5 and moved > 1e-3,
                  f"tiny 14a. stdres {label}: pass 1 == Marin's own Transformer (max |d|={d0:.3g}) and the reference "
                  f"({d1:.3g}), pass 2 == reference ({d2:.3g}), moved {moved:.3g}")
            model3 = stdres_variant(base, key_model, passes=3, **kw)
            fwd = eqx.filter_jit(lambda t, m, model=model3: model.run_passes(t, m, 3, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 14b. stdres {label} 3 passes", tol=0.0)
            causality(fwd, tokens, mask, pos=split, split=split, label=f"tiny 14b. stdres {label} 3 passes, doc start",
                      tol=0.0)
            model8 = stdres_variant(base, key_model, passes=short, **kw)
            jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
            exact = eqx.filter_jit(lambda m, t: ref_stdres_exact(m, t, mask8))(model8, tok8)
            two = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 2, train=False)[0][-1])(model8, tok8)
            check(max_diff(jac, exact) <= 1e-5 and max_diff(two, exact) > 1e-4,
                  f"tiny 14b. stdres {label}: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g}); "
                  f"2 passes differ by {max_diff(two, exact):.3g}")

        model = stdres_variant(base, key_model, passes=2)
        zero = eqx.tree_at(lambda m: m.frontier_logit.weight, model, jnp.full_like(model.frontier_logit.weight, -1e4))
        z = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(zero, tokens)
        kx, kp = jax.random.split(jax.random.fold_in(key_model, 19))
        xr = jax.random.normal(kx, (2, seq, cfg.hidden_dim))
        pv = jax.random.normal(kp, (2, seq, 2 * cfg.num_layers, cfg.hidden_dim))
        valid = jnp.ones((2, seq), bool).at[:, 0].set(False)
        mixed = [am.residual_mix(j, xr, (pv, valid, None), zero.frontier_logit.weight) for j in range(2 * cfg.num_layers)]
        live = am.residual_mix(0, xr, (pv, valid, None), model.frontier_logit.weight)
        exact = all(bool(jnp.array_equal(m, xr)) for m in mixed) and bool(jnp.array_equal(live[:, 0], xr[:, 0]))
        check(exact and max_diff(live, xr) > 1e-3 and max_diff(z[0], z[1]) <= 1e-6 and max_diff(z[0], plain_marin) <= 1e-5,
              f"tiny 14c. stdres zero share: residual_mix returns the stream bit for bit (every sublayer, and positions "
              f"without feedback at a live share: {exact}); a full pass 2 == pass 1 (max |d|={max_diff(z[0], z[1]):.3g}, "
              f"separately compiled) == Marin's Transformer ({max_diff(z[0], plain_marin):.3g})")

        _, _, g1 = grads(stdres_variant(base, key_model, passes=1, randomize=False))
        loss2, metrics2, g2 = grads(stdres_variant(base, key_model, passes=2, randomize=False))
        names = ("frontier_logit", "frontier_query", "frontier_proj_down", "frontier_proj_up")
        dormant = max(max_abs(getattr(g1, f).weight) for f in names)
        live = {f: max_abs(getattr(g2, f).weight) for f in names}
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
        ce = [float(metrics2[f"train/cross_entropy_loss_pass{k}"]) for k in (1, 2)]
        groups = am.optimizer_groups(stdres_variant(base, key_model, passes=2, randomize=False),
                                     optimizer_config(cfg, num_train_steps=100, batch_size=8))
        check(dormant == 0.0 and min(live["frontier_logit"], live["frontier_query"], live["frontier_proj_up"]) > 0
              and live["frontier_proj_down"] == 0.0 and finite and abs(float(loss2) - ce[0] - ce[1]) <= 1e-5
              and len(groups) == len(names) and all(v == "adam" for v in groups.values()),
              f"tiny 14d. stdres pool: one pass -> new-parameter gradients zero ({dormant:.3g}); first two-pass step "
              f"max |g| { {k: float('%.3g' % v) for k, v in live.items()} } (down zero while up is zero); finite={finite}; "
              f"loss {float(loss2):.5f} = pass-1 + pass-2 CE {ce[0]:.5f} + {ce[1]:.5f}; new parameters -> {groups}")
        fbt_kw = dict(share="fbt", query="shared", proj="none", proj_rank=0, randomize=False)
        _, _, g1 = grads(stdres_variant(base, key_model, passes=1, **fbt_kw))
        _, _, g2 = grads(stdres_variant(base, key_model, passes=2, **fbt_kw))
        dormant = max(max_abs(g1.fbt_state_proj.weight), max_abs(g1.fbt_token_gate.weight))
        live = min(max_abs(g2.fbt_state_proj.weight), max_abs(g2.fbt_token_gate.weight))
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
        groups = am.optimizer_groups(stdres_variant(base, key_model, passes=2, **fbt_kw),
                                     optimizer_config(cfg, num_train_steps=100, batch_size=8))
        check(dormant == 0.0 and live > 0 and finite and {k.split(".")[0] for k in groups} == {"fbt_state_proj", "fbt_token_gate"}
              and all(v == "adam" for v in groups.values()),
              f"tiny 14d. stdres fbt: one pass -> fusion gradients zero ({dormant:.3g}); two passes -> min |g| {live:.3g}, "
              f"finite={finite}; new parameters -> {groups}")

        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint({"model": base}, step=1, checkpoint_path=tmp, is_temporary=False)
            for share, query, proj, r in (("pool", "own", "slot", 4), ("fbt", "shared", "none", 0)):
                template = stdres_variant(base, key_model, share=share, query=query, passes=2, proj=proj, proj_rank=r,
                                          randomize=False)
                try:
                    load_checkpoint({"model": template}, tmp, allow_partial=False)
                    refused = False
                except Exception:  # noqa: BLE001 - any refusal of the strict load is the expected outcome
                    refused = True
                restored = load_checkpoint({"model": template}, tmp, allow_partial=True)["model"]
                kept = max_diff([getattr(restored, f) for f in fields], [getattr(base, f) for f in fields])
                filled = max_diff([getattr(restored, f) for f in STDRES_NEW if getattr(template, f) is not None],
                                  [getattr(template, f) for f in STDRES_NEW if getattr(template, f) is not None])
                runs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(restored, tokens)
                pass1 = max_diff(runs[0], plain_marin)
                check(refused and kept == 0.0 and filled == 0.0 and restored.attnres_query is None and pass1 <= 1e-5,
                      f"tiny 14e. stdres {share} branched from a plain Marin checkpoint: strict load refused={refused}, "
                      f"saved leaves kept (max |d|={kept:.3g}), new leaves from the template ({filled:.3g}), pass 1 == "
                      f"Marin's Transformer ({pass1:.3g}); pass 2 at the branch moves {max_diff(runs[1], runs[0]):.3g}")

        drop, flops = grug_train._drop_metrics, grug_train._compute_flops
        am.install_stdres(passes=2, eval_passes=3, share="pool", query="own", proj="slot", proj_rank=4, bos_id=BOS)
        built = grug_train.Transformer.init(cfg, key=key_model)
        grug_train.Transformer, grug_train._drop_metrics, grug_train._compute_flops = gm.Transformer, drop, flops
        am._STDRES_SPEC.update(passes=1, eval_passes=1, share="pool", query="own", proj="none", proj_rank=0, bos_id=-1)
        backbone = max_diff([getattr(built, f) for f in fields], [getattr(base, f) for f in fields])
        check(isinstance(built, am.StdResTransformer) and built.passes == 2 and built.eval_passes == 3
              and built.feedback_share == "pool" and built.frontier_proj_up.weight.shape == (2 * cfg.num_layers, 4, cfg.hidden_dim)
              and backbone == 0.0 and built.attnres_query is None,
              f"tiny 14f. install_stdres(): Transformer.init -> StdResTransformer(passes 2, eval 3, pool, slot maps "
              f"{built.frontier_proj_up.weight.shape}), backbone == Marin init (|d|={backbone}), no AttnRes parameters")

        rejected = []
        for kw in ({"share": "pool", "query": "shared"}, {"share": "fbt", "proj": "slot", "proj_rank": 4},
                   {"share": "joint"}, {"share": "pool", "proj": "source", "proj_rank": 4}):
            try:
                am.StdResTransformer.from_base(base, key=key_model, **kw)
            except ValueError:
                rejected.append(kw)
        check(len(rejected) == 4, f"tiny 14g. stdres rejects pool + shared query, fbt + maps, share joint, proj source "
                                  f"({len(rejected)} / 4 rejected)")


def stdres_d768_checks() -> None:
    """d768 (2 layers, bf16 compute, FA4 attention, sonic MoE): variant stdres pool (rank 64) and fbt."""
    import jmp
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.recipe import model_config

    cfg = model_config(shape="d768", layers=2)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    seq, split = cfg.max_seq_len, 2500
    key_tok, key_model = jax.random.split(jax.random.key(768), 2)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, 128, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32)
        plain = eqx.filter_jit(lambda m: mp.cast_to_compute(m).next_token_loss(
            tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4))(base)
        for share, query, proj, r in (("pool", "own", "slot", 64), ("fbt", "shared", "none", 0)):
            model = stdres_variant(base, key_model, share=share, query=query, passes=2, proj=proj, proj_rank=r)

            def loss_fn(m):
                return mp.cast_to_compute(m).next_token_loss(tokens, loss_weight, mask=mask, reduction="mean",
                                                             logsumexp_weight=1e-4, return_router_metrics=True)

            (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
            ce1 = float(metrics["train/cross_entropy_loss_pass1"])
            check(bool(jnp.isfinite(loss)) and finite and abs(ce1 - float(plain)) <= 5e-3
                  and metrics["qb_beta_per_layer"].shape == (cfg.num_layers, cfg.num_experts),
                  f"d768 stdres {share}: loss {float(loss):.4f}, pass-1 CE {ce1:.5f} vs Marin's Transformer {float(plain):.5f}, "
                  f"gradients finite={finite}, max |grad|={max_abs(grads):.3g}")
            fwd = eqx.filter_jit(lambda t, m, model=model: mp.cast_to_compute(model).run_passes(t, m, 2, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=3000, split=split, label=f"d768 stdres {share} 2 passes", tol=1e-2)


FBT_NORMS = ("unit", "embed", "match")


def fbtnorm_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _fbtnorm_checks()


def _fbtnorm_checks() -> None:
    """Group 17: fbt_input_norm of FBT's fused input (share fbt) on Marin's residual trunk and on AttnRes."""
    import tempfile

    from levanter.checkpoint import load_checkpoint, save_checkpoint

    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261112), 2)
    split, seq, short = 7, cfg.max_seq_len, 8
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        # a small embedding gain, as in a trained Marin run (x0 far below RMS 1), so the three scales differ
        base = eqx.tree_at(lambda m: m.embed_norm.weight, base, 0.1 * jnp.ones_like(base.embed_norm.weight))
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        plain_marin = eqx.filter_jit(lambda m, t: m(t, mask)[0])(base, tokens)

        def build(trunk, norm, passes, randomize=True):
            if trunk == "stdres":
                return stdres_variant(base, key_model, share="fbt", query="shared", passes=passes, proj="none",
                                      proj_rank=0, randomize=randomize, fbt_input_norm=norm)
            return variant(base, key_model, share="fbt", passes=passes, inject_rank=4, randomize=randomize,
                           fbt_input_norm=norm)

        def slot0(m, t):
            ctx = m._context(t, mask)
            x0 = m._embed(t)
            top = m.run_passes(t, mask, 1, train=False)[0][-1]
            return m._fbt_input(t, x0, top, ctx, train=False, pass_idx=0), x0, ctx["fb_valid"]

        def rms(x):
            return np.sqrt(np.mean(np.square(np.asarray(x, np.float64)), axis=-1))

        for trunk in ("stdres", "attnres"):
            ref_passes, ref_exact = ((ref_stdres_passes, ref_stdres_exact) if trunk == "stdres"
                                     else (ref_fbt_passes, ref_fbt_exact))
            outs, inputs = {}, {}
            for norm in FBT_NORMS:
                model = build(trunk, norm, 2)
                outs[norm] = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
                refs = eqx.filter_jit(lambda m, t: ref_passes(m, t, mask, 2))(model, tokens)
                d1, d2 = max_diff(outs[norm][0], refs[0]), max_diff(outs[norm][1], refs[1])
                same1 = max_diff(outs[norm][0], outs["unit"][0])
                check(model.fbt_input_norm == norm and d1 <= 1e-5 and d2 <= 1e-5 and same1 == 0.0,
                      f"tiny 17a. {trunk} fbt {norm}: pass 1 == reference ({d1:.3g}) and == unit's pass 1 ({same1:.3g}), "
                      f"pass 2 == reference ({d2:.3g})")
                inputs[norm] = eqx.filter_jit(slot0)(model, tokens)
            fbv = np.asarray(inputs["unit"][2])
            x0 = inputs["unit"][1]
            r_x0 = rms(x0)[fbv]
            r_unit, r_embed, r_match = (rms(inputs[n][0])[fbv] for n in ("unit", "embed", "match"))
            r_target = np.sqrt(r_x0 ** 2 + am.ATTN_RES_EPS)  # the scale carries the norm's eps
            embed_ref = base.embed_gated_norm(base.embed_norm(inputs["unit"][0]))
            d_embed = float(np.abs(np.asarray(inputs["embed"][0], np.float32) - np.asarray(embed_ref, np.float32))[fbv].max())
            starts_kept = max(float(np.abs(np.asarray(inputs[n][0]) - np.asarray(x0))[~fbv].max()) for n in FBT_NORMS)
            check(np.abs(r_unit - 1).max() < 1e-3 and np.abs(r_match / r_target - 1).max() < 1e-4 and d_embed <= 1e-5
                  and starts_kept == 0.0 and np.median(r_unit / r_x0) > 3.0,
                  f"tiny 17b. {trunk} fused input scale at feedback positions: unit RMS 1 (max |r-1|={np.abs(r_unit - 1).max():.2g}, "
                  f"{np.median(r_unit / r_x0):.1f}x x0), match = sqrt(rms(x0)^2 + eps) "
                  f"(max rel {np.abs(r_match / r_target - 1).max():.2g}), "
                  f"embed = the embedding norms of unit's input ({d_embed:.2g}; {np.median(r_embed / r_x0):.2f}x x0); "
                  f"document starts keep x0 ({starts_kept:.2g})")
            moved = min(max_diff(outs[n][1], outs["unit"][1]) for n in ("embed", "match"))
            check(moved > 1e-3, f"tiny 17a. {trunk}: embed / match move pass 2 off unit's by >= {moved:.3g}")
            for norm in ("embed", "match"):
                model3 = build(trunk, norm, 3)
                fwd = eqx.filter_jit(lambda t, m, model=model3: model.run_passes(t, m, 3, train=False)[0][-1])
                causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 17c. {trunk} fbt {norm} 3 passes", tol=0.0)
                causality(fwd, tokens, mask, pos=split, split=split, label=f"tiny 17c. {trunk} fbt {norm} 3 passes, doc start",
                          tol=0.0)
                model8 = build(trunk, norm, short)
                jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, short, train=False)[0][-1])(model8, tok8)
                exact = eqx.filter_jit(lambda m, t: ref_exact(m, t, mask8))(model8, tok8)
                check(max_diff(jac, exact) <= 1e-5,
                      f"tiny 17d. {trunk} fbt {norm}: {short} Jacobi passes == exact recursion (max |d|={max_diff(jac, exact):.3g})")

                def grads(m):
                    def loss_of(mm):
                        return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                                  return_router_metrics=True)
                    (loss, _), g = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(m)
                    return loss, g

                _, g1 = grads(build(trunk, norm, 1, randomize=False))
                loss2, g2 = grads(build(trunk, norm, 2, randomize=False))
                dormant = max(max_abs(g1.fbt_state_proj.weight), max_abs(g1.fbt_token_gate.weight))
                live = min(max_abs(g2.fbt_state_proj.weight), max_abs(g2.fbt_token_gate.weight))
                finite = bool(jnp.isfinite(loss2)) and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g2))
                check(dormant == 0.0 and live > 0 and finite,
                      f"tiny 17e. {trunk} fbt {norm}: one pass -> fusion gradients zero ({dormant:.3g}); two passes -> "
                      f"min |g| {live:.3g}, finite={finite}")

        rejected = []
        for make in (lambda: am.StdResTransformer.from_base(base, key=key_model, share="pool", fbt_input_norm="embed"),
                     lambda: am.AttnResTransformer.from_base(base, key=key_model, share="pool", fbt_input_norm="match"),
                     lambda: am.StdResTransformer.from_base(base, key=key_model, share="fbt", query="shared",
                                                            fbt_input_norm="rms1")):
            try:
                make()
            except ValueError:
                rejected.append(make)
        check(len(rejected) == 3, f"tiny 17f. rejects embed / match off share fbt and an unknown fbt_input_norm "
                                  f"({len(rejected)} / 3 rejected)")

        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint({"model": base}, step=1, checkpoint_path=tmp, is_temporary=False)
            for norm in ("embed", "match"):
                template = build("stdres", norm, 2, randomize=False)
                restored = load_checkpoint({"model": template}, tmp, allow_partial=True)["model"]
                runs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(restored, tokens)
                pass1 = max_diff(runs[0], plain_marin)
                check(restored.fbt_input_norm == norm and pass1 <= 1e-5,
                      f"tiny 17g. stdres fbt {norm} branched from a plain Marin checkpoint (partial restore): pass 1 == "
                      f"Marin's Transformer ({pass1:.3g}), pass 2 moves {max_diff(runs[1], runs[0]):.3g}")

        with tempfile.TemporaryDirectory() as run_dir:
            first = am.read_fbt_input_norm(run_dir)
            am.record_fbt_input_norm(run_dir, "match")
            am.record_fbt_input_norm(run_dir, "match")
            outcomes = []
            for call in (lambda: am.record_fbt_input_norm(run_dir, "unit"),
                         lambda: am.run_fbt_input_norm(run_dir, "pool")):
                try:
                    call()
                    outcomes.append(False)
                except ValueError:
                    outcomes.append(True)
            got = am.run_fbt_input_norm(run_dir, "fbt")
            check(first == "unit" and got == "match" and all(outcomes),
                  f"tiny 17h. run record: no file -> {first!r}; recorded match -> {got!r}; a resume with unit refused="
                  f"{outcomes[0]}; evaluating it as share pool refused={outcomes[1]}")



def fbtnorm_d768_checks() -> None:
    """d768 (2 layers, bf16 compute, FA4 attention, sonic MoE): stdres fbt with embed / match."""
    import jmp
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.recipe import model_config

    cfg = model_config(shape="d768", layers=2)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    seq, split = cfg.max_seq_len, 2500
    key_tok, key_model = jax.random.split(jax.random.key(768), 2)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, 128, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32)
        plain = eqx.filter_jit(lambda m: mp.cast_to_compute(m).next_token_loss(
            tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4))(base)
        for norm in ("embed", "match"):
            model = stdres_variant(base, key_model, share="fbt", query="shared", passes=2, proj="none", proj_rank=0,
                                   fbt_input_norm=norm)

            def loss_fn(m):
                return mp.cast_to_compute(m).next_token_loss(tokens, loss_weight, mask=mask, reduction="mean",
                                                             logsumexp_weight=1e-4, return_router_metrics=True)

            (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
            ce1 = float(metrics["train/cross_entropy_loss_pass1"])
            live = min(max_abs(grads.fbt_state_proj.weight), max_abs(grads.fbt_token_gate.weight))
            check(bool(jnp.isfinite(loss)) and finite and live > 0 and abs(ce1 - float(plain)) <= 5e-3,
                  f"d768 stdres fbt {norm}: loss {float(loss):.4f}, pass-1 CE {ce1:.5f} vs Marin's Transformer "
                  f"{float(plain):.5f}, gradients finite={finite}, fusion min |g| {live:.3g}")
            fwd = eqx.filter_jit(lambda t, m, model=model: mp.cast_to_compute(model).run_passes(t, m, 2, train=False)[0][-1])
            causality(fwd, tokens, mask, pos=3000, split=split, label=f"d768 stdres fbt {norm} 2 passes", tol=1e-2)


def fbt_feedback_mask_ref(starts, draws):
    """FBT's build_feedback_mask(prefix_mixin=True) written out in numpy, the document's U[0, 1) being draws[start] / 2^19
    (the quantized uniform of the model's prefix_mixin_draws): plain_length = 1 + floor(u * length), read where the
    offset from the document start is >= plain_length."""
    starts = np.asarray(starts, bool).copy()
    starts[:, 0] = True
    b, t = starts.shape
    out = np.zeros((b, t), bool)
    for r in range(b):
        bounds = list(np.flatnonzero(starts[r])) + [t]
        for a, e in zip(bounds[:-1], bounds[1:]):
            plain = 1 + ((int(draws[r, a]) * (e - a)) >> am.PREFIX_MIXIN_BITS)
            out[r, a + plain:e] = True
    return out


def ref_stdres_passes_read(model, tokens, mask, n, reads):
    """ref_stdres_passes with pass k + 1 reading the previous position only where reads[k] (prefix mixin)."""
    out, outs = ref_stdres_trunk(model, tokens, mask)
    res = [out]
    for k in range(1, n):
        out, outs = ref_stdres_trunk(model, tokens, mask, **ref_stdres_next(model, tokens, reads[k], out, outs))
        res.append(out)
    return res


def ref_fbt_input_jitter(model, tokens, top, fb_valid, pass_idx):
    """ref_fbt_input with the training jitter of _fbt_input: U[-j, j] times each position's RMS, same key."""
    h = jnp.pad(top[:, :-1], ((0, 0), (1, 0), (0, 0)))
    rms = jnp.sqrt(jnp.mean(h * h, -1, keepdims=True))
    noise = jax.random.uniform(model._jitter_key(tokens, pass_idx), h.shape, jnp.float32, -model.jitter, model.jitter,
                               out_sharding=jax.sharding.PartitionSpec(*tuple(jax.typeof(top).sharding.spec)))
    h = h + noise * rms
    e = gm._embedding_gather(model.token_embed, tokens)
    x0 = model.embed_gated_norm(model.embed_norm(e))
    fused = ref_norm((h @ model.fbt_state_proj.weight) * jax.nn.sigmoid(ref_norm(e) @ model.fbt_token_gate.weight))
    return jnp.where(fb_valid[..., None], ref_fbt_scaled(model, fused, x0), x0)


def pm_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _pm_checks()


def _pm_checks() -> None:
    """Group 15: FBT's training recipe on variant stdres, prefix mixin and jitter."""
    import experiments.grug.moe_hero_ep.train as grug_train

    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261112), 2)
    seq, split = cfg.max_seq_len, 7
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        rng = np.random.default_rng(20261111)
        n_rows, t_long = 48, 64
        starts = rng.random((n_rows, t_long)) < 0.12
        starts[:, 0] = True
        fb_valid = jnp.asarray(~starts)
        spec = jax.sharding.PartitionSpec(*tuple(jax.typeof(fb_valid).sharding.spec))
        mask_jit = jax.jit(am.prefix_mixin_mask)
        mismatches, jit_diff, start_read, full_plain, frac, masks = 0, 0, 0, 0, [], []
        for i in range(24):
            key = jax.random.key(4000 + i)
            got = np.asarray(am.prefix_mixin_mask(fb_valid, key))
            draws = np.asarray(am.prefix_mixin_draws(key, (n_rows, t_long), spec))
            mismatches += int((got != fbt_feedback_mask_ref(starts, draws)).sum())
            jit_diff += int((np.asarray(mask_jit(fb_valid, key)) != got).sum())
            start_read += int((got & starts).sum())
            for r in range(n_rows):
                bounds = list(np.flatnonzero(starts[r])) + [t_long]
                for a, e in zip(bounds[:-1], bounds[1:]):
                    full_plain += int(not got[r, a:e].any())
                    frac.append(float(got[r, a:e].mean()))
            masks.append(got)
        distinct = len({m.tobytes() for m in masks})
        check(mismatches == 0 and jit_diff == 0 and start_read == 0 and full_plain > 0 and 0.3 < np.mean(frac) < 0.6
              and distinct == len(masks),
              f"tiny 15a. prefix_mixin_mask == FBT build_feedback_mask(prefix_mixin=True) on {len(masks)} draws x {n_rows} "
              f"packed rows ({mismatches} mismatches, jit {jit_diff}); document starts never read ({start_read}); "
              f"{full_plain} documents drew a whole-document plain prefix; mean read fraction {np.mean(frac):.3f}; "
              f"{distinct} / {len(masks)} keys give distinct masks")

        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)
        for share, query, proj, r in (("pool", "own", "slot", 4), ("fbt", "shared", "none", 0)):
            model = dataclasses.replace(stdres_variant(base, key_model, share=share, query=query, passes=3, proj=proj,
                                                       proj_rank=r), prefix_mixin=True)
            plain_model = dataclasses.replace(model, prefix_mixin=False)
            train_outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=True)[0])(model, tokens)
            fbv = model._context(tokens, mask)["fb_valid"]
            reads = [fbv] + [am.prefix_mixin_mask(fbv, model._jitter_key(tokens, am.PREFIX_MIXIN_KEY_OFFSET + k))
                             for k in (1, 2)]
            refs = eqx.filter_jit(lambda m, t, rd: ref_stdres_passes_read(m, t, mask, 3, rd))(model, tokens, reads)
            d = max(max_diff(o, rf) for o, rf in zip(train_outs, refs))
            no_pm = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=True)[0])(plain_model, tokens)
            moved = max_diff(train_outs[1], no_pm[1])
            ev = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=False)[0])(model, tokens)
            ev_plain = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=False)[0])(plain_model, tokens)
            same_eval = max(max_diff(a, b) for a, b in zip(ev, ev_plain))
            read_frac = float(np.mean([np.asarray(rd).mean() for rd in reads[1:]]))
            check(d <= 1e-5 and moved > 1e-4 and same_eval == 0.0 and max_diff(train_outs[0], no_pm[0]) == 0.0,
                  f"tiny 15b. stdres {share} prefix mixin (train): passes 1-3 == the reference reading only where each pass's "
                  f"mask says (max |d|={d:.3g}; masks read {read_frac:.2f} of positions), pass 2 moved {moved:.3g} from no "
                  f"mixin, pass 1 unchanged; evaluation unaffected (max |d|={same_eval:.3g})")

        model = dataclasses.replace(stdres_variant(base, key_model, share="fbt", query="shared", passes=2, proj="none",
                                                   proj_rank=0), jitter=0.02)
        outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=True)[0])(model, tokens)
        fbv = model._context(tokens, mask)["fb_valid"]
        ref2 = eqx.filter_jit(lambda m, t, o: ref_stdres_trunk(m, t, mask, x0=ref_fbt_input_jitter(m, t, o, fbv, 0))[0])(
            model, tokens, outs[0])
        quiet = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=True)[0])(dataclasses.replace(model, jitter=0.0), tokens)
        ev = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
        ev0 = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(dataclasses.replace(model, jitter=0.0), tokens)
        dj, moved = max_diff(outs[1], ref2), max_diff(outs[1], quiet[1])
        check(dj <= 1e-5 and moved > 1e-5 and max(max_diff(a, b) for a, b in zip(ev, ev0)) == 0.0,
              f"tiny 15c. stdres fbt jitter 0.02 (train): pass 2 == the reference with the same U[-0.02, 0.02] x RMS noise on "
              f"the previous top state (max |d|={dj:.3g}), moved {moved:.3g} from no jitter; evaluation noise-free")

        for share, query, proj, r in (("pool", "own", "slot", 4), ("fbt", "shared", "none", 0)):
            m = dataclasses.replace(stdres_variant(base, key_model, share=share, query=query, passes=2, proj=proj,
                                                   proj_rank=r, randomize=False), prefix_mixin=True, jitter=0.02)

            def loss_of(mm):
                return mm.next_token_loss(tokens, loss_weight, mask=mask, reduction="mean", logsumexp_weight=1e-4,
                                          return_router_metrics=True)

            (loss, _), g = eqx.filter_jit(eqx.filter_value_and_grad(loss_of, has_aux=True))(m)
            finite = bool(jnp.isfinite(loss)) and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g))
            new = [f for f in STDRES_NEW if getattr(g, f) is not None]
            live = {f: max_abs(getattr(g, f).weight) for f in new}
            check(finite and any(v > 0 for v in live.values()),
                  f"tiny 15d. stdres {share} prefix mixin + jitter 0.02: loss {float(loss):.4f}, gradients finite={finite}, "
                  f"max |g| { {k: float('%.3g' % v) for k, v in live.items()} }")

        drop, flops = grug_train._drop_metrics, grug_train._compute_flops
        am.install_stdres(passes=2, eval_passes=3, share="fbt", query="shared", jitter=0.02, prefix_mixin=True, bos_id=BOS)
        built = grug_train.Transformer.init(cfg, key=key_model)
        grug_train.Transformer, grug_train._drop_metrics, grug_train._compute_flops = gm.Transformer, drop, flops
        am._STDRES_SPEC.update(passes=1, eval_passes=1, share="pool", query="own", jitter=0.0, prefix_mixin=False, bos_id=-1)
        rejected = False
        try:
            am.StdResTransformer.from_base(base, key=key_model, prefix_mixin=1)
        except ValueError:
            rejected = True
        check(isinstance(built, am.StdResTransformer) and built.prefix_mixin and built.jitter == 0.02
              and built.feedback_share == "fbt" and rejected,
              f"tiny 15e. install_stdres(prefix_mixin=True, jitter=0.02) -> StdResTransformer(prefix_mixin="
              f"{built.prefix_mixin}, jitter {built.jitter}); a non-bool prefix_mixin is rejected ({rejected})")


def pm_d768_checks() -> None:
    """d768 real kernels with FBT's recipe on variant stdres: prefix mixin + jitter 0.02, training path (loss and gradients)."""
    import jmp
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.recipe import model_config

    cfg = model_config(shape="d768", layers=2)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    seq, split = cfg.max_seq_len, 2500
    key_tok, key_model = jax.random.split(jax.random.key(769), 2)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, 128, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32)
        for share, query, proj, r in (("fbt", "shared", "none", 0), ("pool", "own", "slot", 64)):
            model = dataclasses.replace(stdres_variant(base, key_model, share=share, query=query, passes=2, proj=proj,
                                                       proj_rank=r), prefix_mixin=True, jitter=0.02)

            def loss_fn(m):
                return mp.cast_to_compute(m).next_token_loss(tokens, loss_weight, mask=mask, reduction="mean",
                                                             logsumexp_weight=1e-4, return_router_metrics=True)

            (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
            check(bool(jnp.isfinite(loss)) and finite,
                  f"d768 stdres {share} prefix mixin + jitter 0.02: loss {float(loss):.4f} (pass 1 "
                  f"{float(metrics['train/cross_entropy_loss_pass1']):.4f}, "
                  f"pass 2 {float(metrics['train/cross_entropy_loss_pass2']):.4f}), "
                  f"gradients finite={finite}, max |grad|={max_abs(grads):.3g}")


def prefill_family(model, tokens, mask):
    """(trunk(**reads) -> (top, outs), nxt(fb_valid, top, outs) -> reads of the next pass) for the reference of `model`."""
    if isinstance(model, am.StdResTransformer):
        return (lambda **kw: ref_stdres_trunk(model, tokens, mask, **kw),
                lambda fbv, top, outs: ref_stdres_next(model, tokens, fbv, top, outs))

    def trunk(**kw):
        out, outs, _ = ref_trunk(model, tokens, mask, **kw)
        return out, outs

    def nxt(fbv, top, outs):
        if model.feedback_share == "fbt":
            return {"x0": ref_fbt_input(model, tokens, top, fbv)}
        return {"prev": ref_prev(model, outs, fbv)}

    return trunk, nxt


def pick_reads(context, held, fresh):
    """Per position: the prompt's reads (held) at prompt positions, the decoded reads (fresh) elsewhere."""
    def sel(h, f):
        return jnp.where(jnp.reshape(context, context.shape + (1,) * (f.ndim - 2)), h, f)

    return jax.tree.map(sel, held, fresh)


def ref_prefill_exact(model, tokens, mask, context, c):
    """Prompt (context) through c Jacobi passes, kept; the rest decoded token by token, position t reading the exact state
    of t - 1 (the prompt's kept state when t - 1 is in the prompt). c = 1: SOFT, the prompt is a plain prefill."""
    trunk, nxt = prefill_family(model, tokens, mask)
    _, _, fbv = ref_masks(model, tokens, mask)
    top, outs = trunk()
    hist = [(top, outs)]
    for _ in range(c - 1):
        top, outs = trunk(**nxt(fbv, top, outs))
        hist.append((top, outs))
    held = None if c == 1 else nxt(fbv, *hist[c - 2])
    exact_top, exact_outs = jnp.zeros_like(top), [jnp.zeros_like(v) for v in outs]
    for t in range(tokens.shape[1]):
        if t:
            fresh = nxt(fbv & ~context if c == 1 else fbv, exact_top, exact_outs)
            top, outs = trunk(**(fresh if held is None else pick_reads(context, held, fresh)))
        elif held is not None:
            top, outs = trunk(**held)
        exact_top = exact_top.at[:, t].set(top[:, t])
        exact_outs = [e.at[:, t].set(v[:, t]) for e, v in zip(exact_outs, outs)]
    return exact_top


def prompt_mask(model, tokens, mask, prompt_len):
    fbv = model._context(tokens, mask)["fb_valid"]
    return am.document_offsets(fbv) < prompt_len


PREFILL_CASES = (("stdres", "pool"), ("stdres", "fbt"), ("attnres", "pool"), ("attnres", "fbt"))


def prefill_variant(base, key, backbone, share, passes):
    # `backbone` names the trunk, not the helper `variant()` below, which builds the model.
    if backbone == "stdres":
        return stdres_variant(base, key, share=share, query="own" if share == "pool" else "shared", passes=passes,
                              proj="slot" if share == "pool" else "none", proj_rank=4 if share == "pool" else 0)
    return variant(base, key, share=share, query="own" if share == "pool" else "shared", passes=passes,
                   proj="slot" if share == "pool" else "none", proj_rank=4 if share == "pool" else 0)


def prefill_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _prefill_checks()
        _turns_checks()


def turns_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _turns_checks()


def _turns_checks() -> None:
    """16e: multi-turn decoding, a context mask that alternates plain user turns with generated turns inside every
    document; K = T Jacobi passes == user turns through one plain pass and generated turns decoded token by token from
    the exact state of the previous position, user turns attending to the generated ones."""
    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261114), 2)
    short = 8
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        for backbone, share in PREFILL_CASES:
            label = f"{backbone} {share}"
            m8 = prefill_variant(base, key_model, backbone, share, short)
            off = am.document_offsets(m8._context(tok8, mask8)["fb_valid"])
            # turns of two tokens: row 0 holds two documents (user, generated each), row 1 one document with two of each
            ctx = (off // 2) % 2 == 0
            jac = eqx.filter_jit(lambda m, t, c: m.run_passes(t, mask8, short, train=False, context=c,
                                                              context_passes=1)[0][-1])(m8, tok8, ctx)
            ref = eqx.filter_jit(lambda m, t, c: ref_prefill_exact(m, t, mask8, c, 1))(m8, tok8, ctx)
            prompt = eqx.filter_jit(lambda m, t, c: ref_prefill_exact(m, t, mask8, c, 1))(m8, tok8, off < 2)
            n_turns = int(np.asarray(ctx[1]).sum())
            check(max_diff(jac, ref) <= 1e-5 and max_diff(ref, prompt) > 1e-5 and n_turns == 4,
                  f"tiny 16e. {label}: {short} Jacobi passes with alternating two-token turns == plain user turns and "
                  f"generated turns decoded token by token (max |d|={max_diff(jac, ref):.3g}); differs from a single "
                  f"prompt by {max_diff(ref, prompt):.3g}")


def _prefill_checks() -> None:
    """Group 16: decoding modes (run_passes context / context_passes): SOFT, FUSED with a 2-pass prompt, standard."""
    cfg = tiny_config()
    key_tok, key_model = jax.random.split(jax.random.key(20261113), 2)
    seq, split, short = cfg.max_seq_len, 7, 8
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, cfg.vocab_size, key_tok)
        tok8, mask8 = packed_batch(short, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        loss_weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0.0)

        rng = np.random.default_rng(20261113)
        starts = rng.random((32, 64)) < 0.1
        starts[:, 0] = True
        got = np.asarray(am.document_offsets(jnp.asarray(~starts)))
        want = np.zeros_like(got)
        for r in range(starts.shape[0]):
            last = 0
            for t in range(starts.shape[1]):
                last = t if starts[r, t] else last
                want[r, t] = t - last
        check(np.array_equal(got, want), f"tiny 16a. document_offsets == the offset from the last document start "
                                         f"({int((got != want).sum())} mismatches over 32 packed rows)")

        for backbone, share in PREFILL_CASES:
            label = f"{backbone} {share}"
            model = prefill_variant(base, key_model, backbone, share, 3)
            ctx = prompt_mask(model, tokens, mask, 3)
            run = eqx.filter_jit(lambda m, t, c, cp: m.run_passes(t, mask, 3, train=False, context=c, context_passes=cp)[0],
                                 )
            full = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=False)[0])(model, tokens)
            none_ctx = jnp.zeros_like(ctx)
            d_empty = max(max_diff(run(model, tokens, none_ctx, cp), full) for cp in (1, 2))
            d_all = max(max_diff(run(model, tokens, ctx, cp), full) for cp in (3, 5))
            soft = run(model, tokens, ctx, 1)
            fused2 = run(model, tokens, ctx, 2)
            c3 = ctx[..., None]
            soft_prompt = max(max_diff(jnp.where(c3, o, 0.0), jnp.where(c3, full[0], 0.0)) for o in soft)
            f2_prompt = max(max_diff(jnp.where(c3, o, 0.0), jnp.where(c3, full[1], 0.0)) for o in fused2[1:])
            moved = min(max_diff(soft[-1], full[-1]), max_diff(fused2[-1], full[-1]), max_diff(soft[-1], fused2[-1]))
            check(d_empty <= 1e-6 and d_all <= 1e-6 and soft_prompt <= 1e-6 and f2_prompt <= 1e-6 and moved > 1e-5,
                  f"tiny 16b. {label}: an empty prompt == every position reading (c = 1, 2; max |d|={d_empty:.3g}); "
                  f"c >= passes == every position reading ({d_all:.3g}); SOFT keeps the prompt at pass 1 in every pass "
                  f"({soft_prompt:.3g}), c = 2 keeps it at pass 2 ({f2_prompt:.3g}); modes differ after the prompt "
                  f"(min {moved:.3g})")

            m8 = prefill_variant(base, key_model, backbone, share, short)
            # c passes leave the first c prompt positions exact, so a prompt longer than c is what makes c visible
            ctx8 = prompt_mask(m8, tok8, mask8, 3)
            for cp in (1, 2):
                jac = eqx.filter_jit(lambda m, t, c, cp=cp: m.run_passes(t, mask8, short, train=False, context=c,
                                                                         context_passes=cp)[0][-1])(m8, tok8, ctx8)
                ref = eqx.filter_jit(lambda m, t, c, cp=cp: ref_prefill_exact(m, t, mask8, c, cp))(m8, tok8, ctx8)
                exact = eqx.filter_jit(lambda m, t: ref_prefill_exact(m, t, mask8, jnp.zeros_like(ctx8), 1))(m8, tok8)
                check(max_diff(jac, ref) <= 1e-5 and max_diff(ref, exact) > 1e-5,
                      f"tiny 16c. {label} c = {cp}: {short} Jacobi passes == prompt (3 tokens per document) through {cp} "
                      f"pass{'es' if cp > 1 else ''} then exact token-by-token decoding (max |d|={max_diff(jac, ref):.3g}); "
                      f"differs from decoding with no prompt by {max_diff(ref, exact):.3g}")

            fwd = eqx.filter_jit(lambda t, m, model=model, cp=1: model.run_passes(
                t, m, 3, train=False, context=prompt_mask(model, t, m, 3), context_passes=cp)[0][-1])
            causality(fwd, tokens, mask, pos=11, split=split, label=f"tiny 16d. {label} SOFT", tol=0.0)
            fwd2 = eqx.filter_jit(lambda t, m, model=model, cp=2: model.run_passes(
                t, m, 3, train=False, context=prompt_mask(model, t, m, 3), context_passes=cp)[0][-1])
            causality(fwd2, tokens, mask, pos=11, split=split, label=f"tiny 16d. {label} FUSED c = 2", tol=0.0)

            losses = eqx.filter_jit(lambda m, t, c: m.pass_losses(t, loss_weight, mask=mask, n_passes=3, context=c,
                                                                  context_passes=1))(model, tokens, ctx)
            m_eval = dataclasses.replace(model, eval_passes=3)
            plain_loss = eqx.filter_jit(lambda m, t: m.next_token_loss(t, loss_weight, mask=mask, reduction="none"))(m_eval, tokens)
            loss_full = eqx.filter_jit(lambda m, t: m.pass_losses(t, loss_weight, mask=mask, n_passes=3, keep=(1, 3)))(model, tokens)
            labels = np.pad(np.asarray(tokens)[:, 1:], ((0, 0), (0, 1)))
            logits = np.asarray(soft[-1], np.float64) @ np.asarray(model.output_proj, np.float64)
            top_logit = logits.max(-1, keepdims=True)
            lse = top_logit[..., 0] + np.log(np.exp(logits - top_logit).sum(-1))
            lw = np.asarray(loss_weight, np.float64)
            ref_ce = (lse - np.take_along_axis(logits, labels[..., None], -1)[..., 0]) * lw
            d_ce = float(np.max(np.abs(np.asarray(losses[2], np.float64) * lw - ref_ce)))
            check(max_diff(loss_full[1], plain_loss) <= 1e-5 and d_ce <= 1e-4 and len(loss_full) == 2,
                  f"tiny 16e. {label} pass_losses: last pass == next_token_loss(reduction none) (max |d|="
                  f"{max_diff(loss_full[1], plain_loss):.3g}); SOFT pass 3 == CE of its hidden state ({d_ce:.3g})")

        model = prefill_variant(base, key_model, "stdres", "pool", 2)
        ctx = prompt_mask(model, tokens, mask, 3)
        refused = []
        for fn in (lambda: model.run_passes(tokens, mask, 2, train=True, context=ctx),
                   lambda: dataclasses.replace(model, prefix_mixin=True).run_passes(tokens, mask, 2, train=True, context=ctx),
                   lambda: model.run_passes(tokens, mask, 2, train=False, context=ctx, context_passes=0)):
            try:
                fn()
            except ValueError:
                refused.append(True)
        check(len(refused) == 3, f"tiny 16f. context refused in training (with and without prefix mixin) and "
                                 f"context_passes 0 refused ({len(refused)} / 3)")


def prefill_d768_checks() -> None:
    """d768 real kernels (bf16, FA4, sonic MoE): SOFT / FUSED c = 2 on variant stdres pool and fbt run, keep the prompt."""
    import jmp
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.recipe import model_config

    cfg = model_config(shape="d768", layers=2)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    seq, split = cfg.max_seq_len, 2500
    key_tok, key_model = jax.random.split(jax.random.key(770), 2)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, 128, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32)
        for share in ("pool", "fbt"):
            model = mp.cast_to_compute(prefill_variant(base, key_model, "stdres", share, 3))
            ctx = prompt_mask(model, tokens, mask, 1024)
            run = eqx.filter_jit(lambda m, t, c, cp: m.run_passes(t, mask, 3, train=False, context=c, context_passes=cp)[0])
            full = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 3, train=False)[0])(model, tokens)
            soft, fused2, fused3 = run(model, tokens, ctx, 1), run(model, tokens, ctx, 2), run(model, tokens, ctx, 3)
            c3 = ctx[..., None]
            soft_prompt = max_diff(jnp.where(c3, soft[-1], 0.0), jnp.where(c3, full[0], 0.0))
            f2_prompt = max_diff(jnp.where(c3, fused2[-1], 0.0), jnp.where(c3, full[1], 0.0))
            losses = eqx.filter_jit(lambda m, t, c: m.pass_losses(t, loss_weight, mask=mask, n_passes=3, context=c,
                                                                  context_passes=1, keep=(3,)))(model, tokens, ctx)
            finite = bool(jnp.all(jnp.isfinite(losses[0])))
            check(finite and soft_prompt <= 1e-2 and f2_prompt <= 1e-2 and max_diff(fused3[-1], full[-1]) <= 1e-3,
                  f"d768 stdres {share}: SOFT keeps the 1,024-token prompt at pass 1 (max |d|={soft_prompt:.3g}), c = 2 at "
                  f"pass 2 ({f2_prompt:.3g}), c = 3 == every position reading ({max_diff(fused3[-1], full[-1]):.3g}); "
                  f"SOFT loss finite={finite}, mean {float(jnp.mean(losses[0])):.4f}")


def joint_share(model, tokens, mask):
    """Mean weight of the previous slots over mixing points 0..2L-1 at feedback positions (pass 2 on the plain pass 1)."""
    cfg = model.config
    n_mix, eps_d = 2 * cfg.num_layers, cfg.hidden_dim * am.ATTN_RES_EPS
    ctx = model._context(tokens, mask)
    x0 = model._embed(tokens)
    _, slots, _ = model._trunk(x0, ctx, None)
    q = model.attnres_query.weight
    scores = [am.slot_scores(s, q, eps_d) for s in [x0, *slots]]
    _, valid, ps = model.feedback_inputs(slots, ctx, train=False)
    shares = []
    for j in range(n_mix):
        same = jnp.stack([sc[..., j] for sc in scores[: j + 1]], -1)
        prev = ps[:, :, j:, j] + model.frontier_entry_bias.weight[j]
        w = jax.nn.softmax(jnp.concatenate([same, prev], -1), -1)
        shares.append(jnp.sum(w[..., j + 1:], -1))
    s = jnp.stack(shares, 0)
    return jnp.sum(s * valid[None]) / (n_mix * jnp.sum(valid))


def d768_checks() -> None:
    import jmp
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.recipe import model_config

    cfg = model_config(shape="d768", layers=2)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    seq, split = cfg.max_seq_len, 2500
    key_tok, key_model = jax.random.split(jax.random.key(768), 2)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(seq, split, 128, key_tok)
        loss_weight = jnp.ones(tokens.shape, jnp.float32)
        cases = (("pool", "shared", 1), ("joint", "shared", 2), ("pool", "shared", 2), ("inject", "shared", 2))
        if ONLY_INJECT:
            cases = (("inject", "shared", 2),)
        for share, query, passes in cases:
            model = variant(base, key_model, share=share, query=query, passes=passes,
                            jitter=0.0 if share == "inject" else 0.02, inject_rank=16 if share == "inject" else 0)

            def loss_fn(m):
                return mp.cast_to_compute(m).next_token_loss(tokens, loss_weight, mask=mask, reduction="mean",
                                                             logsumexp_weight=1e-4, return_router_metrics=True)

            (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
            check(bool(jnp.isfinite(loss)) and finite and metrics["qb_beta_per_layer"].shape == (cfg.num_layers, cfg.num_experts),
                  f"d768 {share}/{query} passes={passes}: loss {float(loss):.4f} (ln V = {math.log(cfg.vocab_size):.4f}), "
                  f"gradients finite={finite}, max |grad|={max_abs(grads):.3g}")
            if passes > 1:
                fwd = eqx.filter_jit(lambda t, m, model=model: mp.cast_to_compute(model).run_passes(t, m, passes, train=False)[0][-1])
                causality(fwd, tokens, mask, pos=3000, split=split, label=f"d768 {share} {passes} passes", tol=1e-2)


def window_checks() -> None:
    with jax.default_matmul_precision("highest"):
        _window_checks()


def _window_checks() -> None:
    """Group 24 (the fixed nearest window): --feedback-depth win<k> reads the k outputs from each sublayer's own depth
    upward and averages them with equal weight, with no depth softmax. A wrong window cannot raise, because win<k>
    and whole build the same parameter tree, so these checks compare against the written-out reference, show that the
    window changes the result, and show that a window as wide as the depth is the whole depth again."""
    cfg = tiny_config()
    n_mix = 2 * cfg.num_layers
    key_tok, key_model = jax.random.split(jax.random.key(2026112224), 2)
    with set_mesh(explicit_mesh()):
        base = gm.Transformer.init(cfg, key=key_model)
        tokens, mask = packed_batch(cfg.max_seq_len, 7, cfg.vocab_size, key_tok)
        tok8, mask8 = packed_batch(8, 4, cfg.vocab_size, jax.random.fold_in(key_tok, 8))
        plain_marin = eqx.filter_jit(lambda m, t: m(t, mask)[0])(base, tokens)

        def win_model(win, passes=2, query="uniform"):
            return stdres_variant(base, key_model, share="pool", query=query, proj="slot", proj_rank=4,
                                  passes=passes, depth=f"win{win}")

        # 24a. every mixing point's read is the equal-weight mean over its own window.
        for win in (2, 4):
            model = win_model(win)
            outs = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(model, tokens)
            refs = eqx.filter_jit(lambda m, t: ref_stdres_passes(m, t, mask, 2))(model, tokens)
            d = max(max_diff(a, b) for a, b in zip(outs, refs))
            check(d <= 2e-6, f"tiny 24a. win{win}: both passes == the written-out equal-weight window (max |d|={d:.3g})")

        # 24b. a window at least as wide as the depth is the whole depth again.
        wide = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0][-1])(win_model(n_mix), tokens)
        whole_model = stdres_variant(base, key_model, share="pool", query="uniform", proj="slot", proj_rank=4,
                                     passes=2, depth="whole")
        whole = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0][-1])(whole_model, tokens)
        check(max_diff(wide, whole) == 0.0,
              f"tiny 24b. win{n_mix} == whole under the uniform query (max |d|={max_diff(wide, whole):.3g})")

        # 24c. the window matters. Nothing raises when it is wrong, so this is the check that would catch a model
        # trained or scored under the other depth.
        narrow_model = win_model(4)
        narrow = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0][-1])(narrow_model, tokens)
        moved = max_diff(narrow, whole)
        check(moved > 1e-3, f"tiny 24c. win4 differs from whole on the same weights (max |d|={moved:.3g})")

        # 24d. a zero share leaves the backbone as it is, with the window in place: residual_mix hands the stream
        # back bit for bit, and a whole pass is Marin's own transformer to the tolerance group 14 uses.
        dead = eqx.tree_at(lambda m: m.frontier_logit.weight, narrow_model,
                           jnp.full_like(narrow_model.frontier_logit.weight, -1e4))
        z = eqx.filter_jit(lambda m, t: m.run_passes(t, mask, 2, train=False)[0])(dead, tokens)
        kx, kp = jax.random.split(jax.random.fold_in(key_model, 24))
        xr = jax.random.normal(kx, (2, cfg.max_seq_len, cfg.hidden_dim))
        pv = jax.random.normal(kp, (2, cfg.max_seq_len, n_mix, cfg.hidden_dim))
        valid = jnp.ones((2, cfg.max_seq_len), bool).at[:, 0].set(False)
        bitwise = all(bool(jnp.array_equal(am.residual_mix(j, xr, (pv, valid, None), dead.frontier_logit.weight, 4), xr))
                      for j in range(n_mix))
        live_win = am.residual_mix(0, xr, (pv, valid, None), narrow_model.frontier_logit.weight, 4)
        check(bitwise and max_diff(live_win, xr) > 1e-3 and max_diff(z[0], z[1]) <= 1e-6
              and max_diff(z[0], plain_marin) <= 1e-5,
              f"tiny 24d. win4 at a zero share: residual_mix returns the stream bit for bit at every sublayer "
              f"({bitwise}); pass 2 == pass 1 (max |d|={max_diff(z[0], z[1]):.3g}) == Marin's Transformer "
              f"({max_diff(z[0], plain_marin):.3g})")

        # 24e. K = T passes reach the token-by-token recursion, as for the other depths.
        m8 = win_model(4, passes=8)
        jac = eqx.filter_jit(lambda m, t: m.run_passes(t, mask8, 8, train=False)[0][-1])(m8, tok8)
        exact = eqx.filter_jit(lambda m, t: ref_stdres_exact(m, t, mask8))(m8, tok8)
        check(max_diff(jac, exact) <= 5e-6,
              f"tiny 24e. win4: 8 Jacobi passes == the exact recursion (max |d|={max_diff(jac, exact):.3g})")

        # 24f. a window with a learned query is refused: the window has no depth softmax to learn.
        refused = ""
        try:
            win_model(4, query="own")
        except (ValueError, SystemExit) as exc:
            refused = str(exc)
        check("win" in refused, f"tiny 24f. win4 with the own query is refused ({refused[:60] or 'not refused'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-d768", action="store_true")
    ap.add_argument("--only-d768", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="first tiny check group to run (1-6; 7-8 always run with 6)")
    ap.add_argument("--only-inject", action="store_true", help="group 9 (share inject) and the d768 inject case only")
    ap.add_argument("--only-proj", action="store_true", help="group 10 (proj source, the final method) only")
    ap.add_argument("--only-depth", action="store_true", help="group 13 (depth top, the feedback-depth ablation) only")
    ap.add_argument("--only-baselines", action="store_true",
                    help="groups 11 / 12 (share fbt, variant src: the d768 comparison variants) only")
    ap.add_argument("--only-stdres", action="store_true",
                    help="group 14 (variant stdres: the standard residual trunk + the sublayer feedback path or FBT's "
                         "fusion) and its d768 case")
    ap.add_argument("--only-window", action="store_true",
                    help="group 24 (the fixed nearest window, --feedback-depth win<k>)")
    ap.add_argument("--only-pm", action="store_true",
                    help="group 15 (variant stdres with FBT's prefix mixin and jitter) and its d768 cases")
    ap.add_argument("--only-prefill", action="store_true",
                    help="group 16 (decoding modes: SOFT / FUSED with a c-pass prompt) and its d768 cases")
    ap.add_argument("--only-fbtnorm", action="store_true",
                    help="group 17 (fbt_input_norm of FBT's fused input) and its d768 cases")
    ap.add_argument("--only-stdres-top", action="store_true",
                    help="group 18 (variant stdres with depth top: the last-block ablation on the standard residual) only")
    ap.add_argument("--only-inject-ablation", action="store_true",
                    help="group 19 (variant stdres: FBT's injection with ALF's read, ALF's injection with FBT's read) "
                         "only")
    ap.add_argument("--only-embed-inject", action="store_true",
                    help="group 20 (variant stdres: the sublayer feedback path plus the previous top state added "
                         "to the token input) and its d768 cases")
    ap.add_argument("--only-embed-plain", action="store_true",
                    help="group 22 (variant stdres: embed_inject plain:<c>, Recycler's adapter as it is)")
    ap.add_argument("--only-embed-only", action="store_true",
                    help="group 21 (variant stdres: embed_inject only, the input injection without the feedback path)")
    ap.add_argument("--only-embed-plain-attnres", action="store_true",
                    help="group 23 (variant feedback, the AttnRes trunk: embed_inject plain:<c> / plainonly:<c>) and its "
                         "d768 cases")
    ap.add_argument("--only-turns", action="store_true",
                    help="check 16e alone (multi-turn decoding: alternating plain user turns and generated turns)")
    args = ap.parse_args()
    global START, ONLY_INJECT
    START = args.start
    ONLY_INJECT = args.only_inject
    from alf.marin.compat import register_gpu_kinds_for_pallas

    register_gpu_kinds_for_pallas()
    print(f"[test_model] devices={jax.devices()}", flush=True)
    if args.only_embed_plain_attnres:
        if not args.only_d768:
            embed_plain_attnres_checks()
        if not args.skip_d768:
            embed_plain_attnres_d768_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_embed_plain:
        embed_plain_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_embed_only:
        embed_only_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_embed_inject:
        embed_inject_checks()
        if not args.skip_d768:
            embed_inject_d768_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_inject_ablation:
        inject_ablation_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_stdres_top:
        stdres_top_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_turns:
        turns_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_fbtnorm:
        fbtnorm_checks()
        if not args.skip_d768:
            fbtnorm_d768_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_prefill:
        prefill_checks()
        if not args.skip_d768:
            prefill_d768_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_pm:
        pm_checks()
        if not args.skip_d768:
            pm_d768_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_window:
        window_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_stdres:
        stdres_checks()
        if not args.skip_d768:
            stdres_d768_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_baselines:
        baseline_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_depth:
        depth_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if args.only_proj:
        proj_checks()
        print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
        sys.exit(1 if FAILED else 0)
    if not args.only_d768:
        if not args.only_inject:
            tiny_checks()
        inject_checks()
        proj_checks()
        depth_checks()
        baseline_checks()
        stdres_checks()
        window_checks()
        pm_checks()
        fbtnorm_checks()
    if not args.skip_d768:
        d768_checks()
        stdres_d768_checks()
        pm_d768_checks()
        fbtnorm_d768_checks()
    print(f"[test_model] {'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
