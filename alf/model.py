"""Attention Latent Feedback (ALF, arXiv:2611.31154): the sublayer feedback and input injection paths, on two backbones.

ALF is built on Marin's MoE hero Transformer (`experiments.grug.moe_hero_ep.model`, marin-community/marin at
8577d258214dd0dcb992c04458ee4eb10496d4e5), whose recipe, sharding and trainer are reused unchanged. Three models
subclass it:
  AttnResTransformer  the attention-residual backbone (AttnRes, arXiv:2603.15031) with the feedback path;
  StdResTransformer   the standard pre-norm residual backbone with the feedback path;
  SrcTransformer      the single-source feedback route of the Full-bandwidth Transformer (FBT, arXiv:2608.08888),
                      kept as a baseline on the standard backbone.
A model runs K passes over the sequence; pass 1 is the ordinary one-pass model, and passes 2..K are the ones that
read the previous position's latents. The FBT baselines follow the published method item by item.

The attention-residual backbone (one pass). Slot 0 is the embedded token after Marin's embedding norms, slot k + 1 the
output of sublayer k (attention of layer i is sublayer 2i, its MoE + shared experts 2i + 1, each after its branch short
conv). The input of sublayer j and the final output (j = 2L) are mixtures sum_s softmax_s(w_j . v_s / sqrt(|v_s|^2 +
D eps)) v_s over the slots produced so far, w_j a zero-initialized pseudo-query; the mixture then goes through the
sublayer's own pre-norms (RMSNorm + GatedNorm), or the final norms. A slot is scored by every query once, when it is
produced.

The sublayer feedback path (passes 2..K). Mixing point j < 2L also reads the previous token's slots j + 1 .. 2L from the
previous pass (shifted right by one position; positions without feedback hold zeros and read nothing):
  share "joint": the previous slots join the same softmax, each logit offset by an entry bias b_j (leaf
    frontier_entry_bias, not frontier_logit: a joint run branched from a pool / AttnRes base then takes the biases
    calibrated at the branch point from its initialization, spec entry_bias, instead of the base's dormant pool logits);
  share "pool": h_j = (1 - rho_j) m_j + rho_j n_j, m_j the same-token mixture, n_j a softmax (query "shared": the
    mixing point's pseudo-query, "own": a zero-initialized second query, "uniform": the mean) over the previous slots,
    rho_j = sigmoid(a_j) starting at init_share.
  share "inject": the previous token is read only through one new AttnRes source per layer: at layer i's attention
    input, the pool read n_2i is RMS-normalized and mapped by layer i's own up_i(down_i(.)) (rank inject_rank, up
    zero-initialized); that mixing point and every later one, the final output included, weigh it like any slot.
    Without a previous state (pass 1) every layer still gets a zero source, as a masked position does (as in FBT).
  share "fbt": no previous slot is read; the previous position's top state h (the final-normalized output of the
    previous pass, the LM head's input) enters only through FBT's layer-0 fusion gate product: slot 0 becomes
    norm(W_s h * sigmoid(W_g norm(e))) at positions with feedback, e the raw token embedding, W_s / W_g D x D
    uniform(+-sqrt(3 / D)), norm parameter-free; the fused input replaces the token input (it is not the identity at the
    switch, as in FBT) and is source 0 of every depth mixture. FBT's reference implementation feeds norm(token
    embedding), RMS 1 like the fused input; Marin's embedding norms leave the token input far below RMS 1 (the plain
    standard-residual model at d768 / MoE-1.6B: median 0.09 per position). fbt_input_norm "unit" (default) keeps the
    fused input at RMS 1; "embed" puts it through Marin's embedding norms (embed_norm, embed_gated_norm), as the token
    it replaces; "match" rescales it to the RMS of the token input it replaces, per position (FBT's equality of the
    two).
rank > 0 maps every previous slot through one shared low-rank map (up projection zero-initialized). proj "source"
(as in FBT) maps the previous token's two slots of layer i (its attention and MoE outputs) by
M_i v = v + Delta_i v, once, before any mixing point reads them: Delta_i = up_i(down_i(.)) of rank proj_rank (up
zero-initialized), or a full D x D matrix (zero-initialized) for proj_rank 0. proj "slot" gives every previous slot its
own map instead (2L maps: M_k v = v + Delta_k v for the output of sublayer k). inject_rank > 0 builds the per-layer maps
whatever the share, so a pool / AttnRes run carries them dormant and can be branched into an inject run (and back).
During training each previous slot gets Uniform[-jitter, jitter] noise times its own RMS (per position). The previous
slots are not detached. The next pass is offered the sublayer outputs only (not the injected sources).
Positions without feedback: every document start (BOS, a segment-id change, position 0).

StdResTransformer (--variant stdres) keeps Marin's residual trunk instead of AttnRes. Share "pool" is the feedback path
on a standard transformer: sublayer j reads (1 - rho_j) x + rho_j c n_j, x the residual stream, n_j the read above (own
query, per-output maps), c = rms(x) / rms(n_j), as in FBT's Qwen3.5 adapter; the stream itself is unchanged. Share "fbt"
fuses the previous top state into the start of the residual stream (FBT's own design). With prefix_mixin (training
only, FBT's prefix mixin) every feedback pass gives each document an independently sampled plain prefix of
1 + floor(u * length) tokens that reads nothing.

The fixed nearest window (--feedback-depth win<k>, Section 6.2's "Fixed nearest-4") is the depth rule that keeps no
depth softmax at all, on the standard residual backbone: sublayer j reads the previous position's outputs
j .. min(j + k - 1, 2L - 1), equally weighted, with no pseudo-query (query "uniform"), and residual_mix's c rescales the
mean to the stream as it does for the full read, so the window changes what is read and not its size. The rule depends
on j alone -- not on the trained maps and not on 2L -- so one k transfers across widths and depths; a window of 4
covers 102-103% of the mass that the trained softmax puts on its own three best outputs (MoE-1.6B 0.762, MoE-2.6B
0.731, MoE-4.0B 0.699, against an oracle top-3 of 0.749, 0.719, 0.680). parse_depth reads the window out of the depth
value; FEEDBACK_DEPTHS "whole" and "top" are the softmax rules it keeps.

The input injection path (embed_inject, with share "pool" on either backbone): besides the feedback path of every
sublayer, passes 2..K also add the previous position's top state h (the final-normalized output of the previous pass, as
share "fbt" reads) to the token input, as Recycler adds it to the next embedding:
u = norm(h) W_E / sqrt(D), norm parameter-free (RMS 1 whatever the gains of the final norms), W_E D x D zero-initialized
(leaf embed_inject_proj.weight, Adam), so the switch to two passes starts from the sublayer feedback path alone. The
1 / sqrt(D) is the fan-in scale: entries of W_E of order one give u RMS one, and one Adam step (every entry moves by the
learning rate) moves u by about lr instead of lr sqrt(D) (without it, the first update after branching the d768 /
MoE-1.6B model put the pass-2 loss 0.60 above the sublayer feedback path alone at the same step, and 0.15-0.20 above it
over the next four). Scale (Marin's token input is far below RMS 1, see share "fbt"): "match" adds rms(x0_t) u_t to the
embedded token x0_t (the injection is measured in units of the token input it joins, per position; the token input
itself is untouched); "embed" adds rms(e_t) u_t to the raw embedding e_t before Marin's embedding norms, which then fix
the scale of the sum. Positions without feedback keep x0. "plain:<c>" is the final form of the path and the injection of
LoRM's FeedbackTransformer.mix (github.com/ifm-ai/LoRM, lorm/model/feedback.py, u_t = e_t + W rms(h_{t-1}); Recycler's
Emb(x) + Linear(RMSNorm(h)) is the same form): x0_t + c * norm(h_{t-1}) W_E, h the stream after the last block, before
the final norms (LoRM's `top`; the other modes read it after them), norm parameter-free, no per-position scale and no
fan-in scale, only the constant c, W_E zero-initialized as in Recycler (the passes are switched on at a checkpoint).
With Adam, no weight decay and a zero start, training W_E under the factor c is the same as training W' = c W_E without
it at c times the learning rate (up to Adam's epsilon), so c is the learning-rate multiplier of Recycler's projection
and can be folded into the weights of a trained model; "plainonly:<c>" is "plain" without the sublayer feedback path.
"only" is the ablation without the sublayer feedback path: the injection of "match" alone, no sublayer reads the
previous position (the path's parameters are built and stay dormant), which is Recycler's design on this backbone. The
modes "match", "embed" and "only" are retained for checkpoints trained before the final form, on the standard residual
backbone alone. The mode is recorded in <run-dir>/embed_inject.txt (record_embed_inject), which evaluations rebuild the
model from.

The input injection path on the attention-residual backbone (AttnResTransformer, share "pool") takes the plain modes
only: in passes 2..K slot 0 becomes x0_t + c * norm(h_{t-1}) W_E at positions with feedback (x0 elsewhere), and it is
weighed in every depth mixture as the embedded token is. Here h is the previous pass's final mixture -- mixing point 2L,
the input of final_norm, which is this backbone's analogue of the stream after the last block and before the final norms
-- and _trunk hands it out as it computes it (with_top=True, through _trunk_top and _top_state). "plain" keeps the pool
read of the previous slots (the method on this backbone: query "own", proj "slot"); "plainonly" drops it, leaving the
attention-residual backbone and the injection alone. Shares "none", "inject" and "fbt" refuse the knob, and so does
"joint", whose entry biases are calibrated at the branch point on a model without the injection. A one-pass base is
built without W_E: the branch builds the leaf at zero and takes it from its initialization through the partial restore
(--init-from with --allow-partial-checkpoint), as the standard-residual runs do from a plain Marin checkpoint.

The injection source depth ("@<k>" after a plain mode's constant; static field embed_inject_src, read by
embed_inject_source(); the paper's method section and its appendix on what the feedback reads) moves the state the
injection reads: the previous position's residual stream after block k -- the token input plus the outputs of the first
k blocks, that is 2k sublayer outputs -- instead of after the last block. It is a standard-residual-backbone knob only,
because the attention-residual backbone has no residual stream to cut at block k; 0 (no suffix) is the last block.

Decoding modes (evaluation, run_passes context / context_passes): a prompt is processed with c passes and keeps its
pass-c state, the rest reads it and runs every pass (the Jacobi form of decoding token by token): c = 1 is FBT's SOFT
(the prompt is a plain prefill), c >= the pass count is FUSED with the prompt in every pass; pass 1 alone is standard.

Loss: pass-1 cross-entropy + mean of the later passes' (each with the recipe's z-loss). Evaluation (reduction "none")
and __call__ run eval_passes passes and use the last one. QB routing: a layer's bias update is the mean of its passes'
betas. With passes = 1 the feedback parameters are dormant (zero gradient), so an AttnRes run carrying them can later be
branched into a feedback run from any of its checkpoints.

New parameters are leaves named `<field>.weight`: Marin's optimizer mask sends `.weight` leaves to Adam. Otherwise the
2-D pseudo-queries would land in MuonH, whose norm-preserving update keeps a zero-initialized matrix at zero.

Frozen parameter names. The fields listed in NEW_PARAM_FIELDS are exactly what a checkpoint stores (as the leaves
`<field>.weight`), so they are an on-disk contract and are deliberately *not* renamed to the paper's vocabulary: a
partial restore (`--init-from` with `--allow-partial-checkpoint`, the documented way to branch one run from another)
takes every leaf the checkpoint does not have from the initialization, and these parameters are zero-initialized by
design, so a renamed leaf would load as zeros -- the feedback would silently do nothing and the run would report a
plausible one-pass loss instead of failing. The two file names "fbt_input_norm.txt" and "embed_inject.txt" are frozen
for the same reason. Their meaning in the paper:

  attnres_query          w_j, the pseudo-queries of the attention-residual backbone, [2L + 1, D]
  frontier_logit         a_j, the share logits of shares "pool" / "inject" (rho_j = sigmoid(a_j)), [2L]
  frontier_entry_bias    b_j, the entry biases of share "joint", [2L]
  frontier_query         the feedback-only second query set (query "own"), [2L, D]
  frontier_down          the one shared low-rank map on every previous slot: its down projection, [D, r], rank r > 0
  frontier_up            the same map's up projection, [r, D], zero-initialized
  frontier_inject_down   down_i, layer i's map on the pooled read of share "inject"
  frontier_inject_up     up_i of that map, zero-initialized
  frontier_proj_down     down_k of Delta_k in the per-source / per-slot maps M_k = I + Delta_k
  frontier_proj_up       up_k of Delta_k, zero-initialized
  frontier_proj_full     Delta_k as a full D x D matrix, zero-initialized (proj_rank 0)
  fbt_state_proj         W_s, the state projection of FBT's fusion
  fbt_token_gate         W_g, the token gate of FBT's fusion
  src_down               down_l of W_H, the per-target-block map of FBT's single-source route
  src_up                 up_l of W_H, zero-initialized
  embed_inject_proj      W_E, the projection of the input injection path

The equinox *static* fields (feedback_share, feedback_query_kind, feedback_proj, feedback_depth, fbt_input_norm,
embed_inject, embed_inject_scale, embed_inject_src, ...) are part of the template rather than the checkpoint, and carry
no such contract.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

import experiments.grug.moe_hero_ep.model as gm

ATTN_RES_EPS = 1e-6
FEEDBACK_SHARES = ("none", "joint", "pool", "inject", "fbt")
FEEDBACK_QUERIES = ("shared", "own", "uniform")
FEEDBACK_PROJS = ("none", "source", "slot")
# depth "whole": mixing point j reads the previous token's slots j+1..2L; "top": only those of its last layer
# (slots 2L-1, 2L), the ablation that reads the previous position's last block instead of its whole depth.
FEEDBACK_DEPTHS = ("whole", "top")
# depth "win<k>" (Section 6.2, "Fixed nearest-4"): the fixed nearest-window rule -- mixing point j reads only the
# previous position's outputs j..min(j+k-1, 2L-1), equally weighted, with no depth softmax and no pseudo-query
# (query "uniform"). The rule depends on j alone, not on the trained maps and not on 2L, so one k transfers across
# widths and depths; a window of 4 covers 102-103% of the mass the trained softmax puts on its own three best outputs
# (MoE-1.6B 0.762 / MoE-2.6B 0.731 / MoE-4.0B 0.699 against an oracle top-3 of 0.749 / 0.719 / 0.680). It is a rule of
# the standard residual backbone (StdResTransformer, share "pool"); parse_depth reads the window out of the value.


def parse_depth(value: str) -> tuple[str, int]:
    """"whole" | "top" | "win<k>" -> (mode, window). The window is 0 for "whole" / "top" and k >= 1 for "win<k>", the
    fixed nearest-window rule of Section 6.2 (residual_mix's `win`)."""
    v = str(value)
    if v in FEEDBACK_DEPTHS:
        return v, 0
    if v.startswith("win") and v[3:].isdigit() and int(v[3:]) >= 1:
        return "win", int(v[3:])
    raise ValueError(f"depth must be one of {FEEDBACK_DEPTHS} or win<k> with k >= 1, got {value!r}")
FBT_INPUT_NORMS = ("unit", "embed", "match")
EMBED_INJECTS = ("none", "match", "embed", "only", "plain", "plainonly")
# modes whose injection reads the top state before the final norms (LoRM's `top`)
PLAIN_INJECTS = ("plain", "plainonly")


def parse_embed_inject(value: str) -> tuple[str, float]:
    """"none" | "match" | "embed" | "only" | "plain:<c>" | "plainonly:<c>" -> (mode, c). c (the plain modes only,
    > 0) is the constant in front of the projection; the other modes return 1.0. A plain mode may carry an "@<k>"
    suffix, which moves the state the injection reads to the residual stream after block k; this function only
    validates it, and embed_inject_source() returns the k."""
    value, _, src = str(value).partition("@")
    if src and not (value.partition(":")[0] in ("plain", "plainonly") and src.isdigit() and int(src) > 0):
        raise ValueError(f"embed_inject @<block> goes with plain:<c> / plainonly:<c> and a positive block count, got {src!r}")
    mode, _, scale = value.partition(":")
    if mode not in EMBED_INJECTS or (mode in ("plain", "plainonly")) != bool(scale):
        raise ValueError(f"embed_inject must be none, match, embed, only, plain:<c> or plainonly:<c>, got {value!r}")
    c = float(scale) if scale else 1.0
    if not 0.0 < c < float("inf"):
        raise ValueError(f"embed_inject constant must be positive and finite, got {value!r}")
    return mode, c
EMBED_INJECT_FILE = "embed_inject.txt"  # in the run directory: the embed_inject its checkpoints were trained with
EMBED_INJECT_KEY_OFFSET = 500  # jitter on the injected top state of pass k draws from _jitter_key(tokens, 500 + k)
FBT_INPUT_NORM_FILE = "fbt_input_norm.txt"  # in the run directory: the fbt_input_norm its checkpoints were trained with
# The parameters ALF adds, as a checkpoint stores them (leaves `<field>.weight`). These names are an on-disk contract
# and are deliberately not renamed; the table at the end of the module docstring gives the paper's symbol for each.
NEW_PARAM_FIELDS = ("attnres_query", "frontier_logit", "frontier_entry_bias", "frontier_query", "frontier_down",
                    "frontier_up", "frontier_inject_down", "frontier_inject_up", "frontier_proj_down", "frontier_proj_up",
                    "frontier_proj_full", "fbt_state_proj", "fbt_token_gate", "src_down", "src_up", "embed_inject_proj")
_SPEC = {"passes": 1, "eval_passes": 1, "share": "none", "query": "shared", "rank": 0, "init_share": 0.01,
         "jitter": 0.0, "bos_id": -1, "jitter_seed": 20261106, "inject_rank": 0, "proj": "none", "proj_rank": 0,
         "entry_bias": None, "depth": "whole", "fbt_input_norm": "unit", "embed_inject": "none"}


class Weight(eqx.Module):
    weight: jax.Array


def _replicated(x: jax.Array) -> jax.Array:
    return reshard(x, P(*([None] * x.ndim)))


def _layer_slice(tree, layer_id: int):
    """Layer `layer_id` of the layer-stacked blocks, each leaf resharded to its spec minus the layer axis (explicit
    sharding: without it the slice's cotangent comes back unsharded and the transpose rejects it)."""

    def take(w):
        spec = tuple(jax.typeof(w).sharding.spec)
        return reshard(w[layer_id], P(*spec[1:]))

    return jax.tree.map(take, tree)


def _segments(mask):
    seg = mask.segment_ids if isinstance(mask, gm.AttentionMask) else None
    return seg[0] if seg is not None else None


def slot_scores(v: jax.Array, q: jax.Array, eps_d: float) -> jax.Array:
    """[..., D] x [J, D] -> [..., J] float32: q_j . RMSNorm(v) / sqrt(D) = q_j . v / sqrt(|v|^2 + D eps)."""
    lead = tuple(jax.typeof(v).sharding.spec)[: v.ndim - 1]
    lead = lead + (None,) * (v.ndim - 1 - len(lead))
    dots = jnp.einsum("...d,jd->...j", v, q.astype(v.dtype), preferred_element_type=jnp.float32,
                      out_sharding=P(*lead, None))
    sq = jnp.sum(jnp.square(v.astype(jnp.float32)), axis=-1, keepdims=True)
    return dots * jax.lax.rsqrt(sq + eps_d)


def _weighted(w: jax.Array, values) -> jax.Array:
    out = w[..., 0:1] * values[0]
    for s in range(1, len(values)):
        out = out + w[..., s:s + 1] * values[s]
    return out


def mix(j: int, slots, scores, prev, flogit, mode: str, n_mix: int, depth: str = "whole") -> jax.Array:
    """Input of mixing point j: the same-token mixture over slots 0..j, plus the previous token's slots j+1..2L (depth
    "whole") or only those of its last layer (depth "top")."""
    dtype = slots[0].dtype
    s0 = j if depth == "whole" else max(j, n_mix - 2)
    same = jnp.stack([sc[..., j] for sc in scores], axis=-1)
    use_prev = prev is not None and mode != "none" and j < n_mix
    if use_prev and mode == "joint":
        pv, valid, ps = prev
        prev_logits = jnp.where(valid[..., None], ps[:, :, s0:, j] + flogit[j].astype(jnp.float32), -jnp.inf)
        w = jax.nn.softmax(jnp.concatenate([same, prev_logits], axis=-1), axis=-1).astype(dtype)
        spec = tuple(jax.typeof(slots[0]).sharding.spec)
        return _weighted(w[..., : j + 1], slots) + jnp.einsum("bts,btsd->btd", w[..., j + 1:], pv[:, :, s0:],
                                                              out_sharding=P(*spec))
    w = jax.nn.softmax(same, axis=-1).astype(dtype)
    out = _weighted(w, slots)
    if use_prev and mode == "pool":
        pv, valid, ps = prev
        window = pv[:, :, s0:]
        if ps is None:
            pooled = jnp.mean(window.astype(jnp.float32), axis=2).astype(dtype)
        else:
            pw = jax.nn.softmax(ps[:, :, s0:, j], axis=-1).astype(dtype)
            pooled = jnp.einsum("bts,btsd->btd", pw, window, out_sharding=P(*tuple(jax.typeof(out).sharding.spec)))
        rho = jax.nn.sigmoid(flogit[j].astype(jnp.float32)) * valid.astype(jnp.float32)
        out = out + rho[..., None].astype(dtype) * (pooled - out)
    return out


def _attn_branch(block, h, layer_mask, flag):
    attn_in = block.attn_gated_norm(block.rms_attn(h))
    out = block.attn(attn_in, layer_mask, disable_rope=flag, is_global=flag)
    if block.sconv_attn is not None:
        out = block.sconv_attn(out, _segments(layer_mask))
    return out


def _mlp_branch(block, h, layer_mask):
    mlp_in = block.mlp_gated_norm(block.rms_mlp(h))
    token_valid = gm.token_validity_from_attention_mask(layer_mask, batch_size=h.shape[0], sequence_length=h.shape[1])
    out, stats = block.mlp(mlp_in, token_valid)
    if block.shared is not None:
        for shared_expert in block.shared:
            out = out + shared_expert(mlp_in, activation=gm.ActivationFunctionEnum.silu)
    if block.sconv_mlp is not None:
        out = block.sconv_mlp(out, _segments(layer_mask))
    return out, stats


def _mixed_sublayer(sub, j, mode, n_mix, block, slots, scores, prev, flogit, layer_mask, flag, depth="whole"):
    """Mixture j and sublayer j in one remat region: its inputs (the slots) stay alive anyway, so the backward recomputes
    the mixture instead of keeping one per sublayer."""
    h = mix(j, slots, scores, prev, flogit, mode, n_mix, depth)
    if sub == 0:
        return _attn_branch(block, h, layer_mask, flag), None
    return _mlp_branch(block, h, layer_mask)


class AttnResTransformer(gm.Transformer):
    attnres_query: Weight
    frontier_logit: Weight | None  # [2L], shares "pool" / "inject"
    frontier_entry_bias: Weight | None  # [2L], share "joint"
    frontier_query: Weight | None
    frontier_down: Weight | None
    frontier_up: Weight | None
    frontier_inject_down: Weight | None  # [L, D, r]
    frontier_inject_up: Weight | None  # [L, r, D], zero-initialized
    frontier_proj_down: Weight | None  # [L, D, r] for proj "source", [2L, D, r] for proj "slot", proj_rank r > 0
    frontier_proj_up: Weight | None  # [L, r, D], zero-initialized
    frontier_proj_full: Weight | None  # [L or 2L, D, D], zero-initialized, proj "source" / "slot" with proj_rank 0
    fbt_state_proj: Weight | None = None  # [D, D], share "fbt"
    fbt_token_gate: Weight | None = None  # [D, D], share "fbt"
    passes: int = eqx.field(static=True, default=1)
    eval_passes: int = eqx.field(static=True, default=1)
    feedback_share: str = eqx.field(static=True, default="none")
    feedback_query_kind: str = eqx.field(static=True, default="shared")
    feedback_proj: str = eqx.field(static=True, default="none")
    feedback_depth: str = eqx.field(static=True, default="whole")
    jitter: float = eqx.field(static=True, default=0.0)
    bos_id: int = eqx.field(static=True, default=-1)
    jitter_seed: int = eqx.field(static=True, default=20261106)
    fbt_input_norm: str = eqx.field(static=True, default="unit")
    embed_inject_proj: Weight | None = None  # [D, D], zero-initialized, every embed_inject but "none"
    embed_inject: str = eqx.field(static=True, default="none")
    embed_inject_scale: float = eqx.field(static=True, default=1.0)  # the constant c of "plain" / "plainonly"
    embed_inject_src: int = eqx.field(static=True, default=0)  # "@<k>": read after block k (StdResTransformer); 0 = last

    @staticmethod
    def init(cfg_or_vocab, config=None, *, key) -> "AttnResTransformer":
        base = gm.Transformer.init(cfg_or_vocab, config, key=key)
        return AttnResTransformer.from_base(base, key=key)

    @staticmethod
    def from_base(base: gm.Transformer, *, key, **overrides) -> "AttnResTransformer":
        """The base model's weights (Marin's own initialization for the same key) plus the new parameters."""
        spec = dict(_SPEC, **overrides)
        _validate(spec)
        cfg = base.config
        d, n_mix = cfg.hidden_dim, 2 * cfg.num_layers
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(gm.Transformer)}
        frontier_logit = entry_bias = frontier_query = frontier_down = frontier_up = None
        fbt_state = fbt_gate = None
        if spec["share"] == "fbt":
            bound = math.sqrt(3.0 / d)
            fbt_state = Weight(_replicated(random.uniform(random.fold_in(key, 20261109), (d, d), jnp.float32, -bound, bound)))
            fbt_gate = Weight(_replicated(random.uniform(random.fold_in(key, 20261110), (d, d), jnp.float32, -bound, bound)))
        elif spec["share"] != "none":
            p = float(spec["init_share"])
            init = jnp.full((n_mix,), math.log(p / (1.0 - p)), jnp.float32)
            if spec["share"] == "joint":
                if spec["entry_bias"] is not None:
                    init = jnp.asarray(spec["entry_bias"], jnp.float32)
                    if init.shape != (n_mix,):
                        raise ValueError(f"entry_bias needs {n_mix} values (one per mixing point), got {init.shape}")
                entry_bias = Weight(_replicated(init))
            else:
                frontier_logit = Weight(_replicated(init))
            if spec["query"] == "own":
                frontier_query = Weight(_replicated(jnp.zeros((n_mix, d), jnp.float32)))
            if spec["rank"] > 0:
                k = random.fold_in(key, 20261106)
                frontier_down = Weight(_replicated(gm._init_weight(k, (d, spec["rank"]), cfg.initializer_std)))
                frontier_up = Weight(_replicated(jnp.zeros((spec["rank"], d), jnp.float32)))
        inject_down = inject_up = None
        if spec["inject_rank"] > 0:
            r, n_layer = int(spec["inject_rank"]), cfg.num_layers
            k = random.fold_in(key, 20261107)
            inject_down = Weight(_replicated(gm._init_weight(k, (n_layer, d, r), cfg.initializer_std)))
            inject_up = Weight(_replicated(jnp.zeros((n_layer, r, d), jnp.float32)))
        proj_down = proj_up = proj_full = None
        if spec["proj"] in ("source", "slot"):
            n_layer = cfg.num_layers * (2 if spec["proj"] == "slot" else 1)
            r = int(spec["proj_rank"])
            if r > 0:
                k = random.fold_in(key, 20261108)
                proj_down = Weight(_replicated(gm._init_weight(k, (n_layer, d, r), cfg.initializer_std)))
                proj_up = Weight(_replicated(jnp.zeros((n_layer, r, d), jnp.float32)))
            else:
                proj_full = Weight(_replicated(jnp.zeros((n_layer, d, d), jnp.float32)))
        embed_proj = None
        embed_mode, embed_scale = parse_embed_inject(spec["embed_inject"])
        if embed_mode != "none":
            embed_proj = Weight(_replicated(jnp.zeros((d, d), jnp.float32)))
        return AttnResTransformer(
            **fields,
            attnres_query=Weight(_replicated(jnp.zeros((n_mix + 1, d), jnp.float32))),
            frontier_logit=frontier_logit,
            frontier_entry_bias=entry_bias,
            frontier_query=frontier_query,
            frontier_down=frontier_down,
            frontier_up=frontier_up,
            frontier_inject_down=inject_down,
            frontier_inject_up=inject_up,
            frontier_proj_down=proj_down,
            frontier_proj_up=proj_up,
            frontier_proj_full=proj_full,
            fbt_state_proj=fbt_state,
            fbt_token_gate=fbt_gate,
            passes=int(spec["passes"]),
            eval_passes=int(spec["eval_passes"]),
            feedback_share=spec["share"],
            feedback_query_kind=spec["query"],
            feedback_proj=spec["proj"],
            feedback_depth=spec["depth"],
            jitter=float(spec["jitter"]),
            bos_id=int(spec["bos_id"]),
            jitter_seed=int(spec["jitter_seed"]),
            fbt_input_norm=spec["fbt_input_norm"],
            embed_inject_proj=embed_proj,
            embed_inject=embed_mode,
            embed_inject_scale=embed_scale,
        )

    # ----------------------------------------------------------------------------------------------------------------
    def _context(self, token_ids, mask):
        cfg = self.config
        if mask is None:
            mask = gm.AttentionMask.causal()
        segment_ids = None
        if isinstance(mask, gm.AttentionMask) and mask.segment_ids is not None:
            q_seg = gm._batch_reshard(mask.segment_ids[0])
            segment_ids = (q_seg, q_seg)
        short_mask = gm.AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = gm.AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)
        if cfg.remat_mode == "save_moe":
            policy = jax.checkpoint_policies.save_only_these_names(*gm.MOE_REMAT_SAVE_NAMES)
        elif cfg.remat_mode == gm.OFFLOAD_CARRY_REMAT_MODE:
            # The layer carry does not exist here (the slots are the analogue); keep the MoE output as Marin does.
            policy = jax.checkpoint_policies.save_only_these_names(gm.MOE_OUTPUT_REMAT_NAME)
        else:
            policy = None
        b, s = token_ids.shape
        long_lb, valid = gm.fa4_cute_segment_bounds(long_mask, batch_size=b, seq_len=s, sliding_window=None)
        short_lb, _ = gm.fa4_cute_segment_bounds(short_mask, batch_size=b, seq_len=s, sliding_window=cfg.sliding_window)
        bounds_spec = P(gm._BATCH_AXES, gm._seq_axis(get_abstract_mesh()))
        long_lb, short_lb, valid = (reshard(x, bounds_spec) for x in (long_lb, short_lb, valid))
        with jax.ensure_compile_time_eval():
            flags = [bool(x) for x in np.asarray(gm._long_layer_schedule(cfg.num_layers, cfg.global_every))]
        start = jnp.broadcast_to(jnp.arange(s)[None, :] == 0, (b, s))
        start = reshard(start, P(gm._BATCH_AXES, None))
        if self.bos_id >= 0:
            start = start | (token_ids == self.bos_id)
        if segment_ids is not None:
            q = segment_ids[0]
            start = start | jnp.pad(q[:, 1:] != q[:, :-1], ((0, 0), (1, 0)))
        return {"long_mask": long_mask, "long_lb": long_lb, "short_lb": short_lb, "valid": valid, "flags": flags,
                "policy": policy, "fb_valid": ~start}

    def _embed(self, token_ids):
        return self.embed_gated_norm(self.embed_norm(gm._embedding_gather(self.token_embed, token_ids)))

    def _trunk(self, x0, ctx, prev, *, with_top: bool = False):
        """One pass; returns (final-normalized state, sublayer outputs 1..2L, per-layer router stats); with_top: also the
        final mixture h (mixing point 2L, the input of final_norm), this backbone's state after the last block and
        before the final norms, which the input injection path of the next pass reads (_top_state)."""
        cfg = self.config
        n_mix = 2 * cfg.num_layers
        eps_d = cfg.hidden_dim * ATTN_RES_EPS
        q = self.attnres_query.weight.astype(x0.dtype)
        if self.feedback_share == "joint":
            flogit = self.frontier_entry_bias.weight
        else:
            flogit = None if self.frontier_logit is None else self.frontier_logit.weight
        inject = self.feedback_share == "inject"
        mode = self.feedback_share if prev is not None and not inject else "none"
        slots, scores, stats, outs = [x0], [slot_scores(x0, q, eps_d)], [], []
        for i in range(cfg.num_layers):
            block = _layer_slice(self.stacked_blocks.stacked, i)
            is_long = ctx["flags"][i]
            layer_mask = ctx["long_mask"].with_fa4_bounds(ctx["long_lb"] if is_long else ctx["short_lb"], ctx["valid"])
            flag = jnp.asarray(is_long)
            if inject:
                u = jnp.zeros_like(x0) if prev is None else self._inject_source(i, prev, x0)
                slots.append(u)
                scores.append(slot_scores(u, q, eps_d))
            for sub in (0, 1):
                j = 2 * i + sub
                fn = functools.partial(_mixed_sublayer, sub, j, mode, n_mix, depth=self.feedback_depth)
                out, layer_stats = eqx.filter_checkpoint(fn, policy=ctx["policy"])(
                    block, tuple(slots), tuple(scores), prev, flogit, layer_mask, flag)
                slots.append(out)
                scores.append(slot_scores(out, q, eps_d))
                outs.append(out)
                if sub == 1:
                    stats.append(layer_stats)
        h = mix(n_mix, slots, scores, None, None, "none", n_mix)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *stats)
        out = self.final_gated_norm(self.final_norm(h))
        return (out, outs, stacked, h) if with_top else (out, outs, stacked)

    def _inject_source(self, i: int, prev, x0):
        """Layer i's injected source: up_i(down_i(RMSNorm(n_2i))) at positions with feedback, zero elsewhere."""
        pv, valid, ps = prev
        j, dtype = 2 * i, x0.dtype
        spec = tuple(jax.typeof(x0).sharding.spec)
        window = pv[:, :, j:]
        if ps is None:
            pooled = jnp.mean(window.astype(jnp.float32), axis=2).astype(dtype)
        else:
            pw = jax.nn.softmax(ps[:, :, j:, j], axis=-1).astype(dtype)
            pooled = jnp.einsum("bts,btsd->btd", pw, window, out_sharding=P(*spec))
        p32 = pooled.astype(jnp.float32)
        normed = (p32 * jax.lax.rsqrt(jnp.mean(jnp.square(p32), axis=-1, keepdims=True) + ATTN_RES_EPS)).astype(dtype)
        z = jnp.einsum("btd,dr->btr", normed, self.frontier_inject_down.weight[i].astype(dtype),
                       out_sharding=P(*spec[:2], None))
        u = jnp.einsum("btr,rd->btd", z, self.frontier_inject_up.weight[i].astype(dtype), out_sharding=P(*spec))
        return u * valid[..., None].astype(dtype)

    def _jitter_key(self, token_ids, pass_idx: int):
        n = min(256, token_ids.shape[1])
        weights = jnp.arange(1, n + 1, dtype=jnp.uint32)
        digest = jnp.sum(token_ids[:, :n].astype(jnp.uint32) * weights[None, :], dtype=jnp.uint32)
        return random.fold_in(random.fold_in(random.key(self.jitter_seed), digest), pass_idx)

    def feedback_inputs(self, slots, ctx, *, train: bool, token_ids=None, pass_idx: int = 0):
        """(previous slots [B, T, 2L, D], positions with feedback [B, T], their scores [B, T, 2L, 2L] or None)."""
        cfg = self.config
        n_mix = 2 * cfg.num_layers
        dtype = slots[0].dtype
        spec = tuple(jax.typeof(slots[0]).sharding.spec)
        stacked_spec = P(*spec[:2], None, *spec[2:])
        pv = reshard(jnp.stack(slots, axis=2), stacked_spec)
        if train and self.jitter > 0.0:
            rms = jnp.sqrt(jnp.mean(jnp.square(jax.lax.stop_gradient(pv).astype(jnp.float32)), axis=-1, keepdims=True))
            noise = random.uniform(self._jitter_key(token_ids, pass_idx), pv.shape, jnp.float32, -self.jitter,
                                   self.jitter, out_sharding=stacked_spec)
            pv = pv + (noise * rms).astype(dtype)
        pv = jnp.pad(pv[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0)))
        valid = ctx["fb_valid"]
        pv = pv * valid[:, :, None, None].astype(dtype)
        if self.frontier_down is not None:
            pv = jnp.einsum("btsd,dr->btsr", pv, self.frontier_down.weight.astype(dtype),
                            out_sharding=P(*stacked_spec[:3], None))
            pv = jnp.einsum("btsr,rd->btsd", pv, self.frontier_up.weight.astype(dtype), out_sharding=stacked_spec)
        if self.feedback_proj in ("source", "slot"):
            pv = self._source_maps(pv, stacked_spec)
        if self.feedback_query_kind == "uniform":
            return pv, valid, None
        if self.feedback_query_kind == "shared":
            fq = self.attnres_query.weight[:n_mix]
        else:
            fq = self.frontier_query.weight
        return pv, valid, slot_scores(pv, fq, cfg.hidden_dim * ATTN_RES_EPS)

    def _source_maps(self, pv, stacked_spec):
        """proj "source": M_i v = v + Delta_i v on the previous token's slots 2i, 2i + 1 (layer i's attention and MoE
        outputs). proj "slot": M_k v = v + Delta_k v on each previous slot k separately."""
        dtype = pv.dtype
        low = P(*stacked_spec[:3], None)
        if self.feedback_proj == "slot":
            if self.frontier_proj_full is not None:
                delta = jnp.einsum("btsd,sde->btse", pv, self.frontier_proj_full.weight.astype(dtype), out_sharding=stacked_spec)
            else:
                z = jnp.einsum("btsd,sdr->btsr", pv, self.frontier_proj_down.weight.astype(dtype), out_sharding=low)
                delta = jnp.einsum("btsr,srd->btsd", z, self.frontier_proj_up.weight.astype(dtype), out_sharding=stacked_spec)
            return pv + delta
        mapped = []
        for i in range(self.config.num_layers):
            v = pv[:, :, 2 * i:2 * i + 2]
            if self.frontier_proj_full is not None:
                delta = jnp.einsum("btkd,de->btke", v, self.frontier_proj_full.weight[i].astype(dtype), out_sharding=stacked_spec)
            else:
                z = jnp.einsum("btkd,dr->btkr", v, self.frontier_proj_down.weight[i].astype(dtype), out_sharding=low)
                delta = jnp.einsum("btkr,rd->btkd", z, self.frontier_proj_up.weight[i].astype(dtype), out_sharding=stacked_spec)
            mapped.append(v + delta)
        return reshard(jnp.concatenate(mapped, axis=2), stacked_spec)

    def _fbt_input(self, token_ids, x0, top, ctx, *, train: bool, pass_idx: int):
        """Share "fbt": slot 0 = norm(W_s h_{t-1} * sigmoid(W_g norm(e_t))) where position t has feedback, else x0;
        fbt_input_norm "embed" passes the fused vector through the embedding norms that make x0, "match" scales it to
        rms(x0_t)."""
        spec = P(*tuple(jax.typeof(x0).sharding.spec))
        h = jnp.pad(top[:, :-1], ((0, 0), (1, 0), (0, 0)))
        if train and self.jitter > 0.0:
            rms = jnp.sqrt(jnp.mean(jnp.square(jax.lax.stop_gradient(h).astype(jnp.float32)), axis=-1, keepdims=True))
            noise = random.uniform(self._jitter_key(token_ids, pass_idx), h.shape, jnp.float32, -self.jitter,
                                   self.jitter, out_sharding=spec)
            h = h + (noise * rms).astype(h.dtype)
        return self._fbt_fuse(token_ids, x0, h, ctx)

    def _fbt_fuse(self, token_ids, x0, h, ctx):
        """FBT's gate product on the already shifted previous-position vector h [B, T, D]: slot 0 =
        norm(W_s h * sigmoid(W_g norm(e_t))) where position t has feedback, else x0 (fbt_input_norm as _fbt_input)."""
        dtype = x0.dtype
        spec = P(*tuple(jax.typeof(x0).sharding.spec))

        def norm(x):
            x32 = x.astype(jnp.float32)
            return x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + ATTN_RES_EPS)

        e = gm._embedding_gather(self.token_embed, token_ids)
        value = jnp.einsum("btd,de->bte", h.astype(dtype), self.fbt_state_proj.weight.astype(dtype), out_sharding=spec)
        gate = jnp.einsum("btd,de->bte", norm(e).astype(dtype), self.fbt_token_gate.weight.astype(dtype), out_sharding=spec)
        fused = norm(value.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(dtype)
        if self.fbt_input_norm == "embed":
            fused = self.embed_gated_norm(self.embed_norm(fused)).astype(dtype)
        elif self.fbt_input_norm == "match":
            x32 = x0.astype(jnp.float32)
            scale = jnp.sqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + ATTN_RES_EPS)
            fused = (fused.astype(jnp.float32) * scale).astype(dtype)
        return jnp.where(ctx["fb_valid"][..., None], fused, x0)

    def _embed_inject_input(self, token_ids, x0, top, ctx, *, train: bool, pass_idx: int):
        """The input injection path: the start of the residual stream (slot 0 on the attention-residual backbone) where
        position t has feedback, from the previous pass's top state h (_top_state). u_t = norm(h_{t-1}) W_E / sqrt(D)
        for "match" (x0_t + rms(x0_t) u_t) and "embed" (EmbedNorms(e_t + rms(e_t) u_t)), the scales rms(.) without
        gradient (units, not pathways); x0_t + c * norm(h_{t-1}) W_E for "plain" / "plainonly". Elsewhere x0."""
        dtype = x0.dtype
        spec = P(*tuple(jax.typeof(x0).sharding.spec))

        def rms(x32):
            return jnp.sqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + ATTN_RES_EPS)

        h = jnp.pad(top[:, :-1], ((0, 0), (1, 0), (0, 0)))
        if train and self.jitter > 0.0:
            scale = jnp.sqrt(jnp.mean(jnp.square(jax.lax.stop_gradient(h).astype(jnp.float32)), axis=-1, keepdims=True))
            noise = random.uniform(self._jitter_key(token_ids, EMBED_INJECT_KEY_OFFSET + pass_idx), h.shape, jnp.float32,
                                   -self.jitter, self.jitter, out_sharding=spec)
            h = h + (noise * scale).astype(h.dtype)
        h32 = h.astype(jnp.float32)
        normed = (h32 / rms(h32)).astype(dtype)
        valid = ctx["fb_valid"][..., None]
        u = jnp.einsum("btd,de->bte", normed, self.embed_inject_proj.weight.astype(dtype), out_sharding=spec)
        if self.embed_inject in ("plain", "plainonly"):
            delta = (u.astype(jnp.float32) * self.embed_inject_scale).astype(dtype)
            return x0 + jnp.where(valid, delta, jnp.zeros_like(delta))
        u = u * (1.0 / math.sqrt(x0.shape[-1]))
        if self.embed_inject == "embed":
            e = gm._embedding_gather(self.token_embed, token_ids)
            mixed = e.astype(jnp.float32) + u.astype(jnp.float32) * jax.lax.stop_gradient(rms(e.astype(jnp.float32)))
            return jnp.where(valid, self.embed_gated_norm(self.embed_norm(mixed.astype(e.dtype))).astype(dtype), x0)
        delta = (u.astype(jnp.float32) * jax.lax.stop_gradient(rms(x0.astype(jnp.float32)))).astype(dtype)
        return x0 + jnp.where(valid, delta, jnp.zeros_like(delta))

    def _top_state(self, x_in, out, slots, mixed=None):
        """What the input injection path of the next pass reads: the final-normalized output, or (the plain modes, as
        LoRM) the state after the last block and before the final norms. On the attention-residual backbone that is the
        final mixture `mixed`, the very tensor _trunk feeds to final_norm (_trunk(..., with_top=True));
        StdResTransformer rebuilds its residual stream instead."""
        if self.embed_inject not in PLAIN_INJECTS:
            return out
        if mixed is None:
            raise ValueError("the plain embedding injection reads the final mixture of _trunk(..., with_top=True)")
        return mixed

    def _trunk_top(self, x_in, ctx, prev):
        """_trunk plus the top state the input injection path of the next pass reads (_top_state)."""
        out, slots, layer_stats, mixed = self._trunk(x_in, ctx, prev, with_top=True)
        return out, slots, layer_stats, self._top_state(x_in, out, slots, mixed)

    def run_passes(self, token_ids, mask, n_passes: int, *, train: bool, context=None, context_passes: int = 1):
        """context [B, T] bool (evaluation only): prompt positions, processed with context_passes passes before the rest
        is decoded. A prompt position reads the previous position only in passes 2..context_passes and keeps its pass-
        context_passes state (and so its keys and values) in later passes, which the other positions read. context_passes
        1 is FBT's SOFT (the prompt never reads); context_passes >= n_passes is every position reading in every pass."""
        if context is not None and train:
            raise ValueError("context is evaluation-only")
        if context_passes < 1:
            raise ValueError(f"context_passes must be >= 1, got {context_passes}")
        ctx = self._context(token_ids, mask)
        x0 = self._embed(token_ids)
        fbt = self.feedback_share == "fbt"
        hold = context is not None and 1 < context_passes < n_passes
        read_ctx = ctx
        if context is not None:
            context = reshard(context, P(*tuple(jax.typeof(ctx["fb_valid"]).sharding.spec)))
            if context_passes == 1:
                read_ctx = dict(ctx, fb_valid=ctx["fb_valid"] & ~context)
        emb = self.embed_inject != "none"
        outs, stats, prev, held, top = [], [], None, None, None
        for k in range(n_passes):
            x_in = self._fbt_input(token_ids, x0, outs[-1], read_ctx, train=train, pass_idx=k - 1) if fbt and k else x0
            if emb and k:
                x_in = self._embed_inject_input(token_ids, x0, top, read_ctx, train=train, pass_idx=k - 1)
            reads = prev
            if hold and k + 1 == context_passes:
                held = x_in if fbt else (x_in, prev) if emb else prev
            elif hold and k + 1 > context_passes:
                if fbt:
                    x_in = hold_context(context, held, x_in)
                elif emb:
                    x_in = hold_context(context, held[0], x_in)
                    reads = hold_context(context, held[1], prev)
                else:
                    reads = hold_context(context, held, prev)
            if self.embed_inject in ("only", "plainonly"):
                reads = None
            if emb and k + 1 < n_passes:
                out, slots, layer_stats, top = self._trunk_top(x_in, ctx, reads)
            else:
                out, slots, layer_stats = self._trunk(x_in, ctx, reads)
            outs.append(out)
            stats.append(layer_stats)
            if k + 1 < n_passes and not fbt:
                prev = self.feedback_inputs(slots, read_ctx, train=train, token_ids=token_ids, pass_idx=k)
        return outs, stats

    def pass_losses(self, token_ids, loss_weight, *, mask=None, n_passes: int, context=None, context_passes: int = 1,
                    keep=None, loss_dtype=jnp.float32):
        """Per-position next-token loss (next_token_loss's reduction "none") of the passes in `keep` (1-based, default
        all), each [B, T]; context / context_passes as run_passes."""
        outs, _ = self.run_passes(token_ids, mask, n_passes, train=False, context=context, context_passes=context_passes)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        keep = range(1, n_passes + 1) if keep is None else keep
        return [gm.fused_linear_softmax_cross_entropy_loss(
            outs[p - 1], self.output_proj, labels, weight=loss_weight.astype(loss_dtype), reduction="none",
            logsumexp_weight=None, dtype=loss_dtype, implementation="xla_fast_bwd", block_sizes=gm._CE_BLOCK_SIZES)
            for p in keep]

    def router_metrics(self, stats_per_pass):
        cfg = self.config
        stacked = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *stats_per_pass)
        reduced = gm.reduce_moe_routing_stats(stacked, num_experts=cfg.num_experts,
                                              num_experts_per_token=cfg.num_experts_per_token)
        k = len(stats_per_pass)
        qb_beta = reduced["qb_beta"]
        qb_beta = jnp.mean(jnp.reshape(qb_beta, (k, cfg.num_layers, *qb_beta.shape[1:])), axis=0)
        return {
            "routing_entropy_per_layer": reduced["routing_entropy"],
            "routing_counts_per_layer": reduced["routing_counts"],
            "load_balancing_loss_per_layer": reduced["load_balancing_loss"],
            "router_z_loss_per_layer": reduced["router_z_loss"],
            "qb_beta_per_layer": qb_beta,
            "capacity_overflow_per_layer": stacked["capacity_overflow"],
            "sender_capacity_overflow_per_layer": stacked["sender_capacity_overflow"],
            "receiver_capacity_overflow_per_layer": stacked["receiver_capacity_overflow"],
            "skipped_assignments_per_layer": stacked["skipped_assignments"],
            "margin_min_per_layer": stacked["margin_min"],
            "margin_max_per_layer": stacked["margin_max"],
        }

    def __call__(self, token_ids, mask=None):
        outs, stats = self.run_passes(token_ids, mask, self.eval_passes, train=False)
        return outs[-1], self.router_metrics(stats)

    def next_token_loss(self, token_ids, loss_weight, *, mask=None, reduction="mean", logsumexp_weight=None,
                        loss_dtype=jnp.float32, return_router_metrics=False):
        train = reduction != "none"
        outs, stats = self.run_passes(token_ids, mask, self.passes if train else self.eval_passes, train=train)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)
        block = self.config.ce_grad_in_forward_token_block_size if reduction != "none" else None

        def ce(hidden):
            return gm.fused_linear_softmax_cross_entropy_loss(
                hidden, self.output_proj, labels, weight=loss_weight, reduction=reduction,
                logsumexp_weight=logsumexp_weight, dtype=loss_dtype,
                implementation=None if block is not None else "xla_fast_bwd",
                block_sizes=None if block is not None else gm._CE_BLOCK_SIZES,
                grad_in_forward_token_block_size=block,
            )

        if not train:
            return ce(outs[-1])
        ces = [ce(o) for o in outs]
        loss = ces[0] if len(ces) == 1 else ces[0] + sum(ces[1:]) / (len(ces) - 1)
        if not return_router_metrics:
            return loss
        router_metrics = self.router_metrics(stats)
        metrics = gm._summarize_router_metrics(router_metrics)
        metrics["train/cross_entropy_loss"] = ces[-1]
        for k, c in enumerate(ces):
            metrics[f"train/cross_entropy_loss_pass{k + 1}"] = c
        n_calls = router_metrics["router_z_loss_per_layer"].shape[0]
        metrics["train/router/z_loss_logging_only"] = jnp.sum(router_metrics["router_z_loss_per_layer"]) / n_calls
        if self.config.report_capacity_overflow:
            metrics[gm.MOE_DROPPED_ASSIGNMENTS_METRIC] = router_metrics["capacity_overflow_per_layer"]
            metrics[gm.MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC] = router_metrics["sender_capacity_overflow_per_layer"]
            metrics[gm.MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC] = router_metrics["receiver_capacity_overflow_per_layer"]
            metrics[gm.MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC] = router_metrics["skipped_assignments_per_layer"]
            metrics[gm.MOE_VALID_ASSIGNMENTS_METRIC] = jnp.sum(router_metrics["routing_counts_per_layer"], axis=-1,
                                                               dtype=jnp.int32)
        return loss, metrics


_SRC_SPEC = {"passes": 1, "eval_passes": 1, "src_block": -1, "src_rank": 64, "src_alpha": 1.0, "jitter": 0.0,
             "bos_id": -1, "jitter_seed": 20261106}
_ATTNRES_FIELDS = ("attnres_query", "frontier_logit", "frontier_entry_bias", "frontier_query", "frontier_down",
                   "frontier_up", "frontier_inject_down", "frontier_inject_up", "frontier_proj_down", "frontier_proj_up",
                   "frontier_proj_full")


class SrcTransformer(AttnResTransformer):
    """Marin's residual trunk with one fixed source of the previous position: FBT's single-source route (its residual
    mode, one low-rank W_H per target block; arXiv:2608.08888). Passes 2..K add
    alpha * up_l(down_l(norm(x_s))) of the previous position to the residual stream before every block l <= s, x_s the
    stream after block s (FBT captures norm(x), parameter-free); down uniform(+-sqrt(3 / D)), up zero, so the switch to
    two passes starts from the plain model. A plain Marin checkpoint branches into it with a partial restore (the maps
    are the only new leaves). Loss, evaluation passes and QB routing as AttnResTransformer."""
    src_down: Weight | None = None  # [s + 1, D, r]
    src_up: Weight | None = None  # [s + 1, r, D], zero-initialized
    src_block: int = eqx.field(static=True, default=-1)
    src_alpha: float = eqx.field(static=True, default=1.0)

    @staticmethod
    def init(cfg_or_vocab, config=None, *, key) -> "SrcTransformer":
        base = gm.Transformer.init(cfg_or_vocab, config, key=key)
        return SrcTransformer.from_base(base, key=key)

    @staticmethod
    def from_base(base: gm.Transformer, *, key, **overrides) -> "SrcTransformer":
        spec = dict(_SRC_SPEC, **overrides)
        cfg = base.config
        _validate_src(spec, cfg.num_layers)
        d, s, r = cfg.hidden_dim, int(spec["src_block"]), int(spec["src_rank"])
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(gm.Transformer)}
        bound = math.sqrt(3.0 / d)
        down = random.uniform(random.fold_in(key, 20261111), (s + 1, d, r), jnp.float32, -bound, bound)
        return SrcTransformer(
            **fields,
            **{name: None for name in _ATTNRES_FIELDS},
            src_down=Weight(_replicated(down)),
            src_up=Weight(_replicated(jnp.zeros((s + 1, r, d), jnp.float32))),
            passes=int(spec["passes"]),
            eval_passes=int(spec["eval_passes"]),
            feedback_share="src",
            jitter=float(spec["jitter"]),
            bos_id=int(spec["bos_id"]),
            jitter_seed=int(spec["jitter_seed"]),
            src_block=s,
            src_alpha=float(spec["src_alpha"]),
        )

    def _trunk(self, x0, ctx, prev):
        """One pass; returns (final-normalized state, the stream after block s, per-layer router stats)."""
        cfg = self.config
        x, source, stats = x0, None, []
        for i in range(cfg.num_layers):
            block = _layer_slice(self.stacked_blocks.stacked, i)
            is_long = ctx["flags"][i]
            layer_mask = ctx["long_mask"].with_fa4_bounds(ctx["long_lb"] if is_long else ctx["short_lb"], ctx["valid"])
            flag = jnp.asarray(is_long)
            if prev is not None and i <= self.src_block:
                x = x + self._src_delta(i, prev[0], x)
            x, layer_stats = eqx.filter_checkpoint(block, policy=ctx["policy"])(x, layer_mask, flag, flag)
            stats.append(layer_stats)
            if i == self.src_block:
                source = x
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *stats)
        return self.final_gated_norm(self.final_norm(x)), source, stacked

    def _src_delta(self, i: int, h, x):
        dtype = x.dtype
        spec = tuple(jax.typeof(x).sharding.spec)
        z = jnp.einsum("btd,dr->btr", h.astype(dtype), self.src_down.weight[i].astype(dtype), out_sharding=P(*spec[:2], None))
        u = jnp.einsum("btr,rd->btd", z, self.src_up.weight[i].astype(dtype), out_sharding=P(*spec))
        return self.src_alpha * u

    def feedback_inputs(self, source, ctx, *, train: bool, token_ids=None, pass_idx: int = 0):
        """(norm(x_s) of the previous position [B, T, D], zero without feedback; positions with feedback; None)."""
        dtype = source.dtype
        spec = P(*tuple(jax.typeof(source).sharding.spec))
        s32 = source.astype(jnp.float32)
        h = s32 * jax.lax.rsqrt(jnp.mean(jnp.square(s32), axis=-1, keepdims=True) + ATTN_RES_EPS)
        if train and self.jitter > 0.0:
            rms = jnp.sqrt(jnp.mean(jnp.square(jax.lax.stop_gradient(h)), axis=-1, keepdims=True))
            noise = random.uniform(self._jitter_key(token_ids, pass_idx), h.shape, jnp.float32, -self.jitter,
                                   self.jitter, out_sharding=spec)
            h = h + noise * rms
        h = jnp.pad(h[:, :-1], ((0, 0), (1, 0), (0, 0)))
        valid = ctx["fb_valid"]
        return (h * valid[..., None]).astype(dtype), valid, None


def _validate_src(spec, num_layers: int) -> None:
    if not 0 <= int(spec["src_block"]) < num_layers:
        raise ValueError(f"src_block must be in [0, {num_layers}), got {spec['src_block']}")
    if int(spec["src_rank"]) <= 0 or spec["passes"] < 1 or spec["eval_passes"] < 1 or spec["jitter"] < 0.0:
        raise ValueError(f"bad src spec {spec}")


STDRES_SHARES = ("pool", "fbt")
_STDRES_SPEC = {"passes": 1, "eval_passes": 1, "share": "pool", "query": "own", "init_share": 0.01, "proj": "none",
                "proj_rank": 0, "jitter": 0.0, "bos_id": -1, "jitter_seed": 20261106, "prefix_mixin": False,
                "fbt_input_norm": "unit", "depth": "whole", "inject": "residual", "source": "outputs",
                "embed_inject": "none"}
# Injection ablations on the residual trunk, which cross the two methods: inject "input" feeds sublayer 0's read over
# the whole depth through FBT's gate product in place of the token input (FBT's injection, ALF's read); source "top"
# has every sublayer read only the previous position's final state, through its own map, mixed in as ALF mixes (ALF's
# injection, FBT's read).
FEEDBACK_INJECTS = ("residual", "input")
FEEDBACK_SOURCES = ("outputs", "top")
PREFIX_MIXIN_BITS = 19  # position * 2^19 + draw fits int32 for sequences up to 4,096 tokens
PREFIX_MIXIN_KEY_OFFSET = 1000  # pass k draws its mask from _jitter_key(tokens, 1000 + k), apart from the jitter streams


def prefix_mixin_draws(key, shape, spec) -> jax.Array:
    """Uniform integers in [0, 2^19) per position; a document uses the draw at its first position."""
    u = random.uniform(key, shape, jnp.float32, 0.0, 1.0, out_sharding=spec)
    return jnp.minimum(jnp.floor(u * (1 << PREFIX_MIXIN_BITS)), (1 << PREFIX_MIXIN_BITS) - 1).astype(jnp.int32)


def prefix_mixin_mask(fb_valid: jax.Array, key) -> jax.Array:
    """Positions that read the previous position on one training pass under FBT's prefix mixin (arXiv:2608.08888):
    each document, starting where fb_valid is False, keeps a plain prefix of 1 + floor(u * length)
    tokens, u ~ U[0, 1) drawn independently per document, which may cover the whole document; the rest reads the
    previous position. fb_valid [B, T] -> [B, T] bool. Prefix scans only (no gather): the document's draw rides on
    position * 2^19 + draw through a running max."""
    start = ~fb_valid
    b, t = start.shape
    if t > (1 << (31 - PREFIX_MIXIN_BITS)):
        raise ValueError(f"prefix_mixin_mask supports up to {1 << (31 - PREFIX_MIXIN_BITS)} positions, got {t}")
    spec = P(*tuple(jax.typeof(fb_valid).sharding.spec))
    pos = reshard(jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (b, t)), spec)
    draws = prefix_mixin_draws(key, (b, t), spec)
    packed = jax.lax.cummax(jnp.where(start, pos * (1 << PREFIX_MIXIN_BITS) + draws, -1), axis=1)
    start_pos = packed >> PREFIX_MIXIN_BITS
    draw = packed & ((1 << PREFIX_MIXIN_BITS) - 1)
    next_start = jnp.pad(jnp.where(start, pos, t)[:, 1:], ((0, 0), (0, 1)), constant_values=t)
    end_pos = jax.lax.cummin(next_start, axis=1, reverse=True)
    plain = 1 + ((draw * (end_pos - start_pos)) >> PREFIX_MIXIN_BITS)
    return (pos - start_pos) >= plain


def document_offsets(fb_valid: jax.Array) -> jax.Array:
    """Offset of every position from the start of its document (a position where fb_valid is False: BOS, a new segment,
    position 0), [B, T] int32."""
    start = ~fb_valid
    b, t = start.shape
    spec = P(*tuple(jax.typeof(fb_valid).sharding.spec))
    pos = reshard(jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (b, t)), spec)
    return pos - jax.lax.cummax(jnp.where(start, pos, 0), axis=1)


def hold_context(context: jax.Array, held, fresh):
    """Per position: `held` at prompt positions, `fresh` elsewhere (pytrees of [B, T, ...] leaves; None leaves pass)."""
    def pick(h, f):
        c = jnp.reshape(context, context.shape + (1,) * (f.ndim - 2))
        c = reshard(jnp.broadcast_to(c, f.shape), P(*tuple(jax.typeof(f).sharding.spec)))
        return jnp.where(c, h, f)

    return jax.tree.map(pick, held, fresh)


def residual_mix(j: int, x: jax.Array, prev, flogit: jax.Array, win: int = 0) -> jax.Array:
    """Input of sublayer j on a residual trunk with a previous pass: (1 - rho_j) x + rho_j c n_j, x the residual stream
    before sublayer j, n_j the feedback read (the previous position's outputs j..2L-1 under sublayer j's query, as `mix`
    share "pool"), c = sqrt((|x|^2 + D eps) / (|n_j|^2 + D eps)), so that rho_j is the share of the input whatever the
    size of x (as in FBT's Qwen3.5 adapter, arXiv:2608.08888). rho_j = 0 (a zero share, or no feedback) gives x
    exactly. win > 0 (depth "win<k>", the fixed nearest window of Section 6.2) reads only outputs
    j..min(j + win - 1, 2L - 1), equally weighted (query "uniform", so there is no softmax to narrow); c rescales the
    mean to the stream either way, so the window does not change the size of what is mixed in."""
    pv, valid, ps = prev
    dtype = x.dtype
    spec = tuple(jax.typeof(x).sharding.spec)
    window = pv[:, :, j:] if win <= 0 else pv[:, :, j:j + win]
    if ps is None:
        n = jnp.mean(window.astype(jnp.float32), axis=2).astype(dtype)
    else:
        w = jax.nn.softmax(ps[:, :, j:, j], axis=-1).astype(dtype)
        n = jnp.einsum("bts,btsd->btd", w, window, out_sharding=P(*spec))
    eps_d = x.shape[-1] * ATTN_RES_EPS
    sq_x = jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    sq_n = jnp.sum(jnp.square(n.astype(jnp.float32)), axis=-1, keepdims=True)
    rho = (jax.nn.sigmoid(flogit[j].astype(jnp.float32)) * valid.astype(jnp.float32))[..., None]
    coef = rho * jnp.sqrt((sq_x + eps_d) / (sq_n + eps_d))
    return x * (1.0 - rho).astype(dtype) + n * coef.astype(dtype)


def _stdres_layer(i: int, block, x, prev, flogit, layer_mask, flag, win: int = 0):
    """Layer i of Marin's residual trunk in one remat region, as Block.__call__ under Transformer's per-layer
    checkpoint: (stream after the layer, attention output, MoE output, router stats). With a previous pass each sublayer
    reads residual_mix of the stream over `win` of the previous position's outputs (0: all of them from its own index
    on); the outputs are still added to the stream itself."""
    h = x if prev is None else residual_mix(2 * i, x, prev, flogit, win)
    attn_out = _attn_branch(block, h, layer_mask, flag)
    x = x + attn_out
    h = x if prev is None else residual_mix(2 * i + 1, x, prev, flogit, win)
    mlp_out, stats = _mlp_branch(block, h, layer_mask)
    return x + mlp_out, attn_out, mlp_out, stats


class StdResTransformer(AttnResTransformer):
    """Marin's residual trunk -- the standard-residual backbone -- reading the previous position in passes 2..K:
      share "pool", the feedback path on a standard transformer: sublayer j reads residual_mix of the residual stream,
        n_j a softmax over the previous position's outputs j..2L-1 with its own zero-initialized query (query "own";
        "uniform": their mean), each output through its own map M_k = I + up_k(down_k(.)) (proj "slot", rank proj_rank,
        up zero-initialized; proj_rank 0: a full zero-initialized D x D Delta_k), share rho_j = sigmoid(a_j) starting at
        init_share; the stream, the final norms and the LM head are Marin's. depth "win<k>" drops the softmax for the
        fixed nearest window of Section 6.2: equal weights over outputs j..min(j + k - 1, 2L - 1), with query
        "uniform";
      share "fbt": no previous output is read; the previous pass's top state enters through FBT's layer-0 gate product
        (AttnResTransformer._fbt_input, the same fusion as on the attention-residual backbone), which replaces the
        embedded token as the start of the residual stream at positions with feedback; the reported FBT runs use
        fbt_input_norm "unit" (RMS 1, ~11x the token input), while "embed" / "match" bring it to the token input's
        norms / scale.
    The input injection path (embed_inject) rides on share "pool" here, in any of its modes; a plain mode's "@<k>"
    reads the stream after block k instead of after the last one (embed_inject_src, _top_state).
    One pass (or a zero share) is Marin's Transformer, so a plain Marin checkpoint (--variant plain) branches into it
    with a partial restore: the new parameters are its only new leaves, initialized as the AttnRes variants initialize
    the same leaves.
    Loss, evaluation passes, jitter, document starts and QB routing as AttnResTransformer; attnres_query is None.
    prefix_mixin (training passes only): pass k >= 2 reads the previous position only where prefix_mixin_mask says so,
    a fresh mask per pass and batch, so every document also trains an independently sampled plain prefix."""
    prefix_mixin: bool = eqx.field(static=True, default=False)
    feedback_replaces_input: str = eqx.field(static=True, default="residual")
    feedback_source: str = eqx.field(static=True, default="outputs")

    @staticmethod
    def init(cfg_or_vocab, config=None, *, key) -> "StdResTransformer":
        base = gm.Transformer.init(cfg_or_vocab, config, key=key)
        return StdResTransformer.from_base(base, key=key)

    @staticmethod
    def from_base(base: gm.Transformer, *, key, **overrides) -> "StdResTransformer":
        spec = dict(_STDRES_SPEC, **overrides)
        _validate_stdres(spec)
        cfg = base.config
        d, n_mix = cfg.hidden_dim, 2 * cfg.num_layers
        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(gm.Transformer)}
        new = {name: None for name in _ATTNRES_FIELDS}
        fbt_state = fbt_gate = None
        if spec["share"] == "fbt" or spec["inject"] == "input":
            bound = math.sqrt(3.0 / d)
            fbt_state = Weight(_replicated(random.uniform(random.fold_in(key, 20261109), (d, d), jnp.float32, -bound, bound)))
            fbt_gate = Weight(_replicated(random.uniform(random.fold_in(key, 20261110), (d, d), jnp.float32, -bound, bound)))
        if spec["share"] != "fbt":
            p = float(spec["init_share"])
            new["frontier_logit"] = Weight(_replicated(jnp.full((n_mix,), math.log(p / (1.0 - p)), jnp.float32)))
            if spec["query"] == "own":
                new["frontier_query"] = Weight(_replicated(jnp.zeros((n_mix, d), jnp.float32)))
            if spec["proj"] == "slot":
                r = int(spec["proj_rank"])
                if r > 0:
                    k = random.fold_in(key, 20261108)
                    new["frontier_proj_down"] = Weight(_replicated(gm._init_weight(k, (n_mix, d, r), cfg.initializer_std)))
                    new["frontier_proj_up"] = Weight(_replicated(jnp.zeros((n_mix, r, d), jnp.float32)))
                else:
                    new["frontier_proj_full"] = Weight(_replicated(jnp.zeros((n_mix, d, d), jnp.float32)))
        embed_proj = None
        embed_mode, embed_scale = parse_embed_inject(spec["embed_inject"])
        embed_src = embed_inject_source(spec["embed_inject"])
        if not 0 <= embed_src < cfg.num_layers:
            raise ValueError(f"embed_inject @<k> needs 0 < k < num_layers ({cfg.num_layers}), got {embed_src}")
        if embed_mode != "none":
            embed_proj = Weight(_replicated(jnp.zeros((d, d), jnp.float32)))
        return StdResTransformer(
            **fields,
            **new,
            fbt_state_proj=fbt_state,
            fbt_token_gate=fbt_gate,
            passes=int(spec["passes"]),
            eval_passes=int(spec["eval_passes"]),
            feedback_share=spec["share"],
            feedback_query_kind=spec["query"],
            feedback_proj=spec["proj"],
            jitter=float(spec["jitter"]),
            bos_id=int(spec["bos_id"]),
            jitter_seed=int(spec["jitter_seed"]),
            prefix_mixin=bool(spec["prefix_mixin"]),
            fbt_input_norm=spec["fbt_input_norm"],
            feedback_depth=spec["depth"],
            feedback_replaces_input=spec["inject"],
            feedback_source=spec["source"],
            embed_inject_proj=embed_proj,
            embed_inject=embed_mode,
            embed_inject_scale=embed_scale,
            embed_inject_src=embed_src,
        )

    def feedback_inputs(self, slots, ctx, *, train: bool, token_ids=None, pass_idx: int = 0):
        """As AttnResTransformer.feedback_inputs; depth "top" (the last-block ablation on the residual trunk)
        sets the score of every previous output below the last layer to -inf for sublayers j < 2L - 2, so residual_mix's
        softmax over outputs j..2L-1 is exactly a softmax over max(j, 2L-2)..2L-1, as `mix` reads with depth "top"."""
        pv, valid, ps = super().feedback_inputs(slots, ctx, train=train, token_ids=token_ids, pass_idx=pass_idx)
        if self.feedback_depth == "top" and ps is not None:
            n_mix = 2 * self.config.num_layers
            s_idx = jnp.arange(n_mix)[:, None]
            j_idx = jnp.arange(n_mix)[None, :]
            keep = s_idx >= jnp.maximum(j_idx, n_mix - 2)
            ps = jnp.where(keep, ps, jnp.asarray(-jnp.inf, ps.dtype))
        if self.feedback_source == "top" and ps is not None:
            # every slot holds the previous final state, slot k through map M_k: sublayer j reads slot j alone
            n_mix = 2 * self.config.num_layers
            keep = jnp.arange(n_mix)[:, None] == jnp.arange(n_mix)[None, :]
            ps = jnp.where(keep, ps, jnp.asarray(-jnp.inf, ps.dtype))
        return pv, valid, ps

    def _feedback_input(self, token_ids, x0, prev, ctx):
        """inject "input": the read of sublayer 0 (softmax of its query over all 2L previous outputs, each through its
        map, as residual_mix's n_0) enters through FBT's gate product in place of the token input."""
        pv, valid, ps = prev
        w = jax.nn.softmax(ps[:, :, :, 0], axis=-1).astype(pv.dtype)
        n = jnp.einsum("bts,btsd->btd", w, pv, out_sharding=P(*tuple(jax.typeof(x0).sharding.spec)))
        return self._fbt_fuse(token_ids, x0, n, ctx)

    def _run_ablation(self, token_ids, mask, n_passes: int, *, train: bool):
        """Passes for inject "input" or source "top" (no decoding modes, no prefix mixin)."""
        ctx = self._context(token_ids, mask)
        x0 = self._embed(token_ids)
        n_mix = 2 * self.config.num_layers
        outs, stats, prev, x_in = [], [], None, x0
        for k in range(n_passes):
            out, slots, layer_stats = self._trunk(x_in, ctx, None if self.feedback_replaces_input == "input" else prev)
            outs.append(out)
            stats.append(layer_stats)
            if k + 1 < n_passes:
                src = [out] * n_mix if self.feedback_source == "top" else slots
                prev = self.feedback_inputs(src, ctx, train=train, token_ids=token_ids, pass_idx=k)
                if self.feedback_replaces_input == "input":
                    x_in = self._feedback_input(token_ids, x0, prev, ctx)
        return outs, stats

    def run_passes(self, token_ids, mask, n_passes: int, *, train: bool, context=None, context_passes: int = 1):
        if self.feedback_replaces_input != "residual" or self.feedback_source != "outputs":
            if context is not None:
                raise ValueError("decoding modes (context) are not implemented for inject 'input' / source 'top'")
            return self._run_ablation(token_ids, mask, n_passes, train=train)
        if not (train and self.prefix_mixin):
            return super().run_passes(token_ids, mask, n_passes, train=train, context=context,
                                      context_passes=context_passes)
        if context is not None:
            raise ValueError("context is evaluation-only")
        ctx = self._context(token_ids, mask)
        x0 = self._embed(token_ids)
        fbt = self.feedback_share == "fbt"
        reads = [ctx] + [dict(ctx, fb_valid=prefix_mixin_mask(ctx["fb_valid"],
                                                              self._jitter_key(token_ids, PREFIX_MIXIN_KEY_OFFSET + k)))
                         for k in range(1, n_passes)]
        outs, stats, prev, top = [], [], None, None
        for k in range(n_passes):
            x_in = self._fbt_input(token_ids, x0, outs[-1], reads[k], train=train, pass_idx=k - 1) if fbt and k else x0
            if self.embed_inject != "none" and k:
                x_in = self._embed_inject_input(token_ids, x0, top, reads[k], train=train, pass_idx=k - 1)
            out, slots, layer_stats = self._trunk(x_in, ctx, None if self.embed_inject in ("only", "plainonly") else prev)
            outs.append(out)
            stats.append(layer_stats)
            if self.embed_inject != "none" and k + 1 < n_passes:
                top = self._top_state(x_in, out, slots)
            if k + 1 < n_passes and not fbt:
                prev = self.feedback_inputs(slots, reads[k + 1], train=train, token_ids=token_ids, pass_idx=k)
        return outs, stats

    def _top_state(self, x_in, out, slots, mixed=None):
        """What the input injection path of the next pass reads: the final-normalized output, or (the plain modes, as
        LoRM) the residual stream after the last block, rebuilt as the trunk builds it -- x_in plus the sublayer outputs
        in order. embed_inject_src k > 0 ("@<k>") stops after the first k blocks, that is after 2k sublayer outputs."""
        if self.embed_inject not in PLAIN_INJECTS:
            return out
        x = x_in
        for s in (slots if self.embed_inject_src <= 0 else slots[:2 * self.embed_inject_src]):  # "@<k>": after block k
            x = x + s
        return x

    def _trunk_top(self, x_in, ctx, prev):
        """_trunk plus the top state the input injection path of the next pass reads (_top_state)."""
        out, slots, layer_stats = self._trunk(x_in, ctx, prev)
        return out, slots, layer_stats, self._top_state(x_in, out, slots)

    def _trunk(self, x0, ctx, prev):
        """One pass; returns (final-normalized state, sublayer outputs 1..2L, per-layer router stats)."""
        flogit = None if self.frontier_logit is None else self.frontier_logit.weight
        x, outs, stats = x0, [], []
        for i in range(self.config.num_layers):
            block = _layer_slice(self.stacked_blocks.stacked, i)
            is_long = ctx["flags"][i]
            layer_mask = ctx["long_mask"].with_fa4_bounds(ctx["long_lb"] if is_long else ctx["short_lb"], ctx["valid"])
            flag = jnp.asarray(is_long)
            fn = functools.partial(_stdres_layer, i, win=parse_depth(self.feedback_depth)[1])
            x, attn_out, mlp_out, layer_stats = eqx.filter_checkpoint(fn, policy=ctx["policy"])(
                block, x, prev, flogit, layer_mask, flag)
            outs += [attn_out, mlp_out]
            stats.append(layer_stats)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *stats)
        return self.final_gated_norm(self.final_norm(x)), outs, stacked


def _validate_stdres(spec) -> None:
    if spec["share"] not in STDRES_SHARES:
        raise ValueError(f"stdres share must be one of {STDRES_SHARES}, got {spec['share']!r}")
    if spec["passes"] < 1 or spec["eval_passes"] < 1 or spec["jitter"] < 0.0 or not 0.0 < spec["init_share"] < 1.0:
        raise ValueError(f"bad stdres spec {spec}")
    if spec["query"] not in FEEDBACK_QUERIES or spec["proj"] not in ("none", "slot") or spec["proj_rank"] < 0:
        raise ValueError(f"stdres query must be one of {FEEDBACK_QUERIES}, proj none or slot (one map per previous output) "
                         f"and proj_rank >= 0, got {spec['query']!r} / {spec['proj']!r} / {spec['proj_rank']}")
    if spec["share"] == "pool" and spec["query"] == "shared":
        raise ValueError("stdres share pool scores with its own query (own) or not at all (uniform): the residual trunk "
                         "has no AttnRes pseudo-query to share")
    if spec["share"] == "fbt" and spec["proj"] != "none":
        raise ValueError("stdres share fbt reads no previous output: proj none")
    if not isinstance(spec["prefix_mixin"], bool):
        raise ValueError(f"prefix_mixin must be a bool, got {spec['prefix_mixin']!r}")
    depth_mode, depth_win = parse_depth(spec["depth"])
    if depth_mode == "win":
        if spec["share"] != "pool" or spec["query"] != "uniform":
            # residual_mix's softmax branch reads ps[:, :, j:, j], which is not windowed: a window with a query would
            # be a shape mismatch, not a third variant. Windowing ps too is what a future "window + softmax" needs.
            raise ValueError("stdres depth 'win<k>' (the fixed nearest-window rule) goes with share 'pool' and query "
                             "'uniform': equal weights over the window, no depth softmax and no pseudo-query")
    elif spec["depth"] != "whole" and (spec["share"] != "pool" or spec["query"] != "own"):
        raise ValueError("stdres depth 'top' (read only the previous position's last layer) goes with share 'pool' and "
                         "query 'own'")
    if spec["inject"] not in FEEDBACK_INJECTS or spec["source"] not in FEEDBACK_SOURCES:
        raise ValueError(f"inject must be one of {FEEDBACK_INJECTS} and source one of {FEEDBACK_SOURCES}, got "
                         f"{spec['inject']!r} / {spec['source']!r}")
    if (spec["inject"], spec["source"]) != ("residual", "outputs"):
        if spec["inject"] != "residual" and spec["source"] != "outputs":
            raise ValueError("inject 'input' with source 'top' is FBT's fusion: use share 'fbt'")
        if (spec["share"] != "pool" or spec["query"] != "own" or spec["depth"] != "whole" or spec["prefix_mixin"]
                or spec["jitter"] != 0.0):
            raise ValueError("inject 'input' / source 'top' go with share 'pool', query 'own', depth 'whole', no prefix "
                             "mixin and no jitter")
        if spec["source"] == "top" and spec["proj"] != "slot":
            raise ValueError("source 'top' passes the final state through each sublayer's own map: proj 'slot'")
    if parse_embed_inject(spec["embed_inject"])[0] != "none" and (spec["share"] != "pool" or spec["inject"] != "residual"
                                           or spec["source"] != "outputs"):
        raise ValueError("embed_inject adds the previous top state to the token input of the feedback path: share "
                         "'pool', inject 'residual', source 'outputs'")
    _validate_fbt_input_norm(dict(spec, share="fbt") if spec["inject"] == "input" else spec)


def embed_inject_source(value: str) -> int:
    """The block the input injection path reads, from a "plain:<c>@<k>" / "plainonly:<c>@<k>" value: k means the
    previous position's residual stream after block k (k sublayer pairs) instead of after the last block, on the
    standard-residual backbone (StdResTransformer) only; 0 (no suffix) is the last block. The paper's method section
    and its appendix on what the feedback reads cover the choice."""
    parse_embed_inject(value)
    _, _, src = str(value).partition("@")
    return int(src) if src else 0


def read_embed_inject(run_dir: str) -> str:
    """The embed_inject a run's checkpoints were trained with (EMBED_INJECT_FILE; no file: "none")."""
    path = os.path.join(run_dir, EMBED_INJECT_FILE)
    if not os.path.exists(path):
        return "none"
    with open(path) as f:
        value = f.read().strip()
    try:
        parse_embed_inject(value)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    return value


def record_embed_inject(run_dir: str, value: str) -> None:
    """Training start of a run with embed_inject: write the mode, or refuse a resume with another one."""
    if parse_embed_inject(value)[0] == "none":
        raise ValueError(f"record_embed_inject takes a mode other than none, got {value!r}")
    path = os.path.join(run_dir, EMBED_INJECT_FILE)
    if os.path.exists(path):
        old = read_embed_inject(run_dir)
        if old != value:
            raise ValueError(f"{run_dir} was trained with embed_inject {old!r} ({path}); this start asks for {value!r}")
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(value + "\n")
    os.replace(path + ".tmp", path)


def _validate_fbt_input_norm(spec) -> None:
    if spec["fbt_input_norm"] not in FBT_INPUT_NORMS or (spec["fbt_input_norm"] != "unit" and spec["share"] != "fbt"):
        raise ValueError(f"fbt_input_norm must be one of {FBT_INPUT_NORMS}, and only share 'fbt' leaves 'unit'; got "
                         f"{spec['fbt_input_norm']!r} with share {spec['share']!r}")


def read_fbt_input_norm(run_dir: str) -> str:
    """The fbt_input_norm a run's checkpoints were trained with (FBT_INPUT_NORM_FILE; no file: "unit", every run
    before the file existed)."""
    path = os.path.join(run_dir, FBT_INPUT_NORM_FILE)
    if not os.path.exists(path):
        return "unit"
    with open(path) as f:
        value = f.read().strip()
    if value not in FBT_INPUT_NORMS:
        raise ValueError(f"{path}: unknown fbt_input_norm {value!r}")
    return value


def record_fbt_input_norm(run_dir: str, value: str) -> None:
    """Training start of a share-"fbt" run: write its fbt_input_norm, or refuse a resume with another one (the
    checkpoints were trained with the recorded input path)."""
    if value not in FBT_INPUT_NORMS:
        raise ValueError(f"unknown fbt_input_norm {value!r}")
    path = os.path.join(run_dir, FBT_INPUT_NORM_FILE)
    if os.path.exists(path):
        old = read_fbt_input_norm(run_dir)
        if old != value:
            raise ValueError(f"{run_dir} was trained with fbt_input_norm {old!r} ({path}); this start asks for {value!r}")
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(value + "\n")
    os.replace(path + ".tmp", path)


def run_fbt_input_norm(run_dir: str, share: str) -> str:
    """fbt_input_norm to rebuild a run's model with for evaluation: the run's record, which only share 'fbt' may carry
    as other than 'unit'."""
    value = read_fbt_input_norm(run_dir)
    if value != "unit" and share != "fbt":
        raise ValueError(f"{run_dir} trained FBT's fusion with fbt_input_norm {value!r}; evaluated here as share {share!r}")
    return value


def calibrate_joint_bias(model: AttnResTransformer, token_ids, mask=None, target: float | None = None, iters: int = 60):
    """Entry biases b [2L] (share "joint") so that, on this batch, the previous slots start with `target` of the weight:
    the second pass with the previous slots at zero weight is the first pass, whose per-mixing-point score gap
    LSE(previous) - LSE(same token) at feedback positions gives share = sigmoid(b_j + gap_j); b_j by bisection on the
    mean (global over shards). The same rule as FBT's calibration (arXiv:2608.08888), run once at the branch point."""
    if model.feedback_share != "joint":
        raise ValueError("calibration applies to feedback_share='joint'")
    target = _SPEC["init_share"] if target is None else target
    cfg = model.config
    n_mix = 2 * cfg.num_layers
    eps_d = cfg.hidden_dim * ATTN_RES_EPS
    ctx = model._context(token_ids, mask)
    x0 = model._embed(token_ids)
    _, slots, _ = model._trunk(x0, ctx, None)
    q = model.attnres_query.weight.astype(x0.dtype)
    scores = [slot_scores(s, q, eps_d) for s in [x0, *slots]]
    _, valid, ps = model.feedback_inputs(slots, ctx, train=False)
    validf = valid.astype(jnp.float32)
    count = jnp.sum(validf)
    gaps = []
    for j in range(n_mix):
        same = jnp.stack([sc[..., j] for sc in scores[: j + 1]], axis=-1)
        gaps.append(jax.nn.logsumexp(ps[:, :, j:, j], axis=-1) - jax.nn.logsumexp(same, axis=-1))
    gaps = jnp.stack(gaps, axis=0)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        share = jnp.sum(jax.nn.sigmoid(mid[:, None, None] + gaps) * validf[None], axis=(1, 2)) / count
        below = share < target
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iters, body, (jnp.full((n_mix,), -80.0), jnp.full((n_mix,), 80.0)))
    return 0.5 * (lo + hi)


def _validate(spec) -> None:
    if spec["share"] not in FEEDBACK_SHARES:
        raise ValueError(f"share must be one of {FEEDBACK_SHARES}, got {spec['share']!r}")
    if spec["query"] not in FEEDBACK_QUERIES:
        raise ValueError(f"query must be one of {FEEDBACK_QUERIES}, got {spec['query']!r}")
    if spec["share"] == "joint" and spec["query"] != "shared":
        raise ValueError("share 'joint' scores the previous slots with the mixing point's own pseudo-query")
    if spec["share"] == "none" and (spec["passes"] > 1 or spec["eval_passes"] > 1):
        raise ValueError("more than one pass needs a feedback share")
    if spec["passes"] < 1 or spec["eval_passes"] < 1 or spec["rank"] < 0 or not 0.0 < spec["init_share"] < 1.0:
        raise ValueError(f"bad spec {spec}")
    if spec["jitter"] < 0.0:
        raise ValueError("jitter must be >= 0")
    if spec["inject_rank"] < 0 or (spec["share"] == "inject" and spec["inject_rank"] == 0):
        raise ValueError("share 'inject' needs inject_rank > 0 (the rank of each layer's map); inject_rank >= 0")
    if spec["proj"] not in FEEDBACK_PROJS or spec["proj_rank"] < 0:
        raise ValueError(f"proj must be one of {FEEDBACK_PROJS} and proj_rank >= 0, got {spec['proj']!r} / {spec['proj_rank']}")
    if spec["proj"] != "none" and (spec["share"] not in ("joint", "pool") or spec["rank"] != 0):
        raise ValueError("proj 'source' / 'slot' goes with share 'joint' or 'pool' and rank 0 (as in FBT)")
    if spec["depth"] not in FEEDBACK_DEPTHS:
        raise ValueError(f"variant feedback takes depth {FEEDBACK_DEPTHS} (win<k> is the residual trunk's rule, variant "
                         f"stdres), got {spec['depth']!r}")
    if spec["depth"] != "whole" and spec["share"] != "pool":
        raise ValueError("depth 'top' (read only the previous token's last layer) goes with share 'pool'")
    if spec["entry_bias"] is not None and spec["share"] != "joint":
        raise ValueError("entry_bias (calibrated entry biases) goes with share 'joint'")
    if spec["share"] == "fbt" and (spec["rank"] != 0 or spec["query"] != "shared"):
        raise ValueError("share 'fbt' reads no previous slot: rank 0 and the default query")
    embed_mode = parse_embed_inject(spec["embed_inject"])[0]
    if embed_inject_source(spec["embed_inject"]):
        raise ValueError("embed_inject @<k> reads the residual stream after block k: residual trunk (StdResTransformer) only")
    if embed_mode not in ("none", "plain", "plainonly"):
        raise ValueError(f"the AttnRes trunk takes embed_inject none, plain:<c> or plainonly:<c>; {embed_mode!r} is "
                         "retained for checkpoints trained before the final form and exists on the residual trunk "
                         "only (StdResTransformer)")
    if embed_mode != "none" and spec["share"] != "pool":
        raise ValueError("embed_inject plain:<c> / plainonly:<c> adds the previous position's final mixture to slot 0 of "
                         f"a share 'pool' model (joint, inject, fbt and none do not take it), got share {spec['share']!r}")
    _validate_fbt_input_norm(spec)


def install(**spec) -> None:
    """Point Marin's trainer at AttnResTransformer: passes / eval_passes / share / query / rank / init_share / jitter /
    bos_id / jitter_seed / inject_rank / proj / proj_rank / entry_bias / depth / fbt_input_norm / embed_inject (none,
    plain:<c> or plainonly:<c>, with share pool). Assignment accounting and analytic FLOPs count
    passes x layers MoE calls (the AttnRes mixing and the feedback reading are not priced)."""
    if spec.get("entry_bias") is not None:
        spec["entry_bias"] = tuple(float(b) for b in spec["entry_bias"])
    unknown = set(spec) - set(_SPEC)
    if unknown:
        raise ValueError(f"unknown spec keys {sorted(unknown)}")
    merged = dict(_SPEC, **spec)
    _validate(merged)
    _patch_trainer(AttnResTransformer, int(merged["passes"]))
    _SPEC.update(merged)


def install_src(**spec) -> None:
    """Point Marin's trainer at SrcTransformer: passes / eval_passes / src_block / src_rank / src_alpha / jitter / bos_id
    / jitter_seed (accounting as install())."""
    unknown = set(spec) - set(_SRC_SPEC)
    if unknown:
        raise ValueError(f"unknown spec keys {sorted(unknown)}")
    merged = dict(_SRC_SPEC, **spec)
    if merged["src_block"] < 0:
        raise ValueError("install_src needs src_block")
    _patch_trainer(SrcTransformer, int(merged["passes"]))
    _SRC_SPEC.update(merged)


def install_stdres(**spec) -> None:
    """Point Marin's trainer at StdResTransformer: passes / eval_passes / share / query / init_share / proj / proj_rank /
    jitter / bos_id / jitter_seed / prefix_mixin / fbt_input_norm / depth / inject / source / embed_inject (accounting as
    install())."""
    unknown = set(spec) - set(_STDRES_SPEC)
    if unknown:
        raise ValueError(f"unknown spec keys {sorted(unknown)}")
    merged = dict(_STDRES_SPEC, **spec)
    _validate_stdres(merged)
    _patch_trainer(StdResTransformer, int(merged["passes"]))
    _STDRES_SPEC.update(merged)


def _patch_trainer(cls, passes: int) -> None:
    import experiments.grug.moe_hero_ep.train as grug_train

    if grug_train.Transformer is not gm.Transformer:
        raise RuntimeError("another model patch is already installed")
    grug_train.Transformer = cls
    drop_metrics, compute_flops = grug_train._drop_metrics, grug_train._compute_flops

    def _drop_metrics(*args, num_layers: int, **kwargs):
        return drop_metrics(*args, num_layers=num_layers * passes, **kwargs)

    def _compute_flops(*, model_config):
        return compute_flops(model_config=dataclasses.replace(model_config, num_layers=model_config.num_layers * passes))

    grug_train._drop_metrics = _drop_metrics
    grug_train._compute_flops = _compute_flops


def optimizer_groups(params, optimizer_config) -> dict[str, str]:
    """Group of every new parameter leaf under Marin's optimizer mask (expected: all "adam")."""
    from levanter.utils.jax_utils import leaf_key_paths

    labels = optimizer_config.create_mask(params)
    paths = leaf_key_paths(params)
    out = {}
    for label, path in zip(jax.tree.leaves(labels), jax.tree.leaves(paths)):
        name = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        if name.split(".")[0] in NEW_PARAM_FIELDS:
            out[name] = label
    return out
