"""The three rungs of the scaling ladder: their shapes and their schedules.

Two rungs, d768 (MoE-1.6B) and d1024 (MoE-4.0B), are Marin's own; d896 (MoE-2.6B) interpolates between them and is
registered into Marin's SMALL_SHAPES in this process only, so no file of the Marin checkout is modified. Everything
is derived with Marin's own functions, so a rung here is the rung Marin's ladder would have.

The interpolation, step by step: width sqrt(768 * 1024) = 887 rounds to 896, the only multiple of the 128-wide head
between the two rungs; depth comes from Marin's MoE heuristic round(d / (55 + 4 log2 d)), rounded up to even, giving
10; heads are d / 128 = 7; KV heads are local heads // 4 = 1 and global max(1, heads // 8) = 1; the MoE is 384 routed
experts, top-8, of width d / 2 = 448 in a 448-wide latent, plus two shared experts of 448; the batch is
1024 * sqrt(2) = 1448 rounded to 1440 sequences (180 rows per GPU, which divides into 2 to 6 micro-batches); the step
count follows Marin's 791 tokens per active parameter, and the optimizer from Marin's MoE heuristic at that budget.

The rungs are selected with `--shape`; `alf.rungs.schedule` gives the interpolated rung its (batch, steps) and
`register_mid_shapes` makes d896 known to Marin for this process.
"""
from __future__ import annotations


TOKENS_PER_ACTIVE_PARAM = 791
MID_BATCH = 1440
MID_SHAPES = {"d896": (896, 10, 7, 1, 1)}  # hidden, layers, heads, local KV heads, global KV heads (Marin SmallShape order)


def register_mid_shapes() -> None:
    from experiments.grug.moe_hero_ep.small_scale_abl_launch import SMALL_SHAPES, SmallShape

    for name, dims in MID_SHAPES.items():
        SMALL_SHAPES.setdefault(name, SmallShape(*dims))


def schedule(active: int, seq_len: int = 4096) -> tuple[int, int]:
    """The interpolated rung's (batch, steps). Marin's own ladder_schedule covers its own two rungs."""
    return MID_BATCH, max(1, round(TOKENS_PER_ACTIVE_PARAM * active / (MID_BATCH * seq_len)))
