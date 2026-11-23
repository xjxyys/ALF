"""Gradient accumulation for Marin's grug MoE hero train step (``marin-community/marin`` at
``8577d258214dd0dcb992c04458ee4eb10496d4e5``), which has none: its jitted train_step calls the module-level
_loss_and_grads once per step, and install(n) makes that call a lax.scan over n microbatches. Row i of every batch
shard goes to microbatch i mod n, so the split moves no data and every microbatch is spread over all shards like the
full batch. This is how a rung's batch is trained on fewer devices than the recipe assumes.

Gradients and the loss are averaged (each microbatch's loss is its own token mean, and the microbatches hold equally
many rows); expert-assignment counts are summed, so the trainer's assignment accounting still covers the full batch;
the QB betas, the cross-entropy and the other per-step means are averaged; summaries that do not average (histograms,
margins) come from the last microbatch. The one change against a single full batch: the QB thresholds are the mean of
n smaller-batch estimates instead of one full-batch estimate.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

import experiments.grug.moe_hero_ep.model as gm

_SUMMED = (
    gm.MOE_DROPPED_ASSIGNMENTS_METRIC,
    gm.MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC,
    gm.MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC,
    gm.MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
    gm.MOE_VALID_ASSIGNMENTS_METRIC,
    "train/router/routing_counts_per_layer",
)


def _split(x: jax.Array, n: int) -> jax.Array:
    """[B, ...] -> [n, B / n, ...] with microbatch j = rows j, j + n, ... of every shard."""
    spec = jax.typeof(x).sharding.spec
    lead, rest = (spec[0] if len(spec) else None), tuple(spec[1:]) + (None,) * (x.ndim - max(len(spec), 1))
    y = jnp.reshape(x, (x.shape[0] // n, n, *x.shape[1:]), out_sharding=P(lead, None, *rest))
    return jnp.moveaxis(y, 1, 0)


def install(n: int) -> None:
    import experiments.grug.moe_hero_ep.train as grug_train

    if n <= 1:
        return
    if getattr(grug_train._loss_and_grads, "_alf_microbatches", None):
        raise RuntimeError("microbatch.install() was already called")
    single = grug_train._loss_and_grads

    def _loss_and_grads(params, batch, mp, z_loss):
        if batch.tokens.shape[0] % n:
            raise ValueError(f"batch of {batch.tokens.shape[0]} rows does not split into {n} microbatches")
        micro = jax.tree.map(lambda x: _split(x, n), batch)
        zeros = jax.tree.map(lambda p: reshard(jnp.zeros(p.shape, jnp.float32), jax.typeof(p).sharding.spec), params)

        def body(acc, mb):
            loss_sum, grad_sum = acc
            (loss, metrics), grads = single(params, mb, mp, z_loss)
            grad_sum = jax.tree.map(lambda a, g: a + g.astype(jnp.float32), grad_sum, grads)
            return (loss_sum + loss.astype(jnp.float32), grad_sum), metrics

        (loss_sum, grad_sum), stacked = jax.lax.scan(body, (jnp.zeros((), jnp.float32), zeros), micro)
        grads = jax.tree.map(lambda g, p: (g / n).astype(p.dtype), grad_sum, params)
        metrics = {}
        for key, value in stacked.items():
            if key in _SUMMED:
                metrics[key] = jnp.sum(value, axis=0)
            elif isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.floating):
                metrics[key] = jnp.mean(value, axis=0)
            else:
                metrics[key] = jax.tree.map(lambda x: x[-1], value)
        return (loss_sum / n, metrics), grads

    _loss_and_grads._alf_microbatches = n
    grug_train._loss_and_grads = _loss_and_grads
