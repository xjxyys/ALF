# Copied from kschwethelm/looped-lm-scaling at ad7d8f3055501e4c84ced7797f772dce2f8e2107
# (dev/eval_bundles/owned/prepare_owned_bundle.py), MIT License; see third_party/looped-lm-scaling.LICENSE.
#
# Trimmed to the induction-head generator. Removed: the YAML config, the CLI and the local-storage helpers
# (`get_bundle_dir`, `_task`, `ALL_TASKS`, `write_yaml`, `main`). `prepare_induction_head` returns its records instead
# of writing a JSONL file; the generator itself, and so the RNG stream and every example it draws, is unchanged.
"""Prepare the owned-taxonomy eval bundle.

Tasks:
- Induction-head (compositional_symbolic) — minimal in-context key/value
  lookup task, inspired by Olsson et al. 2022's induction-head circuit.
"""

import random
import string


def _random_token(rng: random.Random, length: int = 4) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def generate_induction_head_example(rng: random.Random, n_pairs: int = 8) -> dict:
    keys = [_random_token(rng) for _ in range(n_pairs)]
    vals = [_random_token(rng, length=3) for _ in range(n_pairs)]
    lines = [f"{k} -> {v}" for k, v in zip(keys, vals)]
    query_idx = rng.randrange(n_pairs)
    query_key = keys[query_idx]
    gold_val = vals[query_idx]
    context = "\n".join(lines) + f"\n{query_key} ->"
    # No leading space in continuation — the suite's continuation_delimiter " " adds it.
    return {"context": context, "continuation": gold_val,
            "choices": vals, "gold": query_idx}


def prepare_induction_head(n_examples: int = 1000, seed: int = 1337) -> list[dict]:
    rng = random.Random(seed)
    return [generate_induction_head_example(rng, n_pairs=8) for _ in range(n_examples)]
