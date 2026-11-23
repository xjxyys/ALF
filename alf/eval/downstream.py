#!/usr/bin/env python3
"""The downstream bench on a Marin MoE checkpoint (JAX), scored question by question at every Jacobi pass.

The suite is the 20 tasks of Schwethelm et al. (arXiv:2604.21106, v3) plus 6 further tasks of the CORE bundle; the
bundle itself is built by `alf.bench.prepare`, which pins every source. Sampling, few-shot prompting and judging
follow that paper's scorer: each task is shuffled with seed 1337 and cut to the first 10,000 questions, the few-shot
examples of question i are drawn with `Random(1234 + i)` from the other questions of that pool, the prompts are
rendered and tokenized by the vendored CORE functions (`alf.vendor.core_eval`), and the continuation delimiter comes
from the bundle's manifest. The few-shot draw and the two judging rules below are nanochat's `evaluate_example`
(MIT, see NOTICE); the forward pass and the packing are this repository's. Each question records the summed NLL of the right answer and its token count, and
whether it was right (language modeling: every answer token is the teacher-forced argmax; multiple choice: the
choice with the lowest mean loss); a task's loss is the summed NLL over the summed token count. The model is loaded
and run with the Marin tokenizer, BOS prepended, a model with feedback scored at passes 1..P, a
backbone without feedback at pass 1 only. A question longer than the model's context is an error, not a truncation,
as in the paper's scorer. At evaluation the single-device MoE takes the dropless local path, where rows never
interact, so questions are sorted by length and packed 8 rows to a forward over five fixed lengths
(256 / 512 / 1024 / 2048 / 4096).

`--decode` chooses how a model with feedback reads during scoring: `fused` (the default, the paper's protocol) has
every position read the previous position in every pass, so pass 1 is the standard backbone; `soft` runs each row's
prompt (its few-shot examples and the question) through one pass with no feedback read -- an ordinary prefill, the
SOFT mode of FBT (arXiv:2608.08888) -- and reads feedback only in the continuation; `soft1` is `soft` with the
prompt's last token decoded rather than prefilled, so the answer's first token also reads feedback; `fused2` runs
the prompt through two passes. `--tasks` scores only the named tasks (the merge then reports those tasks and their
macro, not the paper's five axes).

  shard: python -m alf.eval.downstream --run-dir RUN [--step N] --variant plain|attnres|feedback|src|stdres
             [the kind's feedback / src arguments] --passes P --bundle B --out DIR --shard I
             --num-shards N [--max-per-task K] [--decode fused|soft|soft1|fused2] [--tasks a,b,...]
  merge: python -m alf.eval.downstream --merge --run-dir RUN [--step N] --bundle B --out DIR --num-shards N

Every argument of the kind table (alf.kinds) is accepted. The injection ablations of the standard
residual (--feedback-replaces-input input, --feedback-source top) hold no prompt context -- their passes run through
alf.model.StdResTransformer._run_ablation -- so they are scored with --decode fused only.

A shard writes one line per question to <out>/shard_<i>/<task>.jsonl (`i` the index in the pool, `src` the index in
the task, `t` the answer's token count, `nll` the summed NLL per pass, `ok` right or wrong per pass, and for
multiple choice `cand`, each choice's mean loss per pass). Each task is renamed into place when it is finished, so a
restart skips the tasks already done; a shard that finishes every task writes <out>/shard_<i>.json. A multiple-choice
question whose choices tokenize so that one is a prefix of another (nothing left to score once the common prefix is
removed, where the paper's scorer raises) is recorded as skipped and not scored; this depends only on the tokenizer,
so it is the same for every model. The merge writes <out>/bench.json (per task and pass, loss and accuracy; the
paper's five axes and its aggregate; the 26-task macro loss) and done.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from alf.bench.bundle import read_bundle

# The bundle written by alf.bench.prepare (manifest.json, suite.json, records/), the default of --bundle.
BUNDLE = os.environ.get("BENCH_BUNDLE", "")
BATCH_ROWS = 8
LENGTHS = (256, 512, 1024, 2048, 4096)
DECODES = ("fused", "soft", "soft1", "fused2")
# (passes of the prompt, prompt tokens left to decoding): soft1 decodes the prompt's last token instead of prefilling it
DECODE_PROMPT = {"soft": (1, 0), "soft1": (1, 1), "fused2": (2, 0)}
MAX_PER_TASK = 10000
SHUFFLE_SEED, FEWSHOT_SEED = 1337, 1234



class MarinBatchTokenizer:
    """The nanochat interface core_eval needs: tokenizer(texts, prepend=bos) -> id lists, get_bos_token_id()."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_bos_token_id(self) -> int:
        return int(self.tokenizer.bos_token_id)

    def __call__(self, texts, prepend=None):
        head = [] if prepend is None else [int(prepend)]
        return [head + list(self.tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def complete_steps(run_dir: str) -> list[int]:
    root = Path(run_dir) / "checkpoints"
    return sorted(int(p.name.split("-", 1)[1]) for p in root.glob("step-*") if (p / "metadata.json").exists())


def resolve_step(args) -> int:
    steps = complete_steps(args.run_dir)
    if args.step is not None:
        if args.step not in steps:
            raise SystemExit(f"no complete checkpoint step-{args.step} under {args.run_dir}/checkpoints (have {steps})")
        return args.step
    if not steps:
        raise SystemExit(f"no complete checkpoint under {args.run_dir}/checkpoints")
    return steps[-1]

def check_backbone_args(args) -> None:
    """Arguments of the kind table that only some backbones carry: refused rather than silently dropped."""
    if args.variant not in ("attnres", "feedback") and (args.feedback_rank or args.feedback_inject_rank):
        raise SystemExit(f"--feedback-rank / --feedback-inject-rank are parameters of the attention residual backbone "
                         f"(--variant attnres / feedback), which --variant {args.variant} does not build")
    if args.variant != "src" and (args.src_block != -1 or args.src_rank != 64):
        raise SystemExit(f"--src-block / --src-rank are parameters of FBT's single-source route (--variant src), which "
                         f"--variant {args.variant} does not build")
    ablation = (args.feedback_replaces_input, args.feedback_source) != ("residual", "outputs")
    if ablation and args.variant != "stdres":
        raise SystemExit("--feedback-replaces-input input / --feedback-source top are injection ablations of the sublayer "
                         "feedback path on the standard residual: they need --variant stdres (kinds standard-alf-gate / "
                         "standard-alf-finalstate)")
    if ablation and args.decode != "fused":
        raise SystemExit(f"--decode {args.decode} needs a prompt context, which --feedback-replaces-input input / "
                         f"--feedback-source top hold no state for (alf.model.StdResTransformer._run_ablation): score "
                         f"these models with --decode fused")


def task_filter(spec: str, suite: dict):
    """The set of task names in `spec` (comma-separated; empty = None, the whole suite); unknown names are an error."""
    if not spec.strip():
        return None
    wanted = {t.strip() for t in spec.split(",") if t.strip()}
    unknown = wanted - {t["name"] for t in suite["tasks"]}
    if unknown:
        raise SystemExit(f"unknown tasks {sorted(unknown)}")
    return wanted


def load_bundle(bundle: str):
    """The bundle's suite, its manifest and its records by task, verified against the manifest (see
    `alf.bench.bundle.read_bundle` for what is checked)."""
    if not bundle:
        raise SystemExit("set --bundle (or BENCH_BUNDLE) to a bundle written by alf.bench.prepare")
    read = read_bundle(bundle)
    return read.suite, read.manifest, read.records


def task_pool(records: list, limit: int) -> list:
    indexed = list(enumerate(records))
    random.Random(SHUFFLE_SEED).shuffle(indexed)
    return indexed[:min(MAX_PER_TASK, limit) if limit > 0 else MAX_PER_TASK]


def render_raw_lm(item, continuation_delimiter, fewshot_examples=None):
    """Suite tasks with "render": "raw" (0-shot language modeling only): context and continuation joined verbatim, with no
    trim / strip, so a code line keeps the prefix's trailing newline and its own indentation. The tokens of the context must
    still be a prefix of the tokens of context + continuation (batch_sequences_lm asserts it)."""
    head = item["context"] + continuation_delimiter
    return [head, head + item["continuation"]]


def build_example(tokenizer, indexed: list, index: int, task: dict, delimiter: str, max_len: int):
    """tokens / starts / ends / gold for pool item `index`, built as the paper's scorer builds an example (tasks
    without a "render" field; "render": "raw" swaps in render_raw_lm)."""
    from alf.vendor import core_eval  # imported here, not at the top, so the merge needs no tokenizer stack

    kind, shots = task["task_type"], task["fewshot"]
    if len(indexed) <= shots:
        raise ValueError(f"{task['name']} needs at least {shots + 1} examples for {shots}-shot")
    candidates = [i for i in range(len(indexed)) if i != index]
    chosen = random.Random(FEWSHOT_SEED + index).sample(candidates, shots)
    render, tokenize = {"language_modeling": (core_eval.render_prompts_lm, core_eval.batch_sequences_lm),
                        "multiple_choice": (core_eval.render_prompts_mc, core_eval.batch_sequences_mc)}[kind]
    if task.get("render", "template") != "template":
        if task["render"] != "raw" or kind != "language_modeling" or shots:
            raise ValueError(f"{task['name']}: render={task['render']!r} supports only 0-shot language_modeling as 'raw'")
        render = render_raw_lm
    item = indexed[index][1]
    tokens, starts, ends = tokenize(tokenizer, render(item, delimiter, [indexed[i][1] for i in chosen]))
    for row, seq in enumerate(tokens):
        if len(seq) > max_len:
            raise ValueError(f"{task['name']} pool item {index} row {row}: {len(seq)} tokens exceeds {max_len} "
                             "(no truncation, as in the paper's scorer)")
        if kind == "multiple_choice" and starts[row] == ends[row] == len(seq):
            return None  # one choice tokenizes to a prefix of another: nothing left to score after the common prefix
        if not 1 <= starts[row] < ends[row] <= len(seq):
            raise ValueError(f"{task['name']} pool item {index} row {row}: empty or invalid continuation span")
    return tokens, starts, ends, (item["gold"] if kind == "multiple_choice" else 0)


def score_task(task: dict, indexed: list, delimiter: str, *, tokenizer, bos: int, max_len: int, passes: int,
               score_fn, shard: int, num_shards: int, selfcheck: bool = False, selfcheck_tol: float = 0.02):
    """Score this shard's pool items of one task. score_fn(int32 [BATCH_ROWS, L], int32 prompt lengths [BATCH_ROWS]) ->
    (float64 losses [P, B, L], int preds [P, B, L]), position t scoring token t + 1; a row's prompt is its tokens before
    the continuation (0 for padding rows). Returns ({pool index: record}, max |packed - single| NLL when selfcheck)."""
    name = task["name"]
    selfcheck_max = 0.0
    recs, pending = {}, []
    for index in range(shard, len(indexed), num_shards):
        built = build_example(tokenizer, indexed, index, task, delimiter, max_len)
        if built is None:
            recs[index] = {"i": index, "src": indexed[index][0], "skip": "choice_is_prefix"}
        elif len(built[0]) > BATCH_ROWS:
            raise SystemExit(f"{name} pool item {index}: {len(built[0])} rows exceed BATCH_ROWS={BATCH_ROWS}")
        else:
            pending.append((index, *built))
    # Rows never interact at eval (single-device MoE takes the dropless local path), so examples are packed
    # BATCH_ROWS rows at a time, sorted by length so a batch shares the smallest padding bucket.
    pending.sort(key=lambda e: max(len(x) for x in e[1]))
    batches, cur, used = [], [], 0
    for ex in pending:
        if cur and used + len(ex[1]) > BATCH_ROWS:
            batches.append(cur)
            cur, used = [], 0
        cur.append(ex)
        used += len(ex[1])
    if cur:
        batches.append(cur)
    for batch in batches:
        rows = [seq for ex in batch for seq in ex[1]]
        length = next((L for L in LENGTHS if L >= max(len(x) for x in rows)), None)
        if length is None:
            raise SystemExit(f"{name}: a row longer than {LENGTHS[-1]} tokens")
        padded = np.full((BATCH_ROWS, length), bos, np.int32)
        for r, seq in enumerate(rows):
            padded[r, :len(seq)] = seq
        prompt = np.zeros(BATCH_ROWS, np.int32)
        prompt[:len(rows)] = [s for ex in batch for s in ex[2]]
        losses, preds = score_fn(padded, prompt)
        if selfcheck and batch is batches[0]:
            r0 = 0
            for _, tokens, starts, ends, _ in batch:
                alone = np.full((BATCH_ROWS, length), bos, np.int32)
                for r, seq in enumerate(tokens):
                    alone[r, :len(seq)] = seq
                alone_prompt = np.zeros(BATCH_ROWS, np.int32)
                alone_prompt[:len(tokens)] = starts
                la = score_fn(alone, alone_prompt)[0]
                for r in range(len(tokens)):
                    s, e = starts[r], ends[r]
                    diff = float(np.max(np.abs(la[:, r, s - 1:e - 1].sum(-1) - losses[:, r0 + r, s - 1:e - 1].sum(-1))))
                    selfcheck_max = max(selfcheck_max, diff)
                r0 += len(tokens)
            if selfcheck_max > selfcheck_tol:
                raise SystemExit(f"{name}: packed vs single-example NLL differs by {selfcheck_max:.4g} nats "
                                 f"(> {selfcheck_tol}); rows are not independent")
        r0 = 0
        for index, tokens, starts, ends, gold in batch:
            rec = {"i": index, "src": indexed[index][0], "t": ends[gold] - starts[gold], "nll": [], "ok": []}
            if task["task_type"] == "multiple_choice":
                rec["cand"] = []
            for k in range(passes):
                sums, hits = [], []
                for r in range(len(tokens)):
                    s, e = starts[r], ends[r]
                    sums.append(float(losses[k, r0 + r, s - 1:e - 1].sum()))
                    hits.append(bool(np.array_equal(preds[k, r0 + r, s - 1:e - 1], np.asarray(tokens[r][s:e]))))
                if not all(math.isfinite(x) for x in sums):
                    raise ValueError(f"{name} pool item {index} pass {k + 1}: non-finite continuation loss")
                if task["task_type"] == "multiple_choice":
                    means = [sm / (ends[r] - starts[r]) for r, sm in enumerate(sums)]
                    rec["ok"].append(means.index(min(means)) == gold)
                    rec["cand"].append([round(x, 6) for x in means])
                else:
                    rec["ok"].append(hits[0])
                rec["nll"].append(round(sums[gold], 6))
            recs[index] = rec
            r0 += len(tokens)
    return recs, selfcheck_max


def run_shard(args) -> None:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jmp
    from jax.sharding import AxisType, Mesh, set_mesh

    from alf import model as am
    import experiments.grug.moe_hero_ep.model as gm
    from alf import rungs as mid_rung
    from experiments.grug.moe_hero_ep.hero_recipe import HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE
    from experiments.grug.moe_hero_ep.train import MasterParamMode
    from alf.marin.compat import register_gpu_kinds_for_pallas
    from levanter.checkpoint import load_checkpoint
    from levanter.tokenizers import load_tokenizer
    from alf.marin.recipe import model_config

    mid_rung.register_mid_shapes()
    register_gpu_kinds_for_pallas()
    suite, manifest, records = load_bundle(args.bundle)
    step = resolve_step(args)
    tokenizer = MarinBatchTokenizer(load_tokenizer(os.environ.get("MARIN_TOKENIZER_DIR", "marin-community/marin-tokenizer")))
    bos = tokenizer.get_bos_token_id()
    passes = args.passes if args.variant in ("feedback", "src", "stdres") else 1
    cfg = model_config(shape=args.shape)
    max_len = min(int(cfg.max_seq_len), LENGTHS[-1])
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1), ("replica_dcn", "data", "context", "expert", "model"),
                axis_types=(AxisType.Explicit,) * 5)
    mp = jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[MasterParamMode.DEVICE])
    out = Path(args.out)
    sdir = out / f"shard_{args.shard}"
    sdir.mkdir(parents=True, exist_ok=True)
    fbt_input_norm = am.run_fbt_input_norm(args.run_dir, "fbt" if args.feedback_replaces_input == "input" else
                                           args.feedback_share if args.variant in ("feedback", "stdres") else "none")
    t0 = time.time()
    with set_mesh(mesh):
        key = jax.random.key(0)
        base = gm.Transformer.init(cfg, key=key)
        if args.variant == "plain":
            model = base
        elif args.variant == "src":
            model = am.SrcTransformer.from_base(base, key=key, src_block=args.src_block, src_rank=args.src_rank,
                                                passes=max(passes, 1), eval_passes=passes, jitter=0.0, bos_id=bos)
        elif args.variant == "stdres":
            model = am.StdResTransformer.from_base(base, key=key, share=args.feedback_share, query=args.feedback_query,
                                                   proj=args.feedback_proj, proj_rank=args.feedback_proj_rank,
                                                   passes=max(passes, 1), eval_passes=passes, jitter=0.0, bos_id=bos,
                                                   fbt_input_norm=fbt_input_norm, depth=args.feedback_depth,
                                                   inject=args.feedback_replaces_input, source=args.feedback_source,
                                                   embed_inject=am.read_embed_inject(args.run_dir))
        else:
            model = am.AttnResTransformer.from_base(
                base, key=key, share=args.feedback_share, query=args.feedback_query, rank=args.feedback_rank,
                passes=max(passes, 1), eval_passes=passes, jitter=0.0, bos_id=bos, inject_rank=args.feedback_inject_rank,
                proj=args.feedback_proj, proj_rank=args.feedback_proj_rank, depth=args.feedback_depth,
                fbt_input_norm=fbt_input_norm, embed_inject=am.read_embed_inject(args.run_dir))
        ckpt = f"{args.run_dir}/checkpoints/step-{step}"
        model = load_checkpoint({"params": model}, ckpt, axis_mapping=None, mesh=mesh, allow_partial=False)["params"]
        model = mp.cast_to_compute(model)
        print(f"[bench] shard {args.shard}/{args.num_shards}: loaded {ckpt} variant={args.variant} passes={passes} "
              f"query={args.feedback_query} depth={args.feedback_depth} "
              f"fbt_input_norm={fbt_input_norm} max_len={max_len} ({time.time() - t0:.0f}s)", flush=True)

        @eqx.filter_jit
        def run_hiddens(m, tokens, prompt):
            mask = gm.AttentionMask.causal()
            if isinstance(m, am.AttnResTransformer) and args.decode != "fused":
                c, shift = DECODE_PROMPT[args.decode]
                context = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :] < (prompt - shift)[:, None]
                hiddens, _ = m.run_passes(tokens, mask, passes, train=False, context=context, context_passes=c)
            elif isinstance(m, am.AttnResTransformer):
                hiddens, _ = m.run_passes(tokens, mask, passes, train=False)
            else:
                hiddens = [m(tokens, mask=mask)[0]]
            return hiddens

        @eqx.filter_jit
        def score_hidden(output_proj, h, labels):
            logits = jnp.einsum("bsh,hd->bsd", h, output_proj.astype(h.dtype), preferred_element_type=jnp.float32,
                                out_sharding=gm._activation_spec(h))
            target = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
            return jax.nn.logsumexp(logits, axis=-1) - target, jnp.argmax(logits, axis=-1).astype(jnp.int32)

        def score_np(padded, prompt):
            tok_j = jnp.asarray(padded)
            labels = jnp.roll(tok_j, -1, axis=1)
            hiddens = run_hiddens(model, tok_j, jnp.asarray(prompt))
            losses, preds = [], []
            for h in hiddens:
                l, p = score_hidden(model.output_proj, h, labels)  # noqa: E741
                losses.append(np.asarray(l, np.float64))
                preds.append(np.asarray(p))
            return np.stack(losses), np.stack(preds)

        done_tasks = []
        selfcheck_max = 0.0
        wanted = task_filter(args.tasks, suite)
        for task in suite["tasks"]:
            name = task["name"]
            if wanted is not None and name not in wanted:
                continue
            final = sdir / f"{name}.jsonl"
            if final.exists():
                done_tasks.append(name)
                continue
            delimiter = manifest["tasks"][name]["continuation_delimiter"]
            indexed = task_pool(records[name], args.max_per_task)
            recs, check = score_task(task, indexed, delimiter, tokenizer=tokenizer, bos=bos,
                                     max_len=max_len, passes=passes, score_fn=score_np, shard=args.shard,
                                     num_shards=args.num_shards, selfcheck=args.selfcheck, selfcheck_tol=args.selfcheck_tol)
            selfcheck_max = max(selfcheck_max, check)
            tmp = sdir / f"{name}.jsonl.tmp"
            with tmp.open("w") as stream:
                for index in sorted(recs):
                    stream.write(json.dumps(recs[index]) + "\n")
            scored = [r for r in recs.values() if "skip" not in r]
            n, tok = len(scored), sum(r["t"] for r in scored)
            nll_last = math.fsum(r["nll"][-1] for r in scored)
            correct_last = sum(int(r["ok"][-1]) for r in scored)
            os.replace(tmp, final)
            done_tasks.append(name)
            print(f"[bench] shard {args.shard} task={name} n={n} loss(last pass)={nll_last / max(tok, 1):.4f} "
                  f"acc(last pass)={correct_last / max(n, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
    report = {"run_dir": args.run_dir, "step": step, "variant": args.variant, "passes": passes, "shard": args.shard,
              "num_shards": args.num_shards, "max_per_task": args.max_per_task, "bundle": args.bundle,
              "decode": args.decode, "task_filter": sorted(wanted) if wanted is not None else None,
              "tasks": done_tasks, "selfcheck_max_abs_dnll": selfcheck_max if args.selfcheck else None,
              "seconds": time.time() - t0}
    tmp = out / f"shard_{args.shard}.json.tmp"
    json.dump(report, open(tmp, "w"), indent=1)
    os.replace(tmp, out / f"shard_{args.shard}.json")


def merge(args) -> None:

    suite = json.loads((Path(args.bundle) / "suite.json").read_text())
    step = resolve_step(args)
    out = Path(args.out)
    shards = [json.load(open(out / f"shard_{i}.json")) for i in range(args.num_shards)]
    passes = {s["passes"] for s in shards}
    steps = {s["step"] for s in shards}
    decodes = {s.get("decode", "fused") for s in shards}
    filters = {tuple(s["task_filter"]) if s.get("task_filter") else None for s in shards}
    if len(passes) != 1 or steps != {step} or len(decodes) != 1 or len(filters) != 1:
        raise SystemExit(f"shards disagree: passes {passes}, steps {steps} (expected step {step}), decode {decodes}, "
                         f"task filters {filters}")
    p = passes.pop()
    wanted = filters.pop()
    tasks = {}
    for task in suite["tasks"]:
        name = task["name"]
        if wanted is not None and name not in wanted:
            continue
        rows = []
        for i in range(args.num_shards):
            with open(out / f"shard_{i}" / f"{name}.jsonl") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
        idx = sorted(r["i"] for r in rows)
        if idx != list(range(len(idx))):
            raise SystemExit(f"{name}: pool indices are not a complete 0..n-1 cover across shards")
        skipped = sum("skip" in r for r in rows)
        rows = [r for r in rows if "skip" not in r]
        tok = sum(r["t"] for r in rows)
        tasks[name] = {"axis": task["axis"], "task_type": task["task_type"], "fewshot": task["fewshot"], "n": len(rows),
                       "skipped": skipped,
                       "tokens": tok, "loss": [math.fsum(r["nll"][k] for r in rows) / tok for k in range(p)],
                       "accuracy": [sum(r["ok"][k] for r in rows) / len(rows) for k in range(p)]}
    paper = [t for t in suite["tasks"] if t["axis"] != "core_extra"]
    paper2604 = None
    if all(t["name"] in tasks for t in paper):
        axes = {}
        for axis in dict.fromkeys(t["axis"] for t in paper):
            names = [t["name"] for t in paper if t["axis"] == axis]
            axes[axis] = {"tasks": names,
                          "loss": [math.fsum(tasks[n]["loss"][k] for n in names) / len(names) for k in range(p)],
                          "accuracy": [math.fsum(tasks[n]["accuracy"][k] for n in names) / len(names) for k in range(p)]}
        paper2604 = {"axes": axes,
                     "task_macro_loss": [math.fsum(tasks[t["name"]]["loss"][k] for t in paper) / len(paper) for k in range(p)],
                     "axis_macro_loss": [math.fsum(a["loss"][k] for a in axes.values()) / len(axes) for k in range(p)],
                     "owned_metric": [math.fsum(a["accuracy"][k] for a in axes.values()) / len(axes) for k in range(p)]}
    report = {"run_dir": args.run_dir, "step": step, "variant": shards[0]["variant"], "passes": p, "bundle": args.bundle,
              "decode": next(iter(decodes)), "task_filter": list(wanted) if wanted is not None else None,
              "protocol": suite["protocol"], "max_per_task": shards[0]["max_per_task"], "tasks": tasks,
              "paper2604": paper2604,
              "all_task_macro_loss": [math.fsum(v["loss"][k] for v in tasks.values()) / len(tasks) for k in range(p)],
              "all_task_macro_accuracy": [math.fsum(v["accuracy"][k] for v in tasks.values()) / len(tasks) for k in range(p)],
              "seconds": max(s["seconds"] for s in shards)}
    owned = paper2604["owned_metric"] if paper2604 else None
    json.dump(report, open(out / "bench.json", "w"), indent=1)
    json.dump({"all_task_macro_loss": report["all_task_macro_loss"], "owned_metric": owned, "decode": report["decode"],
               "seconds": report["seconds"]}, open(out / "done.json", "w"))
    print(f"[{report['decode']}] task-macro loss by pass: " + " / ".join(f"{x:.4f}" for x in report["all_task_macro_loss"])
          + ("" if owned is None else " | 2604 owned_metric by pass: " + " / ".join(f"{x:.4f}" for x in owned)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, default=None, help="default: newest complete checkpoint")
    ap.add_argument("--variant", choices=("plain", "attnres", "feedback", "src", "stdres"), default="attnres")
    ap.add_argument("--feedback-share", choices=("pool", "joint", "inject", "fbt"), default="pool")
    ap.add_argument("--src-block", type=int, default=-1)
    ap.add_argument("--src-rank", type=int, default=64)
    ap.add_argument("--feedback-query", choices=("shared", "own", "uniform"), default="shared")
    ap.add_argument("--feedback-rank", type=int, default=0)
    ap.add_argument("--feedback-inject-rank", type=int, default=0)
    ap.add_argument("--feedback-proj", choices=("none", "source", "slot"), default="none")
    ap.add_argument("--feedback-proj-rank", type=int, default=0)
    ap.add_argument("--feedback-depth", default="whole",
                    help="whole: deeper slots; top: last layer; win<k>: nearest k slots, equal weights, no softmax "
                         "(stdres pool/uniform). Must match how the run was trained -- win<k> and whole share the "
                         "same parameter tree, so a mismatch is silent")
    ap.add_argument("--feedback-replaces-input", choices=("residual", "input"), default="residual",
                    help="variant stdres ablation: input = sublayer 0's read replaces the token input through FBT's gate")
    ap.add_argument("--feedback-source", choices=("outputs", "top"), default="outputs",
                    help="variant stdres ablation: top = every sublayer reads the previous position's final state")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--shape", default="d768")
    ap.add_argument("--max-per-task", type=int, default=-1, help="-1 = the paper's 10,000 cap; K > 0 shrinks the pool (debug)")
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selfcheck", action="store_true", help="re-score each task's first packed batch one example at a time")
    ap.add_argument("--selfcheck-tol", type=float, default=0.02, help="max |packed - single| NLL per example (nats)")
    ap.add_argument("--decode", choices=DECODES, default="fused",
                    help="fused: every position reads in every pass (pass 1 = standard); soft: the prompt (each row's "
                         "tokens before its continuation) through one pass; soft1: soft with the prompt's last token "
                         "decoded; fused2: the prompt through two passes")
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: the whole suite)")
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("need 0 <= shard < num_shards")
    if args.decode != "fused" and (args.variant not in ("feedback", "src", "stdres")
                                   or args.passes < (3 if args.decode == "fused2" else 2)):
        raise SystemExit(f"--decode {args.decode} needs a feedback variant and more passes than the prompt gets "
                         f"(variant {args.variant}, passes {args.passes})")
    check_backbone_args(args)
    if args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
