"""The models of the paper, as the arguments that build them.

A checkpoint does not record which model it holds: the parameter tree does, and the tree follows from the arguments
the run was trained with. Every evaluation therefore takes models as ``<run>:<kind>``, and this table turns a kind
into the arguments that rebuild it. The kind must match the run's training arguments, as must the ranks
(``--inject-rank`` / ``--proj-rank``, and ``--src-block`` / ``--src-rank`` for the single-source route).

The two backbones are the standard pre-norm residual stream and the attention residual (every sublayer input is a
learned softmax over the sublayer outputs below it, arXiv:2603.15031). On either one, ALF adds the sublayer feedback
path, and, when the run was trained with it, the input injection path; a run's injection is recorded in its run
directory and picked up automatically, so a run with and a run without it share the kind ``*-alf``.

"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Kind:
    """One model: what to call it, how to build it, and what it is."""

    name: str
    args: tuple[str, ...]
    #: feedback passes apply; a backbone without feedback is always evaluated with one pass
    passes: bool
    what: str


def _k(name, passes, what, *args):
    return Kind(name=name, args=tuple(args), passes=passes, what=what)


# Placeholders filled by model_args: {inject_rank}, {proj_rank}, {src_block}, {src_rank}.
_TABLE = [
    # ── the standard residual backbone ──
    _k("standard", False, "the backbone alone", "--variant", "plain"),
    _k("standard-fbt", True, "FBT: the previous position's top state fused into the token input",
       "--variant", "stdres", "--feedback-share", "fbt"),
    _k("standard-alf", True, "ALF: the sublayer feedback path (and the input injection path, if trained with it)",
       "--variant", "stdres", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "slot",
       "--feedback-proj-rank", "{proj_rank}"),
    _k("standard-alf-nomaps", True, "ablation: the sublayer feedback without the per-output maps",
       "--variant", "stdres", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "none",
       "--feedback-proj-rank", "0"),
    _k("standard-alf-top", True, "ablation: every sublayer reads only the previous position's last block",
       "--variant", "stdres", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "slot",
       "--feedback-depth", "top", "--feedback-proj-rank", "{proj_rank}"),
    _k("standard-alf-gate", True,
       "ablation: sublayer 0's read replaces the token input through FBT's gate product instead of being mixed in",
       "--variant", "stdres", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "slot",
       "--feedback-proj-rank", "{proj_rank}", "--feedback-replaces-input", "input"),
    _k("standard-alf-finalstate", True,
       "ablation: every sublayer reads only the previous position's final state, each through its own map",
       "--variant", "stdres", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "slot",
       "--feedback-proj-rank", "{proj_rank}", "--feedback-source", "top"),
    _k("standard-nearest4", True,
       "the fixed nearest window in place of the depth softmax: an equal-weight average of the four outputs from "
       "each sublayer's own depth upward (--feedback-depth win<k> takes other widths)",
       "--variant", "stdres", "--feedback-share", "pool", "--feedback-query", "uniform", "--feedback-proj", "slot",
       "--feedback-depth", "win4", "--feedback-proj-rank", "{proj_rank}"),
    _k("standard-fbt-source", True,
       "FBT's single-source route: one fixed depth of the previous position, injected into every block up to it",
       "--variant", "src", "--src-block", "{src_block}", "--src-rank", "{src_rank}"),
    # ── the attention residual backbone ──
    _k("attnres", False, "the backbone alone (the feedback parameters exist but stay dormant)",
       "--variant", "attnres", "--feedback-share", "pool", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-fbt", True, "FBT: the previous position's top state fused into the token input",
       "--variant", "feedback", "--feedback-share", "fbt", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf", True, "ALF: the sublayer feedback path (and the input injection path, if trained with it)",
       "--variant", "feedback", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "slot",
       "--feedback-proj-rank", "{proj_rank}", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf-shared", True, "ablation: the mixing point's own query reads the previous position, no maps",
       "--variant", "feedback", "--feedback-share", "pool", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf-own", True, "ablation: a second query for the previous position, no maps",
       "--variant", "feedback", "--feedback-share", "pool", "--feedback-query", "own",
       "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf-source", True, "ablation: one map per source block instead of one per output",
       "--variant", "feedback", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "source",
       "--feedback-proj-rank", "{proj_rank}", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf-joint", True, "ablation: one softmax over both positions, maps per source block",
       "--variant", "feedback", "--feedback-share", "joint", "--feedback-query", "shared", "--feedback-proj", "source",
       "--feedback-proj-rank", "{proj_rank}", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf-joint-maps", True, "ablation: one softmax over both positions, maps per output",
       "--variant", "feedback", "--feedback-share", "joint", "--feedback-query", "shared", "--feedback-proj", "slot",
       "--feedback-proj-rank", "{proj_rank}", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf-top", True, "ablation: every sublayer reads only the previous position's last block",
       "--variant", "feedback", "--feedback-share", "pool", "--feedback-query", "own", "--feedback-proj", "slot",
       "--feedback-depth", "top", "--feedback-proj-rank", "{proj_rank}", "--feedback-inject-rank", "{inject_rank}"),
    _k("attnres-alf-newsource", True,
       "ablation: the previous position enters only as one new attention-residual source per layer",
       "--variant", "feedback", "--feedback-share", "inject", "--feedback-inject-rank", "{inject_rank}"),
]

KINDS: dict[str, Kind] = {k.name: k for k in _TABLE}


def resolve(name: str) -> Kind:
    if name in KINDS:
        return KINDS[name]
    raise SystemExit(f"unknown kind {name!r}; known kinds: " + ", ".join(sorted(KINDS)))


def model_args(name: str, *, inject_rank: int = 64, proj_rank: int = 64, src_block: int | None = None,
               src_rank: int = 64) -> list[str]:
    """The arguments that rebuild the kind, with the ranks filled in."""
    kind = resolve(name)
    if "{src_block}" in kind.args and src_block is None:
        raise SystemExit(f"kind {kind.name} needs --src-block")
    fill = {"inject_rank": inject_rank, "proj_rank": proj_rank, "src_block": src_block, "src_rank": src_rank}
    return [a.format(**fill) if "{" in a else a for a in kind.args]


def parse_models(spec: str) -> list[tuple[str, str]]:
    """``run:kind[,run:kind...]`` -> [(run, kind), ...], with the kinds checked."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item.count(":") != 1:
            raise SystemExit(f"expected <run>:<kind>, got {item!r}")
        run, kind = item.split(":")
        out.append((run, resolve(kind).name))
    if not out:
        raise SystemExit("no models given")
    return out


def table() -> str:
    """The kinds as a table, for `--list-kinds`."""
    width = max(len(k) for k in KINDS)
    lines = [f"{'kind'.ljust(width)}  passes  model"]
    for k in _TABLE:
        lines.append(f"{k.name.ljust(width)}  {'yes' if k.passes else 'one':6}  {k.what}")
    return "\n".join(lines)
