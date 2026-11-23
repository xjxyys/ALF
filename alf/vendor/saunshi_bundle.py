# Copied from kschwethelm/looped-lm-scaling at ad7d8f3055501e4c84ced7797f772dce2f8e2107
# (dev/eval_bundles/saunshi/prepare_saunshi_bundle.py), MIT License; see third_party/looped-lm-scaling.LICENSE.
#
# Trimmed to the record builders the bundle needs. Removed: the YAML config, the CLI, the local-storage helpers
# (`get_bundle_dir`, `_write_jsonl`, `write_yaml`, `main`) and `prepare_tydiqa_nocontext`, which this suite does not
# use. Each builder now takes `load_dataset` from its caller (the original imported it inside the function body) and
# returns its records instead of writing a JSONL file; `prepare_reasoning_primitives` returns the four synthetic task
# lists in its original generation order, so the one shared RNG stream is unchanged. The record-building logic -- the
# prompt scaffolding, the field selection, the filtering and the number formatting -- is byte-for-byte the original's,
# because the task content depends on it.
"""
Prepare the Saunshi downstream eval bundle.

Based on the benchmark from Saunshi et al. (2502.17416), §5.1.
15 of 16 tasks implemented — QuAC is excluded due to a broken legacy
HuggingFace datasets loader (allenai/quac requires parquet workaround
and has inconsistent schema across versions).

JSONL format follows the CORE eval bundle convention:
- context includes the answer delimiter with trailing space (e.g. "...\\nAnswer: ")
- continuation has no leading whitespace (e.g. "Denver Broncos")
- continuation_delimiter in the suite is " " (default), except for reasoning primitives
"""

import random
import re

# ==============================================================================
# Math word problems
# Format: "Question: {q}\nAnswer: " + "{numeric_answer}"
# Matches CORE TriviaQA-style prompt format.
# ==============================================================================


def prepare_svamp(load_dataset) -> list[dict]:
    ds = load_dataset("ChilleD/SVAMP", split="test")
    records = []
    for item in ds:
        question = item["Body"].strip().rstrip(".") + ". " + item["Question"].strip()
        fval = float(item["Answer"])
        answer = str(int(fval)) if fval == int(fval) else str(fval)
        records.append({"context": f"Question: {question}\nAnswer: ", "continuation": answer})
    return records


def prepare_asdiv(load_dataset) -> list[dict]:
    ds = load_dataset("EleutherAI/asdiv", split="validation")
    records = []
    for item in ds:
        question = item["body"].strip().rstrip(".") + " " + item["question"].strip()
        raw_answer = item["answer"].strip()
        # Strip units in parentheses e.g. "9 (apples)" → "9"
        answer = re.sub(r"\s*\(.*?\)", "", raw_answer).strip()
        records.append({"context": f"Question: {question}\nAnswer: ", "continuation": answer})
    return records


def prepare_mawps(load_dataset) -> list[dict]:
    ds = load_dataset("mwpt5/MAWPS", split="train")
    records = []
    skipped = 0
    for item in ds:
        question_raw = item["Question"].strip()
        numbers_str = item.get("Numbers", "") or ""
        answer_raw = str(item["Answer"]).strip()

        # Substitute N_00, N_01, ... placeholders with actual numbers (space-separated).
        def fmt_num(s: str) -> str:
            try:
                f = float(s)
                return str(int(f)) if f == int(f) else s
            except ValueError:
                return s

        if numbers_str.strip():
            numbers = [fmt_num(n) for n in numbers_str.split()]
            question = question_raw
            for i, num in enumerate(numbers):
                placeholder = f"N_{i:02d}"
                question = question.replace(placeholder, num)
        else:
            question = question_raw

        if re.search(r"N_\d{2}", question):
            skipped += 1
            continue

        try:
            fval = float(answer_raw)
            answer = str(int(fval)) if fval == int(fval) else answer_raw
        except ValueError:
            answer = answer_raw
        records.append({"context": f"Question: {question}\nAnswer: ", "continuation": answer})
    return records


# ==============================================================================
# Closed book QA
# Format: "Question: {q}\nAnswer: " + "{answer}"
# Matches CORE TriviaQA format exactly.
# ==============================================================================


def prepare_triviaqa(load_dataset) -> list[dict]:
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    records = []
    for item in ds:
        answer = item["answer"]["value"].strip()
        if not answer:
            continue
        records.append({"context": f"Question: {item['question'].strip()}\nAnswer: ", "continuation": answer})
    return records


def prepare_nq(load_dataset) -> list[dict]:
    ds = load_dataset("nq_open", split="validation")
    records = []
    for item in ds:
        answers = item["answer"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}\nAnswer: ", "continuation": answers[0].strip()})
    return records


def prepare_webq(load_dataset) -> list[dict]:
    ds = load_dataset("web_questions", split="test")
    records = []
    for item in ds:
        answers = item["answers"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}\nAnswer: ", "continuation": answers[0].strip()})
    return records


# ==============================================================================
# Open book QA
# Format: "Context: {passage}\nQuestion: {q}\nAnswer: " + "{answer}"
# Matches CORE SQuAD format (uses "Context:" prefix, not "Passage:").
# CoQA uses CORE's multi-turn format with "Story:" prefix and QA history.
# ==============================================================================


def prepare_tydiqa_goldp(load_dataset) -> list[dict]:
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    records = []
    for item in ds:
        if item["id"].split("-")[0] != "english":
            continue
        answers = item["answers"]["text"]
        if not answers:
            continue
        records.append({
            "context": f"Context: {item['context'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: ",
            "continuation": answers[0].strip(),
        })
    return records


def prepare_squadv2(load_dataset) -> list[dict]:
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    records = []
    for item in ds:
        answers = item["answers"]["text"]
        if not answers:  # unanswerable
            continue
        records.append({
            "context": f"Context: {item['context'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: ",
            "continuation": answers[0].strip(),
        })
    return records


def prepare_drop(load_dataset) -> list[dict]:
    ds = load_dataset("ucinlp/drop", split="validation")
    records = []
    for item in ds:
        spans = item["answers_spans"]["spans"]
        if not spans:
            continue
        records.append({
            "context": f"Context: {item['passage'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: ",
            "continuation": spans[0].strip(),
        })
    return records


def prepare_coqa(load_dataset) -> list[dict]:
    """Prepare CoQA with multi-turn context, matching the CORE eval bundle format.

    CORE format:
        Below is a story followed by a series of related questions. ...
        Story: {passage}
        Preceding questions:
        Question: {q1}
        Answer: {a1}
        ...
        Final question:
        Question: {current_q}
        Answer:
    """
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    preamble = ("Below is a story followed by a series of related questions. "
                "Please answer the final question by referring to the story "
                "and the previous questions.")
    records = []
    for item in ds:
        passage = item["story"].strip()
        questions = item["questions"]
        answers = item["answers"]["input_text"]
        for turn_idx, (q, a) in enumerate(zip(questions, answers)):
            if not a.strip():
                continue
            # Build context with QA history (matching CORE format)
            parts = [f"{preamble}\nStory: {passage}"]
            if turn_idx > 0:
                parts.append("Preceding questions:")
                for prev_q, prev_a in zip(questions[:turn_idx], answers[:turn_idx]):
                    parts.append(f"Question: {prev_q.strip()}\nAnswer: {prev_a.strip()}")
            parts.append(f"\nFinal question:\nQuestion: {q.strip()}\nAnswer: ")
            records.append({
                "context": "\n".join(parts),
                "continuation": a.strip(),
            })
    return records


# NOTE: QuAC (allenai/quac) is excluded from the benchmark. The HuggingFace
# datasets loader is broken for this dataset (requires parquet workaround with
# inconsistent schema across versions). This is the only task from Saunshi et al.
# that we drop. Open-book QA group has 4 tasks instead of the paper's 5.


# ==============================================================================
# Reasoning primitives — synthetic variable assignment
#
# Reimplements the depth-0 / depth-1 variable-assignment probes from Saunshi
# et al. 2024 (arXiv:2409.19044, Appendix B); reused verbatim in Saunshi et al.
# 2025 (arXiv:2502.17416). The task format is:
#
#   math:   "The following is a set of simple mathematical equations."\n\n
#           n=22\nr=16\n...\n
#           What is the numerical value of n?\nAnswer:
#   code:   "The following is a very short Python program. ..."\nProgram:\n
#           n = 22\nr = 16\n...\n
#           \nQuestion:\nWhat is the value of n?\nAnswer:
#
# Depth-0: 5 base vars, each assigned a direct integer in [1, 25];
#          query targets one of the 5 base vars.
# Depth-1: 5 base vars + 5 aliases, with a strict 1-to-1 base <-> alias map
#          (each alias references a distinct base). Bases listed first, then
#          aliases (no forward references). Query targets one of the 5 aliases.
#
# Random-guess baseline is 1/5 = 20% (the model can choose among the 5 base
# values that appear in the prompt). Continuation is the answer token
# (e.g. "22"); continuation_delimiter=" " is baked into the scaffolding.
# ==============================================================================

_VAR_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
_VAL_LO, _VAL_HI = 1, 25
_N_VARS = 5

_MATH_HEADER = "The following is a set of simple mathematical equations."
_MATH_QUERY_TMPL = "What is the numerical value of {var}?\nAnswer:"
_CODE_HEADER = ("The following is a very short Python program. "
                "Use the program to resolve the value of the variable in the question.")
_CODE_QUERY_TMPL = "Question:\nWhat is the value of {var}?\nAnswer:"


def _make_var_assign_d0(rng: random.Random) -> dict:
    """5 base vars, direct integer assignments, query one base var."""
    bases = rng.sample(_VAR_LETTERS, _N_VARS)
    base_vals = {v: str(rng.randint(_VAL_LO, _VAL_HI)) for v in bases}
    query_var = rng.choice(bases)
    assignments = [(v, base_vals[v]) for v in bases]
    return {"assignments": assignments, "query": query_var,
            "answer": base_vals[query_var]}


def _make_var_assign_d1(rng: random.Random) -> dict:
    """5 base vars + 5 aliases (strict 1-to-1). Bases first then aliases."""
    all_vars = rng.sample(_VAR_LETTERS, 2 * _N_VARS)
    bases = all_vars[:_N_VARS]
    aliases = all_vars[_N_VARS:]
    base_vals = {v: str(rng.randint(_VAL_LO, _VAL_HI)) for v in bases}
    shuffled_bases = bases[:]
    rng.shuffle(shuffled_bases)
    alias_map = dict(zip(aliases, shuffled_bases))
    query_var = rng.choice(aliases)
    answer = base_vals[alias_map[query_var]]
    assignments = (
        [(v, base_vals[v]) for v in bases]          # bases first
        + [(a, alias_map[a]) for a in aliases]      # then aliases
    )
    return {"assignments": assignments, "query": query_var, "answer": answer}


def _to_math(ex: dict) -> dict:
    body = "\n".join(f"{v}={val}" for v, val in ex["assignments"])
    ctx = (f"{_MATH_HEADER}\n\n{body}\n"
           f"{_MATH_QUERY_TMPL.format(var=ex['query'])}")
    return {"context": ctx, "continuation": ex["answer"]}


def _to_code(ex: dict) -> dict:
    body = "\n".join(f"{v} = {val}" for v, val in ex["assignments"])
    ctx = (f"{_CODE_HEADER}\nProgram:\n{body}\n\n"
           f"{_CODE_QUERY_TMPL.format(var=ex['query'])}")
    return {"context": ctx, "continuation": ex["answer"]}


# The order of this list is the order of the single RNG stream: changing it changes every task after the first.
REASONING_PRIMITIVES = [
    ("var_assign_d0_math", _make_var_assign_d0, _to_math),
    ("var_assign_d0_code", _make_var_assign_d0, _to_code),
    ("var_assign_d1_math", _make_var_assign_d1, _to_math),
    ("var_assign_d1_code", _make_var_assign_d1, _to_code),
]


def prepare_reasoning_primitives(n_examples: int = 1000, seed: int = 42) -> dict[str, list[dict]]:
    """The four variable-assignment tasks, drawn in `REASONING_PRIMITIVES` order from one `random.Random(seed)`."""
    rng = random.Random(seed)
    out = {}
    for label, maker, formatter in REASONING_PRIMITIVES:
        out[label] = [formatter(maker(rng)) for _ in range(n_examples)]
    return out
