#!/usr/bin/env python3
"""The downstream bundle: read one and verify it, or rebuild one from the sources its manifest pins.

A bundle is a directory of three things:

    manifest.json        what every task is and where each record came from
    suite.json           the task list the scorer iterates (the copy in this package is the authority)
    records/<task>.jsonl one JSON object per line, one line per question

The suite is the 20 tasks of Schwethelm et al. (arXiv:2604.21106, v3) plus 6 further tasks of the CORE bundle of the
DCLM paper (arXiv:2406.11794). The records come from three kinds of source, and the manifest pins each one:

* `huggingface` (10 tasks) -- a public dataset at a named repository revision, turned into records by the builders in
  `alf.vendor.saunshi_bundle`. Two repositories are stored under their legacy un-namespaced ids, which recent
  `huggingface_hub` releases reject; `repo_aliases` in the suite maps them to the current names, and the manifest
  keeps the id the builder asked for.
* `core_archive` (11 tasks) -- a member of the published CORE archive, copied verbatim. `core.yaml` inside the
  archive gives each task its member path and its continuation delimiter, which for four tasks is not the suite's
  default " " but "\\nAnswer: "; the scorer reads the delimiter from the manifest, so this matters.
* `synthetic` (5 tasks) -- generated offline from a fixed seed by the upstream generators in
  `alf.vendor.owned_bundle` and `alf.vendor.saunshi_bundle`. The four variable-assignment tasks share one RNG
  stream, so they are always generated together, in the manifest's `generation_order`.

Reproducibility is by construction: the records of every task are a pure function of the pinned revision (or
archive, or seed), and the manifest carries the SHA-256 of each file, so a rebuild can be compared byte for byte
against an earlier bundle. `prepare_bundle(..., pin=<earlier bundle>)` does exactly that -- it takes the revisions
from that bundle's manifest and reports every task whose bytes differ.

`alf.bench.prepare` is the command line around `prepare_bundle`; `alf.eval.downstream` reads a
bundle with `read_bundle`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from alf.vendor import owned_bundle, saunshi_bundle

SCHEMA_VERSIONS = (1,)
TASK_TYPES = ("language_modeling", "multiple_choice")
SOURCE_KINDS = ("huggingface", "core_archive", "synthetic")
MAX_PER_TASK = 10000  # the paper's cap on the number of scored questions per task

# The published CORE archive. Its SHA-256 is checked before anything is read out of it.
CORE_ARCHIVE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
CORE_ARCHIVE_SHA256 = "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
CORE_ARCHIVE_NAME = "eval_bundle.zip"
CORE_CONFIG_MEMBER = "eval_bundle/core.yaml"
CORE_DATA_PREFIX = "eval_bundle/eval_data/"
CORE_DEFAULT_DELIMITER = " "

INDUCTION_HEAD_SEED = 1337
VAR_ASSIGN_SEED = 42
VAR_ASSIGN_TASKS = tuple(label for label, _, _ in saunshi_bundle.REASONING_PRIMITIVES)
SYNTHETIC_TASKS = ("induction_head",) + VAR_ASSIGN_TASKS

# The HuggingFace-sourced tasks and the vendored builder that turns each dataset into records.
HF_BUILDERS = {
    "triviaqa": saunshi_bundle.prepare_triviaqa,
    "nq": saunshi_bundle.prepare_nq,
    "webq": saunshi_bundle.prepare_webq,
    "tydiqa_goldp": saunshi_bundle.prepare_tydiqa_goldp,
    "squadv2": saunshi_bundle.prepare_squadv2,
    "drop": saunshi_bundle.prepare_drop,
    "coqa": saunshi_bundle.prepare_coqa,
    "svamp": saunshi_bundle.prepare_svamp,
    "asdiv": saunshi_bundle.prepare_asdiv,
    "mawps": saunshi_bundle.prepare_mawps,
}

SUITE_FILE = Path(__file__).with_name("suite.json")


class BundleError(ValueError):
    """A bundle, or a source it pins, does not match what the manifest says it is."""


# -----------------------------------------------------------------------------
# Reading


@dataclass(frozen=True)
class Task:
    """One task of a bundle: its records and the suite and manifest entries that describe them."""

    name: str
    axis: str
    task_type: str
    fewshot: int
    continuation_delimiter: str
    expected_examples: int
    group: str
    source: dict
    sha256: str
    path: Path
    spec: dict
    meta: dict
    records: list[dict]

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class Bundle:
    """A verified bundle. `tasks` is in suite order; `suite` and `manifest` are the files as written."""

    path: Path
    protocol: str
    suite: dict
    manifest: dict
    tasks: dict[str, Task]

    @property
    def records(self) -> dict[str, list[dict]]:
        return {name: task.records for name, task in self.tasks.items()}

    @property
    def names(self) -> list[str]:
        return list(self.tasks)

    def __getitem__(self, name: str) -> Task:
        return self.tasks[name]

    def __contains__(self, name: object) -> bool:
        return name in self.tasks

    def __len__(self) -> int:
        return len(self.tasks)


def read_bundle(path, *, verify_sha256: bool = True, check_records: bool = True, max_per_task: int | None = None,
                allow_count_mismatch: bool = False) -> Bundle:
    """Read `path` and verify it against its own manifest; raise `BundleError` on the first thing that disagrees.

    Checked: the suite's schema version; that the suite and the manifest name the same protocol and the same tasks;
    and, per task, that the records file sits inside the bundle, hashes to the manifest's `sha256` (unless
    `verify_sha256` is off), holds exactly `records` lines of valid JSON objects, and that the suite and the manifest
    agree on `task_type` (one of the two the scorer implements), `fewshot` (and that the pool is larger than it, as
    the few-shot sampler needs) and `axis`. `continuation_delimiter` must be a string in the manifest -- it is the
    one the scorer uses, and for the CORE tasks it legitimately differs from the suite's. The source must be one of
    the three kinds, with the fields that kind needs. With `check_records`, every record is checked for the fields
    its task type scores: `context`/`continuation` for language modeling, `query`/`choices`/`gold` for multiple
    choice, with `gold` in range. Finally `paper_count_matches` must hold and agree with the manifest, i.e.
    `min(records, max_per_task)` must be the suite's `expected_examples`, unless `allow_count_mismatch`.
    `max_per_task` defaults to the suite's own `defaults.max_per_task`, and to `MAX_PER_TASK` when it has none.
    """
    root = Path(path)
    if not root.is_dir():
        raise BundleError(f"not a bundle directory: {root}")
    suite = _read_json(root / "suite.json")
    manifest = _read_json(root / "manifest.json")
    if max_per_task is None:
        max_per_task = suite.get("defaults", {}).get("max_per_task", MAX_PER_TASK)

    version = suite.get("schema_version")
    if version not in SCHEMA_VERSIONS:
        raise BundleError(f"{root}: suite schema_version {version!r} is not one of {list(SCHEMA_VERSIONS)}")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, str) or not protocol:
        raise BundleError(f"{root}: manifest has no protocol")
    if suite.get("protocol") != protocol:
        raise BundleError(f"{root}: suite protocol {suite.get('protocol')!r} != manifest protocol {protocol!r}")

    specs = suite.get("tasks")
    metas = manifest.get("tasks")
    if not isinstance(specs, list) or not specs:
        raise BundleError(f"{root}: suite has no tasks")
    if not isinstance(metas, dict) or not metas:
        raise BundleError(f"{root}: manifest has no tasks")
    names = [spec.get("name") for spec in specs]
    if len(set(names)) != len(names):
        raise BundleError(f"{root}: the suite names a task twice: {sorted(n for n in names if names.count(n) > 1)}")
    missing = [n for n in names if n not in metas]
    extra = sorted(set(metas) - set(names))
    if missing or extra:
        raise BundleError(f"{root}: suite and manifest disagree on the task list "
                          f"(not in the manifest: {missing}; not in the suite: {extra})")

    tasks = {}
    for spec in specs:
        tasks[spec["name"]] = _read_task(root, spec, metas[spec["name"]], verify_sha256=verify_sha256,
                                         check_records=check_records, max_per_task=max_per_task,
                                         allow_count_mismatch=allow_count_mismatch)
    return Bundle(path=root, protocol=protocol, suite=suite, manifest=manifest, tasks=tasks)


def _read_task(root: Path, spec: dict, meta: dict, *, verify_sha256: bool, check_records: bool, max_per_task: int,
               allow_count_mismatch: bool) -> Task:
    name = spec["name"]
    rel = meta.get("file")
    if not isinstance(rel, str) or not rel:
        raise BundleError(f"{name}: the manifest entry has no records file")
    path = (root / rel).resolve()
    if os.path.isabs(rel) or root.resolve() not in path.parents:
        raise BundleError(f"{name}: records file {rel!r} is not inside the bundle")
    if not path.is_file():
        raise BundleError(f"{name}: missing records file {path}")

    raw = path.read_bytes()
    if verify_sha256:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != meta.get("sha256"):
            raise BundleError(f"{name}: {rel} hashes to {digest}, the manifest says {meta.get('sha256')}")
    records = _parse_jsonl(raw, name)
    if len(records) != meta.get("records"):
        raise BundleError(f"{name}: {rel} holds {len(records)} records, the manifest says {meta.get('records')}")

    task_type = _agree(name, "task_type", spec, meta)
    if task_type not in TASK_TYPES:
        raise BundleError(f"{name}: task_type {task_type!r} is not one of {list(TASK_TYPES)}")
    fewshot = _agree(name, "fewshot", spec, meta)
    if not isinstance(fewshot, int) or isinstance(fewshot, bool) or fewshot < 0:
        raise BundleError(f"{name}: fewshot {fewshot!r} is not a non-negative integer")
    if len(records) <= fewshot:
        raise BundleError(f"{name}: {len(records)} records cannot serve {fewshot}-shot prompts "
                          f"(the few-shot sampler draws from the other records, so it needs more than {fewshot})")
    axis = _agree(name, "axis", spec, meta)
    if not isinstance(axis, str) or not axis:
        raise BundleError(f"{name}: axis {axis!r} is not a name")
    delimiter = meta.get("continuation_delimiter")
    if not isinstance(delimiter, str):
        raise BundleError(f"{name}: the manifest has no continuation_delimiter")
    source = meta.get("source")
    _check_source(name, source)
    expected = spec.get("expected_examples")
    scored = min(len(records), max_per_task)
    matches = scored == expected
    if meta.get("paper_count_matches") != matches:
        raise BundleError(f"{name}: the manifest says paper_count_matches={meta.get('paper_count_matches')}, but "
                          f"{len(records)} records capped at {max_per_task} give {scored} scored questions and the "
                          f"suite expects {expected}")
    if not matches and not allow_count_mismatch:
        raise BundleError(f"{name}: {scored} scored questions, the suite expects {expected}")
    if check_records:
        _check_records(name, task_type, records)
    return Task(name=name, axis=axis, task_type=task_type, fewshot=fewshot, continuation_delimiter=delimiter,
                expected_examples=expected, group=spec.get("bundle", ""), source=source, sha256=meta.get("sha256"),
                path=path, spec=spec, meta=meta, records=records)


def _agree(name: str, key: str, spec: dict, meta: dict):
    if key not in spec or key not in meta:
        raise BundleError(f"{name}: {key!r} is missing from the "
                          f"{'suite' if key not in spec else 'manifest'} entry")
    if spec[key] != meta[key]:
        raise BundleError(f"{name}: suite {key}={spec[key]!r} != manifest {key}={meta[key]!r}")
    return spec[key]


def _check_source(name: str, source) -> None:
    if not isinstance(source, dict):
        raise BundleError(f"{name}: the manifest entry has no source")
    kind = source.get("kind")
    if kind not in SOURCE_KINDS:
        raise BundleError(f"{name}: source kind {kind!r} is not one of {list(SOURCE_KINDS)}")
    if kind == "huggingface":
        pinned = source.get("datasets")
        if not isinstance(pinned, list) or not pinned:
            raise BundleError(f"{name}: a huggingface source must pin at least one dataset")
        for entry in pinned:
            for key in ("repository", "revision", "split"):
                if not entry.get(key):
                    raise BundleError(f"{name}: a pinned dataset has no {key}")
    elif kind == "core_archive":
        for key in ("url", "archive_sha256", "member"):
            if not source.get(key):
                raise BundleError(f"{name}: a core_archive source has no {key}")
        if not source["member"].startswith(CORE_DATA_PREFIX):
            raise BundleError(f"{name}: archive member {source['member']!r} is outside {CORE_DATA_PREFIX}")
    elif source.get("seed") is None:
        raise BundleError(f"{name}: a synthetic source has no seed")


def _check_records(name: str, task_type: str, records: list[dict]) -> None:
    for index, item in enumerate(records):
        if task_type == "language_modeling":
            if not isinstance(item.get("context"), str) or not isinstance(item.get("continuation"), str):
                raise BundleError(f"{name} record {index}: needs string 'context' and 'continuation'")
        else:
            choices = item.get("choices")
            gold = item.get("gold")
            if not isinstance(item.get("query"), str):
                raise BundleError(f"{name} record {index}: needs a string 'query'")
            if not isinstance(choices, list) or not choices or not all(isinstance(c, str) for c in choices):
                raise BundleError(f"{name} record {index}: needs a non-empty list of string 'choices'")
            if not isinstance(gold, int) or isinstance(gold, bool) or not 0 <= gold < len(choices):
                raise BundleError(f"{name} record {index}: 'gold' {gold!r} is not an index into {len(choices)} choices")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise BundleError(f"missing {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BundleError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{path}: expected a JSON object")
    return value


def _parse_jsonl(raw: bytes, name: str) -> list[dict]:
    """One record per line. Split on b"\\n" only: a record may hold U+2028 and other characters that
    `str.splitlines` would split on, which would cut a record in two."""
    if raw and not raw.endswith(b"\n"):
        raise BundleError(f"{name}: the records file does not end with a newline")
    records = []
    for number, line in enumerate(raw.split(b"\n")[:-1], 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise BundleError(f"{name} line {number}: {error}") from error
        if not isinstance(item, dict):
            raise BundleError(f"{name} line {number}: expected a JSON object")
        records.append(item)
    return records


# -----------------------------------------------------------------------------
# Rebuilding


def default_suite() -> dict:
    """The suite this package ships: all 26 tasks, their axes, few-shot counts and expected question counts."""
    return _read_json(SUITE_FILE)


def prepare_bundle(dest, suite: dict | None = None, *, pin=None, data_root=None, archive=None, online: bool = True,
                   allow_count_mismatch: bool = False, dry_run: bool = False,
                   log=print) -> dict:
    """Build the bundle `suite` describes into `dest`, and return the manifest written there.

    `pin` is an earlier bundle (a path) or its manifest (a dict). Its recorded dataset revisions are then used
    instead of resolving each repository's current head, and every task whose bytes differ from the pinned SHA-256 is
    reported; this is what makes a rebuild comparable byte for byte. `archive` is a local copy of the CORE archive;
    without one it is fetched into `data_root` (default: the operating system's temporary directory) once, and either
    way its SHA-256 must be `CORE_ARCHIVE_SHA256`. `online=False` refuses anything that would reach the network.

    `dry_run=True` writes nothing and returns a plan instead of a manifest: the synthetic tasks are generated and
    hashed, the CORE tasks are resolved and hashed if an archive is at hand, and the HuggingFace tasks are reported
    with the repository id, the alias it resolves to, the subset, the split and the pinned revision. It needs no
    network, so it is the way to check a `pin` and this code against each other before downloading anything.
    """
    suite = default_suite() if suite is None else suite
    pinned = _pinned_tasks(pin)
    specs = suite.get("tasks") or []
    if not specs:
        raise BundleError("the suite has no tasks")
    aliases = dict(suite.get("repo_aliases") or {})
    kinds = {spec["name"]: _source_kind(spec) for spec in specs}

    dest = Path(dest)
    if not dry_run:
        if dest.exists() and any(dest.iterdir()):
            raise BundleError(f"{dest} exists and is not empty; a bundle is never written over")
        (dest / "records").mkdir(parents=True, exist_ok=True)

    core = None
    if "core_archive" in kinds.values():
        core = _CoreArchive.open(archive=archive, data_root=data_root, online=online and not dry_run,
                                 required=not dry_run, log=log)
    loader = None
    if not dry_run and "huggingface" in kinds.values():
        if not online:
            raise BundleError("the HuggingFace-sourced tasks need network access; pass online=True")
        loader = _HuggingFaceLoader(aliases, pinned)
    synthetic = _synthetic_records([name for name, kind in kinds.items() if kind == "synthetic"])

    entries, mismatches = {}, []
    for spec in specs:
        name = spec["name"]
        kind = kinds[name]
        delimiter = spec["continuation_delimiter"]
        plan = {}
        if kind == "synthetic":
            data = _jsonl(synthetic[name])
            source = _synthetic_source(name)
        elif kind == "core_archive":
            member, delimiter, data = core.task(name, pinned.get(name))
            source = {"kind": "core_archive", "url": CORE_ARCHIVE_URL, "archive_sha256": CORE_ARCHIVE_SHA256,
                      "member": member, "continuation_delimiter": delimiter}
        else:
            repository, subset, split = _hf_plan(name, pinned.get(name))
            plan = {"repository": repository, "resolves_to": aliases.get(repository, repository), "subset": subset,
                    "split": split, "revision": _pinned_revision(pinned.get(name))}
            if dry_run:
                data, source = None, {"kind": "huggingface", "datasets": _pinned_datasets(pinned.get(name))}
            else:
                loader.reset()
                data = _jsonl(HF_BUILDERS[name](loader))
                source = {"kind": "huggingface", "datasets": loader.loaded}

        entry = {"file": f"records/{name}.jsonl", "records": None, "sha256": None, "task_type": spec["task_type"],
                 "axis": spec["axis"], "fewshot": spec["fewshot"], "continuation_delimiter": delimiter,
                 "paper_count_matches": None, "source": source}
        if data is not None:
            entry["records"] = data.count(b"\n")
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            scored = min(entry["records"], suite.get("defaults", {}).get("max_per_task", MAX_PER_TASK))
            entry["paper_count_matches"] = scored == spec["expected_examples"]
            if not entry["paper_count_matches"] and not allow_count_mismatch:
                raise BundleError(f"{name}: {scored} scored questions, the suite expects "
                                  f"{spec['expected_examples']}; pass allow_count_mismatch to write it anyway")
            if not dry_run:
                (dest / "records" / f"{name}.jsonl").write_bytes(data)
        if plan and dry_run:  # the plan is dry-run reporting; a manifest carries only what was actually loaded
            entry["plan"] = plan
        note = _compare_pin(name, entry, pinned.get(name))
        if note:
            mismatches.append(note)
        entries[name] = entry
        log(f"{name:28s} kind={kind:13s} records={entry['records']!s:>6s} "
            f"delim={entry['continuation_delimiter']!r} {note or 'ok'}")

    if dry_run:
        return {"dry_run": True, "protocol": suite.get("protocol"), "dest": str(dest),
                "archive": None if core is None or core.path is None else str(core.path),
                "pinned": bool(pinned), "mismatches": mismatches, "tasks": entries}

    manifest = {"protocol": suite.get("protocol"), "upstreams": suite.get("upstreams", {}), "tasks": entries}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest / "suite.json").write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if mismatches:
        raise BundleError(f"{dest} was written, but it does not reproduce the bundle it was pinned to:\n  "
                          + "\n  ".join(mismatches))
    return manifest


def _source_kind(spec: dict) -> str:
    name = spec.get("name")
    if name in SYNTHETIC_TASKS:
        return "synthetic"
    if spec.get("bundle") == "core":
        return "core_archive"
    if name in HF_BUILDERS:
        return "huggingface"
    raise BundleError(f"{name}: no source for this task; it is neither synthetic, nor a CORE archive member, "
                      f"nor one of the HuggingFace-sourced tasks {sorted(HF_BUILDERS)}")


def _synthetic_records(names: list[str]) -> dict[str, list[dict]]:
    """The requested synthetic tasks. The four variable-assignment tasks come from one RNG stream, so they are all
    generated whenever any of them is wanted: generating a subset would change the records of every task but the
    first."""
    out = {}
    if "induction_head" in names:
        out["induction_head"] = owned_bundle.prepare_induction_head(seed=INDUCTION_HEAD_SEED)
    if any(name in VAR_ASSIGN_TASKS for name in names):
        out.update(saunshi_bundle.prepare_reasoning_primitives(seed=VAR_ASSIGN_SEED))
    unknown = sorted(set(names) - set(out))
    if unknown:
        raise BundleError(f"no synthetic generator for {unknown}")
    return out


def _synthetic_source(name: str) -> dict:
    if name == "induction_head":
        return {"kind": "synthetic", "seed": INDUCTION_HEAD_SEED}
    return {"kind": "synthetic", "seed": VAR_ASSIGN_SEED,
            "generation_order": ",".join(label.removeprefix("var_assign_") for label in VAR_ASSIGN_TASKS)}


def _jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")


def _pinned_tasks(pin) -> dict:
    """The `tasks` table of the manifest to reproduce: `pin` is a bundle directory, a manifest file or a manifest."""
    if pin is None:
        return {}
    if isinstance(pin, dict):
        manifest = pin
    else:
        path = Path(pin)
        manifest = _read_json(path / "manifest.json" if path.is_dir() else path)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise BundleError(f"the bundle pinned with pin={pin!r} has no tasks")
    return tasks


def _pinned_datasets(meta: dict | None) -> list[dict]:
    return list((meta or {}).get("source", {}).get("datasets") or [])


def _pinned_revision(meta: dict | None) -> str | None:
    pinned = _pinned_datasets(meta)
    return pinned[0].get("revision") if pinned else None


def _hf_plan(name: str, meta: dict | None) -> tuple[str, str | None, str | None]:
    """What the vendored builder for `name` will ask for: repository id, subset and split, from the pin when there is
    one. The builder is the authority -- a pin only says which revision of that repository to take."""
    pinned = _pinned_datasets(meta)
    if pinned:
        first = pinned[0]
        return first.get("repository"), first.get("subset"), first.get("split")
    return _BUILDER_SOURCES[name]


# What each vendored builder loads, for the dry-run plan when no pin says so. The builders remain the authority:
# `prepare_bundle` records what they actually asked the loader for.
_BUILDER_SOURCES = {
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation"),
    "nq": ("nq_open", None, "validation"),
    "webq": ("web_questions", None, "test"),
    "tydiqa_goldp": ("google-research-datasets/tydiqa", "secondary_task", "validation"),
    "squadv2": ("rajpurkar/squad_v2", None, "validation"),
    "drop": ("ucinlp/drop", None, "validation"),
    "coqa": ("stanfordnlp/coqa", None, "validation"),
    "svamp": ("ChilleD/SVAMP", None, "test"),
    "asdiv": ("EleutherAI/asdiv", None, "validation"),
    "mawps": ("mwpt5/MAWPS", None, "train"),
}


def _compare_pin(name: str, entry: dict, meta: dict | None) -> str | None:
    """Why `entry` differs from the pinned manifest entry `meta`, or None when it does not (or cannot be told)."""
    if not meta:
        return None
    for key in ("continuation_delimiter", "records", "sha256"):
        want, got = meta.get(key), entry.get(key)
        if got is None:
            continue
        if want != got:
            return f"{name}: {key} is {got!r}, the pinned bundle has {want!r}"
    want_member = meta.get("source", {}).get("member")
    got_member = entry.get("source", {}).get("member")
    if want_member and got_member and want_member != got_member:
        return f"{name}: archive member is {got_member!r}, the pinned bundle has {want_member!r}"
    return None


class _HuggingFaceLoader:
    """`load_dataset` for the vendored builders: resolves the legacy repository ids, pins the revision and records
    the provenance of every dataset it hands out.

    The builders call it positionally, as `load_dataset(repository, subset, split=...)`, and the manifest keeps the
    repository id the builder asked for, not the alias it resolves to.
    """

    def __init__(self, aliases: dict, pinned: dict | None = None):
        self.aliases = dict(aliases or {})
        self.pinned = pinned or {}
        self.loaded: list[dict] = []

    def reset(self) -> None:
        self.loaded = []

    def __call__(self, path: str, name: str | None = None, *, split: str | None = None, **kwargs):
        import datasets

        repository = self.aliases.get(path, path)
        revision = self._revision(path, name, split, repository)
        dataset = datasets.load_dataset(repository, name, split=split, revision=revision, **kwargs)
        self.loaded.append({"repository": path, "revision": revision, "subset": name, "split": split,
                            "fingerprint": getattr(dataset, "_fingerprint", None), "source_records": len(dataset)})
        return dataset

    def _revision(self, path: str, subset: str | None, split: str | None, repository: str) -> str:
        for meta in self.pinned.values():
            for entry in _pinned_datasets(meta):
                if (entry.get("repository"), entry.get("subset"), entry.get("split")) == (path, subset, split):
                    return entry["revision"]
        import huggingface_hub

        return huggingface_hub.HfApi().dataset_info(repository).sha


class _CoreArchive:
    """The published CORE archive, verified by SHA-256, with `core.yaml` resolving each task to a member and a
    continuation delimiter. `path` is None when no archive was at hand (a dry run can still report member paths from
    a pin)."""

    def __init__(self, path: Path | None, config: dict):
        self.path = path
        self.config = config

    @classmethod
    def open(cls, *, archive=None, data_root=None, online: bool, required: bool, log=print) -> "_CoreArchive":
        path = cls._locate(archive=archive, data_root=data_root, online=online, required=required, log=log)
        if path is None:
            return cls(None, {})
        digest = _sha256_file(path)
        if digest != CORE_ARCHIVE_SHA256:
            raise BundleError(f"{path} hashes to {digest}, not the published {CORE_ARCHIVE_SHA256}")
        import yaml

        with zipfile.ZipFile(path) as zf:
            config = yaml.safe_load(zf.read(CORE_CONFIG_MEMBER).decode("utf-8"))
        tasks = {}
        for item in config.get("icl_tasks") or []:
            tasks[item["label"]] = (CORE_DATA_PREFIX + item["dataset_uri"],
                                    item.get("continuation_delimiter", CORE_DEFAULT_DELIMITER))
        if not tasks:
            raise BundleError(f"{path}: {CORE_CONFIG_MEMBER} lists no tasks")
        return cls(path, tasks)

    @staticmethod
    def _locate(*, archive, data_root, online: bool, required: bool, log) -> Path | None:
        if archive:
            path = Path(archive)
            if not path.is_file():
                raise BundleError(f"no CORE archive at {path}")
            return path
        root = Path(data_root) if data_root else Path(tempfile.gettempdir())
        cached = root / CORE_ARCHIVE_NAME
        if cached.is_file():
            return cached
        if not online:
            if required:
                raise BundleError(f"no CORE archive at {cached}; download {CORE_ARCHIVE_URL} there, pass it as "
                                  f"archive=..., or allow the download with online=True")
            return None
        root.mkdir(parents=True, exist_ok=True)
        log(f"downloading {CORE_ARCHIVE_URL} -> {cached}")
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
            with urlopen(CORE_ARCHIVE_URL) as response:  # noqa: S310 -- a fixed https URL, checked by SHA-256 below
                shutil.copyfileobj(response, tmp)
        os.replace(tmp.name, cached)
        return cached

    def task(self, name: str, meta: dict | None) -> tuple[str, str, bytes | None]:
        """`(member, continuation_delimiter, member bytes)` for one CORE task; the bytes are None without an
        archive. CORE records are copied out of the archive verbatim, so that a rebuilt bundle is byte-identical."""
        if name in self.config:
            member, delimiter = self.config[name]
        else:
            pinned = (meta or {}).get("source", {})
            member, delimiter = pinned.get("member"), pinned.get("continuation_delimiter")
            if not member:
                raise BundleError(f"{name}: no such task in {CORE_CONFIG_MEMBER} and no pinned archive member")
            delimiter = CORE_DEFAULT_DELIMITER if delimiter is None else delimiter
        if self.path is None:
            return member, delimiter, None
        with zipfile.ZipFile(self.path) as zf:
            try:
                data = zf.read(member)
            except KeyError as error:
                raise BundleError(f"{name}: {member} is not in {self.path}") from error
        if data and not data.endswith(b"\n"):
            data += b"\n"
        return member, delimiter, data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
