#!/usr/bin/env python3
"""Tokenize the training corpus into the levanter caches every run of the paper reads.

The corpus is Dolma 3 HQ 300B shuffled: a sample of the Dolma 3 high-quality mix at its official proportions, released
as 256 parquet shards named ``train-<nnnnn>-of-00256.parquet`` with one ``text`` column per document (the README's
data section names the release and the command that fetches it). ``--src`` is a local directory of those shards and
``--src-revision`` is the revision of the release they were taken from, which is recorded in the manifest so a cache
can be traced back to an exact version of the dataset. Nothing here is specific to Dolma: any directory of parquet
shards with a ``text`` column and that file naming can be tokenized the same way.

Tokenization uses the Marin tokenizer (``marin-community/marin-tokenizer``, 128,256 entries) and levanter's own
TextLmDatasetFormat preprocessor, with BOS + EOS around every document, exactly as the Marin recipe's own caches are
built; ``alf.marin.recipe.dolma_data_config`` then concatenates the caches into one training component.

A work unit is (shard, part): the shard's parquet row groups are split into --parts contiguous ranges and one process
tokenizes its range and writes a TreeCache with SerialCacheWriter to <out>/s<shard>_p<part>. Each unit is written
under a temporary name and renamed once its ledger is complete, so a killed run leaves no partial cache under a final
name; finished units are skipped on a rerun, which makes the whole job restartable. manifest.json (written last) lists
every unit with its document and token counts, in shard / part order, and the trainer checks the token total against
the run's length. The first run records --src in <out>/source.json, and a later run with another --src is refused
rather than mixing two corpora in one directory. Several machines can fill disjoint shard ranges of one directory with
--no-manifest; a final run over the whole range then skips every unit and writes the manifest. --check writes nothing:
it reports how many units of the requested range are complete and exits non-zero if any are missing, which is how a
directory is verified before a training run starts.

  python -m alf.tokenize_dolma --out DIR --shards 0-15 --parts 4 --workers 64
  python -m alf.tokenize_dolma --out DIR --shards 0-15 --parts 4 --check
  python -m alf.tokenize_dolma --src <parquet dir> --src-revision <release revision> --out DIR --shards 0-255 \
      --parts 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

SRC = os.environ.get("DOLMA_SRC", "")  # default --src: a local directory of the corpus's parquet shards
# The revision of the released dataset the default shards were taken from; recorded with every cache it produces.
SRC_HF_REVISION = "b9d34c9c68d0032a95d20f0322be9ab9e06c4f32"
LEDGER = "shard_ledger.json"
SOURCE_FILE = "source.json"
DOCS_PER_CALL = 4096


def parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for item in spec.split(","):
        lo, _, hi = item.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def unit_name(shard: int, part: int) -> str:
    return f"s{shard:05d}_p{part}"


def ledger_counts(path: str) -> tuple[int, int] | None:
    try:
        with open(os.path.join(path, LEDGER)) as f:
            ledger = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not ledger.get("is_finished"):
        return None
    return int(ledger["total_num_rows"]), int(ledger["field_counts"]["input_ids"])


def claim_source(out: str, src: str, revision: str) -> None:
    recorded = None
    for name in (SOURCE_FILE, "manifest.json"):  # caches written before source.json was added only have the manifest
        try:
            with open(os.path.join(out, name)) as f:
                recorded = json.load(f)["source"]
            break
        except FileNotFoundError:
            continue
    if recorded is not None and os.path.abspath(recorded) != os.path.abspath(src) \
            and os.path.realpath(recorded) != os.path.realpath(src):
        sys.exit(f"[tok] {out} holds caches of {recorded}; refusing to add units from {src}")
    if not os.path.exists(os.path.join(out, SOURCE_FILE)):
        tmp = os.path.join(out, f".{SOURCE_FILE}.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump({"source": src, "source_hf_revision": revision}, f)
            f.write("\n")
        os.replace(tmp, os.path.join(out, SOURCE_FILE))


def _preprocessor(tokenizer_dir: str):
    from levanter.data.text.formats import TextLmDatasetFormat, preprocessor_for_format
    from levanter.tokenizers import load_tokenizer

    return preprocessor_for_format(TextLmDatasetFormat(), load_tokenizer(tokenizer_dir), enforce_bos=True, enforce_eos=True)


def run_unit(task: tuple[int, int, int, str, str, str]) -> dict:
    shard, part, parts, src, out, tokenizer_dir = task
    import pyarrow.parquet as pq
    from levanter.store.cache import CacheMetadata, SerialCacheWriter

    name = unit_name(shard, part)
    final = os.path.join(out, name)
    counts = ledger_counts(final)
    if counts is not None:
        return {"name": name, "shard": shard, "part": part, "docs": counts[0], "tokens": counts[1], "seconds": 0.0, "skipped": True}
    t0 = time.time()
    proc = _preprocessor(tokenizer_dir)
    pf = pq.ParquetFile(os.path.join(src, f"train-{shard:05d}-of-00256.parquet"))
    n_groups = pf.metadata.num_row_groups
    lo, hi = part * n_groups // parts, (part + 1) * n_groups // parts
    tmp = os.path.join(out, f".tmp_{name}_{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    with SerialCacheWriter(tmp, proc.output_exemplar, metadata=CacheMetadata(preprocessor_metadata=proc.metadata)) as writer:
        for group in range(lo, hi):
            texts = pf.read_row_group(group, columns=["text"]).column("text").to_pylist()
            for i in range(0, len(texts), DOCS_PER_CALL):
                writer.write_batch(proc([{"text": t} for t in texts[i : i + DOCS_PER_CALL]]))
    counts = ledger_counts(tmp)
    if counts is None:
        raise RuntimeError(f"{name}: ledger missing after write")
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(tmp, final)
    return {"name": name, "shard": shard, "part": part, "row_groups": [lo, hi], "docs": counts[0], "tokens": counts[1],
            "seconds": round(time.time() - t0, 1), "skipped": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", required=True, help="e.g. 0-15 or 0,3,7-9")
    ap.add_argument("--parts", type=int, default=4, help="row-group ranges per shard, one process each")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--src-revision", default=SRC_HF_REVISION,
                    help="revision of the released dataset --src came from; recorded in manifest.json so a cache can "
                         "be traced to an exact version of the corpus")
    ap.add_argument("--tokenizer", default=os.environ.get("MARIN_TOKENIZER_DIR", "marin-community/marin-tokenizer"))
    ap.add_argument("--check", action="store_true",
                    help="write nothing: report how many units of the range are complete and exit non-zero if any "
                         "are missing")
    ap.add_argument("--no-manifest", action="store_true",
                    help="tokenize only; leave manifest.json to a run over the full range")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    units = [(s, p) for s in parse_shards(args.shards) for p in range(args.parts)]
    if args.check:
        missing = [unit_name(s, p) for s, p in units if ledger_counts(os.path.join(args.out, unit_name(s, p))) is None]
        print(f"[tok] {len(units) - len(missing)}/{len(units)} units complete" + (f"; missing {missing[:8]}" if missing else ""))
        sys.exit(1 if missing else 0)
    if not os.path.isdir(args.src):
        raise SystemExit(f"--src must be a directory of train-<nnnnn>-of-00256.parquet shards (got {args.src!r}); "
                         "set DOLMA_SRC in .env or pass --src")
    claim_source(args.out, args.src, args.src_revision)

    import multiprocessing as mp

    tasks = [(s, p, args.parts, args.src, args.out, args.tokenizer) for s, p in units]
    todo = sum(ledger_counts(os.path.join(args.out, unit_name(s, p))) is None for s, p in units)
    print(f"[tok] {len(units)} units ({todo} to do) from {args.src} -> {args.out}, {args.workers} workers", flush=True)
    t0, results = time.time(), []
    with mp.get_context("spawn").Pool(min(args.workers, max(1, todo))) as pool:
        for r in pool.imap_unordered(run_unit, tasks):
            results.append(r)
            if not r["skipped"]:
                print(f"[tok] {r['name']} docs={r['docs']} tokens={r['tokens']} {r['seconds']:.0f}s "
                      f"({len(results)}/{len(units)}, {time.time() - t0:.0f}s)", flush=True)
    results.sort(key=lambda r: (r["shard"], r["part"]))
    if args.no_manifest:
        print(f"[tok] {len(results)} units, {sum(r['docs'] for r in results)} docs, {sum(r['tokens'] for r in results)} tokens, "
              f"{time.time() - t0:.0f}s (no manifest)", flush=True)
        return
    proc = _preprocessor(args.tokenizer)
    manifest = {
        "source": args.src, "source_hf_revision": args.src_revision, "shards": parse_shards(args.shards), "parts": args.parts,
        "tokenizer_metadata": proc.metadata, "units": [{k: r[k] for k in ("name", "shard", "part", "docs", "tokens")} for r in results],
        "total_docs": sum(r["docs"] for r in results), "total_tokens": sum(r["tokens"] for r in results),
        "written_at": time.strftime("%Y%m%dT%H%M%S%z"),
    }
    tmp = os.path.join(args.out, ".manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    os.replace(tmp, os.path.join(args.out, "manifest.json"))
    print(f"[tok] manifest: {len(results)} units, {manifest['total_docs']} docs, {manifest['total_tokens']} tokens, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
