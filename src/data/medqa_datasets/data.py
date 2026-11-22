"""Load MedMCQA and MedQA, convert both to one format, deduplicate, and build fixed splits.

Entry point: load_splits("medmcqa" | "medqa") -> {split name: [examples]}
MedMCQA: train, hp_dev, calib, thresh, test.  MedQA: train, val, test.
Example format: {"id", "source", "question", "options": {"A": ..., "B": ...}, "answer": "B"}
"""
import hashlib
import importlib.metadata
import json
import math
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

from src.utils.paths import CACHE_DIR, DATA_DIR, HF_CACHE_DIR, PROCESSED_DIR, SPLITS_DIR

# -------------------------------------------------------------------------------------------------
# Shared configuration. Everything here defines the splits. If you change any of it, run build_all()
# again and re-share the splits folder: load_splits refuses to run on splits built with another config.
# -------------------------------------------------------------------------------------------------
SEED = 42
SPLIT_SIZES = {  # rows carved out of each dataset's train set, in this order (the rest stays train)
    "medmcqa": {"hp_dev": 2000, "calib": 1500, "thresh": 2500},
    "medqa": {"val": 1000},
}
SIMILARITY_FILTER = {  # subset rows must have no other train question with Levenshtein similarity >= threshold
    "medmcqa": {"threshold": 0.85, "pool_factor": 2},  # pool_factor: candidates examined = pool_factor * total subset rows
}
SHORT_STEM_WORDS = 10  # leak_key: stems with fewer words must also match on options
DEDUPE_SCHEME = "two-tier-v1"  # exact_key inside train, leak_key against val/test; bump if keys change
HF_REPOS = {"medmcqa": "openlifescienceai/medmcqa", "medqa": "GBaker/MedQA-USMLE-4-options"}
HF_REVISIONS = {  # pinned dataset commits (None would mean "latest")
    "medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    "medqa": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
}
SPLIT_PRIORITY = ("test", "val", "hp_dev", "calib", "thresh", "train")  # a repeated question stays in the earliest; train last
DATASET_ORDER = ("medqa", "medmcqa")  # tie-break inside a split: MedQA first (protects MedAbstain)
LETTERS = string.ascii_uppercase  # option labels by position, so any option count up to 26 works

RESOLVED_REVISIONS = {}  # filled by _load_hf: the dataset commits actually used
_PUNCT = re.compile(r"[^\w\s]")


def current_config() -> dict:
    """Snapshot of every setting that changes the splits (JSON-serializable)."""
    return {
        "seed": SEED,
        "split_sizes": SPLIT_SIZES,
        "similarity_filter": SIMILARITY_FILTER,
        "short_stem_words": SHORT_STEM_WORDS,
        "dedupe_scheme": DEDUPE_SCHEME,
        "hf_repos": HF_REPOS,
        "hf_revisions": HF_REVISIONS,
        "split_priority": list(SPLIT_PRIORITY),
        "dataset_order": list(DATASET_ORDER),
    }


def normalize(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def _answer_text(ex: dict) -> str:
    """Normalized text of the correct option (to detect duplicates that disagree on the answer)."""
    return normalize(ex["options"][ex["answer"]])


def _options_key(ex: dict) -> str:
    return " | ".join(sorted(normalize(v) for v in ex["options"].values()))


def exact_key(ex: dict) -> str:
    """True-copy identity: same question and same set of options. Used to deduplicate train."""
    return f"{normalize(ex['question'])} || {_options_key(ex)}"


def leak_key(ex: dict) -> str:
    """Strict identity for protecting val/test: long stems match on the question alone."""
    q = normalize(ex["question"])
    if len(q.split()) >= SHORT_STEM_WORDS:
        return q
    return f"{q} || {_options_key(ex)}"


def _load_hf(name: str):
    """Download one dataset (pinned revision if set) and remember which commit was used."""
    from datasets import load_dataset

    rev = HF_REVISIONS[name]
    ds = load_dataset(HF_REPOS[name], revision=rev, cache_dir=str(HF_CACHE_DIR))
    try:
        from huggingface_hub import HfApi

        RESOLVED_REVISIONS[name] = HfApi().dataset_info(HF_REPOS[name], revision=rev).sha
    except Exception:
        RESOLVED_REVISIONS[name] = rev
    return ds


def load_medmcqa_raw() -> dict:
    """MedMCQA in the common format. Official validation is used as our test (official test has no answers)."""
    ds = _load_hf("medmcqa")
    out = {}
    for hf_split, ours in (("train", "train"), ("validation", "test")):
        rows = []
        for r in ds[hf_split]:
            if r["cop"] not in (0, 1, 2, 3):
                continue
            opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
            rows.append({
                "id": f"medmcqa-{r['id']}",
                "source": "medmcqa",
                "question": r["question"].strip(),
                "options": {LETTERS[i]: str(o).strip() for i, o in enumerate(opts)},
                "answer": LETTERS[r["cop"]],
            })
        out[ours] = rows
    return out


def load_medqa_raw() -> dict:
    """MedQA (English, 4 options) in the common format. HF has train and test only."""
    ds = _load_hf("medqa")
    out = {}
    for split in ("train", "test"):
        rows = []
        for i, r in enumerate(ds[split]):
            opts = {k: str(v).strip() for k, v in sorted(r["options"].items())}
            assert r["answer_idx"] in opts, (split, i)
            rows.append({
                "id": f"medqa-{split}-{i}",
                "source": "medqa",
                "question": r["question"].strip(),
                "options": opts,
                "answer": r["answer_idx"],
            })
        out[split] = rows
    return out


def dedupe_within(rows: list, key_fn):
    """Keep the first row per key_fn(row). Returns (kept, n_removed, n_removed_with_conflicting_answer)."""
    first, kept = {}, []
    removed = conflicts = 0
    for r in rows:
        k = key_fn(r)
        if k not in first:
            first[k] = r
            kept.append(r)
            continue
        removed += 1
        if _answer_text(first[k]) != _answer_text(r):
            conflicts += 1
    return kept, removed, conflicts


def split_names(name: str) -> tuple:
    """Split names of one dataset: train, the carved subsets (SPLIT_SIZES order), test."""
    return ("train", *SPLIT_SIZES[name], "test")


def near_duplicate_ids(rows: list, reference: list, threshold: float, cache_dir=None) -> set:
    """Ids of rows with a different-text question in reference at Levenshtein similarity >= threshold."""
    key = hashlib.sha256("\n".join(
        [f"threshold={threshold}", *(e["id"] for e in rows), "--", *(e["id"] for e in reference)]
    ).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"similarity_{key}.json" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        return set(json.loads(cache_file.read_text()))

    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    ref_text = [normalize(e["question"]) for e in reference]
    positions = {}
    for i, t in enumerate(ref_text):
        positions.setdefault(t, []).append(i)
    flagged = []
    for start in range(0, len(rows), 100):
        block = [normalize(e["question"]) for e in rows[start:start + 100]]
        sims = process.cdist(block, ref_text, scorer=Levenshtein.normalized_similarity,
                             score_cutoff=threshold, dtype=np.float32, workers=-1)
        for r, text in enumerate(block):
            sims[r, positions.get(text, [])] = 0.0           # ignore identical text (copies, and the row itself)
            if sims[r].max() >= threshold:
                flagged.append(rows[start + r]["id"])
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(sorted(flagged)))
    return set(flagged)


def carve_subsets(rows: list, sizes: dict, seed: int = SEED, flag_fn=None, pool_factor: float = 2):
    """Deterministically carve disjoint named subsets from rows. Returns (remaining_train, {name: rows}).

    flag_fn(candidate_rows) -> set of ids that are not allowed to enter any subset (optional)."""
    if any(not isinstance(k, int) or k <= 0 for k in sizes.values()):
        raise ValueError(f"subset sizes must be positive integers: {sizes}")
    total = sum(sizes.values())
    if total >= len(rows):
        raise ValueError(f"subsets need {total} rows but only {len(rows)} are available")
    ordered = sorted(rows, key=lambda e: e["id"])
    random.Random(seed).shuffle(ordered)
    limit, blocked = len(ordered), set()
    if flag_fn is not None:
        limit = min(len(ordered), math.ceil(total * pool_factor))
        blocked = set(flag_fn(ordered[:limit]))
    subsets, pos = {}, 0
    for name, k in sizes.items():
        picked = []
        while len(picked) < k:
            if pos >= limit:
                raise ValueError(f"only {len(picked)} eligible rows for {name} in a pool of {limit}; raise pool_factor")
            if ordered[pos]["id"] not in blocked:
                picked.append(ordered[pos])
            pos += 1
        subsets[name] = picked
    chosen = {e["id"] for part in subsets.values() for e in part}
    return [e for e in ordered if e["id"] not in chosen], subsets


def global_dedupe(pools: dict):
    """Two-tier cross-split pass: leak_key protects every eval split, exact_key cleans train."""
    assert SPLIT_PRIORITY[-1] == "train", "train must be processed after every eval split"
    present = {s for parts in pools.values() for s in parts}
    assert present == set(SPLIT_PRIORITY), f"split names {sorted(present)} do not match SPLIT_PRIORITY"
    eval_seen, train_seen = set(), set()
    final = {name: {} for name in pools}
    removed = Counter({f"{n}/{s}/cross": 0 for n in pools for s in pools[n]})
    removed.update({f"{n}/train/cross_dup": 0 for n in pools})
    for split in SPLIT_PRIORITY:
        for name in DATASET_ORDER:
            if split not in pools[name]:
                continue
            kept = []
            for ex in pools[name][split]:
                lk = leak_key(ex)
                if lk in eval_seen:
                    removed[f"{name}/{split}/cross"] += 1
                    continue
                if split == "train":
                    ek = exact_key(ex)
                    if ek in train_seen:
                        removed[f"{name}/train/cross_dup"] += 1
                        continue
                    train_seen.add(ek)
                else:
                    eval_seen.add(lk)
                kept.append(ex)
            final[name][split] = kept
    return final, removed


def _environment() -> dict:
    """Versions that could in principle influence the result; stored in stats.json for comparison across machines."""
    def version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {"python": sys.version.split()[0], "datasets": version("datasets"), "rapidfuzz": version("rapidfuzz")}


def build_all(out_dir=None, verbose: bool = True, cache_dir=None) -> dict:
    """Build every split from scratch and write ids, records and stats to out_dir."""
    splits_dir = Path(out_dir) / "splits" if out_dir else SPLITS_DIR
    processed_dir = Path(out_dir) / "processed" if out_dir else PROCESSED_DIR
    cache_dir = Path(cache_dir or CACHE_DIR)
    splits_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = {"medmcqa": load_medmcqa_raw(), "medqa": load_medqa_raw()}
    raw_sizes = {n: {s: len(v) for s, v in parts.items()} for n, parts in raw.items()}

    removed, conflicts, pools, similarity_info = Counter(), {}, {}, {}
    for name, parts in raw.items():
        train, n_removed, n_conf = dedupe_within(parts["train"], exact_key)
        removed[f"{name}/train/within"], conflicts[f"{name}/train"] = n_removed, n_conf
        test, n_removed, n_conf = dedupe_within(parts["test"], leak_key)
        removed[f"{name}/test/within"], conflicts[f"{name}/test"] = n_removed, n_conf
        flt = SIMILARITY_FILTER.get(name)
        if flt:
            info = similarity_info[name] = {"threshold": flt["threshold"]}

            def flag_fn(pool, reference=train, thr=flt["threshold"], info=info):
                info["pool"] = len(pool)
                info["flagged_in_pool"] = len(bad := near_duplicate_ids(pool, reference, thr, cache_dir))
                return bad

            train, subsets = carve_subsets(train, SPLIT_SIZES[name], SEED, flag_fn, flt["pool_factor"])
        else:
            train, subsets = carve_subsets(train, SPLIT_SIZES[name], SEED)
        pools[name] = {"train": train, **subsets, "test": test}

    final, cross_removed = global_dedupe(pools)
    removed.update(cross_removed)

    stats = {
        "config": json.loads(json.dumps(current_config())),
        "resolved_revisions": dict(RESOLVED_REVISIONS),
        "environment": _environment(),
        "raw_sizes": raw_sizes,
        "removed": dict(removed),
        "similarity_filter": similarity_info,
        "duplicate_answer_conflicts": conflicts,
        "sizes": {n: {s: len(final[n][s]) for s in split_names(n)} for n in final},
    }

    for name, parts in final.items():
        names = split_names(name)
        ids = {s: [e["id"] for e in parts[s]] for s in names}
        (splits_dir / f"{name}.json").write_text(json.dumps(ids))
        with open(processed_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for s in names:
                for e in parts[s]:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (splits_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    if verbose:
        print(json.dumps(stats, indent=2))
    return stats


def load_splits(name: str, data_dir=None, check_config: bool = True) -> dict:
    """Return {split name: [examples]} for "medmcqa" or "medqa" (see split_names)."""
    if name not in HF_REPOS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(HF_REPOS)}")
    splits_dir = Path(data_dir) / "splits" if data_dir else SPLITS_DIR
    processed_dir = Path(data_dir) / "processed" if data_dir else PROCESSED_DIR
    split_file = splits_dir / f"{name}.json"
    stats_file = splits_dir / "stats.json"
    if not split_file.exists() or not stats_file.exists():
        build_all(data_dir)

    if check_config:
        saved = json.loads(stats_file.read_text())["config"]
        now = json.loads(json.dumps(current_config()))
        if saved != now:
            raise RuntimeError(
                "Saved splits were built with a different configuration than the current code.\n"
                f"  saved: {saved}\n  now:   {now}\n"
                "Revert the setting, or run build_all() and share the new splits folder."
            )

    ids = json.loads(split_file.read_text())
    with open(processed_dir / f"{name}.jsonl", encoding="utf-8") as f:
        by_id = {e["id"]: e for e in map(json.loads, f)}
    return {s: [by_id[i] for i in ids[s]] for s in split_names(name)}
