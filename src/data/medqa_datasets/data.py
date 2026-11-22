"""Load MedMCQA and MedQA, convert both to one format, deduplicate, and build fixed splits.

Entry point: load_splits("medmcqa" | "medqa") -> {"train": [...], "val": [...], "test": [...]}
Example format: {"id", "source", "question", "options": {"A": ..., "B": ...}, "answer": "B"}
"""
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path

# -------------------------------------------------------------------------------------------------
# Shared configuration. Everything here defines the splits. If you change any of it, run build_all()
# again and re-share the splits folder: load_splits refuses to run on splits built with another config.
# -------------------------------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/shared/data"))  # set the env var BEFORE importing
SEED = 42
VAL_SIZE = {"medmcqa": 4000, "medqa": 1000}
SHORT_STEM_WORDS = 10  # leak_key: stems with fewer words must also match on options
DEDUPE_SCHEME = "two-tier-v1"  # exact_key inside train, leak_key against val/test; bump if keys change
HF_REPOS = {"medmcqa": "openlifescienceai/medmcqa", "medqa": "GBaker/MedQA-USMLE-4-options"}
HF_REVISIONS = {  # pinned dataset commits (None would mean "latest")
    "medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    "medqa": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
}
SPLIT_PRIORITY = ("test", "val", "train")  # when a question repeats, the earliest split keeps it
DATASET_ORDER = ("medqa", "medmcqa")  # tie-break inside a split: MedQA first (protects MedAbstain)
SPLITS = ("train", "val", "test")
LETTERS = string.ascii_uppercase  # option labels by position, so any option count up to 26 works

RESOLVED_REVISIONS = {}  # filled by _load_hf: the dataset commits actually used
_PUNCT = re.compile(r"[^\w\s]")


def current_config() -> dict:
    """Snapshot of every setting that changes the splits (JSON-serializable)."""
    return {
        "seed": SEED,
        "val_size": VAL_SIZE,
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
    ds = load_dataset(HF_REPOS[name], revision=rev, cache_dir=str(DATA_DIR / "hf_cache"))
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


def carve_val(rows: list, k: int, seed: int = SEED):
    """Deterministically split rows into (train, val) with |val| = k."""
    if not 0 <= k < len(rows):
        raise ValueError(f"val size {k} is not valid for {len(rows)} rows")
    ordered = sorted(rows, key=lambda e: e["id"])
    random.Random(seed).shuffle(ordered)
    return ordered[k:], ordered[:k]


def global_dedupe(pools: dict):
    """Two-tier cross-split pass: leak_key protects val/test, exact_key cleans train."""
    assert SPLIT_PRIORITY[-1] == "train", "train must be processed after every eval split"
    eval_seen, train_seen = set(), set()
    final = {name: {} for name in pools}
    removed = Counter({f"{n}/{s}/cross": 0 for n in pools for s in SPLITS})
    removed.update({f"{n}/train/cross_dup": 0 for n in pools})
    for split in SPLIT_PRIORITY:
        for name in DATASET_ORDER:
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


def build_all(out_dir=None, verbose: bool = True) -> dict:
    """Build every split from scratch and write ids, records and stats to out_dir."""
    out_dir = Path(out_dir or DATA_DIR)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)

    raw = {"medmcqa": load_medmcqa_raw(), "medqa": load_medqa_raw()}
    raw_sizes = {n: {s: len(v) for s, v in parts.items()} for n, parts in raw.items()}

    removed, conflicts, pools = Counter(), {}, {}
    for name, parts in raw.items():
        train, n_removed, n_conf = dedupe_within(parts["train"], exact_key)
        removed[f"{name}/train/within"], conflicts[f"{name}/train"] = n_removed, n_conf
        test, n_removed, n_conf = dedupe_within(parts["test"], leak_key)
        removed[f"{name}/test/within"], conflicts[f"{name}/test"] = n_removed, n_conf
        train, val = carve_val(train, VAL_SIZE[name], SEED)
        pools[name] = {"train": train, "val": val, "test": test}

    final, cross_removed = global_dedupe(pools)
    removed.update(cross_removed)

    stats = {
        "config": json.loads(json.dumps(current_config())),
        "resolved_revisions": dict(RESOLVED_REVISIONS),
        "raw_sizes": raw_sizes,
        "removed": dict(removed),
        "duplicate_answer_conflicts": conflicts,
        "sizes": {n: {s: len(final[n][s]) for s in SPLITS} for n in final},
    }

    for name, parts in final.items():
        ids = {s: [e["id"] for e in parts[s]] for s in SPLITS}
        (out_dir / "splits" / f"{name}.json").write_text(json.dumps(ids))
        with open(out_dir / "processed" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for s in SPLITS:
                for e in parts[s]:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out_dir / "splits" / "stats.json").write_text(json.dumps(stats, indent=2))
    if verbose:
        print(json.dumps(stats, indent=2))
    return stats


def load_splits(name: str, data_dir=None, check_config: bool = True) -> dict:
    """Return {"train": [...], "val": [...], "test": [...]} for "medmcqa" or "medqa"."""
    if name not in HF_REPOS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(HF_REPOS)}")
    data_dir = Path(data_dir or DATA_DIR)
    split_file = data_dir / "splits" / f"{name}.json"
    stats_file = data_dir / "splits" / "stats.json"
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
    with open(data_dir / "processed" / f"{name}.jsonl", encoding="utf-8") as f:
        by_id = {e["id"]: e for e in map(json.loads, f)}
    return {s: [by_id[i] for i in ids[s]] for s in SPLITS}
