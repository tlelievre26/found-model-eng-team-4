import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.medqa_datasets import data
from src.data.medqa_datasets.data import LETTERS, SPLIT_SIZES, exact_key, leak_key, load_medqa_raw, load_splits, split_names

NAMES = ["medmcqa", "medqa"]
ROOT = Path(__file__).resolve().parents[1]  # the repo root (contains src/)


def eval_names(name):
    return [s for s in split_names(name) if s != "train"]


@pytest.fixture(scope="module")
def splits():
    return {n: load_splits(n) for n in NAMES}


def test_split_names(splits):
    assert list(splits["medmcqa"]) == ["train", "hp_dev", "calib", "thresh", "test"]
    assert list(splits["medqa"]) == ["train", "val", "test"]


def test_medmcqa_splits_have_no_questions_in_common(splits):
    parts = splits["medmcqa"]
    ids = {s: {e["id"] for e in rows} for s, rows in parts.items()}
    keys = {s: {leak_key(e) for e in rows} for s, rows in parts.items()}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not ids[a] & ids[b], f"shared ids between {a} and {b}"
            assert not keys[a] & keys[b], f"shared questions between {a} and {b}"


def test_no_eval_question_in_any_train(splits):
    train_keys = {leak_key(e) for n in NAMES for e in splits[n]["train"]}
    for n in NAMES:
        for sp in eval_names(n):
            leaked = [e["id"] for e in splits[n][sp] if leak_key(e) in train_keys]
            assert not leaked, (n, sp, leaked[:5])


def test_eval_sets_have_no_duplicates(splits):
    keys = [leak_key(e) for n in NAMES for sp in eval_names(n) for e in splits[n][sp]]
    assert len(keys) == len(set(keys))


def test_train_has_no_exact_duplicates(splits):
    keys = [exact_key(e) for n in NAMES for e in splits[n]["train"]]
    assert len(keys) == len(set(keys))


def test_subset_sizes_close_to_nominal(splits):
    for n, sizes in SPLIT_SIZES.items():
        for sp, nominal in sizes.items():
            actual = len(splits[n][sp])
            assert 0.98 * nominal <= actual <= nominal, (n, sp, actual, nominal)


def test_subsets_have_no_near_copy_in_train(splits):
    """Sampled check of the similarity filter: no hp_dev/calib/thresh row has a different train question at >= the filter threshold."""
    import random

    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    train_text = [data.normalize(e["question"]) for e in splits["medmcqa"]["train"]]
    identical = {}
    for i, t in enumerate(train_text):
        identical.setdefault(t, []).append(i)
    thr = data.SIMILARITY_FILTER["medmcqa"]["threshold"]
    rng = random.Random(0)
    for sp in ("hp_dev", "calib", "thresh"):
        sample = [data.normalize(e["question"]) for e in rng.sample(splits["medmcqa"][sp], min(300, len(splits["medmcqa"][sp])))]
        sims = process.cdist(sample, train_text, scorer=Levenshtein.normalized_similarity,
                             score_cutoff=thr, dtype=np.float32, workers=-1)
        for r, t in enumerate(sample):
            sims[r, identical.get(t, [])] = 0.0
        assert sims.max() < thr, (sp, float(sims.max()))


def test_ids_unique(splits):
    ids = [e["id"] for n in NAMES for sp in split_names(n) for e in splits[n][sp]]
    assert len(ids) == len(set(ids))


def test_format_all_rows(splits):
    for n in NAMES:
        for sp in split_names(n):
            for e in splits[n][sp]:
                letters = list(e["options"])
                assert letters == list(LETTERS[: len(letters)]), e["id"]
                assert len(letters) >= 2, e["id"]
                assert e["answer"] in e["options"], e["id"]
                assert e["source"] == n and e["id"].startswith(n + "-"), e["id"]
                assert e["question"].strip(), e["id"]


def test_medqa_test_is_untouched(splits):
    n_raw = len(load_medqa_raw()["test"])
    assert {e["id"] for e in splits["medqa"]["test"]} == {f"medqa-test-{i}" for i in range(n_raw)}


def _build_in_fresh_process(out_dir, hash_seed):
    """Build in a new Python process with its own EMPTY similarity cache, so the search is really recomputed."""
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "FME_SHARED_DIR": str(data.DATA_DIR.parent)}
    code = "import sys; from src.data.medqa_datasets.data import build_all; build_all(sys.argv[1], verbose=False, cache_dir=sys.argv[2])"
    subprocess.run([sys.executable, "-c", code, str(out_dir), str(out_dir / "cache")], check=True, env=env, cwd=ROOT)


def test_deterministic_in_fresh_processes(tmp_path, splits):
    _build_in_fresh_process(tmp_path / "a", 0)
    _build_in_fresh_process(tmp_path / "b", 1)
    for n in NAMES:
        a = (tmp_path / "a" / "splits" / f"{n}.json").read_text()
        b = (tmp_path / "b" / "splits" / f"{n}.json").read_text()
        saved = (data.DATA_DIR / "splits" / f"{n}.json").read_text()
        assert a == b == saved, n


def test_config_guard(monkeypatch):
    load_splits("medqa")  # fine with the current config
    monkeypatch.setattr(data, "SEED", data.SEED + 1)
    with pytest.raises(RuntimeError):
        load_splits("medqa")
