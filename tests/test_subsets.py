import random

import pytest

from src.data.medqa_datasets import data
from src.data.medqa_datasets.data import carve_subsets, global_dedupe, leak_key

ALL = ("train", "val", "hp_dev", "calib", "thresh", "test")
LONG = "A 45-year-old man presents with chest pain radiating to the left arm and diaphoresis lasting an hour"


def make_rows(n):
    return [
        {"id": f"r{i:04d}", "source": "t",
         "question": f"Question number {i} about a distinct clinical topic and its management in adults",
         "options": {"A": f"a{i}", "B": f"b{i}"}, "answer": "A"}
        for i in range(n)
    ]


def ex(id_, q, opts):
    return {"id": id_, "source": "medmcqa", "question": q, "options": dict(zip("ABCD", opts)), "answer": "A"}


def pools(**kw):
    base = {n: {s: [] for s in ALL} for n in ("medmcqa", "medqa")}
    for key, rows in kw.items():
        name, split = key.split("__")
        base[name][split] = rows
    return base


def ids(rows):
    return [e["id"] for e in rows]


def test_sizes_and_disjoint():
    train, subs = carve_subsets(make_rows(100), {"a": 10, "b": 20, "c": 30})
    assert [len(subs[k]) for k in "abc"] == [10, 20, 30] and len(train) == 40
    everything = ids(train) + [i for k in "abc" for i in ids(subs[k])]
    assert len(everything) == len(set(everything)) == 100


def test_independent_of_input_order():
    rows = make_rows(100)
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    t1, s1 = carve_subsets(rows, {"a": 10, "b": 20})
    t2, s2 = carve_subsets(shuffled, {"a": 10, "b": 20})
    assert ids(t1) == ids(t2) and all(ids(s1[k]) == ids(s2[k]) for k in s1)


def test_deterministic_and_seed_matters():
    rows = make_rows(100)
    assert ids(carve_subsets(rows, {"a": 10}, seed=1)[1]["a"]) == ids(carve_subsets(rows, {"a": 10}, seed=1)[1]["a"])
    assert ids(carve_subsets(rows, {"a": 10}, seed=1)[1]["a"]) != ids(carve_subsets(rows, {"a": 10}, seed=2)[1]["a"])


def test_smaller_carve_is_prefix_of_larger():
    rows = make_rows(100)
    small = ids(carve_subsets(rows, {"v": 10})[1]["v"])
    big = ids(carve_subsets(rows, {"v": 25})[1]["v"])
    assert big[:10] == small
    first = ids(carve_subsets(rows, {"a": 10, "b": 15})[1]["a"])
    assert first == small


def test_rejects_bad_sizes():
    rows = make_rows(20)
    for bad in ({"a": 0}, {"a": -1}, {"a": 20}, {"a": 10, "b": 10}):
        with pytest.raises(ValueError):
            carve_subsets(rows, bad)


def test_question_in_two_subsets_and_train_is_kept_once():
    hp = ex("1", LONG, ["w", "x", "y", "z"])
    cal = ex("2", LONG, ["p", "q", "r", "s"])      # same long stem, different options
    thr = ex("3", LONG.upper(), ["a", "b", "c", "d"])
    trn = ex("4", LONG + "?", ["e", "f", "g", "h"])
    other = ex("5", "A completely different question about renal physiology and fluid balance", ["a", "b", "c", "d"])
    final, removed = global_dedupe(pools(medmcqa__hp_dev=[hp], medmcqa__calib=[cal],
                                         medmcqa__thresh=[thr], medmcqa__train=[trn, other]))
    parts = final["medmcqa"]
    assert ids(parts["hp_dev"]) == ["1"] and parts["calib"] == [] and parts["thresh"] == []
    assert ids(parts["train"]) == ["5"]
    assert removed["medmcqa/calib/cross"] == 1 and removed["medmcqa/thresh/cross"] == 1
    assert removed["medmcqa/train/cross"] == 1


def test_flagged_rows_never_enter_subsets():
    rows = make_rows(200)
    ordered = sorted(rows, key=lambda e: e["id"])
    random.Random(data.SEED).shuffle(ordered)
    blocked = {e["id"] for e in ordered[:15]}               # block the first rows of the shuffle
    train, subs = carve_subsets(rows, {"a": 10, "b": 20}, flag_fn=lambda pool: blocked, pool_factor=3)
    chosen = {e["id"] for part in subs.values() for e in part}
    assert len(subs["a"]) == 10 and len(subs["b"]) == 20 and not chosen & blocked
    assert blocked <= {e["id"] for e in train} and len(train) == 170


def test_pool_too_small_raises():
    with pytest.raises(ValueError):
        carve_subsets(make_rows(100), {"a": 10}, flag_fn=lambda pool: {e["id"] for e in pool}, pool_factor=2)


def test_flag_fn_receives_only_the_pool():
    seen = []
    carve_subsets(make_rows(100), {"a": 10}, flag_fn=lambda pool: seen.append(len(pool)) or set(), pool_factor=2)
    assert seen == [20]


def test_near_duplicate_ids(tmp_path):
    base = "A 45-year-old man presents with chest pain radiating to the left arm and diaphoresis lasting an hour"
    ref = [ex("r1", base, ["a", "b"]), ex("r2", "Completely unrelated question about renal tubular physiology in adults", ["a", "b"])]
    rows = [ex("n1", base.replace("hour", "hours"), ["a", "b"]),          # one character different: flagged
            ex("n2", base.upper() + "?", ["a", "b"]),                      # identical after normalize: not flagged
            ex("n3", "Which vitamin deficiency causes scurvy in sailors on long voyages", ["a", "b"])]
    first = data.near_duplicate_ids(rows, ref, 0.9, tmp_path)
    assert first == {"n1"}
    assert list(tmp_path.glob("similarity_*.json"))
    assert data.near_duplicate_ids(rows, ref, 0.9, tmp_path) == first      # served from the cache
    assert data.near_duplicate_ids(rows, ref, 0.9) == first                # recomputed without a cache


def test_priority_names_match_split_sizes():
    names = {"train", "test"} | {s for sizes in data.SPLIT_SIZES.values() for s in sizes}
    assert names == set(data.SPLIT_PRIORITY) and data.SPLIT_PRIORITY[-1] == "train"
