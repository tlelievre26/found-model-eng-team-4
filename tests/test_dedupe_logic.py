import pytest

from src.data.medqa_datasets import data
from src.data.medqa_datasets.data import dedupe_within, exact_key, global_dedupe, leak_key

LONG = "A 45-year-old man presents with chest pain radiating to the left arm and diaphoresis lasting an hour"
ALL = ("train", "val", "hp_dev", "calib", "thresh", "test")


def ex(id_, q, opts, ans="A", source="medmcqa"):
    return {"id": id_, "source": source, "question": q,
            "options": dict(zip("ABCDEF", opts)), "answer": ans}


def pools(**kw):
    base = {n: {s: [] for s in ALL} for n in ("medmcqa", "medqa")}
    for key, rows in kw.items():
        name, split = key.split("__")
        base[name][split] = rows
    return base


def test_keys():
    a = ex("1", LONG, ["w", "x", "y", "z"])
    b = ex("2", LONG.upper() + "?", ["z", "y", "x", "w"])          # punctuation, case, option order
    c = ex("3", LONG, ["p", "q", "r", "s"])                         # same stem, different options
    assert exact_key(a) == exact_key(b) and exact_key(a) != exact_key(c)
    assert leak_key(a) == leak_key(b) == leak_key(c)                # long stem: options ignored
    s1, s2 = ex("4", "Which is true?", ["a", "b", "c", "d"]), ex("5", "Which is true?", ["e", "f", "g", "h"])
    assert leak_key(s1) != leak_key(s2)                             # short stem: options matter


def test_train_keeps_same_stem_with_different_options():
    a, c = ex("1", LONG, ["w", "x", "y", "z"]), ex("3", LONG, ["p", "q", "r", "s"])
    final, removed = global_dedupe(pools(medmcqa__train=[a, c]))
    assert [e["id"] for e in final["medmcqa"]["train"]] == ["1", "3"]
    assert sum(removed.values()) == 0


def test_reworded_eval_question_is_removed_from_train():
    train = [ex("1", LONG, ["p", "q", "r", "s"]), ex("2", "Another stem entirely about renal physiology and fluid balance in adults", ["a", "b", "c", "d"])]
    test = [ex("9", LONG, ["w", "x", "y", "z"])]                    # same stem, different options
    final, removed = global_dedupe(pools(medmcqa__train=train, medmcqa__test=test))
    assert [e["id"] for e in final["medmcqa"]["train"]] == ["2"]
    assert removed["medmcqa/train/cross"] == 1


def test_true_copies_across_train_sets_are_removed():
    a = ex("1", LONG, ["w", "x", "y", "z"], source="medqa")
    b = ex("2", LONG, ["z", "y", "x", "w"])
    final, removed = global_dedupe(pools(medqa__train=[a], medmcqa__train=[b]))
    assert len(final["medqa"]["train"]) == 1 and len(final["medmcqa"]["train"]) == 0
    assert removed["medmcqa/train/cross_dup"] == 1


def test_medqa_test_wins_over_medmcqa_test():
    q = ex("q", LONG, ["a", "b", "c", "d"], source="medqa")
    m = ex("m", LONG, ["a", "b", "c", "d"])
    final, removed = global_dedupe(pools(medqa__test=[q], medmcqa__test=[m]))
    assert len(final["medqa"]["test"]) == 1 and len(final["medmcqa"]["test"]) == 0
    assert removed["medmcqa/test/cross"] == 1


def test_dedupe_within_counts_conflicts():
    a = ex("1", LONG, ["w", "x", "y", "z"], "A")
    b = ex("2", LONG, ["w", "x", "y", "z"], "B")                    # exact copy, different answer
    kept, removed, conflicts = dedupe_within([a, b], exact_key)
    assert [e["id"] for e in kept] == ["1"] and removed == 1 and conflicts == 1


def test_dedupe_within_drop_conflicts():
    a = ex("1", LONG, ["w", "x", "y", "z"], "A")
    b = ex("2", LONG, ["w", "x", "y", "z"], "B")                    # exact copy, different answer: both dropped
    c = ex("3", LONG, ["w", "x", "y", "z"], "A")
    d = ex("4", "Another stem entirely about renal physiology and fluid balance in adults", ["a", "b", "c", "d"])
    e = ex("5", d["question"], ["d", "c", "b", "a"], "D")            # same answer text "a": a normal copy, first kept
    kept, removed, conflicts = dedupe_within([a, b, c, d, e], exact_key, drop_conflicts=True)
    assert [r["id"] for r in kept] == ["4"] and removed == 3 and conflicts == 1


def test_medmcqa_explanation_field(monkeypatch):
    row = {"id": "x", "question": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 1}
    fake = {"train": [{**row, "exp": "  Because.  "}, {**row, "id": "y", "exp": " "}, {**row, "id": "z", "exp": None}],
            "validation": []}
    monkeypatch.setattr(data, "_load_hf", lambda name: fake)
    assert [e["explanation"] for e in data.load_medmcqa_raw()["train"]] == ["Because.", None, None]


def test_load_splits_raises_instead_of_building(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "build_all", lambda *a, **k: pytest.fail("load_splits must not build"))
    with pytest.raises(FileNotFoundError, match="build_all"):
        data.load_splits("medqa", data_dir=tmp_path)


def test_train_must_be_last_in_priority(monkeypatch):
    monkeypatch.setattr(data, "SPLIT_PRIORITY", ("train", "val", "test"))
    try:
        global_dedupe(pools())
    except AssertionError:
        return
    raise AssertionError("expected the priority assertion to fire")


def test_medabstain_questions_excluded_before_carving(tmp_path, monkeypatch):
    """Offline build on made-up rows: a MedAbstain test question leaves train before carving, and test; options ignored."""
    import hashlib
    import json

    from src.data.medabstain import FILES, UPSTREAM_COMMIT

    abst = tmp_path / "medabstain"
    abst.mkdir()
    protected = "Which made-up drug is protected by the MedAbstain exclusion in this tiny offline build?"
    (abst / FILES["test"]["NA"]).write_text(json.dumps([{"id": 1, "source": "x", "question": protected,
                                                         "choices": dict(zip("ABCDE", "vwxyz")), "answer": "A"}]))
    for name in {f for files in FILES.values() for f in files.values()} - {FILES["test"]["NA"]}:
        (abst / name).write_text("[]")
    with pytest.raises(RuntimeError, match="MedAbstain"):     # no provenance.json: the build refuses to run
        data.build_all(tmp_path / "out", verbose=False, medabstain_dir=abst)
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in abst.iterdir()}
    (abst / "provenance.json").write_text(json.dumps({"commit": UPSTREAM_COMMIT, "sha256": sha}))

    def rows(split, n, source):
        return [ex(f"{source}-{split}-{i}", f"{LONG} in {source} {split} case {i}", ["a", "b", "c", "d"], source=source)
                for i in range(n)]
    medqa = {"train": rows("train", 20, "medqa"), "test": rows("test", 5, "medqa")}
    medqa["train"].append(ex("medqa-train-hit", protected.upper(), ["p", "q", "r", "s"], source="medqa"))  # 4 other options
    medqa["test"].append(ex("medqa-test-hit", protected, ["p", "q", "r", "s"], source="medqa"))
    monkeypatch.setattr(data, "load_medqa_raw", lambda: medqa)
    monkeypatch.setattr(data, "load_medmcqa_raw", lambda: {"train": rows("train", 30, "medmcqa"), "test": []})
    monkeypatch.setattr(data, "SPLIT_SIZES", {"medmcqa": {"hp_dev": 2, "calib": 2, "thresh": 2}, "medqa": {"val": 10}})
    monkeypatch.setattr(data, "SIMILARITY_FILTER", {})
    carved_from = []
    carve = data.carve_subsets
    monkeypatch.setattr(data, "carve_subsets", lambda rows, *a, **k: carved_from.extend(e["id"] for e in rows) or carve(rows, *a, **k))

    stats = data.build_all(tmp_path / "out", verbose=False, cache_dir=tmp_path / "cache", medabstain_dir=abst)
    assert stats["medabstain_excluded_ids"]["medqa/train"] == ["medqa-train-hit"]
    assert stats["medabstain_excluded_ids"]["medqa/test"] == ["medqa-test-hit"]
    assert "medqa-train-hit" not in carved_from and len(carved_from) == 50
    assert stats["sizes"]["medqa"] == {"train": 10, "val": 10, "test": 5}
