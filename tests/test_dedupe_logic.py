from medsplits import data
from medsplits.data import dedupe_within, exact_key, global_dedupe, leak_key

LONG = "A 45-year-old man presents with chest pain radiating to the left arm and diaphoresis lasting an hour"


def ex(id_, q, opts, ans="A", source="medmcqa"):
    return {"id": id_, "source": source, "question": q,
            "options": dict(zip("ABCDEF", opts)), "answer": ans}


def pools(**kw):
    base = {n: {"train": [], "val": [], "test": []} for n in ("medmcqa", "medqa")}
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


def test_train_must_be_last_in_priority(monkeypatch):
    monkeypatch.setattr(data, "SPLIT_PRIORITY", ("train", "val", "test"))
    try:
        global_dedupe(pools())
    except AssertionError:
        return
    raise AssertionError("expected the priority assertion to fire")
