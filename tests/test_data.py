import pytest

from medsplits import data
from medsplits.data import LETTERS, SPLITS, build_all, exact_key, leak_key, load_medqa_raw, load_splits

NAMES = ["medmcqa", "medqa"]


@pytest.fixture(scope="module")
def splits():
    return {n: load_splits(n) for n in NAMES}


def test_no_eval_question_in_any_train(splits):
    train_keys = {leak_key(e) for n in NAMES for e in splits[n]["train"]}
    for n in NAMES:
        for sp in ("val", "test"):
            leaked = [e["id"] for e in splits[n][sp] if leak_key(e) in train_keys]
            assert not leaked, (n, sp, leaked[:5])


def test_eval_sets_have_no_duplicates(splits):
    keys = [leak_key(e) for n in NAMES for sp in ("val", "test") for e in splits[n][sp]]
    assert len(keys) == len(set(keys))


def test_train_has_no_exact_duplicates(splits):
    keys = [exact_key(e) for n in NAMES for e in splits[n]["train"]]
    assert len(keys) == len(set(keys))


def test_ids_unique(splits):
    ids = [e["id"] for n in NAMES for sp in SPLITS for e in splits[n][sp]]
    assert len(ids) == len(set(ids))


def test_format_all_rows(splits):
    for n in NAMES:
        for sp in SPLITS:
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


def test_deterministic(tmp_path):
    build_all(tmp_path / "a", verbose=False)
    build_all(tmp_path / "b", verbose=False)
    for n in NAMES:
        a = (tmp_path / "a" / "splits" / f"{n}.json").read_text()
        b = (tmp_path / "b" / "splits" / f"{n}.json").read_text()
        assert a == b


def test_config_guard(monkeypatch):
    load_splits("medqa")  # fine with the current config
    monkeypatch.setattr(data, "SEED", data.SEED + 1)
    with pytest.raises(RuntimeError):
        load_splits("medqa")
