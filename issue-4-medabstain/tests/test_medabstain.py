import json
import os
import subprocess
import sys
from pathlib import Path
from copy import deepcopy

import pytest

from src.data.medabstain import ABSTENTION_TEXT, FILES, check_medqa_overlap, load_medabstain
from src.utils import paths
from src.data.medabstain import normalize_question
from scripts.download_medabstain import _install_data, download_medabstain


def record(variant="NA"):
    row = {"id": 1, "source": "synthetic", "question": "Which option?",
           "choices": {"A": "First", "B": "Second"}, "answer": "B"}
    if variant in ("A", "AP"):
        row["choices"] = {"A": "  I AM unable  to answer this\nquestion for certain reasons. ",
                          "B": "First", "C": "Second"}
        row["answer"] = "C"
    if variant in ("NAP", "AP"):
        row.update(original_question="Original complete question?", meta={"removed": "context"})
    return row


def write(tmp_path, variant="NA", split="test", rows=None):
    path = tmp_path / FILES[split][variant]
    path.write_text(json.dumps([record(variant)] if rows is None else rows))
    return path


@pytest.mark.parametrize("split,variant", [(s, v) for s, vs in FILES.items() for v in vs])
def test_mapping_and_schema(tmp_path, split, variant):
    path = write(tmp_path, variant, split)
    raw = json.loads(path.read_text())[0]
    row, = load_medabstain(variant, split, tmp_path)
    for key in ("id", "source", "question", "answer"):
        assert row[key] == raw[key]
    assert row["options"] == raw["choices"]
    assert row["original_answer"] == row["evaluation_target"] == raw["answer"]
    assert row["abstain_option"] == ("A" if variant in ("A", "AP") else None)
    assert row["is_answerable"] is (None if variant in ("NAP", "AP") else True)
    assert row["is_perturbed"] == (variant in ("NAP", "AP"))
    assert row["metadata"]["raw"] == raw
    assert row["metadata"]["variant"] == variant
    assert row["metadata"]["split"] == split
    assert row["metadata"]["filename"] == path.name
    row["options"]["B"] = "changed"
    assert row["metadata"]["raw"]["choices"] == raw["choices"]


def test_default_path(monkeypatch, tmp_path):
    write(tmp_path)
    monkeypatch.setattr(paths, "MEDABSTAIN_DIR", tmp_path)
    assert len(load_medabstain("NA")) == 1


@pytest.mark.parametrize("variant,split,message", [
    ("bad", "test", "Invalid variant"), ("NA", "train", "Invalid split"),
    ("NAP", "val", "No upstream"), ("AP", "val", "No upstream"),
])
def test_invalid_selection(tmp_path, variant, split, message):
    with pytest.raises(ValueError, match=message):
        load_medabstain(variant, split, tmp_path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="medqa_1_test_noabst.json.*JSON files"):
        load_medabstain("NA", data_dir=tmp_path)


@pytest.mark.parametrize("field", ["id", "source", "question", "choices", "answer"])
def test_missing_field(tmp_path, field):
    row = record()
    del row[field]
    write(tmp_path, rows=[row])
    with pytest.raises(ValueError, match=f"record 0: missing required fields.*{field}"):
        load_medabstain("NA", data_dir=tmp_path)


@pytest.mark.parametrize("update,message", [
    ({"id": None}, "id must"), ({"id": True}, "id must"),
    ({"source": ""}, "source must"), ({"question": []}, "question must"),
    ({"answer": None}, "answer must"), ({"answer": "Z"}, "not a choice"),
    ({"choices": []}, "choices must"), ({"choices": {}}, "choices must"),
    ({"choices": {"AA": "One", "B": "Two"}}, "invalid choice"),
    ({"choices": {"A": "One", "B": None}}, "nonempty text"),
    ({"choices": {"A": "One", "B": ABSTENTION_TEXT}}, "requires 0"),
    ({"original_question": ""}, "original_question must"),
])
def test_malformed_examples(tmp_path, update, message):
    row = record()
    row.update(update)
    write(tmp_path, rows=[row])
    with pytest.raises(ValueError, match=f"record 0:.*{message}"):
        load_medabstain("NA", data_dir=tmp_path)


@pytest.mark.parametrize("rows", [[None], [record(), record()]])
def test_invalid_record_or_duplicate_id(tmp_path, rows):
    write(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="record"):
        load_medabstain("NA", data_dir=tmp_path)


@pytest.mark.parametrize("text", ['{', '{}', '[{"id":1,"id":2}]'])
def test_invalid_json(tmp_path, text):
    write(tmp_path).write_text(text)
    with pytest.raises(ValueError):
        load_medabstain("NA", data_dir=tmp_path)


@pytest.mark.parametrize("count", [0, 2])
def test_exactly_one_abstention(tmp_path, count):
    row = record("A")
    row["choices"]["A" if count == 0 else "D"] = "Ordinary" if count == 0 else ABSTENTION_TEXT
    write(tmp_path, "A", rows=[row])
    with pytest.raises(ValueError, match="requires 1"):
        load_medabstain("A", data_dir=tmp_path)


def test_answer_cannot_be_abstention(tmp_path):
    row = record("A")
    row["answer"] = "A"
    write(tmp_path, "A", rows=[row])
    with pytest.raises(ValueError, match="original option"):
        load_medabstain("A", data_dir=tmp_path)


def test_overlap_and_input_preservation(tmp_path):
    write(tmp_path, "NAP")
    row, = load_medabstain("NAP", data_dir=tmp_path)
    splits = {"train": [{"question": " ORIGINAL complete QUESTION!!! ", "options": row["options"]}],
              "val": [{"question": "which option", "options": row["options"]}]}
    before = deepcopy(splits)
    assert check_medqa_overlap([row], splits) == {
        "train": {"overlapping_examples": 1, "ids": [1]},
        "val": {"overlapping_examples": 1, "ids": [1]},
    }
    assert splits == before
    assert check_medqa_overlap([row], {"train": [], "val": []})["train"]["overlapping_examples"] == 0
    with pytest.raises(ValueError, match="pending"):
        check_medqa_overlap([row], {"train": []})
    assert check_medqa_overlap([row], splits, normalizer=lambda _: "same")["train"]["overlapping_examples"] == 1


# Download provenance and protection


def test_install_provenance_idempotence_and_no_overwrite(tmp_path):
    checkout = tmp_path / "upstream"
    (checkout / "data").mkdir(parents=True)
    (checkout / "data" / "fixture.json").write_text("[]")
    (checkout / "LICENSE").write_text("synthetic license fixture")
    target = tmp_path / "shared" / "medabstain"
    manifest = _install_data(checkout, target, "fixture-commit")
    assert json.loads((target / "provenance.json").read_text()) == manifest
    assert manifest["commit"] == "fixture-commit"
    assert len(manifest["sha256"]["fixture.json"]) == 64
    assert _install_data(checkout, target, "fixture-commit") == manifest
    (target / "fixture.json").write_text("user content")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _install_data(checkout, target, "fixture-commit")
    assert (target / "fixture.json").read_text() == "user content"


def test_refuses_repository_destination():
    with pytest.raises(ValueError, match="outside the source repository"):
        download_medabstain(Path(__file__).resolve().parents[1] / "data" / "medabstain")


# Real-data integration checks


def data_directory():
    directory = Path(os.environ.get("MEDABSTAIN_DATA_DIR", paths.MEDABSTAIN_DIR))
    if not directory.is_dir():
        pytest.skip("MedAbstain data directory not found")
    return directory


@pytest.mark.parametrize("split,variant", [(s, v) for s, vs in FILES.items() for v in vs])
def test_published_data(split, variant):
    rows = load_medabstain(variant, split, data_directory())
    assert len(rows) == (1007 if split == "test" else 100)
    assert len({r["id"] for r in rows}) == len(rows)
    expected = variant in ("A", "AP")
    assert all((r["abstain_option"] is not None) == expected for r in rows)
    assert all(r["answer"] in r["options"] for r in rows)
    assert all(r["answer"] == r["original_answer"] == r["evaluation_target"] for r in rows)
    assert len({frozenset(r) for r in rows}) == 1


def test_medqa_team_splits_overlap():
    export = os.environ.get("MEDQA_SPLITS_JSON")
    if not export:
        pytest.skip("MEDQA_SPLITS_JSON is not set")
    splits = json.loads(Path(export).read_text())
    for split, variants in FILES.items():
        for variant in variants:
            rows = load_medabstain(variant, split, data_directory())
            result = check_medqa_overlap(rows, splits)
            assert all(r["overlapping_examples"] == 0
                       for r in result.values()), (split, variant, result)


# Import dependencies


def test_loader_import_and_use_without_med_splits(tmp_path):
    record = {"id": 1, "source": "synthetic", "question": "Synthetic standalone fixture?",
              "choices": {"A": "One", "B": "Two"}, "answer": "A"}
    (tmp_path / "medqa_1_test_noabst.json").write_text(json.dumps([record]))
    # The import hook checks for a dependency on medsplits.
    program = '''
import builtins, sys
original_import = builtins.__import__
def independent_import(name, *args, **kwargs):
    if name == "medsplits" or name.startswith("medsplits."):
        raise AssertionError("Unexpected medsplits import")
    return original_import(name, *args, **kwargs)
builtins.__import__ = independent_import
from src.data.medabstain import load_medabstain, check_medqa_overlap
row, = load_medabstain("NA", data_dir=sys.argv[1])
assert row["answer"] == "A"
result = check_medqa_overlap([row], {"train": [], "val": []})
assert result["train"]["overlapping_examples"] == 0
from scripts.verify_medabstain import verify
'''
    subprocess.run([sys.executable, "-c", program, str(tmp_path)], check=True)


def test_standalone_normalization_matches_observed_team_convention():
    assert normalize_question(" Kidney—disease; A/B  ") == "kidney disease a b"


def test_question_only_protocol_does_not_require_matching_options():
    row = {"id": 1, "question": "Which is true?", "options": {"A": "One", "B": "Two"}}
    result = check_medqa_overlap([row], {"train": [{"question": "WHICH is true!"}], "val": []})
    assert result["train"]["overlapping_examples"] == 1
