"""MedAbstain JSON loading and validation."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable

from src.utils import paths

UPSTREAM_URL = "https://github.com/sravanthi6m/MedAbstain"
UPSTREAM_COMMIT = "091e5c22111fffeb51c0c2e69b65d0a21a1e4164"
ABSTENTION_TEXT = "I am unable to answer this question for certain reasons."
FILES = {
    "test": {
        "NA": "medqa_1_test_noabst.json",
        "A": "medqa_1_test_randabst.json",
        "NAP": "perturbed_medqa_1_test_noabst.json",
        "AP": "perturbed_medqa_1_test_randabst.json",
    },
    "val": {
        "NA": "val_100_medqa_1_noabst.json",
        "A": "val_100_medqa_1_randabst.json",
    },
}


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _to_common_format(raw: dict, variant: str, split: str, filename: str) -> dict:
    """Convert a source record to the common schema, retaining its metadata."""
    return {
        "id": f"medabstain-{split}-{raw['id']}",  # upstream ids are ints that repeat between test and val
        "source": "medabstain",  # dataset name, as in the common format; upstream value stays in metadata.raw
        "question": raw["question"],
        "options": dict(raw["choices"]),
        "answer": raw["answer"],
        "original_answer": raw["answer"],
        "evaluation_target": raw["answer"],
        "abstain_option": next(
            (letter for letter, text in raw["choices"].items()
             if _normalized_text(text) == _normalized_text(ABSTENTION_TEXT)),
            None,
        ),
        "is_answerable": None if variant in ("NAP", "AP") else True,
        "is_perturbed": variant in ("NAP", "AP"),
        "metadata": {
            "dataset": "medabstain",
            "variant": variant,
            "split": split,
            "filename": filename,
            "reference_upstream_commit": UPSTREAM_COMMIT,
            "evaluation_protocol": "upstream_original_answer",
            "raw": deepcopy(raw),
        },
    }


def _validate(raw: object, variant: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError("expected an object")
    required = {"id", "source", "question", "choices", "answer"}
    if required - raw.keys():
        raise ValueError(f"missing required fields: {sorted(required - raw.keys())}")
    if (isinstance(raw["id"], bool) or not isinstance(raw["id"], (str, int))
            or (isinstance(raw["id"], str) and not raw["id"].strip())):
        raise ValueError("id must be a nonempty string or integer")
    for field in ("source", "question", "answer"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    choices = raw["choices"]
    if not isinstance(choices, dict) or len(choices) < 2:
        raise ValueError("choices must be a dictionary with at least two options")
    for letter, text in choices.items():
        if not isinstance(letter, str) or len(letter) != 1 or not "A" <= letter <= "Z":
            raise ValueError(f"invalid choice letter: {letter!r}; expected A-Z")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"choice {letter} must contain nonempty text")
    if raw["answer"] not in choices:
        raise ValueError(f"answer {raw['answer']!r} is not a choice letter")
    abstentions = [k for k, v in choices.items()
                   if _normalized_text(v) == _normalized_text(ABSTENTION_TEXT)]
    expected = int(variant in ("A", "AP"))
    if len(abstentions) != expected:
        raise ValueError(f"variant {variant} requires {expected} abstention option(s); "
                         f"found {len(abstentions)}")
    if raw["answer"] in abstentions:
        raise ValueError("source answer must identify an original option, not abstention")
    if "original_question" in raw and (
        not isinstance(raw["original_question"], str) or not raw["original_question"].strip()
    ):
        raise ValueError("original_question must be nonempty text when present")


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key!r}")
        obj[key] = value
    return obj


def load_medabstain(variant: str, split: str = "test", data_dir=None) -> list[dict]:
    """Load a MedAbstain variant from local JSON files.

    data_dir defaults to paths.MEDABSTAIN_DIR. Test variants are NA/A/NAP/AP;
    validation files are available for NA/A.
    """
    if variant not in FILES["test"]:
        raise ValueError(f"Invalid variant {variant!r}; choose NA, A, NAP, or AP")
    if split not in FILES:
        raise ValueError(f"Invalid split {split!r}; choose 'test' or 'val'")
    if variant not in FILES[split]:
        raise ValueError(f"No upstream {split} file exists for variant {variant}")
    filename = FILES[split][variant]
    directory = Path(data_dir).expanduser() if data_dir is not None else paths.MEDABSTAIN_DIR
    path = directory / filename
    try:
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream, object_pairs_hook=_unique_object)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"MedAbstain file not found: {path}. data_dir must contain the downloaded JSON files."
        ) from exc
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid MedAbstain JSON in {path}: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of records")
    result, seen = [], set()
    for index, raw in enumerate(records):
        try:
            _validate(raw, variant)
            if raw["id"] in seen:
                raise ValueError(f"duplicate id: {raw['id']!r}")
            seen.add(raw["id"])
            result.append(_to_common_format(raw, variant, split, filename))
        except ValueError as exc:
            raise ValueError(f"{path}: record {index}: {exc}") from exc
    return result


def normalize_question(text: str) -> str:
    """Lowercase text, replace punctuation with spaces, and collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def check_medqa_overlap(
    examples: Iterable[dict], medqa_splits: dict,
    normalizer: Callable[[str], str] = normalize_question,
) -> dict:
    """Find normalized question matches in MedQA train and validation records.

    Both displayed and original question text is compared. Choices are ignored,
    so matches involving generic short questions may need review.
    """
    examples = list(examples)
    result = {}
    for split in ("train", "val"):
        if split not in medqa_splits:
            raise ValueError(f"MedQA split {split!r} is missing; overlap check pending")
        questions = {normalizer(row["question"]) for row in medqa_splits[split]}
        ids = []
        for row in examples:
            texts = [row["question"]]
            original = row.get("metadata", {}).get("raw", {}).get("original_question")
            if original:
                texts.append(original)
            if any(normalizer(text) in questions for text in texts):
                ids.append(row["id"])
        result[split] = {"overlapping_examples": len(ids), "ids": ids}
    return result
