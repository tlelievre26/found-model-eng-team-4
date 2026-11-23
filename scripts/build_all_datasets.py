"""Build every dataset once, validate the result and record it in $FME_SHARED_DIR/data/manifest.json.

Steps: download MedAbstain (a no-op when the same verified copy is present), build the MedMCQA/MedQA splits
(build_all, which excludes the MedAbstain test questions), validate, and write manifest.json.

    python -m scripts.build_all_datasets                  # build into $FME_SHARED_DIR/data (refuses to overwrite)
    python -m scripts.build_all_datasets --verify-only    # check an existing or copied build against its manifest

Run the build once, on the team VM, from a clean main, into /shared/data; everyone else uses or copies that build
and checks it with --verify-only (see the root README). The manifest never contains MedAbstain text.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.download_medabstain import download_medabstain
from scripts.verify_medabstain import verify as verify_medabstain
from src.data.medqa_datasets import data
from src.utils import paths

# Disabled until the unify-datasets branch is verified; then True, so builds come only from a clean main.
REQUIRE_CLEAN_MAIN = False
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(paths.DEFAULT_SHARED_DIR) / "data"
MANIFEST = "manifest.json"
NAMES = ("medmcqa", "medqa")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def git_state() -> dict:
    """Commit, branch and whether the working tree is clean (recorded in every manifest)."""
    try:
        return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "clean": _git("status", "--porcelain") == ""}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "branch": None, "clean": None, "error": str(exc)}


def require_clean_main() -> None:
    """Refuse to run unless on main, with a clean tree, and HEAD equal to origin/main (after fetching it)."""
    state = git_state()
    problems = []
    if state["branch"] != "main":
        problems.append(f"the current branch is {state['branch']!r}, not 'main'")
    if not state["clean"]:
        problems.append("the working tree is not clean (git status --porcelain is not empty)")
    try:
        _git("fetch", "-q", "origin", "main")
        origin = _git("rev-parse", "origin/main")
    except (OSError, subprocess.CalledProcessError) as exc:
        problems.append(f"could not fetch origin/main ({exc})")
    else:
        if state["commit"] != origin:
            problems.append(f"HEAD {state['commit']} is not origin/main {origin}")
    if problems:
        raise SystemExit("Refusing to run: the shared datasets are built and verified only from a clean main that "
                         "equals origin/main.\n  - " + "\n  - ".join(problems) + "\nFor a test build into another "
                         "folder, set FME_SHARED_DIR to it and pass --allow-non-main.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_checksums(data_dir: Path) -> dict:
    """SHA-256 of splits/*.json and processed/*.jsonl, keyed by path relative to data_dir."""
    files = sorted([*(data_dir / "splits").glob("*.json"), *(data_dir / "processed").glob("*.jsonl")])
    return {str(p.relative_to(data_dir)): _sha256(p) for p in files}


def validate(data_dir: Path) -> list:
    """Check the build in data_dir. Returns a list of problems (empty when everything holds)."""
    problems = []
    try:  # checksums against provenance.json, record counts, MedAbstain test files vs MedQA train/val
        medqa = data.load_splits("medqa", data_dir=data_dir)
        report = verify_medabstain(paths.MEDABSTAIN_DIR, medqa)
        for key, result in report["medqa_overlap"]["results"].items():
            if key.startswith("test/") and any(result[s]["overlapping_examples"] for s in ("train", "val")):
                problems.append(f"MedAbstain {key} overlaps MedQA train/val")
        splits = {n: data.load_splits(n, data_dir=data_dir) for n in NAMES}  # also the config guard
    except Exception as exc:  # noqa: BLE001 - every failure is reported the same way
        return problems + [f"{type(exc).__name__}: {exc}"]

    ids = [e["id"] for n in NAMES for rows in splits[n].values() for e in rows]
    if len(ids) != len(set(ids)):
        problems.append("an id appears in more than one split")
    eval_seen, train_keys, train_exact = {}, set(), set()
    for n in NAMES:
        for s, rows in splits[n].items():
            for e in rows:
                key = data.leak_key(e)
                if s == "train":
                    train_keys.add(key)
                    if (ek := data.exact_key(e)) in train_exact:
                        problems.append(f"{e['id']}: true copy of another train question")
                    train_exact.add(ek)
                elif key in eval_seen:
                    problems.append(f"{e['id']} ({n}/{s}) shares a question with {eval_seen[key]}")
                else:
                    eval_seen[key] = f"{n}/{s}"
    leaked = [where for key, where in eval_seen.items() if key in train_keys]
    if leaked:
        problems.append(f"{len(leaked)} eval questions also appear in a train set, e.g. in {leaked[:3]}")

    protected = data.medabstain_protected_questions()
    for n in NAMES:
        for s, rows in splits[n].items():
            if hits := [e["id"] for e in rows if data.normalize(e["question"]) in protected]:
                problems.append(f"{n}/{s} contains {len(hits)} MedAbstain test questions, e.g. {hits[:3]}")

    stats = json.loads((data_dir / "splits" / "stats.json").read_text())
    for n in NAMES:
        sizes = {s: len(rows) for s, rows in splits[n].items()}
        if sizes != stats["sizes"][n]:
            problems.append(f"{n}: loaded sizes {sizes} differ from stats.json {stats['sizes'][n]}")
        for s, nominal in data.SPLIT_SIZES[n].items():
            if not 0.98 * nominal <= sizes[s] <= nominal:
                problems.append(f"{n}/{s} has {sizes[s]} rows, nominal {nominal}")
        if not sizes["train"] > sum(data.SPLIT_SIZES[n].values()) or not sizes["test"]:
            problems.append(f"{n}: implausible sizes {sizes}")
    if len(splits["medqa"]["test"]) != stats["raw_sizes"]["medqa"]["test"]:
        problems.append("MedQA test is not the complete official test set")
    return problems


def build_manifest(data_dir: Path) -> dict:
    stats = json.loads((data_dir / "splits" / "stats.json").read_text())
    provenance = json.loads((paths.MEDABSTAIN_DIR / "provenance.json").read_text())
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "environment": data._environment(),
        "hf_revisions": data.HF_REVISIONS,
        "resolved_revisions": stats["resolved_revisions"],
        "medabstain": {"upstream_commit": provenance["commit"], "sha256": provenance["sha256"]},
        "config": json.loads(json.dumps(data.current_config())),
        "sha256": file_checksums(data_dir),
        "sizes": stats["sizes"],
        "removed": stats["removed"],
    }


def verify_build(data_dir: Path) -> list:
    """Compare data_dir with its manifest.json and validate it. Returns a list of problems."""
    path = data_dir / MANIFEST
    if not path.is_file():
        return [f"{path} not found"]
    manifest = json.loads(path.read_text())
    now = file_checksums(data_dir)
    problems = [f"{name}: checksum differs from the manifest" for name, digest in manifest["sha256"].items()
                if now.get(name, digest) != digest]
    problems += [f"{name}: listed in the manifest but missing" for name in manifest["sha256"].keys() - now.keys()]
    problems += [f"{name}: present but not in the manifest" for name in now.keys() - manifest["sha256"].keys()]
    if manifest["config"] != json.loads(json.dumps(data.current_config())):
        problems.append("the manifest's config differs from the current code's current_config()")
    try:
        provenance = json.loads((paths.MEDABSTAIN_DIR / "provenance.json").read_text())
    except OSError as exc:
        problems.append(f"MedAbstain provenance.json unreadable: {exc}")
    else:
        if manifest["medabstain"] != {"upstream_commit": provenance["commit"], "sha256": provenance["sha256"]}:
            problems.append("MedAbstain provenance.json differs from the manifest")
    return problems + validate(data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true",
                        help="do not build: check an existing build against its manifest.json and validate it")
    parser.add_argument("--force", action="store_true", help="overwrite an existing build")
    parser.add_argument("--allow-non-main", action="store_true",
                        help="skip the clean-main check (only when FME_SHARED_DIR is not the default)")
    args = parser.parse_args()
    data_dir = paths.DATA_DIR

    if args.allow_non_main and data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
        raise SystemExit(f"--allow-non-main is not allowed for the default data dir {DEFAULT_DATA_DIR}; "
                         "set FME_SHARED_DIR to another folder")
    if REQUIRE_CLEAN_MAIN and not args.allow_non_main:
        require_clean_main()

    if args.verify_only:
        if problems := verify_build(data_dir):
            raise SystemExit(f"Verification of {data_dir} FAILED:\n  - " + "\n  - ".join(problems))
        manifest = json.loads((data_dir / MANIFEST).read_text())
        print(f"OK: {data_dir} matches {MANIFEST} (built from commit {manifest['git']['commit']}, "
              f"{manifest['created_utc']}) and passes validation")
        print(json.dumps({"sizes": manifest["sizes"], "sha256": manifest["sha256"]}, indent=2))
        return

    existing = [str(p) for p in (data_dir / "splits", data_dir / "processed", data_dir / MANIFEST) if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"Refusing to overwrite an existing build: {', '.join(existing)}\n"
                         "Check it with --verify-only, or pass --force to rebuild (never once tuning has started).")
    (data_dir / MANIFEST).unlink(missing_ok=True)  # a failed rebuild must not leave a manifest for other files

    download_medabstain()
    data.build_all(verbose=False)
    if problems := validate(data_dir):
        raise SystemExit("Validation FAILED, no manifest written:\n  - " + "\n  - ".join(problems))
    manifest = build_manifest(data_dir)
    (data_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("git", "sizes", "removed", "sha256")}, indent=2))
    print(f"Built and validated {data_dir}; wrote {data_dir / MANIFEST}")


if __name__ == "__main__":
    main()
