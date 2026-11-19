"""Download MedAbstain data at a fixed upstream revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from src.data.medabstain import UPSTREAM_COMMIT, UPSTREAM_URL
from src.utils import paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_data(checkout: Path, destination: Path, commit: str) -> dict:
    """Copy the data and checksum manifest to a new destination."""
    files = sorted((checkout / "data").rglob("*"))
    files = [p for p in files if p.is_file()]
    if not files:
        raise ValueError("Upstream data directory is empty")
    manifest = {
        "upstream_url": UPSTREAM_URL,
        "commit": commit,
        "license": "CC-BY-NC-4.0",
        "sha256": {str(p.relative_to(checkout / "data")): _sha256(p) for p in files},
    }
    manifest["sha256"]["LICENSE"] = _sha256(checkout / "LICENSE")
    if destination.exists():
        provenance = destination / "provenance.json"
        if provenance.is_file() and json.loads(provenance.read_text()) == manifest:
            if all((destination / name).is_file() and _sha256(destination / name) == digest
                   for name, digest in manifest["sha256"].items()):
                return manifest
        raise FileExistsError(f"Refusing to overwrite existing data at {destination}; "
                              "use a new destination or verify its provenance")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".medabstain-", dir=destination.parent) as staging:
        staged = Path(staging) / "data"
        shutil.copytree(checkout / "data", staged)
        shutil.copy2(checkout / "LICENSE", staged / "LICENSE")
        (staged / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(destination)
    return manifest


def download_medabstain(data_dir=None) -> dict:
    directory = Path(data_dir).expanduser() if data_dir is not None else paths.MEDABSTAIN_DIR
    destination = directory.resolve()
    # Dataset files are stored outside the source tree.
    repo_root = Path(__file__).resolve().parents[1]
    if destination == repo_root or repo_root in destination.parents:
        raise ValueError("Data destination must be outside the source repository")
    with tempfile.TemporaryDirectory(prefix="medabstain-upstream-") as temporary:
        checkout = Path(temporary) / "upstream"
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM_URL + ".git", str(checkout)], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", UPSTREAM_COMMIT + "^{commit}"], text=True
        ).strip()
        if actual != UPSTREAM_COMMIT:
            raise ValueError(f"Upstream revision mismatch: {actual}")
        subprocess.run(["git", "-C", str(checkout), "checkout", UPSTREAM_COMMIT,
                        "--", "data", "LICENSE"], check=True)
        manifest = _install_data(checkout, destination, actual)
    print(f"MedAbstain data: {destination}\nUpstream commit: {actual}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Override shared data destination")
    args = parser.parse_args()
    download_medabstain(args.data_dir)


if __name__ == "__main__":
    main()
