"""Import completed validation-only Stage 5 fits downloaded from Kaggle."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from src.config import load_config


ARTIFACTS = ("record.json", "metrics.json", "predictions.csv", "model.pt")
NAME = re.compile(
    r"stage5_(inceptionv3|efficientnetv2s|convnext_tiny|swin_tiny)_224_lr(3e-05|0\.0001|0\.0003)_r([1-5])"
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def collect(source, project):
    target = project / "runs"
    target.mkdir(parents=True, exist_ok=True)
    checksum = digest(project / "splits/folds.csv")
    imported = []
    for record_path in sorted(source.rglob("record.json")):
        folder = record_path.parent
        match = NAME.fullmatch(folder.name)
        if not match:
            continue
        record = json.loads(record_path.read_text())
        if record.get("status") != "completed" or not all(
            (folder / filename).is_file() for filename in ARTIFACTS
        ):
            continue
        model, lr, round_ = match.groups()
        config = record["config"]
        expected = {**load_config(f"configs/wholeimage/{model}_224.yaml"),
                    "lr": float(lr), "round": int(round_)}
        expected = json.loads(json.dumps(expected))
        if (
            record.get("run_name") != folder.name
            or record.get("split_checksum") != checksum
            or record.get("code_version") != "d44f65f"
            or config != expected
        ):
            raise ValueError(f"Unexpected provenance/settings: {folder}")
        if any(p.is_symlink() for p in folder.rglob("*")):
            raise ValueError(f"Symlinks are not permitted in a run export: {folder}")
        if any(folder.glob("test_*")):
            raise ValueError(f"Expected validation-only cloud output: {folder}")
        hashes = {filename: digest(folder / filename) for filename in ARTIFACTS}
        destination = target / folder.name
        if destination.exists():
            if any(not (destination / f).is_file() or digest(destination / f) != h
                   for f, h in hashes.items()):
                raise ValueError(f"Conflicting existing run; refusing overwrite: {destination}")
        else:
            with tempfile.TemporaryDirectory(prefix=".import-", dir=target) as temporary:
                staged = Path(temporary) / folder.name
                shutil.copytree(folder, staged)
                staged.rename(destination)
        imported.append({"run": folder.name, "artifacts_sha256": hashes})
    manifest = {"source": str(source.resolve()), "runs": imported}
    (target / f"deadline_import_{source.name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated and imported {len(imported)} complete fits", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--project", type=Path, default=Path("/home/aztrek/projects/ricetein"))
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("source must be an extracted Kaggle output directory")
    collect(args.source, args.project)
