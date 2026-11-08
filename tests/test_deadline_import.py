import copy
import hashlib
import json

import pytest

from scripts.import_deadline_outputs import collect
from src.config import load_config


@pytest.fixture
def export(tmp_path):
    project = tmp_path / "project"
    (project / "splits").mkdir(parents=True)
    split = project / "splits/folds.csv"
    split.write_text("frozen split fixture\n")
    source = tmp_path / "download"
    folder = source / "runs/stage5_inceptionv3_224_lr0.0001_r1"
    folder.mkdir(parents=True)
    record = {
        "run_name": folder.name, "status": "completed", "code_version": "d44f65f",
        "split_checksum": hashlib.sha256(split.read_bytes()).hexdigest(),
        "config": {**load_config("configs/wholeimage/inceptionv3_224.yaml"),
                   "lr": 1e-4, "round": 1},
    }
    (folder / "record.json").write_text(json.dumps(record))
    for name in ("metrics.json", "predictions.csv", "model.pt"):
        (folder / name).write_bytes(b"fixture artifact")
    return source, project, folder, record


def test_import_is_atomic_and_idempotent(export):
    source, project, folder, _ = export
    collect(source, project)
    collect(source, project)
    assert (project / "runs" / folder.name / "model.pt").read_bytes() == b"fixture artifact"


def test_reduced_budget_is_rejected(export):
    source, project, folder, record = export
    altered = copy.deepcopy(record)
    altered["config"]["wholeimage"]["max_epochs"] = 1
    (folder / "record.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="provenance/settings"):
        collect(source, project)
    assert not (project / "runs" / folder.name).exists()


def test_conflicting_artifacts_are_preserved(export):
    source, project, folder, _ = export
    collect(source, project)
    checkpoint = project / "runs" / folder.name / "model.pt"
    checkpoint.write_bytes(b"existing checkpoint")
    with pytest.raises(ValueError, match="refusing overwrite"):
        collect(source, project)
    assert checkpoint.read_bytes() == b"existing checkpoint"
