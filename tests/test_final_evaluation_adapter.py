import json

import pytest

from scripts.run_final_evaluation import digest, normalise_record, recover_preloader_failure


def test_json_keys_are_restored_without_mutating_frozen_record():
    record = {"config": {"wholeimage": {"size": 224, "batch_size": {"224": 32, "448": 16}}}}
    result = normalise_record(record)
    assert result["config"]["wholeimage"]["batch_size"][224] == 32
    assert record["config"]["wholeimage"]["batch_size"] == {"224": 32, "448": 16}


@pytest.mark.parametrize("violation", ["saved_predictions", "different_plan"])
def test_recovery_refuses_unsafe_attempt(tmp_path, violation):
    folder = tmp_path / "runs/stage5_example"
    folder.mkdir(parents=True)
    plan = {"runs": {"stage5_example": {}}}
    plan_path = tmp_path / "runs/deadline_evaluation_plan.json"
    plan_path.write_text(json.dumps(plan))
    marker = {"plan_sha256": digest(plan_path) if violation != "different_plan" else "incorrect"}
    (folder / "test_started.json").write_text(json.dumps(marker))
    if violation == "saved_predictions":
        (folder / "test_predictions.csv").write_text("existing output")
    with pytest.raises(ValueError):
        recover_preloader_failure(tmp_path, plan)
    assert (folder / "test_started.json").exists()
    assert not (folder / "test_started_preloader_failure.json").exists()
