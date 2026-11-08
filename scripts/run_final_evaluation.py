"""Resume the frozen core evaluation after a disclosed JSON batch-key loader failure.

The failed attempt constructed no test loader and read no test images. Training,
selection, checkpoint bytes and the frozen evaluation plan remain unchanged.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise_record(record):
    result = copy.deepcopy(record)
    sizes = result["config"]["wholeimage"]["batch_size"]
    result["config"]["wholeimage"]["batch_size"] = {int(key): value for key, value in sizes.items()}
    return result


def recover_preloader_failure(project, plan):
    """Archive only the first failed loader attempt, with evidence and its original marker."""
    first = next(iter(plan["runs"]))
    folder = project / "runs" / first
    marker = folder / "test_started.json"
    archived = folder / "test_started_preloader_failure.json"
    if archived.exists():
        return
    if not marker.exists():
        raise ValueError("Expected the recorded first-run loader failure")
    if any((folder / name).exists() for name in ("test_predictions.csv", "test_metrics.json")):
        raise ValueError("Refuse recovery after any saved test output")
    started = json.loads(marker.read_text())
    plan_path = project / "runs/deadline_evaluation_plan.json"
    if started["plan_sha256"] != digest(plan_path):
        raise ValueError("Failed attempt does not match the frozen plan")
    log = project / "cache/deadline_finish.log"
    content = log.read_text()
    if 'batch = w["batch_size"][w["size"]]' not in content or "KeyError: 224" not in content:
        raise ValueError("The expected pre-loader failure is not documented")
    evidence = {
        "failure": "JSON serializes integer batch-size keys as strings; KeyError: 224",
        "stage": "before test DataLoader construction or test image iteration",
        "original_started_marker": started,
        "frozen_plan_sha256": digest(plan_path),
        "failure_log_sha256": digest(log),
        "recovery_script_sha256": digest(Path(__file__)),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "training_or_selection_changed": False,
    }
    with archived.open("x") as stream:
        json.dump(evidence, stream, indent=2)
    marker.unlink()


def run(project, controller):
    from src import evaluate
    from src.config import REPO_ROOT

    if REPO_ROOT.resolve() != project.resolve():
        raise ValueError("PYTHONPATH must point at the frozen training project")
    plan = json.loads((project / "runs/deadline_evaluation_plan.json").read_text())
    recover_preloader_failure(project, plan)
    original = evaluate.evaluate_stage5
    evaluate.EVALUATORS["stage5"] = lambda folder, record: original(folder, normalise_record(record))
    spec = importlib.util.spec_from_file_location("frozen_deadline_controller", controller)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.report()
    provenance = project / "results/tables/deadline_report_provenance.json"
    value = json.loads(provenance.read_text())
    value["evaluation_adapter_sha256"] = digest(Path(__file__))
    value["evaluation_recovery"] = "first attempt failed before loader construction; original marker archived"
    provenance.write_text(json.dumps(value, indent=2) + "\n")
    status = project / "runs/deadline_finish_status.json"
    status.write_text(json.dumps({"stage": "complete", "updated_at": datetime.now(timezone.utc).isoformat(),
                                  "code_version": plan["code_version"],
                                  "evaluation_adapter_sha256": digest(Path(__file__))}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.controller.resolve())
