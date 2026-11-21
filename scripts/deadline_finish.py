"""Collect the final cloud batch, finish missing core fits, then evaluate once."""

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from import_deadline_outputs import collect
from src.config import REPO_ROOT, load_config
from src.records import require_clean_code
from src.train_wholeimage import done, run_name


def mark(stage):
    target = REPO_ROOT / "runs/deadline_finish_status.json"
    payload = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
               "code_version": require_clean_code()}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(f"Core study finishing: {stage}", flush=True)


def collect_cloud():
    mark("waiting_for_cloud_batch_1")
    api = KaggleApi()
    api.authenticate()
    kernel = "aztrek/ricetein-deadline-whole-1"
    failures = 0
    while True:
        try:
            state = api.kernels_status(kernel).status.name
            failures = 0
        except Exception:
            failures += 1
            if failures >= 5:
                raise
            time.sleep(30)
            continue
        if state not in ("NEW_SCRIPT", "QUEUED", "RUNNING", "CANCEL_REQUESTED"):
            break
        time.sleep(30)
    mark(f"collecting_cloud_batch_1_{state.lower()}")
    output = REPO_ROOT / "kaggle/outputs/deadline-whole-1"
    for attempt in range(3):
        try:
            api.kernels_output(kernel, path=str(output), file_pattern=r"^runs/", quiet=True)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(30)
    collect(output, REPO_ROOT)


def finish_whole():
    mark("waiting_for_local_convnext_swin")
    while subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "ricetein-deadline-whole.service"],
        check=False,
    ).returncode == 0:
        time.sleep(30)
    for model in ("convnext_tiny", "swin_tiny"):
        config = load_config(f"configs/wholeimage/{model}_224.yaml")
        if not all(done(run_name(config, lr, r)) for lr in config["wholeimage"]["lrs"]
                   for r in range(1, 6)):
            raise RuntimeError(f"Local {model} work stopped before completion; diagnose its log")
    mark("finishing_missing_inception_efficientnet_fits")
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "OMP_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for model in ("inceptionv3", "efficientnetv2s"):
        subprocess.run(
            [sys.executable, "-u", "-m", "src.train_wholeimage", "--config",
             f"configs/wholeimage/{model}_224.yaml"],
            cwd=REPO_ROOT, env=environment, check=True,
        )


def main():
    if require_clean_code() != "d44f65f":
        raise RuntimeError("Keep the original source/HEAD pinned while these studies run")
    with (REPO_ROOT / "cache/deadline_finish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect_cloud()
        finish_whole()
        mark("waiting_for_full_mil_and_probe_budgets")
        marker = REPO_ROOT / "runs/deadline_core_training_complete.json"
        while not marker.exists():
            time.sleep(30)
        training = json.loads(marker.read_text())
        if (training.get("code_version") != "d44f65f"
                or training.get("tuning_trials_each") != 40 or training.get("rounds") != 5):
            raise RuntimeError("Core training marker has unexpected provenance or budgets")
        mark("freezing_and_evaluating_core_models")
        # Never retry a failed test evaluation automatically; its started marker is preserved.
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).with_name("deadline_core.py")), "report"],
            cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, check=True,
        )
        mark("complete")


if __name__ == "__main__":
    main()
