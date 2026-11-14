"""Unified run recorder for experiments on the local machine and on Kaggle.

Every run writes the same files into ``runs/<run_name>/`` (``/kaggle/working/runs/<run_name>/``
on Kaggle):

    record.json      settings, code version, split checksum, data version, hardware,
                     package versions, start/end time, duration, peak GPU memory
    history.csv      one row per epoch
    metrics.json     final scores
    predictions.csv  one row per photo
    run.log          console log of the run
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import psutil

from src.config import REPO_ROOT, on_kaggle, output_root
from src.kaggle_bootstrap import CODE_DATASET_SLUG, find_kaggle_input

# What counts as "the code" for a run's code version (also checked by scripts/kaggle_*.sh).
CODE_PATHS = ("src", "configs", "splits", "pyproject.toml", "uv.lock")

# The module whose code produces a run's numbers, by run-name prefix. A completed run stays reusable
# while that module and everything it imports are unchanged (handbook §24, 2026-11-14).
RUN_MODULES = {"stage4": "src.sanity_baselines", "stage5": "src.train_wholeimage",
               "stage7a": "src.linear_probe", "stage7": "src.train_mil"}
BOOKKEEPING = {"src/records.py"}  # writes records and decides reuse; computes no score

# Package versions written into every record (missing ones are recorded as None).
TRACKED_PACKAGES = ("torch", "torchvision", "timm", "transformers", "numpy", "scikit-learn", "pandas")


def get_default_runs_dir() -> Path:
    """``/kaggle/working/runs`` on Kaggle, otherwise ``<repo>/runs`` (whatever the working directory)."""
    return output_root() / "runs"


def require_clean_code() -> str:
    version = get_code_version()
    if version == "unknown" or version.endswith("-dirty"):
        raise RuntimeError(f"code version {version}: commit before real experiments")
    return version


def completed_run(name: str, artifacts: tuple[str, ...] = ("metrics.json", "predictions.csv")) -> bool:
    folder = get_default_runs_dir() / name
    record = folder / "record.json"
    return (record.exists() and json.loads(record.read_text()).get("status") == "completed"
            and all((folder / artifact).exists() for artifact in artifacts))


def source_files(module: str) -> list[str]:
    """Repo-relative files of ``module`` and of every ``src`` module it imports, directly or not."""
    seen: dict[str, str] = {}
    todo = [module]
    while todo:
        name = todo.pop()
        path = REPO_ROOT / f"{name.replace('.', '/')}.py"
        if name in seen or not path.exists():
            continue
        seen[name] = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "src":
                todo += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Import):
                todo += [alias.name for alias in node.names if alias.name.startswith("src.")]
    return sorted(seen.values())


def code_paths(name: str) -> tuple[str, ...]:
    """What must be unchanged for run ``name`` to be reused. Settings and the split are compared
    separately, so a known run kind only depends on the code it executes and the locked packages."""
    module = RUN_MODULES.get(name.split("_")[0])
    if module is None:
        return CODE_PATHS
    return (*(f for f in source_files(module) if f not in BOOKKEEPING), "pyproject.toml", "uv.lock")


def reusable_run(name: str, config: Mapping[str, Any], checksum: str,
                 artifacts: tuple[str, ...] = ("metrics.json", "predictions.csv")) -> bool:
    """Completed runs may be reused only with the same settings, split and clean source code."""
    if not completed_run(name, artifacts):
        return False
    record = json.loads((get_default_runs_dir() / name / "record.json").read_text())
    expected = json.loads(json.dumps(config, default=_json_default))
    source = record["code_version"]
    if (record["config"] != expected or record["split_checksum"] != checksum
            or source == "unknown" or source.endswith("-dirty")):
        raise ValueError(f"{name}: existing results have different settings, split or untraceable code")
    current = require_clean_code()
    if source != current:
        try:
            differences = _git("diff", source, current, "--", *code_paths(name))
        except Exception as exc:
            raise ValueError(f"{name}: cannot verify the previous code") from exc
        if differences:
            raise ValueError(f"{name}: source code changed; archive affected runs after logging the fix")
    return True


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()


def get_code_version() -> str:
    """Return the code version the run is using.

    On Kaggle: the commit ID written into the ``ricetein-code`` dataset by
    ``scripts/kaggle_sync_code.sh``. On the local machine: the current git commit,
    with ``-dirty`` appended when tracked code has uncommitted changes.
    """
    if on_kaggle():
        code_dir = find_kaggle_input(CODE_DATASET_SLUG)
        if code_dir and (code_dir / "CODE_VERSION").exists():
            return (code_dir / "CODE_VERSION").read_text().strip()
        return "unknown"

    try:
        commit = _git("rev-parse", "--short", "HEAD")
        dirty = _git("status", "--porcelain", "--", *CODE_PATHS)
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "unknown"


def get_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def get_hardware_info() -> dict[str, Any]:
    """Describe the machine and accelerator the run is using."""
    info: dict[str, Any] = {
        "environment": "kaggle" if on_kaggle() else "local",
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": False,
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def _json_default(obj: Any) -> Any:
    """Make numpy numbers/arrays, paths and sets JSON-serialisable."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "tolist"):  # numpy scalars and arrays, torch tensors
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


class RunRecorder:
    """Records one experiment run.

    A run name may only be used once: if ``runs/<run_name>/`` already contains
    files, a ``FileExistsError`` is raised unless ``overwrite=True`` (which
    deletes the old folder first). This keeps results from different runs from
    being silently mixed.
    """

    def __init__(
        self,
        run_name: str,
        config: Mapping[str, Any] | None = None,
        split_checksum: str | None = None,
        data_version: str | Mapping[str, Any] | None = None,
        base_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> None:
        self.run_name = run_name
        self.config = dict(config) if config else {}
        self.split_checksum = split_checksum
        self.data_version = data_version

        base = Path(base_dir) if base_dir else get_default_runs_dir()
        self.run_dir = base / run_name
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"{self.run_dir} already has results. Use a new run name, or pass overwrite=True "
                    "to delete the old results (only after a logged bug fix)."
                )
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        self.start_time = time.time()
        self.code_version = get_code_version()
        self.hardware_info = get_hardware_info()
        self.package_versions = get_package_versions()

        # Peak GPU memory is measured from here, not from earlier runs in the same notebook.
        if self.hardware_info.get("cuda_available"):
            import torch

            torch.cuda.reset_peak_memory_stats()

        self.history_file = self.run_dir / "history.csv"
        self._history_headers: list[str] | None = None

        self.logger = self._setup_logger()
        self.logger.info("=" * 60)
        self.logger.info(f"Run: {self.run_name}")
        self.logger.info(f"Folder: {self.run_dir.resolve()}")
        self.logger.info(f"Code version: {self.code_version}")
        if self.code_version.endswith("-dirty") or self.code_version == "unknown":
            self.logger.warning("Code version is not a clean commit; this run cannot be traced exactly.")
        if self.hardware_info.get("cuda_available"):
            gpu, mem = self.hardware_info.get("gpu_name"), self.hardware_info.get("gpu_total_memory_gb")
            self.logger.info(f"GPU: {gpu} ({mem} GB)")
        else:
            self.logger.info("Device: CPU (CUDA not available)")
        self.logger.info("=" * 60)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"ricetein.{self.run_name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(self.run_dir / "run.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def log_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Append one epoch's metrics to history.csv (the first call fixes the columns)."""
        row: dict[str, Any] = {"epoch": epoch, **metrics}
        if self._history_headers is None:
            self._history_headers = list(row.keys())
            with open(self.history_file, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._history_headers).writeheader()
        elif set(row) != set(self._history_headers):
            raise ValueError(
                f"Epoch {epoch} metrics {sorted(row)} differ from the first epoch's columns "
                f"{sorted(self._history_headers)}"
            )
        with open(self.history_file, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._history_headers).writerow(row)

        shown = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()
        )
        self.logger.info(f"Epoch {epoch:03d} -> {shown}")

    def save_metrics(self, metrics: Mapping[str, Any]) -> Path:
        """Write final scores to metrics.json; include which photos they come from ("partition": "test")."""
        target = self.run_dir / "metrics.json"
        _write_json(target, dict(metrics))
        self.logger.info(f"Saved {target.name}")
        return target

    def save_predictions(self, predictions: Any) -> Path:
        """Save per-photo predictions (a pandas DataFrame or a list of dicts) to predictions.csv."""
        target = self.run_dir / "predictions.csv"
        if hasattr(predictions, "to_csv"):
            predictions.to_csv(target, index=False)
        elif isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
            with open(target, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
                writer.writeheader()
                writer.writerows(predictions)
        else:
            raise ValueError("predictions must be a pandas DataFrame or a non-empty list of dicts")
        self.logger.info(f"Saved {target.name}")
        return target

    def finish(self, status: str = "completed") -> dict[str, Any]:
        """Write record.json with timing and peak GPU memory, and close the log file."""
        duration_seconds = round(time.time() - self.start_time, 2)

        peak_gpu_mb = None
        if self.hardware_info.get("cuda_available"):
            import torch

            peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)

        record = {
            "run_name": self.run_name,
            "status": status,
            "start_time": self.start_timestamp,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration_seconds,
            "code_version": self.code_version,
            "split_checksum": self.split_checksum,
            "data_version": self.data_version,
            "hardware": {**self.hardware_info, "peak_gpu_memory_mb": peak_gpu_mb},
            "packages": self.package_versions,
            "config": self.config,
        }
        _write_json(self.run_dir / "record.json", record)

        self.logger.info(f"Run finished in {duration_seconds}s (status: {status}); saved record.json")
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        return record
