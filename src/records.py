"""Unified run recorder for experiments on the laptop and on Kaggle.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
CODE_DATASET_SLUG = "ricetein-code"

# Package versions written into every record (missing ones are recorded as None).
TRACKED_PACKAGES = ("torch", "torchvision", "timm", "transformers", "numpy", "scikit-learn", "pandas")


def on_kaggle() -> bool:
    return KAGGLE_WORKING.exists()


def get_default_runs_dir() -> Path:
    """``/kaggle/working/runs`` on Kaggle, otherwise ``<repo>/runs`` (whatever the working directory)."""
    return KAGGLE_WORKING / "runs" if on_kaggle() else REPO_ROOT / "runs"


def find_kaggle_input(slug: str) -> Path | None:
    """Locate a mounted Kaggle dataset or notebook output by its slug.

    Kaggle has used both ``/kaggle/input/<slug>`` and
    ``/kaggle/input/datasets/<owner>/<slug>`` layouts, so both are searched.
    """
    if not KAGGLE_INPUT.exists():
        return None
    direct = KAGGLE_INPUT / slug
    if direct.is_dir():
        return direct
    matches = sorted(p for p in KAGGLE_INPUT.glob(f"*/*/{slug}") if p.is_dir())
    return matches[0] if matches else None


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()


def get_code_version() -> str:
    """Return the code version the run is using.

    On Kaggle: the commit ID written into the ``ricetein-code`` dataset by
    ``scripts/kaggle_sync_code.sh``. On the laptop: the current git commit,
    with ``-dirty`` appended when tracked code has uncommitted changes.
    """
    if on_kaggle():
        code_dir = find_kaggle_input(CODE_DATASET_SLUG)
        if code_dir and (code_dir / "CODE_VERSION").exists():
            return (code_dir / "CODE_VERSION").read_text().strip()
        return "unknown"

    try:
        commit = _git("rev-parse", "--short", "HEAD")
        dirty = _git("status", "--porcelain", "--", "src", "configs", "splits", "pyproject.toml", "uv.lock")
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
