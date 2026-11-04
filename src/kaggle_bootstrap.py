"""Run our code on Kaggle inside an isolated environment with the exact locked packages.

Why: Kaggle loads some packages (e.g. numpy) into the notebook's own Python process
before our first cell runs. Upgrading them in place leaves old and new versions mixed
in memory and breaks imports. So instead we build a separate virtual environment with
uv (fast, parallel downloads) and run our code in a fresh Python process inside it.

This module uses the standard library only: it runs before our dependencies exist.

Typical Kaggle notebook:

    from src.kaggle_bootstrap import setup_env, run
    setup_env()                                   # once per session, ~3-5 min
    run("src.some_stage", "--config", "configs/x.yaml", env={"HF_TOKEN": token})
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
CODE_DATASET_SLUG = "ricetein-code"
WEIGHTS_DATASET_SLUG = "ricetein-weights"
DEFAULT_VENV = Path(tempfile.gettempdir()) / "ricetein-venv"  # not /kaggle/working: that is saved as output


def find_kaggle_input(slug: str, root: Path | None = None) -> Path | None:
    """Locate a mounted Kaggle dataset or notebook output by its slug.

    Kaggle has used both ``/kaggle/input/<slug>`` and ``/kaggle/input/<kind>/<owner>/<slug>``.
    """
    root = root or KAGGLE_INPUT
    if not root.exists():
        return None
    if (root / slug).is_dir():
        return root / slug
    matches = sorted(p for p in root.glob(f"*/*/{slug}") if p.is_dir())
    return matches[0] if matches else None


def pretrained_source(repo_id: str) -> str:
    """Where to load a pretrained model from.

    On Kaggle with the private ``ricetein-weights`` dataset attached: its local folder for
    ``repo_id`` (no token or download needed, and the exact pinned version is used).
    Otherwise: the Hugging Face id itself (the local machine uses its logged-in cache).
    """
    root = find_kaggle_input(WEIGHTS_DATASET_SLUG)
    if root is not None:
        folder = repo_id.replace("/", "--")
        for candidate in [root / folder, *sorted(root.glob(f"*/{folder}"))]:
            if (candidate / "config.json").exists():
                return str(candidate)
    return repo_id


def pinned_pretrained(repo_id: str, revision: str) -> tuple[str, dict[str, str]]:
    """``(source, from_pretrained kwargs)`` that load ``repo_id`` at exactly ``revision``.

    On the local machine: the Hugging Face id plus ``revision=``. On Kaggle: the copy in
    ``ricetein-weights``, after checking its REVISIONS.txt lists this exact commit (the upload
    script copies whatever commit Hugging Face had at upload time, which may not be the pin).
    """
    source = pretrained_source(repo_id)
    if source == repo_id:
        return source, {"revision": revision}
    folder = Path(source)
    listing = next((p / "REVISIONS.txt" for p in (folder.parent, folder.parent.parent)
                    if (p / "REVISIONS.txt").exists()), None)
    pins = {line.strip() for line in listing.read_text().splitlines()} if listing else set()
    if f"{repo_id} {revision}" not in pins:
        raise RuntimeError(f"{WEIGHTS_DATASET_SLUG} does not hold {repo_id} at pinned revision {revision}")
    return source, {}


def code_dir() -> Path:
    found = find_kaggle_input(CODE_DATASET_SLUG)
    if found is None:
        raise FileNotFoundError(f"'{CODE_DATASET_SLUG}' is not attached (add it to dataset_sources).")
    return found


def _stream(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a command, printing its output live; raise if it fails."""
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def venv_python(venv_dir: Path = DEFAULT_VENV) -> Path:
    return venv_dir / "bin" / "python"


def setup_env(requirements: Path | None = None, venv_dir: Path = DEFAULT_VENV) -> Path:
    """Create the isolated environment (once per session) and return its Python."""
    python = venv_python(venv_dir)
    if python.exists():
        return python
    requirements = requirements or code_dir() / "requirements-kaggle.txt"
    if shutil.which("uv"):
        uv = ["uv"]
    else:
        _stream([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
        uv = [sys.executable, "-m", "uv"]
    _stream([*uv, "venv", str(venv_dir), "--python", sys.executable, "--quiet"])
    # unsafe-best-match: every version is pinned, so take each package from whichever
    # index has that exact version (torch from the PyTorch index, the rest from PyPI).
    _stream(
        [*uv, "pip", "install", "--python", str(python), "--index-strategy", "unsafe-best-match"]
        + ["--requirement", str(requirements)]
    )
    return python


def run(module: str, *args: object, env: dict[str, str] | None = None, code_root: Path | None = None) -> None:
    """Run ``python -m <module> <args>`` in the isolated environment, with our code importable."""
    python = setup_env()
    root = code_root or code_dir()
    # PYTHONPATH is replaced, not extended, so none of Kaggle's own packages leak in.
    full_env = {**os.environ, "PYTHONPATH": str(root), **(env or {})}
    _stream([str(python), "-u", "-m", module, *map(str, args)], env=full_env)
