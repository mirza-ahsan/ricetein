"""Environment smoke check: packages, GPU, datasets, run records and (optionally) DINOv3.

Kaggle:  run("src.smoke_check", "--require-dinov3")  (see the 00-smoke notebook)
Local:   uv run python -m src.smoke_check [--require-dinov3]
"""

from __future__ import annotations

import argparse
import importlib
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

from src.config import REPO_ROOT, dataset_dir, load_config, on_kaggle
from src.kaggle_bootstrap import find_kaggle_input, pinned_pretrained
from src.records import RunRecorder

CHECKED_PACKAGES = {  # import name -> distribution name in the lock file
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "transformers": "transformers",
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
}


def locked_versions() -> dict[str, str] | None:
    """Pinned versions from the Kaggle requirements file (Kaggle only)."""
    code = find_kaggle_input("ricetein-code")
    if code is None:
        return None
    text = (code / "requirements-kaggle.txt").read_text()
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, re.M))


def check_packages() -> None:
    """Import everything (scipy/sklearn import chains included); check the installed versions
    match the lock and that each package is loaded from this environment, not Kaggle's own."""
    locked = locked_versions()
    problems = {}
    for module_name, dist in CHECKED_PACKAGES.items():
        module = importlib.import_module(module_name)
        installed = metadata.version(dist)
        want = locked.get(dist) if locked else None
        from_env = str(Path(module.__file__).resolve()).startswith(str(Path(sys.prefix).resolve()))
        ok = want in (None, installed) and from_env
        note = "" if ok else ("   <-- VERSION MISMATCH" if from_env else "   <-- LOADED FROM OUTSIDE THE ENV")
        print(f"  {dist:24s} installed {installed:22s} locked {want or '(local: uv.lock)'}{note}")
        if not ok:
            problems[dist] = (installed, want, module.__file__)
    importlib.import_module("scipy.sparse")
    importlib.import_module("sklearn.metrics")
    importlib.import_module("imagehash")
    assert not problems, f"Package problems: {problems}"


def check_gpu() -> None:
    import torch

    assert torch.cuda.is_available(), "No GPU visible"
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    total = float((x @ x).sum())
    print(f"  {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | matmul sum {total:.1f}")


def check_datasets(config: dict) -> None:
    """Both datasets are where base.yaml says (the same lookup every stage uses)."""
    labelled = dataset_dir(config, "primary")
    for cls in sorted(p.name for p in labelled.iterdir()):
        print(f"  primary {cls:10s} {sum(1 for _ in (labelled / cls).glob('*.jpg')):5d} images")
    paddy = dataset_dir(config, "paddy")
    assert (paddy / config["datasets"]["paddy"]["metadata_file"]).exists(), "Paddy Doctor metadata missing"
    print(f"  paddy   {paddy} ({sum(1 for p in paddy.iterdir() if p.is_dir())} classes)")


def check_dinov3(config: dict) -> None:
    import torch
    from transformers import AutoModel

    spec = config["encoders"]["E1"]
    source, pinned = pinned_pretrained(spec["name"], spec["revision"])
    if on_kaggle():
        assert source != spec["name"], "ricetein-weights dataset not attached (add it to dataset_sources)"
    print(f"  loading from: {source} {pinned or '(pinned copy, revision checked)'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(source, **pinned).eval().to(device)
    with torch.no_grad():
        tokens = model(pixel_values=torch.randn(2, 3, 224, 224, device=device)).last_hidden_state
    assert tokens.shape[1:] == (201, 768), tokens.shape
    print(f"  DINOv3 ViT-B/16: output {tuple(tokens.shape)} = 1 summary + 4 registers + 196 patches")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-dinov3", action="store_true")
    parser.add_argument("--run-name", default="smoke_test")
    args = parser.parse_args()
    config = load_config("configs/base.yaml")

    print("Python", platform.python_version(), "| on Kaggle:", on_kaggle())
    pin = REPO_ROOT / ".python-version"  # on Kaggle REPO_ROOT is the code dataset, which carries it
    assert not pin.exists() or platform.python_version() == pin.read_text().strip(), (
        f"Python {platform.python_version()} is not the pinned {pin.read_text().strip()}")
    print("1. Packages")
    check_packages()
    print("2. GPU")
    check_gpu()
    print("3. Datasets")
    check_datasets(config)
    if args.require_dinov3:
        print("4. DINOv3")
        check_dinov3(config)

    rec = RunRecorder(args.run_name, config={"purpose": "environment smoke check"}, overwrite=True)
    rec.log_epoch(1, {"dummy_metric": 1.0})
    rec.save_metrics({"smoke": "passed", "partition": "none", "dinov3_checked": args.require_dinov3})
    record = rec.finish()
    assert record["code_version"] != "unknown", "code version not found"
    print("Smoke test passed. Code version:", record["code_version"])


if __name__ == "__main__":
    main()
