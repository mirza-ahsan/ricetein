"""Control every source of randomness from one seed.

Call ``seed_everything(config["seed"], config["determinism"])`` once at the very start
of a run, before anything touches the GPU.

- ``strict`` (default): deterministic algorithms only, cuDNN autotuning off and full
  fp32 matmuls/convolutions (no TF32). This is what lets the same run give the same
  numbers on the local RTX 4060 and a Kaggle T4: the RTX 4060 supports TF32 but the T4
  does not, so TF32 alone would make the two machines disagree.
- ``fast``: autotuning and TF32 allowed. Repeatable on one machine only roughly; for
  heavy fine-tuning where speed matters more than bit-level agreement.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

MODES = ("strict", "fast")


def seed_everything(seed: int, determinism: str = "strict") -> None:
    if determinism not in MODES:
        raise ValueError(f"determinism must be one of {MODES}, got {determinism!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)  # only reaches child processes (e.g. DataLoader workers)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device

    strict = determinism == "strict"
    if strict:
        # cuBLAS needs this to be deterministic; it must be set before the first cuBLAS call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.benchmark = not strict
    precision = "ieee" if strict else "tf32"
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.fp32_precision = precision

