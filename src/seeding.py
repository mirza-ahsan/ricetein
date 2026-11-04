"""Control every source of randomness from one seed.

Call ``seed_everything(config["seed"], config["determinism"])`` once at the very start
of a run, before anything touches the GPU.

- ``strict`` (default): deterministic algorithms only, cuDNN autotuning off and full fp32
  precision (no TF32) for matmuls, convolutions and RNNs. A run then repeats exactly on
  one machine, and the local RTX 4060 and a Kaggle T4 agree within the Stage 0 tolerance
  (1e-3): the RTX 4060 supports TF32 but the T4 does not, so TF32 alone would make the
  two machines disagree. Some backward passes have no deterministic version on the GPU
  (e.g. ``adaptive_avg_pool2d`` to sizes above 1, ``scatter_add``); they raise an error
  in this mode rather than silently varying.
- ``fast``: cuDNN autotuning, non-deterministic kernels and TF32 (TF32 only on GPUs that
  have it; the T4 does not). Runs then repeat only approximately. For heavy fine-tuning
  where speed matters more than exact agreement; set ``determinism: fast`` in that
  stage's settings file.

Never mix in the legacy ``allow_tf32`` flags or ``torch.set_float32_matmul_precision``:
torch 2.11 rejects mixing them with the ``fp32_precision`` settings used here, and the
latter silently switches matmuls back to TF32.
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
    os.environ["PYTHONHASHSEED"] = str(seed)  # reaches only freshly started interpreters (spawn, subprocess)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device

    strict = determinism == "strict"
    if strict:
        # Older torch versions require this for deterministic cuBLAS; harmless on 2.11.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.benchmark = not strict
    precision = "ieee" if strict else "tf32"
    # Each setting is needed: conv and rnn default to an explicit "tf32" and do not
    # inherit from torch.backends.cudnn.fp32_precision.
    torch.backends.cuda.matmul.fp32_precision = precision
    torch.backends.cudnn.fp32_precision = precision
    torch.backends.cudnn.conv.fp32_precision = precision
    torch.backends.cudnn.rnn.fp32_precision = precision
