"""Checks for src/seeding.py: one seed repeats every random source; modes set the right flags."""

import random

import numpy as np
import pytest
import torch

from src.seeding import seed_everything

PRECISION_SETTINGS = [
    torch.backends.cuda.matmul,
    torch.backends.cudnn,
    torch.backends.cudnn.conv,
    torch.backends.cudnn.rnn,
]


@pytest.fixture(autouse=True)
def restore_global_state(monkeypatch):
    """seed_everything changes process-wide torch state; put it back so other tests are unaffected."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # registers both variables for restoring
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        [s.fp32_precision for s in PRECISION_SETTINGS],
    )
    yield
    deterministic, warn_only, benchmark, precisions = saved
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    for setting, value in zip(PRECISION_SETTINGS, precisions):
        setting.fp32_precision = value


def draw():
    values = [random.random(), float(np.random.rand()), float(torch.rand(1))]
    if torch.cuda.is_available():
        values.append(float(torch.rand(1, device="cuda")))
    return values


def test_same_seed_repeats_every_random_source():
    seed_everything(123)
    first = draw()
    seed_everything(123)
    assert draw() == first
    seed_everything(124)
    assert draw() != first


MODES = [("strict", True, "ieee"), ("fast", False, "tf32")]


@pytest.mark.parametrize(("mode", "deterministic", "precision"), MODES)
def test_mode_sets_determinism_autotuning_and_every_precision(mode, deterministic, precision):
    seed_everything(0, mode)
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.backends.cudnn.benchmark is not deterministic
    assert [s.fp32_precision for s in PRECISION_SETTINGS] == [precision] * len(PRECISION_SETTINGS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_strict_mode_really_computes_full_fp32_on_the_gpu():
    """Catches TF32 whatever the flags say: TF32 errors are ~1e-4, full fp32 ~1e-7.

    The convolution must be large enough for cuDNN to pick a TF32-capable algorithm.
    """
    seed_everything(0, "strict")
    x = torch.randn(8, 64, 56, 56, device="cuda")
    w = torch.randn(128, 64, 3, 3, device="cuda")
    a = torch.randn(512, 512, device="cuda")
    conv_ref = torch.nn.functional.conv2d(x.double(), w.double())
    conv_err = (torch.nn.functional.conv2d(x, w).double() - conv_ref).abs().max() / conv_ref.abs().max()
    mm_ref = a.double() @ a.double()
    mm_err = ((a @ a).double() - mm_ref).abs().max() / mm_ref.abs().max()
    assert conv_err < 1e-5 and mm_err < 1e-5


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="determinism"):
        seed_everything(0, "sloppy")
