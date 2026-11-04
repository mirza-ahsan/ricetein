"""Checks for src/seeding.py: one seed repeats every random source; modes set the right flags."""

import random

import numpy as np
import pytest
import torch

from src.seeding import seed_everything


@pytest.fixture(autouse=True)
def restore_torch_flags():
    yield
    seed_everything(0, "strict")  # leave the process in the project default


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_repeats_every_random_source():
    seed_everything(123)
    first = draw()
    seed_everything(123)
    assert draw() == first
    seed_everything(124)
    assert draw() != first


def test_strict_mode_is_deterministic_and_full_precision():
    seed_everything(0, "strict")
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"


def test_fast_mode_allows_tf32_and_autotuning():
    seed_everything(0, "fast")
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="determinism"):
        seed_everything(0, "sloppy")
