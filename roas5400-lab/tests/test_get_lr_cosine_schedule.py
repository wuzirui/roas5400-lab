"""Cosine schedule with warmup tests."""

import math

import pytest

from .adapters import run_get_lr_cosine_schedule


def test_get_lr_cosine_schedule_warmup_boundaries() -> None:
    num_warmup_steps = 4
    num_training_steps = 20

    assert run_get_lr_cosine_schedule(0, num_warmup_steps, num_training_steps) == pytest.approx(0.0)
    assert run_get_lr_cosine_schedule(2, num_warmup_steps, num_training_steps) == pytest.approx(0.5)
    assert run_get_lr_cosine_schedule(4, num_warmup_steps, num_training_steps) == pytest.approx(1.0)


def test_get_lr_cosine_schedule_cosine_values() -> None:
    num_warmup_steps = 2
    num_training_steps = 10

    step = 6
    progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    expected = 0.5 * (1.0 + math.cos(math.pi * progress))
    actual = run_get_lr_cosine_schedule(step, num_warmup_steps, num_training_steps)
    assert actual == pytest.approx(expected)


def test_get_lr_cosine_schedule_post_training_is_zero() -> None:
    assert run_get_lr_cosine_schedule(10, 2, 10) == pytest.approx(0.0)
    assert run_get_lr_cosine_schedule(999, 2, 10) == pytest.approx(0.0)
