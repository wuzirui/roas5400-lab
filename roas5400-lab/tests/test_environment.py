"""Basic environment smoke tests for ROAS5400 lab."""

import importlib
import sys

import pytest


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "scipy",
    "einops",
    "jaxtyping",
    "yaml",
    "PIL",
    "tqdm",
]


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_modules_are_importable(module_name: str) -> None:
    """Ensure all critical runtime dependencies are importable."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_python_version_is_supported() -> None:
    """Project requires Python 3.10 or newer."""
    assert sys.version_info >= (3, 10)


def test_torch_basic_runtime() -> None:
    """A minimal tensor op should work after environment setup."""
    torch = importlib.import_module("torch")
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    z = x + y
    assert z.shape == (2,)
    assert float(z.sum().item()) == pytest.approx(10.0)
