"""Visualize GELU against ReLU and save a comparison plot."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.adapters import run_gelu


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install it first, e.g. `python -m pip install matplotlib`."
        ) from exc

    out_path = PROJECT_ROOT / "demo" / "gelu_vs_relu.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = torch.linspace(-5.0, 5.0, steps=2000, dtype=torch.float32)

    try:
        y_gelu = run_gelu(x)
    except NotImplementedError as exc:
        raise RuntimeError(
            "run_gelu is not implemented yet. Implement it in tests/adapters.py first."
        ) from exc

    if not isinstance(y_gelu, torch.Tensor):
        y_gelu = torch.as_tensor(y_gelu, dtype=torch.float32)

    x_np = x.detach().cpu().numpy()
    y_gelu_np = y_gelu.detach().cpu().numpy()
    y_relu_np = torch.relu(x).detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 6), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(x_np, y_gelu_np, label="GELU", linewidth=2.5, color="#1f77b4")
    ax.plot(
        x_np,
        y_relu_np,
        label="ReLU (reference)",
        linewidth=2.0,
        linestyle="--",
        color="#ff7f0e",
    )
    ax.axhline(0.0, color="#aaaaaa", linewidth=1.0)
    ax.axvline(0.0, color="#aaaaaa", linewidth=1.0)
    ax.set_title("GELU vs ReLU")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
