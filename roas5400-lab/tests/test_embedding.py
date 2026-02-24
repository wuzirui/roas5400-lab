"""Token embedding module tests."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_embedding


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXPECTED_PATH = ASSETS_DIR / "embedding_expected.npz"


def test_embedding_lookup_matches_expected():
    expected = np.load(EXPECTED_PATH)
    weights = torch.from_numpy(expected["weights"])
    token_ids = torch.from_numpy(expected["token_ids"]).to(torch.long)
    expected_output = expected["output"]

    actual_output = run_embedding(
        vocab_size=int(expected["vocab_size"].item()),
        d_model=int(expected["d_model"].item()),
        weights=weights,
        token_ids=token_ids,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-6, atol=1e-6)


def test_embedding_repeated_ids_share_same_vector():
    vocab_size = 5
    d_model = 4
    weights = torch.arange(vocab_size * d_model, dtype=torch.float32).reshape(vocab_size, d_model)
    token_ids = torch.tensor([[1, 3, 1, 3, 1]], dtype=torch.long)

    actual_output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights,
        token_ids=token_ids,
    ).detach().cpu().numpy()

    np.testing.assert_allclose(actual_output[:, 0, :], actual_output[:, 2, :], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_output[:, 1, :], actual_output[:, 3, :], rtol=0.0, atol=0.0)
