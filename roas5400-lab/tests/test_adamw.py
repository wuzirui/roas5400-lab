"""AdamW optimizer tests against torch.optim.AdamW."""

import torch

from .adapters import build_adamw_optimizer


def _assign_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g.clone()


def test_adamw_matches_torch_single_group() -> None:
    torch.manual_seed(7)

    init_tensors = [
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
    ]
    ours_params = [torch.nn.Parameter(t.clone()) for t in init_tensors]
    ref_params = [torch.nn.Parameter(t.clone()) for t in init_tensors]

    kwargs = {
        "lr": 3e-3,
        "betas": (0.85, 0.97),
        "eps": 1e-9,
        "weight_decay": 0.1,
    }
    ours = build_adamw_optimizer(ours_params, **kwargs)
    ref = torch.optim.AdamW(ref_params, **kwargs)

    for _ in range(10):
        grads = [torch.randn_like(p) for p in ours_params]
        _assign_grads(ours_params, grads)
        _assign_grads(ref_params, grads)
        ours.step()
        ref.step()

    for p_ours, p_ref in zip(ours_params, ref_params):
        torch.testing.assert_close(p_ours, p_ref, rtol=1e-12, atol=1e-12)


def test_adamw_matches_torch_param_groups() -> None:
    torch.manual_seed(13)

    init_a = torch.randn(5, dtype=torch.float64)
    init_b = torch.randn(5, dtype=torch.float64)
    ours_a, ours_b = torch.nn.Parameter(init_a.clone()), torch.nn.Parameter(init_b.clone())
    ref_a, ref_b = torch.nn.Parameter(init_a.clone()), torch.nn.Parameter(init_b.clone())

    group_a = {"params": [ours_a], "lr": 1e-3, "weight_decay": 0.05}
    group_b = {"params": [ours_b], "lr": 5e-4, "weight_decay": 0.0}
    ref_group_a = {"params": [ref_a], "lr": 1e-3, "weight_decay": 0.05}
    ref_group_b = {"params": [ref_b], "lr": 5e-4, "weight_decay": 0.0}

    ours = build_adamw_optimizer([group_a, group_b], betas=(0.9, 0.999), eps=1e-8)
    ref = torch.optim.AdamW([ref_group_a, ref_group_b], betas=(0.9, 0.999), eps=1e-8)

    for _ in range(8):
        grad_a = torch.randn_like(ours_a)
        grad_b = torch.randn_like(ours_b)
        ours_a.grad = grad_a.clone()
        ours_b.grad = grad_b.clone()
        ref_a.grad = grad_a.clone()
        ref_b.grad = grad_b.clone()
        ours.step()
        ref.step()

    torch.testing.assert_close(ours_a, ref_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ours_b, ref_b, rtol=1e-12, atol=1e-12)
