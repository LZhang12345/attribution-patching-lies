import pytest
import torch

from hvp_patching.mib import (
    apply_isolated_destination_updates,
    gemma_attention_endpoint_direction,
    isolated_directional_taylor_terms,
)


def test_independent_edge_derivatives_match_closed_form():
    t = torch.tensor([0.0, 0.4, 0.8], dtype=torch.float64, requires_grad=True)
    losses = 2 * t + 3 * t.square() + t.pow(3)
    first, quadratic, estimate = isolated_directional_taylor_terms(losses, t, step_size=0.2)
    torch.testing.assert_close(first, 0.2 * (2 + 6 * t.detach() + 3 * t.detach().square()))
    torch.testing.assert_close(quadratic, 0.5 * 0.2**2 * (6 + 6 * t.detach()))
    torch.testing.assert_close(estimate, first + quadratic)


def test_replicated_batches_keep_edges_isolated():
    activations = torch.randn(6, 4, 2, 5, dtype=torch.float64)
    directions = torch.randn(3, 2, 4, 5, dtype=torch.float64)
    t = torch.tensor([0.2, 0.0, 0.7], dtype=torch.float64, requires_grad=True)
    changed = apply_isolated_destination_updates(activations, directions, t, destination_head=1)
    torch.testing.assert_close(changed[:, :, 0], activations[:, :, 0])
    torch.testing.assert_close(changed[2:4], activations[2:4])
    expected = activations[:, :, 1] + (directions * t[:, None, None, None]).reshape(6, 4, 5)
    torch.testing.assert_close(changed[:, :, 1], expected)
    per_edge = changed[:, :, 1].square().reshape(3, -1).sum(dim=1)
    derivative = torch.autograd.grad(per_edge[0], t, retain_graph=True)[0]
    assert derivative[1:].abs().sum().item() == 0


def test_gemma_endpoint_normalization_matches_independent_construction():
    torch.manual_seed(0)
    clean = torch.randn(1, 3, 2, 4, dtype=torch.float64)
    corrupt = torch.randn_like(clean)
    bias = torch.randn(4, dtype=torch.float64)

    def normalize(x):
        return x / (x.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()

    result = gemma_attention_endpoint_direction(clean, corrupt, 1, bias, normalize)
    replaced = clean.clone()
    replaced[:, :, 1] = corrupt[:, :, 1]
    expected = normalize(replaced.sum(2) + bias) - normalize(clean.sum(2) + bias)
    torch.testing.assert_close(result, expected)


def test_invalid_replicated_batch_is_rejected():
    with pytest.raises(ValueError, match="replicated activation batch"):
        apply_isolated_destination_updates(
            torch.zeros(5, 4, 5), torch.zeros(3, 2, 4, 5), torch.zeros(3), None
        )
