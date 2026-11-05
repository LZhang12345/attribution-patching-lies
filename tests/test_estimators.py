import pytest
import torch

from hvp_patching import (
    activation_patching,
    attribution_patching,
    hvp,
    multi_step_hvp,
)


DTYPE = torch.float64


def test_quadratic_matches_direct_patching():
    matrix = torch.tensor([[2.0, 4.0], [4.0, -3.0]], dtype=DTYPE)
    linear = torch.tensor([1.0, -0.5], dtype=DTYPE)

    def metric(a):
        return linear @ a + 0.5 * a @ matrix @ a

    clean = torch.tensor([0.3, -0.7], dtype=DTYPE)
    corrupt = torch.tensor([-0.2, 0.6], dtype=DTYPE)
    delta = corrupt - clean
    result = hvp(metric, clean, corrupt)
    torch.testing.assert_close(result.correction, 0.5 * delta @ matrix @ delta)
    truth = activation_patching(metric, clean, corrupt)
    torch.testing.assert_close(result.estimate, truth)
    for steps in (1, 2, 5):
        torch.testing.assert_close(multi_step_hvp(metric, clean, corrupt, steps), truth)


def test_direct_patching_evaluates_the_supplied_endpoint_without_reconstruction():
    clean = torch.tensor([1e9], dtype=torch.float32)
    corrupt = torch.tensor([1.0], dtype=torch.float32)
    # clean + (corrupt - clean) loses the small endpoint in float32.
    metric = torch.cos
    expected = torch.cos(corrupt[0]) - torch.cos(clean[0])
    result = activation_patching(lambda a: metric(a[0]), clean, corrupt)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [1, 2, 5, 10])
def test_cubic_remainder_matches_closed_form(steps):
    clean, corrupt = torch.zeros(1, dtype=DTYPE), torch.ones(1, dtype=DTYPE)

    def metric(a):
        return a[0].pow(3)

    ms_error = 1 - multi_step_hvp(metric, clean, corrupt, steps).item()
    assert ms_error == pytest.approx(1 / steps**2, abs=1e-13)


def test_k1_is_hvp_and_patch_direction_is_corrupt_minus_clean():
    def metric(a):
        return torch.sin(a).sum()

    clean = torch.tensor([0.6, -0.3], dtype=DTYPE)
    corrupt = torch.tensor([-0.1, 0.4], dtype=DTYPE)
    result = hvp(metric, clean, corrupt)
    expected_first = (torch.cos(clean) * (corrupt - clean)).sum()
    torch.testing.assert_close(result.atp, expected_first)
    torch.testing.assert_close(multi_step_hvp(metric, clean, corrupt, 1), result.estimate)


def test_isolated_component_does_not_include_joint_cross_term():
    def metric(a):
        return a[0] * a[1]

    clean = torch.zeros(2, dtype=DTYPE)
    isolated = torch.tensor([1.0, 0.0], dtype=DTYPE)
    joint = torch.ones(2, dtype=DTYPE)
    assert hvp(metric, clean, isolated).estimate.item() == 0
    assert hvp(metric, clean, joint).estimate.item() == 1


@pytest.mark.parametrize("metric", [lambda a: a.sum() + 3, lambda a: a.new_tensor(3.0)])
def test_affine_and_constant_metrics_have_zero_curvature(metric):
    clean, corrupt = torch.zeros(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE)
    result = hvp(metric, clean, corrupt)
    assert result.correction.item() == 0
    torch.testing.assert_close(result.estimate, activation_patching(metric, clean, corrupt))


def test_zero_first_order_uses_absolute_terms():
    clean, corrupt = torch.zeros(1, dtype=DTYPE), torch.ones(1, dtype=DTYPE)
    result = hvp(lambda a: a.square().sum(), clean, corrupt)
    assert result.exposure is None
    assert result.correction.item() == 1


def test_runs_inside_no_grad_without_changing_endpoint_tensors():
    clean = torch.tensor([0.1], dtype=DTYPE, requires_grad=True)
    corrupt = torch.tensor([0.8], dtype=DTYPE)
    before = clean.detach().clone()
    with torch.no_grad():
        result = hvp(lambda a: a.square().sum(), clean, corrupt)
    torch.testing.assert_close(clean.detach(), before)
    assert clean.grad is None
    assert not result.estimate.requires_grad


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_steps_rejected(steps):
    with pytest.raises(ValueError, match="positive integer"):
        multi_step_hvp(lambda a: a.sum(), torch.zeros(1), torch.ones(1), steps)


def test_invalid_endpoints_and_metric_rejected():
    with pytest.raises(ValueError, match="same nonempty shape"):
        hvp(lambda a: a.sum(), torch.zeros(1), torch.ones(2))
    with pytest.raises(ValueError, match="finite"):
        hvp(lambda a: a.sum(), torch.zeros(1), torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="scalar"):
        hvp(lambda a: a, torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError, match="float32 or float64"):
        attribution_patching(lambda a: a.sum(), torch.zeros(1).half(), torch.ones(1).half())
