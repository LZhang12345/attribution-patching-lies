import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from hvp_patching import activation_patching, hvp, multi_step_hvp
from hvp_patching.heads import Head, HeadInterventions


@pytest.fixture(params=["gpt2", "llama"])
def adapter(request):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    if request.param == "gpt2":
        config = GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=32,
            n_layer=2,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            attn_implementation="eager",
        )
        model = GPT2LMHeadModel(config)
    else:
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            attn_implementation="eager",
        )
        model = LlamaForCausalLM(config)
    model.eval().requires_grad_(False)
    return HeadInterventions(model)


def test_isolated_endpoint_matches_independent_replacement(adapter):
    clean = torch.tensor([[1, 5, 8, 3]])
    corrupt = torch.tensor([[1, 9, 8, 3]])
    specs = [Head(0, 0), Head(1, 1)]
    fn = adapter.metric(clean, corrupt, specs, 4, 7)
    zero = torch.zeros(2)
    for index, spec in enumerate(specs):
        endpoint = zero.clone()
        endpoint[index] = 1
        additive = activation_patching(fn, zero, endpoint)
        independent = adapter.replacement_effect(clean, corrupt, spec, 4, 7)
        torch.testing.assert_close(additive, independent, rtol=1e-4, atol=2e-6)
        result = hvp(fn, zero, endpoint)
        assert torch.isfinite(result.estimate)
        torch.testing.assert_close(multi_step_hvp(fn, zero, endpoint, 1), result.estimate)
    for index in range(len(adapter.layers)):
        assert not adapter._projection(index)._forward_pre_hooks


def test_coordinate_gradient_matches_finite_difference(adapter):
    fn = adapter.metric(
        torch.tensor([[1, 5, 8, 3]]),
        torch.tensor([[1, 9, 8, 3]]),
        [Head(0, 0)],
        4,
        7,
    )
    zero = torch.zeros(1)
    result = hvp(fn, zero, torch.ones(1))
    epsilon = 0.05
    with torch.no_grad():
        difference = (fn(zero + epsilon) - fn(zero - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(result.atp, difference, rtol=0.02, atol=2e-6)


def test_bad_input_geometry_is_rejected(adapter):
    clean, corrupt = torch.tensor([[1, 2]]), torch.tensor([[1, 3]])
    with pytest.raises(ValueError, match="unique"):
        adapter.metric(clean, corrupt, [Head(0, 0), Head(0, 0)], 4, 7)
    with pytest.raises(ValueError, match="outside"):
        adapter.metric(clean, corrupt, [Head(99, 0)], 4, 7)
    with pytest.raises(ValueError, match="shape"):
        adapter.metric(clean, torch.tensor([[1, 2, 3]]), [Head(0, 0)], 4, 7)
