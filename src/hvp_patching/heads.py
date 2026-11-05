"""Additive attention-head interventions for Hugging Face decoder models.

The hook acts on concatenated head outputs at the input of the attention
output projection. Changing one coordinate from zero to one reproduces a
corrupt-into-clean head replacement at the selected token position.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class Head:
    layer: int
    head: int

    @property
    def name(self) -> str:
        return f"L{self.layer}H{self.head}"


def _attribute(root, path):
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


class HeadInterventions:
    """Construct a scalar metric F(z) for a single position-matched prompt pair.

    Supported projection layouts: GPT-2 and Llama-style decoder models,
    including Qwen and Gemma. The caller supplies an eval-mode model with
    eager attention. Only one prompt is accepted per call.
    """

    def __init__(self, model):
        if model.training:
            raise ValueError("call model.eval() before computing intervention effects")
        implementation = getattr(model.config, "_attn_implementation", None)
        if implementation != "eager":
            raise ValueError("load the model with attn_implementation='eager' for double backward")
        dtype = next(model.parameters()).dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("this reference adapter requires float32 or float64 model weights")
        self.model = model
        self.dtype = dtype
        self.device = next(model.parameters()).device
        config = model.config
        self.n_heads = getattr(config, "num_attention_heads", None) or config.n_head
        hidden = getattr(config, "hidden_size", None) or config.n_embd
        self.head_dim = getattr(config, "head_dim", None) or hidden // self.n_heads
        for path in ("model.layers", "transformer.h"):
            try:
                self.layers = _attribute(model, path)
                break
            except AttributeError:
                continue
        else:
            raise ValueError("unsupported decoder layout")

    def _projection(self, layer):
        for path in ("self_attn.o_proj", "attn.c_proj"):
            try:
                return _attribute(self.layers[layer], path)
            except AttributeError:
                continue
        raise ValueError(f"unsupported attention projection at layer {layer}")

    def _validate(self, clean_ids, corrupt_ids, heads, position):
        if clean_ids.ndim != 2 or clean_ids.shape[0] != 1 or clean_ids.shape != corrupt_ids.shape:
            raise ValueError("clean/corrupt token tensors must share shape [1, sequence]")
        if clean_ids.device != self.device or corrupt_ids.device != self.device:
            raise ValueError("input tensors must be on the model device")
        if not -clean_ids.shape[1] <= position < clean_ids.shape[1]:
            raise ValueError("patch position is outside the input sequence")
        if not heads or len(set(heads)) != len(heads):
            raise ValueError("heads must be a nonempty sequence of unique Head values")
        for spec in heads:
            if not 0 <= spec.layer < len(self.layers) or not 0 <= spec.head < self.n_heads:
                raise ValueError(f"head is outside the model: {spec}")

    def _forward(self, ids):
        return self.model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits

    @contextmanager
    def _capture(self, layers):
        store, handles = {}, []

        def capture(layer):
            def hook(module, args):
                store[layer] = args[0].detach().clone()

            return hook

        try:
            for layer in layers:
                handles.append(self._projection(layer).register_forward_pre_hook(capture(layer)))
            yield store
        finally:
            for handle in handles:
                handle.remove()

    def metric(
        self,
        clean_ids: Tensor,
        corrupt_ids: Tensor,
        heads: Sequence[Head],
        correct_id: int,
        counter_id: int,
        position: int = -1,
    ):
        """F(z) = correct-minus-counterfactual logit difference at the final token.

        Coordinates modify the live activation, preserving effects propagated
        from earlier layers. Evaluate along e_i to score an isolated head.
        """
        self._validate(clean_ids, corrupt_ids, heads, position)
        if correct_id == counter_id:
            raise ValueError("correct and counterfactual answer tokens must differ")
        if not all(0 <= token < self.model.config.vocab_size for token in (correct_id, counter_id)):
            raise ValueError("answer token is outside the model vocabulary")
        layers = sorted({head.layer for head in heads})
        with torch.no_grad():
            with self._capture(layers) as clean:
                self._forward(clean_ids)
            with self._capture(layers) as corrupt:
                self._forward(corrupt_ids)
        directions = {}
        for index, head in enumerate(heads):
            left = clean[head.layer][:, position].reshape(1, self.n_heads, self.head_dim)
            right = corrupt[head.layer][:, position].reshape(1, self.n_heads, self.head_dim)
            directions[index] = (right[0, head.head] - left[0, head.head]).detach()

        def metric_fn(z):
            if z.shape != (len(heads),) or z.device != self.device or z.dtype != self.dtype:
                raise ValueError("coordinates must match the head count, model dtype, and device")
            handles = []

            def patch(layer):
                entries = [(i, head.head) for i, head in enumerate(heads) if head.layer == layer]

                def hook(module, args):
                    activation = args[0]
                    expected = self.n_heads * self.head_dim
                    if activation.shape[-1] != expected:
                        raise ValueError(
                            "attention output does not match the configured head geometry"
                        )
                    shaped = activation.reshape(1, activation.shape[1], self.n_heads, self.head_dim)
                    update = torch.zeros_like(shaped)
                    for index, head_index in entries:
                        update[:, position, head_index] = z[index] * directions[index]
                    return ((shaped + update).reshape_as(activation),) + args[1:]

                return hook

            try:
                for layer in layers:
                    handles.append(self._projection(layer).register_forward_pre_hook(patch(layer)))
                logits = self._forward(clean_ids)[0, -1]
            finally:
                for handle in handles:
                    handle.remove()
            return logits[correct_id] - logits[counter_id]

        return metric_fn

    def replacement_effect(self, clean_ids, corrupt_ids, head, correct_id, counter_id, position=-1):
        """Independent direct replacement, used to validate the coordinate endpoint."""
        self._validate(clean_ids, corrupt_ids, [head], position)
        with torch.no_grad():
            with self._capture([head.layer]) as cached:
                self._forward(corrupt_ids)
            source = cached[head.layer][:, position].reshape(1, self.n_heads, self.head_dim)
            source = source[0, head.head].clone()
            baseline = self._forward(clean_ids)[0, -1]

            def replace(module, args):
                activation = args[0]
                shaped = activation.reshape(
                    1, activation.shape[1], self.n_heads, self.head_dim
                ).clone()
                shaped[:, position, head.head] = source
                return (shaped.reshape_as(activation),) + args[1:]

            handle = self._projection(head.layer).register_forward_pre_hook(replace)
            try:
                patched = self._forward(clean_ids)[0, -1]
            finally:
                handle.remove()
            return (
                patched[correct_id]
                - patched[counter_id]
                - baseline[correct_id]
                + baseline[counter_id]
            ).detach()
