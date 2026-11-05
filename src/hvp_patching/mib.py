#!/usr/bin/env python3
"""Pure utilities for protocol-faithful, isolated-edge MIB MS-HVP scoring.

The production runner keeps every edge intervention isolated by assigning each
edge to a separate copy of the input batch.  Consequently, loss row ``i``
depends only on scalar path coordinate ``t[i]``.  Under that invariant, two
vector-Jacobian products recover the per-edge first and second directional
derivatives without introducing cross-edge Hessian terms.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor


PROTOCOL_VERSION = "true-mib-isolated-edge-hvp-v1"
MS_HVP_PROTOCOL_VERSION = "true-mib-isolated-edge-ms-hvp-v1"


@dataclass(frozen=True)
class EdgeSpec:
    """Runtime-independent description of one official MIB graph edge."""

    name: str
    source_name: str
    source_hook: str
    source_kind: str
    source_layer: int | None
    source_head: int | None
    destination_hook: str
    destination_head: int | None

    @property
    def destination_key(self) -> tuple[str, int]:
        head = -1 if self.destination_head is None else self.destination_head
        return self.destination_hook, head


@dataclass(frozen=True)
class WorkUnit:
    """A set of isolated edges that can share one destination-hook pass."""

    index: int
    destination_hook: str
    destination_head: int | None
    edges: tuple[EdgeSpec, ...]

    @property
    def edge_names(self) -> tuple[str, ...]:
        return tuple(edge.name for edge in self.edges)


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def object_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    """Durably replace ``path`` with canonical JSON on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_edge_specs(edges: Sequence[EdgeSpec]) -> None:
    if not edges:
        raise ValueError("the graph contains no real edges")
    names = [edge.name for edge in edges]
    if len(names) != len(set(names)):
        raise ValueError("edge names are not unique")
    for edge in edges:
        if edge.source_kind not in {"input", "attention", "mlp"}:
            raise ValueError(f"unsupported source kind for {edge.name}: {edge.source_kind}")
        if edge.source_kind == "attention":
            if edge.source_layer is None or edge.source_head is None:
                raise ValueError(f"attention source metadata missing for {edge.name}")
        elif edge.source_layer is not None or edge.source_head is not None:
            if edge.source_kind == "input" or edge.source_head is not None:
                raise ValueError(f"invalid source metadata for {edge.name}")
        if edge.destination_head is not None and edge.destination_head < 0:
            raise ValueError(f"negative destination head for {edge.name}")


def build_work_units(
    edges: Sequence[EdgeSpec], edge_chunk_size: int
) -> tuple[WorkUnit, ...]:
    """Group edges by exact destination and split deterministically into chunks."""

    if edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be positive")
    validate_edge_specs(edges)
    ordered = sorted(
        edges,
        key=lambda edge: (
            edge.destination_key[0],
            edge.destination_key[1],
            edge.name,
        ),
    )
    units: list[WorkUnit] = []
    for destination_key, group in groupby(
        ordered, key=lambda edge: edge.destination_key
    ):
        grouped_edges = tuple(group)
        for start in range(0, len(grouped_edges), edge_chunk_size):
            chunk = grouped_edges[start : start + edge_chunk_size]
            units.append(
                WorkUnit(
                    index=len(units),
                    destination_hook=destination_key[0],
                    destination_head=(
                        None if destination_key[1] == -1 else destination_key[1]
                    ),
                    edges=chunk,
                )
            )
    if sum(len(unit.edges) for unit in units) != len(edges):
        raise AssertionError("work-unit construction lost edges")
    return tuple(units)


def plan_payload(
    *,
    model: str,
    task: str,
    graph_template_sha256: str,
    edge_chunk_size: int,
    units: Sequence[WorkUnit],
) -> dict[str, object]:
    edge_names = [name for unit in units for name in unit.edge_names]
    unit_records = [
        {
            "index": unit.index,
            "destination_hook": unit.destination_hook,
            "destination_head": unit.destination_head,
            "edge_names": list(unit.edge_names),
        }
        for unit in units
    ]
    body: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "HVP",
        "model": model,
        "task": task,
        "split": "train",
        "num_examples": 100,
        "graph_template_sha256": graph_template_sha256,
        "edge_chunk_size": edge_chunk_size,
        "edge_count": len(edge_names),
        "unit_count": len(units),
        "edge_order_sha256": object_sha256(edge_names),
        "units": unit_records,
    }
    body["plan_sha256"] = object_sha256(body)
    return body


def repeat_edge_major(tensor: Tensor, edge_count: int) -> Tensor:
    if edge_count < 1:
        raise ValueError("edge_count must be positive")
    repeats = (edge_count,) + (1,) * (tensor.ndim - 1)
    return tensor.repeat(repeats)


def apply_isolated_destination_updates(
    activations: Tensor,
    directions: Tensor,
    path_coordinates: Tensor,
    destination_head: int | None,
) -> Tensor:
    """Add one independent edge displacement to each replicated batch block.

    ``directions`` has shape ``[edge, batch, position, d_model]``.  The input
    activation's leading dimension must be ``edge * batch``.  Attention-input
    hooks additionally contain a head dimension, and only ``destination_head``
    is modified.
    """

    if directions.ndim != 4:
        raise ValueError(f"directions must be rank 4, got {directions.shape}")
    edge_count, batch_size, positions, hidden = directions.shape
    if path_coordinates.shape != (edge_count,):
        raise ValueError(
            "path_coordinates must have one scalar per edge: "
            f"expected {(edge_count,)}, got {tuple(path_coordinates.shape)}"
        )
    if activations.shape[0] != edge_count * batch_size:
        raise ValueError("replicated activation batch does not match directions")
    if activations.shape[1] != positions or activations.shape[-1] != hidden:
        raise ValueError("activation position/hidden dimensions do not match directions")

    scale = path_coordinates.to(dtype=activations.dtype).view(
        edge_count, 1, 1, 1
    )
    update = (directions.to(dtype=activations.dtype) * scale).reshape(
        edge_count * batch_size, positions, hidden
    )
    if destination_head is None:
        if activations.ndim != 3:
            raise ValueError(
                f"non-attention destination must be rank 3, got {activations.shape}"
            )
        return activations + update

    if activations.ndim != 4:
        raise ValueError(
            f"attention destination must be rank 4, got {activations.shape}"
        )
    if not 0 <= destination_head < activations.shape[2]:
        raise ValueError(
            f"destination head {destination_head} outside {activations.shape[2]} heads"
        )
    head_mask = torch.nn.functional.one_hot(
        torch.tensor(destination_head, device=activations.device),
        num_classes=activations.shape[2],
    ).to(dtype=activations.dtype)
    return activations + update.unsqueeze(2) * head_mask.view(1, 1, -1, 1)


def gemma_attention_endpoint_direction(
    clean_results: Tensor,
    corrupted_results: Tensor,
    source_head: int,
    output_bias: Tensor,
    post_attention_normalizer: Callable[[Tensor], Tensor],
) -> Tensor:
    """Reproduce official MIB's isolated Gemma attention-source endpoint.

    MIB changes one raw head result, reconstructs the complete attention-layer
    output (including ``b_O``), applies Gemma's post-attention normalization to
    both endpoints, and sends their difference into the destination residual
    input.
    """

    if clean_results.shape != corrupted_results.shape or clean_results.ndim != 4:
        raise ValueError("Gemma attention caches must share [batch,pos,head,hidden]")
    if not 0 <= source_head < clean_results.shape[2]:
        raise ValueError("source_head is outside the cached attention tensor")
    clean_layer_output = clean_results.sum(dim=2) + output_bias
    raw_head_delta = (
        corrupted_results[:, :, source_head] - clean_results[:, :, source_head]
    )
    clean_endpoint = post_attention_normalizer(clean_layer_output)
    patched_endpoint = post_attention_normalizer(clean_layer_output + raw_head_delta)
    return patched_endpoint - clean_endpoint


def isolated_directional_taylor_terms(
    per_edge_losses: Tensor,
    path_coordinates: Tensor,
    *,
    step_size: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return local first- and second-order contributions and their sum.

    The caller must guarantee that ``per_edge_losses[i]`` depends only on
    ``path_coordinates[i]``.  The production runner enforces this structurally
    by mapping each edge to a distinct replicated batch block.

    ``path_coordinates`` identifies the local Taylor base point on the edge's
    clean-to-corrupted activation path. ``step_size`` is the displacement from
    that base point. The default of one preserves the single-step HVP
    estimator. For an equal-width K-step estimator it is ``1 / K``.
    """

    if per_edge_losses.ndim != 1 or path_coordinates.ndim != 1:
        raise ValueError("losses and path coordinates must be vectors")
    if per_edge_losses.shape != path_coordinates.shape:
        raise ValueError("loss and path-coordinate vectors must have equal shape")
    if not path_coordinates.requires_grad:
        raise ValueError("path_coordinates must require gradients")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")

    first_order = torch.autograd.grad(
        per_edge_losses,
        path_coordinates,
        grad_outputs=torch.ones_like(per_edge_losses),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    if first_order.requires_grad:
        second_order = torch.autograd.grad(
            first_order,
            path_coordinates,
            grad_outputs=torch.ones_like(first_order),
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if second_order is None:
            second_order = torch.zeros_like(first_order)
    else:
        second_order = torch.zeros_like(first_order)
    first_contribution = step_size * first_order
    correction = 0.5 * (step_size**2) * second_order
    estimate = first_contribution.detach() + correction.detach()
    return first_contribution.detach(), correction.detach(), estimate


def finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def validate_finite_records(records: Iterable[dict[str, object]]) -> None:
    seen: set[str] = set()
    for record in records:
        name = record.get("edge_name")
        if not isinstance(name, str) or not name:
            raise ValueError("score record has no edge_name")
        if name in seen:
            raise ValueError(f"duplicate score record for {name}")
        seen.add(name)
        for key in ("first_order", "quadratic_correction", "score"):
            finite_float(record.get(key), f"{name}: {key}")
        expected = finite_float(record["first_order"], f"{name}: first_order") + finite_float(
            record["quadratic_correction"], f"{name}: quadratic_correction"
        )
        if not math.isclose(
            finite_float(record["score"], f"{name}: score"),
            expected,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name}: score is inconsistent with its terms")


def validate_ms_hvp_records(
    records: Iterable[dict[str, object]], *, reuse_source_hvp: bool
) -> None:
    """Validate all five per-edge Taylor contributions and their aggregate."""

    materialized = list(records)
    validate_finite_records(materialized)
    expected_steps = list(range(5))
    for record in materialized:
        name = str(record["edge_name"])
        terms = record.get("path_terms")
        if not isinstance(terms, list) or len(terms) != 5:
            raise ValueError(f"{name}: path_terms must contain five entries")
        step_indices: list[int] = []
        total_first = 0.0
        total_correction = 0.0
        for term in terms:
            if not isinstance(term, dict):
                raise ValueError(f"{name}: invalid path term")
            step_index = term.get("step_index")
            if not isinstance(step_index, int):
                raise ValueError(f"{name}: invalid path step index")
            step_indices.append(step_index)
            expected_start = step_index / 5.0
            if not math.isclose(
                finite_float(term.get("path_start"), f"{name}: path_start"),
                expected_start,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{name}: invalid path start for step {step_index}")
            for key in ("first_contribution", "quadratic_contribution"):
                finite_float(term.get(key), f"{name}: {key} at step {step_index}")
            expected_reuse = reuse_source_hvp and step_index == 0
            if term.get("reused_source_hvp") is not expected_reuse:
                raise ValueError(f"{name}: invalid reuse marker at step {step_index}")
            total_first += finite_float(
                term["first_contribution"], f"{name}: first contribution"
            )
            total_correction += finite_float(
                term["quadratic_contribution"], f"{name}: quadratic contribution"
            )
        if step_indices != expected_steps:
            raise ValueError(f"{name}: path steps are not exactly 0 through 4")
        if not math.isclose(
            total_first,
            finite_float(record["first_order"], f"{name}: first_order"),
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name}: first-order path terms do not sum")
        if not math.isclose(
            total_correction,
            finite_float(
                record["quadratic_correction"], f"{name}: quadratic_correction"
            ),
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name}: quadratic path terms do not sum")
        if reuse_source_hvp:
            source_first = record.get("source_hvp_first_order")
            source_correction = record.get("source_hvp_quadratic_correction")
            source_first_value = finite_float(
                source_first, f"{name}: source_hvp_first_order"
            )
            source_correction_value = finite_float(
                source_correction, f"{name}: source_hvp_quadratic_correction"
            )
            if not math.isclose(
                finite_float(
                    terms[0]["first_contribution"], f"{name}: reused first contribution"
                ),
                source_first_value / 5.0,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{name}: reused first-order term has wrong scaling")
            if not math.isclose(
                finite_float(
                    terms[0]["quadratic_contribution"],
                    f"{name}: reused quadratic contribution",
                ),
                source_correction_value / 25.0,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{name}: reused quadratic term has wrong scaling")


def edge_spec_record(edge: EdgeSpec) -> dict[str, object]:
    return asdict(edge)
