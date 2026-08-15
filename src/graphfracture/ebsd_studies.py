"""Strict reducers for the measured-EBSD convergence and control studies.

The reducers in this module deliberately validate the evidence recorded by a
solver run *before* calculating a difference.  They reject incomplete runs,
adaptive load subdivisions, off-schedule load nodes, failed KKT tolerances,
inconsistent runtime mesh diagnostics, and undeclared cross-case differences.

The fixed-mesh control is an ``attribute-location association permutation``:
the measured chain geometry and connectivity stay fixed while the joint chain
attribute rows are reassigned.  It is not a topology reconnection experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 2

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ablation_report",
    "case_audit_report",
    "convergence_report",
    "main",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required result file is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _assert_finite_json(value, path)
    return value


def _assert_finite_json(value: Any, path: Path, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: {location} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, path, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, path, f"{location}[{index}]")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _finite(value: Any, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _optional_finite(value: Any, context: str) -> float | None:
    return None if value is None else _finite(value, context)


def _integer(value: Any, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-11, abs_tol=1.0e-13)


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{context} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _csv_number(row: Mapping[str, str], name: str, path: Path, index: int) -> float:
    text = row.get(name)
    if text is None or not text.strip():
        raise ValueError(f"{path}: row {index} has no numeric {name!r}")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{path}: row {index} {name!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: row {index} {name!r} must be finite")
    return value


def _csv_integer(row: Mapping[str, str], name: str, path: Path, index: int) -> int:
    value = _csv_number(row, name, path, index)
    result = int(value)
    if value != result:
        raise ValueError(f"{path}: row {index} {name!r} must be an integer")
    return result


def _csv_boolean(row: Mapping[str, str], name: str, path: Path, index: int) -> bool:
    value = row.get(name, "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{path}: row {index} {name!r} must be true or false")


def _read_history(path: Path) -> list[dict[str, float | int | bool]]:
    required = {
        "step",
        "scheduled_step",
        "subdivision_level",
        "displacement",
        "reaction_y",
        "regularised_crack_length",
        "rightmost_damaged_x",
        "stagger_iterations",
        "stagger_converged",
        "damage_kkt_relative",
    }
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no CSV header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"{path} has duplicate CSV columns")
            missing = sorted(required - set(reader.fieldnames))
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
            raw_rows = list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required result file is missing: {path}") from exc
    if not raw_rows:
        raise ValueError(f"{path} contains no records")

    rows: list[dict[str, float | int | bool]] = []
    for index, raw in enumerate(raw_rows):
        rows.append(
            {
                "step": _csv_integer(raw, "step", path, index),
                "scheduled_step": _csv_integer(raw, "scheduled_step", path, index),
                "subdivision_level": _csv_integer(raw, "subdivision_level", path, index),
                "displacement": _csv_number(raw, "displacement", path, index),
                "reaction_y": _csv_number(raw, "reaction_y", path, index),
                "regularised_crack_length": _csv_number(
                    raw, "regularised_crack_length", path, index
                ),
                "rightmost_damaged_x": _csv_number(raw, "rightmost_damaged_x", path, index),
                "stagger_iterations": _csv_integer(raw, "stagger_iterations", path, index),
                "stagger_converged": _csv_boolean(raw, "stagger_converged", path, index),
                "damage_kkt_relative": _csv_number(raw, "damage_kkt_relative", path, index),
            }
        )
    return rows


def _validate_runtime_against_config(
    runtime: dict[str, Any], config: dict[str, Any], runtime_path: Path
) -> dict[str, Any]:
    geometry = _mapping(config.get("geometry"), f"{runtime_path}: config.geometry")
    material = _mapping(config.get("material"), f"{runtime_path}: config.material")
    graph = _mapping(config.get("graph"), f"{runtime_path}: config.graph")
    hydrogen = _mapping(config.get("hydrogen"), f"{runtime_path}: config.hydrogen")
    diagnostics = _mapping(runtime.get("diagnostics"), f"{runtime_path}: diagnostics")
    mesh = _mapping(diagnostics.get("mesh"), f"{runtime_path}: diagnostics.mesh")
    resolution = _mapping(diagnostics.get("resolution"), f"{runtime_path}: diagnostics.resolution")
    precrack = _mapping(diagnostics.get("precrack"), f"{runtime_path}: diagnostics.precrack")
    quality = _mapping(
        diagnostics.get("quality_gates"), f"{runtime_path}: diagnostics.quality_gates"
    )
    graph_diagnostics = _mapping(diagnostics.get("graphs"), f"{runtime_path}: diagnostics.graphs")
    hydrogen_diagnostics = _mapping(
        diagnostics.get("hydrogen"), f"{runtime_path}: diagnostics.hydrogen"
    )

    config_schema = _integer(config.get("schema_version"), f"{runtime_path}: config schema")
    runtime_schema = _integer(diagnostics.get("schema_version"), f"{runtime_path}: runtime schema")
    if config_schema != runtime_schema:
        raise ValueError(f"{runtime_path}: runtime schema disagrees with resolved config")
    _sha256(runtime.get("config_sha256"), f"{runtime_path}: config_sha256")

    length = _finite(geometry.get("length"), f"{runtime_path}: geometry.length")
    height = _finite(geometry.get("height"), f"{runtime_path}: geometry.height")
    nx = _integer(geometry.get("nx"), f"{runtime_path}: geometry.nx")
    ny = _integer(geometry.get("ny"), f"{runtime_path}: geometry.ny")
    if length <= 0.0 or height <= 0.0 or nx < 1 or ny < 1:
        raise ValueError(f"{runtime_path}: invalid resolved geometry")
    hx, hy = length / nx, height / ny
    diameter = math.hypot(hx, hy)
    expected_mesh = {
        "cells": 2 * nx * ny,
        "hx": hx,
        "hy": hy,
        "triangle_diameter": diameter,
    }
    for name, expected in expected_mesh.items():
        observed = _finite(mesh.get(name), f"{runtime_path}: mesh.{name}")
        if not _close(observed, float(expected)):
            raise ValueError(f"{runtime_path}: runtime mesh {name} disagrees with resolved config")
    configured_diagonal = _string(
        geometry.get("diagonal", "right"), f"{runtime_path}: geometry.diagonal"
    )
    observed_diagonal = _string(mesh.get("diagonal", "right"), f"{runtime_path}: mesh.diagonal")
    if observed_diagonal != configured_diagonal:
        raise ValueError(f"{runtime_path}: runtime mesh diagonal disagrees with resolved config")

    ell = _finite(material.get("length_scale"), f"{runtime_path}: material.length_scale")
    band = _finite(graph.get("influence_radius"), f"{runtime_path}: graph.influence_radius")
    expected_resolution = {
        "ell_over_triangle_diameter": ell / diameter,
        "gb_width_over_triangle_diameter": band / diameter,
    }
    for name, expected in expected_resolution.items():
        observed = _finite(resolution.get(name), f"{runtime_path}: resolution.{name}")
        if not _close(observed, expected):
            raise ValueError(f"{runtime_path}: runtime resolution disagrees with config")

    requested_tip = _finite(
        geometry.get("precrack_length"), f"{runtime_path}: geometry.precrack_length"
    )
    discrete_tip = math.floor(requested_tip / hx + 1.0e-12) * hx
    if not _close(
        _finite(precrack.get("requested_tip_x"), f"{runtime_path}: requested tip"),
        requested_tip,
    ) or not _close(
        _finite(precrack.get("discrete_tip_x"), f"{runtime_path}: discrete tip"),
        discrete_tip,
    ):
        raise ValueError(f"{runtime_path}: runtime precrack diagnostics disagree with config")

    expected_quality = {
        "phase_field_resolution_pass": ell / diameter >= 2.0,
        "precrack_mesh_alignment_pass": _close(requested_tip, discrete_tip),
        "grain_boundary_band_has_multiple_cells": band / diameter >= 1.5,
    }
    for name, expected in expected_quality.items():
        observed = _boolean(quality.get(name), f"{runtime_path}: quality_gates.{name}")
        if observed != expected:
            raise ValueError(f"{runtime_path}: runtime quality gate {name} disagrees with config")
        if not observed:
            raise ValueError(f"{runtime_path}: required quality gate {name} did not pass")

    graph_enabled = _boolean(graph.get("enabled"), f"{runtime_path}: graph.enabled")
    if (
        _boolean(graph_diagnostics.get("enabled"), f"{runtime_path}: diagnostics.graphs.enabled")
        != graph_enabled
    ):
        raise ValueError(f"{runtime_path}: runtime graph mode disagrees with config")
    nodes = _list(config.get("graph_nodes"), f"{runtime_path}: graph_nodes")
    edges = _list(config.get("graph_edges"), f"{runtime_path}: graph_edges")
    if _integer(graph_diagnostics.get("gb_nodes"), f"{runtime_path}: gb_nodes") != len(nodes):
        raise ValueError(f"{runtime_path}: runtime graph node count disagrees with config")
    if _integer(graph_diagnostics.get("gb_edges"), f"{runtime_path}: gb_edges") != len(edges):
        raise ValueError(f"{runtime_path}: runtime graph edge count disagrees with config")
    chain_artifact = graph.get("chain_artifact") or None
    if graph_diagnostics.get("chain_artifact") != chain_artifact:
        raise ValueError(f"{runtime_path}: runtime chain_artifact disagrees with config")
    confidence = _finite(graph.get("confidence_floor"), f"{runtime_path}: graph.confidence_floor")
    if not _close(
        _finite(
            graph_diagnostics.get("confidence_floor"),
            f"{runtime_path}: diagnostics.graphs.confidence_floor",
        ),
        confidence,
    ):
        raise ValueError(f"{runtime_path}: runtime confidence floor disagrees with config")
    seed = _integer(
        graph.get("attribute_permutation_seed"),
        f"{runtime_path}: graph.attribute_permutation_seed",
    )
    expected_seed = None if seed < 0 else seed
    if graph_diagnostics.get("attribute_permutation_seed") != expected_seed:
        raise ValueError(f"{runtime_path}: runtime permutation seed disagrees with config")

    hydrogen_enabled = _boolean(hydrogen.get("enabled"), f"{runtime_path}: hydrogen.enabled")
    if (
        _boolean(
            hydrogen_diagnostics.get("enabled"),
            f"{runtime_path}: diagnostics.hydrogen.enabled",
        )
        != hydrogen_enabled
    ):
        raise ValueError(f"{runtime_path}: runtime hydrogen mode disagrees with config")
    if hydrogen_diagnostics.get("charging_boundary") != hydrogen.get("charging_boundary"):
        raise ValueError(f"{runtime_path}: runtime charging boundary disagrees with config")
    diffusion_length = math.sqrt(
        _finite(hydrogen.get("diffusivity"), f"{runtime_path}: hydrogen.diffusivity")
        * _finite(hydrogen.get("charging_time"), f"{runtime_path}: hydrogen.charging_time")
    )
    if not _close(
        _finite(
            hydrogen_diagnostics.get("diffusion_length"),
            f"{runtime_path}: hydrogen diffusion_length",
        ),
        diffusion_length,
    ):
        raise ValueError(f"{runtime_path}: runtime hydrogen diagnostics disagree with config")

    return {
        "cells": 2 * nx * ny,
        "nx": nx,
        "ny": ny,
        "hx": hx,
        "hy": hy,
        "triangle_diameter": diameter,
        "ell_over_h_K": ell / diameter,
        "diagonal": configured_diagonal,
    }


def _runtime_identity(runtime: dict[str, Any], path: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for name in ("python", "dolfinx", "ufl", "numpy", "mpi_library", "reference_container"):
        identity[name] = _string(runtime.get(name), f"{path}: {name}")
    identity["mpi_ranks"] = _integer(runtime.get("mpi_ranks"), f"{path}: mpi_ranks")
    identity["petsc"] = deepcopy(_mapping(runtime.get("petsc"), f"{path}: petsc"))
    model = deepcopy(_mapping(runtime.get("model"), f"{path}: model"))
    diagnostics = _mapping(runtime.get("diagnostics"), f"{path}: diagnostics")
    identity["model"] = model
    identity["diagnostic_model"] = deepcopy(
        _mapping(diagnostics.get("model"), f"{path}: diagnostics.model")
    )
    return identity


def _validate_history(
    history: list[dict[str, Any]],
    config: dict[str, Any],
    completion: dict[str, Any],
    history_path: Path,
    completion_path: Path,
) -> dict[str, Any]:
    loading = _mapping(config.get("loading"), f"{history_path}: config.loading")
    steps = _integer(loading.get("steps"), f"{history_path}: loading.steps")
    maximum = _finite(
        loading.get("maximum_displacement"),
        f"{history_path}: loading.maximum_displacement",
    )
    kkt_tolerance = _finite(
        loading.get("damage_kkt_tolerance"),
        f"{history_path}: loading.damage_kkt_tolerance",
    )
    stagger_budget = _integer(
        loading.get("stagger_max_iterations"),
        f"{history_path}: loading.stagger_max_iterations",
    )
    if steps < 1 or maximum <= 0.0 or kkt_tolerance <= 0.0 or stagger_budget < 1:
        raise ValueError(f"{history_path}: invalid resolved loading controls")
    if len(history) != steps + 1:
        raise ValueError(
            f"{history_path}: history records do not equal configured loading.steps + 1"
        )

    expected_nodes = [maximum * index / steps for index in range(steps + 1)]
    for index, (row, expected) in enumerate(zip(history, expected_nodes, strict=True)):
        if row["step"] != index or row["scheduled_step"] != index:
            raise ValueError(f"{history_path}: step/scheduled_step must follow nominal nodes")
        if row["subdivision_level"] != 0:
            raise ValueError(
                f"{history_path}: adaptive load subdivisions are not allowed in strict studies"
            )
        if not _close(float(row["displacement"]), expected):
            raise ValueError(
                f"{history_path}: actual displacement sequence differs from configured "
                "nominal load nodes"
            )
        if row["stagger_converged"] is not True:
            raise ValueError(f"{history_path}: contains a non-converged staggered solve")
        iterations = int(row["stagger_iterations"])
        if iterations < 0 or iterations > stagger_budget:
            raise ValueError(f"{history_path}: stagger iteration count is outside its budget")
        kkt = float(row["damage_kkt_relative"])
        if kkt < 0.0 or kkt > kkt_tolerance * (1.0 + 1.0e-12):
            raise ValueError(
                f"{history_path}: damage KKT relative residual {kkt:.6g} exceeds "
                f"configured tolerance {kkt_tolerance:.6g}"
            )

    if completion.get("status") != "complete" or completion.get("all_steps_converged") is not True:
        raise ValueError(f"{completion_path}: marker does not certify a complete run")
    accepted = _integer(
        completion.get("accepted_load_steps"), f"{completion_path}: accepted_load_steps"
    )
    if accepted != steps:
        raise ValueError(f"{completion_path}: accepted steps disagree with config/history")
    final = _finite(completion.get("final_displacement"), f"{completion_path}: final_displacement")
    if not _close(final, maximum):
        raise ValueError(f"{completion_path}: final displacement disagrees with config/history")

    maximum_kkt = max(float(row["damage_kkt_relative"]) for row in history)
    maximum_iterations = max(int(row["stagger_iterations"]) for row in history)
    return {
        "nodes": expected_nodes,
        "configured_steps": steps,
        "maximum_damage_kkt_relative": maximum_kkt,
        "damage_kkt_tolerance": kkt_tolerance,
        "maximum_stagger_iterations": maximum_iterations,
        "stagger_iteration_budget": stagger_budget,
    }


def _validate_grain_boundary_graph(
    graph_metrics: dict[str, Any],
    config: dict[str, Any],
    graph_path: Path,
) -> dict[str, Any] | None:
    graph = _mapping(config.get("graph"), f"{graph_path}: config.graph")
    enabled = _boolean(graph.get("enabled"), f"{graph_path}: graph.enabled")
    gb = _mapping(graph_metrics.get("G_GB"), f"{graph_path}: G_GB")
    if not enabled:
        if gb.get("enabled") is not False:
            raise ValueError(f"{graph_path}: disabled graph case has active G_GB metadata")
        return None
    if gb.get("kind") != "G_GB_chains":
        raise ValueError(f"{graph_path}: EBSD study requires a G_GB_chains field")
    artifact = _string(graph.get("chain_artifact"), f"{graph_path}: chain_artifact")
    if not artifact.lower().endswith(".npz"):
        raise ValueError(f"{graph_path}: chain_artifact must be an NPZ artifact")

    provenance = _mapping(gb.get("provenance"), f"{graph_path}: G_GB.provenance")
    semantic_sha = _sha256(
        provenance.get("artifact_semantic_sha256"),
        f"{graph_path}: artifact_semantic_sha256",
    )
    artifact_sha = _sha256(
        provenance.get("artifact_sha256"),
        f"{graph_path}: artifact_sha256",
    )
    ctf_sha = _sha256(
        provenance.get("ctf_sha256"),
        f"{graph_path}: ctf_sha256",
    )
    recorded_path = _string(
        provenance.get("chain_artifact"), f"{graph_path}: provenance.chain_artifact"
    )
    if Path(recorded_path).name != Path(artifact).name:
        raise ValueError(f"{graph_path}: recorded artifact path disagrees with config")
    manifest_path = _string(
        provenance.get("chain_artifact_manifest"),
        f"{graph_path}: provenance.chain_artifact_manifest",
    )
    if Path(manifest_path).name != Path(artifact).with_suffix(".json").name:
        raise ValueError(f"{graph_path}: recorded artifact manifest disagrees with config")
    coordinate_units = _string(
        provenance.get("coordinate_units"), f"{graph_path}: coordinate_units"
    )
    coordinate_convention = _string(
        provenance.get("coordinate_convention"),
        f"{graph_path}: coordinate_convention",
    )
    coordinate_min_raw = _list(provenance.get("coordinate_min"), f"{graph_path}: coordinate_min")
    coordinate_max_raw = _list(provenance.get("coordinate_max"), f"{graph_path}: coordinate_max")
    if len(coordinate_min_raw) != 2 or len(coordinate_max_raw) != 2:
        raise ValueError(f"{graph_path}: coordinate extent must contain two 2-D points")
    coordinate_min = [
        _finite(value, f"{graph_path}: coordinate_min[{index}]")
        for index, value in enumerate(coordinate_min_raw)
    ]
    coordinate_max = [
        _finite(value, f"{graph_path}: coordinate_max[{index}]")
        for index, value in enumerate(coordinate_max_raw)
    ]
    if any(lower > upper for lower, upper in zip(coordinate_min, coordinate_max, strict=True)):
        raise ValueError(f"{graph_path}: coordinate_min exceeds coordinate_max")
    source_chains = _integer(provenance.get("source_chains"), f"{graph_path}: source_chains")
    kept_chains = _integer(provenance.get("kept_chains"), f"{graph_path}: kept_chains")
    excluded = _integer(
        provenance.get("excluded_low_confidence_chains"),
        f"{graph_path}: excluded_low_confidence_chains",
    )
    if source_chains < 1 or kept_chains < 1 or source_chains != kept_chains + excluded:
        raise ValueError(f"{graph_path}: inconsistent source/kept/excluded chain counts")
    confidence = _finite(
        provenance.get("confidence_floor"), f"{graph_path}: provenance.confidence_floor"
    )
    if not _close(
        confidence,
        _finite(graph.get("confidence_floor"), f"{graph_path}: graph.confidence_floor"),
    ):
        raise ValueError(f"{graph_path}: applied confidence floor disagrees with config")
    config_seed = _integer(
        graph.get("attribute_permutation_seed"), f"{graph_path}: permutation seed"
    )
    expected_seed = None if config_seed < 0 else config_seed
    if provenance.get("attribute_permutation_seed") != expected_seed:
        raise ValueError(f"{graph_path}: applied permutation seed disagrees with config")
    permutation_enabled = _boolean(
        provenance.get("attribute_location_permutation"),
        f"{graph_path}: attribute_location_permutation",
    )
    if permutation_enabled != (expected_seed is not None):
        raise ValueError(f"{graph_path}: permutation flag disagrees with config")
    permutation_sha = _sha256(
        provenance.get("attribute_location_permutation_sha256"),
        f"{graph_path}: attribute_location_permutation_sha256",
    )

    segments = _integer(gb.get("segments"), f"{graph_path}: G_GB.segments")
    parent_chains = _integer(gb.get("parent_chains"), f"{graph_path}: G_GB.parent_chains")
    total_length = _finite(gb.get("total_boundary_length"), f"{graph_path}: total_boundary_length")
    influence_radius = _finite(gb.get("influence_radius"), f"{graph_path}: G_GB.influence_radius")
    if segments < 1 or parent_chains != kept_chains or total_length <= 0.0:
        raise ValueError(f"{graph_path}: G_GB geometry must contain positive segments/length")
    if not _close(
        influence_radius,
        _finite(graph.get("influence_radius"), f"{graph_path}: graph.influence_radius"),
    ):
        raise ValueError(f"{graph_path}: applied boundary width disagrees with config")
    index_cells = _list(gb.get("index_cells"), f"{graph_path}: G_GB.index_cells")
    if len(index_cells) != 2 or any(type(value) is not int or value < 1 for value in index_cells):
        raise ValueError(f"{graph_path}: G_GB.index_cells must contain two positive integers")

    return {
        "artifact_semantic_sha256": semantic_sha,
        "artifact_sha256": artifact_sha,
        "ctf_sha256": ctf_sha,
        "coordinate_units": coordinate_units,
        "coordinate_convention": coordinate_convention,
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "source_chains": source_chains,
        "kept_chains": kept_chains,
        "excluded_low_confidence_chains": excluded,
        "confidence_floor": confidence,
        "segments": segments,
        "parent_chains": parent_chains,
        "total_boundary_length": total_length,
        "influence_radius": influence_radius,
        "index_cells": index_cells,
        "attribute_distribution_signature": {
            "minimum_segment_toughness_ratio": _finite(
                gb.get("minimum_segment_toughness_ratio"),
                f"{graph_path}: minimum_segment_toughness_ratio",
            ),
            "maximum_segment_diffusivity_ratio": _finite(
                gb.get("maximum_segment_diffusivity_ratio"),
                f"{graph_path}: maximum_segment_diffusivity_ratio",
            ),
            "sum_parent_chain_trap_density": _finite(
                gb.get("sum_parent_chain_trap_density"),
                f"{graph_path}: sum_parent_chain_trap_density",
            ),
        },
        "attribute_location_permutation": permutation_enabled,
        "attribute_permutation_seed": expected_seed,
        "attribute_location_permutation_sha256": permutation_sha,
    }


def _validate_crack_graph(
    graph_metrics: dict[str, Any], config: dict[str, Any], graph_path: Path
) -> dict[str, Any]:
    crack = _mapping(graph_metrics.get("G_crack"), f"{graph_path}: G_crack")
    graph = _mapping(config.get("graph"), f"{graph_path}: config.graph")
    threshold = _finite(graph.get("crack_threshold"), f"{graph_path}: crack_threshold")
    if not _close(_finite(crack.get("threshold"), f"{graph_path}: G_crack.threshold"), threshold):
        raise ValueError(f"{graph_path}: G_crack threshold disagrees with config")
    return {
        "threshold": threshold,
        "tip_x": _optional_finite(crack.get("tip_x"), f"{graph_path}: G_crack.tip_x"),
        "tortuosity": _optional_finite(
            crack.get("tortuosity"), f"{graph_path}: G_crack.tortuosity"
        ),
        "spans_left_to_right": _boolean(
            crack.get("spans_left_to_right"),
            f"{graph_path}: G_crack.spans_left_to_right",
        ),
    }


def _validate_case(case_directory: str | Path) -> dict[str, Any]:
    directory = Path(case_directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {directory}")
    config_path = directory / "config.resolved.json"
    runtime_path = directory / "runtime.json"
    completion_path = directory / "completion.json"
    graph_path = directory / "graph_metrics.json"
    history_path = directory / "history.csv"
    config = _read_json(config_path)
    runtime = _read_json(runtime_path)
    completion = _read_json(completion_path)
    graph_metrics = _read_json(graph_path)
    history = _read_history(history_path)

    mesh = _validate_runtime_against_config(runtime, config, runtime_path)
    loading = _validate_history(history, config, completion, history_path, completion_path)
    artifact = _validate_grain_boundary_graph(graph_metrics, config, graph_path)
    crack = _validate_crack_graph(graph_metrics, config, graph_path)

    reactions = [float(row["reaction_y"]) for row in history]
    maximum_index = max(range(len(history)), key=reactions.__getitem__)
    displacements = [float(row["displacement"]) for row in history]
    external_work = sum(
        0.5
        * (reactions[index - 1] + reactions[index])
        * (displacements[index] - displacements[index - 1])
        for index in range(1, len(history))
    )
    quantities = {
        "window_maximum_reaction_y": reactions[maximum_index],
        "window_maximum_reaction_displacement": displacements[maximum_index],
        "window_maximum_reaction_at_right_endpoint": maximum_index == len(history) - 1,
        "final_regularised_crack_length": float(history[-1]["regularised_crack_length"]),
        "final_main_crack_tip_x": crack["tip_x"],
        "crack_tortuosity": crack["tortuosity"],
        "crack_spans_left_to_right": crack["spans_left_to_right"],
        "external_work_over_observed_window": external_work,
    }
    positive_rows = [row for row in history if float(row["displacement"]) > 0.0]
    first_loaded = positive_rows[0]
    final_loaded = positive_rows[-1]
    first_secant = float(first_loaded["reaction_y"]) / float(first_loaded["displacement"])
    final_secant = float(final_loaded["reaction_y"]) / float(final_loaded["displacement"])
    first_incremental = (reactions[1] - reactions[0]) / (displacements[1] - displacements[0])
    final_incremental = (reactions[-1] - reactions[-2]) / (displacements[-1] - displacements[-2])
    initial_crack_length = float(history[0]["regularised_crack_length"])
    final_crack_length = float(history[-1]["regularised_crack_length"])
    secant_loss = None if abs(first_secant) < 1.0e-30 else 1.0 - final_secant / first_secant
    incremental_loss = (
        None if abs(first_incremental) < 1.0e-30 else 1.0 - final_incremental / first_incremental
    )
    crack_length_change = (
        None
        if abs(initial_crack_length) < 1.0e-30
        else (final_crack_length - initial_crack_length) / initial_crack_length
    )
    response_diagnostics = {
        "initial_secant_stiffness": first_secant,
        "final_secant_stiffness": final_secant,
        "relative_secant_stiffness_loss": secant_loss,
        "first_incremental_stiffness": first_incremental,
        "final_incremental_stiffness": final_incremental,
        "relative_incremental_stiffness_loss": incremental_loss,
        "relative_regularised_crack_length_change": crack_length_change,
        "threshold_front_advance": (
            float(history[-1]["rightmost_damaged_x"]) - float(history[0]["rightmost_damaged_x"])
        ),
    }
    return {
        "case": directory.name,
        "directory": str(directory),
        "config": config,
        "runtime": runtime,
        "runtime_identity": _runtime_identity(runtime, runtime_path),
        "mesh": mesh,
        "loading_validation": loading,
        "artifact_identity": artifact,
        "quantities": quantities,
        "response_diagnostics": response_diagnostics,
        "graph_enabled": bool(_mapping(config["graph"], "graph")["enabled"]),
        "permutation_seed": int(_mapping(config["graph"], "graph")["attribute_permutation_seed"]),
    }


def _pop_nested(value: dict[str, Any], *path: str) -> None:
    current = value
    for key in path[:-1]:
        current = _mapping(current.get(key), ".".join(path[:-1]))
    current.pop(path[-1], None)


def _canonical_config(config: dict[str, Any], study: str) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("source_path", None)
    _pop_nested(value, "output", "directory")
    if study == "convergence":
        _pop_nested(value, "geometry", "nx")
        _pop_nested(value, "geometry", "ny")
    elif study == "association_permutation":
        _pop_nested(value, "graph", "enabled")
        _pop_nested(value, "graph", "chain_artifact")
        _pop_nested(value, "graph", "attribute_permutation_seed")
    else:  # pragma: no cover - internal programming error
        raise RuntimeError(f"unknown study {study}")
    return value


def _first_difference(left: Any, right: Any, location: str = "$") -> str | None:
    if type(left) is not type(right):
        return location
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{location}.{sorted(set(left) ^ set(right))[0]}"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{location}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{location}.length"
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(first, second, f"{location}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else location


def _require_equal(values: Sequence[Any], label: str) -> None:
    reference = values[0]
    for value in values[1:]:
        difference = _first_difference(reference, value)
        if difference is not None:
            raise ValueError(f"{label} differ at {difference}")


def _same_nodes(cases: Sequence[dict[str, Any]]) -> None:
    reference = cases[0]["loading_validation"]["nodes"]
    for case in cases[1:]:
        nodes = case["loading_validation"]["nodes"]
        if len(nodes) != len(reference) or any(
            not _close(float(observed), float(expected))
            for observed, expected in zip(nodes, reference, strict=True)
        ):
            raise ValueError("actual nominal load nodes differ across cases")


def _case_record(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case["case"],
        "directory": case["directory"],
        "mesh": case["mesh"],
        "loading_validation": {
            key: value for key, value in case["loading_validation"].items() if key != "nodes"
        },
        "artifact_identity": case["artifact_identity"],
        "quantities": case["quantities"],
        "response_diagnostics": case["response_diagnostics"],
    }


def _relative_absolute_difference(value: Any, reference: Any) -> float | None:
    if value is None or reference is None:
        return None
    scale = max(abs(float(reference)), 1.0e-30)
    return abs(float(value) - float(reference)) / scale


def _difference(value: Any, reference: Any) -> float | None:
    if value is None or reference is None:
        return None
    return float(value) - float(reference)


_CONVERGENCE_QUANTITIES = (
    "window_maximum_reaction_y",
    "window_maximum_reaction_displacement",
    "final_regularised_crack_length",
    "final_main_crack_tip_x",
    "external_work_over_observed_window",
)


def case_audit_report(case_directory: str | Path) -> dict[str, Any]:
    """Strictly validate one measured-EBSD result without inventing a comparison."""
    case = _validate_case(case_directory)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study": "measured_ebsd_single_case_numerical_validation",
        "all_checks_passed": True,
        "case": _case_record(case),
        "validation": {
            "completion_marker_required": True,
            "damage_kkt_checked_against_config": True,
            "zero_adaptive_subdivisions_required": True,
            "configured_and_actual_nominal_load_nodes_verified": True,
            "runtime_mesh_and_provenance_verified": True,
        },
        "limitations": [
            "A single validated run is not a mesh-convergence, material-calibration, "
            "control, uncertainty or statistical study.",
            "Reaction is the signed maximum observed inside the configured displacement "
            "window; a right-endpoint maximum is window-censored, not a global peak.",
            "The threshold front and final connected crack graph are numerical diagnostics, "
            "not experimental crack-onset evidence.",
        ],
    }


def convergence_report(case_directories: list[str | Path]) -> dict[str, Any]:
    """Validate and compare one measured material on two or more FE meshes."""
    if len(case_directories) < 2:
        raise ValueError("a convergence study needs at least two meshes")
    cases = [_validate_case(directory) for directory in case_directories]
    if any(not case["graph_enabled"] or case["artifact_identity"] is None for case in cases):
        raise ValueError("mesh convergence cases must all use measured G_GB_chains fields")
    _require_equal(
        [_canonical_config(case["config"], "convergence") for case in cases],
        "convergence resolved configs outside the nx/ny/output allowlist",
    )
    _require_equal(
        [case["runtime_identity"] for case in cases],
        "convergence runtime environments",
    )
    _require_equal(
        [case["artifact_identity"] for case in cases],
        "convergence artifact identities",
    )
    _same_nodes(cases)
    grid_sizes = [(case["mesh"]["nx"], case["mesh"]["ny"]) for case in cases]
    if len(set(grid_sizes)) != len(grid_sizes):
        raise ValueError("convergence cases must use distinct nx/ny meshes")

    cases.sort(key=lambda case: case["mesh"]["ell_over_h_K"])
    reference = cases[-1]["quantities"]
    records = []
    for case in cases:
        record = _case_record(case)
        record["relative_absolute_difference_from_finest"] = {
            name: _relative_absolute_difference(case["quantities"][name], reference[name])
            for name in _CONVERGENCE_QUANTITIES
        }
        records.append(record)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study": "measured_ebsd_mesh_convergence",
        "reference": "finest accepted mesh (largest ell/h_K)",
        "validation": {
            "complete_runs_required": True,
            "damage_kkt_checked_against_each_config": True,
            "zero_adaptive_subdivisions_required": True,
            "configured_and_actual_nominal_load_nodes_verified": True,
            "config_difference_allowlist": [
                "geometry.nx",
                "geometry.ny",
                "output.directory",
                "source_path",
            ],
            "runtime_environment_identity_required": True,
            "artifact_identity_required": True,
        },
        "all_checks_passed": True,
        "cases": records,
        "limitations": [
            "Reaction is the signed maximum observed inside the configured displacement window; "
            "it is not called a global peak.",
            "A maximum at the right endpoint is window-censored and requires a longer load "
            "window before a post-peak claim.",
            "The report contains deterministic discretisation differences, not statistical "
            "uncertainty or significance.",
            "Connected main-crack topology is available only for the final recorded state.",
        ],
    }


def _association_runtime_identity(case: dict[str, Any]) -> dict[str, Any]:
    identity = deepcopy(case["runtime_identity"])
    # A runtime may describe graph coupling differently when the graph factor is
    # disabled.  This one descriptive string is an explicitly allowed factor
    # consequence; mechanics, libraries, MPI and the diagnostic model remain fixed.
    _mapping(identity.get("model"), "runtime identity model").pop("graph_coupling", None)
    return identity


def ablation_report(
    homogeneous: str | Path, original: str | Path, shuffled: str | Path
) -> dict[str, Any]:
    """Validate the fixed-topology attribute-location association control."""
    cases = {
        "homogeneous": _validate_case(homogeneous),
        "original": _validate_case(original),
        "permuted": _validate_case(shuffled),
    }
    ordered = [cases["homogeneous"], cases["original"], cases["permuted"]]
    if cases["homogeneous"]["graph_enabled"]:
        raise ValueError("homogeneous control must have graph.enabled = false")
    if not cases["original"]["graph_enabled"] or cases["original"]["permutation_seed"] >= 0:
        raise ValueError("original case must use an unpermuted measured chain artifact")
    if not cases["permuted"]["graph_enabled"] or cases["permuted"]["permutation_seed"] < 0:
        raise ValueError("permuted case must use a non-negative attribute permutation seed")

    _require_equal(
        [_canonical_config(case["config"], "association_permutation") for case in ordered],
        "control resolved configs outside the declared graph-factor/output allowlist",
    )
    _require_equal(
        [_association_runtime_identity(case) for case in ordered],
        "control runtime environments",
    )
    _same_nodes(ordered)
    if len({(case["mesh"]["nx"], case["mesh"]["ny"]) for case in ordered}) != 1:
        raise ValueError("association permutation control requires an identical FE mesh")

    original_artifact = cases["original"]["artifact_identity"]
    permuted_artifact = cases["permuted"]["artifact_identity"]
    assert original_artifact is not None and permuted_artifact is not None
    original_geometry = {
        key: original_artifact[key]
        for key in (
            "artifact_semantic_sha256",
            "artifact_sha256",
            "ctf_sha256",
            "coordinate_units",
            "coordinate_convention",
            "coordinate_min",
            "coordinate_max",
            "source_chains",
            "kept_chains",
            "excluded_low_confidence_chains",
            "confidence_floor",
            "segments",
            "parent_chains",
            "total_boundary_length",
            "influence_radius",
            "index_cells",
        )
    }
    permuted_geometry = {key: permuted_artifact[key] for key in original_geometry}
    difference = _first_difference(original_geometry, permuted_geometry)
    if difference is not None:
        raise ValueError(
            f"original and permuted cases do not share artifact/kept-chain geometry at {difference}"
        )
    if (
        original_artifact["attribute_distribution_signature"]
        != permuted_artifact["attribute_distribution_signature"]
    ):
        raise ValueError(
            "original and permuted cases do not preserve the recorded attribute distribution"
        )
    if (
        original_artifact["attribute_location_permutation_sha256"]
        == permuted_artifact["attribute_location_permutation_sha256"]
    ):
        raise ValueError(
            "permuted case records the same attribute-location assignment as the original"
        )

    quantities = {name: case["quantities"] for name, case in cases.items()}
    contrast_keys = (
        "window_maximum_reaction_y",
        "final_regularised_crack_length",
        "final_main_crack_tip_x",
        "crack_tortuosity",
        "external_work_over_observed_window",
    )

    def contrast(candidate: str, reference: str) -> dict[str, float | None]:
        return {
            key: _difference(quantities[candidate][key], quantities[reference][key])
            for key in contrast_keys
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study": "fixed_topology_attribute_location_association_permutation_control",
        "validation": {
            "complete_runs_required": True,
            "damage_kkt_checked_against_each_config": True,
            "zero_adaptive_subdivisions_required": True,
            "identical_actual_nominal_load_nodes_required": True,
            "identical_mesh_required": True,
            "config_difference_allowlist": [
                "graph.enabled",
                "graph.chain_artifact",
                "graph.attribute_permutation_seed",
                "output.directory",
                "source_path",
            ],
            "runtime_environment_identity_required": True,
            "original_permuted_artifact_and_ctf_hash_identity_required": True,
            "original_permuted_kept_chain_geometry_identity_required": True,
            "permutation_assignment_hash_recorded": True,
        },
        "all_checks_passed": True,
        "cases": {name: _case_record(case) for name, case in cases.items()},
        "contrasts": {
            "network_original_minus_homogeneous": contrast("original", "homogeneous"),
            "network_permuted_minus_homogeneous": contrast("permuted", "homogeneous"),
            "attribute_location_association_original_minus_permuted": contrast(
                "original", "permuted"
            ),
        },
        "interpretation": (
            "Original versus permuted measures sensitivity to the association between joint "
            "chain attributes and their locations on one fixed measured geometry/connectivity."
        ),
        "limitations": [
            "This control does not reconnect, rewire, or otherwise change graph topology.",
            "The permutation preserves the recorded chain-level joint attributes but need not "
            "preserve a boundary-length-weighted or DG0 field histogram.",
            "One deterministic permutation seed is not an ensemble and supports no statistical "
            "significance claim.",
            "Reaction is the signed maximum observed inside the configured displacement window; "
            "a right-endpoint maximum is window-censored, not a demonstrated global peak.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphfracture-ebsd-studies",
        description="Strictly validate and reduce EBSD convergence/control result directories.",
    )
    subparsers = parser.add_subparsers(dest="study", required=True)
    validation = subparsers.add_parser("validate", help="strict single-case numerical audit")
    validation.add_argument("case", type=Path)
    validation.add_argument("--output", "-o", type=Path)
    convergence = subparsers.add_parser("convergence", help="strict ell/h_K refinement report")
    convergence.add_argument("cases", nargs="+", type=Path)
    convergence.add_argument("--output", "-o", type=Path)
    association = subparsers.add_parser(
        "ablation", help="fixed-topology attribute-location association permutation control"
    )
    association.add_argument("homogeneous", type=Path)
    association.add_argument("original", type=Path)
    association.add_argument("shuffled", type=Path)
    association.add_argument("--output", "-o", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.study == "validate":
            payload = case_audit_report(args.case)
        elif args.study == "convergence":
            payload = convergence_report(args.cases)
        else:
            payload = ablation_report(args.homogeneous, args.original, args.shuffled)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        print(f"ebsd-studies error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
