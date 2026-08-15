"""Strict post-processing for the graph x hydrogen 2x2 onset study.

The comparison deliberately has no DOLFINx dependency.  It accepts exactly
four completed result directories and checks that their recorded resolved
inputs and runtime fields differ only in ``graph.enabled`` and
``hydrogen.enabled`` before calculating deterministic effect sizes. A
separate, explicit opt-in can permit only ``loading.stagger_max_iterations``
to differ, subject to a strict unused-iteration-margin check in every case.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

COMPARISON_SCHEMA_VERSION = 1
FACTOR_LEVELS = ((False, False), (False, True), (True, False), (True, True))

__all__ = ["COMPARISON_SCHEMA_VERSION", "compare_cases", "main"]


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


def _bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _finite(value: Any, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer(value: Any, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-11, abs_tol=1.0e-13)


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
    integer = int(value)
    if value != integer:
        raise ValueError(f"{path}: row {index} {name!r} must be an integer")
    return integer


def _csv_bool(row: Mapping[str, str], name: str, path: Path, index: int) -> bool:
    text = row.get(name, "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{path}: row {index} {name!r} must be true or false")


def _read_history(path: Path) -> list[dict[str, float | int | bool]]:
    required = {
        "step",
        "displacement",
        "reaction_y",
        "regularised_crack_length",
        "rightmost_damaged_x",
        "stagger_converged",
        "stagger_iterations",
        "subdivision_level",
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
        raise ValueError(f"{path} contains no data rows")

    rows: list[dict[str, float | int | bool]] = []
    for index, raw in enumerate(raw_rows):
        rows.append(
            {
                "step": _csv_integer(raw, "step", path, index),
                "displacement": _csv_number(raw, "displacement", path, index),
                "reaction_y": _csv_number(raw, "reaction_y", path, index),
                "regularised_crack_length": _csv_number(
                    raw, "regularised_crack_length", path, index
                ),
                "rightmost_damaged_x": _csv_number(raw, "rightmost_damaged_x", path, index),
                "stagger_converged": _csv_bool(raw, "stagger_converged", path, index),
                "stagger_iterations": _csv_integer(raw, "stagger_iterations", path, index),
                "subdivision_level": _csv_integer(raw, "subdivision_level", path, index),
            }
        )
    return rows


def _canonical_nodes(value: Any, context: str) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(_list(value, context)):
        node = _mapping(raw, f"{context}[{index}]")
        name = node.get("name")
        point = node.get("point")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context}[{index}].name must be a non-empty string")
        if name in names:
            raise ValueError(f"{context} contains duplicate node {name!r}")
        names.add(name)
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{context}[{index}].point must contain two numbers")
        canonical = deepcopy(node)
        canonical["point"] = [
            _finite(point[0], f"{context}[{index}].point[0]"),
            _finite(point[1], f"{context}[{index}].point[1]"),
        ]
        nodes.append(canonical)
    return sorted(nodes, key=lambda node: node["name"])


def _canonical_edges(value: Any, context: str) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    endpoints_seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(_list(value, context)):
        edge = _mapping(raw, f"{context}[{index}]")
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"{context}[{index}] endpoints must be strings")
        endpoints = tuple(sorted((source, target)))
        if endpoints[0] == endpoints[1]:
            raise ValueError(f"{context}[{index}] must not be a self-loop")
        if endpoints in endpoints_seen:
            raise ValueError(f"{context} contains duplicate edge {endpoints!r}")
        endpoints_seen.add(endpoints)
        canonical = deepcopy(edge)
        canonical["source"], canonical["target"] = endpoints
        edges.append(canonical)
    return sorted(edges, key=lambda edge: (edge["source"], edge["target"]))


def _canonical_config(config: dict[str, Any], path: Path) -> tuple[dict[str, Any], bool, bool]:
    value = deepcopy(config)
    graph = _mapping(value.get("graph"), f"{path}: graph")
    hydrogen = _mapping(value.get("hydrogen"), f"{path}: hydrogen")
    graph_enabled = _bool(graph.get("enabled"), f"{path}: graph.enabled")
    hydrogen_enabled = _bool(hydrogen.get("enabled"), f"{path}: hydrogen.enabled")
    graph.pop("enabled")
    hydrogen.pop("enabled")

    value["graph_nodes"] = _canonical_nodes(value.get("graph_nodes"), f"{path}: graph_nodes")
    value["graph_edges"] = _canonical_edges(value.get("graph_edges"), f"{path}: graph_edges")
    value.pop("source_path", None)
    output = _mapping(value.get("output"), f"{path}: output")
    output.pop("directory", None)
    return value, graph_enabled, hydrogen_enabled


def _canonical_runtime(runtime: dict[str, Any], path: Path) -> dict[str, Any]:
    value = deepcopy(runtime)
    value.pop("config_sha256", None)
    model = _mapping(value.get("model"), f"{path}: runtime.model")
    model.pop("hydrogen_coupling", None)
    diagnostics = _mapping(value.get("diagnostics"), f"{path}: runtime.diagnostics")
    diagnostics.pop("output_directory", None)
    graphs = _mapping(diagnostics.get("graphs"), f"{path}: runtime.diagnostics.graphs")
    graphs.pop("enabled", None)
    hydrogen = _mapping(diagnostics.get("hydrogen"), f"{path}: runtime.diagnostics.hydrogen")
    hydrogen.pop("enabled", None)
    return value


def _first_difference(left: Any, right: Any, location: str = "$") -> str | None:
    if type(left) is not type(right):
        return location
    if isinstance(left, dict):
        if set(left) != set(right):
            missing = sorted(set(left) ^ set(right))[0]
            return f"{location}.{missing}"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{location}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{location}.length"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{location}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else location


def _validate_runtime_against_config(
    runtime: dict[str, Any], config: dict[str, Any], path: Path, graph: bool, hydrogen: bool
) -> tuple[float, float]:
    geometry = _mapping(config.get("geometry"), f"{path}: geometry")
    material = _mapping(config.get("material"), f"{path}: material")
    graph_config = _mapping(config.get("graph"), f"{path}: graph")
    hydrogen_config = _mapping(config.get("hydrogen"), f"{path}: hydrogen")
    diagnostics = _mapping(runtime.get("diagnostics"), f"{path}: runtime.diagnostics")
    mesh = _mapping(diagnostics.get("mesh"), f"{path}: runtime.diagnostics.mesh")
    precrack = _mapping(diagnostics.get("precrack"), f"{path}: runtime.diagnostics.precrack")
    graph_diagnostics = _mapping(diagnostics.get("graphs"), f"{path}: runtime.diagnostics.graphs")
    hydrogen_diagnostics = _mapping(
        diagnostics.get("hydrogen"), f"{path}: runtime.diagnostics.hydrogen"
    )
    quality = _mapping(
        diagnostics.get("quality_gates"), f"{path}: runtime.diagnostics.quality_gates"
    )
    resolution = _mapping(diagnostics.get("resolution"), f"{path}: runtime.diagnostics.resolution")

    config_schema = _integer(config.get("schema_version"), f"{path}: config schema_version")
    runtime_schema = _integer(
        diagnostics.get("schema_version"), f"{path}: runtime diagnostics schema_version"
    )
    if runtime_schema != config_schema:
        raise ValueError(f"{path}: runtime diagnostics schema disagrees with resolved config")

    length = _finite(geometry.get("length"), f"{path}: geometry.length")
    height = _finite(geometry.get("height"), f"{path}: geometry.height")
    nx = _integer(geometry.get("nx"), f"{path}: geometry.nx")
    ny = _integer(geometry.get("ny"), f"{path}: geometry.ny")
    hx, hy = length / nx, height / ny
    expected = {
        "cells": 2 * nx * ny,
        "hx": hx,
        "hy": hy,
        "triangle_diameter": math.hypot(hx, hy),
    }
    for name, expected_value in expected.items():
        observed = _finite(mesh.get(name), f"{path}: runtime.diagnostics.mesh.{name}")
        if not _close(observed, float(expected_value)):
            raise ValueError(
                f"{path}: runtime mesh {name}={observed!r} is inconsistent with "
                f"resolved config value {expected_value!r}"
            )
    configured_diagonal = _string(geometry.get("diagonal", "right"), f"{path}: geometry.diagonal")
    observed_diagonal = _string(
        mesh.get("diagonal", "right"), f"{path}: runtime.diagnostics.mesh.diagonal"
    )
    if observed_diagonal != configured_diagonal:
        raise ValueError(f"{path}: runtime mesh diagonal is inconsistent with resolved config")

    requested_tip = _finite(geometry.get("precrack_length"), f"{path}: geometry.precrack_length")
    observed_requested = _finite(
        precrack.get("requested_tip_x"),
        f"{path}: runtime.diagnostics.precrack.requested_tip_x",
    )
    expected_discrete = math.floor(requested_tip / hx + 1.0e-12) * hx
    observed_discrete = _finite(
        precrack.get("discrete_tip_x"),
        f"{path}: runtime.diagnostics.precrack.discrete_tip_x",
    )
    if not _close(observed_requested, requested_tip) or not _close(
        observed_discrete, expected_discrete
    ):
        raise ValueError(f"{path}: runtime precrack diagnostics disagree with resolved config")

    if _bool(graph_diagnostics.get("enabled"), f"{path}: runtime graph enabled") != graph:
        raise ValueError(f"{path}: runtime graph factor disagrees with resolved config")
    if _integer(graph_diagnostics.get("gb_nodes"), f"{path}: runtime gb_nodes") != len(
        _list(config.get("graph_nodes"), f"{path}: graph_nodes")
    ):
        raise ValueError(f"{path}: runtime graph node count disagrees with resolved config")
    if _integer(graph_diagnostics.get("gb_edges"), f"{path}: runtime gb_edges") != len(
        _list(config.get("graph_edges"), f"{path}: graph_edges")
    ):
        raise ValueError(f"{path}: runtime graph edge count disagrees with resolved config")
    if _bool(hydrogen_diagnostics.get("enabled"), f"{path}: runtime hydrogen enabled") != hydrogen:
        raise ValueError(f"{path}: runtime hydrogen factor disagrees with resolved config")
    if hydrogen_diagnostics.get("charging_boundary") != hydrogen_config.get("charging_boundary"):
        raise ValueError(f"{path}: runtime hydrogen boundary disagrees with resolved config")
    diffusion_length = math.sqrt(
        _finite(hydrogen_config.get("diffusivity"), f"{path}: hydrogen.diffusivity")
        * _finite(hydrogen_config.get("charging_time"), f"{path}: hydrogen.charging_time")
    )
    observed_diffusion_length = _finite(
        hydrogen_diagnostics.get("diffusion_length"),
        f"{path}: runtime hydrogen diffusion_length",
    )
    if not _close(observed_diffusion_length, diffusion_length):
        raise ValueError(f"{path}: runtime hydrogen diagnostics disagree with resolved config")

    triangle_diameter = math.hypot(hx, hy)
    length_scale = _finite(material.get("length_scale"), f"{path}: material.length_scale")
    influence_radius = _finite(
        graph_config.get("influence_radius"), f"{path}: graph.influence_radius"
    )
    expected_resolution = {
        "ell_over_triangle_diameter": length_scale / triangle_diameter,
        "gb_width_over_triangle_diameter": influence_radius / triangle_diameter,
    }
    for name, expected_value in expected_resolution.items():
        observed = _finite(resolution.get(name), f"{path}: runtime resolution {name}")
        if not _close(observed, expected_value):
            raise ValueError(
                f"{path}: runtime resolution diagnostics disagree with resolved config"
            )

    expected_quality = {
        "phase_field_resolution_pass": length_scale / triangle_diameter >= 2.0,
        "precrack_mesh_alignment_pass": _close(requested_tip, expected_discrete),
        "grain_boundary_band_has_multiple_cells": influence_radius / triangle_diameter >= 1.5,
    }
    for name, expected_value in expected_quality.items():
        observed = _bool(quality.get(name), f"{path}: runtime quality gate {name}")
        if observed != expected_value:
            raise ValueError(f"{path}: runtime quality gates disagree with resolved config")
        if not observed:
            raise ValueError(f"{path}: runtime quality gate {name!r} did not pass")

    runtime_model = _mapping(runtime.get("model"), f"{path}: runtime.model")
    coupling = runtime_model.get("hydrogen_coupling")
    if hydrogen and (not isinstance(coupling, str) or coupling.lower() == "disabled"):
        raise ValueError(f"{path}: runtime hydrogen model disagrees with enabled factor")
    if not hydrogen and coupling != "disabled":
        raise ValueError(f"{path}: runtime hydrogen model disagrees with disabled factor")

    _finite(graph_config.get("crack_threshold"), f"{path}: graph.crack_threshold")
    return hx, observed_discrete


def _validate_case(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {path}")
    config_path = path / "config.resolved.json"
    runtime_path = path / "runtime.json"
    completion_path = path / "completion.json"
    graph_path = path / "graph_metrics.json"
    config = _read_json(config_path)
    runtime = _read_json(runtime_path)
    config_sha256 = runtime.get("config_sha256")
    if (
        not isinstance(config_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", config_sha256) is None
    ):
        raise ValueError(f"{runtime_path}: config_sha256 must be a 64-character hexadecimal string")
    completion = _read_json(completion_path)
    graph_metrics = _read_json(graph_path)
    history = _read_history(path / "history.csv")

    canonical_config, graph_enabled, hydrogen_enabled = _canonical_config(config, config_path)
    hx, precrack_tip = _validate_runtime_against_config(
        runtime, config, runtime_path, graph_enabled, hydrogen_enabled
    )

    if completion.get("status") != "complete" or completion.get("all_steps_converged") is not True:
        raise ValueError(f"{path}: completion marker does not certify a complete result")
    accepted = _integer(
        completion.get("accepted_load_steps"), f"{completion_path}: accepted_load_steps"
    )
    if accepted != len(history) - 1:
        raise ValueError(f"{path}: completion accepted_load_steps disagrees with history")

    if [row["step"] for row in history] != list(range(len(history))):
        raise ValueError(f"{path / 'history.csv'}: step must be consecutive from zero")
    if not _close(float(history[0]["displacement"]), 0.0):
        raise ValueError(f"{path / 'history.csv'}: first displacement must be zero")
    for previous, current in zip(history, history[1:], strict=False):
        if float(current["displacement"]) <= float(previous["displacement"]):
            raise ValueError(f"{path / 'history.csv'}: displacement must be strictly increasing")
    if not all(bool(row["stagger_converged"]) for row in history):
        raise ValueError(f"{path / 'history.csv'} contains a non-converged record")
    if any(int(row["subdivision_level"]) != 0 for row in history):
        raise ValueError(
            f"{path / 'history.csv'} contains adaptive subdivisions; strict factorial "
            "comparison requires identical nominal load nodes, so rerun all four cases "
            "with unified load nodes (统一载荷节点重跑)"
        )

    loading = _mapping(config.get("loading"), f"{config_path}: loading")
    configured_steps = _integer(loading.get("steps"), f"{config_path}: loading.steps")
    if accepted != configured_steps:
        raise ValueError(
            f"{path}: accepted load steps ({accepted}) do not equal configured loading.steps "
            f"({configured_steps}); rerun all four cases with unified nominal load nodes "
            "(统一载荷节点重跑)"
        )
    configured_stagger_budget = _integer(
        loading.get("stagger_max_iterations"),
        f"{config_path}: loading.stagger_max_iterations",
    )
    if configured_stagger_budget < 1:
        raise ValueError(f"{config_path}: loading.stagger_max_iterations must be positive")
    observed_stagger_iterations = [int(row["stagger_iterations"]) for row in history]
    if any(iterations < 0 for iterations in observed_stagger_iterations):
        raise ValueError(f"{path / 'history.csv'}: stagger_iterations must be non-negative")
    observed_maximum_stagger_iterations = max(observed_stagger_iterations)
    if observed_maximum_stagger_iterations >= configured_stagger_budget:
        raise ValueError(
            f"{path}: observed maximum stagger iterations "
            f"({observed_maximum_stagger_iterations}) must be strictly below configured "
            f"loading.stagger_max_iterations ({configured_stagger_budget}); the iteration "
            "budget was exhausted or exceeded"
        )
    configured_final = _finite(
        loading.get("maximum_displacement"), f"{config_path}: loading.maximum_displacement"
    )
    history_final = float(history[-1]["displacement"])
    completion_final = _finite(
        completion.get("final_displacement"), f"{completion_path}: final_displacement"
    )
    if not (_close(configured_final, history_final) and _close(history_final, completion_final)):
        raise ValueError(f"{path}: config, history and completion final displacement disagree")

    crack = _mapping(graph_metrics.get("G_crack"), f"{graph_path}: G_crack")
    grain_boundary_graph = _mapping(graph_metrics.get("G_GB"), f"{graph_path}: G_GB")
    if graph_enabled and grain_boundary_graph.get("kind") != "G_GB":
        raise ValueError(f"{path}: final G_GB metadata disagrees with enabled graph factor")
    if not graph_enabled and grain_boundary_graph.get("enabled") is not False:
        raise ValueError(f"{path}: final G_GB metadata disagrees with disabled graph factor")
    graph_config = _mapping(config.get("graph"), f"{config_path}: graph")
    threshold = _finite(graph_config.get("crack_threshold"), f"{config_path}: threshold")
    if not _close(_finite(crack.get("threshold"), f"{graph_path}: G_crack.threshold"), threshold):
        raise ValueError(f"{path}: final crack threshold disagrees with resolved config")

    scalar_names = (
        "active_nodes",
        "active_edges",
        "components",
        "largest_component_nodes",
        "main_component_nodes",
    )
    final_crack: dict[str, Any] = {"threshold": threshold}
    for name in scalar_names:
        final_crack[name] = _integer(crack.get(name), f"{graph_path}: G_crack.{name}")
    final_crack["spans_left_to_right"] = _bool(
        crack.get("spans_left_to_right"), f"{graph_path}: G_crack.spans_left_to_right"
    )
    for name in ("tip_x", "path_length", "tortuosity"):
        value = crack.get(name)
        final_crack[name] = (
            None if value is None else _finite(value, f"{graph_path}: G_crack.{name}")
        )
    final_crack["rightmost_active_damage_x"] = float(history[-1]["rightmost_damaged_x"])
    final_crack["regularised_crack_length"] = float(history[-1]["regularised_crack_length"])
    final_crack["regularised_crack_length_increment"] = float(
        history[-1]["regularised_crack_length"]
    ) - float(history[0]["regularised_crack_length"])

    return {
        "case": path.name,
        "directory": str(path),
        "factor": (graph_enabled, hydrogen_enabled),
        "config": config,
        "canonical_config": canonical_config,
        "runtime": runtime,
        "canonical_runtime": _canonical_runtime(runtime, runtime_path),
        "history": history,
        "hx": hx,
        "precrack_tip": precrack_tip,
        "threshold": threshold,
        "final_crack": final_crack,
        "stagger_iteration_budget": {
            "configured": configured_stagger_budget,
            "observed_maximum": observed_maximum_stagger_iterations,
            "margin": configured_stagger_budget - observed_maximum_stagger_iterations,
        },
    }


def _merged_grid(cases: Sequence[dict[str, Any]]) -> list[float]:
    values = sorted(float(row["displacement"]) for case in cases for row in case["history"])
    result: list[float] = []
    for value in values:
        if not result or not _close(value, result[-1]):
            result.append(value)
    return result


def _interpolate(history: Sequence[dict[str, Any]], displacement: float) -> float:
    coordinates = [float(row["displacement"]) for row in history]
    index = bisect.bisect_left(coordinates, displacement)
    if index < len(coordinates) and _close(coordinates[index], displacement):
        return float(history[index]["reaction_y"])
    if index == 0 or index == len(coordinates):
        raise ValueError("interpolation point lies outside the history range")
    left, right = history[index - 1], history[index]
    left_u, right_u = float(left["displacement"]), float(right["displacement"])
    weight = (displacement - left_u) / (right_u - left_u)
    return float(left["reaction_y"]) + weight * (
        float(right["reaction_y"]) - float(left["reaction_y"])
    )


def _trapezoid(values: Sequence[float], grid: Sequence[float]) -> float:
    return sum(
        0.5 * (left_value + right_value) * (right_u - left_u)
        for left_value, right_value, left_u, right_u in zip(
            values, values[1:], grid, grid[1:], strict=False
        )
    )


def _absolute_piecewise_linear_integral(values: Sequence[float], grid: Sequence[float]) -> float:
    """Integrate the absolute value of a piecewise-linear scalar function.

    A sign-changing segment is split at its linear zero.  Applying the
    trapezoid rule directly to ``abs(values)`` would instead connect two
    positive endpoint magnitudes and overestimate the two triangular areas.
    """
    if len(values) != len(grid):
        raise ValueError("piecewise-linear values and grid must have equal length")
    integral = 0.0
    for left, right, left_u, right_u in zip(values, values[1:], grid, grid[1:], strict=False):
        width = right_u - left_u
        if width <= 0.0:
            raise ValueError("piecewise-linear grid must be strictly increasing")
        if left == 0.0 and right == 0.0:
            continue
        if left * right >= 0.0:
            integral += 0.5 * (abs(left) + abs(right)) * width
            continue
        magnitude_sum = abs(left) + abs(right)
        integral += 0.5 * width * (left * left + right * right) / magnitude_sum
    return integral


def _relative(numerator: float, denominator: float, label: str) -> dict[str, Any]:
    if abs(denominator) <= 1.0e-30:
        return {"value": None, "reason": f"reference {label} is zero"}
    return {"value": numerator / denominator, "reason": None}


def _effect(candidate: float, reference: float, label: str) -> dict[str, Any]:
    difference = candidate - reference
    return {
        "absolute_difference": difference,
        "relative_difference": _relative(difference, reference, label),
    }


def _front_onset(case: dict[str, Any]) -> dict[str, Any]:
    history = case["history"]
    hx = float(case["hx"])
    tip = float(case["precrack_tip"])
    tolerance = 1.0e-10 * hx
    detected_index: int | None = None
    for index, row in enumerate(history):
        if float(row["rightmost_damaged_x"]) - tip >= hx - tolerance:
            detected_index = index
            break
    common = {
        "quantity": "rightmost active damage node; per-step connectivity is not checked",
        "connected_main_crack_tip": False,
        "damage_threshold": float(case["threshold"]),
        "criterion": "rightmost_damaged_x - discrete_precrack_tip_x >= 1 * hx",
        "hx": hx,
    }
    if detected_index is None:
        return {
            **common,
            "status": "right_censored",
            "lower_displacement_exclusive": float(history[-1]["displacement"]),
            "upper_displacement_inclusive": None,
            "observed_rightmost_damage_advance": float(history[-1]["rightmost_damaged_x"]) - tip,
            "observed_mesh_columns": (float(history[-1]["rightmost_damaged_x"]) - tip) / hx,
        }
    current = history[detected_index]
    lower = 0.0 if detected_index == 0 else float(history[detected_index - 1]["displacement"])
    advance = float(current["rightmost_damaged_x"]) - tip
    return {
        **common,
        "status": "present_at_initial_state" if detected_index == 0 else "interval_detected",
        "lower_displacement_exclusive": lower if detected_index > 0 else None,
        "upper_displacement_inclusive": float(current["displacement"]),
        "observed_rightmost_damage_advance": advance,
        "observed_mesh_columns": advance / hx,
    }


def _case_metrics(case: dict[str, Any], grid: Sequence[float]) -> dict[str, Any]:
    reaction = [_interpolate(case["history"], displacement) for displacement in grid]
    maximum = max(reaction)
    maximum_index = reaction.index(maximum)
    area = _trapezoid(reaction, grid)
    return {
        "case": case["case"],
        "factors": {
            "graph_enabled": case["factor"][0],
            "hydrogen_enabled": case["factor"][1],
        },
        "history_records": len(case["history"]),
        "stagger_iteration_budget": case["stagger_iteration_budget"],
        "window_maximum_reaction_y": {
            "value": maximum,
            "displacement": grid[maximum_index],
            "at_right_endpoint": maximum_index == len(grid) - 1,
        },
        "final_reaction_y": reaction[-1],
        "reaction_curve_signed_area": area,
        "first_active_damage_front_advance": _front_onset(case),
        "final_crack": case["final_crack"],
        "_reaction": reaction,
    }


def _pair_effects(
    candidate: dict[str, Any], reference: dict[str, Any], grid: Sequence[float]
) -> dict[str, Any]:
    candidate_reaction = candidate["_reaction"]
    reference_reaction = reference["_reaction"]
    difference = [
        candidate_value - reference_value
        for candidate_value, reference_value in zip(
            candidate_reaction, reference_reaction, strict=True
        )
    ]
    signed_difference_area = _trapezoid(difference, grid)
    l1_difference_area = _absolute_piecewise_linear_integral(difference, grid)
    reference_signed_area = float(reference["reaction_curve_signed_area"])
    reference_l1_area = _absolute_piecewise_linear_integral(reference_reaction, grid)
    return {
        "candidate": candidate["case"],
        "reference": reference["case"],
        "window_maximum_reaction_y": _effect(
            float(candidate["window_maximum_reaction_y"]["value"]),
            float(reference["window_maximum_reaction_y"]["value"]),
            "window maximum reaction",
        ),
        "final_reaction_y": _effect(
            float(candidate["final_reaction_y"]),
            float(reference["final_reaction_y"]),
            "final reaction",
        ),
        "reaction_curve": {
            "signed_difference_area": signed_difference_area,
            "signed_area_effect": _relative(
                signed_difference_area, reference_signed_area, "signed reaction area"
            ),
            "l1_difference_area": l1_difference_area,
            "l1_effect": _relative(l1_difference_area, reference_l1_area, "absolute reaction area"),
        },
    }


def _interaction_scalar(values: Mapping[tuple[bool, bool], float], label: str) -> dict[str, Any]:
    baseline = values[(False, False)]
    interaction = values[(True, True)] - values[(True, False)] - values[(False, True)] + baseline
    return {
        "value": interaction,
        "relative_to_no_graph_no_hydrogen": _relative(interaction, baseline, label),
    }


def _optional_interaction_scalar(
    values: Mapping[tuple[bool, bool], float | None], label: str
) -> dict[str, Any]:
    if any(value is None for value in values.values()):
        return {
            "value": None,
            "relative_to_no_graph_no_hydrogen": {
                "value": None,
                "reason": f"{label} is unavailable in at least one case",
            },
        }
    return _interaction_scalar({key: float(value) for key, value in values.items()}, label)


def _additive_interaction(
    metrics: Mapping[tuple[bool, bool], dict[str, Any]], grid: Sequence[float]
) -> dict[str, Any]:
    interaction_curve = [
        metrics[(True, True)]["_reaction"][index]
        - metrics[(True, False)]["_reaction"][index]
        - metrics[(False, True)]["_reaction"][index]
        + metrics[(False, False)]["_reaction"][index]
        for index in range(len(grid))
    ]
    baseline_curve = metrics[(False, False)]["_reaction"]
    signed_area = _trapezoid(interaction_curve, grid)
    l1_area = _absolute_piecewise_linear_integral(interaction_curve, grid)
    baseline_signed = _trapezoid(baseline_curve, grid)
    baseline_l1 = _absolute_piecewise_linear_integral(baseline_curve, grid)

    maximum_values = {
        factor: float(metric["window_maximum_reaction_y"]["value"])
        for factor, metric in metrics.items()
    }
    final_values = {factor: float(metric["final_reaction_y"]) for factor, metric in metrics.items()}
    crack_interactions: dict[str, Any] = {}
    for name in (
        "tip_x",
        "path_length",
        "tortuosity",
        "regularised_crack_length_increment",
    ):
        crack_interactions[name] = _optional_interaction_scalar(
            {factor: metric["final_crack"][name] for factor, metric in metrics.items()},
            f"final crack {name}",
        )

    return {
        "definition": "Y(graph=1,H=1) - Y(graph=1,H=0) - Y(graph=0,H=1) + Y(graph=0,H=0)",
        "window_maximum_reaction_y": _interaction_scalar(
            maximum_values, "baseline window maximum reaction"
        ),
        "final_reaction_y": _interaction_scalar(final_values, "baseline final reaction"),
        "reaction_curve": {
            "signed_area": signed_area,
            "signed_area_relative_to_baseline": _relative(
                signed_area, baseline_signed, "baseline signed reaction area"
            ),
            "l1_area": l1_area,
            "l1_area_relative_to_baseline": _relative(
                l1_area, baseline_l1, "baseline absolute reaction area"
            ),
        },
        "final_crack": crack_interactions,
        "first_active_damage_front_advance": {
            "value": None,
            "reason": (
                "onset is interval- or right-censored; no scalar additive interaction is reported"
            ),
        },
    }


def compare_cases(
    case_directories: Sequence[str | Path],
    *,
    allow_iteration_budget_difference: bool = False,
) -> dict[str, Any]:
    """Validate and compare one complete graph x hydrogen 2x2 onset experiment.

    By default, the staggered iteration budget must be identical across all
    cases. ``allow_iteration_budget_difference=True`` ignores only that one
    cross-case config field; every history must still converge strictly before
    its own configured budget.
    """
    if type(allow_iteration_budget_difference) is not bool:
        raise TypeError("allow_iteration_budget_difference must be a boolean")
    if len(case_directories) != 4:
        raise ValueError("strict 2x2 comparison requires exactly four result directories")
    cases = [_validate_case(directory) for directory in case_directories]
    by_factor: dict[tuple[bool, bool], dict[str, Any]] = {}
    for case in cases:
        factor = case["factor"]
        if factor in by_factor:
            raise ValueError(
                "resolved configs do not uniquely cover the graph.enabled x "
                f"hydrogen.enabled design; duplicate factor combination {factor}"
            )
        by_factor[factor] = case
    if set(by_factor) != set(FACTOR_LEVELS):
        missing = sorted(set(FACTOR_LEVELS) - set(by_factor))
        raise ValueError(
            "resolved configs do not cover all graph.enabled x hydrogen.enabled "
            f"combinations; missing {missing}"
        )

    reference = by_factor[(False, False)]
    reference_config = deepcopy(reference["canonical_config"])
    if allow_iteration_budget_difference:
        _mapping(reference_config.get("loading"), "reference loading config").pop(
            "stagger_max_iterations"
        )
    for case in cases:
        candidate_config = deepcopy(case["canonical_config"])
        if allow_iteration_budget_difference:
            _mapping(candidate_config.get("loading"), "candidate loading config").pop(
                "stagger_max_iterations"
            )
        config_difference = _first_difference(reference_config, candidate_config)
        if config_difference is not None:
            raise ValueError(
                f"unexpected resolved-config difference at {config_difference}: "
                f"{reference['case']!r} versus {case['case']!r}; only graph.enabled and "
                "hydrogen.enabled may vary"
                + (
                    ", plus loading.stagger_max_iterations under the explicit opt-in"
                    if allow_iteration_budget_difference
                    else ""
                )
            )
        runtime_difference = _first_difference(
            reference["canonical_runtime"], case["canonical_runtime"]
        )
        if runtime_difference is not None:
            raise ValueError(
                f"unexpected runtime difference at {runtime_difference}: "
                f"{reference['case']!r} versus {case['case']!r}"
            )

    reference_displacements = [float(row["displacement"]) for row in reference["history"]]
    for case in cases:
        displacements = [float(row["displacement"]) for row in case["history"]]
        if len(displacements) != len(reference_displacements) or any(
            not _close(observed, expected)
            for observed, expected in zip(displacements, reference_displacements, strict=False)
        ):
            raise ValueError(
                "actual displacement sequences differ; strict factorial comparison requires "
                "pointwise-identical nominal load nodes. Rerun all four cases with unified "
                "load nodes (统一载荷节点重跑)"
            )

    grid = _merged_grid(cases)
    upper = float(reference["history"][-1]["displacement"])
    if not _close(grid[0], 0.0) or not _close(grid[-1], upper):
        raise ValueError("case histories do not share the complete configured displacement window")

    metrics = {factor: _case_metrics(case, grid) for factor, case in by_factor.items()}
    baseline = metrics[(False, False)]
    case_results: list[dict[str, Any]] = []
    for factor in FACTOR_LEVELS:
        metric = metrics[factor]
        effect = _pair_effects(metric, baseline, grid)
        clean_metric = {key: value for key, value in metric.items() if key != "_reaction"}
        clean_metric["effect_vs_no_graph_no_hydrogen"] = effect
        case_results.append(clean_metric)

    factor_effects = {
        "graph_at_hydrogen_disabled": _pair_effects(
            metrics[(True, False)], metrics[(False, False)], grid
        ),
        "graph_at_hydrogen_enabled": _pair_effects(
            metrics[(True, True)], metrics[(False, True)], grid
        ),
        "hydrogen_at_graph_disabled": _pair_effects(
            metrics[(False, True)], metrics[(False, False)], grid
        ),
        "hydrogen_at_graph_enabled": _pair_effects(
            metrics[(True, True)], metrics[(True, False)], grid
        ),
    }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "design": {
            "kind": "strict 2x2 graph.enabled x hydrogen.enabled onset comparison",
            "reference_factor": {"graph_enabled": False, "hydrogen_enabled": False},
            "allow_iteration_budget_difference": allow_iteration_budget_difference,
            "allowed_resolved_config_differences": [
                "graph.enabled",
                "hydrogen.enabled",
                "output.directory",
                "source_path",
                *(["loading.stagger_max_iterations"] if allow_iteration_budget_difference else []),
            ],
            "iteration_budget_acceptance": (
                "observed maximum stagger_iterations must be strictly less than each "
                "case's configured loading.stagger_max_iterations"
            ),
            "rightmost_damage_front_caveat": (
                "rightmost_damaged_x is the rightmost threshold-active node at each load step; "
                "it is not a connected main-crack tip"
            ),
        },
        "limitations": {
            "provenance": (
                "Only fields recorded in config.resolved.json and runtime.json are checked for "
                "cross-case and internal consistency. runtime.config_sha256 is checked only "
                "for 64-character hexadecimal syntax and is not content-bound here. No solver "
                "source hash, run identifier, or output-file manifest is recorded, so solver, "
                "run, and file provenance is not certified."
            ),
            "iteration_budget": (
                "loading.stagger_max_iterations is a computational budget, not a physical "
                "factor. Allowing it to differ is explicit and recorded, never silent; even "
                "when every case converges with positive margin, changing this budget may "
                "affect nonlinear local-branch search. No tolerance, KKT criterion, solver, "
                "physical parameter, load node, or other config/runtime difference is ignored."
            ),
        },
        "common_displacement_grid": grid,
        "common_displacement_window": {"lower": grid[0], "upper": grid[-1]},
        "cases": case_results,
        "factor_effects": factor_effects,
        "additive_interaction": _additive_interaction(metrics, grid),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphfracture-comparison",
        description="Strictly compare four completed graph x hydrogen onset cases.",
    )
    parser.add_argument("cases", nargs="+", type=Path, help="four completed result directories")
    parser.add_argument("--output", "-o", type=Path, help="write the comparison JSON")
    parser.add_argument(
        "--allow-iteration-budget-difference",
        action="store_true",
        help=(
            "explicitly allow only loading.stagger_max_iterations to differ; each case must "
            "converge strictly below its configured budget"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = compare_cases(
            args.cases,
            allow_iteration_budget_difference=args.allow_iteration_budget_difference,
        )
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is None:
            print(encoded)
        else:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
    except (OSError, ValueError, TypeError) as exc:
        print(f"comparison error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
