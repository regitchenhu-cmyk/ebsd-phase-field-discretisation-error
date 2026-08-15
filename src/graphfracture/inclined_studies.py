"""Strict reducers for same-material inclined weak-interface DOLFINx studies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import (
    CONTINUATION_CONTROL_FIELDS,
    INTERFACE_DAMAGE_THRESHOLDS,
    continuation_control_increase,
)
from .damage_control import fracture_energy_control_residual_certificate

INTERFACE_CLASSIFICATIONS = frozenset(
    {
        "pre_impact_right_censored",
        "arrested_or_unresolved",
        "deflection_candidate",
        "penetration_candidate",
        "mixed_or_branched",
    }
)
CANDIDATE_CLASSIFICATIONS = frozenset(
    {"deflection_candidate", "penetration_candidate", "mixed_or_branched"}
)
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
INCREMENT_PAIR_RESPONSE_FIELDS = (
    "displacement",
    "reaction_y",
    "elastic_energy",
    "fracture_energy",
    "total_internal_energy",
    "regularised_crack_length",
)
INCREMENT_PAIR_RESPONSE_ABSOLUTE_TOLERANCE = 1.0e-12
WINDOW_EXTENSION_RESPONSE_FIELDS = (
    "displacement",
    "load_factor",
    "reaction_y",
    "elastic_energy",
    "fracture_energy",
    "total_internal_energy",
    "regularised_crack_length",
    "rightmost_damaged_x",
)
WINDOW_EXTENSION_SCHEDULER_FLOAT_FIELDS = (
    "reference_displacement",
    "path_coordinate",
    "path_increment",
    "control_target",
)
WINDOW_EXTENSION_INTERFACE_FLOAT_FIELDS = (
    "closest_main_node_to_impact",
    "interface_forward_advance",
    "penetration_forward_advance",
    "interface_active_edge_length",
    "penetration_active_edge_length",
)
ENERGY_PREDICTOR_LATCH_FIELD = (
    "energy_predictor_disable_after_nominal_path_step"
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required result file does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    _assert_finite_json(value, path, "$")
    return value


def _assert_finite_json(value: Any, path: Path, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: {location} must be finite")
    if isinstance(value, dict):
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


def _energy_predictor_policy(
    path_control: dict[str, Any],
    context: str,
) -> tuple[bool, int]:
    use_energy_predictor = _boolean(
        path_control.get("use_energy_predictor"),
        f"{context}.use_energy_predictor",
    )
    disable_after = _integer(
        path_control.get(ENERGY_PREDICTOR_LATCH_FIELD, -1),
        f"{context}.{ENERGY_PREDICTOR_LATCH_FIELD}",
    )
    if disable_after < -1:
        raise ValueError(
            f"{context}.{ENERGY_PREDICTOR_LATCH_FIELD} must be at least -1"
        )
    return use_energy_predictor, disable_after


def _effective_energy_predictor_last_enabled_step(
    use_energy_predictor: bool,
    disable_after: int,
    configured_steps: int,
) -> int:
    if not use_energy_predictor:
        return 0
    if disable_after < 0:
        return configured_steps
    return min(disable_after, configured_steps)


def _energy_predictor_window_summary(
    *,
    use_energy_predictor: bool,
    disable_after: int,
    configured_steps: int,
    target_increment: float,
) -> dict[str, Any]:
    effective_last_enabled = _effective_energy_predictor_last_enabled_step(
        use_energy_predictor,
        disable_after,
        configured_steps,
    )
    right_censored = (
        use_energy_predictor and effective_last_enabled == configured_steps
    )
    return {
        "use_energy_predictor": use_energy_predictor,
        ENERGY_PREDICTOR_LATCH_FIELD: disable_after,
        "configured_path_targets": configured_steps,
        "effective_last_enabled_nominal_path_step": effective_last_enabled,
        "effective_last_enabled_control_offset": (
            effective_last_enabled * target_increment
        ),
        "first_disabled_nominal_path_step_within_window": (
            None if right_censored else effective_last_enabled + 1
        ),
        "cutoff_right_censored_by_configured_window": right_censored,
    }


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _close(left: float, right: float, *, rel: float = 1.0e-10, abs_: float = 1.0e-12) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=abs_)


def _csv_redundant_scalar_close(left: float, right: float) -> bool:
    """Compare two CSV columns that serialise the same computed scalar."""
    scale = max(1.0, abs(left), abs(right))
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=16.0 * math.ulp(scale),
    )


def _path_control_increment_meets_minimum(
    interval: float,
    minimum_increment: float,
    *,
    accepted_control: float,
    target_control: float,
) -> bool:
    coordinate_scale = max(abs(accepted_control), abs(target_control))
    tolerance = max(
        16.0 * math.ulp(coordinate_scale),
        1.0e-14 * max(abs(interval), abs(minimum_increment), 1.0e-30),
    )
    return interval >= minimum_increment - tolerance


def _read_history(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except FileNotFoundError:
        raise FileNotFoundError(f"required result file does not exist: {path}") from None
    except csv.Error as exc:
        raise ValueError(f"{path}: invalid CSV: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: history must contain at least the zero-load record")
    return rows


def _hdf5_signature_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "signature_valid": False}
    try:
        with path.open("rb") as stream:
            prefix = stream.read(65_544)
    except OSError as exc:
        return {
            "present": True,
            "signature_valid": False,
            "read_error": f"{type(exc).__name__}: {exc}",
        }
    offsets = (0, 512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536)
    signature_offset = next(
        (
            offset
            for offset in offsets
            if prefix[offset : offset + len(HDF5_SIGNATURE)] == HDF5_SIGNATURE
        ),
        None,
    )
    return {
        "present": True,
        "signature_valid": signature_offset is not None,
        "signature_offset": signature_offset,
        "size_bytes": path.stat().st_size,
    }


def _continuation_summary(path: Path, loading: dict[str, Any]) -> dict[str, Any]:
    base = {
        "stagger_max_iterations": _integer(
            loading.get("stagger_max_iterations"), "loading.stagger_max_iterations"
        ),
        "maximum_subdivisions": _integer(
            loading.get("maximum_subdivisions"), "loading.maximum_subdivisions"
        ),
        "minimum_increment": _finite(
            loading.get("minimum_increment"), "loading.minimum_increment"
        ),
    }
    try:
        continuation_control_increase(base, base, allow_equal=True)
    except ValueError as exc:
        raise ValueError(f"loading has invalid continuation controls: {exc}") from exc
    if not path.is_file():
        return {
            "present": False,
            "base_controls": base,
            "effective_controls": base,
            "sessions": [],
            "session_count": 0,
            "controls_changed": False,
        }
    payload = _read_json(path)
    if _integer(payload.get("schema_version"), f"{path}: schema_version") != 1:
        raise ValueError(f"{path}: unsupported continuation history schema")
    _string(payload.get("policy"), f"{path}: policy")
    recorded_base = _mapping(payload.get("base_controls"), f"{path}: base_controls")
    effective = _mapping(payload.get("effective_controls"), f"{path}: effective_controls")
    sessions = _list(payload.get("sessions"), f"{path}: sessions")
    try:
        continuation_control_increase(base, recorded_base, allow_equal=True)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid recorded base controls: {exc}") from exc
    if recorded_base != base:
        raise ValueError(f"{path}: base controls disagree with config")
    try:
        continuation_control_increase(base, effective, allow_equal=True)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid effective continuation controls: {exc}") from exc

    previous = base
    normalised_sessions: list[dict[str, Any]] = []
    previous_parent_generation = -1
    previous_accepted_step = -1
    previous_accepted_displacement = -math.inf
    for index, raw in enumerate(sessions, start=1):
        context = f"{path}: sessions[{index - 1}]"
        session = _mapping(raw, context)
        if _integer(session.get("session"), f"{context}.session") != index:
            raise ValueError(f"{path}: continuation session numbering is not contiguous")
        _string(session.get("started_at_utc"), f"{context}.started_at_utc")
        parent_generation = _integer(
            session.get("parent_generation"), f"{context}.parent_generation"
        )
        if parent_generation < 0 or parent_generation <= previous_parent_generation:
            raise ValueError(f"{context}.parent_generation must strictly increase")
        accepted_state = _mapping(session.get("accepted_state"), f"{context}.accepted_state")
        accepted_step = _integer(
            accepted_state.get("accepted_step"), f"{context}.accepted_state.accepted_step"
        )
        accepted_displacement = _finite(
            accepted_state.get("displacement"),
            f"{context}.accepted_state.displacement",
        )
        if (
            accepted_step < 0
            or accepted_step < previous_accepted_step
            or accepted_displacement < 0.0
            or accepted_displacement + 1.0e-12 < previous_accepted_displacement
        ):
            raise ValueError(f"{context}.accepted_state is not monotone")
        requested = _mapping(
            session.get("requested_overrides"), f"{context}.requested_overrides"
        )
        unknown_overrides = set(requested) - set(CONTINUATION_CONTROL_FIELDS)
        if unknown_overrides:
            raise ValueError(f"{context}.requested_overrides contains unknown controls")
        before = _mapping(session.get("controls_before"), f"{context}.controls_before")
        after = _mapping(session.get("controls_after"), f"{context}.controls_after")
        if before != previous:
            raise ValueError(f"{context}.controls_before breaks the continuation chain")
        try:
            continuation_control_increase(before, after, allow_equal=True)
        except ValueError as exc:
            raise ValueError(f"{context}.controls_after is invalid: {exc}") from exc
        for name in CONTINUATION_CONTROL_FIELDS:
            if name in requested and (
                type(requested[name]) is not type(after[name])
                or requested[name] != after[name]
            ):
                raise ValueError(f"{context}.requested_overrides disagrees on {name}")
            if after[name] != before[name] and name not in requested:
                raise ValueError(f"{context}.controls_after changes unrequested {name}")
        mpi_ranks = _integer(session.get("mpi_ranks"), f"{context}.mpi_ranks")
        if mpi_ranks < 1:
            raise ValueError(f"{context}.mpi_ranks must be positive")
        runtime_identity = _mapping(
            session.get("runtime_identity"), f"{context}.runtime_identity"
        )
        if _integer(
            runtime_identity.get("mpi_ranks"), f"{context}.runtime_identity.mpi_ranks"
        ) != mpi_ranks:
            raise ValueError(f"{context}.runtime_identity disagrees on mpi_ranks")
        normalised_sessions.append(
            {
                **session,
                "accepted_state": {
                    "accepted_step": accepted_step,
                    "displacement": accepted_displacement,
                },
                "controls_before": before,
                "controls_after": after,
                "requested_overrides": requested,
                "parent_generation": parent_generation,
                "mpi_ranks": mpi_ranks,
            }
        )
        previous = after
        previous_parent_generation = parent_generation
        previous_accepted_step = accepted_step
        previous_accepted_displacement = accepted_displacement
    if previous != effective:
        raise ValueError(f"{path}: continuation sessions disagree with effective controls")
    return {
        "present": True,
        "base_controls": base,
        "effective_controls": effective,
        "sessions": normalised_sessions,
        "session_count": len(normalised_sessions),
        "controls_changed": effective != base,
    }


def _csv_float(row: dict[str, str], name: str, path: Path, index: int) -> float:
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


def _csv_int(row: dict[str, str], name: str, path: Path, index: int) -> int:
    value = _csv_float(row, name, path, index)
    result = int(value)
    if value != result:
        raise ValueError(f"{path}: row {index} {name!r} must be an integer")
    return result


def _csv_bool(row: dict[str, str], name: str, path: Path, index: int) -> bool:
    text = row.get(name)
    if text == "True":
        return True
    if text == "False":
        return False
    raise ValueError(f"{path}: row {index} {name!r} must be True or False")


def _csv_string(row: dict[str, str], name: str, path: Path, index: int) -> str:
    text = row.get(name)
    if text is None or not text.strip():
        raise ValueError(f"{path}: row {index} has no {name!r}")
    return text.strip()


def _csv_optional_float(
    row: dict[str, str], name: str, path: Path, index: int
) -> float | None:
    if name not in row:
        raise ValueError(f"{path}: history has no {name!r} column")
    text = row[name]
    if text is None or not text.strip():
        return None
    return _csv_float(row, name, path, index)


def _csv_optional_string(
    row: dict[str, str], name: str, path: Path, index: int
) -> str | None:
    if name not in row:
        raise ValueError(f"{path}: history has no {name!r} column")
    text = row[name]
    if text is None or not text.strip():
        return None
    return text.strip()


def _validate_threshold_results(value: Any, context: str) -> list[dict[str, Any]]:
    results = _list(value, context)
    if len(results) != len(INTERFACE_DAMAGE_THRESHOLDS):
        raise ValueError(f"{context} must contain the three registered damage thresholds")
    normalised: list[dict[str, Any]] = []
    for index, (raw, expected_threshold) in enumerate(
        zip(results, INTERFACE_DAMAGE_THRESHOLDS, strict=True)
    ):
        item = _mapping(raw, f"{context}[{index}]")
        threshold = _finite(item.get("threshold"), f"{context}[{index}].threshold")
        if not _close(threshold, expected_threshold):
            raise ValueError(f"{context}[{index}] has an unexpected threshold")
        classification = _string(
            item.get("geometric_classification"),
            f"{context}[{index}].geometric_classification",
        )
        if classification not in INTERFACE_CLASSIFICATIONS:
            raise ValueError(f"{context}[{index}] has an unknown classification")
        reached = _boolean(item.get("reached_interface"), f"{context}[{index}].reached_interface")
        closest = _optional_finite(
            item.get("closest_main_node_to_impact"),
            f"{context}[{index}].closest_main_node_to_impact",
        )
        interface_advance = _finite(
            item.get("interface_forward_advance"),
            f"{context}[{index}].interface_forward_advance",
        )
        penetration_advance = _finite(
            item.get("penetration_forward_advance"),
            f"{context}[{index}].penetration_forward_advance",
        )
        interface_edges = _finite(
            item.get("interface_active_edge_length"),
            f"{context}[{index}].interface_active_edge_length",
        )
        penetration_edges = _finite(
            item.get("penetration_active_edge_length"),
            f"{context}[{index}].penetration_active_edge_length",
        )
        if min(interface_advance, penetration_advance, interface_edges, penetration_edges) < 0.0:
            raise ValueError(f"{context}[{index}] contains a negative geometric measure")
        if classification == "pre_impact_right_censored" and reached:
            raise ValueError(f"{context}[{index}] marks a reached crack as pre-impact")
        if classification != "pre_impact_right_censored" and not reached:
            raise ValueError(f"{context}[{index}] classifies an unreached interface post-impact")
        normalised.append(
            {
                "threshold": threshold,
                "classification": classification,
                "reached_interface": reached,
                "closest_main_node_to_impact": closest,
                "interface_forward_advance": interface_advance,
                "penetration_forward_advance": penetration_advance,
                "interface_active_edge_length": interface_edges,
                "penetration_active_edge_length": penetration_edges,
                "raw": item,
            }
        )
    return normalised


def _consensus(results: list[dict[str, Any]]) -> str:
    values = {item["classification"] for item in results}
    return values.pop() if len(values) == 1 else "threshold_ambiguous"


def _impact_bracket(records: list[dict[str, Any]], *, require_all: bool) -> dict[str, Any]:
    first_index = None
    for index, record in enumerate(records):
        flags = [item["reached_interface"] for item in record["threshold_results"]]
        if all(flags) if require_all else any(flags):
            first_index = index
            break
    if first_index is None:
        return {
            "status": "right_censored",
            "lower_displacement": records[-1]["displacement"],
            "upper_displacement": None,
        }
    if first_index == 0:
        return {
            "status": "at_or_before_zero_load",
            "lower_displacement": None,
            "upper_displacement": 0.0,
        }
    return {
        "status": "interval_censored",
        "lower_displacement": records[first_index - 1]["displacement"],
        "upper_displacement": records[first_index]["displacement"],
    }


def _candidate_persistence(
    records: list[dict[str, Any]],
    *,
    consecutive_states: int = 2,
) -> dict[str, Any]:
    if consecutive_states < 2:
        raise ValueError("candidate persistence requires at least two states")
    first_confirmation = None
    for end in range(consecutive_states - 1, len(records)):
        window = records[end - consecutive_states + 1 : end + 1]
        values = {record["threshold_consensus"] for record in window}
        if len(values) == 1 and next(iter(values)) in CANDIDATE_CLASSIFICATIONS:
            first_confirmation = end
            break
    final_window = records[-consecutive_states:]
    final_values = {record["threshold_consensus"] for record in final_window}
    final_classification = (
        next(iter(final_values))
        if len(final_window) == consecutive_states
        and len(final_values) == 1
        and next(iter(final_values)) in CANDIDATE_CLASSIFICATIONS
        else None
    )
    if first_confirmation is None:
        return {
            "required_consecutive_states": consecutive_states,
            "status": "not_confirmed",
            "first_confirmed_classification": None,
            "first_confirmation_displacement": None,
            "final_gate_pass": False,
            "final_confirmed_classification": final_classification,
        }
    first = first_confirmation - consecutive_states + 1
    return {
        "required_consecutive_states": consecutive_states,
        "status": "confirmed",
        "first_confirmed_classification": records[first]["threshold_consensus"],
        "first_candidate_displacement": records[first]["displacement"],
        "first_confirmation_displacement": records[first_confirmation]["displacement"],
        "final_gate_pass": final_classification is not None,
        "final_confirmed_classification": final_classification,
    }


def _validate_case(
    case_directory: str | Path,
    *,
    allow_underresolved: bool,
    allow_subdivisions: bool,
) -> dict[str, Any]:
    directory = Path(case_directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {directory}")
    paths = {
        name: directory / filename
        for name, filename in {
            "config": "config.resolved.json",
            "runtime": "runtime.json",
            "completion": "completion.json",
            "continuation": "continuation_history.json",
            "graph": "graph_metrics.json",
            "interface": "interface_history.json",
            "history": "history.csv",
        }.items()
    }
    config = _read_json(paths["config"])
    configured_loading = _mapping(
        config.get("loading"),
        f"{paths['config']}: loading",
    )
    if _path_control_progress_contract(
        config,
        configured_loading,
        paths["config"],
    ) is not None:
        return _validate_completed_hybrid_case(
            directory,
            allow_underresolved=allow_underresolved,
            allow_subdivisions=allow_subdivisions,
        )
    runtime = _read_json(paths["runtime"])
    completion = _read_json(paths["completion"])
    graph_metrics = _read_json(paths["graph"])
    interface_history = _read_json(paths["interface"])
    history = _read_history(paths["history"])

    if _string(completion.get("status"), f"{paths['completion']}: status") != "complete":
        raise ValueError(f"{paths['completion']}: case is not complete")
    if not _boolean(
        completion.get("all_steps_converged"), f"{paths['completion']}: all_steps_converged"
    ):
        raise ValueError(f"{paths['completion']}: not all accepted steps converged")

    geometry = _mapping(config.get("geometry"), f"{paths['config']}: geometry")
    material = _mapping(config.get("material"), f"{paths['config']}: material")
    loading = _mapping(config.get("loading"), f"{paths['config']}: loading")
    continuation = _continuation_summary(paths["continuation"], loading)
    graph = _mapping(config.get("graph"), f"{paths['config']}: graph")
    hydrogen = _mapping(config.get("hydrogen"), f"{paths['config']}: hydrogen")
    if not _boolean(graph.get("enabled"), f"{paths['config']}: graph.enabled"):
        raise ValueError(f"{paths['config']}: inclined-interface study requires graph.enabled")
    if _boolean(hydrogen.get("enabled"), f"{paths['config']}: hydrogen.enabled"):
        raise ValueError(
            f"{paths['config']}: baseline inclined-interface study requires no hydrogen"
        )
    if graph.get("chain_artifact") not in {None, ""}:
        raise ValueError(f"{paths['config']}: inclined-interface study requires a synthetic graph")
    protocol_names = (
        _string(graph.get("interface_start_node"), f"{paths['config']}: interface_start_node"),
        _string(graph.get("interface_impact_node"), f"{paths['config']}: interface_impact_node"),
        _string(graph.get("interface_end_node"), f"{paths['config']}: interface_end_node"),
    )

    length = _finite(geometry.get("length"), f"{paths['config']}: geometry.length")
    height = _finite(geometry.get("height"), f"{paths['config']}: geometry.height")
    nx = _integer(geometry.get("nx"), f"{paths['config']}: geometry.nx")
    ny = _integer(geometry.get("ny"), f"{paths['config']}: geometry.ny")
    ell = _finite(material.get("length_scale"), f"{paths['config']}: material.length_scale")
    triangle_diameter = math.hypot(length / nx, height / ny)
    resolution = ell / triangle_diameter
    resolution_pass = resolution >= 2.0
    if not resolution_pass and not allow_underresolved:
        raise ValueError(
            f"{directory}: ell/h_K={resolution:.6g} is below 2; "
            "use --allow-underresolved only for load-window screening"
        )

    diagnostics = _mapping(runtime.get("diagnostics"), f"{paths['runtime']}: diagnostics")
    mesh_diagnostics = _mapping(diagnostics.get("mesh"), f"{paths['runtime']}: diagnostics.mesh")
    model_diagnostics = _mapping(diagnostics.get("model"), f"{paths['runtime']}: diagnostics.model")
    diagonal = _string(geometry.get("diagonal"), f"{paths['config']}: geometry.diagonal")
    pin = _string(geometry.get("x_pin_corner"), f"{paths['config']}: geometry.x_pin_corner")
    if mesh_diagnostics.get("diagonal") != diagonal:
        raise ValueError(f"{paths['runtime']}: mesh diagonal disagrees with config")
    if model_diagnostics.get("horizontal_rigid_body_pin") != pin:
        raise ValueError(f"{paths['runtime']}: rigid-body pin disagrees with config")
    recorded_resolution = _finite(
        _mapping(diagnostics.get("resolution"), f"{paths['runtime']}: diagnostics.resolution").get(
            "ell_over_triangle_diameter"
        ),
        f"{paths['runtime']}: ell_over_triangle_diameter",
    )
    if not _close(recorded_resolution, resolution):
        raise ValueError(f"{paths['runtime']}: recorded resolution disagrees with config")
    mpi_ranks = _integer(runtime.get("mpi_ranks"), f"{paths['runtime']}: mpi_ranks")
    if mpi_ranks < 1:
        raise ValueError(f"{paths['runtime']}: mpi_ranks must be positive")
    if any(session["mpi_ranks"] != mpi_ranks for session in continuation["sessions"]):
        raise ValueError(f"{paths['continuation']}: session MPI ranks disagree with runtime")

    accepted_steps = _integer(
        completion.get("accepted_load_steps"), f"{paths['completion']}: accepted_load_steps"
    )
    if accepted_steps != len(history) - 1:
        raise ValueError(f"{paths['completion']}: accepted step count disagrees with history")
    configured_steps = _integer(loading.get("steps"), f"{paths['config']}: loading.steps")
    configured_maximum = _finite(
        loading.get("maximum_displacement"),
        f"{paths['config']}: loading.maximum_displacement",
    )
    kkt_tolerance = _finite(
        loading.get("damage_kkt_tolerance"),
        f"{paths['config']}: loading.damage_kkt_tolerance",
    )
    history_values: list[dict[str, Any]] = []
    for index, row in enumerate(history):
        value = {
            "step": _csv_int(row, "step", paths["history"], index),
            "scheduled_step": _csv_int(row, "scheduled_step", paths["history"], index),
            "subdivision_level": _csv_int(row, "subdivision_level", paths["history"], index),
            "displacement": _csv_float(row, "displacement", paths["history"], index),
            "load_increment": _csv_float(row, "load_increment", paths["history"], index),
            "reaction_y": _csv_float(row, "reaction_y", paths["history"], index),
            "elastic_energy": _csv_float(row, "elastic_energy", paths["history"], index),
            "fracture_energy": _csv_float(row, "fracture_energy", paths["history"], index),
            "total_internal_energy": _csv_float(
                row, "total_internal_energy", paths["history"], index
            ),
            "regularised_crack_length": _csv_float(
                row, "regularised_crack_length", paths["history"], index
            ),
            "damage_kkt_relative": _csv_float(row, "damage_kkt_relative", paths["history"], index),
            "stagger_iterations": _csv_int(
                row, "stagger_iterations", paths["history"], index
            ),
            "stagger_converged": _csv_bool(row, "stagger_converged", paths["history"], index),
        }
        if value["step"] != index:
            raise ValueError(f"{paths['history']}: accepted step numbering is not contiguous")
        if not (
            0
            <= value["subdivision_level"]
            <= continuation["effective_controls"]["maximum_subdivisions"]
        ):
            raise ValueError(
                f"{paths['history']}: row {index} exceeds the effective subdivision budget"
            )
        if not (
            0
            <= value["stagger_iterations"]
            <= continuation["effective_controls"]["stagger_max_iterations"]
        ):
            raise ValueError(
                f"{paths['history']}: row {index} exceeds the effective stagger budget"
            )
        if not value["stagger_converged"]:
            raise ValueError(f"{paths['history']}: row {index} is not converged")
        if value["damage_kkt_relative"] > kkt_tolerance * (1.0 + 1.0e-8):
            raise ValueError(
                f"{paths['history']}: row {index} exceeds the configured KKT tolerance"
            )
        if index:
            previous = history_values[-1]
            if value["displacement"] <= previous["displacement"]:
                raise ValueError(
                    f"{paths['history']}: accepted displacements must strictly increase"
                )
            increment = value["displacement"] - previous["displacement"]
            if not _close(value["load_increment"], increment):
                raise ValueError(f"{paths['history']}: row {index} load increment disagrees")
            minimum_increment = continuation["effective_controls"]["minimum_increment"]
            if value["load_increment"] + max(1.0e-15, 1.0e-12 * minimum_increment) < (
                minimum_increment
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} is below the effective minimum increment"
                )
        history_values.append(value)
    if not _close(history_values[0]["displacement"], 0.0):
        raise ValueError(f"{paths['history']}: first record must be zero load")
    if not _close(history_values[0]["load_increment"], 0.0):
        raise ValueError(f"{paths['history']}: zero-load record must have zero increment")
    if not _close(history_values[-1]["displacement"], configured_maximum):
        raise ValueError(f"{paths['history']}: final displacement disagrees with config")
    if history_values[-1]["scheduled_step"] != configured_steps:
        raise ValueError(f"{paths['history']}: final scheduled step disagrees with config")
    subdivisions = [item["subdivision_level"] for item in history_values]
    if any(level > 0 for level in subdivisions) and not allow_subdivisions:
        raise ValueError(
            f"{paths['history']}: adaptive subdivisions require --allow-subdivisions for screening"
        )
    if not any(level > 0 for level in subdivisions) and accepted_steps != configured_steps:
        raise ValueError(f"{paths['history']}: nominal step count disagrees with config")
    for session in continuation["sessions"]:
        accepted_state = session["accepted_state"]
        accepted_step = accepted_state["accepted_step"]
        if accepted_step >= len(history_values) or not _close(
            accepted_state["displacement"],
            history_values[accepted_step]["displacement"],
        ):
            raise ValueError(
                f"{paths['continuation']}: session accepted state disagrees with history"
            )

    if _string(interface_history.get("status"), f"{paths['interface']}: status") != "complete":
        raise ValueError(f"{paths['interface']}: interface history is not complete")
    protocol = _mapping(interface_history.get("protocol"), f"{paths['interface']}: protocol")
    if tuple(protocol.get(name) for name in ("start_node", "impact_node", "end_node")) != (
        protocol_names
    ):
        raise ValueError(f"{paths['interface']}: protocol node names disagree with config")
    thresholds = _list(protocol.get("thresholds"), f"{paths['interface']}: thresholds")
    if len(thresholds) != len(INTERFACE_DAMAGE_THRESHOLDS) or any(
        not _close(_finite(value, f"{paths['interface']}: threshold"), expected)
        for value, expected in zip(thresholds, INTERFACE_DAMAGE_THRESHOLDS, strict=True)
    ):
        raise ValueError(f"{paths['interface']}: protocol thresholds are not registered")
    raw_interface_records = _list(
        interface_history.get("records"), f"{paths['interface']}: records"
    )
    if len(raw_interface_records) != len(history_values):
        raise ValueError(f"{paths['interface']}: record count disagrees with mechanical history")
    interface_records: list[dict[str, Any]] = []
    previous_thresholds: list[dict[str, Any]] | None = None
    for index, (raw, mechanical) in enumerate(
        zip(raw_interface_records, history_values, strict=True)
    ):
        item = _mapping(raw, f"{paths['interface']}: records[{index}]")
        for field in ("accepted_step", "scheduled_step", "subdivision_level"):
            if (
                _integer(item.get(field), f"{paths['interface']}: records[{index}].{field}")
                != (mechanical["step" if field == "accepted_step" else field])
            ):
                raise ValueError(f"{paths['interface']}: records[{index}] disagrees on {field}")
        displacement = _finite(
            item.get("displacement"), f"{paths['interface']}: records[{index}].displacement"
        )
        if not _close(displacement, mechanical["displacement"]):
            raise ValueError(f"{paths['interface']}: records[{index}] displacement disagrees")
        threshold_results = _validate_threshold_results(
            item.get("threshold_results"),
            f"{paths['interface']}: records[{index}].threshold_results",
        )
        consensus = _string(
            item.get("threshold_consensus"),
            f"{paths['interface']}: records[{index}].threshold_consensus",
        )
        if consensus != _consensus(threshold_results):
            raise ValueError(f"{paths['interface']}: records[{index}] consensus is inconsistent")
        if previous_thresholds is not None:
            for threshold_index, (previous, current) in enumerate(
                zip(previous_thresholds, threshold_results, strict=True)
            ):
                context = f"{paths['interface']}: threshold index {threshold_index}"
                if previous["reached_interface"] and not current["reached_interface"]:
                    raise ValueError(f"{context} violates irreversible interface reachability")
                for name in ("interface_forward_advance", "penetration_forward_advance"):
                    if current[name] + 1.0e-12 < previous[name]:
                        raise ValueError(f"{context} has decreasing {name}")
                if (
                    previous["closest_main_node_to_impact"] is not None
                    and current["closest_main_node_to_impact"] is not None
                    and current["closest_main_node_to_impact"]
                    > previous["closest_main_node_to_impact"] + 1.0e-12
                ):
                    raise ValueError(f"{context} has increasing closest impact distance")
        interface_records.append(
            {
                "accepted_step": index,
                "scheduled_step": mechanical["scheduled_step"],
                "subdivision_level": mechanical["subdivision_level"],
                "displacement": displacement,
                "threshold_consensus": consensus,
                "threshold_results": threshold_results,
                "raw": item,
            }
        )
        previous_thresholds = threshold_results

    final_interaction = _mapping(
        graph_metrics.get("interface_interaction"),
        f"{paths['graph']}: interface_interaction",
    )
    if final_interaction.get("protocol") != protocol:
        raise ValueError(f"{paths['graph']}: final protocol disagrees with interface history")
    if final_interaction.get("threshold_results") != raw_interface_records[-1].get(
        "threshold_results"
    ):
        raise ValueError(f"{paths['graph']}: final thresholds disagree with interface history")
    if final_interaction.get("threshold_consensus") != interface_records[-1]["threshold_consensus"]:
        raise ValueError(f"{paths['graph']}: final consensus disagrees with interface history")
    crack = _mapping(graph_metrics.get("G_crack"), f"{paths['graph']}: G_crack")
    crack_threshold = _finite(graph.get("crack_threshold"), f"{paths['config']}: crack_threshold")
    if not _close(
        _finite(crack.get("threshold"), f"{paths['graph']}: G_crack.threshold"),
        crack_threshold,
    ):
        raise ValueError(f"{paths['graph']}: G_crack threshold disagrees with config")

    reactions = [item["reaction_y"] for item in history_values]
    peak_index = max(range(len(reactions)), key=reactions.__getitem__)
    return {
        "case": directory.name,
        "directory": str(directory),
        "screen_only": not resolution_pass or any(level > 0 for level in subdivisions),
        "mesh": {
            "nx": nx,
            "ny": ny,
            "diagonal": diagonal,
            "horizontal_rigid_body_pin": pin,
            "ell_over_triangle_diameter": resolution,
            "resolution_gate_pass": resolution_pass,
        },
        "runtime": {"mpi_ranks": mpi_ranks},
        "continuation": continuation,
        "loading": {
            "configured_steps": configured_steps,
            "accepted_steps": accepted_steps,
            "maximum_displacement": configured_maximum,
            "maximum_subdivision_level": max(subdivisions),
            "window_maximum_reaction_y": reactions[peak_index],
            "window_maximum_reaction_displacement": history_values[peak_index]["displacement"],
            "window_maximum_reaction_at_right_endpoint": peak_index == len(reactions) - 1,
        },
        "interface": {
            "protocol": protocol,
            "final_threshold_consensus": interface_records[-1]["threshold_consensus"],
            "first_any_threshold_impact": _impact_bracket(interface_records, require_all=False),
            "first_all_threshold_impact": _impact_bracket(interface_records, require_all=True),
            "candidate_persistence": _candidate_persistence(interface_records),
            "records": [item["raw"] for item in interface_records],
        },
        "all_checks_passed": True,
        "limitations": [
            "Candidate classifications are threshold- and mesh-dependent geometric screens.",
            "The model is a same-material isotropic weak plane without elastic mismatch, crystal "
            "anisotropy or a tension/compression split.",
            "A right-endpoint reaction maximum is window-censored, not a demonstrated global peak.",
        ],
        "_config": config,
        "_history": history_values,
        "_interface_records": interface_records,
    }


def _public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if not key.startswith("_")}


def case_audit_report(
    case_directory: str | Path,
    *,
    allow_underresolved: bool = False,
    allow_subdivisions: bool = False,
) -> dict[str, Any]:
    """Strictly validate one completed inclined-interface result directory."""
    return _public_case(
        _validate_case(
            case_directory,
            allow_underresolved=allow_underresolved,
            allow_subdivisions=allow_subdivisions,
        )
    )


def _path_control_progress_contract(
    config: dict[str, Any], loading: dict[str, Any], path: Path
) -> dict[str, Any] | None:
    raw = config.get("path_control")
    if raw is None:
        return None
    section = _mapping(raw, f"{path}: path_control")
    if not _boolean(section.get("enabled"), f"{path}: path_control.enabled"):
        return None
    functional = _string(section.get("functional"), f"{path}: path_control.functional")
    if functional != "fracture_energy":
        raise ValueError(f"{path}: unsupported path-control functional {functional!r}")

    configured_steps = _integer(loading.get("steps"), f"{path}: loading.steps")
    configured_maximum = _finite(
        loading.get("maximum_displacement"),
        f"{path}: loading.maximum_displacement",
    )
    switch = _finite(
        section.get("switch_displacement"),
        f"{path}: path_control.switch_displacement",
    )
    increment = _finite(
        section.get("target_increment"),
        f"{path}: path_control.target_increment",
    )
    targets = _integer(section.get("steps"), f"{path}: path_control.steps")
    use_energy_predictor, predictor_disable_after = _energy_predictor_policy(
        section,
        f"{path}: path_control",
    )
    adaptive = _boolean(section.get("adaptive"), f"{path}: path_control.adaptive")
    maximum_subdivisions = _integer(
        section.get("maximum_subdivisions"),
        f"{path}: path_control.maximum_subdivisions",
    )
    minimum_increment = _finite(
        section.get("minimum_increment"),
        f"{path}: path_control.minimum_increment",
    )
    lower = _finite(
        section.get("load_lower_bound"),
        f"{path}: path_control.load_lower_bound",
    )
    upper = _finite(
        section.get("load_upper_bound"),
        f"{path}: path_control.load_upper_bound",
    )
    snes_max_iterations = _integer(
        section.get("snes_max_iterations"),
        f"{path}: path_control.snes_max_iterations",
    )
    residual_tolerance = _finite(
        section.get("residual_tolerance"),
        f"{path}: path_control.residual_tolerance",
    )
    tolerance = _finite(
        section.get("control_tolerance"),
        f"{path}: path_control.control_tolerance",
    )
    raw_absolute_tolerance = section.get("control_absolute_tolerance")
    if raw_absolute_tolerance is None:
        absolute_tolerance = None
        certificate_mode = "legacy_relative_only"
    else:
        absolute_tolerance = _finite(
            raw_absolute_tolerance,
            f"{path}: path_control.control_absolute_tolerance",
        )
        certificate_mode = "composite_relative_absolute"
    if not (
        configured_steps > 0
        and configured_maximum > 0.0
        and 0.0 < switch < configured_maximum
        and increment > 0.0
        and targets > 0
        and maximum_subdivisions >= 0
        and 0.0 < minimum_increment <= increment
        and 0.0 <= lower < switch < upper <= configured_maximum
        and snes_max_iterations > 0
        and residual_tolerance > 0.0
        and tolerance > 0.0
        and (absolute_tolerance is None or absolute_tolerance > 0.0)
    ):
        raise ValueError(f"{path}: invalid path-control progress contract")
    nominal_increment = configured_maximum / configured_steps
    switch_index = round(switch / nominal_increment)
    if not _close(switch, switch_index * nominal_increment):
        raise ValueError(f"{path}: path-control switch is not a nominal loading node")
    return {
        "switch_displacement": switch,
        "switch_index": switch_index,
        "target_increment": increment,
        "steps": targets,
        "use_energy_predictor": use_energy_predictor,
        ENERGY_PREDICTOR_LATCH_FIELD: predictor_disable_after,
        "adaptive": adaptive,
        "maximum_subdivisions": maximum_subdivisions,
        "minimum_increment": minimum_increment,
        "load_lower_bound": lower,
        "load_upper_bound": upper,
        "snes_max_iterations": snes_max_iterations,
        "residual_tolerance": residual_tolerance,
        "control_tolerance": tolerance,
        "control_absolute_tolerance": absolute_tolerance,
        "control_certificate_mode": certificate_mode,
    }


def _hybrid_method_status_summary(
    path: Path,
    *,
    path_contract: dict[str, Any] | None,
    last: dict[str, Any],
    accepted_path_steps: int,
    completion_present: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if path_contract is None:
        raise ValueError(f"{path}: hybrid method status requires enabled path control")

    payload = _read_json(path)
    if _integer(payload.get("schema_version"), f"{path}: schema_version") != 1:
        raise ValueError(f"{path}: unsupported method-status schema")
    method = _string(payload.get("method"), f"{path}: method")
    if method != "hybrid_displacement_fracture_energy":
        raise ValueError(f"{path}: unknown hybrid method")
    status = _string(payload.get("status"), f"{path}: status")
    if status not in {"initialising", "running", "complete", "failed"}:
        raise ValueError(f"{path}: unknown method status")
    resume_supported = _boolean(
        payload.get("resume_supported"), f"{path}: resume_supported"
    )
    checkpoint_schema = _integer(
        payload.get("checkpoint_schema"), f"{path}: checkpoint_schema"
    )
    if checkpoint_schema != 3:
        raise ValueError(f"{path}: unsupported checkpoint schema")
    checkpoint_policy = _string(
        payload.get("checkpoint_policy"), f"{path}: checkpoint_policy"
    )
    control_phase = _string(payload.get("control_phase"), f"{path}: control_phase")
    if control_phase not in {"displacement", "fracture_energy"}:
        raise ValueError(f"{path}: unknown control phase")
    accepted_step = _integer(payload.get("accepted_step"), f"{path}: accepted_step")
    accepted_displacement = _finite(
        payload.get("accepted_displacement"), f"{path}: accepted_displacement"
    )
    accepted_control = _finite(
        payload.get("accepted_control"), f"{path}: accepted_control"
    )
    recorded_path_steps = _integer(
        payload.get("accepted_path_steps"), f"{path}: accepted_path_steps"
    )
    configured_targets = _integer(
        payload.get("configured_path_targets"), f"{path}: configured_path_targets"
    )
    pending_targets = _integer(
        payload.get("pending_control_targets"), f"{path}: pending_control_targets"
    )
    completion_condition = _string(
        payload.get("completion_condition"), f"{path}: completion_condition"
    )
    if (
        accepted_step < 0
        or accepted_displacement < 0.0
        or accepted_control < 0.0
        or recorded_path_steps < 0
        or configured_targets <= 0
        or pending_targets < 0
    ):
        raise ValueError(f"{path}: method-status progress counters are invalid")
    if (
        accepted_step != last["step"]
        or not _close(accepted_displacement, last["displacement"])
        or not _close(accepted_control, last["path_coordinate"])
        or recorded_path_steps != accepted_path_steps
        or configured_targets != path_contract["steps"]
    ):
        raise ValueError(f"{path}: accepted method status disagrees with history or config")
    if completion_condition != "fracture_energy_queue_exhausted":
        raise ValueError(f"{path}: unknown completion condition")
    if completion_present and status != "complete":
        raise ValueError(f"{path}: method status disagrees with completion.json")
    if status == "complete" and pending_targets != 0:
        raise ValueError(f"{path}: complete method status retains pending targets")

    failure_type = payload.get("failure_type")
    failure_message = payload.get("failure_message")
    if status == "failed":
        failure_type = _string(failure_type, f"{path}: failure_type")
        failure_message = _string(failure_message, f"{path}: failure_message")
    elif failure_type is not None or failure_message is not None:
        raise ValueError(f"{path}: non-failed method status contains failure metadata")

    return {
        "present": True,
        "method": method,
        "status": status,
        "resume_supported": resume_supported,
        "checkpoint_schema": checkpoint_schema,
        "checkpoint_policy": checkpoint_policy,
        "control_phase": control_phase,
        "accepted_step": accepted_step,
        "accepted_displacement": accepted_displacement,
        "accepted_control": accepted_control,
        "accepted_path_steps": recorded_path_steps,
        "configured_path_targets": configured_targets,
        "pending_control_targets": pending_targets,
        "completion_condition": completion_condition,
        "failure_type": failure_type,
        "failure_message": failure_message,
    }


def case_progress_report(case_directory: str | Path) -> dict[str, Any]:
    """Strictly inspect accepted states from a complete or partial inclined case.

    Unlike :func:`case_audit_report`, this reducer does not claim that the
    configured load window was completed.  It validates only persisted,
    converged states and reports all event brackets as censored at the last
    accepted displacement when no interface impact has yet been observed.
    """
    directory = Path(case_directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {directory}")
    paths = {
        name: directory / filename
        for name, filename in {
            "config": "config.resolved.json",
            "runtime": "runtime.json",
            "completion": "completion.json",
            "continuation": "continuation_history.json",
            "failure": "failure.json",
            "method_status": "method_status.json",
            "attempts": "attempt_history.json",
            "interface": "interface_history.json",
            "history": "history.csv",
        }.items()
    }
    config = _read_json(paths["config"])
    runtime = _read_json(paths["runtime"])
    interface_history = _read_json(paths["interface"])
    history = _read_history(paths["history"])

    geometry = _mapping(config.get("geometry"), f"{paths['config']}: geometry")
    material = _mapping(config.get("material"), f"{paths['config']}: material")
    loading = _mapping(config.get("loading"), f"{paths['config']}: loading")
    path_contract = _path_control_progress_contract(config, loading, paths["config"])
    path_control_enabled = path_contract is not None
    displacement_phase_steps = 0
    accepted_path_steps = 0
    path_phase_started = False
    continuation = _continuation_summary(paths["continuation"], loading)
    graph = _mapping(config.get("graph"), f"{paths['config']}: graph")
    hydrogen = _mapping(config.get("hydrogen"), f"{paths['config']}: hydrogen")
    if not _boolean(graph.get("enabled"), f"{paths['config']}: graph.enabled"):
        raise ValueError(f"{paths['config']}: inclined-interface study requires graph.enabled")
    if _boolean(hydrogen.get("enabled"), f"{paths['config']}: hydrogen.enabled"):
        raise ValueError(
            f"{paths['config']}: baseline inclined-interface study requires no hydrogen"
        )
    if graph.get("chain_artifact") not in {None, ""}:
        raise ValueError(f"{paths['config']}: inclined-interface study requires a synthetic graph")
    protocol_names = (
        _string(graph.get("interface_start_node"), f"{paths['config']}: interface_start_node"),
        _string(graph.get("interface_impact_node"), f"{paths['config']}: interface_impact_node"),
        _string(graph.get("interface_end_node"), f"{paths['config']}: interface_end_node"),
    )

    length = _finite(geometry.get("length"), f"{paths['config']}: geometry.length")
    height = _finite(geometry.get("height"), f"{paths['config']}: geometry.height")
    nx = _integer(geometry.get("nx"), f"{paths['config']}: geometry.nx")
    ny = _integer(geometry.get("ny"), f"{paths['config']}: geometry.ny")
    ell = _finite(material.get("length_scale"), f"{paths['config']}: material.length_scale")
    triangle_diameter = math.hypot(length / nx, height / ny)
    resolution = ell / triangle_diameter

    diagnostics = _mapping(runtime.get("diagnostics"), f"{paths['runtime']}: diagnostics")
    recorded_resolution = _finite(
        _mapping(diagnostics.get("resolution"), f"{paths['runtime']}: diagnostics.resolution").get(
            "ell_over_triangle_diameter"
        ),
        f"{paths['runtime']}: ell_over_triangle_diameter",
    )
    if not _close(recorded_resolution, resolution):
        raise ValueError(f"{paths['runtime']}: recorded resolution disagrees with config")
    mpi_ranks = _integer(runtime.get("mpi_ranks"), f"{paths['runtime']}: mpi_ranks")
    if mpi_ranks < 1:
        raise ValueError(f"{paths['runtime']}: mpi_ranks must be positive")
    if any(session["mpi_ranks"] != mpi_ranks for session in continuation["sessions"]):
        raise ValueError(f"{paths['continuation']}: session MPI ranks disagree with runtime")

    configured_steps = _integer(loading.get("steps"), f"{paths['config']}: loading.steps")
    configured_maximum = _finite(
        loading.get("maximum_displacement"),
        f"{paths['config']}: loading.maximum_displacement",
    )
    kkt_tolerance = _finite(
        loading.get("damage_kkt_tolerance"),
        f"{paths['config']}: loading.damage_kkt_tolerance",
    )
    history_values: list[dict[str, Any]] = []
    for index, row in enumerate(history):
        value = {
            "step": _csv_int(row, "step", paths["history"], index),
            "scheduled_step": _csv_int(row, "scheduled_step", paths["history"], index),
            "subdivision_level": _csv_int(row, "subdivision_level", paths["history"], index),
            "displacement": _csv_float(row, "displacement", paths["history"], index),
            "load_increment": _csv_float(row, "load_increment", paths["history"], index),
            "reaction_y": _csv_float(row, "reaction_y", paths["history"], index),
            "regularised_crack_length": _csv_float(
                row, "regularised_crack_length", paths["history"], index
            ),
            "rightmost_damaged_x": _csv_float(
                row, "rightmost_damaged_x", paths["history"], index
            ),
            "stagger_iterations": _csv_int(
                row, "stagger_iterations", paths["history"], index
            ),
            "stagger_error": _csv_float(row, "stagger_error", paths["history"], index),
            "stagger_converged": _csv_bool(row, "stagger_converged", paths["history"], index),
            "damage_kkt_relative": _csv_float(
                row, "damage_kkt_relative", paths["history"], index
            ),
        }
        if path_control_enabled:
            value.update(
                {
                    "fracture_energy": _csv_float(
                        row, "fracture_energy", paths["history"], index
                    ),
                    "control_phase": _csv_string(
                        row, "control_phase", paths["history"], index
                    ),
                    "phase_step": _csv_int(
                        row, "phase_step", paths["history"], index
                    ),
                    "load_factor": _csv_optional_float(
                        row, "load_factor", paths["history"], index
                    ),
                    "reference_displacement": _csv_optional_float(
                        row, "reference_displacement", paths["history"], index
                    ),
                    "path_coordinate": _csv_float(
                        row, "path_coordinate", paths["history"], index
                    ),
                    "path_increment": _csv_float(
                        row, "path_increment", paths["history"], index
                    ),
                    "control_target": _csv_float(
                        row, "control_target", paths["history"], index
                    ),
                    "control_value": _csv_float(
                        row, "control_value", paths["history"], index
                    ),
                    "control_residual_relative": _csv_float(
                        row,
                        "control_residual_relative",
                        paths["history"],
                        index,
                    ),
                    "load_factor_bound_status": _csv_optional_string(
                        row,
                        "load_factor_bound_status",
                        paths["history"],
                        index,
                    ),
                }
            )
            if value["control_phase"] not in {"displacement", "fracture_energy"}:
                raise ValueError(
                    f"{paths['history']}: row {index} has an unknown control phase"
                )
        if value["step"] != index:
            raise ValueError(f"{paths['history']}: accepted step numbering is not contiguous")
        fracture_phase = (
            path_control_enabled and value["control_phase"] == "fracture_energy"
        )
        subdivision_budget = (
            path_contract["maximum_subdivisions"]
            if fracture_phase
            else continuation["effective_controls"]["maximum_subdivisions"]
        )
        if not (
            0
            <= value["subdivision_level"]
            <= subdivision_budget
        ):
            raise ValueError(
                f"{paths['history']}: row {index} exceeds the effective "
                f"{'path-control' if fracture_phase else 'subdivision'} budget"
            )
        iteration_budget = (
            path_contract["snes_max_iterations"]
            if fracture_phase
            else continuation["effective_controls"]["stagger_max_iterations"]
        )
        if not (
            0
            <= value["stagger_iterations"]
            <= iteration_budget
        ):
            raise ValueError(
                f"{paths['history']}: row {index} exceeds the effective "
                f"{'path-control nonlinear' if fracture_phase else 'stagger'} budget"
            )
        scheduled_step_upper = (
            path_contract["switch_index"] + path_contract["steps"]
            if fracture_phase
            else configured_steps
        )
        if not 0 <= value["scheduled_step"] <= scheduled_step_upper:
            raise ValueError(f"{paths['history']}: row {index} has an invalid scheduled step")
        if not value["stagger_converged"]:
            raise ValueError(f"{paths['history']}: row {index} is not converged")
        if value["stagger_error"] < 0.0:
            raise ValueError(
                f"{paths['history']}: row {index} has a negative stagger error"
            )
        if not 0.0 <= value["damage_kkt_relative"] <= kkt_tolerance * (
            1.0 + 1.0e-8
        ):
            raise ValueError(
                f"{paths['history']}: row {index} has an invalid relative KKT certificate"
            )
        if index:
            previous = history_values[-1]
            if value["scheduled_step"] < previous["scheduled_step"]:
                raise ValueError(f"{paths['history']}: scheduled steps must not decrease")
        if not path_control_enabled:
            if index:
                previous = history_values[-1]
                if value["displacement"] <= previous["displacement"]:
                    raise ValueError(
                        f"{paths['history']}: accepted displacements must increase"
                    )
                increment = value["displacement"] - previous["displacement"]
                if not _close(value["load_increment"], increment):
                    raise ValueError(
                        f"{paths['history']}: row {index} load increment disagrees"
                    )
                minimum_increment = continuation["effective_controls"][
                    "minimum_increment"
                ]
                if value["load_increment"] + max(
                    1.0e-15, 1.0e-12 * minimum_increment
                ) < minimum_increment:
                    raise ValueError(
                        f"{paths['history']}: row {index} is below the effective "
                        "minimum increment"
                    )
        elif value["control_phase"] == "displacement":
            if path_phase_started:
                raise ValueError(
                    f"{paths['history']}: displacement control cannot resume after path control"
                )
            if value["phase_step"] != displacement_phase_steps:
                raise ValueError(
                    f"{paths['history']}: displacement phase_step is not contiguous"
                )
            displacement_phase_steps += 1
            if value["scheduled_step"] > path_contract["switch_index"]:
                raise ValueError(
                    f"{paths['history']}: displacement preload exceeds the switch index"
                )
            if value["load_factor"] is not None or value["reference_displacement"] is not None:
                raise ValueError(
                    f"{paths['history']}: displacement rows must not define a load factor"
                )
            if value["load_factor_bound_status"] is not None:
                raise ValueError(
                    f"{paths['history']}: displacement row has a load-factor bound status"
                )
            for name in ("path_coordinate", "control_target", "control_value"):
                if not _close(value[name], value["displacement"]):
                    raise ValueError(
                        f"{paths['history']}: row {index} {name} disagrees with displacement"
                    )
            if not _close(value["path_increment"], value["load_increment"]):
                raise ValueError(
                    f"{paths['history']}: row {index} path increment disagrees"
                )
            if not 0.0 <= value["control_residual_relative"] <= path_contract[
                "control_tolerance"
            ] * (1.0 + 1.0e-8):
                raise ValueError(
                    f"{paths['history']}: row {index} exceeds the control tolerance"
                )
            if value["displacement"] > path_contract["switch_displacement"] + max(
                1.0e-12, 1.0e-10 * path_contract["switch_displacement"]
            ):
                raise ValueError(
                    f"{paths['history']}: displacement preload exceeds the switch"
                )
            if index:
                previous = history_values[-1]
                if value["displacement"] <= previous["displacement"]:
                    raise ValueError(
                        f"{paths['history']}: preload displacements must increase"
                    )
                increment = value["displacement"] - previous["displacement"]
                if not _close(value["load_increment"], increment):
                    raise ValueError(
                        f"{paths['history']}: row {index} load increment disagrees"
                    )
                minimum_increment = continuation["effective_controls"][
                    "minimum_increment"
                ]
                if value["load_increment"] + max(
                    1.0e-15, 1.0e-12 * minimum_increment
                ) < minimum_increment:
                    raise ValueError(
                        f"{paths['history']}: row {index} is below the effective "
                        "minimum increment"
                    )
        else:
            if not index:
                raise ValueError(
                    f"{paths['history']}: fracture-energy phase cannot start at zero load"
                )
            previous = history_values[-1]
            if not path_phase_started:
                if previous["control_phase"] != "displacement" or not _close(
                    previous["displacement"], path_contract["switch_displacement"]
                ):
                    raise ValueError(
                        f"{paths['history']}: path control must start from the switch state"
                    )
                previous_control = previous["fracture_energy"]
                previous_actual_control = previous["fracture_energy"]
                path_phase_started = True
            else:
                previous_control = previous["control_target"]
                previous_actual_control = previous["control_value"]
            expected_phase_step = accepted_path_steps + 1
            if value["phase_step"] != expected_phase_step:
                raise ValueError(
                    f"{paths['history']}: fracture-energy phase_step is not contiguous"
                )
            accepted_path_steps += 1
            if value["control_target"] <= previous_control:
                raise ValueError(
                    f"{paths['history']}: fracture-energy targets must increase"
                )
            control_increment = value["control_target"] - previous_control
            if not _close(value["path_increment"], control_increment):
                raise ValueError(
                    f"{paths['history']}: row {index} control increment disagrees"
                )
            expected_dyadic_increment = math.ldexp(
                path_contract["target_increment"],
                -value["subdivision_level"],
            )
            if not _close(control_increment, expected_dyadic_increment):
                raise ValueError(
                    f"{paths['history']}: row {index} violates dyadic "
                    "subdivision provenance"
                )
            switch_record = next(
                item
                for item in reversed(history_values)
                if item["control_phase"] == "displacement"
            )
            nominal_ratio = (
                value["control_target"] - switch_record["fracture_energy"]
            ) / path_contract["target_increment"]
            ratio_tolerance = 1.0e-12 * max(1.0, abs(nominal_ratio))
            if not (
                0.0 < nominal_ratio <= path_contract["steps"] + ratio_tolerance
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} target is outside the configured "
                    "path window"
                )
            nominal_path_index = min(
                path_contract["steps"],
                max(1, math.ceil(nominal_ratio - ratio_tolerance)),
            )
            if value["scheduled_step"] != (
                path_contract["switch_index"] + nominal_path_index
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} scheduled step disagrees with "
                    "the nominal control target"
                )
            minimum_increment = path_contract["minimum_increment"]
            if control_increment + max(
                1.0e-15, 1.0e-12 * minimum_increment
            ) < minimum_increment:
                raise ValueError(
                    f"{paths['history']}: row {index} is below the path-control "
                    "minimum increment"
                )
            bound_slack = max(
                1.0e-12,
                1.0e-10
                * max(
                    abs(path_contract["load_lower_bound"]),
                    abs(path_contract["load_upper_bound"]),
                ),
            )
            if not (
                path_contract["load_lower_bound"] - bound_slack
                <= value["displacement"]
                <= path_contract["load_upper_bound"] + bound_slack
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} displacement is outside load bounds"
                )
            if value["load_factor"] is None or value["reference_displacement"] is None:
                raise ValueError(
                    f"{paths['history']}: row {index} has no path-control load factor"
                )
            if not _close(
                value["reference_displacement"],
                path_contract["switch_displacement"],
            ) or not _close(
                value["displacement"],
                value["load_factor"] * value["reference_displacement"],
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} load factor is inconsistent"
                )
            if value["load_factor_bound_status"] is not None:
                raise ValueError(
                    f"{paths['history']}: accepted path state hit a load-factor bound"
                )
            if not _close(value["path_coordinate"], value["control_target"]):
                raise ValueError(
                    f"{paths['history']}: row {index} path coordinate disagrees"
                )
            if not _csv_redundant_scalar_close(
                value["control_value"],
                value["fracture_energy"],
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} control value disagrees with energy"
                )
            actual_control_increment = (
                value["control_target"] - previous_actual_control
            )
            if actual_control_increment <= 0.0:
                raise ValueError(
                    f"{paths['history']}: row {index} has a non-positive actual "
                    "control increment"
                )
            absolute_residual = abs(
                value["control_value"] - value["control_target"]
            )
            computed_residual = absolute_residual / actual_control_increment
            if not _close(
                value["control_residual_relative"],
                computed_residual,
                rel=1.0e-8,
                abs_=1.0e-12,
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} control residual disagrees"
                )
            if path_contract["control_certificate_mode"] == "legacy_relative_only":
                if not (
                    0.0
                    <= value["control_residual_relative"]
                    <= path_contract["control_tolerance"] * (1.0 + 1.0e-8)
                ):
                    raise ValueError(
                        f"{paths['history']}: row {index} exceeds the control tolerance"
                    )
            else:
                control_certificate = fracture_energy_control_residual_certificate(
                    value["control_value"] - value["control_target"],
                    accepted_value=previous_actual_control,
                    target_value=value["control_target"],
                    relative_tolerance=path_contract["control_tolerance"],
                    absolute_tolerance=path_contract["control_absolute_tolerance"],
                )
                if not control_certificate.certified:
                    raise ValueError(
                        f"{paths['history']}: row {index} exceeds the control certificate"
                    )
            if not _close(
                value["load_increment"],
                value["displacement"] - previous["displacement"],
            ):
                raise ValueError(
                    f"{paths['history']}: row {index} load increment disagrees"
                )
        history_values.append(value)
    if not _close(history_values[0]["displacement"], 0.0):
        raise ValueError(f"{paths['history']}: first record must be zero load")
    if not _close(history_values[0]["load_increment"], 0.0):
        raise ValueError(f"{paths['history']}: zero-load record must have zero increment")
    if (
        not path_control_enabled
        and history_values[-1]["displacement"] > configured_maximum * (1.0 + 1.0e-12)
    ):
        raise ValueError(f"{paths['history']}: accepted displacement exceeds the configured window")
    for session in continuation["sessions"]:
        accepted_state = session["accepted_state"]
        accepted_step = accepted_state["accepted_step"]
        if accepted_step >= len(history_values) or not _close(
            accepted_state["displacement"],
            history_values[accepted_step]["displacement"],
        ):
            raise ValueError(
                f"{paths['continuation']}: session accepted state disagrees with history"
            )

    interface_status = _string(
        interface_history.get("status"), f"{paths['interface']}: status"
    )
    if interface_status not in {"partial", "complete"}:
        raise ValueError(f"{paths['interface']}: status must be partial or complete")
    protocol = _mapping(interface_history.get("protocol"), f"{paths['interface']}: protocol")
    if tuple(protocol.get(name) for name in ("start_node", "impact_node", "end_node")) != (
        protocol_names
    ):
        raise ValueError(f"{paths['interface']}: protocol node names disagree with config")
    thresholds = _list(protocol.get("thresholds"), f"{paths['interface']}: thresholds")
    if len(thresholds) != len(INTERFACE_DAMAGE_THRESHOLDS) or any(
        not _close(_finite(value, f"{paths['interface']}: threshold"), expected)
        for value, expected in zip(thresholds, INTERFACE_DAMAGE_THRESHOLDS, strict=True)
    ):
        raise ValueError(f"{paths['interface']}: protocol thresholds are not registered")
    raw_interface_records = _list(
        interface_history.get("records"), f"{paths['interface']}: records"
    )
    if len(raw_interface_records) != len(history_values):
        raise ValueError(f"{paths['interface']}: record count disagrees with mechanical history")
    interface_records: list[dict[str, Any]] = []
    previous_thresholds: list[dict[str, Any]] | None = None
    for index, (raw, mechanical) in enumerate(
        zip(raw_interface_records, history_values, strict=True)
    ):
        item = _mapping(raw, f"{paths['interface']}: records[{index}]")
        expected_fields = {
            "accepted_step": mechanical["step"],
            "scheduled_step": mechanical["scheduled_step"],
            "subdivision_level": mechanical["subdivision_level"],
        }
        for field, expected in expected_fields.items():
            observed = _integer(
                item.get(field),
                f"{paths['interface']}: records[{index}].{field}",
            )
            if observed != expected:
                raise ValueError(f"{paths['interface']}: records[{index}] disagrees on {field}")
        displacement = _finite(
            item.get("displacement"), f"{paths['interface']}: records[{index}].displacement"
        )
        if not _close(displacement, mechanical["displacement"]):
            raise ValueError(f"{paths['interface']}: records[{index}] displacement disagrees")
        threshold_results = _validate_threshold_results(
            item.get("threshold_results"),
            f"{paths['interface']}: records[{index}].threshold_results",
        )
        consensus = _string(
            item.get("threshold_consensus"),
            f"{paths['interface']}: records[{index}].threshold_consensus",
        )
        if consensus != _consensus(threshold_results):
            raise ValueError(f"{paths['interface']}: records[{index}] consensus is inconsistent")
        if previous_thresholds is not None:
            for threshold_index, (previous, current) in enumerate(
                zip(previous_thresholds, threshold_results, strict=True)
            ):
                context = f"{paths['interface']}: threshold index {threshold_index}"
                if previous["reached_interface"] and not current["reached_interface"]:
                    raise ValueError(f"{context} violates irreversible interface reachability")
                for name in ("interface_forward_advance", "penetration_forward_advance"):
                    if current[name] + 1.0e-12 < previous[name]:
                        raise ValueError(f"{context} has decreasing {name}")
                previous_distance = previous["closest_main_node_to_impact"]
                current_distance = current["closest_main_node_to_impact"]
                if (
                    previous_distance is not None
                    and current_distance is not None
                    and current_distance > previous_distance + 1.0e-12
                ):
                    raise ValueError(f"{context} has increasing closest impact distance")
        interface_records.append(
            {
                "displacement": displacement,
                "threshold_consensus": consensus,
                "threshold_results": threshold_results,
                "raw": item,
            }
        )
        previous_thresholds = threshold_results

    completion = _read_json(paths["completion"]) if paths["completion"].is_file() else None
    if completion is not None:
        completion_status = _string(
            completion.get("status"), f"{paths['completion']}: status"
        )
        if completion_status != "complete":
            raise ValueError(f"{paths['completion']}: unknown completion status")
        if interface_status != "complete":
            raise ValueError(f"{paths['completion']}: complete case has partial interface history")
        if not _close(
            _finite(
                completion.get("final_displacement"),
                f"{paths['completion']}: final_displacement",
            ),
            history_values[-1]["displacement"],
        ):
            raise ValueError(f"{paths['completion']}: final displacement disagrees with history")
        if path_control_enabled:
            final = history_values[-1]
            final_phase = _string(
                completion.get("final_control_phase"),
                f"{paths['completion']}: final_control_phase",
            )
            final_target = _finite(
                completion.get("final_control_target"),
                f"{paths['completion']}: final_control_target",
            )
            final_value = _finite(
                completion.get("final_control_value"),
                f"{paths['completion']}: final_control_value",
            )
            if (
                final_phase != final["control_phase"]
                or not _close(final_target, final["control_target"])
                or not _close(final_value, final["control_value"])
            ):
                raise ValueError(
                    f"{paths['completion']}: final control state disagrees with history"
                )
            if final_phase != "fracture_energy":
                raise ValueError(
                    f"{paths['completion']}: hybrid completion must end in path control"
                )
            if not _boolean(
                completion.get("control_targets_exhausted"),
                f"{paths['completion']}: control_targets_exhausted",
            ):
                raise ValueError(
                    f"{paths['completion']}: hybrid control targets are not exhausted"
                )
            if _integer(
                completion.get("accepted_path_steps"),
                f"{paths['completion']}: accepted_path_steps",
            ) != accepted_path_steps:
                raise ValueError(
                    f"{paths['completion']}: accepted path-step count disagrees with history"
                )
            if _integer(
                completion.get("configured_path_targets"),
                f"{paths['completion']}: configured_path_targets",
            ) != path_contract["steps"]:
                raise ValueError(
                    f"{paths['completion']}: configured path-target count disagrees"
                )
    elif interface_status == "complete":
        raise ValueError(f"{paths['interface']}: complete history requires completion.json")

    attempts: list[dict[str, Any]] = []
    if paths["attempts"].is_file():
        attempt_payload = _read_json(paths["attempts"])
        attempt_status = _string(
            attempt_payload.get("status"), f"{paths['attempts']}: status"
        )
        if attempt_status not in {"partial", "complete"}:
            raise ValueError(f"{paths['attempts']}: status must be partial or complete")
        if attempt_status != interface_status:
            raise ValueError(f"{paths['attempts']}: status disagrees with interface history")
        raw_attempts = _list(attempt_payload.get("records"), f"{paths['attempts']}: records")
        continuation_states = [
            continuation["base_controls"],
            *(session["controls_after"] for session in continuation["sessions"]),
        ]
        for index, raw in enumerate(raw_attempts, start=1):
            item = _mapping(raw, f"{paths['attempts']}: records[{index - 1}]")
            if _integer(item.get("attempt"), f"{paths['attempts']}: attempt") != index:
                raise ValueError(f"{paths['attempts']}: attempt numbering is not contiguous")
            attempt_phase = (
                _string(
                    item.get("control_phase"),
                    f"{paths['attempts']}: control_phase",
                )
                if path_control_enabled
                else "displacement"
            )
            if attempt_phase not in {"displacement", "fracture_energy"}:
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} has an unknown control phase"
                )
            accepted_displacement = _finite(
                item.get("accepted_displacement"),
                f"{paths['attempts']}: accepted_displacement",
            )
            scheduled_step = _integer(
                item.get("scheduled_step"), f"{paths['attempts']}: scheduled_step"
            )
            subdivision_level = _integer(
                item.get("subdivision_level"),
                f"{paths['attempts']}: subdivision_level",
            )
            iterations = _integer(
                item.get("iterations"), f"{paths['attempts']}: iterations"
            )
            attempt_scheduled_upper = (
                path_contract["switch_index"] + path_contract["steps"]
                if attempt_phase == "fracture_energy"
                else configured_steps
            )
            if not 1 <= scheduled_step <= attempt_scheduled_upper:
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} has an invalid scheduled step"
                )
            failure_type = _string(
                item.get("failure_type"), f"{paths['attempts']}: failure_type"
            )
            error = _optional_finite(item.get("error"), f"{paths['attempts']}: error")
            will_subdivide = _boolean(
                item.get("will_subdivide"), f"{paths['attempts']}: will_subdivide"
            )
            if attempt_phase == "displacement":
                if path_control_enabled and scheduled_step > path_contract["switch_index"]:
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} exceeds the switch index"
                    )
                target_displacement = _finite(
                    item.get("target_displacement"),
                    f"{paths['attempts']}: target_displacement",
                )
                load_increment = _finite(
                    item.get("load_increment"), f"{paths['attempts']}: load_increment"
                )
                if (
                    accepted_displacement < 0.0
                    or target_displacement <= accepted_displacement
                    or target_displacement > configured_maximum * (1.0 + 1.0e-12)
                    or not _close(
                        load_increment, target_displacement - accepted_displacement
                    )
                ):
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} has invalid loads"
                    )
                raw_attempt_controls = item.get("effective_continuation_controls")
                if raw_attempt_controls is None:
                    attempt_controls = continuation["effective_controls"]
                else:
                    attempt_controls = _mapping(
                        raw_attempt_controls,
                        f"{paths['attempts']}: records[{index - 1}]"
                        ".effective_continuation_controls",
                    )
                    try:
                        continuation_control_increase(
                            continuation["base_controls"],
                            attempt_controls,
                            allow_equal=True,
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"{paths['attempts']}: record {index - 1} has invalid "
                            f"effective continuation controls: {exc}"
                        ) from exc
                    if attempt_controls not in continuation_states:
                        raise ValueError(
                            f"{paths['attempts']}: record {index - 1} uses controls "
                            "outside the continuation session chain"
                        )
                if not (
                    0
                    <= subdivision_level
                    <= attempt_controls["maximum_subdivisions"]
                ):
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} exceeds the "
                        "effective subdivision budget"
                    )
                if not 0 <= iterations <= attempt_controls["stagger_max_iterations"]:
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} exceeds the "
                        "effective stagger budget"
                    )
                minimum_increment = attempt_controls["minimum_increment"]
                if load_increment + max(
                    1.0e-15, 1.0e-12 * minimum_increment
                ) < minimum_increment:
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} is below the "
                        "effective minimum increment"
                    )
                attempts.append(
                    {
                        "attempt": index,
                        "control_phase": attempt_phase,
                        "accepted_displacement": accepted_displacement,
                        "target_displacement": target_displacement,
                        "load_increment": load_increment,
                        "scheduled_step": scheduled_step,
                        "subdivision_level": subdivision_level,
                        "failure_type": failure_type,
                        "iterations": iterations,
                        "error": error,
                        "will_subdivide": will_subdivide,
                        "raw": item,
                    }
                )
                continue

            accepted_step_before = _integer(
                item.get("accepted_step_before_attempt"),
                f"{paths['attempts']}: accepted_step_before_attempt",
            )
            if not 0 <= accepted_step_before < len(history_values):
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} has an invalid accepted step"
                )
            accepted_record = history_values[accepted_step_before]
            accepted_control = _finite(
                item.get("accepted_control"),
                f"{paths['attempts']}: accepted_control",
            )
            target_control = _finite(
                item.get("target_control"),
                f"{paths['attempts']}: target_control",
            )
            trial_displacement = _optional_finite(
                item.get("trial_displacement"),
                f"{paths['attempts']}: trial_displacement",
            )
            load_increment = _optional_finite(
                item.get("load_increment"), f"{paths['attempts']}: load_increment"
            )
            control_increment = _finite(
                item.get("control_increment"),
                f"{paths['attempts']}: control_increment",
            )
            accepted_record_control = (
                accepted_record["control_target"]
                if accepted_record["control_phase"] == "fracture_energy"
                else accepted_record["fracture_energy"]
            )
            if (
                (
                    accepted_record["control_phase"] == "displacement"
                    and not _close(
                        accepted_record["displacement"],
                        path_contract["switch_displacement"],
                    )
                )
                or not _close(accepted_displacement, accepted_record["displacement"])
                or not _close(accepted_control, accepted_record_control)
                or target_control <= accepted_control
                or not _close(control_increment, target_control - accepted_control)
            ):
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} has invalid path controls"
                )
            switch_record = next(
                record
                for record in reversed(history_values)
                if record["control_phase"] == "displacement"
            )
            nominal_ratio = (
                target_control - switch_record["fracture_energy"]
            ) / path_contract["target_increment"]
            ratio_tolerance = 1.0e-12 * max(1.0, abs(nominal_ratio))
            if not (
                0.0 < nominal_ratio <= path_contract["steps"] + ratio_tolerance
            ):
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} target is outside the "
                    "configured path window"
                )
            nominal_path_index = min(
                path_contract["steps"],
                max(1, math.ceil(nominal_ratio - ratio_tolerance)),
            )
            if scheduled_step != path_contract["switch_index"] + nominal_path_index:
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} scheduled step disagrees "
                    "with the nominal control target"
                )
            if trial_displacement is None:
                if load_increment is not None:
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} has an orphan load increment"
                    )
            else:
                if not (
                    path_contract["load_lower_bound"]
                    <= trial_displacement
                    <= path_contract["load_upper_bound"]
                ) or load_increment is None or not _close(
                    load_increment, trial_displacement - accepted_displacement
                ):
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} has an invalid "
                        "trial displacement"
                    )
            if item.get("target_displacement") is not None:
                legacy_target = _finite(
                    item.get("target_displacement"),
                    f"{paths['attempts']}: target_displacement",
                )
                if trial_displacement is None or not _close(
                    legacy_target, trial_displacement
                ):
                    raise ValueError(
                        f"{paths['attempts']}: record {index - 1} has inconsistent "
                        "trial displacement aliases"
                    )
            if not (
                0
                <= subdivision_level
                <= path_contract["maximum_subdivisions"]
            ):
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} exceeds the "
                    "path-control subdivision budget"
                )
            if not -1 <= iterations <= path_contract["snes_max_iterations"]:
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} exceeds the "
                    "path-control nonlinear budget"
                )
            minimum_increment = path_contract["minimum_increment"]
            expected_dyadic_increment = math.ldexp(
                path_contract["target_increment"],
                -subdivision_level,
            )
            if not _close(control_increment, expected_dyadic_increment):
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} violates dyadic "
                    "subdivision provenance"
                )
            if not _path_control_increment_meets_minimum(
                control_increment,
                minimum_increment,
                accepted_control=accepted_control,
                target_control=target_control,
            ):
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} is below the "
                    "path-control minimum increment"
                )
            if will_subdivide and not (
                path_contract["adaptive"]
                and subdivision_level < path_contract["maximum_subdivisions"]
                and _path_control_increment_meets_minimum(
                    0.5 * control_increment,
                    minimum_increment,
                    accepted_control=accepted_control,
                    target_control=target_control,
                )
            ):
                raise ValueError(
                    f"{paths['attempts']}: record {index - 1} cannot be subdivided"
                )
            attempts.append(
                {
                    "attempt": index,
                    "control_phase": attempt_phase,
                    "accepted_step_before_attempt": accepted_step_before,
                    "accepted_displacement": accepted_displacement,
                    "accepted_control": accepted_control,
                    "target_control": target_control,
                    "trial_displacement": trial_displacement,
                    "load_increment": load_increment,
                    "control_increment": control_increment,
                    "scheduled_step": scheduled_step,
                    "subdivision_level": subdivision_level,
                    "failure_type": failure_type,
                    "iterations": iterations,
                    "error": error,
                    "will_subdivide": will_subdivide,
                    "raw": item,
                }
            )
        attempt_summary = {
            "present": True,
            "status": attempt_status,
            "unaccepted_attempts": len(attempts),
            "last_unaccepted_attempt": attempts[-1]["raw"] if attempts else None,
        }
    else:
        attempt_summary = {
            "present": False,
            "status": None,
            "unaccepted_attempts": None,
            "last_unaccepted_attempt": None,
        }

    nominal_increment = configured_maximum / configured_steps
    completed_nominal_steps = 0
    nominal_step_limit = (
        path_contract["switch_index"] if path_control_enabled else configured_steps
    )
    displacement_records = (
        [
            item
            for item in history_values
            if item["control_phase"] == "displacement"
        ]
        if path_control_enabled
        else history_values
    )
    for scheduled_step in range(1, nominal_step_limit + 1):
        target = scheduled_step * nominal_increment
        if any(_close(item["displacement"], target) for item in displacement_records):
            completed_nominal_steps = scheduled_step
        else:
            break
    last = history_values[-1]
    next_nominal_target = (
        None
        if completed_nominal_steps == nominal_step_limit
        else (completed_nominal_steps + 1) * nominal_increment
    )
    current_control_phase = "displacement"
    next_control_target: float | None = next_nominal_target
    if path_control_enabled:
        current_control_phase = last["control_phase"]
        current_path_attempt = next(
            (
                attempt
                for attempt in reversed(attempts)
                if attempt["control_phase"] == "fracture_energy"
                and attempt["accepted_step_before_attempt"] == last["step"]
            ),
            None,
        )
        if current_path_attempt is not None or (
            last["control_phase"] == "displacement"
            and _close(last["displacement"], path_contract["switch_displacement"])
        ):
            current_control_phase = "fracture_energy"
        if completion is not None:
            next_control_target = None
        elif current_control_phase == "displacement":
            next_control_target = next_nominal_target
        elif current_path_attempt is not None:
            accepted_control = current_path_attempt["accepted_control"]
            target_control = current_path_attempt["target_control"]
            next_control_target = (
                accepted_control + 0.5 * (target_control - accepted_control)
                if current_path_attempt["will_subdivide"]
                else target_control
            )
        elif accepted_path_steps == 0:
            next_control_target = last["fracture_energy"] + path_contract[
                "target_increment"
            ]
        else:
            path_records = [
                item
                for item in history_values
                if item["control_phase"] == "fracture_energy"
            ]
            final_path = path_records[-1]
            switch_record = displacement_records[-1]
            nominal_ratio = (
                final_path["control_target"] - switch_record["fracture_energy"]
            ) / path_contract["target_increment"]
            nearest_nominal = round(nominal_ratio)
            if _close(nominal_ratio, nearest_nominal) and nearest_nominal >= path_contract[
                "steps"
            ]:
                next_control_target = None
            elif _close(nominal_ratio, nearest_nominal):
                next_control_target = (
                    final_path["control_target"] + path_contract["target_increment"]
                )
            else:
                next_control_target = (
                    final_path["control_target"] + final_path["path_increment"]
                )
    reactions = [item["reaction_y"] for item in history_values]
    peak_index = max(range(len(reactions)), key=reactions.__getitem__)
    subdivisions = [item["subdivision_level"] for item in history_values]
    method_status = _hybrid_method_status_summary(
        paths["method_status"],
        path_contract=path_contract,
        last=last,
        accepted_path_steps=accepted_path_steps,
        completion_present=completion is not None,
    )
    failure = _read_json(paths["failure"]) if paths["failure"].is_file() else None
    if failure is None and method_status is not None and method_status["status"] == "failed":
        failure = {
            "source": "method_status.json",
            "failure_type": method_status["failure_type"],
            "failure_message": method_status["failure_message"],
        }
    run_status = (
        "complete"
        if completion is not None
        else (
            "failed"
            if method_status is not None and method_status["status"] == "failed"
            else "partial"
        )
    )
    accepted_increment_records = displacement_records[1:]
    progress = {
        "configured_steps": configured_steps,
        "accepted_states_including_zero": len(history_values),
        "completed_nominal_steps": completed_nominal_steps,
        "last_scheduled_step": last["scheduled_step"],
        "last_accepted_displacement": last["displacement"],
        "configured_maximum_displacement": configured_maximum,
        "nominal_increment": nominal_increment,
        "next_nominal_target": next_nominal_target,
        "maximum_subdivision_level": max(subdivisions),
        "minimum_accepted_increment": min(
            item["load_increment"] for item in accepted_increment_records
        )
        if accepted_increment_records
        else 0.0,
        "maximum_stagger_iterations": max(
            item["stagger_iterations"] for item in history_values
        ),
        "last_stagger_iterations": last["stagger_iterations"],
        "window_maximum_reaction_y": reactions[peak_index],
        "window_maximum_reaction_displacement": history_values[peak_index][
            "displacement"
        ],
        "window_maximum_reaction_at_last_accepted_state": peak_index
        == len(history_values) - 1,
    }
    if path_control_enabled:
        progress.update(
            {
                "current_control_phase": current_control_phase,
                "accepted_path_steps": accepted_path_steps,
                "configured_path_targets": path_contract["steps"],
                "next_control_target": next_control_target,
                "use_energy_predictor": path_contract["use_energy_predictor"],
                ENERGY_PREDICTOR_LATCH_FIELD: path_contract[
                    ENERGY_PREDICTOR_LATCH_FIELD
                ],
            }
        )
    else:
        progress["displacement_fraction"] = last["displacement"] / configured_maximum
    return {
        "case": directory.name,
        "directory": str(directory),
        "run_status": run_status,
        "screen_only": resolution < 2.0 or any(level > 0 for level in subdivisions),
        "mesh": {
            "nx": nx,
            "ny": ny,
            "ell_over_triangle_diameter": resolution,
            "resolution_gate_pass": resolution >= 2.0,
        },
        "runtime": {"mpi_ranks": mpi_ranks},
        "continuation": continuation,
        "progress": progress,
        "interface": {
            "history_status": interface_status,
            "final_threshold_consensus": raw_interface_records[-1]["threshold_consensus"],
            "first_any_threshold_impact": _impact_bracket(
                interface_records, require_all=False
            ),
            "first_all_threshold_impact": _impact_bracket(
                interface_records, require_all=True
            ),
            "candidate_persistence": _candidate_persistence(interface_records),
            "final_threshold_results": raw_interface_records[-1]["threshold_results"],
        },
        "failure": failure,
        "method_status": method_status,
        "attempts": attempt_summary,
        "field_outputs": {
            name: _hdf5_signature_status(directory / filename)
            for name, filename in {
                "damage": "damage.h5",
                "displacement": "displacement.h5",
                "material": "material.h5",
            }.items()
        },
        "all_persisted_states_passed": True,
        "interpretation": (
            (
                "A complete progress report validates the persisted completion and all "
                "accepted states. Right-censored interface events remain censored physical "
                "observations."
            )
            if run_status == "complete"
            else (
                "A failed run report validates persisted accepted states and exposes the "
                "hybrid method failure certificate. The accepted checkpoint may remain "
                "resumable under its recorded restart policy."
                if run_status == "failed"
                else
                "A partial report validates persisted accepted states only. A right-censored "
                "interface event and a reaction maximum at the last accepted state are not "
                "completed-window conclusions."
            )
        ),
    }


def _validate_completed_hybrid_case(
    directory: Path,
    *,
    allow_underresolved: bool,
    allow_subdivisions: bool,
) -> dict[str, Any]:
    """Promote a completed hybrid progress report to the strict audit contract."""
    progress_report = case_progress_report(directory)
    if progress_report["run_status"] != "complete":
        raise ValueError(f"{directory}: hybrid case is not complete")

    paths = {
        name: directory / filename
        for name, filename in {
            "config": "config.resolved.json",
            "runtime": "runtime.json",
            "completion": "completion.json",
            "graph": "graph_metrics.json",
            "attempts": "attempt_history.json",
            "interface": "interface_history.json",
            "history": "history.csv",
        }.items()
    }
    config = _read_json(paths["config"])
    runtime = _read_json(paths["runtime"])
    completion = _read_json(paths["completion"])
    graph_metrics = _read_json(paths["graph"])
    interface_history = _read_json(paths["interface"])
    history = _read_history(paths["history"])
    loading = _mapping(config.get("loading"), f"{paths['config']}: loading")
    graph = _mapping(config.get("graph"), f"{paths['config']}: graph")
    path_contract = _path_control_progress_contract(config, loading, paths["config"])
    if path_contract is None:  # pragma: no cover - dispatch invariant.
        raise ValueError(f"{paths['config']}: hybrid audit requires path control")

    completion_context = str(paths["completion"])
    if _string(completion.get("status"), f"{completion_context}: status") != "complete":
        raise ValueError(f"{completion_context}: hybrid case is not complete")
    if not _boolean(
        completion.get("all_steps_converged"),
        f"{completion_context}: all_steps_converged",
    ):
        raise ValueError(f"{completion_context}: not all accepted steps converged")
    accepted_steps = _integer(
        completion.get("accepted_load_steps"),
        f"{completion_context}: accepted_load_steps",
    )
    if accepted_steps != len(history) - 1:
        raise ValueError(
            f"{completion_context}: accepted step count disagrees with history"
        )
    completion_condition = _string(
        completion.get("completion_condition"),
        f"{completion_context}: completion_condition",
    )
    if completion_condition != "fracture_energy_queue_exhausted":
        raise ValueError(
            f"{completion_context}: hybrid completion did not exhaust the energy queue"
        )
    if not _boolean(
        completion.get("control_targets_exhausted"),
        f"{completion_context}: control_targets_exhausted",
    ):
        raise ValueError(f"{completion_context}: hybrid control targets are not exhausted")

    continuation = progress_report["continuation"]
    completion_controls = _mapping(
        completion.get("effective_continuation_controls"),
        f"{completion_context}: effective_continuation_controls",
    )
    if completion_controls != continuation["effective_controls"]:
        raise ValueError(
            f"{completion_context}: effective continuation controls disagree"
        )
    if _integer(
        completion.get("continuation_sessions"),
        f"{completion_context}: continuation_sessions",
    ) != continuation["session_count"]:
        raise ValueError(f"{completion_context}: continuation session count disagrees")

    attempts = progress_report["attempts"]
    if not attempts["present"] or attempts["status"] != "complete":
        raise ValueError(
            f"{directory / 'attempt_history.json'}: completed hybrid audit requires "
            "a complete attempt history"
        )
    if attempts["unaccepted_attempts"] != 0:
        if not allow_subdivisions:
            raise ValueError(
                f"{paths['attempts']}: completed hybrid audit requires "
                "no unaccepted nonlinear attempts"
            )
        attempt_payload = _read_json(paths["attempts"])
        if _integer(
            attempt_payload.get("schema_version"),
            f"{paths['attempts']}: schema_version",
        ) != 1:
            raise ValueError(f"{paths['attempts']}: unsupported attempt-history schema")
        raw_attempts = _list(
            attempt_payload.get("records"),
            f"{paths['attempts']}: records",
        )
        if len(raw_attempts) != attempts["unaccepted_attempts"]:
            raise ValueError(f"{paths['attempts']}: attempt count disagrees with progress audit")
        if any(
            not _boolean(
                _mapping(raw, f"{paths['attempts']}: records[{index}]").get(
                    "will_subdivide"
                ),
                f"{paths['attempts']}: records[{index}].will_subdivide",
            )
            for index, raw in enumerate(raw_attempts)
        ):
            raise ValueError(
                f"{paths['attempts']}: a completed adaptive run contains a terminal "
                "unaccepted attempt"
            )

    history_values: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    switch_record: dict[str, Any] | None = None
    for index, row in enumerate(history):
        value = {
            "step": _csv_int(row, "step", paths["history"], index),
            "scheduled_step": _csv_int(
                row,
                "scheduled_step",
                paths["history"],
                index,
            ),
            "subdivision_level": _csv_int(
                row,
                "subdivision_level",
                paths["history"],
                index,
            ),
            "displacement": _csv_float(
                row,
                "displacement",
                paths["history"],
                index,
            ),
            "load_increment": _csv_float(
                row,
                "load_increment",
                paths["history"],
                index,
            ),
            "reaction_y": _csv_float(row, "reaction_y", paths["history"], index),
            "elastic_energy": _csv_float(
                row,
                "elastic_energy",
                paths["history"],
                index,
            ),
            "fracture_energy": _csv_float(
                row,
                "fracture_energy",
                paths["history"],
                index,
            ),
            "total_internal_energy": _csv_float(
                row,
                "total_internal_energy",
                paths["history"],
                index,
            ),
            "regularised_crack_length": _csv_float(
                row,
                "regularised_crack_length",
                paths["history"],
                index,
            ),
            "damage_kkt_relative": _csv_float(
                row,
                "damage_kkt_relative",
                paths["history"],
                index,
            ),
            "stagger_iterations": _csv_int(
                row,
                "stagger_iterations",
                paths["history"],
                index,
            ),
            "stagger_converged": _csv_bool(
                row,
                "stagger_converged",
                paths["history"],
                index,
            ),
            "control_phase": _csv_string(
                row,
                "control_phase",
                paths["history"],
                index,
            ),
            "phase_step": _csv_int(row, "phase_step", paths["history"], index),
            "control_target": _csv_float(
                row,
                "control_target",
                paths["history"],
                index,
            ),
            "control_value": _csv_float(
                row,
                "control_value",
                paths["history"],
                index,
            ),
        }
        if value["control_phase"] == "displacement":
            switch_record = value
        else:
            path_records.append(value)
        history_values.append(value)

    if switch_record is None or not _close(
        switch_record["displacement"],
        path_contract["switch_displacement"],
    ):
        raise ValueError(f"{paths['history']}: hybrid switch state disagrees with config")
    configured_targets = path_contract["steps"]
    accepted_targets = len(path_records)
    if accepted_targets < configured_targets or (
        accepted_targets != configured_targets and not allow_subdivisions
    ):
        raise ValueError(
            f"{paths['history']}: accepted fracture-energy target count disagrees "
            "with config"
        )
    switch_energy = switch_record["fracture_energy"]
    if not allow_subdivisions:
        for target_index, record in enumerate(path_records, start=1):
            expected_target = (
                switch_energy + target_index * path_contract["target_increment"]
            )
            if (
                record["phase_step"] != target_index
                or record["subdivision_level"] != 0
                or not _close(record["control_target"], expected_target)
            ):
                raise ValueError(
                    f"{paths['history']}: fracture-energy target {target_index} "
                    "disagrees with the configured queue"
                )
    final = path_records[-1]
    final_target = _finite(
        completion.get("final_control_target"),
        f"{completion_context}: final_control_target",
    )
    final_value = _finite(
        completion.get("final_control_value"),
        f"{completion_context}: final_control_value",
    )
    if (
        _string(
            completion.get("final_control_phase"),
            f"{completion_context}: final_control_phase",
        )
        != "fracture_energy"
        or not _close(final_target, final["control_target"])
        or not _close(final_value, final["control_value"])
        or not _close(
            final_target,
            switch_energy
            + configured_targets * path_contract["target_increment"],
        )
    ):
        raise ValueError(
            f"{completion_context}: final fracture-energy control disagrees with "
            "the exhausted queue"
        )
    completion_accepted_targets = _integer(
        completion.get("accepted_path_steps"),
        f"{completion_context}: accepted_path_steps",
    )
    completion_configured_targets = _integer(
        completion.get("configured_path_targets"),
        f"{completion_context}: configured_path_targets",
    )
    if (
        completion_accepted_targets != accepted_targets
        or completion_configured_targets != configured_targets
    ):
        raise ValueError(
            f"{completion_context}: configured fracture-energy target count disagrees"
        )

    maximum_subdivision = max(
        value["subdivision_level"] for value in history_values
    )
    if maximum_subdivision > 0 and not allow_subdivisions:
        raise ValueError(
            f"{paths['history']}: adaptive subdivisions require --allow-subdivisions "
            "for screening"
        )
    unaccepted_attempts = _integer(
        attempts["unaccepted_attempts"],
        f"{paths['attempts']}: unaccepted attempt count",
    )
    if (maximum_subdivision > 0) != (unaccepted_attempts > 0):
        raise ValueError(
            f"{paths['attempts']}: accepted refined states and unaccepted attempt "
            "provenance disagree"
        )
    method_status = progress_report["method_status"]
    if maximum_subdivision > 0 and method_status is None:
        raise ValueError(
            f"{directory / 'method_status.json'}: adaptive hybrid completion requires "
            "a method-status certificate"
        )
    if method_status is not None and (
        method_status["status"] != "complete"
        or method_status["accepted_step"] != accepted_steps
        or method_status["accepted_path_steps"] != accepted_targets
        or method_status["configured_path_targets"] != configured_targets
        or method_status["pending_control_targets"] != 0
        or method_status["control_phase"] != "fracture_energy"
        or method_status["completion_condition"] != completion_condition
        or not _close(method_status["accepted_control"], final_target)
        or not _close(method_status["accepted_displacement"], final["displacement"])
    ):
        raise ValueError(
            f"{directory / 'method_status.json'}: method completion disagrees with "
            "history, completion, or exhausted queue"
        )
    resolution = progress_report["mesh"]["ell_over_triangle_diameter"]
    resolution_pass = progress_report["mesh"]["resolution_gate_pass"]
    if not resolution_pass and not allow_underresolved:
        raise ValueError(
            f"{directory}: ell/h_K={resolution:.6g} is below 2; "
            "use --allow-underresolved only for load-window screening"
        )

    geometry = _mapping(config.get("geometry"), f"{paths['config']}: geometry")
    diagnostics = _mapping(runtime.get("diagnostics"), f"{paths['runtime']}: diagnostics")
    mesh_diagnostics = _mapping(
        diagnostics.get("mesh"),
        f"{paths['runtime']}: diagnostics.mesh",
    )
    model_diagnostics = _mapping(
        diagnostics.get("model"),
        f"{paths['runtime']}: diagnostics.model",
    )
    diagonal = _string(geometry.get("diagonal"), f"{paths['config']}: geometry.diagonal")
    pin = _string(
        geometry.get("x_pin_corner"),
        f"{paths['config']}: geometry.x_pin_corner",
    )
    if mesh_diagnostics.get("diagonal") != diagonal:
        raise ValueError(f"{paths['runtime']}: mesh diagonal disagrees with config")
    if model_diagnostics.get("horizontal_rigid_body_pin") != pin:
        raise ValueError(f"{paths['runtime']}: rigid-body pin disagrees with config")

    protocol = _mapping(
        interface_history.get("protocol"),
        f"{paths['interface']}: protocol",
    )
    raw_interface_records = _list(
        interface_history.get("records"),
        f"{paths['interface']}: records",
    )
    interface_records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_interface_records):
        item = _mapping(raw, f"{paths['interface']}: records[{index}]")
        interface_records.append(
            {
                "accepted_step": index,
                "scheduled_step": history_values[index]["scheduled_step"],
                "subdivision_level": history_values[index]["subdivision_level"],
                "displacement": history_values[index]["displacement"],
                "threshold_consensus": _string(
                    item.get("threshold_consensus"),
                    f"{paths['interface']}: records[{index}].threshold_consensus",
                ),
                "threshold_results": _validate_threshold_results(
                    item.get("threshold_results"),
                    f"{paths['interface']}: records[{index}].threshold_results",
                ),
                "raw": item,
            }
        )
    final_interaction = _mapping(
        graph_metrics.get("interface_interaction"),
        f"{paths['graph']}: interface_interaction",
    )
    if final_interaction.get("protocol") != protocol:
        raise ValueError(f"{paths['graph']}: final protocol disagrees with interface history")
    if final_interaction.get("threshold_results") != raw_interface_records[-1].get(
        "threshold_results"
    ):
        raise ValueError(f"{paths['graph']}: final thresholds disagree with interface history")
    if final_interaction.get("threshold_consensus") != interface_records[-1][
        "threshold_consensus"
    ]:
        raise ValueError(f"{paths['graph']}: final consensus disagrees with interface history")
    crack = _mapping(graph_metrics.get("G_crack"), f"{paths['graph']}: G_crack")
    crack_threshold = _finite(
        graph.get("crack_threshold"),
        f"{paths['config']}: crack_threshold",
    )
    if not _close(
        _finite(crack.get("threshold"), f"{paths['graph']}: G_crack.threshold"),
        crack_threshold,
    ):
        raise ValueError(f"{paths['graph']}: G_crack threshold disagrees with config")

    reactions = [value["reaction_y"] for value in history_values]
    peak_index = max(range(len(reactions)), key=reactions.__getitem__)
    configured_steps = _integer(
        loading.get("steps"),
        f"{paths['config']}: loading.steps",
    )
    configured_maximum = _finite(
        loading.get("maximum_displacement"),
        f"{paths['config']}: loading.maximum_displacement",
    )
    return {
        "case": directory.name,
        "directory": str(directory),
        "screen_only": not resolution_pass or maximum_subdivision > 0,
        "mesh": {
            **progress_report["mesh"],
            "diagonal": diagonal,
            "horizontal_rigid_body_pin": pin,
        },
        "runtime": progress_report["runtime"],
        "continuation": continuation,
        "loading": {
            "configured_steps": configured_steps,
            "accepted_steps": accepted_steps,
            "maximum_displacement": configured_maximum,
            "maximum_subdivision_level": maximum_subdivision,
            "window_maximum_reaction_y": reactions[peak_index],
            "window_maximum_reaction_displacement": history_values[peak_index][
                "displacement"
            ],
            "window_maximum_reaction_at_right_endpoint": peak_index
            == len(history_values) - 1,
        },
        "path_control": {
            "functional": "fracture_energy",
            "switch_displacement": path_contract["switch_displacement"],
            "switch_index": path_contract["switch_index"],
            "target_increment": path_contract["target_increment"],
            "use_energy_predictor": path_contract["use_energy_predictor"],
            ENERGY_PREDICTOR_LATCH_FIELD: path_contract[
                ENERGY_PREDICTOR_LATCH_FIELD
            ],
            "configured_targets": configured_targets,
            "accepted_targets": len(path_records),
            **(
                {
                    "unaccepted_attempts": unaccepted_attempts,
                    "adaptive_refinement_used": True,
                }
                if maximum_subdivision > 0
                else {}
            ),
            "final_control_target": final_target,
            "final_control_value": final_value,
            "completion_condition": completion_condition,
            "control_targets_exhausted": True,
        },
        "interface": {
            "protocol": protocol,
            "final_threshold_consensus": interface_records[-1][
                "threshold_consensus"
            ],
            "first_any_threshold_impact": progress_report["interface"][
                "first_any_threshold_impact"
            ],
            "first_all_threshold_impact": progress_report["interface"][
                "first_all_threshold_impact"
            ],
            "candidate_persistence": progress_report["interface"][
                "candidate_persistence"
            ],
            "records": raw_interface_records,
        },
        "all_checks_passed": True,
        "limitations": [
            "Candidate classifications are threshold- and mesh-dependent geometric screens.",
            "The model is a same-material isotropic weak plane without elastic mismatch, "
            "crystal anisotropy or a tension/compression split.",
            "Fracture-energy continuation does not require the final displacement to equal "
            "the displacement-control window maximum.",
        ],
        "_config": config,
        "_history": history_values,
        "_interface_records": interface_records,
    }


def _canonical_mirror_config(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("source_path", None)
    _mapping(value.get("output"), "config.output").pop("directory", None)
    geometry = _mapping(value.get("geometry"), "config.geometry")
    geometry.pop("diagonal", None)
    geometry.pop("x_pin_corner", None)
    value.pop("graph_nodes", None)
    return value


def _canonical_continuation_config(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("source_path", None)
    _mapping(value.get("output"), "config.output").pop("directory", None)
    loading = _mapping(value.get("loading"), "config.loading")
    for name in ("stagger_max_iterations", "maximum_subdivisions", "minimum_increment"):
        loading.pop(name, None)
    solver = _mapping(value.get("solver"), "config.solver")
    solver.pop("aitken_max_relaxation", None)
    return value


def _continuation_history(directory: Path) -> list[dict[str, Any]]:
    path = directory / "history.csv"
    rows = _read_history(path)
    fields = (
        "reaction_y",
        "elastic_energy",
        "fracture_energy",
        "total_internal_energy",
        "regularised_crack_length",
    )
    return [
        {
            "index": index,
            "displacement": _csv_float(row, "displacement", path, index),
            **{name: _csv_float(row, name, path, index) for name in fields},
        }
        for index, row in enumerate(rows)
    ]


def continuation_path_report(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    relative_tolerance: float = 1.0e-4,
) -> dict[str, Any]:
    """Compare common accepted states while changing only continuation controls."""
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    directories = [Path(value).resolve() for value in (reference_directory, candidate_directory)]
    reports = [case_progress_report(directory) for directory in directories]
    configs = [_read_json(directory / "config.resolved.json") for directory in directories]
    if _canonical_continuation_config(configs[0]) != _canonical_continuation_config(configs[1]):
        raise ValueError(
            "continuation configurations differ beyond output, iteration, subdivision, "
            "minimum-increment and Aitken controls"
        )
    if reports[0]["runtime"]["mpi_ranks"] != reports[1]["runtime"]["mpi_ranks"]:
        raise ValueError("continuation cases use different MPI rank counts")

    histories = [_continuation_history(directory) for directory in directories]
    interfaces = [
        _list(
            _read_json(directory / "interface_history.json").get("records"),
            f"{directory / 'interface_history.json'}: records",
        )
        for directory in directories
    ]
    common: list[tuple[dict[str, Any], dict[str, Any]]] = []
    first_index = second_index = 0
    while first_index < len(histories[0]) and second_index < len(histories[1]):
        first = histories[0][first_index]
        second = histories[1][second_index]
        if _close(first["displacement"], second["displacement"]):
            common.append((first, second))
            first_index += 1
            second_index += 1
        elif first["displacement"] < second["displacement"]:
            first_index += 1
        else:
            second_index += 1
    if len(common) < 2:
        raise ValueError("continuation cases have fewer than two common accepted states")

    response_fields = (
        "reaction_y",
        "elastic_energy",
        "fracture_energy",
        "total_internal_energy",
        "regularised_crack_length",
    )
    maximum_errors = {name: 0.0 for name in response_fields}
    for first, second in common:
        for name in response_fields:
            scale = max(abs(first[name]), abs(second[name]), 1.0)
            error = abs(first[name] - second[name]) / scale
            maximum_errors[name] = max(maximum_errors[name], error)
            if error > relative_tolerance:
                raise ValueError(
                    f"continuation histories differ in {name} at "
                    f"u={first['displacement']:.12g}"
                )

        first_interface = _mapping(
            interfaces[0][first["index"]],
            f"{directories[0]}: interface record {first['index']}",
        )
        second_interface = _mapping(
            interfaces[1][second["index"]],
            f"{directories[1]}: interface record {second['index']}",
        )
        if first_interface.get("threshold_consensus") != second_interface.get(
            "threshold_consensus"
        ):
            raise ValueError(
                "continuation interface consensus differs at "
                f"u={first['displacement']:.12g}"
            )
        threshold_pairs = zip(
            _validate_threshold_results(
                first_interface.get("threshold_results"),
                f"{directories[0]}: threshold results",
            ),
            _validate_threshold_results(
                second_interface.get("threshold_results"),
                f"{directories[1]}: threshold results",
            ),
            strict=True,
        )
        for threshold_index, (left, right) in enumerate(threshold_pairs):
            if (
                left["classification"] != right["classification"]
                or left["reached_interface"] != right["reached_interface"]
            ):
                raise ValueError(
                    "continuation threshold classification differs at "
                    f"u={first['displacement']:.12g}, index={threshold_index}"
                )

    return {
        "study": "inclined-interface numerical continuation path control",
        "relative_tolerance": relative_tolerance,
        "common_accepted_states": len(common),
        "common_displacement_maximum": common[-1][0]["displacement"],
        "maximum_normalised_response_errors": maximum_errors,
        "all_common_states_agree": True,
        "reference": reports[0],
        "candidate": reports[1],
        "interpretation": (
            "Agreement shows that the tested continuation controls recover the same accepted "
            "branch over common loads. It does not validate the under-resolved physical path."
        ),
    }


def _canonical_increment_pair_config(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("source_path", None)
    _mapping(value.get("output"), "config.output").pop("directory", None)
    path_control = _mapping(value.get("path_control"), "config.path_control")
    for name in (
        "target_increment",
        "steps",
        "minimum_increment",
        "use_energy_predictor",
    ):
        if name not in path_control:
            raise ValueError(f"config.path_control has no explicit {name}")
        path_control.pop(name)
    path_control.pop(ENERGY_PREDICTOR_LATCH_FIELD, None)
    return value


def _increment_pair_case_data(directory: Path) -> dict[str, Any]:
    history_path = directory / "history.csv"
    interface_path = directory / "interface_history.json"
    rows = _read_history(history_path)
    interface_records = _list(
        _read_json(interface_path).get("records"),
        f"{interface_path}: records",
    )
    if len(rows) != len(interface_records):
        raise ValueError(f"{directory}: history and interface record counts differ")

    states: list[dict[str, Any]] = []
    switch_energy: float | None = None
    first_all_impact: dict[str, Any] | None = None
    for index, (row, raw_interface) in enumerate(
        zip(rows, interface_records, strict=True)
    ):
        phase = _csv_string(row, "control_phase", history_path, index)
        fracture_energy = _csv_float(row, "fracture_energy", history_path, index)
        target = (
            _csv_float(row, "control_target", history_path, index)
            if phase == "fracture_energy"
            else fracture_energy
        )
        interface = _mapping(
            raw_interface,
            f"{interface_path}: records[{index}]",
        )
        thresholds = _validate_threshold_results(
            interface.get("threshold_results"),
            f"{interface_path}: records[{index}].threshold_results",
        )
        if first_all_impact is None and all(
            item["reached_interface"] for item in thresholds
        ):
            first_all_impact = {
                "status": "observed",
                "accepted_step": index,
                "control_phase": phase,
                "control_target": target,
                "fracture_energy": fracture_energy,
            }
        if phase == "displacement":
            switch_energy = fracture_energy
            continue
        if phase != "fracture_energy":
            raise ValueError(f"{history_path}: row {index} has an unknown control phase")
        states.append(
            {
                "accepted_step": index,
                "phase_step": _csv_int(row, "phase_step", history_path, index),
                "subdivision_level": _csv_int(
                    row,
                    "subdivision_level",
                    history_path,
                    index,
                ),
                "target": target,
                **{
                    name: _csv_float(row, name, history_path, index)
                    for name in INCREMENT_PAIR_RESPONSE_FIELDS
                },
                "rightmost_damaged_x": _csv_float(
                    row,
                    "rightmost_damaged_x",
                    history_path,
                    index,
                ),
                "threshold_consensus": _string(
                    interface.get("threshold_consensus"),
                    f"{interface_path}: records[{index}].threshold_consensus",
                ),
                "threshold_signature": tuple(
                    (
                        item["threshold"],
                        item["classification"],
                        item["reached_interface"],
                    )
                    for item in thresholds
                ),
            }
        )
    if switch_energy is None or not states:
        raise ValueError(f"{directory}: no hybrid switch/path states")
    if first_all_impact is None:
        first_all_impact = {
            "status": "right_censored",
            "accepted_step": None,
            "control_phase": None,
            "control_target": None,
            "fracture_energy": None,
        }
    return {
        "switch_energy": switch_energy,
        "path": states,
        "first_all_impact": first_all_impact,
    }


def _increment_pair_normalised_difference(
    left: float,
    right: float,
    relative_tolerance: float,
) -> float:
    scale = max(
        abs(left),
        abs(right),
        INCREMENT_PAIR_RESPONSE_ABSOLUTE_TOLERANCE / relative_tolerance,
    )
    return abs(left - right) / scale


def increment_pair_report(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    allow_reference_subdivisions: bool = False,
    relative_tolerance: float = 1.0e-2,
) -> dict[str, Any]:
    """Strictly audit one completed path against a half-increment completed path."""
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    reference = Path(reference_directory).resolve()
    candidate = Path(candidate_directory).resolve()
    if reference == candidate:
        raise ValueError("increment-pair roles must reference distinct result directories")

    reference_audit = case_audit_report(
        reference,
        allow_subdivisions=allow_reference_subdivisions,
    )
    candidate_audit = case_audit_report(candidate)
    reference_config = _read_json(reference / "config.resolved.json")
    candidate_config = _read_json(candidate / "config.resolved.json")
    reference_path_config = _mapping(
        reference_config.get("path_control"),
        f"{reference}: path_control",
    )
    candidate_path_config = _mapping(
        candidate_config.get("path_control"),
        f"{candidate}: path_control",
    )
    reference_predictor, reference_predictor_disable_after = (
        _energy_predictor_policy(
            reference_path_config,
            f"{reference}: path_control",
        )
    )
    candidate_predictor, candidate_predictor_disable_after = (
        _energy_predictor_policy(
            candidate_path_config,
            f"{candidate}: path_control",
        )
    )
    if _canonical_increment_pair_config(
        reference_config
    ) != _canonical_increment_pair_config(candidate_config):
        raise ValueError(
            "increment-pair configurations differ beyond output directory and the "
            "registered target/minimum increments, target count, and semantically "
            "authenticated energy-predictor policy"
        )

    reference_increment = _finite(
        reference_path_config.get("target_increment"),
        f"{reference}: target_increment",
    )
    candidate_increment = _finite(
        candidate_path_config.get("target_increment"),
        f"{candidate}: target_increment",
    )
    reference_steps = _integer(
        reference_path_config.get("steps"),
        f"{reference}: path steps",
    )
    candidate_steps = _integer(
        candidate_path_config.get("steps"),
        f"{candidate}: path steps",
    )
    reference_minimum = _finite(
        reference_path_config.get("minimum_increment"),
        f"{reference}: minimum_increment",
    )
    candidate_minimum = _finite(
        candidate_path_config.get("minimum_increment"),
        f"{candidate}: minimum_increment",
    )
    if not (
        _close(candidate_increment, 0.5 * reference_increment)
        and candidate_steps == 2 * reference_steps
        and _close(candidate_minimum, 0.5 * reference_minimum)
    ):
        raise ValueError(
            "candidate must use exactly half the target/minimum increment and twice "
            "the configured path targets"
        )
    reference_predictor_policy = _energy_predictor_window_summary(
        use_energy_predictor=reference_predictor,
        disable_after=reference_predictor_disable_after,
        configured_steps=reference_steps,
        target_increment=reference_increment,
    )
    candidate_predictor_policy = _energy_predictor_window_summary(
        use_energy_predictor=candidate_predictor,
        disable_after=candidate_predictor_disable_after,
        configured_steps=candidate_steps,
        target_increment=candidate_increment,
    )
    if (
        candidate_predictor_policy[
            "effective_last_enabled_nominal_path_step"
        ]
        != 2
        * reference_predictor_policy[
            "effective_last_enabled_nominal_path_step"
        ]
    ):
        raise ValueError(
            "increment-pair effective predictor policies differ over the "
            "authenticated physical control window"
        )

    reference_runtime = _read_json(reference / "runtime.json")
    candidate_runtime = _read_json(candidate / "runtime.json")
    runtime_fields = (
        "mpi_ranks",
        "implementation_fingerprint",
        "runtime_fingerprint",
    )
    for name in runtime_fields:
        if reference_runtime.get(name) != candidate_runtime.get(name):
            raise ValueError(f"increment-pair {name} differs")
    if reference_audit["mesh"] != candidate_audit["mesh"]:
        raise ValueError("increment-pair authenticated meshes differ")
    if candidate_audit["loading"]["maximum_subdivision_level"] != 0:
        raise ValueError("increment-pair candidate must not contain accepted subdivisions")
    if candidate_audit["path_control"].get("unaccepted_attempts", 0) != 0:
        raise ValueError("increment-pair candidate must not contain unaccepted attempts")

    reference_data = _increment_pair_case_data(reference)
    candidate_data = _increment_pair_case_data(candidate)
    if not _close(
        reference_data["switch_energy"],
        candidate_data["switch_energy"],
    ):
        raise ValueError("increment-pair accepted switch fracture energies differ")
    cumulative = reference_increment * reference_steps
    if not _close(cumulative, candidate_increment * candidate_steps):
        raise ValueError("increment-pair terminal energy windows differ")
    expected_terminal = reference_data["switch_energy"] + cumulative
    if not (
        _close(
            reference_audit["path_control"]["final_control_target"],
            expected_terminal,
        )
        and _close(
            candidate_audit["path_control"]["final_control_target"],
            expected_terminal,
        )
    ):
        raise ValueError("increment-pair authenticated terminal targets differ")

    reference_path = reference_data["path"]
    candidate_path = candidate_data["path"]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for nominal_index in range(1, reference_steps + 1):
        expected_target = (
            reference_data["switch_energy"] + nominal_index * reference_increment
        )
        reference_matches = [
            state for state in reference_path if _close(state["target"], expected_target)
        ]
        if len(reference_matches) != 1:
            raise ValueError(
                f"reference nominal target {nominal_index} is missing or duplicated"
            )
        candidate_index = 2 * nominal_index - 1
        if candidate_index >= len(candidate_path):
            raise ValueError(
                f"candidate has no phase_step {2 * nominal_index} for nominal pairing"
            )
        candidate_state = candidate_path[candidate_index]
        if (
            candidate_state["phase_step"] != 2 * nominal_index
            or not _close(candidate_state["target"], expected_target)
        ):
            raise ValueError(
                f"candidate phase_step {2 * nominal_index} disagrees with the "
                "reference nominal target"
            )
        pairs.append((reference_matches[0], candidate_state))
    if len(pairs) != reference_steps:
        raise ValueError("increment-pair did not authenticate every reference nominal target")

    maximum_errors = {name: 0.0 for name in INCREMENT_PAIR_RESPONSE_FIELDS}
    maximum_tip_difference = 0.0
    classification_mismatches = 0
    first_classification_mismatch: dict[str, Any] | None = None
    for nominal_index, (left, right) in enumerate(pairs, start=1):
        for name in INCREMENT_PAIR_RESPONSE_FIELDS:
            maximum_errors[name] = max(
                maximum_errors[name],
                _increment_pair_normalised_difference(
                    left[name],
                    right[name],
                    relative_tolerance,
                ),
            )
        maximum_tip_difference = max(
            maximum_tip_difference,
            abs(left["rightmost_damaged_x"] - right["rightmost_damaged_x"]),
        )
        if (
            left["threshold_consensus"] != right["threshold_consensus"]
            or left["threshold_signature"] != right["threshold_signature"]
        ):
            classification_mismatches += 1
            if first_classification_mismatch is None:
                first_classification_mismatch = {
                    "reference_nominal_index": nominal_index,
                    "control_target": left["target"],
                    "reference_consensus": left["threshold_consensus"],
                    "candidate_consensus": right["threshold_consensus"],
                    "reference_thresholds": left["threshold_signature"],
                    "candidate_thresholds": right["threshold_signature"],
                }

    maximum_response_error = max(maximum_errors.values())
    response_passed = maximum_response_error <= relative_tolerance
    geometry = _mapping(reference_config.get("geometry"), f"{reference}: geometry")
    mesh_h = _finite(geometry.get("length"), f"{reference}: geometry.length") / _integer(
        geometry.get("nx"), f"{reference}: geometry.nx"
    )
    tip_passed = maximum_tip_difference <= mesh_h + 1.0e-12
    classification_passed = classification_mismatches == 0

    reference_impact = reference_data["first_all_impact"]
    candidate_impact = candidate_data["first_all_impact"]
    impact_drift: float | None = None
    if (
        reference_impact["status"] == "observed"
        and candidate_impact["status"] == "observed"
    ):
        impact_drift = abs(
            reference_impact["control_target"]
            - candidate_impact["control_target"]
        )
        impact_passed = impact_drift <= candidate_increment + 1.0e-12
        impact_status = "observed_in_both"
    elif reference_impact["status"] == candidate_impact["status"]:
        impact_passed = False
        impact_status = "right_censored_in_both"
    else:
        impact_passed = False
        impact_status = "observation_status_mismatch"

    all_passed = (
        response_passed
        and tip_passed
        and classification_passed
        and impact_passed
    )
    reference_subdivisions = reference_audit["loading"][
        "maximum_subdivision_level"
    ]
    return {
        "schema_version": 1,
        "study": "fracture_energy_single_increment_halving",
        "configuration_authentication": {
            "passed": True,
            "reference_target_increment": reference_increment,
            "candidate_target_increment": candidate_increment,
            "reference_steps": reference_steps,
            "candidate_steps": candidate_steps,
            "reference_minimum_increment": reference_minimum,
            "candidate_minimum_increment": candidate_minimum,
            "switch_fracture_energy": reference_data["switch_energy"],
            "terminal_control_target": expected_terminal,
            "use_energy_predictor": reference_predictor,
            "energy_predictor_policy": {
                "reference": reference_predictor_policy,
                "candidate": candidate_predictor_policy,
                "candidate_nominal_steps_per_reference_step": 2,
                "effective_physical_cutoff_authenticated": True,
            },
            "mpi_ranks": reference_runtime["mpi_ranks"],
            "implementation_fingerprint": reference_runtime[
                "implementation_fingerprint"
            ],
            "runtime_fingerprint": reference_runtime["runtime_fingerprint"],
            "mesh_h": mesh_h,
        },
        "reference": {
            "directory": str(reference),
            "configured_targets": reference_steps,
            "accepted_targets": reference_audit["path_control"]["accepted_targets"],
            "maximum_subdivision_level": reference_subdivisions,
            "unaccepted_attempts": reference_audit["path_control"].get(
                "unaccepted_attempts", 0
            ),
            "adaptive_subdivisions_explicitly_allowed": allow_reference_subdivisions,
        },
        "candidate": {
            "directory": str(candidate),
            "configured_targets": candidate_steps,
            "accepted_targets": candidate_audit["path_control"]["accepted_targets"],
            "maximum_subdivision_level": 0,
            "unaccepted_attempts": 0,
        },
        "pairing": {
            "rule": "reference nominal target k equals candidate phase_step 2k",
            "expected_reference_nominal_targets": reference_steps,
            "compared_targets": len(pairs),
            "first_control_target": pairs[0][0]["target"],
            "last_control_target": pairs[-1][0]["target"],
            "all_reference_nominal_targets_authenticated": True,
        },
        "response_convergence": {
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": INCREMENT_PAIR_RESPONSE_ABSOLUTE_TOLERANCE,
            "maximum_normalised_difference": maximum_response_error,
            "maximum_normalised_difference_by_field": maximum_errors,
            "passed": response_passed,
        },
        "crack_tip_convergence": {
            "tolerance": mesh_h,
            "maximum_absolute_difference": maximum_tip_difference,
            "passed": tip_passed,
        },
        "threshold_classification_convergence": {
            "registered_thresholds": list(INTERFACE_DAMAGE_THRESHOLDS),
            "mismatch_count": classification_mismatches,
            "first_mismatch": first_classification_mismatch,
            "passed": classification_passed,
        },
        "first_all_threshold_impact_energy": {
            "status": impact_status,
            "reference": reference_impact,
            "candidate": candidate_impact,
            "absolute_drift": impact_drift,
            "tolerance": candidate_increment,
            "passed": impact_passed,
        },
        "gate_summary": {
            "configuration_authenticated": True,
            "all_nominal_targets_compared": len(pairs) == reference_steps,
            "response_convergence_passed": response_passed,
            "crack_tip_convergence_passed": tip_passed,
            "threshold_classification_convergence_passed": classification_passed,
            "first_all_threshold_impact_energy_passed": impact_passed,
        },
        "all_checks_passed": all_passed,
        "screen_only": reference_subdivisions > 0,
        "interpretation": (
            "Passing establishes increment-halving stability at every reference nominal "
            "fracture-energy target for this mesh and finite control window. It does not "
            "establish mesh convergence or a final interface mechanism."
        ),
    }


def _canonical_window_extension_config(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("source_path", None)
    value.pop("output", None)
    path_control = _mapping(value.get("path_control"), "config.path_control")
    if "steps" not in path_control:
        raise ValueError("config.path_control has no explicit 'steps'")
    path_control.pop("steps")
    path_control.pop(ENERGY_PREDICTOR_LATCH_FIELD, None)
    return value


def _window_extension_case_data(directory: Path) -> dict[str, Any]:
    history_path = directory / "history.csv"
    interface_path = directory / "interface_history.json"
    rows = _read_history(history_path)
    interface_records = _list(
        _read_json(interface_path).get("records"),
        f"{interface_path}: records",
    )
    if len(rows) != len(interface_records):
        raise ValueError(f"{directory}: history and interface record counts differ")

    states: list[dict[str, Any]] = []
    for index, (row, raw_interface) in enumerate(
        zip(rows, interface_records, strict=True)
    ):
        interface = _mapping(
            raw_interface,
            f"{interface_path}: records[{index}]",
        )
        normalised_thresholds = _validate_threshold_results(
            interface.get("threshold_results"),
            f"{interface_path}: records[{index}].threshold_results",
        )
        thresholds: list[dict[str, Any]] = []
        for threshold_index, item in enumerate(normalised_thresholds):
            raw_threshold = item["raw"]
            thresholds.append(
                {
                    "threshold": item["threshold"],
                    "classification": item["classification"],
                    "reached_interface": item["reached_interface"],
                    "main_component_nodes": _integer(
                        raw_threshold.get("main_component_nodes"),
                        f"{interface_path}: records[{index}].threshold_results"
                        f"[{threshold_index}].main_component_nodes",
                    ),
                    "closest_main_node_to_impact": item[
                        "closest_main_node_to_impact"
                    ],
                    "interface_forward_advance": item[
                        "interface_forward_advance"
                    ],
                    "penetration_forward_advance": item[
                        "penetration_forward_advance"
                    ],
                    "interface_active_edge_length": item[
                        "interface_active_edge_length"
                    ],
                    "penetration_active_edge_length": item[
                        "penetration_active_edge_length"
                    ],
                }
            )
        phase = _csv_string(row, "control_phase", history_path, index)
        if phase not in {"displacement", "fracture_energy"}:
            raise ValueError(f"{history_path}: row {index} has an unknown control phase")
        states.append(
            {
                "accepted_step": _csv_int(row, "step", history_path, index),
                "scheduled_step": _csv_int(
                    row,
                    "scheduled_step",
                    history_path,
                    index,
                ),
                "subdivision_level": _csv_int(
                    row,
                    "subdivision_level",
                    history_path,
                    index,
                ),
                "control_phase": phase,
                "phase_step": _csv_int(row, "phase_step", history_path, index),
                "load_factor_bound_status": _csv_optional_string(
                    row,
                    "load_factor_bound_status",
                    history_path,
                    index,
                ),
                "scheduler": {
                    "reference_displacement": _csv_optional_float(
                        row,
                        "reference_displacement",
                        history_path,
                        index,
                    ),
                    "path_coordinate": _csv_float(
                        row,
                        "path_coordinate",
                        history_path,
                        index,
                    ),
                    "path_increment": _csv_float(
                        row,
                        "path_increment",
                        history_path,
                        index,
                    ),
                    "control_target": _csv_float(
                        row,
                        "control_target",
                        history_path,
                        index,
                    ),
                },
                "responses": {
                    "displacement": _csv_float(
                        row,
                        "displacement",
                        history_path,
                        index,
                    ),
                    "load_factor": _csv_optional_float(
                        row,
                        "load_factor",
                        history_path,
                        index,
                    ),
                    "reaction_y": _csv_float(
                        row,
                        "reaction_y",
                        history_path,
                        index,
                    ),
                    "elastic_energy": _csv_float(
                        row,
                        "elastic_energy",
                        history_path,
                        index,
                    ),
                    "fracture_energy": _csv_float(
                        row,
                        "fracture_energy",
                        history_path,
                        index,
                    ),
                    "total_internal_energy": _csv_float(
                        row,
                        "total_internal_energy",
                        history_path,
                        index,
                    ),
                    "regularised_crack_length": _csv_float(
                        row,
                        "regularised_crack_length",
                        history_path,
                        index,
                    ),
                    "rightmost_damaged_x": _csv_float(
                        row,
                        "rightmost_damaged_x",
                        history_path,
                        index,
                    ),
                },
                "threshold_consensus": _string(
                    interface.get("threshold_consensus"),
                    f"{interface_path}: records[{index}].threshold_consensus",
                ),
                "thresholds": thresholds,
            }
        )
    return {
        "states": states,
        "displacement": [
            state for state in states if state["control_phase"] == "displacement"
        ],
        "path": [
            state for state in states if state["control_phase"] == "fracture_energy"
        ],
    }


def _window_tolerance_ratio(
    left: float,
    right: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[float, float]:
    difference = abs(left - right)
    scale = max(
        absolute_tolerance,
        relative_tolerance * max(abs(left), abs(right)),
    )
    return difference, difference / scale


def _window_state_marker(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_step": state["accepted_step"],
        "scheduled_step": state["scheduled_step"],
        "control_phase": state["control_phase"],
        "phase_step": state["phase_step"],
        "control_target": state["scheduler"]["control_target"],
        "responses": dict(state["responses"]),
        "threshold_consensus": state["threshold_consensus"],
        "threshold_results": [dict(item) for item in state["thresholds"]],
    }


def _window_event_marker(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_step": state["accepted_step"],
        "control_phase": state["control_phase"],
        "phase_step": state["phase_step"],
        "control_target": state["scheduler"]["control_target"],
        "displacement": state["responses"]["displacement"],
        "fracture_energy": state["responses"]["fracture_energy"],
        "threshold_consensus": state["threshold_consensus"],
    }


def _window_first_all_threshold_event(
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    for state in states:
        if all(item["reached_interface"] for item in state["thresholds"]):
            return {"status": "observed", **_window_event_marker(state)}
    return {
        "status": "right_censored",
        "accepted_step": None,
        "control_phase": None,
        "phase_step": None,
        "control_target": None,
        "displacement": None,
        "fracture_energy": None,
        "threshold_consensus": None,
    }


def _window_first_directional_measure(
    states: list[dict[str, Any]],
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    for state in states:
        for threshold in state["thresholds"]:
            for direction, field in (
                ("interface", "interface_forward_advance"),
                ("penetration", "penetration_forward_advance"),
            ):
                if threshold[field] > absolute_tolerance:
                    return {
                        "status": "observed",
                        **_window_event_marker(state),
                        "threshold": threshold["threshold"],
                        "direction": direction,
                        "forward_advance": threshold[field],
                    }
    return {"status": "not_observed"}


def _window_first_candidate(
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    for state in states:
        if state["threshold_consensus"] in CANDIDATE_CLASSIFICATIONS:
            return {
                "status": "observed",
                "classification": state["threshold_consensus"],
                **_window_event_marker(state),
            }
    return {"status": "not_observed"}


def _window_candidate_persistence(
    states: list[dict[str, Any]],
    *,
    consecutive_states: int = 2,
) -> dict[str, Any]:
    first_window: list[dict[str, Any]] | None = None
    for end in range(consecutive_states - 1, len(states)):
        window = states[end - consecutive_states + 1 : end + 1]
        values = {state["threshold_consensus"] for state in window}
        if len(values) == 1 and next(iter(values)) in CANDIDATE_CLASSIFICATIONS:
            first_window = window
            break
    final_window = states[-consecutive_states:]
    final_values = {state["threshold_consensus"] for state in final_window}
    final_gate_pass = (
        len(final_window) == consecutive_states
        and len(final_values) == 1
        and next(iter(final_values)) in CANDIDATE_CLASSIFICATIONS
    )
    if first_window is None:
        return {
            "required_consecutive_states": consecutive_states,
            "status": "not_confirmed",
            "first_candidate": None,
            "first_confirmation": None,
            "final_gate_pass": final_gate_pass,
            "final_confirmed_classification": (
                next(iter(final_values)) if final_gate_pass else None
            ),
        }
    return {
        "required_consecutive_states": consecutive_states,
        "status": "confirmed",
        "first_confirmed_classification": first_window[0][
            "threshold_consensus"
        ],
        "first_candidate": _window_event_marker(first_window[0]),
        "first_confirmation": _window_event_marker(first_window[-1]),
        "final_gate_pass": final_gate_pass,
        "final_confirmed_classification": (
            next(iter(final_values)) if final_gate_pass else None
        ),
    }


def _authenticate_window_scheduler_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for name in (
        "accepted_step",
        "scheduled_step",
        "subdivision_level",
        "control_phase",
        "phase_step",
        "load_factor_bound_status",
    ):
        if reference[name] != candidate[name]:
            raise ValueError(
                "window-extension prefix scheduler differs at accepted step "
                f"{reference['accepted_step']} in {name}"
            )
    for name in WINDOW_EXTENSION_SCHEDULER_FLOAT_FIELDS:
        left = reference["scheduler"][name]
        right = candidate["scheduler"][name]
        if left is None or right is None:
            if left is not right:
                raise ValueError(
                    "window-extension prefix scheduler nullability differs at "
                    f"accepted step {reference['accepted_step']} in {name}"
                )
        elif not _close(left, right):
            raise ValueError(
                "window-extension prefix scheduler differs at accepted step "
                f"{reference['accepted_step']} in {name}"
            )


def _window_response_comparison(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    maximum_absolute = {name: 0.0 for name in WINDOW_EXTENSION_RESPONSE_FIELDS}
    maximum_ratio = {name: 0.0 for name in WINDOW_EXTENSION_RESPONSE_FIELDS}
    mismatch_count = 0
    nullability_mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    for reference, candidate in pairs:
        for name in WINDOW_EXTENSION_RESPONSE_FIELDS:
            left = reference["responses"][name]
            right = candidate["responses"][name]
            if left is None or right is None:
                if left is right:
                    continue
                mismatch_count += 1
                nullability_mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "accepted_step": reference["accepted_step"],
                        "control_phase": reference["control_phase"],
                        "phase_step": reference["phase_step"],
                        "field": name,
                        "reference": left,
                        "candidate": right,
                        "reason": "nullability_mismatch",
                    }
                continue
            difference, ratio = _window_tolerance_ratio(
                left,
                right,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            maximum_absolute[name] = max(maximum_absolute[name], difference)
            maximum_ratio[name] = max(maximum_ratio[name], ratio)
            if ratio > 1.0:
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "accepted_step": reference["accepted_step"],
                        "control_phase": reference["control_phase"],
                        "phase_step": reference["phase_step"],
                        "field": name,
                        "reference": left,
                        "candidate": right,
                        "absolute_difference": difference,
                        "tolerance_ratio": ratio,
                    }
    return {
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "maximum_absolute_difference_by_field": maximum_absolute,
        "maximum_tolerance_ratio_by_field": maximum_ratio,
        "mismatch_count": mismatch_count,
        "nullability_mismatch_count": nullability_mismatches,
        "first_mismatch": first_mismatch,
        "passed": mismatch_count == 0,
    }


def _window_interface_comparison(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    maximum_absolute = {
        name: 0.0 for name in WINDOW_EXTENSION_INTERFACE_FLOAT_FIELDS
    }
    maximum_ratio = {
        name: 0.0 for name in WINDOW_EXTENSION_INTERFACE_FLOAT_FIELDS
    }
    discrete_mismatches = 0
    continuous_mismatches = 0
    nullability_mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    for reference, candidate in pairs:
        if reference["threshold_consensus"] != candidate["threshold_consensus"]:
            discrete_mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "accepted_step": reference["accepted_step"],
                    "field": "threshold_consensus",
                    "reference": reference["threshold_consensus"],
                    "candidate": candidate["threshold_consensus"],
                }
        for threshold_index, (left, right) in enumerate(
            zip(reference["thresholds"], candidate["thresholds"], strict=True)
        ):
            for name in (
                "threshold",
                "classification",
                "reached_interface",
                "main_component_nodes",
            ):
                if left[name] != right[name]:
                    discrete_mismatches += 1
                    if first_mismatch is None:
                        first_mismatch = {
                            "accepted_step": reference["accepted_step"],
                            "threshold_index": threshold_index,
                            "field": name,
                            "reference": left[name],
                            "candidate": right[name],
                        }
            for name in WINDOW_EXTENSION_INTERFACE_FLOAT_FIELDS:
                left_value = left[name]
                right_value = right[name]
                if left_value is None or right_value is None:
                    if left_value is right_value:
                        continue
                    continuous_mismatches += 1
                    nullability_mismatches += 1
                    if first_mismatch is None:
                        first_mismatch = {
                            "accepted_step": reference["accepted_step"],
                            "threshold_index": threshold_index,
                            "field": name,
                            "reference": left_value,
                            "candidate": right_value,
                            "reason": "nullability_mismatch",
                        }
                    continue
                difference, ratio = _window_tolerance_ratio(
                    left_value,
                    right_value,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                )
                maximum_absolute[name] = max(maximum_absolute[name], difference)
                maximum_ratio[name] = max(maximum_ratio[name], ratio)
                if ratio > 1.0:
                    continuous_mismatches += 1
                    if first_mismatch is None:
                        first_mismatch = {
                            "accepted_step": reference["accepted_step"],
                            "threshold_index": threshold_index,
                            "field": name,
                            "reference": left_value,
                            "candidate": right_value,
                            "absolute_difference": difference,
                            "tolerance_ratio": ratio,
                        }
    mismatch_count = discrete_mismatches + continuous_mismatches
    return {
        "registered_thresholds": list(INTERFACE_DAMAGE_THRESHOLDS),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "maximum_absolute_difference_by_field": maximum_absolute,
        "maximum_tolerance_ratio_by_field": maximum_ratio,
        "discrete_mismatch_count": discrete_mismatches,
        "continuous_mismatch_count": continuous_mismatches,
        "nullability_mismatch_count": nullability_mismatches,
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
        "passed": mismatch_count == 0,
    }


def _window_impact_comparison(
    reference_states: list[dict[str, Any]],
    candidate_prefix: list[dict[str, Any]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    reference = _window_first_all_threshold_event(reference_states)
    candidate = _window_first_all_threshold_event(candidate_prefix)
    mismatches: list[str] = []
    if reference["status"] != candidate["status"]:
        mismatches.append("status")
    elif reference["status"] == "observed":
        for name in (
            "accepted_step",
            "control_phase",
            "phase_step",
            "threshold_consensus",
        ):
            if reference[name] != candidate[name]:
                mismatches.append(name)
        for name in ("control_target", "displacement", "fracture_energy"):
            _, ratio = _window_tolerance_ratio(
                reference[name],
                candidate[name],
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            if ratio > 1.0:
                mismatches.append(name)
    return {
        "reference": reference,
        "candidate_prefix": candidate,
        "mismatched_fields": mismatches,
        "passed": not mismatches,
    }


def _window_endpoint_delta(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    response_delta = {
        name: (
            candidate["responses"][name] - reference["responses"][name]
            if candidate["responses"][name] is not None
            and reference["responses"][name] is not None
            else None
        )
        for name in WINDOW_EXTENSION_RESPONSE_FIELDS
    }
    threshold_delta = []
    for left, right in zip(
        reference["thresholds"], candidate["thresholds"], strict=True
    ):
        threshold_delta.append(
            {
                "threshold": left["threshold"],
                "reference_classification": left["classification"],
                "candidate_classification": right["classification"],
                "reached_interface_changed": left["reached_interface"]
                != right["reached_interface"],
                "main_component_nodes_increment": right["main_component_nodes"]
                - left["main_component_nodes"],
                **{
                    f"{name}_increment": (
                        right[name] - left[name]
                        if right[name] is not None and left[name] is not None
                        else None
                    )
                    for name in WINDOW_EXTENSION_INTERFACE_FLOAT_FIELDS
                },
            }
        )
    return {
        "control_target_increment": candidate["scheduler"]["control_target"]
        - reference["scheduler"]["control_target"],
        "responses": response_delta,
        "threshold_results": threshold_delta,
    }


def window_extension_report(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Authenticate a complete path as the exact prefix of a longer path."""
    if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("window-extension tolerances must be positive")
    reference = Path(reference_directory).resolve()
    candidate = Path(candidate_directory).resolve()
    if reference == candidate:
        raise ValueError("window-extension roles must reference distinct directories")

    reference_audit = _validate_case(
        reference,
        allow_underresolved=False,
        allow_subdivisions=False,
    )
    candidate_audit = _validate_case(
        candidate,
        allow_underresolved=False,
        allow_subdivisions=False,
    )
    reference_config = _read_json(reference / "config.resolved.json")
    candidate_config = _read_json(candidate / "config.resolved.json")
    reference_path_config = _mapping(
        reference_config.get("path_control"),
        f"{reference}: path_control",
    )
    candidate_path_config = _mapping(
        candidate_config.get("path_control"),
        f"{candidate}: path_control",
    )
    reference_predictor, reference_predictor_disable_after = (
        _energy_predictor_policy(
            reference_path_config,
            f"{reference}: path_control",
        )
    )
    candidate_predictor, candidate_predictor_disable_after = (
        _energy_predictor_policy(
            candidate_path_config,
            f"{candidate}: path_control",
        )
    )
    if _canonical_window_extension_config(
        reference_config
    ) != _canonical_window_extension_config(candidate_config):
        raise ValueError(
            "window-extension configurations differ beyond source, output, and "
            "path_control.steps or a common-prefix-equivalent energy-predictor latch"
        )

    reference_steps = _integer(
        reference_path_config.get("steps"),
        f"{reference}: path_control.steps",
    )
    candidate_steps = _integer(
        candidate_path_config.get("steps"),
        f"{candidate}: path_control.steps",
    )
    if candidate_steps <= reference_steps:
        raise ValueError(
            "window-extension candidate must configure more path targets than reference"
        )
    target_increment = _finite(
        reference_path_config.get("target_increment"),
        f"{reference}: path_control.target_increment",
    )
    reference_predictor_policy = _energy_predictor_window_summary(
        use_energy_predictor=reference_predictor,
        disable_after=reference_predictor_disable_after,
        configured_steps=reference_steps,
        target_increment=target_increment,
    )
    candidate_predictor_policy = _energy_predictor_window_summary(
        use_energy_predictor=candidate_predictor,
        disable_after=candidate_predictor_disable_after,
        configured_steps=candidate_steps,
        target_increment=target_increment,
    )
    reference_prefix_last_enabled = reference_predictor_policy[
        "effective_last_enabled_nominal_path_step"
    ]
    candidate_prefix_last_enabled = (
        _effective_energy_predictor_last_enabled_step(
            candidate_predictor,
            candidate_predictor_disable_after,
            reference_steps,
        )
    )
    if candidate_prefix_last_enabled != reference_prefix_last_enabled:
        raise ValueError(
            "window-extension energy-predictor policies differ within the "
            "authenticated common prefix"
        )

    reference_runtime = _read_json(reference / "runtime.json")
    candidate_runtime = _read_json(candidate / "runtime.json")
    runtime_identity: dict[str, Any] = {}
    for name in ("mpi_ranks", "implementation_fingerprint", "runtime_fingerprint"):
        reference_value = (
            _integer(reference_runtime.get(name), f"{reference}: runtime.{name}")
            if name == "mpi_ranks"
            else _string(reference_runtime.get(name), f"{reference}: runtime.{name}")
        )
        candidate_value = (
            _integer(candidate_runtime.get(name), f"{candidate}: runtime.{name}")
            if name == "mpi_ranks"
            else _string(candidate_runtime.get(name), f"{candidate}: runtime.{name}")
        )
        if reference_value != candidate_value:
            raise ValueError(f"window-extension {name} differs")
        runtime_identity[name] = reference_value
    if reference_audit["mesh"] != candidate_audit["mesh"]:
        raise ValueError("window-extension authenticated meshes differ")
    for role, audit, expected_steps in (
        ("reference", reference_audit, reference_steps),
        ("candidate", candidate_audit, candidate_steps),
    ):
        path_control = audit["path_control"]
        if (
            path_control["configured_targets"] != expected_steps
            or path_control["accepted_targets"] != expected_steps
            or not path_control["control_targets_exhausted"]
            or audit["loading"]["maximum_subdivision_level"] != 0
            or path_control.get("unaccepted_attempts", 0) != 0
        ):
            raise ValueError(
                f"window-extension {role} is not an exact zero-attempt, "
                "zero-subdivision completed path"
            )

    reference_data = _window_extension_case_data(reference)
    candidate_data = _window_extension_case_data(candidate)
    reference_displacement = reference_data["displacement"]
    candidate_displacement = candidate_data["displacement"]
    reference_path = reference_data["path"]
    candidate_path = candidate_data["path"]
    if len(reference_displacement) != len(candidate_displacement):
        raise ValueError("window-extension displacement-phase state counts differ")
    if len(reference_path) != reference_steps or len(candidate_path) != candidate_steps:
        raise ValueError("window-extension authenticated path counts disagree with config")
    if any(
        state["phase_step"] != index
        for index, state in enumerate(reference_path, start=1)
    ) or any(
        state["phase_step"] != index
        for index, state in enumerate(candidate_path, start=1)
    ):
        raise ValueError("window-extension energy phase steps are not contiguous")

    pairs = [
        *zip(reference_displacement, candidate_displacement, strict=True),
        *zip(reference_path, candidate_path[:reference_steps], strict=True),
    ]
    for left, right in pairs:
        _authenticate_window_scheduler_pair(left, right)
    extension_states = candidate_path[reference_steps:]
    if len(extension_states) != candidate_steps - reference_steps:
        raise ValueError("window-extension candidate extension length is inconsistent")

    response_comparison = _window_response_comparison(
        pairs,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    interface_comparison = _window_interface_comparison(
        pairs,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    impact_comparison = _window_impact_comparison(
        reference_data["states"],
        [
            *candidate_displacement,
            *candidate_path[:reference_steps],
        ],
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    all_passed = (
        response_comparison["passed"]
        and interface_comparison["passed"]
        and impact_comparison["passed"]
    )
    reference_endpoint = reference_path[-1]
    candidate_endpoint = candidate_path[-1]
    return {
        "schema_version": 1,
        "study": "fracture_energy_same_increment_window_extension",
        "configuration_authentication": {
            "passed": True,
            "allowed_differences": [
                "source_path",
                "output",
                "path_control.steps",
                f"path_control.{ENERGY_PREDICTOR_LATCH_FIELD}",
            ],
            "reference_steps": reference_steps,
            "candidate_steps": candidate_steps,
            "extension_steps": candidate_steps - reference_steps,
            "target_increment": target_increment,
            "use_energy_predictor": reference_predictor,
            "energy_predictor_policy": {
                "reference": reference_predictor_policy,
                "candidate": candidate_predictor_policy,
                "common_prefix_steps": reference_steps,
                "reference_effective_last_enabled_nominal_path_step_on_common_prefix": (
                    reference_prefix_last_enabled
                ),
                "candidate_effective_last_enabled_nominal_path_step_on_common_prefix": (
                    candidate_prefix_last_enabled
                ),
                "common_prefix_sequence_equivalent": True,
            },
            "runtime_identity": runtime_identity,
            "mesh": reference_audit["mesh"],
            "solver": reference_config.get("solver"),
        },
        "completion_authentication": {
            "reference": {
                "configured_targets": reference_steps,
                "accepted_targets": reference_steps,
                "pending_control_targets": 0,
                "unaccepted_attempts": 0,
                "maximum_subdivision_level": 0,
            },
            "candidate": {
                "configured_targets": candidate_steps,
                "accepted_targets": candidate_steps,
                "pending_control_targets": 0,
                "unaccepted_attempts": 0,
                "maximum_subdivision_level": 0,
            },
            "passed": True,
        },
        "prefix_pairing": {
            "rule": (
                "all displacement states pair in order; reference energy phase_step k "
                "pairs with candidate energy phase_step k at the same control target"
            ),
            "displacement_states": len(reference_displacement),
            "energy_states": reference_steps,
            "compared_states": len(pairs),
            "first_energy_control_target": reference_path[0]["scheduler"][
                "control_target"
            ],
            "last_energy_control_target": reference_endpoint["scheduler"][
                "control_target"
            ],
            "scheduler_authenticated": True,
        },
        "prefix_response_comparison": response_comparison,
        "prefix_interface_comparison": interface_comparison,
        "prefix_first_all_threshold_event": impact_comparison,
        "extension": {
            "states": len(extension_states),
            "first_phase_step": extension_states[0]["phase_step"],
            "last_phase_step": extension_states[-1]["phase_step"],
            "reference_endpoint": _window_state_marker(reference_endpoint),
            "candidate_endpoint": _window_state_marker(candidate_endpoint),
            "increment_from_reference_endpoint": _window_endpoint_delta(
                reference_endpoint,
                candidate_endpoint,
            ),
            "candidate_first_all_threshold_event": (
                _window_first_all_threshold_event(candidate_data["states"])
            ),
            "first_directional_measure_in_extension": (
                _window_first_directional_measure(
                    extension_states,
                    absolute_tolerance=absolute_tolerance,
                )
            ),
            "first_direction_candidate_in_extension": _window_first_candidate(
                extension_states
            ),
            "candidate_persistence": _window_candidate_persistence(
                candidate_data["states"]
            ),
        },
        "gate_summary": {
            "configuration_authenticated": True,
            "completion_authenticated": True,
            "scheduler_authenticated": True,
            "prefix_response_passed": response_comparison["passed"],
            "prefix_interface_passed": interface_comparison["passed"],
            "prefix_first_all_threshold_event_passed": impact_comparison[
                "passed"
            ],
        },
        "all_checks_passed": all_passed,
        "screen_only": False,
        "interpretation": (
            "Passing proves that the shorter completed path is reproduced over its full "
            "displacement and fracture-energy prefix before the candidate adds a longer "
            "same-increment observation window. Candidate classifications remain mesh- and "
            "threshold-dependent. 'arrested_or_unresolved' is not a confirmed arrest."
        ),
    }


def mirror_report(
    first_directory: str | Path,
    second_directory: str | Path,
    *,
    allow_underresolved: bool = False,
    allow_subdivisions: bool = False,
    relative_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Strictly compare a y-reflected angle/diagonal/pin result pair."""
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    cases = [
        _validate_case(
            directory,
            allow_underresolved=allow_underresolved,
            allow_subdivisions=allow_subdivisions,
        )
        for directory in (first_directory, second_directory)
    ]
    first, second = cases
    if _canonical_mirror_config(first["_config"]) != _canonical_mirror_config(second["_config"]):
        raise ValueError("mirror configurations differ beyond diagonal, pin, nodes and output")
    first_geometry = _mapping(first["_config"]["geometry"], "first.geometry")
    second_geometry = _mapping(second["_config"]["geometry"], "second.geometry")
    if {first_geometry["diagonal"], second_geometry["diagonal"]} != {"left", "right"}:
        raise ValueError("mirror pair must use opposite left/right diagonals")
    if {first_geometry["x_pin_corner"], second_geometry["x_pin_corner"]} != {
        "bottom_left",
        "top_left",
    }:
        raise ValueError("mirror pair must use opposite bottom/top horizontal pins")
    height = _finite(first_geometry["height"], "first.geometry.height")
    first_protocol = first["interface"]["protocol"]
    second_protocol = second["interface"]["protocol"]
    role_map = {
        first_protocol["start_node"]: second_protocol["end_node"],
        first_protocol["impact_node"]: second_protocol["impact_node"],
        first_protocol["end_node"]: second_protocol["start_node"],
    }
    first_nodes = {
        item["name"]: item["point"] for item in _list(first["_config"]["graph_nodes"], "nodes")
    }
    second_nodes = {
        item["name"]: item["point"] for item in _list(second["_config"]["graph_nodes"], "nodes")
    }
    for first_name, second_name in role_map.items():
        x_coordinate, y_coordinate = first_nodes[first_name]
        expected = (float(x_coordinate), height - float(y_coordinate))
        observed = tuple(float(value) for value in second_nodes[second_name])
        if not all(_close(a, b) for a, b in zip(expected, observed, strict=True)):
            raise ValueError("configured interface nodes are not exact y-mirrors")

    first_history = first["_history"]
    second_history = second["_history"]
    if len(first_history) != len(second_history):
        raise ValueError("mirror histories have different accepted load counts")
    maximum_relative_error = 0.0
    for index, (left, right) in enumerate(zip(first_history, second_history, strict=True)):
        if not _close(left["displacement"], right["displacement"]):
            raise ValueError(f"mirror histories differ at displacement index {index}")
        for name in (
            "reaction_y",
            "elastic_energy",
            "fracture_energy",
            "total_internal_energy",
            "regularised_crack_length",
        ):
            scale = max(abs(left[name]), abs(right[name]), 1.0e-30)
            relative_error = abs(left[name] - right[name]) / scale
            maximum_relative_error = max(maximum_relative_error, relative_error)
            if relative_error > relative_tolerance:
                raise ValueError(f"mirror histories differ in {name} at accepted step {index}")

    for index, (left, right) in enumerate(
        zip(first["_interface_records"], second["_interface_records"], strict=True)
    ):
        if left["threshold_consensus"] != right["threshold_consensus"]:
            raise ValueError(f"mirror interface consensus differs at accepted step {index}")
        for threshold_index, (left_threshold, right_threshold) in enumerate(
            zip(left["threshold_results"], right["threshold_results"], strict=True)
        ):
            if left_threshold["classification"] != right_threshold["classification"]:
                raise ValueError(
                    f"mirror classification differs at step {index}, threshold {threshold_index}"
                )
            for name in (
                "closest_main_node_to_impact",
                "interface_forward_advance",
                "penetration_forward_advance",
                "interface_active_edge_length",
                "penetration_active_edge_length",
            ):
                left_value, right_value = left_threshold[name], right_threshold[name]
                if left_value is None or right_value is None:
                    if left_value is not right_value:
                        raise ValueError(f"mirror {name} nullability differs at step {index}")
                elif not _close(left_value, right_value, rel=relative_tolerance):
                    raise ValueError(f"mirror {name} differs at step {index}")

    return {
        "study": "inclined-interface exact y-mirror control",
        "relative_tolerance": relative_tolerance,
        "maximum_relative_response_error": maximum_relative_error,
        "all_checks_passed": True,
        "cases": [_public_case(case) for case in cases],
        "interpretation": (
            "Passing establishes a discrete angle/diagonal/pin mirror control. It does not "
            "establish a physical penetration-deflection transition."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphfracture-inclined-studies",
        description="Strictly validate same-material inclined weak-interface result directories.",
    )
    subparsers = parser.add_subparsers(dest="study", required=True)
    validation = subparsers.add_parser("validate", help="strict single-case audit")
    validation.add_argument("case", type=Path)
    progress = subparsers.add_parser(
        "progress", help="strict inspection of accepted states from a partial or complete case"
    )
    progress.add_argument("case", type=Path)
    continuation = subparsers.add_parser(
        "path-control",
        help="compare common accepted states under different continuation controls",
    )
    continuation.add_argument("reference", type=Path)
    continuation.add_argument("candidate", type=Path)
    increment_pair = subparsers.add_parser(
        "increment-pair",
        help="audit a completed path against an exact half-increment completed path",
    )
    increment_pair.add_argument("reference", type=Path)
    increment_pair.add_argument("candidate", type=Path)
    window_extension = subparsers.add_parser(
        "window-extension",
        help="audit a completed path as the exact prefix of a longer same-increment path",
    )
    window_extension.add_argument("reference", type=Path)
    window_extension.add_argument("candidate", type=Path)
    mirror = subparsers.add_parser("mirror", help="strict y-reflected pair audit")
    mirror.add_argument("first", type=Path)
    mirror.add_argument("second", type=Path)
    for command in (validation, mirror):
        command.add_argument("--allow-underresolved", action="store_true")
        command.add_argument("--allow-subdivisions", action="store_true")
        command.add_argument("--output", "-o", type=Path)
    progress.add_argument("--output", "-o", type=Path)
    continuation.add_argument("--relative-tolerance", type=float, default=1.0e-4)
    continuation.add_argument("--output", "-o", type=Path)
    increment_pair.add_argument(
        "--allow-reference-subdivisions",
        action="store_true",
        help="allow a strictly authenticated adaptive-complete reference only",
    )
    increment_pair.add_argument("--relative-tolerance", type=float, default=1.0e-2)
    increment_pair.add_argument("--output", "-o", type=Path)
    window_extension.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    window_extension.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    window_extension.add_argument("--output", "-o", type=Path)
    mirror.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.study == "validate":
            payload = case_audit_report(
                args.case,
                allow_underresolved=args.allow_underresolved,
                allow_subdivisions=args.allow_subdivisions,
            )
        elif args.study == "progress":
            payload = case_progress_report(args.case)
        elif args.study == "path-control":
            payload = continuation_path_report(
                args.reference,
                args.candidate,
                relative_tolerance=args.relative_tolerance,
            )
        elif args.study == "increment-pair":
            payload = increment_pair_report(
                args.reference,
                args.candidate,
                allow_reference_subdivisions=args.allow_reference_subdivisions,
                relative_tolerance=args.relative_tolerance,
            )
        elif args.study == "window-extension":
            payload = window_extension_report(
                args.reference,
                args.candidate,
                relative_tolerance=args.relative_tolerance,
                absolute_tolerance=args.absolute_tolerance,
            )
        else:
            payload = mirror_report(
                args.first,
                args.second,
                allow_underresolved=args.allow_underresolved,
                allow_subdivisions=args.allow_subdivisions,
                relative_tolerance=args.relative_tolerance,
            )
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        print(f"inclined-studies error: {exc}", file=sys.stderr)
        return 2
    if args.study in {"increment-pair", "window-extension"} and not payload[
        "all_checks_passed"
    ]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
