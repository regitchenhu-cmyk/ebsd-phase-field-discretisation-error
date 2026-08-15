"""Pure prospective contract for the P0-A qualification study.

This module freezes scientific choices before any qualification solve.  It is
deliberately independent of DOLFINx, PETSc, MPI and subprocess execution.  A
future launch preregistration must bind the hash of the science-protocol
artifact produced from this contract and separately authenticate its complete
toolchain, preflight evidence, argv, environment and output namespace.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from graphfracture.p0a_protocol import (
    FORMAL_FIXED_TABLE_POLICY,
    PHASE_B_FIREWALL,
    PHYSICAL_MODEL,
    QUALIFICATION_SOURCE_POLICY,
    QUALIFICATION_STAGES,
    RIGHT_CENSORED_LINEAGE_IDENTITIES,
    SOURCE_TEMPLATE_FILENAMES,
    SOURCE_TEMPLATE_IDENTITIES,
    SOURCE_TEMPLATE_OUTPUT_DIRECTORIES,
    prospective_design,
)

SCHEMA_VERSION = 1
PROTOCOL_REVISION = 1
PROTOCOL_NAME = "p0a_homogeneous_sent_qualification_science_protocol_v1"
EVIDENCE_CLASS = "p0a_homogeneous_sent_qualification_science_protocol_not_execution_authorization"

COARSE_INCREMENT = 2.0e-6
HALF_INCREMENT = 1.0e-6
SMALL_STRAIN_LIMIT = 0.01
CANDIDATE_END_STEPS = tuple(range(80, 241, 20))
CANDIDATE_END_DISPLACEMENTS = tuple(step * COARSE_INCREMENT for step in CANDIDATE_END_STEPS)
MINIMUM_POSTPEAK_NOMINAL_NODES = 5
MAXIMUM_ENDPOINT_TO_PEAK_REACTION_RATIO = 0.95

BINARY64_EPSILON = 2.0**-52
REPEATABILITY_MARGIN_FACTOR = 4.0
REPEATABILITY_CAPS = {
    "primary_global_curve_energy_displacement": 1.0e-4,
    "damage_field_and_crack_summary": 1.0e-3,
}
CROSS_MESH_BAND = 0.02
CROSS_MESH_CONTRACTION = 0.75
CROSS_MESH_FLOOR_CAP = 0.01

WINDOW_FORMULA_ID = "p0a_signed_reaction_postpeak_window_v1"
SINGLE_RUN_FORMULA_ID = "p0a_single_run_science_gates_v1"
Q05_FORMULA_ID = "p0a_q05_coarse_vs_half_v1"
REPEATABILITY_FORMULA_ID = "p0a_mixed_repeatability_literal_v1"
CROSS_MESH_FLOOR_FORMULA_ID = "p0a_cross_mesh_numerical_floor_v1"
CROSS_MESH_GATE_FORMULA_ID = "p0a_cross_mesh_band_contraction_floor_v1"

Q05_LIMITS = {
    "peak_reaction_relative": 0.005,
    "peak_displacement_in_coarse_steps": 0.5,
    "endpoint_fracture_dissipation_relative": 0.005,
    "common_node_reaction_and_energy_normalized_linf": 0.01,
    "endpoint_displacement_normalized_weighted_l2": 0.01,
    "endpoint_damage_normalized_weighted_l2": 0.01,
}

SINGLE_RUN_LIMITS = {
    "damage_kkt_relative": 1.0e-6,
    "damage_bound_violation": 1.0e-10,
    "irreversibility_violation": 1.0e-10,
    "stagger_error": 1.0e-5,
    "stagger_iterations": 320,
    "normalized_energy_closure": 1.0e-4,
    "normalized_reaction_balance": 1.0e-4,
    "mirror_displacement_normalized_weighted_l2": 1.0e-3,
    "mirror_damage_normalized_weighted_l2": 1.0e-3,
}

QUALIFICATION_EXECUTION_BLOCKERS = (
    "qualification_launch_preregistration_not_frozen",
    "qualification_config_and_load_table_hashes_not_frozen",
    "qualification_stage_runner_not_frozen",
    "qualification_qoi_extractor_not_frozen",
    "qualification_source_whitelist_validator_not_frozen",
    "phase_b_pollution_negative_tests_not_passed",
    "qualification_receipt_and_ledger_schemas_not_frozen",
    "qualification_generic_runner_isolation_not_verified",
    "qualification_zero_solve_preflight_not_frozen",
    "qualification_zero_solve_preflight_not_passed",
    "qualification_launcher_and_terminalizer_not_frozen",
    "qualification_argv_environment_and_runtime_not_frozen",
    "qualification_output_namespace_not_preflighted",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float), f"{label} must be a finite JSON number")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be a finite JSON number")
    return result


def _require_strict_json_value(value: Any, label: str) -> None:
    if type(value) in (type(None), bool, int, str):
        return
    if type(value) is float:
        _require(math.isfinite(value), f"{label} contains a nonfinite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_strict_json_value(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            _require(type(key) is str, f"{label} contains a non-string key")
            _require_strict_json_value(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} contains a non-JSON type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the compact canonical bytes used for nested contract digests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_value(value: Any) -> Any:
    """Return a detached JSON value with tuples normalized to arrays."""

    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def round_up_125(value: float) -> float:
    """Round a non-negative finite value upward to {1, 2, 5} x 10**k."""

    number = _finite(value, "round125 input")
    _require(number >= 0.0, "round125 input must be non-negative")
    if number == 0.0:
        return 0.0
    decimal_value = Decimal(str(number))
    exponent = decimal_value.adjusted()
    unit = Decimal(10) ** exponent
    for multiplier in (Decimal(1), Decimal(2), Decimal(5), Decimal(10)):
        candidate = multiplier * unit
        if decimal_value <= candidate:
            result = float(candidate)
            _require(
                math.isfinite(result) and result >= number,
                "round125 result is not a finite upward binary64 value",
            )
            return result
    raise AssertionError("unreachable round125 branch")


def physical_scales(selected_endpoint: float | None = None) -> dict[str, float | str]:
    """Return the preregistered unit-thickness scales.

    The displacement scale is result-bound because Q01 selects one member of
    the already frozen candidate set.  No new scale may be constructed after
    seeing qualification results.
    """

    model = PHYSICAL_MODEL
    plane_strain_modulus = model.young_modulus / (1.0 - model.poisson_ratio**2)
    at2_first_peak_stress = math.sqrt(
        27.0 * plane_strain_modulus * model.fracture_toughness / (256.0 * model.length_scale)
    )
    result: dict[str, float | str] = {
        "plane_strain_modulus": plane_strain_modulus,
        "at2_first_local_peak_stress": at2_first_peak_stress,
        "reaction_force_unit_thickness": at2_first_peak_stress * model.length,
        "fracture_energy": model.fracture_toughness * (model.length - model.precrack_length),
        "damage": 1.0,
        "crack_length": model.length - model.precrack_length,
        "interpretation": (
            "fixed normalization scales; the homogeneous AT2 stress is not a SENT peak prediction"
        ),
    }
    if selected_endpoint is None:
        result["displacement"] = "selected_candidate_endpoint"
    else:
        endpoint = _finite(selected_endpoint, "selected endpoint")
        _require(
            endpoint in CANDIDATE_END_DISPLACEMENTS,
            "selected endpoint is not a frozen candidate",
        )
        result["displacement"] = endpoint
    return result


def candidate_scale_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step, displacement in zip(CANDIDATE_END_STEPS, CANDIDATE_END_DISPLACEMENTS, strict=True):
        records.append(
            {
                "candidate_id": f"n{step:03d}",
                "end_step": step,
                "end_displacement": displacement,
                "end_nominal_strain": displacement / PHYSICAL_MODEL.height,
                "scales": physical_scales(displacement),
            }
        )
    return records


def _authenticate_window_history(
    records: Sequence[Mapping[str, Any]], *, maximum_step: int
) -> list[dict[str, int | float]]:
    _require(type(records) is list, "window history must be a JSON array")
    _require_strict_json_value(records, "window history")
    _require(type(maximum_step) is int and maximum_step >= 1, "maximum step is invalid")
    expected_keys = {"scheduled_step", "subdivision_level", "displacement", "reaction_y"}
    authenticated: list[dict[str, int | float]] = []
    previous_displacement = -math.inf
    for index, raw in enumerate(records):
        _require(type(raw) is dict, f"history record {index} must be a JSON object")
        _require(set(raw) == expected_keys, f"history record {index} key set mismatch")
        scheduled_step = raw["scheduled_step"]
        subdivision = raw["subdivision_level"]
        _require(type(scheduled_step) is int, f"history record {index} step must be an integer")
        _require(type(subdivision) is int, f"history record {index} subdivision must be an integer")
        _require(
            0 <= scheduled_step <= maximum_step,
            f"history record {index} step is outside the frozen table",
        )
        _require(subdivision >= 0, f"history record {index} subdivision is negative")
        displacement = _finite(raw["displacement"], f"history record {index} displacement")
        reaction = _finite(raw["reaction_y"], f"history record {index} reaction")
        _require(
            displacement > previous_displacement,
            f"history record {index} displacement is not strictly increasing",
        )
        previous_displacement = displacement
        authenticated.append(
            {
                "scheduled_step": scheduled_step,
                "subdivision_level": subdivision,
                "displacement": displacement,
                "reaction_y": reaction,
            }
        )
    _require(bool(authenticated), "history is empty")
    return authenticated


def evaluate_candidate_window(
    records: Sequence[Mapping[str, Any]], *, end_step: int, increment: float
) -> dict[str, Any]:
    """Authenticate and evaluate one fixed endpoint without selecting another."""

    _require(type(end_step) is int and end_step >= 1, "candidate end step is invalid")
    increment_value = _finite(increment, "candidate increment")
    _require(increment_value > 0.0, "candidate increment must be positive")
    authenticated = _authenticate_window_history(records, maximum_step=end_step)
    by_step: dict[int, dict[str, int | float]] = {}
    for record in authenticated:
        _require(
            record["subdivision_level"] == 0,
            "fixed window contains an adaptive subdivided state",
        )
        step = int(record["scheduled_step"])
        _require(step not in by_step, "duplicate nominal scheduled step")
        by_step[step] = record
    _require(
        set(by_step) == set(range(end_step + 1)),
        "fixed window must contain every nominal step through its endpoint",
    )
    for step, record in by_step.items():
        expected = step * increment_value
        observed = float(record["displacement"])
        _require(
            observed.hex() == float(expected).hex(),
            f"nominal displacement mismatch at step {step}",
        )
    reactions = [float(by_step[index]["reaction_y"]) for index in range(end_step + 1)]
    peak_reaction = max(reactions)
    peak_step = reactions.index(peak_reaction)
    endpoint_reaction = reactions[-1]
    postpeak_nodes = end_step - peak_step
    ratio = endpoint_reaction / peak_reaction if peak_reaction > 0.0 else None
    gates = {
        "signed_peak_positive": peak_reaction > 0.0,
        "minimum_postpeak_nominal_nodes": postpeak_nodes >= MINIMUM_POSTPEAK_NOMINAL_NODES,
        "endpoint_drop": (
            peak_reaction > 0.0
            and endpoint_reaction <= MAXIMUM_ENDPOINT_TO_PEAK_REACTION_RATIO * peak_reaction
        ),
    }
    return {
        "formula_id": WINDOW_FORMULA_ID,
        "end_step": end_step,
        "end_displacement": end_step * increment_value,
        "increment": increment_value,
        "peak_step": peak_step,
        "peak_displacement": peak_step * increment_value,
        "peak_reaction": peak_reaction,
        "endpoint_reaction": endpoint_reaction,
        "postpeak_nominal_nodes": postpeak_nodes,
        "endpoint_to_peak_reaction_ratio": ratio,
        "gates": gates,
        "passed": all(gates.values()),
    }


def select_window(nominal_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the earliest frozen post-peak candidate from a complete Q01 trace."""

    maximum_step = CANDIDATE_END_STEPS[-1]
    authenticated = _authenticate_window_history(nominal_records, maximum_step=maximum_step)
    by_step: dict[int, dict[str, int | float]] = {}
    for record in authenticated:
        if record["subdivision_level"] != 0:
            continue
        step = int(record["scheduled_step"])
        _require(step not in by_step, "duplicate nominal scheduled step")
        by_step[step] = record
    _require(
        set(by_step) == set(range(maximum_step + 1)),
        "Q01 nominal trace must contain every step through the small-strain ceiling",
    )
    for step, record in by_step.items():
        expected = step * COARSE_INCREMENT
        observed = float(record["displacement"])
        _require(
            observed.hex() == float(expected).hex(),
            f"Q01 nominal displacement mismatch at step {step}",
        )
    nominal = [by_step[step] for step in range(maximum_step + 1)]
    for candidate in candidate_scale_records():
        end_step = int(candidate["end_step"])
        prefix = nominal[: end_step + 1]
        evaluation = evaluate_candidate_window(
            prefix, end_step=end_step, increment=COARSE_INCREMENT
        )
        if evaluation["passed"]:
            return {
                **candidate,
                **evaluation,
                "selection_rule": "earliest_frozen_candidate",
            }
    raise ValueError("Q01 did not bracket a frozen post-peak candidate")


def _finite_vector(values: Any, label: str) -> list[float]:
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
        f"{label} must be a sequence",
    )
    return [_finite(value, f"{label}[{index}]") for index, value in enumerate(values)]


def _weighted_relative_l2(
    left: Sequence[float],
    right: Sequence[float],
    weights: Sequence[float],
    *,
    fixed_scale: float,
) -> dict[str, float]:
    _require(
        len(left) == len(right) == len(weights) and len(left) > 0,
        "weighted-L2 vectors must have the same positive length",
    )
    weight_sum = sum(weights)
    _require(weight_sum > 0.0, "weighted-L2 weight sum must be positive")
    difference = math.sqrt(
        sum(weight * (a - b) ** 2 for a, b, weight in zip(left, right, weights, strict=True))
        / weight_sum
    )
    left_rms = math.sqrt(
        sum(weight * a**2 for a, weight in zip(left, weights, strict=True)) / weight_sum
    )
    right_rms = math.sqrt(
        sum(weight * b**2 for b, weight in zip(right, weights, strict=True)) / weight_sum
    )
    denominator = max(fixed_scale, left_rms, right_rms)
    return {
        "absolute_weighted_l2": difference,
        "normalization": denominator,
        "normalized_weighted_l2": difference / denominator,
    }


def evaluate_q05_load_discretization(
    coarse: Mapping[str, Any], half: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the fully frozen Q04/Q05 coarse-versus-half-step gate."""

    _require(type(coarse) is dict and type(half) is dict, "Q05 inputs must be JSON objects")
    _require_strict_json_value(coarse, "Q04 comparison record")
    _require_strict_json_value(half, "Q05 comparison record")
    expected_run_keys = {
        "run_passed",
        "scheduled_step",
        "subdivision_level",
        "displacement",
        "reaction_y",
        "elastic_energy",
        "fracture_energy",
        "external_work",
        "endpoint_samples",
    }
    expected_sample_keys = {"sample_ids", "weights", "displacement", "damage"}
    _require(set(coarse) == expected_run_keys, "Q04 comparison record key set mismatch")
    _require(set(half) == expected_run_keys, "Q05 comparison record key set mismatch")
    _require(type(coarse["run_passed"]) is bool, "Q04 run_passed must be Boolean")
    _require(type(half["run_passed"]) is bool, "Q05 run_passed must be Boolean")

    coarse_steps = list(coarse["scheduled_step"])
    half_steps = list(half["scheduled_step"])
    _require(
        all(type(value) is int for value in coarse_steps),
        "Q04 scheduled steps must be integers",
    )
    _require(
        all(type(value) is int for value in half_steps),
        "Q05 scheduled steps must be integers",
    )
    _require(coarse_steps == list(range(len(coarse_steps))), "Q04 scheduled steps differ")
    _require(half_steps == list(range(len(half_steps))), "Q05 scheduled steps differ")
    _require(
        len(half_steps) == 2 * (len(coarse_steps) - 1) + 1,
        "Q05 half-step table length does not refine Q04 by two",
    )

    coarse_subdivision = list(coarse["subdivision_level"])
    half_subdivision = list(half["subdivision_level"])
    _require(
        len(coarse_subdivision) == len(coarse_steps)
        and all(type(value) is int and value == 0 for value in coarse_subdivision),
        "Q04 contains a subdivided state",
    )
    _require(
        len(half_subdivision) == len(half_steps)
        and all(type(value) is int and value == 0 for value in half_subdivision),
        "Q05 contains a subdivided state",
    )

    vector_names = (
        "displacement",
        "reaction_y",
        "elastic_energy",
        "fracture_energy",
        "external_work",
    )
    coarse_vectors = {name: _finite_vector(coarse[name], f"Q04 {name}") for name in vector_names}
    half_vectors = {name: _finite_vector(half[name], f"Q05 {name}") for name in vector_names}
    _require(
        all(len(values) == len(coarse_steps) for values in coarse_vectors.values()),
        "Q04 vector length mismatch",
    )
    _require(
        all(len(values) == len(half_steps) for values in half_vectors.values()),
        "Q05 vector length mismatch",
    )
    for index, displacement in enumerate(coarse_vectors["displacement"]):
        expected = index * COARSE_INCREMENT
        _require(float(displacement).hex() == float(expected).hex(), "Q04 load table differs")
    for index, displacement in enumerate(half_vectors["displacement"]):
        expected = index * HALF_INCREMENT
        _require(float(displacement).hex() == float(expected).hex(), "Q05 load table differs")
    _require(
        all(
            coarse_vectors["displacement"][index].hex()
            == half_vectors["displacement"][2 * index].hex()
            for index in range(len(coarse_steps))
        ),
        "Q05 even nodes do not equal Q04 nodes in binary64",
    )

    endpoint = coarse_vectors["displacement"][-1]
    _require(
        endpoint.hex() == half_vectors["displacement"][-1].hex(),
        "Q04/Q05 endpoints differ",
    )
    scales = physical_scales(endpoint)
    reaction_scale = float(scales["reaction_force_unit_thickness"])
    energy_scale = float(scales["fracture_energy"])
    displacement_scale = float(scales["displacement"])

    coarse_peak = max(coarse_vectors["reaction_y"])
    half_peak = max(half_vectors["reaction_y"])
    coarse_peak_index = coarse_vectors["reaction_y"].index(coarse_peak)
    half_peak_index = half_vectors["reaction_y"].index(half_peak)
    peak_reaction_relative = abs(coarse_peak - half_peak) / max(
        reaction_scale, abs(coarse_peak), abs(half_peak)
    )
    peak_displacement_in_coarse_steps = (
        abs(
            coarse_vectors["displacement"][coarse_peak_index]
            - half_vectors["displacement"][half_peak_index]
        )
        / COARSE_INCREMENT
    )

    coarse_dissipation = (
        coarse_vectors["fracture_energy"][-1] - coarse_vectors["fracture_energy"][0]
    )
    half_dissipation = half_vectors["fracture_energy"][-1] - half_vectors["fracture_energy"][0]
    fracture_dissipation_relative = abs(coarse_dissipation - half_dissipation) / max(
        energy_scale, abs(coarse_dissipation), abs(half_dissipation)
    )

    common_node_normalized_linf = 0.0
    common_node_components: dict[str, float] = {}
    for name in ("reaction_y", "elastic_energy", "fracture_energy", "external_work"):
        fixed_scale = reaction_scale if name == "reaction_y" else energy_scale
        component = max(
            abs(coarse_value - half_vectors[name][2 * index])
            / max(fixed_scale, abs(coarse_value), abs(half_vectors[name][2 * index]))
            for index, coarse_value in enumerate(coarse_vectors[name])
        )
        common_node_components[name] = component
        common_node_normalized_linf = max(common_node_normalized_linf, component)

    coarse_samples = coarse["endpoint_samples"]
    half_samples = half["endpoint_samples"]
    _require(isinstance(coarse_samples, Mapping), "Q04 endpoint samples must be an object")
    _require(isinstance(half_samples, Mapping), "Q05 endpoint samples must be an object")
    _require(set(coarse_samples) == expected_sample_keys, "Q04 sample key set mismatch")
    _require(set(half_samples) == expected_sample_keys, "Q05 sample key set mismatch")
    _require(
        canonical_json_bytes(list(coarse_samples["sample_ids"]))
        == canonical_json_bytes(list(half_samples["sample_ids"])),
        "Q04/Q05 physical sample identities differ",
    )
    sample_ids = list(coarse_samples["sample_ids"])
    _require(len(sample_ids) > 0, "endpoint physical sample set is empty")
    _require(
        len({canonical_json_bytes(value) for value in sample_ids}) == len(sample_ids),
        "endpoint physical sample identities are not unique",
    )
    _require(
        canonical_json_bytes(list(coarse_samples["weights"]))
        == canonical_json_bytes(list(half_samples["weights"])),
        "Q04/Q05 physical sample weights differ",
    )
    weights = _finite_vector(coarse_samples["weights"], "endpoint sample weights")
    _require(len(weights) == len(sample_ids), "endpoint sample weight count differs")
    _require(all(weight > 0.0 for weight in weights), "endpoint sample weights must be positive")
    coarse_u = _finite_vector(coarse_samples["displacement"], "Q04 endpoint displacement")
    half_u = _finite_vector(half_samples["displacement"], "Q05 endpoint displacement")
    coarse_d = _finite_vector(coarse_samples["damage"], "Q04 endpoint damage")
    half_d = _finite_vector(half_samples["damage"], "Q05 endpoint damage")
    _require(
        len(coarse_u) == len(half_u) == len(coarse_d) == len(half_d) == len(weights),
        "endpoint field sample count differs",
    )
    displacement_l2 = _weighted_relative_l2(
        coarse_u, half_u, weights, fixed_scale=displacement_scale
    )
    damage_l2 = _weighted_relative_l2(coarse_d, half_d, weights, fixed_scale=1.0)

    coarse_window = evaluate_candidate_window(
        [
            {
                "scheduled_step": index,
                "subdivision_level": 0,
                "displacement": displacement,
                "reaction_y": coarse_vectors["reaction_y"][index],
            }
            for index, displacement in enumerate(coarse_vectors["displacement"])
        ],
        end_step=len(coarse_steps) - 1,
        increment=COARSE_INCREMENT,
    )
    half_window = evaluate_candidate_window(
        [
            {
                "scheduled_step": index,
                "subdivision_level": 0,
                "displacement": displacement,
                "reaction_y": half_vectors["reaction_y"][index],
            }
            for index, displacement in enumerate(half_vectors["displacement"])
        ],
        end_step=len(half_steps) - 1,
        increment=HALF_INCREMENT,
    )
    half_common_postpeak_passed = (
        int(half_window["postpeak_nominal_nodes"]) >= 2 * MINIMUM_POSTPEAK_NOMINAL_NODES
    )

    gates = {
        "both_single_run_gates_passed": coarse["run_passed"] and half["run_passed"],
        "coarse_window_passed": bool(coarse_window["passed"]),
        "half_window_passed": bool(half_window["passed"]) and half_common_postpeak_passed,
        "peak_reaction_passed": peak_reaction_relative <= Q05_LIMITS["peak_reaction_relative"],
        "peak_displacement_passed": peak_displacement_in_coarse_steps
        <= Q05_LIMITS["peak_displacement_in_coarse_steps"],
        "endpoint_fracture_dissipation_passed": fracture_dissipation_relative
        <= Q05_LIMITS["endpoint_fracture_dissipation_relative"],
        "common_node_curve_and_energy_passed": common_node_normalized_linf
        <= Q05_LIMITS["common_node_reaction_and_energy_normalized_linf"],
        "endpoint_displacement_field_passed": displacement_l2["normalized_weighted_l2"]
        <= Q05_LIMITS["endpoint_displacement_normalized_weighted_l2"],
        "endpoint_damage_field_passed": damage_l2["normalized_weighted_l2"]
        <= Q05_LIMITS["endpoint_damage_normalized_weighted_l2"],
    }
    return {
        "formula_id": Q05_FORMULA_ID,
        "coarse_node_count": len(coarse_steps),
        "half_node_count": len(half_steps),
        "endpoint": endpoint,
        "peak_reaction_relative": peak_reaction_relative,
        "peak_displacement_in_coarse_steps": peak_displacement_in_coarse_steps,
        "endpoint_fracture_dissipation_relative": fracture_dissipation_relative,
        "common_node_component_normalized_linf": common_node_components,
        "common_node_normalized_linf": common_node_normalized_linf,
        "endpoint_displacement_weighted_l2": displacement_l2,
        "endpoint_damage_weighted_l2": damage_l2,
        "coarse_window": coarse_window,
        "half_window": half_window,
        "half_common_postpeak_passed": half_common_postpeak_passed,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_single_run_science_gates(
    record: Mapping[str, Any],
    *,
    window_records: Sequence[Mapping[str, Any]],
    end_step: int,
    increment: float,
) -> dict[str, Any]:
    """Evaluate the preregistered per-run physics and completion gates."""

    _require(type(record) is dict, "single-run science record must be a JSON object")
    _require_strict_json_value(record, "single-run science record")
    expected_keys = {
        "source_authenticated",
        "artifacts_complete",
        "history_finite_and_ordered",
        "snes_reason",
        "ksp_reasons",
        "damage_kkt_relative",
        "damage_lower_bound_violation",
        "damage_upper_bound_violation",
        "irreversibility_violation",
        "stagger_error",
        "stagger_iterations",
        "elastic_energy_start",
        "elastic_energy_end",
        "fracture_energy_start",
        "fracture_energy_end",
        "external_work",
        "reaction_top_y",
        "reaction_bottom_y",
        "mirror_displacement_weighted_l2",
        "mirror_damage_weighted_l2",
    }
    _require(set(record) == expected_keys, "single-run science record key set mismatch")
    for name in ("source_authenticated", "artifacts_complete", "history_finite_and_ordered"):
        _require(type(record[name]) is bool, f"single-run {name} must be Boolean")
    _require(type(record["snes_reason"]) is int, "single-run SNES reason must be an integer")
    ksp_reasons = list(record["ksp_reasons"])
    _require(
        bool(ksp_reasons) and all(type(value) is int for value in ksp_reasons),
        "single-run KSP reasons must be a nonempty integer array",
    )
    _require(
        type(record["stagger_iterations"]) is int,
        "single-run stagger iterations must be an integer",
    )
    numeric_names = expected_keys - {
        "source_authenticated",
        "artifacts_complete",
        "history_finite_and_ordered",
        "snes_reason",
        "ksp_reasons",
        "stagger_iterations",
    }
    values = {name: _finite(record[name], f"single-run {name}") for name in numeric_names}
    for name in (
        "damage_kkt_relative",
        "damage_lower_bound_violation",
        "damage_upper_bound_violation",
        "irreversibility_violation",
        "stagger_error",
        "mirror_displacement_weighted_l2",
        "mirror_damage_weighted_l2",
    ):
        _require(values[name] >= 0.0, f"single-run {name} must be non-negative")
    window = evaluate_candidate_window(window_records, end_step=end_step, increment=increment)
    scales = physical_scales(float(window["end_displacement"]))
    reaction_scale = float(scales["reaction_force_unit_thickness"])
    energy_scale = float(scales["fracture_energy"])
    displacement_scale = float(scales["displacement"])
    delta_elastic = values["elastic_energy_end"] - values["elastic_energy_start"]
    delta_fracture = values["fracture_energy_end"] - values["fracture_energy_start"]
    energy_residual = delta_elastic + delta_fracture - values["external_work"]
    normalized_energy_closure = abs(energy_residual) / max(
        energy_scale,
        abs(delta_elastic),
        abs(delta_fracture),
        abs(values["external_work"]),
    )
    reaction_residual = values["reaction_top_y"] + values["reaction_bottom_y"]
    normalized_reaction_balance = abs(reaction_residual) / max(
        reaction_scale,
        abs(values["reaction_top_y"]),
        abs(values["reaction_bottom_y"]),
    )
    mirror_displacement = values["mirror_displacement_weighted_l2"] / displacement_scale
    mirror_damage = values["mirror_damage_weighted_l2"]
    nonnegative_energy_guard = 1.0e-12 * energy_scale
    gates = {
        "source_authenticated": record["source_authenticated"],
        "artifacts_complete": record["artifacts_complete"],
        "history_finite_and_ordered": record["history_finite_and_ordered"],
        "window_complete_postpeak": bool(window["passed"]),
        "positive_snes_reason": record["snes_reason"] > 0,
        "all_positive_ksp_reasons": all(value > 0 for value in ksp_reasons),
        "damage_kkt": values["damage_kkt_relative"] <= SINGLE_RUN_LIMITS["damage_kkt_relative"],
        "damage_lower_bound": values["damage_lower_bound_violation"]
        <= SINGLE_RUN_LIMITS["damage_bound_violation"],
        "damage_upper_bound": values["damage_upper_bound_violation"]
        <= SINGLE_RUN_LIMITS["damage_bound_violation"],
        "irreversibility": values["irreversibility_violation"]
        <= SINGLE_RUN_LIMITS["irreversibility_violation"],
        "stagger_error": values["stagger_error"] <= SINGLE_RUN_LIMITS["stagger_error"],
        "stagger_iterations": 0
        < record["stagger_iterations"]
        <= SINGLE_RUN_LIMITS["stagger_iterations"],
        "nonnegative_final_energies": values["elastic_energy_end"] >= -nonnegative_energy_guard
        and values["fracture_energy_end"] >= -nonnegative_energy_guard,
        "nondecreasing_fracture_energy": delta_fracture >= -nonnegative_energy_guard,
        "energy_closure": normalized_energy_closure
        <= SINGLE_RUN_LIMITS["normalized_energy_closure"],
        "reaction_balance": normalized_reaction_balance
        <= SINGLE_RUN_LIMITS["normalized_reaction_balance"],
        "mirror_displacement": mirror_displacement
        <= SINGLE_RUN_LIMITS["mirror_displacement_normalized_weighted_l2"],
        "mirror_damage": mirror_damage <= SINGLE_RUN_LIMITS["mirror_damage_normalized_weighted_l2"],
    }
    return {
        "formula_id": SINGLE_RUN_FORMULA_ID,
        "window": window,
        "delta_elastic_energy": delta_elastic,
        "incremental_fracture_dissipation": delta_fracture,
        "external_work": values["external_work"],
        "energy_closure_residual": energy_residual,
        "normalized_energy_closure": normalized_energy_closure,
        "reaction_balance_residual": reaction_residual,
        "normalized_reaction_balance": normalized_reaction_balance,
        "mirror_displacement_normalized_weighted_l2": mirror_displacement,
        "mirror_damage_normalized_weighted_l2": mirror_damage,
        "gates": gates,
        "passed": all(gates.values()),
    }


def qualification_quantity_registry(
    selected_endpoint: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the exact quantity-to-scale/cap/index registry."""

    scales = physical_scales(selected_endpoint)
    primary_cap = REPEATABILITY_CAPS["primary_global_curve_energy_displacement"]
    damage_cap = REPEATABILITY_CAPS["damage_field_and_crack_summary"]
    return {
        "primary.peak_reaction_y": {
            "scale_key": "reaction_force_unit_thickness",
            "scale": scales["reaction_force_unit_thickness"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "one_signed_peak_reaction",
        },
        "primary.peak_displacement": {
            "scale_key": "displacement",
            "scale": scales["displacement"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "one_earliest_tied_peak_displacement",
        },
        "primary.incremental_fracture_dissipation_at_endpoint": {
            "scale_key": "fracture_energy",
            "scale": scales["fracture_energy"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "one_endpoint_incremental_fracture_dissipation",
        },
        "curve.reaction_y": {
            "scale_key": "reaction_force_unit_thickness",
            "scale": scales["reaction_force_unit_thickness"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "all_selected_common_nominal_nodes",
        },
        "curve.elastic_energy": {
            "scale_key": "fracture_energy",
            "scale": scales["fracture_energy"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "all_selected_common_nominal_nodes",
        },
        "curve.fracture_energy": {
            "scale_key": "fracture_energy",
            "scale": scales["fracture_energy"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "all_selected_common_nominal_nodes",
        },
        "curve.external_work": {
            "scale_key": "fracture_energy",
            "scale": scales["fracture_energy"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "all_selected_common_nominal_nodes",
        },
        "field.endpoint_displacement": {
            "scale_key": "displacement",
            "scale": scales["displacement"],
            "cap_class": "primary_global_curve_energy_displacement",
            "cap": primary_cap,
            "index_set": "all_preregistered_endpoint_physical_samples",
        },
        "field.endpoint_damage": {
            "scale_key": "damage",
            "scale": scales["damage"],
            "cap_class": "damage_field_and_crack_summary",
            "cap": damage_cap,
            "index_set": "all_preregistered_endpoint_physical_samples",
        },
        "crack.regularised_crack_length": {
            "scale_key": "crack_length",
            "scale": scales["crack_length"],
            "cap_class": "damage_field_and_crack_summary",
            "cap": damage_cap,
            "index_set": "one_endpoint_regularised_crack_length",
        },
    }


def cross_mesh_primary_registry(selected_endpoint: float | None = None) -> dict[str, Any]:
    quantity_registry = qualification_quantity_registry(selected_endpoint)
    return {
        quantity_id: {
            "scale_key": quantity_registry[quantity_id]["scale_key"],
            "scale": quantity_registry[quantity_id]["scale"],
            "band": CROSS_MESH_BAND,
            "contraction": CROSS_MESH_CONTRACTION,
            "numerical_floor_cap": CROSS_MESH_FLOOR_CAP,
        }
        for quantity_id in (
            "primary.peak_reaction_y",
            "primary.peak_displacement",
            "primary.incremental_fracture_dissipation_at_endpoint",
        )
    }


def derive_repeatability_literal(
    left: Sequence[float],
    right: Sequence[float],
    *,
    mode: str,
    quantity_id: str,
    selected_endpoint: float,
) -> dict[str, Any]:
    """Derive one preregistered engineering envelope from an eligible Q pair."""

    _require(type(left) is list and type(right) is list, "comparison arrays must be JSON arrays")
    _require_strict_json_value(left, "left comparison array")
    _require_strict_json_value(right, "right comparison array")
    _require(len(left) == len(right) and len(left) > 0, "comparison index sets differ")
    _require(mode in ("cold", "serial_mpi"), "repeatability mode is not frozen")
    registry = qualification_quantity_registry(selected_endpoint)
    _require(quantity_id in registry, "repeatability quantity is not frozen")
    quantity = registry[quantity_id]
    scale_value = _finite(quantity["scale"], "repeatability scale")
    cap_value = _finite(quantity["cap"], "repeatability cap")
    absolute = round_up_125(64.0 * BINARY64_EPSILON * scale_value)
    observed_relative = 0.0
    maximum_magnitude = scale_value
    for index, (raw_left, raw_right) in enumerate(zip(left, right, strict=True)):
        left_value = _finite(raw_left, f"left[{index}]")
        right_value = _finite(raw_right, f"right[{index}]")
        denominator = max(scale_value, abs(left_value), abs(right_value))
        maximum_magnitude = max(maximum_magnitude, abs(left_value), abs(right_value))
        observed_relative = max(
            observed_relative,
            max(0.0, abs(left_value - right_value) - absolute) / denominator,
        )
    relative = round_up_125(
        max(64.0 * BINARY64_EPSILON, REPEATABILITY_MARGIN_FACTOR * observed_relative)
    )
    qualification_envelope_fraction = absolute / scale_value + relative * (
        maximum_magnitude / scale_value
    )
    return {
        "formula_id": REPEATABILITY_FORMULA_ID,
        "mode": mode,
        "quantity_id": quantity_id,
        "scale_key": quantity["scale_key"],
        "comparison_index_set": quantity["index_set"],
        "cap_class": quantity["cap_class"],
        "absolute": absolute,
        "relative": relative,
        "scale": scale_value,
        "maximum_magnitude": maximum_magnitude,
        "observed_relative_after_absolute_guard": observed_relative,
        "qualification_envelope_fraction": qualification_envelope_fraction,
        "hard_cap": cap_value,
        "passed": qualification_envelope_fraction <= cap_value,
        "inference_limit": "single-pair engineering envelope; not a confidence interval",
    }


def validate_repeatability_literal(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "repeatability literal must be a JSON object")
    _require_strict_json_value(payload, "repeatability literal")
    expected_keys = {
        "formula_id",
        "mode",
        "quantity_id",
        "scale_key",
        "comparison_index_set",
        "cap_class",
        "absolute",
        "relative",
        "scale",
        "maximum_magnitude",
        "observed_relative_after_absolute_guard",
        "qualification_envelope_fraction",
        "hard_cap",
        "passed",
        "inference_limit",
    }
    _require(set(payload) == expected_keys, "repeatability literal key set mismatch")
    mode = payload["mode"]
    quantity_id = payload["quantity_id"]
    _require(mode in ("cold", "serial_mpi"), "repeatability literal mode mismatch")
    _require(type(quantity_id) is str, "repeatability literal quantity must be a string")
    selected_endpoint = None
    if payload["scale_key"] == "displacement":
        selected_endpoint = _finite(payload["scale"], "repeatability displacement scale")
    registry = qualification_quantity_registry(selected_endpoint)
    _require(quantity_id in registry, "repeatability literal quantity mismatch")
    quantity = registry[quantity_id]
    scale = _finite(payload["scale"], "repeatability literal scale")
    hard_cap = _finite(payload["hard_cap"], "repeatability literal cap")
    absolute = _finite(payload["absolute"], "repeatability literal absolute")
    relative = _finite(payload["relative"], "repeatability literal relative")
    observed = _finite(
        payload["observed_relative_after_absolute_guard"],
        "repeatability literal observed relative",
    )
    envelope = _finite(
        payload["qualification_envelope_fraction"],
        "repeatability literal envelope",
    )
    maximum_magnitude = _finite(
        payload["maximum_magnitude"], "repeatability literal maximum magnitude"
    )
    _require(
        payload["formula_id"] == REPEATABILITY_FORMULA_ID
        and payload["scale_key"] == quantity["scale_key"]
        and payload["comparison_index_set"] == quantity["index_set"]
        and payload["cap_class"] == quantity["cap_class"]
        and payload["inference_limit"]
        == "single-pair engineering envelope; not a confidence interval",
        "repeatability literal semantic binding mismatch",
    )
    _require(
        scale.hex() == float(quantity["scale"]).hex()
        and hard_cap.hex() == float(quantity["cap"]).hex(),
        "repeatability literal scale or cap differs from registry",
    )
    _require(
        absolute.hex() == round_up_125(64.0 * BINARY64_EPSILON * scale).hex(),
        "repeatability absolute literal formula mismatch",
    )
    expected_relative = round_up_125(
        max(64.0 * BINARY64_EPSILON, REPEATABILITY_MARGIN_FACTOR * observed)
    )
    _require(relative.hex() == expected_relative.hex(), "repeatability relative formula mismatch")
    _require(
        absolute >= 0.0
        and relative >= 0.0
        and observed >= 0.0
        and maximum_magnitude >= scale
        and envelope >= 0.0,
        "repeatability literal domain mismatch",
    )
    expected_envelope = absolute / scale + relative * (maximum_magnitude / scale)
    _require(
        envelope.hex() == expected_envelope.hex(),
        "repeatability qualification-envelope formula mismatch",
    )
    _require(type(payload["passed"]) is bool, "repeatability passed must be Boolean")
    _require(
        payload["passed"] is (envelope <= hard_cap),
        "repeatability passed flag differs from hard cap",
    )


def derive_cross_mesh_floor(
    *,
    q04_coarse: float,
    q05_half: float,
    q06_cold: float,
    primary_id: str,
    selected_endpoint: float,
    cold_literal: Mapping[str, Any],
) -> dict[str, Any]:
    registry = cross_mesh_primary_registry(selected_endpoint)
    _require(primary_id in registry, "cross-mesh primary is not frozen")
    primary = registry[primary_id]
    scale_value = _finite(primary["scale"], "mesh scale")
    q04_value = _finite(q04_coarse, "Q04 value")
    q05_value = _finite(q05_half, "Q05 value")
    q06_value = _finite(q06_cold, "Q06 value")
    validate_repeatability_literal(cold_literal)
    _require(
        cold_literal["mode"] == "cold"
        and cold_literal["quantity_id"] == primary_id
        and cold_literal["scale_key"] == primary["scale_key"]
        and cold_literal["passed"] is True,
        "cold literal is not an eligible frozen primary envelope",
    )
    absolute = _finite(cold_literal["absolute"], "cold absolute literal")
    relative = _finite(cold_literal["relative"], "cold relative literal")
    _require(absolute >= 0.0 and relative >= 0.0, "cold literals must be non-negative")
    discretization = abs(q04_value - q05_value) / scale_value
    cold_envelope = absolute / scale_value + relative * (
        max(scale_value, abs(q04_value), abs(q06_value)) / scale_value
    )
    floor = round_up_125(discretization + cold_envelope)
    return {
        "formula_id": CROSS_MESH_FLOOR_FORMULA_ID,
        "primary_id": primary_id,
        "scale_key": primary["scale_key"],
        "scale": scale_value,
        "load_discretization_fraction": discretization,
        "cold_envelope_fraction": cold_envelope,
        "numerical_floor": floor,
        "maximum_allowed_floor": CROSS_MESH_FLOOR_CAP,
        "passed": 0.0 < floor <= CROSS_MESH_FLOOR_CAP,
    }


def evaluate_cross_mesh_gate(
    q2: float,
    q3: float,
    q4: float,
    *,
    primary_id: str,
    selected_endpoint: float,
    numerical_floor: float,
) -> dict[str, Any]:
    registry = cross_mesh_primary_registry(selected_endpoint)
    _require(primary_id in registry, "cross-mesh primary is not frozen")
    primary = registry[primary_id]
    scale_value = _finite(primary["scale"], "mesh scale")
    floor_value = _finite(numerical_floor, "mesh numerical floor")
    _require(scale_value > 0.0, "mesh scale must be positive")
    _require(0.0 < floor_value <= CROSS_MESH_FLOOR_CAP, "mesh floor is outside its domain")
    q2_value = _finite(q2, "q2")
    q3_value = _finite(q3, "q3")
    q4_value = _finite(q4, "q4")
    d23 = abs(q2_value - q3_value) / scale_value
    d34 = abs(q3_value - q4_value) / scale_value
    band_pass = d34 <= CROSS_MESH_BAND
    trend_pass = d34 <= CROSS_MESH_CONTRACTION * d23
    floor_pass = d23 <= floor_value and d34 <= floor_value
    return {
        "formula_id": CROSS_MESH_GATE_FORMULA_ID,
        "primary_id": primary_id,
        "scale_key": primary["scale_key"],
        "scale": scale_value,
        "d23": d23,
        "d34": d34,
        "band_pass": band_pass,
        "trend_pass": trend_pass,
        "floor_pass": floor_pass,
        "passed": band_pass and (trend_pass or floor_pass),
    }


def qualification_science_contract() -> dict[str, Any]:
    """Return the immutable scientific block for the qualification protocol."""

    theoretical_ceiling = SMALL_STRAIN_LIMIT * PHYSICAL_MODEL.height
    maximum_candidate = CANDIDATE_END_DISPLACEMENTS[-1]
    base_design = canonical_json_value(prospective_design())
    source_lineage = canonical_json_value(
        {
            "source_template_filenames": SOURCE_TEMPLATE_FILENAMES,
            "source_template_identities": SOURCE_TEMPLATE_IDENTITIES,
            "source_template_output_directories": SOURCE_TEMPLATE_OUTPUT_DIRECTORIES,
            "right_censored_lineage_identities": RIGHT_CENSORED_LINEAGE_IDENTITIES,
            "runtime_authentication_requirement": (
                "future_launch_preregistration_must_reauthenticate_every_record_from_bytes"
            ),
        }
    )
    stage_plan = []
    for stage in QUALIFICATION_STAGES:
        stage_plan.append(
            {
                **{
                    "stage_id": stage.stage_id,
                    "mesh": stage.mesh,
                    "mpi_ranks": stage.mpi_ranks,
                    "schedule": stage.schedule,
                    "role": stage.role,
                    "execution_mode": stage.execution_mode,
                    "depends_on": list(stage.depends_on),
                    "always_run": stage.always_run,
                },
                "dependency_policy": (
                    "all_predecessors_terminal" if stage.always_run else "all_predecessors_passed"
                ),
            }
        )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_name": PROTOCOL_NAME,
        "base_prospective_design": base_design,
        "base_prospective_design_sha256": canonical_sha256(base_design),
        "source_lineage": source_lineage,
        "source_lineage_sha256": canonical_sha256(source_lineage),
        "window_discovery": {
            "formula_id": WINDOW_FORMULA_ID,
            "control_mode": "ordinary_displacement_only_in_revision_1",
            "q01_mesh": "q2",
            "q01_process": "cold_mpi2",
            "coarse_increment": COARSE_INCREMENT,
            "half_increment": HALF_INCREMENT,
            "candidate_end_steps": list(CANDIDATE_END_STEPS),
            "candidate_end_displacements": list(CANDIDATE_END_DISPLACEMENTS),
            "theoretical_small_strain_ceiling": theoretical_ceiling,
            "small_strain_limit": SMALL_STRAIN_LIMIT,
            "maximum_candidate": maximum_candidate,
            "maximum_candidate_nominal_strain": maximum_candidate / PHYSICAL_MODEL.height,
            "minimum_postpeak_nominal_nodes": MINIMUM_POSTPEAK_NOMINAL_NODES,
            "maximum_endpoint_to_peak_reaction_ratio": (MAXIMUM_ENDPOINT_TO_PEAK_REACTION_RATIO),
            "peak_tie_rule": "earliest_nominal_step_at_signed_maximum_reaction",
            "selection_rule": "earliest_candidate_passing_all_window_checks",
            "adaptive_inserted_states_eligible_for_selection": False,
            "no_candidate_action": "fail_revision",
            "confirmation_failure_action": "fail_revision_without_advancing_candidate",
            "hybrid_controller_requires_new_revision": True,
            "candidate_scale_records": candidate_scale_records(),
        },
        "q01_execution_policy": {
            "adaptive_inserted_states_may_exist": True,
            "all_inserted_states_must_be_type_finite_order_authenticated": True,
            "inserted_states_eligible_for_window_selection_or_literals": False,
            "must_run_once_through_full_small_strain_ceiling": True,
            "candidate_branch_is_membership_selection_only": True,
        },
        "stage_plan": stage_plan,
        "q02_q07_fixed_table_policy": canonical_json_value(FORMAL_FIXED_TABLE_POLICY),
        "q05_load_discretization": {
            "formula_id": Q05_FORMULA_ID,
            "role": "accept_or_reject_q04_coarse_only",
            "eligible_as_formal_schedule": False,
            "half_nodes_equal_even_indexed_coarse_nodes": True,
            "load_table_identity_rule": (
                "Q04_u_j=j*2e-6_and_Q05_u_j=j*1e-6_with_binary64_exact_even_node_binding"
            ),
            "peak_reaction_formula": (
                "abs(Fpeak_coarse-Fpeak_half)/max(S_F,abs(Fpeak_coarse),abs(Fpeak_half))"
            ),
            "peak_reaction_relative_limit": Q05_LIMITS["peak_reaction_relative"],
            "peak_displacement_formula": "abs(upeak_coarse-upeak_half)/coarse_increment",
            "peak_displacement_in_coarse_steps_limit": Q05_LIMITS[
                "peak_displacement_in_coarse_steps"
            ],
            "endpoint_fracture_dissipation_formula": (
                "abs(Dend_coarse-Dend_half)/max(S_E,abs(Dend_coarse),abs(Dend_half))"
            ),
            "endpoint_fracture_dissipation_relative_limit": Q05_LIMITS[
                "endpoint_fracture_dissipation_relative"
            ],
            "common_node_formula": (
                "max_over_reaction_and_Eel_Efrac_Wext_and_coarse_nodes_of_"
                "abs(a-b)/max(fixed_physical_scale,abs(a),abs(b))"
            ),
            "common_node_reaction_and_energy_normalized_linf_limit": Q05_LIMITS[
                "common_node_reaction_and_energy_normalized_linf"
            ],
            "weighted_l2_formula": (
                "sqrt(sum(w*(a-b)^2)/sum(w))/max(fixed_physical_scale,"
                "sqrt(sum(w*a^2)/sum(w)),sqrt(sum(w*b^2)/sum(w)))"
            ),
            "physical_sample_identity_and_weights_must_be_recursive_type_exact": True,
            "physical_sample_weights_must_be_positive": True,
            "endpoint_displacement_normalized_weighted_l2_limit": Q05_LIMITS[
                "endpoint_displacement_normalized_weighted_l2"
            ],
            "endpoint_damage_weighted_l2_limit": Q05_LIMITS[
                "endpoint_damage_normalized_weighted_l2"
            ],
            "coarse_and_half_must_each_pass_single_run_and_window_gates": True,
            "half_minimum_postpeak_requirement_in_common_coarse_intervals": (
                MINIMUM_POSTPEAK_NOMINAL_NODES
            ),
            "coordinate_weight_or_semantic_mismatch_action": "authentication_failure",
            "failure_action": "fail_revision_without_promoting_half_step",
        },
        "single_run_science_gates": {
            "formula_id": SINGLE_RUN_FORMULA_ID,
            "limits": canonical_json_value(SINGLE_RUN_LIMITS),
            "energy_closure_formula": (
                "abs(delta_Eel+delta_Efrac-Wext)/max(S_E,abs(delta_Eel),abs(delta_Efrac),abs(Wext))"
            ),
            "reaction_balance_formula": (
                "abs(Rtop_y+Rbottom_y)/max(S_F,abs(Rtop_y),abs(Rbottom_y))"
            ),
            "mirror_displacement_formula": "weighted_L2_mirror_error/selected_endpoint",
            "mirror_damage_formula": "weighted_L2_mirror_error/1",
            "energy_nonnegative_guard": "1e-12*S_E",
            "signed_reaction_window_gate_required": True,
            "positive_snes_and_all_ksp_reasons_required": True,
            "source_authentication_and_complete_artifacts_required": True,
        },
        "physical_scales": physical_scales(),
        "quantity_registry": qualification_quantity_registry(),
        "repeatability_literal_derivation": {
            "formula_id": REPEATABILITY_FORMULA_ID,
            "modes": ["cold", "serial_mpi"],
            "mode_namespaces_are_disjoint": True,
            "rounding": "round_up_to_1_2_5_times_power_of_ten",
            "binary64_epsilon": BINARY64_EPSILON,
            "absolute_formula": "round125(64*epsilon*S_q)",
            "observed_formula": ("max_i(max(0,abs(a_i-b_i)-A_mq)/max(S_q,abs(a_i),abs(b_i)))"),
            "relative_formula": "round125(max(64*epsilon,4*r_observed_mq))",
            "margin_factor": REPEATABILITY_MARGIN_FACTOR,
            "caps": canonical_json_value(REPEATABILITY_CAPS),
            "cap_exceedance_action": "fail_qualification_without_clipping",
            "quantity_registry_is_the_only_scale_cap_and_index_source": True,
            "exact_endpoint_crack_summary_fields": [
                "active_node_count",
                "connected_component_count",
                "main_component_status",
            ],
            "exact_endpoint_crack_summary_fields_use_literal": False,
            "inference_limit": "single-pair_engineering_envelope_not_ci",
            "formal_holdout_rule": (
                "q4_all_three_cold_pairs_and_q3_serial_mpi_pair_use_frozen_literals"
            ),
        },
        "cross_mesh": {
            "floor_formula_id": CROSS_MESH_FLOOR_FORMULA_ID,
            "gate_formula_id": CROSS_MESH_GATE_FORMULA_ID,
            "primary_registry": cross_mesh_primary_registry(),
            "band": CROSS_MESH_BAND,
            "contraction": CROSS_MESH_CONTRACTION,
            "numerical_floor_cap": CROSS_MESH_FLOOR_CAP,
            "band_and_contraction_are_result_derived": False,
            "numerical_floor_formula": (
                "round125(abs(q_Q04-q_Q05)/S_q + A_cold_q/S_q + "
                "R_cold_q*max(S_q,abs(q_Q04),abs(q_Q06))/S_q)"
            ),
            "gate": "d34<=0.02 and (d34<=0.75*d23 or (d23<=N and d34<=N))",
            "all_primary_must_pass": True,
            "qualification_mesh_differences_may_tune_band_or_contraction": False,
        },
        "result_bound_literals": {
            "fields": [
                "selected_candidate_id",
                "selected_endpoint",
                "selected_coarse_load_table_sha256",
                "A_cold_q",
                "R_cold_q",
                "S_cold_q",
                "A_serial_mpi_q",
                "R_serial_mpi_q",
                "S_serial_mpi_q",
                "S_mesh_q",
                "N_mesh_q",
                "raw_discrepancies",
                "qualification_report_and_manifest_sha256",
            ],
            "values_are_present_in_this_protocol": False,
            "scale_binding_rule": (
                "S_values_are_selected_endpoint_substitutions_into_the_exact_quantity_registry"
            ),
            "cross_mesh_band_and_contraction_are_not_result_bound": True,
            "numerical_floor_domain": "0<N_mesh_q<=0.01",
            "binding_rule": (
                "future_terminal_must_apply_only_the_frozen_formula_id_to_eligible_Q_records"
            ),
            "unknown_or_ineligible_derivation_source_action": "fail_qualification",
        },
        "stop_and_terminal_policy": {
            "automatic_retry_allowed": False,
            "replacement_stage_allowed": False,
            "restart_or_resume_allowed": False,
            "first_hard_failure_stops_later_solve_stages": True,
            "later_receipt_status": "skipped_due_to_upstream_failure",
            "q08_dependency_policy": "all_predecessors_terminal",
            "q08_imports_or_calls_solver": False,
            "interrupted_missing_receipt_status": "not_observed_interruption",
            "terminal_only_recovery_may_run_solve": False,
            "qualification_pass_does_not_authorize_formal_p0a": True,
            "required_solve_stage_receipt_count": 7,
            "required_solve_stage_receipt_ids": [f"Q0{index}" for index in range(1, 8)],
            "exactly_one_receipt_per_solve_stage": True,
            "solve_stage_terminal_statuses": [
                "passed",
                "failed",
                "skipped_due_to_upstream_failure",
                "not_observed_interruption",
            ],
            "skip_receipt_launch_attempted": False,
            "skip_receipt_return_code": None,
            "missing_receipt_normalization_before_q08": (
                "launcher_or_terminal_only_recovery_synthesizes_not_observed_interruption_"
                "then_verifies_all_seven_predecessors_terminal_before_invoking_Q08"
            ),
            "authoritative_exit_zero_rule": (
                "all_Q01_Q07_passed_and_Q08_authentication_derivation_and_science_gates_passed"
            ),
            "authoritative_exit_nonzero_rule": (
                "any_failed_skipped_not_observed_or_Q08_gate_failure"
            ),
            "q08_failure_after_attempt_ledger_action": (
                "preserve_attempt_ledger_and_all_receipts_then_stop_without_retry"
            ),
        },
        "future_launch_preregistration_requirements": {
            "bind_this_science_artifact_path_sha256_and_bytes": True,
            "freeze_all_nine_candidate_branches_before_q01": True,
            "candidate_branch_count": len(CANDIDATE_END_STEPS),
            "each_branch_freezes_q02_q07_configs_and_load_tables": True,
            "each_branch_freezes_stage_argv_environment_output_and_temp_paths": True,
            "selected_load_table_is_membership_only": True,
            "creating_or_modifying_a_branch_after_q01": "forbidden",
            "future_toolchain_and_zero_solve_preflight_identity_required": True,
        },
        "phase_b_firewall": canonical_json_value(PHASE_B_FIREWALL),
        "eligible_evidence_policy": canonical_json_value(QUALIFICATION_SOURCE_POLICY),
    }
    return contract


def qualification_science_contract_sha256() -> str:
    return canonical_sha256(qualification_science_contract())


def validate_qualification_science_contract(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "qualification science contract must be a JSON object")
    _require_strict_json_value(payload, "qualification science contract")
    expected = qualification_science_contract()
    _require(set(payload) == set(expected), "qualification science-contract key set mismatch")
    _require(
        canonical_json_bytes(dict(payload)) == canonical_json_bytes(expected),
        "qualification science contract differs from the frozen value",
    )
    _require(
        payload["base_prospective_design_sha256"]
        == canonical_sha256(payload["base_prospective_design"]),
        "base prospective-design digest mismatch",
    )
    _require(
        payload["source_lineage_sha256"] == canonical_sha256(payload["source_lineage"]),
        "source-lineage digest mismatch",
    )


def qualification_protocol_envelope() -> dict[str, Any]:
    """Return the complete non-executable protocol block used by the generator."""

    science = qualification_science_contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "document_role": "science_contract_only",
        "not_execution_authorization": True,
        "create_only": True,
        "protocol_name": PROTOCOL_NAME,
        "protocol_revision": PROTOCOL_REVISION,
        "execution_authorized": False,
        "qualification_execution_authorized": False,
        "writes_solver_outputs": False,
        "writes_production_case_data": False,
        "science_contract": science,
        "science_contract_sha256": canonical_sha256(science),
        "execution_blockers": list(QUALIFICATION_EXECUTION_BLOCKERS),
        "future_launch_preregistration": {
            "must_bind_this_artifact_sha256": True,
            "must_preserve_science_contract_value_and_sha256": True,
            "must_authenticate_toolchain_preflight_argv_environment_and_outputs": True,
            "may_change_science_contract": False,
        },
        "immutability_contract": {
            "scientific_field_change_requires_new_revision": True,
            "future_launch_may_only_add_execution_bindings": True,
            "copying_without_path_sha256_and_bytes_binding_allowed": False,
        },
        "result": {
            "passed": True,
            "science_protocol_frozen": True,
            "all_scientific_degrees_of_freedom_frozen": True,
            "execution_authorized": False,
            "qualification_campaign_authorized": False,
            "formal_p0a_execution_authorized": False,
            "future_launch_preregistration_required": True,
            "future_toolchain_identity_required": True,
            "future_zero_solve_preflight_required": True,
        },
        "interpretation_limit": (
            "freezes qualification science choices only and authorizes no Q or formal solve"
        ),
    }


def validate_qualification_protocol_envelope(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "qualification protocol must be a JSON object")
    _require_strict_json_value(payload, "qualification protocol")
    expected = qualification_protocol_envelope()
    _require(set(payload) == set(expected), "qualification protocol key set mismatch")
    _require(
        canonical_json_bytes(dict(payload)) == canonical_json_bytes(expected),
        "qualification protocol envelope differs",
    )
    validate_qualification_science_contract(payload["science_contract"])
    _require(
        payload["science_contract_sha256"] == canonical_sha256(payload["science_contract"]),
        "qualification science-contract digest mismatch",
    )


def assert_qualification_execution_authorized() -> None:
    blockers = ", ".join(QUALIFICATION_EXECUTION_BLOCKERS)
    raise RuntimeError(f"P0-A qualification execution is not authorized: {blockers}")
