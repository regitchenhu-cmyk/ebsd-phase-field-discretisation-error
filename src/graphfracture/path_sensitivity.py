"""Strict 2x2 fracture-energy path-sensitivity audit.

The audit is deliberately post-processing only.  By default it delegates every
input directory to the completed hybrid inclined-interface audit.  An explicit
certified-failure mode additionally permits predictor-off cases that exhausted
their configured dyadic refinement budget, while keeping that robustness
outcome distinct from four-path convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inclined_studies import case_audit_report, case_progress_report

PATH_SENSITIVITY_SCHEMA_VERSION = 3
ROLE_ORDER = (
    "predon_dE32",
    "predoff_dE32",
    "predon_half64",
    "predoff_half64",
)
ROLE_CONTRACT = {
    "predon_dE32": (True, 32),
    "predoff_dE32": (False, 32),
    "predon_half64": (True, 64),
    "predoff_half64": (False, 64),
}
RESPONSE_FIELDS = (
    "displacement",
    "reaction_y",
    "elastic_energy",
    "total_internal_energy",
    "regularised_crack_length",
)
PREDICTOR_RESPONSE_TOLERANCE = 1.0e-4
HALF_INCREMENT_RESPONSE_TOLERANCE = 1.0e-2
RESPONSE_ABSOLUTE_TOLERANCE = 1.0e-12
POST_PEAK_MINIMUM_STATES = 3
POST_PEAK_MINIMUM_DROP = 1.0e-3
NEGATIVE_DISPLACEMENT_THRESHOLD = -1.0e-8
NEGATIVE_DISPLACEMENT_CONSECUTIVE_STATES = 2
TARGET_RELATIVE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class _State:
    phase_step: int
    target: float
    displacement: float
    load_increment: float
    reaction_y: float
    elastic_energy: float
    total_internal_energy: float
    regularised_crack_length: float
    rightmost_damaged_x: float
    threshold_consensus: str
    threshold_signature: tuple[tuple[float, str, bool], ...]


@dataclass(frozen=True)
class _Case:
    role: str
    directory: Path
    audit: dict[str, Any]
    run_status: str
    failure_certificate: dict[str, Any] | None
    config: dict[str, Any]
    predictor: bool
    predictor_latch: int
    target_increment: float
    configured_steps: int
    minimum_increment: float
    mesh_h: float
    mpi_ranks: int
    implementation_fingerprint: str
    runtime_fingerprint: str
    switch: _State
    path: tuple[_State, ...]


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be a JSON object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{context} must be a JSON array")
    return value


def _finite(value: Any, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _integer(value: Any, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _string(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    return _mapping(value, str(path))


def _csv_float(row: dict[str, str], name: str, path: Path, index: int) -> float:
    raw = row.get(name)
    if raw is None or not raw.strip():
        raise ValueError(f"{path}: row {index} has no {name}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: row {index} has invalid {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: row {index} has non-finite {name}")
    return value


def _csv_int(row: dict[str, str], name: str, path: Path, index: int) -> int:
    raw = row.get(name)
    if raw is None or not raw.strip():
        raise ValueError(f"{path}: row {index} has no {name}")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: row {index} has invalid {name}") from exc


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=TARGET_RELATIVE_TOLERANCE,
        abs_tol=1.0e-12,
    )


def _threshold_signature(value: Any, context: str) -> tuple[tuple[float, str, bool], ...]:
    signature: list[tuple[float, str, bool]] = []
    for index, raw in enumerate(_list(value, context)):
        item = _mapping(raw, f"{context}[{index}]")
        classification = item.get("geometric_classification")
        if type(classification) is not str or not classification:
            raise ValueError(f"{context}[{index}].geometric_classification must be a string")
        signature.append(
            (
                _finite(item.get("threshold"), f"{context}[{index}].threshold"),
                classification,
                _boolean(
                    item.get("reached_interface"),
                    f"{context}[{index}].reached_interface",
                ),
            )
        )
    if len(signature) != 3:
        raise ValueError(f"{context} must contain the registered three thresholds")
    return tuple(signature)


def _state(
    row: dict[str, str],
    *,
    row_index: int,
    history_path: Path,
    target: float,
    phase_step: int,
    interface_record: Any,
) -> _State:
    interface = _mapping(
        interface_record,
        f"{history_path.parent / 'interface_history.json'}: records[{row_index}]",
    )
    consensus = interface.get("threshold_consensus")
    if type(consensus) is not str or not consensus:
        raise ValueError(
            f"{history_path.parent / 'interface_history.json'}: "
            f"records[{row_index}].threshold_consensus must be a string"
        )
    return _State(
        phase_step=phase_step,
        target=target,
        displacement=_csv_float(row, "displacement", history_path, row_index),
        load_increment=_csv_float(row, "load_increment", history_path, row_index),
        reaction_y=_csv_float(row, "reaction_y", history_path, row_index),
        elastic_energy=_csv_float(row, "elastic_energy", history_path, row_index),
        total_internal_energy=_csv_float(
            row,
            "total_internal_energy",
            history_path,
            row_index,
        ),
        regularised_crack_length=_csv_float(
            row,
            "regularised_crack_length",
            history_path,
            row_index,
        ),
        rightmost_damaged_x=_csv_float(
            row,
            "rightmost_damaged_x",
            history_path,
            row_index,
        ),
        threshold_consensus=consensus,
        threshold_signature=_threshold_signature(
            interface.get("threshold_results"),
            f"{history_path.parent / 'interface_history.json'}: "
            f"records[{row_index}].threshold_results",
        ),
    )


def _coordinate_ulp_tolerance(*coordinates: float) -> float:
    return 16.0 * max(math.ulp(abs(value)) for value in coordinates)


def _coordinate_close(left: float, right: float, *coordinates: float) -> bool:
    return abs(left - right) <= _coordinate_ulp_tolerance(
        left,
        right,
        *coordinates,
    )


def _certify_failed_prefix(
    directory: Path,
    progress: dict[str, Any],
    config: dict[str, Any],
    rows: list[dict[str, str]],
    states: list[_State],
) -> dict[str, Any]:
    context = f"{directory}: certified predictor-off failure"
    if (
        progress.get("run_status") != "failed"
        or progress.get("all_persisted_states_passed") is not True
    ):
        raise ValueError(f"{context} did not pass persisted-state validation")
    if not states:
        raise ValueError(f"{context} has no accepted fracture-energy prefix")

    method = _mapping(progress.get("method_status"), f"{context}.method_status")
    if method.get("status") != "failed":
        raise ValueError(f"{context} has no explicit failed method certificate")
    if method.get("resume_supported") is not True:
        raise ValueError(f"{context} is not restartable from its accepted prefix")
    failure_type = _string(
        method.get("failure_type"), f"{context}.method_status.failure_type"
    )
    failure_message = _string(
        method.get("failure_message"), f"{context}.method_status.failure_message"
    )
    if failure_type != "RuntimeError" or not (
        failure_message.startswith(
            "hybrid path solve failed at absolute fracture energy "
        )
        and "; minimum attempted control increment was " in failure_message
    ):
        raise ValueError(f"{context} is not a terminal hybrid path-solve failure")

    path_control = _mapping(config.get("path_control"), f"{context}.path_control")
    target_increment = _finite(
        path_control.get("target_increment"), f"{context}.target_increment"
    )
    minimum_increment = _finite(
        path_control.get("minimum_increment"), f"{context}.minimum_increment"
    )
    maximum_subdivisions = _integer(
        path_control.get("maximum_subdivisions"),
        f"{context}.maximum_subdivisions",
    )
    configured_targets = _integer(
        path_control.get("steps"), f"{context}.configured_path_targets"
    )
    accepted_step = _integer(
        method.get("accepted_step"), f"{context}.method_status.accepted_step"
    )
    accepted_displacement = _finite(
        method.get("accepted_displacement"),
        f"{context}.method_status.accepted_displacement",
    )
    accepted_control = _finite(
        method.get("accepted_control"),
        f"{context}.method_status.accepted_control",
    )
    accepted_path_steps = _integer(
        method.get("accepted_path_steps"),
        f"{context}.method_status.accepted_path_steps",
    )
    pending_targets = _integer(
        method.get("pending_control_targets"),
        f"{context}.method_status.pending_control_targets",
    )
    if (
        accepted_step != len(rows) - 1
        or accepted_path_steps != len(states)
        or _integer(
            method.get("configured_path_targets"),
            f"{context}.method_status.configured_path_targets",
        )
        != configured_targets
        or pending_targets <= 0
        or method.get("completion_condition") != "fracture_energy_queue_exhausted"
        or not _close(accepted_control, states[-1].target)
        or not _close(
            accepted_displacement,
            _csv_float(rows[-1], "displacement", directory / "history.csv", len(rows) - 1),
        )
    ):
        raise ValueError(f"{context} accepted-prefix certificate is inconsistent")

    summary = _mapping(progress.get("attempts"), f"{context}.attempts")
    if summary.get("present") is not True or summary.get("status") != "partial":
        raise ValueError(f"{context} has no partial attempt provenance")
    attempt_count = _integer(
        summary.get("unaccepted_attempts"), f"{context}.unaccepted_attempts"
    )
    attempt_path = directory / "attempt_history.json"
    attempt_payload = _read_json(attempt_path)
    if (
        _integer(attempt_payload.get("schema_version"), f"{attempt_path}: schema_version")
        != 1
        or attempt_payload.get("status") != "partial"
    ):
        raise ValueError(f"{context} has an invalid attempt-history certificate")
    attempts = _list(attempt_payload.get("records"), f"{attempt_path}: records")
    if attempt_count <= 0 or len(attempts) != attempt_count:
        raise ValueError(f"{context} attempt count is inconsistent")
    terminal = _mapping(attempts[-1], f"{attempt_path}: terminal attempt")
    terminal_attempt = _integer(
        terminal.get("attempt"), f"{attempt_path}: terminal attempt number"
    )
    terminal_accepted_step = _integer(
        terminal.get("accepted_step_before_attempt"),
        f"{attempt_path}: terminal accepted step",
    )
    terminal_accepted_control = _finite(
        terminal.get("accepted_control"),
        f"{attempt_path}: terminal accepted control",
    )
    terminal_accepted_displacement = _finite(
        terminal.get("accepted_displacement"),
        f"{attempt_path}: terminal accepted displacement",
    )
    failed_target = _finite(
        terminal.get("target_control"), f"{attempt_path}: terminal target control"
    )
    control_increment = _finite(
        terminal.get("control_increment"), f"{attempt_path}: terminal control increment"
    )
    subdivision_level = _integer(
        terminal.get("subdivision_level"),
        f"{attempt_path}: terminal subdivision level",
    )
    scheduled_step = _integer(
        terminal.get("scheduled_step"), f"{attempt_path}: terminal scheduled step"
    )
    will_subdivide = _boolean(
        terminal.get("will_subdivide"), f"{attempt_path}: terminal will_subdivide"
    )
    terminal_failure_type = _string(
        terminal.get("failure_type"), f"{attempt_path}: terminal failure_type"
    )
    terminal_failure_message = _string(
        terminal.get("failure_message"), f"{attempt_path}: terminal failure_message"
    )
    expected_minimum = math.ldexp(target_increment, -maximum_subdivisions)
    if (
        terminal.get("control_phase") != "fracture_energy"
        or terminal_attempt != attempt_count
        or terminal_accepted_step != accepted_step
        or not _close(terminal_accepted_control, accepted_control)
        or not _close(terminal_accepted_displacement, accepted_displacement)
        or failed_target <= accepted_control
        or not _coordinate_close(
            control_increment,
            failed_target - accepted_control,
            accepted_control,
            failed_target,
        )
        or subdivision_level != maximum_subdivisions
        or will_subdivide
        or not _coordinate_close(
            control_increment,
            minimum_increment,
            accepted_control,
            failed_target,
        )
        or not _coordinate_close(
            control_increment,
            expected_minimum,
            accepted_control,
            failed_target,
        )
    ):
        raise ValueError(
            f"{context} did not terminate at the authenticated minimum increment"
        )

    return {
        "certified": True,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "failed_control_target": failed_target,
        "minimum_increment": minimum_increment,
        "maximum_subdivisions": maximum_subdivisions,
        "accepted_prefix": {
            "accepted_step": accepted_step,
            "accepted_displacement": accepted_displacement,
            "accepted_control": accepted_control,
            "accepted_path_steps": accepted_path_steps,
            "pending_control_targets": pending_targets,
        },
        "attempts": {
            "count": attempt_count,
            "terminal_attempt": terminal_attempt,
            "scheduled_step": scheduled_step,
            "subdivision_level": subdivision_level,
            "control_increment": control_increment,
            "will_subdivide": will_subdivide,
            "failure_type": terminal_failure_type,
            "failure_message": terminal_failure_message,
        },
        "minimum_increment_reached": True,
    }


def _read_case(
    role: str,
    case_directory: str | Path,
    *,
    allow_certified_failures: bool,
) -> _Case:
    directory = Path(case_directory).resolve()
    completion_present = (directory / "completion.json").is_file()
    if not allow_certified_failures or completion_present:
        audit = case_audit_report(directory)
        if audit.get("all_checks_passed") is not True or "path_control" not in audit:
            raise ValueError(f"{directory}: did not pass the completed hybrid audit")
        run_status = "complete"
    else:
        if ROLE_CONTRACT[role][0]:
            raise ValueError(
                f"{role}: predictor-on case must pass the completed hybrid audit"
            )
        audit = case_progress_report(directory)
        if (
            audit.get("run_status") != "failed"
            or audit.get("all_persisted_states_passed") is not True
        ):
            raise ValueError(f"{directory}: is not a certified failed hybrid prefix")
        run_status = "failed"

    config_path = directory / "config.resolved.json"
    config = _read_json(config_path)
    path_control = _mapping(config.get("path_control"), f"{config_path}: path_control")
    predictor = _boolean(
        path_control.get("use_energy_predictor"),
        f"{config_path}: path_control.use_energy_predictor",
    )
    predictor_latch = _integer(
        path_control.get(
            "energy_predictor_disable_after_nominal_path_step",
            -1,
        ),
        (
            f"{config_path}: "
            "path_control.energy_predictor_disable_after_nominal_path_step"
        ),
    )
    if predictor_latch < -1:
        raise ValueError(
            f"{config_path}: "
            "path_control.energy_predictor_disable_after_nominal_path_step "
            "must be at least -1"
        )
    target_increment = _finite(
        path_control.get("target_increment"),
        f"{config_path}: path_control.target_increment",
    )
    configured_steps = _integer(
        path_control.get("steps"),
        f"{config_path}: path_control.steps",
    )
    minimum_increment = _finite(
        path_control.get("minimum_increment"),
        f"{config_path}: path_control.minimum_increment",
    )

    geometry = _mapping(config.get("geometry"), f"{config_path}: geometry")
    length = _finite(geometry.get("length"), f"{config_path}: geometry.length")
    nx = _integer(geometry.get("nx"), f"{config_path}: geometry.nx")
    if length <= 0.0 or nx <= 0:
        raise ValueError(f"{config_path}: invalid horizontal mesh spacing")
    mesh_h = length / nx
    audit_runtime = _mapping(audit.get("runtime"), f"{directory}: hybrid audit runtime")
    mpi_ranks = _integer(
        audit_runtime.get("mpi_ranks"),
        f"{directory}: runtime.mpi_ranks",
    )
    runtime_path = directory / "runtime.json"
    runtime = _read_json(runtime_path)
    implementation_fingerprint = _string(
        runtime.get("implementation_fingerprint"),
        f"{runtime_path}: implementation_fingerprint",
    )
    runtime_fingerprint = _string(
        runtime.get("runtime_fingerprint"),
        f"{runtime_path}: runtime_fingerprint",
    )

    history_path = directory / "history.csv"
    try:
        with history_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{history_path}: missing CSV header")
            rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"invalid CSV in {history_path}: {exc}") from exc
    interface_path = directory / "interface_history.json"
    interface_records = _list(
        _read_json(interface_path).get("records"),
        f"{interface_path}: records",
    )
    if len(interface_records) != len(rows):
        raise ValueError(f"{directory}: history and interface record counts differ")

    switch_index: int | None = None
    path_indices: list[int] = []
    path_started = False
    for index, row in enumerate(rows):
        phase = row.get("control_phase")
        if phase == "displacement":
            if path_started:
                raise ValueError(f"{history_path}: displacement phase resumes after path control")
            switch_index = index
        elif phase == "fracture_energy":
            path_started = True
            path_indices.append(index)
        else:
            raise ValueError(f"{history_path}: row {index} has invalid control_phase")
    if switch_index is None or not path_indices:
        raise ValueError(f"{history_path}: no hybrid switch/path records")

    switch_row = rows[switch_index]
    switch_energy = _csv_float(
        switch_row,
        "fracture_energy",
        history_path,
        switch_index,
    )
    switch = _state(
        switch_row,
        row_index=switch_index,
        history_path=history_path,
        target=switch_energy,
        phase_step=0,
        interface_record=interface_records[switch_index],
    )
    states: list[_State] = []
    for expected_phase_step, row_index in enumerate(path_indices, start=1):
        row = rows[row_index]
        phase_step = _csv_int(row, "phase_step", history_path, row_index)
        if phase_step != expected_phase_step:
            raise ValueError(f"{history_path}: fracture-energy phase steps are not contiguous")
        target = _csv_float(row, "control_target", history_path, row_index)
        states.append(
            _state(
                row,
                row_index=row_index,
                history_path=history_path,
                target=target,
                phase_step=phase_step,
                interface_record=interface_records[row_index],
            )
        )
    if run_status == "complete" and len(states) != configured_steps:
        raise ValueError(f"{history_path}: path record count disagrees with config")
    failure_certificate = (
        _certify_failed_prefix(directory, audit, config, rows, states)
        if run_status == "failed"
        else None
    )

    return _Case(
        role=role,
        directory=directory,
        audit=audit,
        run_status=run_status,
        failure_certificate=failure_certificate,
        config=config,
        predictor=predictor,
        predictor_latch=predictor_latch,
        target_increment=target_increment,
        configured_steps=configured_steps,
        minimum_increment=minimum_increment,
        mesh_h=mesh_h,
        mpi_ranks=mpi_ranks,
        implementation_fingerprint=implementation_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        switch=switch,
        path=tuple(states),
    )


def _canonical_matrix_config(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value.pop("source_path", None)
    output = _mapping(value.get("output"), "config.output")
    output.pop("directory", None)
    path_control = _mapping(value.get("path_control"), "config.path_control")
    for name in (
        "use_energy_predictor",
        "target_increment",
        "steps",
        "minimum_increment",
    ):
        if name not in path_control:
            raise ValueError(f"config.path_control has no explicit {name}")
        path_control.pop(name)
    path_control.pop(
        "energy_predictor_disable_after_nominal_path_step",
        None,
    )
    return value


def _validate_matrix(cases: dict[str, _Case]) -> dict[str, Any]:
    reference = cases[ROLE_ORDER[0]]
    canonical = _canonical_matrix_config(reference.config)
    for role in ROLE_ORDER:
        case = cases[role]
        expected_predictor, expected_steps = ROLE_CONTRACT[role]
        if case.predictor is not expected_predictor:
            raise ValueError(f"{role}: predictor policy disagrees with the matrix role")
        if case.predictor_latch != -1:
            raise ValueError(
                f"{role}: the fixed 2x2 matrix requires "
                "energy_predictor_disable_after_nominal_path_step=-1 so each "
                "case represents a global predictor-on/off policy"
            )
        if case.configured_steps != expected_steps:
            raise ValueError(f"{role}: expected {expected_steps} configured path targets")
        if _canonical_matrix_config(case.config) != canonical:
            raise ValueError(
                f"{role}: configuration differs beyond predictor, target increment, "
                "path steps, minimum increment and output directory"
            )
        if case.mpi_ranks != reference.mpi_ranks:
            raise ValueError(f"{role}: MPI rank count differs across the matrix")
        if case.implementation_fingerprint != reference.implementation_fingerprint:
            raise ValueError(f"{role}: implementation fingerprint differs across the matrix")
        if case.runtime_fingerprint != reference.runtime_fingerprint:
            raise ValueError(f"{role}: runtime fingerprint differs across the matrix")
        if not _close(case.mesh_h, reference.mesh_h):
            raise ValueError(f"{role}: horizontal mesh spacing differs across the matrix")

    full_on = cases["predon_dE32"]
    full_off = cases["predoff_dE32"]
    half_on = cases["predon_half64"]
    half_off = cases["predoff_half64"]
    if not _close(full_on.target_increment, full_off.target_increment):
        raise ValueError("full-increment predictor pair uses different increments")
    if not _close(half_on.target_increment, half_off.target_increment):
        raise ValueError("half-increment predictor pair uses different increments")
    if not _close(2.0 * half_on.target_increment, full_on.target_increment):
        raise ValueError("half-increment cases do not use exactly half the reference increment")
    if not _close(full_on.minimum_increment, full_off.minimum_increment):
        raise ValueError("full-increment predictor pair uses different minimum increments")
    if not _close(half_on.minimum_increment, half_off.minimum_increment):
        raise ValueError("half-increment predictor pair uses different minimum increments")
    if not _close(2.0 * half_on.minimum_increment, full_on.minimum_increment):
        raise ValueError("half-increment minimum increment is not scaled by one half")
    if not _close(
        full_on.minimum_increment / full_on.target_increment,
        half_on.minimum_increment / half_on.target_increment,
    ):
        raise ValueError("relative adaptive subdivision floor differs across increments")
    path_control = _mapping(reference.config.get("path_control"), "config.path_control")
    maximum_subdivisions = _integer(
        path_control.get("maximum_subdivisions"),
        "config.path_control.maximum_subdivisions",
    )
    expected_relative_floor = math.ldexp(1.0, -maximum_subdivisions)
    if not _close(
        full_on.minimum_increment / full_on.target_increment,
        expected_relative_floor,
    ):
        raise ValueError(
            "minimum increment does not preserve the configured full dyadic depth"
        )

    cumulative = full_on.target_increment * full_on.configured_steps
    if not _close(cumulative, half_on.target_increment * half_on.configured_steps):
        raise ValueError("full and half increments do not have a common energy window")
    final_control_target = reference.switch.target + cumulative
    for role in ROLE_ORDER[1:]:
        if not _close(cases[role].switch.target, reference.switch.target):
            raise ValueError(f"{role}: accepted switch fracture energy differs")
    for role in ROLE_ORDER:
        case = cases[role]
        if case.run_status == "complete":
            if not _close(case.path[-1].target, final_control_target):
                raise ValueError(f"{role}: final control target differs")
        else:
            certificate = _mapping(
                case.failure_certificate,
                f"{role}: failure certificate",
            )
            failed_target = _finite(
                certificate.get("failed_control_target"),
                f"{role}: failed control target",
            )
            if not reference.switch.target < failed_target <= final_control_target + 1.0e-12:
                raise ValueError(f"{role}: certified failure lies outside the path window")

    return {
        "passed": True,
        "case_run_statuses": {role: cases[role].run_status for role in ROLE_ORDER},
        "reference_increment": full_on.target_increment,
        "half_increment": half_on.target_increment,
        "reference_steps": full_on.configured_steps,
        "half_steps": half_on.configured_steps,
        "maximum_subdivisions": maximum_subdivisions,
        "relative_minimum_increment": expected_relative_floor,
        "switch_fracture_energy": reference.switch.target,
        "cumulative_energy_increment": cumulative,
        "final_control_target": final_control_target,
        "mesh_h": reference.mesh_h,
        "mpi_ranks": reference.mpi_ranks,
        "implementation_fingerprint": reference.implementation_fingerprint,
        "runtime_fingerprint": reference.runtime_fingerprint,
        "energy_predictor_policy": "global_on_off_without_nominal_step_latch",
        "energy_predictor_disable_after_nominal_path_step": -1,
    }


def _normalised_difference(left: float, right: float, relative_tolerance: float) -> float:
    scale = max(
        abs(left),
        abs(right),
        RESPONSE_ABSOLUTE_TOLERANCE / relative_tolerance,
    )
    return abs(left - right) / scale


def _pair_report(
    name: str,
    left: _Case,
    right: _Case,
    pairs: list[tuple[_State, _State]],
    *,
    response_tolerance: float,
    tip_tolerance: float,
    comparison_scope: str = "complete_window",
) -> dict[str, Any]:
    if len(pairs) < 2:
        raise ValueError(f"{name}: no common accepted path targets beyond the switch")
    maxima = {field: 0.0 for field in RESPONSE_FIELDS}
    maximum_tip = 0.0
    mismatch_count = 0
    first_mismatch: dict[str, Any] | None = None
    for left_state, right_state in pairs:
        if not _close(left_state.target, right_state.target):
            raise ValueError(
                f"{name}: common-target pairing disagrees at "
                f"{left_state.target:.17g}/{right_state.target:.17g}"
            )
        for field in RESPONSE_FIELDS:
            error = _normalised_difference(
                float(getattr(left_state, field)),
                float(getattr(right_state, field)),
                response_tolerance,
            )
            maxima[field] = max(maxima[field], error)
        maximum_tip = max(
            maximum_tip,
            abs(left_state.rightmost_damaged_x - right_state.rightmost_damaged_x),
        )
        classifications_match = (
            left_state.threshold_consensus == right_state.threshold_consensus
            and left_state.threshold_signature == right_state.threshold_signature
        )
        if not classifications_match:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "control_target": left_state.target,
                    "left_consensus": left_state.threshold_consensus,
                    "right_consensus": right_state.threshold_consensus,
                    "left_thresholds": left_state.threshold_signature,
                    "right_thresholds": right_state.threshold_signature,
                }
    maximum_response = max(maxima.values())
    classification_consistent = mismatch_count == 0
    passed = (
        maximum_response <= response_tolerance
        and maximum_tip <= tip_tolerance + 1.0e-12
        and classification_consistent
    )
    return {
        "name": name,
        "left": left.role,
        "right": right.role,
        "comparison_scope": comparison_scope,
        "common_target_rule": (
            "only accepted control targets aligned within the registered target tolerance"
        ),
        "common_states_including_switch": len(pairs),
        "common_path_targets": len(pairs) - 1,
        "switch_fracture_energy": pairs[0][0].target,
        "first_control_target": pairs[1][0].target,
        "last_control_target": pairs[-1][0].target,
        "response_tolerance": response_tolerance,
        "absolute_response_tolerance": RESPONSE_ABSOLUTE_TOLERANCE,
        "maximum_normalised_response_difference": maximum_response,
        "maximum_normalised_response_difference_by_field": maxima,
        "tip_tolerance": tip_tolerance,
        "maximum_absolute_tip_difference": maximum_tip,
        "classification_consistent": classification_consistent,
        "classification_mismatch_count": mismatch_count,
        "first_classification_mismatch": first_mismatch,
        "passed": passed,
    }


def _same_grid_pairs(left: _Case, right: _Case) -> list[tuple[_State, _State]]:
    if len(left.path) != len(right.path):
        raise ValueError(f"{left.role}/{right.role}: same-grid path lengths differ")
    return [(left.switch, right.switch), *zip(left.path, right.path, strict=True)]


def _full_half_pairs(full: _Case, half: _Case) -> list[tuple[_State, _State]]:
    if len(half.path) != 2 * len(full.path):
        raise ValueError(f"{full.role}/{half.role}: half-step path is not twice as long")
    return [(full.switch, half.switch)] + [
        (full_state, half.path[2 * index + 1])
        for index, full_state in enumerate(full.path)
    ]


def _common_target_pairs(left: _Case, right: _Case) -> list[tuple[_State, _State]]:
    left_states = (left.switch, *left.path)
    right_states = (right.switch, *right.path)
    pairs: list[tuple[_State, _State]] = []
    left_index = right_index = 0
    while left_index < len(left_states) and right_index < len(right_states):
        left_state = left_states[left_index]
        right_state = right_states[right_index]
        if _close(left_state.target, right_state.target):
            pairs.append((left_state, right_state))
            left_index += 1
            right_index += 1
        elif left_state.target < right_state.target:
            left_index += 1
        else:
            right_index += 1
    return pairs


def _case_event_report(case: _Case) -> dict[str, Any]:
    branch = (case.switch, *case.path)
    maximum_reaction = max(state.reaction_y for state in branch)
    peak_index = max(
        index
        for index, state in enumerate(branch)
        if state.reaction_y == maximum_reaction
    )
    peak = branch[peak_index]
    post_peak_states = len(branch) - peak_index - 1
    cumulative_drop = (peak.reaction_y - branch[-1].reaction_y) / max(
        abs(peak.reaction_y),
        1.0e-30,
    )
    peak_confirmed = (
        post_peak_states >= POST_PEAK_MINIMUM_STATES
        and cumulative_drop >= POST_PEAK_MINIMUM_DROP
    )

    maximum_negative_run = 0
    current_run = 0
    first_negative_phase_step: int | None = None
    first_consecutive_phase_step: int | None = None
    fold_target: float | None = None
    fold_interval: list[float] | None = None
    previous_target = case.switch.target
    for state in case.path:
        if state.load_increment < NEGATIVE_DISPLACEMENT_THRESHOLD:
            if first_negative_phase_step is None:
                first_negative_phase_step = state.phase_step
                fold_target = state.target
                fold_interval = [previous_target, state.target]
            current_run += 1
            if (
                current_run == NEGATIVE_DISPLACEMENT_CONSECUTIVE_STATES
                and first_consecutive_phase_step is None
            ):
                first_consecutive_phase_step = (
                    state.phase_step - NEGATIVE_DISPLACEMENT_CONSECUTIVE_STATES + 1
                )
            maximum_negative_run = max(maximum_negative_run, current_run)
        else:
            current_run = 0
        previous_target = state.target
    negative_displacement_confirmed = (
        maximum_negative_run >= NEGATIVE_DISPLACEMENT_CONSECUTIVE_STATES
    )
    return {
        "peak": {
            "reaction_y": peak.reaction_y,
            "fracture_energy": peak.target,
            "displacement": peak.displacement,
            "phase_step": peak.phase_step,
            "at_right_endpoint": peak_index == len(branch) - 1,
        },
        "post_peak_states": post_peak_states,
        "final_cumulative_reaction_drop_relative": cumulative_drop,
        "peak_confirmation": {
            "minimum_post_peak_states": POST_PEAK_MINIMUM_STATES,
            "minimum_cumulative_drop_relative": POST_PEAK_MINIMUM_DROP,
            "passed": peak_confirmed,
        },
        "negative_displacement": {
            "increment_threshold": NEGATIVE_DISPLACEMENT_THRESHOLD,
            "required_consecutive_states": NEGATIVE_DISPLACEMENT_CONSECUTIVE_STATES,
            "maximum_consecutive_states": maximum_negative_run,
            "first_negative_phase_step": first_negative_phase_step,
            "first_consecutive_run_phase_step": first_consecutive_phase_step,
            "passed": negative_displacement_confirmed,
        },
        "fold": {
            "definition": "first accepted target with load_increment < -1e-8",
            "fracture_energy": fold_target,
            "bracketing_control_targets": fold_interval,
        },
        "passed": peak_confirmed and negative_displacement_confirmed,
    }


def _event_drift_report(
    cases: dict[str, _Case],
    events: dict[str, dict[str, Any]],
    reference_increment: float,
) -> dict[str, Any]:
    role_pairs = (
        ("predictor_full", "predon_dE32", "predoff_dE32"),
        ("predictor_half", "predon_half64", "predoff_half64"),
        ("increment_predon", "predon_dE32", "predon_half64"),
        ("increment_predoff", "predoff_dE32", "predoff_half64"),
    )
    reports: list[dict[str, Any]] = []
    for name, left_role, right_role in role_pairs:
        left_peak = float(events[left_role]["peak"]["fracture_energy"])
        right_peak = float(events[right_role]["peak"]["fracture_energy"])
        left_fold = events[left_role]["fold"]["fracture_energy"]
        right_fold = events[right_role]["fold"]["fracture_energy"]
        peak_drift = abs(left_peak - right_peak)
        fold_drift = (
            None
            if left_fold is None or right_fold is None
            else abs(float(left_fold) - float(right_fold))
        )
        passed = (
            peak_drift <= reference_increment + 1.0e-12
            and fold_drift is not None
            and fold_drift <= reference_increment + 1.0e-12
        )
        reports.append(
            {
                "name": name,
                "left": cases[left_role].role,
                "right": cases[right_role].role,
                "peak_fracture_energy_drift": peak_drift,
                "fold_fracture_energy_drift": fold_drift,
                "tolerance": reference_increment,
                "passed": passed,
            }
        )
    return {
        "definition": (
            "peak uses the accepted maximum reaction; fold uses the first accepted "
            "target with load_increment < -1e-8"
        ),
        "pairs": reports,
        "passed": all(item["passed"] for item in reports),
    }


def _predictor_robustness_report(
    cases: dict[str, _Case],
    predictor_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_by_name = {item["name"]: item for item in predictor_pairs}
    specifications = (
        ("predoff_dE32", "predon_dE32", "predictor_full"),
        ("predoff_half64", "predon_half64", "predictor_half"),
    )
    outcomes: list[dict[str, Any]] = []
    failed_outcomes: list[dict[str, Any]] = []
    for off_role, on_role, pair_name in specifications:
        off = cases[off_role]
        comparison = pair_by_name[pair_name]
        outcome = {
            "predictor_off_role": off_role,
            "predictor_on_role": on_role,
            "predictor_on_completed": cases[on_role].run_status == "complete",
            "predictor_off_status": off.run_status,
            "common_prefix_response_consistent": comparison["passed"],
            "common_path_targets": comparison["common_path_targets"],
            "maximum_normalised_response_difference": comparison[
                "maximum_normalised_response_difference"
            ],
            "failure_certificate": off.failure_certificate,
        }
        outcomes.append(outcome)
        if off.run_status == "failed":
            failed_outcomes.append(outcome)

    prefixes_consistent = all(
        item["common_prefix_response_consistent"] for item in failed_outcomes
    )
    supported = bool(failed_outcomes) and prefixes_consistent and all(
        item["predictor_on_completed"] for item in failed_outcomes
    )
    all_off_failed = len(failed_outcomes) == len(specifications)
    if supported and all_off_failed:
        result = "predictor_on_completed_while_both_predictor_off_cases_failed_at_minimum"
    elif supported:
        result = "predictor_on_completed_while_predictor_off_failed_at_minimum"
    else:
        result = "certified_failure_exists_but_common_prefix_robustness_is_not_established"
    return {
        "mode": "certified_failure_prefix",
        "outcome": result,
        "supported": supported,
        "all_predictor_off_cases_failed_at_minimum": all_off_failed,
        "certified_predictor_off_failures": len(failed_outcomes),
        "common_failure_prefixes_consistent": prefixes_consistent,
        "cases": outcomes,
    }


def path_sensitivity_report(
    predon_dE32: str | Path,
    predoff_dE32: str | Path,
    predon_half64: str | Path,
    predoff_half64: str | Path,
    *,
    allow_certified_failures: bool = False,
) -> dict[str, Any]:
    """Audit the fixed 2x2 matrix, optionally admitting certified off-prefix failures."""
    directories = {
        "predon_dE32": predon_dE32,
        "predoff_dE32": predoff_dE32,
        "predon_half64": predon_half64,
        "predoff_half64": predoff_half64,
    }
    resolved = [Path(value).resolve() for value in directories.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("the four matrix roles must reference distinct result directories")
    cases = {
        role: _read_case(
            role,
            directories[role],
            allow_certified_failures=allow_certified_failures,
        )
        for role in ROLE_ORDER
    }
    matrix = _validate_matrix(cases)
    tip_tolerance = float(matrix["mesh_h"])
    failure_mode_active = any(case.run_status == "failed" for case in cases.values())

    def predictor_pair(name: str, on_role: str, off_role: str) -> dict[str, Any]:
        on = cases[on_role]
        off = cases[off_role]
        prefix = on.run_status != "complete" or off.run_status != "complete"
        pairs = _common_target_pairs(on, off) if prefix else _same_grid_pairs(on, off)
        return _pair_report(
            name,
            on,
            off,
            pairs,
            response_tolerance=PREDICTOR_RESPONSE_TOLERANCE,
            tip_tolerance=tip_tolerance,
            comparison_scope="certified_accepted_prefix" if prefix else "complete_window",
        )

    predictor_pairs = [
        predictor_pair("predictor_full", "predon_dE32", "predoff_dE32"),
        predictor_pair("predictor_half", "predon_half64", "predoff_half64"),
    ]
    off_prefix = (
        cases["predoff_dE32"].run_status != "complete"
        or cases["predoff_half64"].run_status != "complete"
    )
    increment_pairs = [
        _pair_report(
            "increment_predon",
            cases["predon_dE32"],
            cases["predon_half64"],
            _full_half_pairs(cases["predon_dE32"], cases["predon_half64"]),
            response_tolerance=HALF_INCREMENT_RESPONSE_TOLERANCE,
            tip_tolerance=tip_tolerance,
        ),
        _pair_report(
            "increment_predoff",
            cases["predoff_dE32"],
            cases["predoff_half64"],
            (
                _common_target_pairs(
                    cases["predoff_dE32"],
                    cases["predoff_half64"],
                )
                if off_prefix
                else _full_half_pairs(
                    cases["predoff_dE32"],
                    cases["predoff_half64"],
                )
            ),
            response_tolerance=HALF_INCREMENT_RESPONSE_TOLERANCE,
            tip_tolerance=tip_tolerance,
            comparison_scope=(
                "certified_accepted_prefix" if off_prefix else "complete_window"
            ),
        ),
    ]
    events = {
        role: {
            "run_status": cases[role].run_status,
            **_case_event_report(cases[role]),
        }
        for role in ROLE_ORDER
    }
    predictor_comparisons_passed = all(item["passed"] for item in predictor_pairs)
    increment_comparisons_passed = all(item["passed"] for item in increment_pairs)
    post_peak_passed = all(
        item["peak_confirmation"]["passed"] for item in events.values()
    )
    snapback_observed = all(
        item["negative_displacement"]["passed"] for item in events.values()
    )
    if failure_mode_active:
        event_drift = {
            "applicable": False,
            "passed": None,
            "pairs": [],
            "reason": (
                "peak/fold drift over four completed paths is unavailable because at "
                "least one predictor-off case is a certified failed prefix"
            ),
        }
        robustness = _predictor_robustness_report(cases, predictor_pairs)
        numerical_equivalence_passed = False
        all_passed = False
    else:
        event_drift = {
            "applicable": True,
            **_event_drift_report(
                cases,
                events,
                float(matrix["reference_increment"]),
            ),
        }
        robustness = {
            "mode": "complete_matrix",
            "outcome": "not_applicable_no_certified_predictor_off_failure",
            "supported": None,
            "certified_predictor_off_failures": 0,
        }
        numerical_equivalence_passed = (
            predictor_comparisons_passed and increment_comparisons_passed
        )
        all_passed = (
            numerical_equivalence_passed
            and post_peak_passed
            and snapback_observed
            and event_drift["passed"]
        )
    return {
        "schema_version": PATH_SENSITIVITY_SCHEMA_VERSION,
        "study": "fracture_energy_predictor_by_increment_path_sensitivity",
        "cases": {
            role: {
                "directory": str(cases[role].directory),
                "run_status": cases[role].run_status,
                "completed_hybrid_audit_passed": cases[role].run_status == "complete",
                "persisted_state_audit_passed": True,
                "use_energy_predictor": cases[role].predictor,
                "energy_predictor_disable_after_nominal_path_step": (
                    cases[role].predictor_latch
                ),
                "energy_predictor_policy": (
                    "global_predictor_on"
                    if cases[role].predictor
                    else "global_predictor_off"
                ),
                "target_increment": cases[role].target_increment,
                "configured_path_targets": cases[role].configured_steps,
                "minimum_increment": cases[role].minimum_increment,
                "switch_fracture_energy": cases[role].switch.target,
                "last_accepted_control_target": cases[role].path[-1].target,
                "accepted_path_steps": len(cases[role].path),
                "failure_certificate": cases[role].failure_certificate,
            }
            for role in ROLE_ORDER
        },
        "configuration_matrix": matrix,
        "path_events": events,
        "predictor_sensitivity": {
            "comparison_scope": (
                "certified accepted prefixes" if failure_mode_active else "complete windows"
            ),
            "pairs": predictor_pairs,
            "common_prefixes_passed": predictor_comparisons_passed,
            "full_window_established": not failure_mode_active,
            "passed": predictor_comparisons_passed if not failure_mode_active else False,
        },
        "increment_halving_sensitivity": {
            "common_target_rule": (
                "full phase_step k equals half phase_step 2k for completed pairs; "
                "failed pairs use only tolerance-aligned public accepted targets"
            ),
            "pairs": increment_pairs,
            "predictor_on_complete_window_passed": increment_pairs[0]["passed"],
            "predictor_off_prefix_passed": increment_pairs[1]["passed"],
            "four_path_increment_convergence_established": not failure_mode_active,
            "passed": (
                increment_comparisons_passed if not failure_mode_active else False
            ),
        },
        "peak_and_fold_energy_drift": event_drift,
        "predictor_robustness": robustness,
        "gate_summary": {
            "completed_hybrid_audits_passed": not failure_mode_active,
            "certified_failure_prefixes_passed": failure_mode_active,
            "four_path_completion_passed": not failure_mode_active,
            "numerical_equivalence_passed": numerical_equivalence_passed,
            "predictor_common_prefixes_passed": predictor_comparisons_passed,
            "predictor_on_increment_halving_passed": increment_pairs[0]["passed"],
            "post_peak_gate_passed": post_peak_passed if not failure_mode_active else False,
            "snapback_observed_in_all_cases": (
                snapback_observed if not failure_mode_active else False
            ),
            "peak_and_fold_energy_drift_passed": (
                event_drift["passed"] if not failure_mode_active else False
            ),
        },
        "all_checks_passed": all_passed,
        "interpretation": (
            (
                "The predictor-off minimum-increment failures are authenticated and compared "
                "with predictor-on solutions only over common accepted control targets. This "
                "is a predictor-robustness outcome, not four-path convergence; therefore the "
                "complete-matrix gate remains failed."
            )
            if failure_mode_active
            else (
                "Passing establishes numerical branch robustness over this finite energy "
                "window. It does not establish an interface penetration/deflection mechanism, "
                "mesh convergence, or mirror symmetry."
            )
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphfracture-path-sensitivity",
        description="Strict 2x2 predictor/increment fracture-energy path audit.",
    )
    parser.add_argument("predon_dE32", type=Path)
    parser.add_argument("predoff_dE32", type=Path)
    parser.add_argument("predon_half64", type=Path)
    parser.add_argument("predoff_half64", type=Path)
    parser.add_argument(
        "--allow-certified-failures",
        action="store_true",
        help=(
            "accept predictor-off prefixes only when a strict terminal minimum-increment "
            "failure certificate is present"
        ),
    )
    parser.add_argument("--output", "-o", type=Path, help="exclusive-create JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = path_sensitivity_report(
            args.predon_dE32,
            args.predoff_dE32,
            args.predon_half64,
            args.predoff_half64,
            allow_certified_failures=args.allow_certified_failures,
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
        print(f"path-sensitivity error: {exc}", file=sys.stderr)
        return 2
    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
