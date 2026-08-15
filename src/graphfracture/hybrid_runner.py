"""Restartable displacement/fracture-energy scheduling for :class:`AT2Solver`.

The runner owns a schema-3 accepted-state journal.  Ordinary displacement
preloading and augmented fracture-energy continuation share one scheduler, so
an interruption cannot silently change coordinates, skip a failed target, or
reconstruct the Real-element load factor from an approximate history value.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dolfinx import io
from petsc4py import PETSc

from .damage_control import (
    ControlPhase,
    ControlState,
    FractureEnergyQueue,
    FractureEnergyTarget,
)
from .hybrid_state import DisplacementTarget, HybridSchedulerState

if TYPE_CHECKING:
    from .dolfinx_solver import AT2Solver

__all__ = ["run_fresh_hybrid"]


_METHOD = "hybrid_displacement_fracture_energy"
_COMPLETION_CONDITION = "fracture_energy_queue_exhausted"
_FINITE_RECORD_FIELDS = (
    "displacement",
    "reaction_y",
    "traction_reaction_y",
    "elastic_energy",
    "fracture_energy",
    "total_internal_energy",
    "regularised_crack_length",
    "maximum_damage",
    "minimum_damage_increment",
)


def _finite_or_none(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class _FreshHybridRunner:
    def __init__(self, solver: AT2Solver) -> None:
        self.solver = solver
        self.config = solver.config
        self.comm = solver.comm
        self.output = self.config.output_directory
        self.history: list[dict[str, Any]] = []
        self.interface_history: list[dict[str, Any]] = []
        self.attempt_history: list[dict[str, Any]] = []
        self.control_phase = "displacement"
        self.accepted_step: int | None = None
        self.accepted_displacement: float | None = None
        self.accepted_control: float | None = None
        self.queue: FractureEnergyQueue | None = None
        self.accepted_path_steps = 0
        self.pending_displacements: list[DisplacementTarget] = []
        self.control_state = ControlState.displacement()
        self.switch_accepted_step: int | None = None
        self._resume = False
        self._owns_output = False

    def _collective_write(self, description: str, writer: Any) -> None:
        """Run an output operation on every rank and broadcast any failure.

        Most solver writers are intentionally root-only after completing any
        required collectives.  Without this wrapper, a root filesystem error
        can let other ranks continue into a later collective and deadlock.
        """
        local_error = None
        try:
            writer()
        except Exception as exc:  # noqa: BLE001 - synchronise all I/O failures.
            local_error = (
                f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
            )
        errors = self.comm.allgather(local_error)
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError(
                f"hybrid {description} failed: " + "; ".join(failures)
            )

    def _write_history(self) -> None:
        self._collective_write(
            "history write",
            lambda: self.solver._write_history(self.output, self.history),
        )

    def _write_interface_history(self, *, complete: bool) -> None:
        self._collective_write(
            "interface-history write",
            lambda: self.solver._write_interface_history(
                self.output,
                self.interface_history,
                complete=complete,
            ),
        )

    def _write_attempt_history(self, *, complete: bool) -> None:
        self._collective_write(
            "attempt-history write",
            lambda: self.solver._write_attempt_history(
                self.output,
                self.attempt_history,
                complete=complete,
            ),
        )

    @staticmethod
    def validate_invocation(
        *,
        resume: bool,
        resume_stagger_max_iterations: int | None,
        resume_maximum_subdivisions: int | None,
        resume_minimum_increment: float | None,
    ) -> None:
        overrides = {
            "resume_stagger_max_iterations": resume_stagger_max_iterations,
            "resume_maximum_subdivisions": resume_maximum_subdivisions,
            "resume_minimum_increment": resume_minimum_increment,
        }
        requested = [name for name, value in overrides.items() if value is not None]
        if requested:
            raise ValueError(
                "schema-3 hybrid restart requires the exact configured continuation "
                "controls; unsupported overrides: " + ", ".join(requested)
            )

    def _prepare_output(self, *, resume: bool) -> None:
        if self.comm.rank == 0:
            try:
                if self.output.exists() and not self.output.is_dir():
                    raise NotADirectoryError(
                        f"output path is not a directory: {self.output}"
                    )
                nonempty = self.output.exists() and any(self.output.iterdir())
                if resume:
                    if not nonempty:
                        raise FileNotFoundError(
                            f"resume requested but the output directory is empty: {self.output}"
                        )
                    if (self.output / "completion.json").exists():
                        raise FileExistsError(
                            f"refusing to resume an already completed result: {self.output}"
                        )
                    if not (self.output / "restart" / "checkpoint.json").is_file():
                        raise FileNotFoundError(
                            "resume requested but no committed restart/checkpoint.json exists"
                        )
                elif not nonempty:
                    self.output.mkdir(parents=True, exist_ok=True)
                output_error: tuple[str, str] | None = None
            except OSError as exc:
                nonempty = False
                output_error = (type(exc).__name__, str(exc))
        else:
            nonempty = None
            output_error = None
        nonempty, output_error = self.comm.bcast((nonempty, output_error), root=0)
        if output_error is not None:
            error_type, message = output_error
            if error_type == "FileExistsError":
                raise FileExistsError(message)
            if error_type == "FileNotFoundError":
                raise FileNotFoundError(message)
            if error_type == "NotADirectoryError":
                raise NotADirectoryError(message)
            raise RuntimeError(
                f"output directory preparation failed: {error_type}: {message}"
            )
        if nonempty and not resume:
            raise FileExistsError(
                f"refusing to mix a fresh hybrid run with existing output files: {self.output}"
            )
        self.comm.barrier()
        self._owns_output = not resume

    def _write_method_status(
        self,
        status: str,
        *,
        failure: BaseException | None = None,
    ) -> None:
        pending = (
            len(self.queue.pending)
            if self.queue is not None
            else len(self.pending_displacements)
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "method": _METHOD,
            "status": status,
            "resume_supported": True,
            "checkpoint_schema": 3,
            "checkpoint_policy": (
                "accepted-state schema-3 journal; same config, implementation, "
                "runtime, MPI size, and partition required"
            ),
            "control_phase": self.control_phase,
            "accepted_step": self.accepted_step,
            "accepted_displacement": self.accepted_displacement,
            "accepted_control": self.accepted_control,
            "accepted_path_steps": self.accepted_path_steps,
            "configured_path_targets": self.config.path_control.steps,
            "pending_control_targets": pending,
            "completion_condition": _COMPLETION_CONDITION,
        }
        if failure is not None:
            payload["failure_type"] = type(failure).__name__
            payload["failure_message"] = str(failure)
        self._collective_write(
            "method-status write",
            lambda: (
                self.solver._atomic_write_json(
                    self.output / "method_status.json",
                    payload,
                )
                if self.comm.rank == 0
                else None
            ),
        )

    def _scheduler_state(self) -> HybridSchedulerState:
        """Return the exact immutable scheduler state for one checkpoint."""
        if self.control_phase != self.control_state.phase.value:
            raise RuntimeError("hybrid string/control-state phases disagree")
        if self.control_phase == ControlPhase.DISPLACEMENT.value:
            if self.accepted_path_steps != 0:
                raise RuntimeError("displacement phase cannot contain accepted path steps")
            state = ControlState.displacement()
            queue = None
            switch_step = None
            pending = tuple(self.pending_displacements)
            phase_step = int(self.accepted_step)
        elif self.control_phase == ControlPhase.FRACTURE_ENERGY.value:
            if self.queue is None or self.switch_accepted_step is None:
                raise RuntimeError("fracture-energy scheduler state is incomplete")
            if self.accepted_path_steps != int(self.accepted_step) - self.switch_accepted_step:
                raise RuntimeError("fracture-energy phase step is not contiguous")
            state = self.control_state
            queue = self.queue
            switch_step = self.switch_accepted_step
            pending = ()
            phase_step = self.accepted_path_steps
        else:
            raise RuntimeError(f"unsupported hybrid control phase: {self.control_phase}")
        return HybridSchedulerState(
            state=state,
            reference_displacement=self.config.path_control.switch_displacement,
            switch_accepted_step=switch_step,
            phase_step=phase_step,
            pending_displacements=pending,
            fracture_energy_queue=queue,
        )

    def _save_checkpoint(self) -> None:
        state = self.solver._snapshot_state()
        scheduler = None
        preflight_error: BaseException | None = None
        try:
            if self.accepted_step is None or self.accepted_displacement is None:
                raise RuntimeError(
                    "hybrid accepted state is unavailable for checkpointing"
                )
            scheduler = self._scheduler_state()
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            preflight_error = exc
        if scheduler is None and preflight_error is None:
            preflight_error = RuntimeError(
                "checkpoint scheduler preflight returned no state"
            )
        self._raise_collective_stage_error(
            "checkpoint scheduler preflight",
            preflight_error,
            state,
        )
        if scheduler is None:  # pragma: no cover - collective gate invariant.
            raise RuntimeError("checkpoint scheduler preflight returned no state")
        self.solver._save_hybrid_restart_checkpoint(
            self.output,
            history=self.history,
            interface_history=self.interface_history,
            attempt_history=self.attempt_history,
            accepted_step=self.accepted_step,
            accepted_displacement=self.accepted_displacement,
            scheduler_state=scheduler,
        )

    def _write_progress_outputs(self) -> None:
        self._write_history()
        self._write_interface_history(complete=False)
        self._write_attempt_history(complete=False)
        self._write_method_status("running")

    def _append_interface_record(
        self,
        *,
        accepted_step: int,
        scheduled_step: int,
        displacement: float,
        subdivision_level: int,
    ) -> None:
        """Synchronise root-only interface diagnostics after their graph gather."""
        record = None
        local_error: BaseException | None = None
        try:
            record = self.solver._interface_history_record(
                accepted_step=accepted_step,
                scheduled_step=scheduled_step,
                displacement=displacement,
                subdivision_level=subdivision_level,
            )
            if record is not None and not isinstance(record, dict):
                raise TypeError("interface history record must be a mapping or None")
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            local_error = exc
        report = (
            None
            if local_error is None
            else (type(local_error).__name__, str(local_error))
        )
        reports = self.comm.allgather(report)
        failures = [
            f"rank {rank}: {item[0]}: {item[1]}"
            for rank, item in enumerate(reports)
            if item is not None
        ]
        if failures:
            raise RuntimeError(
                "hybrid interface-history record failed: " + "; ".join(failures)
            )
        if record is not None:
            self.interface_history.append(record)

    def _transition_to_fracture_energy(self) -> None:
        """Commit the coordinate switch before attempting the first path step."""
        if len(self.history) < 2 or self.accepted_step is None:
            raise RuntimeError("path control requires an accepted preload interval")
        state = self.solver._snapshot_state()
        self._require_collective_callable(
            "path-control initialisation",
            self.solver._initialise_path_control_problem,
        )
        path_problem = None
        initialise_error: BaseException | None = None
        try:
            path_problem = self.solver._initialise_path_control_problem()
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            initialise_error = exc
        try:
            self._raise_collective_stage_error(
                "path-control initialisation",
                initialise_error,
                state,
            )
        except Exception:
            self.solver._path_control_problem = None
            raise
        missing_problem_error = (
            RuntimeError("path-control initialisation returned no problem")
            if path_problem is None
            else None
        )
        try:
            self._raise_collective_stage_error(
                "path-control initialisation result",
                missing_problem_error,
                state,
            )
        except Exception:
            self.solver._path_control_problem = None
            raise

        previous_energy = float(self.history[-2]["fracture_energy"])
        switch_energy = float(self.history[-1]["fracture_energy"])
        measured_energy = math.nan
        energy_error: BaseException | None = None
        try:
            measured_energy = float(path_problem.fracture_energy)
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            energy_error = exc
        try:
            self._raise_collective_stage_error(
                "path-control switch energy",
                energy_error,
                state,
            )
        except Exception:
            self.solver._path_control_problem = None
            raise
        tolerance = max(
            1.0e-12 * max(1.0, abs(switch_energy), abs(measured_energy)),
            self.config.path_control.control_tolerance
            * self.config.path_control.minimum_increment,
        )
        mismatch_error = (
            None
            if math.isclose(
                measured_energy,
                switch_energy,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            else RuntimeError(
                "path-control switch energy disagrees with the accepted preload history"
            )
        )
        try:
            self._raise_collective_stage_error(
                "path-control switch-energy certificate",
                mismatch_error,
                state,
            )
        except Exception:
            self.solver._path_control_problem = None
            raise

        candidate_state = None
        candidate_queue = None
        candidate_error: BaseException | None = None
        try:
            candidate_state = ControlState.displacement().switch_to_fracture_energy(
                previous_fracture_energy=previous_energy,
                current_fracture_energy=switch_energy,
            )
            candidate_queue = FractureEnergyQueue.from_increment(
                accepted_value=switch_energy,
                increment=self.config.path_control.target_increment,
                count=self.config.path_control.steps,
            )
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            candidate_error = exc
        try:
            self._raise_collective_stage_error(
                "path-control scheduler transition",
                candidate_error,
                state,
            )
        except Exception:
            self.solver._path_control_problem = None
            raise
        missing_candidate_error = (
            RuntimeError("path-control scheduler transition returned no state")
            if candidate_state is None or candidate_queue is None
            else None
        )
        try:
            self._raise_collective_stage_error(
                "path-control scheduler result",
                missing_candidate_error,
                state,
            )
        except Exception:
            self.solver._path_control_problem = None
            raise
        candidate_payload = (
            candidate_state.to_payload(),
            candidate_queue.to_payload(),
            measured_energy,
        )
        candidate_payloads = self.comm.allgather(candidate_payload)
        if any(item != candidate_payloads[0] for item in candidate_payloads[1:]):
            self.solver._restore_state(state)
            self.solver._path_control_problem = None
            raise RuntimeError(
                "hybrid path-control switch state differs across MPI ranks"
            )
        self.control_state = candidate_state
        self.queue = candidate_queue
        self.control_phase = ControlPhase.FRACTURE_ENERGY.value
        self.switch_accepted_step = self.accepted_step
        self.accepted_path_steps = 0
        self.accepted_control = switch_energy
        self._save_checkpoint()
        self._write_progress_outputs()

    @staticmethod
    def _certify_record(record: dict[str, Any]) -> None:
        invalid = []
        for name in _FINITE_RECORD_FIELDS:
            value = record[name]
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                invalid.append(name)
        for name in (
            "external_work",
            "energy_balance_residual",
            "energy_balance_relative",
        ):
            value = record[name]
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                invalid.append(name)
        if invalid:
            raise RuntimeError(
                "accepted hybrid record contains non-finite values: "
                + ", ".join(invalid)
            )

    @staticmethod
    def _failed_load_info(maximum_iterations: int) -> dict[str, Any]:
        return {
            "iterations": maximum_iterations,
            "error": math.inf,
            "converged": False,
            "damage_snes_iterations": -1,
            "damage_snes_reason": -1,
            "elastic_ksp_reason": -1,
            "aitken_accepted_iterations": -1,
            "final_aitken_relaxation": None,
        }

    def _raise_collective_stage_error(
        self,
        description: str,
        error: BaseException | None,
        state: dict[str, Any],
    ) -> None:
        """Synchronise a local programming/schema error before any next collective."""
        report = (
            None
            if error is None
            else (type(error).__module__, type(error).__name__, str(error))
        )
        reports = self.comm.allgather(report)
        failures = [item for item in reports if item is not None]
        if not failures:
            return
        self.solver._restore_state(state)
        if (
            error is not None
            and len(failures) == self.comm.size
            and all(item == failures[0] for item in failures)
        ):
            raise error
        rendered = "; ".join(
            f"rank {rank}: {item[1]}: {item[2]}"
            for rank, item in enumerate(reports)
            if item is not None
        )
        raise RuntimeError(f"hybrid {description} differs across MPI ranks: {rendered}")

    def _require_collective_callable(self, description: str, callback: Any) -> None:
        """Reject rank-specific monkeypatch/configuration before a collective call."""
        target = getattr(callback, "__func__", callback)
        code = getattr(target, "__code__", None)
        fingerprint = (
            getattr(target, "__module__", None),
            getattr(target, "__qualname__", None),
            code.co_code if code is not None else None,
        )
        fingerprints = self.comm.allgather(fingerprint)
        if any(item != fingerprints[0] for item in fingerprints[1:]):
            raise RuntimeError(
                f"hybrid {description} callable differs across MPI ranks"
            )

    def _synchronise_solve_outcome(
        self,
        description: str,
        *,
        outcome: str,
        payload: dict[str, Any],
        failure: BaseException | None,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], BaseException | None]:
        """Choose one collective accept/reject branch and one diagnostic payload."""
        report = {
            "outcome": outcome,
            "failure_type": type(failure).__name__ if failure is not None else None,
            "failure_message": str(failure) if failure is not None else None,
        }
        reports = self.comm.allgather(report)
        outcomes = {item["outcome"] for item in reports}
        if len(outcomes) != 1:
            self.solver._restore_state(state)
            rendered = "; ".join(
                f"rank {rank}: {item['outcome']}"
                for rank, item in enumerate(reports)
            )
            raise RuntimeError(
                f"hybrid {description} outcome differs across MPI ranks: {rendered}"
            )
        canonical = self.comm.bcast(
            dict(payload) if self.comm.rank == 0 else None,
            root=0,
        )
        if outcome != "exception":
            return canonical, None
        root_report = reports[0]
        canonical_failure = RuntimeError(
            f"{root_report['failure_type']}: {root_report['failure_message']}"
        )
        return canonical, canonical_failure

    def _append_displacement_attempt(
        self,
        *,
        target: float,
        scheduled_step: int,
        subdivision_level: int,
        solve_info: dict[str, Any],
        failure: BaseException | None,
        can_subdivide: bool,
    ) -> None:
        accepted = float(self.accepted_displacement)
        raw_error = _finite_or_none(solve_info.get("error"))
        self.attempt_history.append(
            {
                "attempt": len(self.attempt_history) + 1,
                "accepted_step_before_attempt": int(self.accepted_step),
                "control_phase": "displacement",
                "accepted_control": accepted,
                "target_control": target,
                "trial_displacement": target,
                "accepted_displacement": accepted,
                "target_displacement": target,
                "load_increment": target - accepted,
                "scheduled_step": scheduled_step,
                "subdivision_level": subdivision_level,
                "failure_type": (
                    type(failure).__name__ if failure is not None else "nonconverged"
                ),
                "failure_message": (
                    str(failure)
                    if failure is not None
                    else "alternate minimisation did not satisfy convergence gates"
                ),
                "iterations": int(solve_info.get("iterations", -1)),
                "error": raw_error,
                "damage_snes_iterations": int(
                    solve_info.get("damage_snes_iterations", -1)
                ),
                "damage_snes_reason": int(solve_info.get("damage_snes_reason", -1)),
                "elastic_ksp_reason": int(solve_info.get("elastic_ksp_reason", -1)),
                "aitken_accepted_iterations": int(
                    solve_info.get("aitken_accepted_iterations", -1)
                ),
                "final_aitken_relaxation": solve_info.get(
                    "final_aitken_relaxation"
                ),
                "will_subdivide": can_subdivide,
            }
        )

    def _append_path_attempt(
        self,
        *,
        target: FractureEnergyTarget,
        scheduled_step: int,
        path_increment: float,
        info: dict[str, Any],
        failure: BaseException | None,
        can_subdivide: bool,
    ) -> None:
        trial_displacement = _finite_or_none(info.get("displacement"))
        accepted_displacement = float(self.accepted_displacement)
        relative_values = [
            _finite_or_none(info.get(name))
            for name in (
                "control_residual_relative",
                "damage_kkt_relative",
                "mechanical_residual_relative",
            )
        ]
        finite_relative = [value for value in relative_values if value is not None]
        self.attempt_history.append(
            {
                "attempt": len(self.attempt_history) + 1,
                "accepted_step_before_attempt": int(self.accepted_step),
                "control_phase": "fracture_energy",
                "accepted_control": float(self.queue.accepted_value),
                "target_control": target.value,
                "trial_displacement": trial_displacement,
                "accepted_displacement": accepted_displacement,
                "target_displacement": trial_displacement,
                "load_increment": (
                    trial_displacement - accepted_displacement
                    if trial_displacement is not None
                    else None
                ),
                "control_increment": path_increment,
                "scheduled_step": scheduled_step,
                "subdivision_level": target.subdivision_level,
                "failure_type": (
                    type(failure).__name__ if failure is not None else "uncertified"
                ),
                "failure_message": (
                    str(failure)
                    if failure is not None
                    else "augmented solve did not satisfy every physical certificate"
                ),
                "iterations": int(info.get("iterations", -1)),
                "error": max(finite_relative) if finite_relative else None,
                "path_snes_reason": int(info.get("snes_reason", -1)),
                "path_ksp_reason": int(info.get("ksp_reason", -1)),
                "load_factor_bound_status": info.get("load_factor_bound_status"),
                "will_subdivide": can_subdivide,
            }
        )

    def _path_nominal_index(self, target: float, switch_energy: float) -> int:
        ratio = (target - switch_energy) / self.config.path_control.target_increment
        tolerance = 1.0e-12 * max(1.0, abs(ratio))
        return min(
            self.config.path_control.steps,
            max(1, math.ceil(ratio - tolerance)),
        )

    def _write_hybrid_completion(self) -> None:
        if self.queue is None or self.queue.pending:
            raise RuntimeError("hybrid completion requires an exhausted energy queue")
        final = self.history[-1]
        payload = {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "accepted_load_steps": len(self.history) - 1,
            "final_displacement": final["displacement"],
            "all_steps_converged": all(
                bool(record["stagger_converged"]) for record in self.history
            ),
            "effective_continuation_controls": self.solver._continuation_controls(
                self.solver._effective_loading
            ),
            "continuation_sessions": len(self.solver._continuation_sessions),
            "final_control_phase": final["control_phase"],
            "final_control_target": final["control_target"],
            "final_control_value": final["control_value"],
            "completion_condition": _COMPLETION_CONDITION,
            "control_targets_exhausted": True,
            "accepted_path_steps": self.accepted_path_steps,
            "configured_path_targets": self.config.path_control.steps,
        }
        self._collective_write(
            "completion write",
            lambda: (
                self.solver._atomic_write_json(
                    self.output / "completion.json",
                    payload,
                )
                if self.comm.rank == 0
                else None
            ),
        )

    def _accept_displacement_target(
        self,
        *,
        target: float,
        subdivision_level: int,
        scheduled_step: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        accepted = float(self.accepted_displacement)
        increment = target - accepted
        state = self.solver._snapshot_state()
        failure: BaseException | None = None
        solve_info = self._failed_load_info(
            self.solver._effective_loading.stagger_max_iterations
        )
        record: dict[str, Any] | None = None
        fatal_error: BaseException | None = None
        self._require_collective_callable(
            "displacement solve",
            self.solver._solve_load_step,
        )
        try:
            solve_info = self.solver._solve_load_step(target)
        except (PETSc.Error, RuntimeError) as exc:
            failure = exc
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            fatal_error = exc
        self._raise_collective_stage_error(
            "displacement solve contract",
            fatal_error,
            state,
        )

        decision_error: BaseException | None = None
        converged = False
        try:
            if not isinstance(solve_info, dict):
                raise TypeError("displacement solve info must be a mapping")
            converged = bool(solve_info["converged"])
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            decision_error = exc
        self._raise_collective_stage_error(
            "displacement solve decision",
            decision_error,
            state,
        )
        outcome = (
            "exception"
            if failure is not None
            else ("accepted" if converged else "rejected")
        )
        solve_info, failure = self._synchronise_solve_outcome(
            "displacement solve",
            outcome=outcome,
            payload=solve_info,
            failure=failure,
            state=state,
        )

        if outcome == "accepted":
            record_error: BaseException | None = None
            self._require_collective_callable(
                "history record",
                self.solver._record,
            )
            try:
                record = self.solver._record(
                    int(self.accepted_step) + 1,
                    target,
                    solve_info,
                    scheduled_step=scheduled_step,
                    subdivision_level=subdivision_level,
                    load_increment=increment,
                )
                if not isinstance(record, dict):
                    raise TypeError("displacement history record must be a mapping")
                self.solver._update_energy_balance(self.history, record)
                self._certify_record(record)
            except Exception as exc:  # noqa: BLE001 - synchronised below.
                record_error = exc
            # History/schema/programming errors are not nonlinear path
            # failures and must never change the continuation branch.
            self._raise_collective_stage_error(
                "displacement history record",
                record_error,
                state,
            )

        if record is not None:
            return record, True

        self.solver._restore_state(state)
        loading = self.solver._effective_loading
        can_subdivide = (
            loading.adaptive
            and subdivision_level < loading.maximum_subdivisions
            and 0.5 * increment >= loading.minimum_increment
        )
        self._append_displacement_attempt(
            target=target,
            scheduled_step=scheduled_step,
            subdivision_level=subdivision_level,
            solve_info=solve_info,
            failure=failure,
            can_subdivide=can_subdivide,
        )
        self._write_attempt_history(complete=False)
        if not can_subdivide:
            # The unaccepted head remains pending in the checkpoint.  A later
            # resume must retry it rather than mistaking the queue for done.
            self._save_checkpoint()
            message = (
                "hybrid preload failed at displacement "
                f"{target:.6e}; minimum attempted increment was {increment:.6e}"
            )
            if failure is not None:
                raise RuntimeError(message) from failure
            raise RuntimeError(message)
        return None, False

    def _accept_path_target(
        self,
        *,
        target: FractureEnergyTarget,
        scheduled_step: int,
        nominal_path_step: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        if self.queue is None:
            raise RuntimeError("fracture-energy queue has not been initialised")
        path_increment = target.value - self.queue.accepted_value
        accepted_displacement = float(self.accepted_displacement)
        state = self.solver._snapshot_state()
        failure: BaseException | None = None
        info: dict[str, Any] = {}
        record: dict[str, Any] | None = None
        fatal_error: BaseException | None = None
        self._require_collective_callable(
            "fracture-energy solve",
            self.solver._solve_path_control_step,
        )
        use_energy_predictor = (
            self.config.path_control.energy_predictor_enabled_for_step(
                nominal_path_step
            )
        )
        try:
            info = dict(
                self.solver._solve_path_control_step(
                    target.value,
                    use_energy_predictor=use_energy_predictor,
                )
            )
        except (PETSc.Error, RuntimeError) as exc:
            failure = exc
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            fatal_error = exc
        self._raise_collective_stage_error(
            "fracture-energy solve contract",
            fatal_error,
            state,
        )

        decision_error: BaseException | None = None
        certified = False
        try:
            if not isinstance(info, dict):
                raise TypeError("fracture-energy solve info must be a mapping")
            certified = bool(info.get("certified", False))
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            decision_error = exc
        self._raise_collective_stage_error(
            "fracture-energy solve decision",
            decision_error,
            state,
        )
        outcome = (
            "exception"
            if failure is not None
            else ("accepted" if certified else "rejected")
        )
        info, failure = self._synchronise_solve_outcome(
            "fracture-energy solve",
            outcome=outcome,
            payload=info,
            failure=failure,
            state=state,
        )

        if outcome == "accepted":
            adapter_error: BaseException | None = None
            solve_info: dict[str, Any] | None = None
            control_info: dict[str, Any] | None = None
            displacement = math.nan
            try:
                displacement = float(info["displacement"])
                solve_info, control_info = self.solver._path_solve_info_for_record(
                    info,
                    phase_step=self.accepted_path_steps + 1,
                    path_increment=path_increment,
                )
                if not isinstance(solve_info, dict) or not isinstance(
                    control_info,
                    dict,
                ):
                    raise TypeError(
                        "fracture-energy record adaptation must return two mappings"
                    )
            except Exception as exc:  # noqa: BLE001 - synchronised below.
                adapter_error = exc
            self._raise_collective_stage_error(
                "fracture-energy record adaptation",
                adapter_error,
                state,
            )
            if solve_info is None or control_info is None:  # pragma: no cover
                raise RuntimeError("fracture-energy record adaptation returned no data")

            record_error: BaseException | None = None
            self._require_collective_callable(
                "history record",
                self.solver._record,
            )
            try:
                record = self.solver._record(
                    int(self.accepted_step) + 1,
                    displacement,
                    solve_info,
                    scheduled_step=scheduled_step,
                    subdivision_level=target.subdivision_level,
                    load_increment=displacement - accepted_displacement,
                    control_info=control_info,
                )
                if not isinstance(record, dict):
                    raise TypeError("fracture-energy history record must be a mapping")
                self.solver._update_energy_balance(self.history, record)
                self._certify_record(record)
            except Exception as exc:  # noqa: BLE001 - synchronised below.
                record_error = exc
            # A certified nonlinear state followed by a recording or schema
            # failure is fatal.  Refining it would select a new irreversible
            # branch for a non-physical reason.
            self._raise_collective_stage_error(
                "fracture-energy history record",
                record_error,
                state,
            )

        if record is not None:
            return record, True

        # The adapter already restores every path and physical field.  This
        # idempotent outer restore also covers a certified nonlinear solve
        # rejected by the independent finite-record gate above.
        self.solver._restore_state(state)
        controls = self.config.path_control
        can_subdivide = (
            controls.adaptive
            and target.subdivision_level < controls.maximum_subdivisions
            and self.solver._hybrid_control_increment_meets_minimum(
                0.5 * path_increment,
                minimum_increment=controls.minimum_increment,
                coordinate_scale=max(
                    abs(self.queue.accepted_value),
                    abs(target.value),
                ),
            )
        )
        self._append_path_attempt(
            target=target,
            scheduled_step=scheduled_step,
            path_increment=path_increment,
            info=info,
            failure=failure,
            can_subdivide=can_subdivide,
        )
        self._write_attempt_history(complete=False)
        if not can_subdivide:
            # Preserve the failed absolute energy target as the queue head
            # before reporting exhaustion of the configured refinements.
            self._save_checkpoint()
            message = (
                "hybrid path solve failed at absolute fracture energy "
                f"{target.value:.6e}; minimum attempted control increment was "
                f"{path_increment:.6e}"
            )
            if failure is not None:
                raise RuntimeError(message) from failure
            raise RuntimeError(message)
        return None, False

    def _initialise_fresh(self) -> None:
        self._prepare_output(resume=False)
        self._write_method_status("initialising")
        self._collective_write(
            "metadata write",
            lambda: self.solver._write_metadata(self.output),
        )
        self.solver._run_hydrogen_precharge(self.output)

        initial_state = self.solver._snapshot_state()
        initial_info = self._failed_load_info(
            self.solver._effective_loading.stagger_max_iterations
        )
        failure: BaseException | None = None
        fatal_error: BaseException | None = None
        self._require_collective_callable(
            "initial displacement solve",
            self.solver._solve_load_step,
        )
        try:
            initial_info = self.solver._solve_load_step(0.0)
        except (PETSc.Error, RuntimeError) as exc:
            failure = exc
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            fatal_error = exc
        self._raise_collective_stage_error(
            "initial displacement solve contract",
            fatal_error,
            initial_state,
        )
        decision_error: BaseException | None = None
        converged = False
        try:
            if not isinstance(initial_info, dict):
                raise TypeError("initial displacement solve info must be a mapping")
            converged = bool(initial_info["converged"])
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            decision_error = exc
        self._raise_collective_stage_error(
            "initial displacement solve decision",
            decision_error,
            initial_state,
        )
        outcome = (
            "exception"
            if failure is not None
            else ("accepted" if converged else "rejected")
        )
        initial_info, failure = self._synchronise_solve_outcome(
            "initial displacement solve",
            outcome=outcome,
            payload=initial_info,
            failure=failure,
            state=initial_state,
        )
        if outcome != "accepted":
            self.solver._restore_state(initial_state)
            message = "pre-crack phase-field equilibration did not converge"
            if failure is not None:
                raise RuntimeError(message) from failure
            raise RuntimeError(message)

        initial_record: dict[str, Any] | None = None
        record_error: BaseException | None = None
        self._require_collective_callable("history record", self.solver._record)
        try:
            initial_record = self.solver._record(
                0,
                0.0,
                initial_info,
                scheduled_step=0,
                subdivision_level=0,
                load_increment=0.0,
            )
            if not isinstance(initial_record, dict):
                raise TypeError("initial history record must be a mapping")
            self._certify_record(initial_record)
        except Exception as exc:  # noqa: BLE001 - synchronised below.
            record_error = exc
        self._raise_collective_stage_error(
            "initial history record",
            record_error,
            initial_state,
        )
        if initial_record is None:
            raise RuntimeError("initial accepted state has no history record")
        self.history = [initial_record]
        self.accepted_step = 0
        self.accepted_displacement = 0.0
        self.accepted_control = 0.0
        self._append_interface_record(
            accepted_step=0,
            scheduled_step=0,
            displacement=0.0,
            subdivision_level=0,
        )

        nominal_increment = (
            self.config.loading.maximum_displacement / self.config.loading.steps
        )
        switch_index = round(
            self.config.path_control.switch_displacement / nominal_increment
        )
        self.pending_displacements = [
            DisplacementTarget(
                displacement=scheduled_step * nominal_increment,
                subdivision_level=0,
                scheduled_step=scheduled_step,
            )
            for scheduled_step in range(1, switch_index + 1)
        ]
        self._save_checkpoint()
        self._write_progress_outputs()

    def _initialise_resume(self) -> None:
        self._prepare_output(resume=True)
        (
            self.history,
            self.interface_history,
            self.attempt_history,
            scheduler,
            self.accepted_step,
            self.accepted_displacement,
        ) = self.solver._load_hybrid_restart_checkpoint(self.output)
        self.control_state = scheduler.state
        self.control_phase = scheduler.state.phase.value
        self.switch_accepted_step = scheduler.switch_accepted_step
        self.accepted_path_steps = (
            scheduler.phase_step
            if scheduler.state.phase is ControlPhase.FRACTURE_ENERGY
            else 0
        )
        self.pending_displacements = list(scheduler.pending_displacements)
        self.queue = scheduler.fracture_energy_queue
        if self.control_phase == ControlPhase.DISPLACEMENT.value:
            self.accepted_control = float(self.accepted_displacement)
        else:
            if self.queue is None:
                raise RuntimeError("restored fracture-energy scheduler has no queue")
            self.accepted_control = float(self.queue.accepted_value)
        self._owns_output = True
        self._write_progress_outputs()

    def _continue_displacement(
        self,
        damage_file: io.XDMFFile,
        displacement_file: io.XDMFFile,
    ) -> None:
        while self.control_phase == ControlPhase.DISPLACEMENT.value:
            if not self.pending_displacements:
                raise RuntimeError(
                    "hybrid displacement scheduler reached the switch without a target"
                )
            pending = self.pending_displacements[0]
            accepted_before = float(self.accepted_displacement)
            record, accepted = self._accept_displacement_target(
                target=pending.displacement,
                subdivision_level=pending.subdivision_level,
                scheduled_step=pending.scheduled_step,
            )
            if not accepted:
                midpoint = accepted_before + 0.5 * (
                    pending.displacement - accepted_before
                )
                next_level = pending.subdivision_level + 1
                self.pending_displacements[0:1] = [
                    DisplacementTarget(
                        midpoint,
                        next_level,
                        pending.scheduled_step,
                    ),
                    DisplacementTarget(
                        pending.displacement,
                        next_level,
                        pending.scheduled_step,
                    ),
                ]
                self._save_checkpoint()
                self._write_progress_outputs()
                continue
            if record is None:
                raise RuntimeError("accepted preload step has no history record")

            self.pending_displacements.pop(0)
            self.accepted_step = int(self.accepted_step) + 1
            self.accepted_displacement = pending.displacement
            self.accepted_control = pending.displacement
            self.history.append(record)
            self._append_interface_record(
                accepted_step=self.accepted_step,
                scheduled_step=pending.scheduled_step,
                displacement=pending.displacement,
                subdivision_level=pending.subdivision_level,
            )
            if self.pending_displacements:
                self._save_checkpoint()
                self._write_progress_outputs()
            else:
                # There is no invalid displacement-phase checkpoint with an
                # empty queue: the accepted switch is atomically published in
                # the new coordinate at phase_step zero.
                self._transition_to_fracture_energy()
            if (
                self.accepted_step % self.config.output.write_every == 0
                or not self.pending_displacements
            ):
                time = float(self.accepted_step)
                damage_file.write_function(self.solver.d, time)
                displacement_file.write_function(self.solver.u, time)

    def _continue_fracture_energy(
        self,
        damage_file: io.XDMFFile,
        displacement_file: io.XDMFFile,
    ) -> None:
        if self.queue is None or self.switch_accepted_step is None:
            raise RuntimeError("fracture-energy continuation state is incomplete")
        switch_index = round(
            self.config.path_control.switch_displacement
            / (self.config.loading.maximum_displacement / self.config.loading.steps)
        )
        switch_energy = float(
            self.history[self.switch_accepted_step]["fracture_energy"]
        )
        while self.queue.pending:
            target = self.queue.next_target
            if target is None:
                raise RuntimeError("non-empty energy queue has no next target")
            nominal_index = self._path_nominal_index(target.value, switch_energy)
            scheduled_step = switch_index + nominal_index
            record, accepted = self._accept_path_target(
                target=target,
                scheduled_step=scheduled_step,
                nominal_path_step=nominal_index,
            )
            if not accepted:
                self.queue = self.queue.subdivide_failed_target()
                self._save_checkpoint()
                self._write_progress_outputs()
                continue
            if record is None:
                raise RuntimeError("accepted path step has no history record")
            self.accepted_step = int(self.accepted_step) + 1
            self.accepted_path_steps += 1
            self.accepted_displacement = float(record["displacement"])
            self.queue = self.queue.accept_next()
            self.accepted_control = self.queue.accepted_value
            self.history.append(record)
            self._append_interface_record(
                accepted_step=self.accepted_step,
                scheduled_step=scheduled_step,
                displacement=self.accepted_displacement,
                subdivision_level=target.subdivision_level,
            )
            # Queue exhaustion is itself an accepted partial checkpoint.  If
            # completion publication is interrupted, resume performs no new
            # nonlinear solve and only finishes the output certificates.
            self._save_checkpoint()
            self._write_progress_outputs()
            if (
                self.accepted_step % self.config.output.write_every == 0
                or not self.queue.pending
            ):
                time = float(self.accepted_step)
                damage_file.write_function(self.solver.d, time)
                displacement_file.write_function(self.solver.u, time)

    def _run(self, *, resume: bool) -> list[dict[str, Any]]:
        self._resume = resume
        if resume:
            self._initialise_resume()
        else:
            self._initialise_fresh()

        field_paths = self.solver._field_output_paths(self.output, resume=resume)
        damage_file = io.XDMFFile(self.comm, field_paths["damage"], "w")
        displacement_file = io.XDMFFile(
            self.comm,
            field_paths["displacement"],
            "w",
        )

        try:
            damage_file.write_mesh(self.solver.domain)
            displacement_file.write_mesh(self.solver.domain)
            accepted_time = float(self.accepted_step)
            damage_file.write_function(self.solver.d, accepted_time)
            displacement_file.write_function(self.solver.u, accepted_time)
            with io.XDMFFile(
                self.comm,
                field_paths["material"],
                "w",
            ) as material_file:
                material_file.write_mesh(self.solver.domain)
                material_file.write_function(self.solver.Gc0, accepted_time)
                material_file.write_function(self.solver.Gc, accepted_time)
                material_file.write_function(self.solver.diffusivity, accepted_time)
                material_file.write_function(self.solver.trap_density, accepted_time)

            self._continue_displacement(damage_file, displacement_file)
            self._continue_fracture_energy(damage_file, displacement_file)
        finally:
            damage_file.close()
            displacement_file.close()
            self._write_history()
            self._write_interface_history(complete=False)
            self._write_attempt_history(complete=False)

        if self.queue is None or self.queue.pending:
            raise RuntimeError("hybrid runner stopped before exhausting the energy queue")
        self._collective_write(
            "graph-metrics write",
            lambda: self.solver._write_graph_metrics(self.output),
        )
        self._write_interface_history(complete=True)
        self._write_attempt_history(complete=True)
        # Publish method status before completion so completion.json remains
        # the final, atomic certificate.  A crash can leave a complete method
        # status without a completion marker, but never a generic-looking
        # completion that omits queue exhaustion.
        self._write_method_status("complete")
        self._write_hybrid_completion()
        self.comm.barrier()
        return self.history

    def run(self, *, resume: bool = False) -> list[dict[str, Any]]:
        try:
            return self._run(resume=resume)
        except Exception as exc:
            if self._owns_output:
                self._write_method_status("failed", failure=exc)
            raise


def run_fresh_hybrid(
    solver: AT2Solver,
    *,
    resume: bool = False,
    resume_stagger_max_iterations: int | None = None,
    resume_maximum_subdivisions: int | None = None,
    resume_minimum_increment: float | None = None,
) -> list[dict[str, Any]]:
    """Run or resume the schema-3 hybrid schedule for one configured solver."""
    _FreshHybridRunner.validate_invocation(
        resume=resume,
        resume_stagger_max_iterations=resume_stagger_max_iterations,
        resume_maximum_subdivisions=resume_maximum_subdivisions,
        resume_minimum_increment=resume_minimum_increment,
    )
    return _FreshHybridRunner(solver).run(resume=resume)
