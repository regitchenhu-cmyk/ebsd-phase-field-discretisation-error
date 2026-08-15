"""Pure scheduling primitives for switching to fracture-energy control.

This module deliberately contains no finite-element or displacement-solver
logic.  It only owns the one-way control phase, scalar fracture-energy
targets, and dyadic refinement in that control coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "ControlResidualCertificate",
    "ControlPhase",
    "ControlState",
    "FractureEnergyQueue",
    "FractureEnergyTarget",
    "finite_nonnegative_control_value",
    "fracture_energy_control_residual_certificate",
    "fracture_energy_reference_increment",
]


_ALGORITHM_VERSION = 1


@dataclass(frozen=True, slots=True)
class ControlResidualCertificate:
    """Scale-aware certificate for one fracture-energy control residual.

    ``relative`` deliberately remains the raw residual divided by the actual
    accepted-to-target increment. It is not clipped when the scale-aware
    absolute branch controls the decision.
    """

    residual: float
    absolute: float
    relative: float
    limit: float
    ratio: float
    increment: float
    scale: float
    certified: bool


def _finite_positive_scalar(value: float | int, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be an int or float")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def fracture_energy_control_residual_certificate(
    residual: float | int,
    *,
    accepted_value: float | int,
    target_value: float | int,
    relative_tolerance: float | int,
    absolute_tolerance: float | int,
) -> ControlResidualCertificate:
    """Certify a signed residual with relative and scale-aware absolute gates.

    The accepted value is the actual regularised fracture energy of the last
    accepted state, not merely its nominal target. The exact rule is

    ``abs(residual) <= max(relative_tolerance * increment,
    absolute_tolerance * max(1, abs(accepted), abs(target)))``.
    """
    if type(residual) not in {int, float}:
        raise TypeError("residual must be an int or float")
    signed = float(residual)
    if not math.isfinite(signed):
        raise ValueError("residual must be finite")
    accepted = finite_nonnegative_control_value(
        accepted_value,
        name="accepted_value",
    )
    target = finite_nonnegative_control_value(
        target_value,
        name="target_value",
    )
    if target <= accepted:
        raise ValueError("target_value must be strictly greater than accepted_value")
    relative_tolerance_value = _finite_positive_scalar(
        relative_tolerance,
        name="relative_tolerance",
    )
    absolute_tolerance_value = _finite_positive_scalar(
        absolute_tolerance,
        name="absolute_tolerance",
    )
    increment = target - accepted
    scale = max(1.0, abs(accepted), abs(target))
    limit = max(
        relative_tolerance_value * increment,
        absolute_tolerance_value * scale,
    )
    absolute = abs(signed)
    return ControlResidualCertificate(
        residual=signed,
        absolute=absolute,
        relative=absolute / increment,
        limit=limit,
        ratio=absolute / limit,
        increment=increment,
        scale=scale,
        certified=absolute <= limit,
    )


def _strict_payload(
    payload: object,
    *,
    expected_keys: frozenset[str],
    name: str,
) -> dict[str, Any]:
    """Return an ordinary dict only when its JSON key set is exact."""
    if type(payload) is not dict:
        raise TypeError(f"{name} payload must be a dict")
    keys = set(payload)
    missing = expected_keys - keys
    unknown = keys - expected_keys
    if missing:
        raise ValueError(f"{name} payload is missing keys: {sorted(missing)}")
    if unknown:
        rendered = sorted(repr(key) for key in unknown)
        raise ValueError(f"{name} payload has unknown keys: {rendered}")
    return payload


def _validate_algorithm_version(value: object, *, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} algorithm_version must be an integer")
    if value != _ALGORITHM_VERSION:
        raise ValueError(
            f"{name} algorithm_version must be {_ALGORITHM_VERSION}, got {value}"
        )


def finite_nonnegative_control_value(
    value: float | int,
    *,
    name: str = "control_value",
) -> float:
    """Return a strictly typed, finite, non-negative scalar control value."""
    if type(name) is not str or not name:
        raise TypeError("name must be a non-empty string")
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be an int or float")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def fracture_energy_reference_increment(
    previous_fracture_energy: float | int,
    current_fracture_energy: float | int,
) -> float:
    """Compute the positive accepted increment used when control switches."""
    previous = finite_nonnegative_control_value(
        previous_fracture_energy,
        name="previous_fracture_energy",
    )
    current = finite_nonnegative_control_value(
        current_fracture_energy,
        name="current_fracture_energy",
    )
    if current <= previous:
        raise ValueError(
            "current_fracture_energy must be strictly greater than "
            "previous_fracture_energy"
        )
    increment = current - previous
    if not math.isfinite(increment):
        raise ValueError("reference fracture-energy increment must be finite")
    return increment


class ControlPhase(StrEnum):
    """The only supported control transition is displacement to fracture energy."""

    DISPLACEMENT = "displacement"
    FRACTURE_ENERGY = "fracture_energy"


@dataclass(frozen=True, slots=True)
class ControlState:
    """Immutable state for the one-way control-phase transition."""

    phase: ControlPhase
    reference_increment: float | None

    def __post_init__(self) -> None:
        if type(self.phase) is not ControlPhase:
            raise TypeError("phase must be a ControlPhase")
        if self.phase is ControlPhase.DISPLACEMENT:
            if self.reference_increment is not None:
                raise ValueError("displacement phase cannot have a reference_increment")
            return
        if self.reference_increment is None:
            raise ValueError("fracture_energy phase requires a positive reference_increment")
        reference = finite_nonnegative_control_value(
            self.reference_increment,
            name="reference_increment",
        )
        if reference <= 0.0:
            raise ValueError("reference_increment must be positive")
        object.__setattr__(self, "reference_increment", reference)

    @classmethod
    def displacement(cls) -> ControlState:
        """Create the initial displacement-control state."""
        return cls(ControlPhase.DISPLACEMENT, None)

    def to_payload(self) -> dict[str, object]:
        """Return the versioned, JSON-compatible control-state payload."""
        return {
            "algorithm_version": _ALGORITHM_VERSION,
            "phase": self.phase.value,
            "reference_increment": self.reference_increment,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ControlState:
        """Restore a control state from an exact version-1 JSON payload."""
        data = _strict_payload(
            payload,
            expected_keys=frozenset(
                {"algorithm_version", "phase", "reference_increment"}
            ),
            name="control state",
        )
        _validate_algorithm_version(
            data["algorithm_version"],
            name="control state",
        )
        raw_phase = data["phase"]
        if type(raw_phase) is not str:
            raise TypeError("control state phase must be a string")
        try:
            phase = ControlPhase(raw_phase)
        except ValueError as exc:
            raise ValueError(f"unsupported control state phase: {raw_phase!r}") from exc
        return cls(phase, data["reference_increment"])

    def switch_to_fracture_energy(
        self,
        *,
        previous_fracture_energy: float | int,
        current_fracture_energy: float | int,
    ) -> ControlState:
        """Switch once, deriving the reference target increment from accepted states."""
        if self.phase is not ControlPhase.DISPLACEMENT:
            raise RuntimeError("control phase is one-way and has already switched")
        reference = fracture_energy_reference_increment(
            previous_fracture_energy,
            current_fracture_energy,
        )
        return ControlState(ControlPhase.FRACTURE_ENERGY, reference)


@dataclass(frozen=True, slots=True)
class FractureEnergyTarget:
    """One regularised fracture-energy target and its dyadic depth."""

    value: float
    subdivision_level: int = 0

    def __post_init__(self) -> None:
        value = finite_nonnegative_control_value(
            self.value,
            name="fracture_energy target",
        )
        if type(self.subdivision_level) is not int:
            raise TypeError("subdivision_level must be an integer")
        if self.subdivision_level < 0:
            raise ValueError("subdivision_level must be non-negative")
        object.__setattr__(self, "value", value)

    def to_payload(self) -> dict[str, object]:
        """Return the compact JSON-compatible target payload."""
        return {
            "value": self.value,
            "subdivision_level": self.subdivision_level,
        }

    @classmethod
    def from_payload(cls, payload: object) -> FractureEnergyTarget:
        """Restore one target from an exact JSON payload."""
        data = _strict_payload(
            payload,
            expected_keys=frozenset({"value", "subdivision_level"}),
            name="fracture_energy target",
        )
        return cls(
            value=data["value"],
            subdivision_level=data["subdivision_level"],
        )


@dataclass(frozen=True, slots=True)
class FractureEnergyQueue:
    """Immutable, strictly increasing fracture-energy target queue."""

    accepted_value: float
    pending: tuple[FractureEnergyTarget, ...]

    def __post_init__(self) -> None:
        accepted = finite_nonnegative_control_value(
            self.accepted_value,
            name="accepted_value",
        )
        if type(self.pending) is not tuple:
            raise TypeError("pending must be a tuple")
        previous = accepted
        for index, target in enumerate(self.pending):
            if type(target) is not FractureEnergyTarget:
                raise TypeError(f"pending[{index}] must be a FractureEnergyTarget")
            if target.value <= previous:
                if index == 0:
                    raise ValueError(
                        "first pending fracture_energy target must be strictly greater "
                        "than accepted_value"
                    )
                raise ValueError("pending fracture_energy targets must be strictly increasing")
            previous = target.value
        object.__setattr__(self, "accepted_value", accepted)

    @classmethod
    def from_increment(
        cls,
        *,
        accepted_value: float | int,
        increment: float | int,
        count: int,
    ) -> FractureEnergyQueue:
        """Schedule ``count`` targets at a fixed positive fracture-energy increment."""
        accepted = finite_nonnegative_control_value(
            accepted_value,
            name="accepted_value",
        )
        step = finite_nonnegative_control_value(increment, name="increment")
        if step <= 0.0:
            raise ValueError("increment must be positive")
        if type(count) is not int:
            raise TypeError("count must be a positive integer")
        if count <= 0:
            raise ValueError("count must be a positive integer")

        pending: list[FractureEnergyTarget] = []
        previous = accepted
        for index in range(1, count + 1):
            value = accepted + index * step
            if not math.isfinite(value):
                raise ValueError("scheduled fracture_energy target must be finite")
            if value <= previous:
                raise ValueError(
                    "scheduled fracture_energy targets are not strictly increasing at "
                    "floating-point resolution"
                )
            pending.append(FractureEnergyTarget(value))
            previous = value
        return cls(accepted, tuple(pending))

    def to_payload(self) -> dict[str, object]:
        """Return the versioned, JSON-compatible target-queue payload."""
        return {
            "algorithm_version": _ALGORITHM_VERSION,
            "accepted_value": self.accepted_value,
            "pending": [target.to_payload() for target in self.pending],
        }

    @classmethod
    def from_payload(cls, payload: object) -> FractureEnergyQueue:
        """Restore and validate an exact version-1 target-queue payload."""
        data = _strict_payload(
            payload,
            expected_keys=frozenset(
                {"algorithm_version", "accepted_value", "pending"}
            ),
            name="fracture_energy queue",
        )
        _validate_algorithm_version(
            data["algorithm_version"],
            name="fracture_energy queue",
        )
        raw_pending = data["pending"]
        if type(raw_pending) is not list:
            raise TypeError("fracture_energy queue pending must be a list")
        pending = tuple(
            FractureEnergyTarget.from_payload(target) for target in raw_pending
        )
        return cls(
            accepted_value=data["accepted_value"],
            pending=pending,
        )

    @property
    def next_target(self) -> FractureEnergyTarget | None:
        """Return the next target without mutating the queue."""
        return self.pending[0] if self.pending else None

    def accept_next(self) -> FractureEnergyQueue:
        """Advance the accepted control value to the next scheduled target."""
        if not self.pending:
            raise IndexError("no pending fracture_energy target to accept")
        return FractureEnergyQueue(self.pending[0].value, self.pending[1:])

    def subdivide_failed_target(self) -> FractureEnergyQueue:
        """Dyadically bisect only the failed interval in fracture-energy coordinates."""
        if not self.pending:
            raise IndexError("no pending fracture_energy target to subdivide")
        failed = self.pending[0]
        midpoint = self.accepted_value + 0.5 * (failed.value - self.accepted_value)
        if not self.accepted_value < midpoint < failed.value:
            raise ValueError(
                "cannot dyadically subdivide fracture_energy target at floating-point resolution"
            )
        level = failed.subdivision_level + 1
        refined = (
            FractureEnergyTarget(midpoint, level),
            FractureEnergyTarget(failed.value, level),
            *self.pending[1:],
        )
        return FractureEnergyQueue(self.accepted_value, refined)
