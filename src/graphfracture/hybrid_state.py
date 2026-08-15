"""Pure restart state for hybrid displacement/fracture-energy scheduling.

The structures in this module contain no finite-element objects.  They define
the immutable, versioned JSON boundary between the hybrid scheduler and the
restart manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .damage_control import ControlPhase, ControlState, FractureEnergyQueue

__all__ = ["DisplacementTarget", "HybridSchedulerState"]


_ALGORITHM_VERSION = 1


def _strict_payload(
    payload: object,
    *,
    expected_keys: frozenset[str],
    name: str,
) -> dict[str, Any]:
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


def _nonnegative_integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_nonnegative_real(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be an int or float")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class DisplacementTarget:
    """One pending displacement target and its continuation provenance."""

    displacement: float
    subdivision_level: int
    scheduled_step: int

    def __post_init__(self) -> None:
        displacement = _finite_nonnegative_real(
            self.displacement,
            name="displacement",
        )
        subdivision_level = _nonnegative_integer(
            self.subdivision_level,
            name="subdivision_level",
        )
        scheduled_step = _nonnegative_integer(
            self.scheduled_step,
            name="scheduled_step",
        )
        if scheduled_step == 0:
            raise ValueError("scheduled_step must be positive")
        object.__setattr__(self, "displacement", displacement)
        object.__setattr__(self, "subdivision_level", subdivision_level)
        object.__setattr__(self, "scheduled_step", scheduled_step)

    def to_payload(self) -> dict[str, float | int]:
        """Return the exact JSON representation used inside scheduler state."""
        return {
            "displacement": self.displacement,
            "subdivision_level": self.subdivision_level,
            "scheduled_step": self.scheduled_step,
        }

    @classmethod
    def from_payload(cls, payload: object) -> DisplacementTarget:
        """Restore one target from an exact JSON object."""
        data = _strict_payload(
            payload,
            expected_keys=frozenset(
                {"displacement", "subdivision_level", "scheduled_step"}
            ),
            name="displacement target",
        )
        return cls(
            displacement=data["displacement"],
            subdivision_level=data["subdivision_level"],
            scheduled_step=data["scheduled_step"],
        )


@dataclass(frozen=True, slots=True)
class HybridSchedulerState:
    """Immutable scheduler state committed in a hybrid restart checkpoint."""

    state: ControlState
    reference_displacement: float
    switch_accepted_step: int | None
    phase_step: int
    pending_displacements: tuple[DisplacementTarget, ...]
    fracture_energy_queue: FractureEnergyQueue | None

    def __post_init__(self) -> None:
        if type(self.state) is not ControlState:
            raise TypeError("state must be a ControlState")
        reference = _finite_nonnegative_real(
            self.reference_displacement,
            name="reference_displacement",
        )
        if reference <= 0.0:
            raise ValueError("reference_displacement must be positive")
        phase_step = _nonnegative_integer(self.phase_step, name="phase_step")
        if type(self.pending_displacements) is not tuple:
            raise TypeError("pending_displacements must be a tuple")

        previous = -math.inf
        for index, target in enumerate(self.pending_displacements):
            if type(target) is not DisplacementTarget:
                raise TypeError(
                    f"pending_displacements[{index}] must be a DisplacementTarget"
                )
            if target.displacement <= previous:
                raise ValueError("pending displacement targets must be strictly increasing")
            previous = target.displacement

        switch_step = self.switch_accepted_step
        if switch_step is not None:
            switch_step = _nonnegative_integer(
                switch_step,
                name="switch_accepted_step",
            )
        queue = self.fracture_energy_queue
        if queue is not None and type(queue) is not FractureEnergyQueue:
            raise TypeError("fracture_energy_queue must be a FractureEnergyQueue or None")

        if self.state.phase is ControlPhase.DISPLACEMENT:
            if switch_step is not None:
                raise ValueError("displacement phase requires switch_accepted_step=None")
            if queue is not None:
                raise ValueError("displacement phase requires fracture_energy_queue=None")
        elif self.state.phase is ControlPhase.FRACTURE_ENERGY:
            if switch_step is None:
                raise ValueError(
                    "fracture_energy phase requires a non-negative switch_accepted_step"
                )
            if self.pending_displacements:
                raise ValueError("fracture_energy phase requires an empty displacement queue")
            if queue is None:
                raise ValueError("fracture_energy phase requires a fracture_energy_queue")
        else:  # pragma: no cover - ControlState already closes this enum.
            raise ValueError(f"unsupported control phase: {self.state.phase!r}")

        object.__setattr__(self, "reference_displacement", reference)
        object.__setattr__(self, "switch_accepted_step", switch_step)
        object.__setattr__(self, "phase_step", phase_step)

    def to_payload(self) -> dict[str, object]:
        """Return the exact version-1, JSON-compatible scheduler payload."""
        return {
            "algorithm_version": _ALGORITHM_VERSION,
            "state": self.state.to_payload(),
            "reference_displacement": self.reference_displacement,
            "switch_accepted_step": self.switch_accepted_step,
            "phase_step": self.phase_step,
            "pending_displacements": [
                target.to_payload() for target in self.pending_displacements
            ],
            "fracture_energy_queue": (
                self.fracture_energy_queue.to_payload()
                if self.fracture_energy_queue is not None
                else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: object) -> HybridSchedulerState:
        """Restore and validate an exact version-1 JSON payload."""
        data = _strict_payload(
            payload,
            expected_keys=frozenset(
                {
                    "algorithm_version",
                    "state",
                    "reference_displacement",
                    "switch_accepted_step",
                    "phase_step",
                    "pending_displacements",
                    "fracture_energy_queue",
                }
            ),
            name="hybrid scheduler state",
        )
        version = data["algorithm_version"]
        if type(version) is not int:
            raise TypeError("hybrid scheduler state algorithm_version must be an integer")
        if version != _ALGORITHM_VERSION:
            raise ValueError(
                "hybrid scheduler state algorithm_version must be "
                f"{_ALGORITHM_VERSION}, got {version}"
            )
        raw_pending = data["pending_displacements"]
        if type(raw_pending) is not list:
            raise TypeError("pending_displacements payload must be a list")
        raw_queue = data["fracture_energy_queue"]
        if raw_queue is not None and type(raw_queue) is not dict:
            raise TypeError("fracture_energy_queue payload must be a dict or None")
        return cls(
            state=ControlState.from_payload(data["state"]),
            reference_displacement=data["reference_displacement"],
            switch_accepted_step=data["switch_accepted_step"],
            phase_step=data["phase_step"],
            pending_displacements=tuple(
                DisplacementTarget.from_payload(target) for target in raw_pending
            ),
            fracture_energy_queue=(
                FractureEnergyQueue.from_payload(raw_queue)
                if raw_queue is not None
                else None
            ),
        )
