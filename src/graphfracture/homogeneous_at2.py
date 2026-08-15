r"""Closed-form homogeneous response of the tensile AT2 verification model.

The DOLFINx baseline degrades the complete plane-strain elastic energy with

.. math::

   g(d)=(1-k)(1-d)^2+k.

For uniform vertical strain, traction-free in-plane lateral sides and zero
out-of-plane strain, the effective tensile modulus is
``Ebar = E / (1 - nu**2)``.  With a spatially uniform damage field, stationarity
of the AT2 energy gives

.. math::

   x=(1-k)\bar E\ell\varepsilon^2/G_c,\qquad d=x/(1+x),

and ``sigma = Ebar*epsilon*(k + (1-k)/(1+x)**2)``.  For ``k < 0.2`` this
curve has a first local maximum.  The later residual-stiffness upturn is a
regularisation artefact and is deliberately not called a material peak.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class HomogeneousAT2State:
    strain: float
    damage: float
    degradation: float
    nominal_stress: float
    dimensionless_driving_force: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HomogeneousAT2Peak:
    strain: float
    stress: float
    damage: float
    degradation: float
    dimensionless_driving_force: float
    effective_modulus: float
    residual_stiffness: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            **asdict(self),
            "kind": "first_local_peak_before_residual_stiffness_upturn",
            "interpretation_limit": (
                "homogeneous tensile constitutive benchmark; not a notched-specimen onset load"
            ),
        }


__all__ = [
    "HomogeneousAT2Peak",
    "HomogeneousAT2State",
    "homogeneous_at2_first_peak",
    "homogeneous_at2_state",
    "plane_strain_uniaxial_modulus",
]


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def plane_strain_uniaxial_modulus(young_modulus: float, poisson_ratio: float) -> float:
    """Return ``sigma_yy/epsilon_yy`` for free in-plane lateral contraction."""
    young = _finite_real(young_modulus, "young_modulus")
    poisson = _finite_real(poisson_ratio, "poisson_ratio")
    if young <= 0.0:
        raise ValueError("young_modulus must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    return young / (1.0 - poisson**2)


def homogeneous_at2_state(
    strain: float,
    *,
    effective_modulus: float,
    fracture_toughness: float,
    length_scale: float,
    residual_stiffness: float = 0.0,
) -> HomogeneousAT2State:
    """Return the stationary uniform-damage state at a non-negative strain."""
    epsilon = _finite_real(strain, "strain")
    modulus = _finite_real(effective_modulus, "effective_modulus")
    toughness = _finite_real(fracture_toughness, "fracture_toughness")
    ell = _finite_real(length_scale, "length_scale")
    residual = _finite_real(residual_stiffness, "residual_stiffness")
    if epsilon < 0.0:
        raise ValueError("strain must be non-negative")
    if modulus <= 0.0 or toughness <= 0.0 or ell <= 0.0:
        raise ValueError("effective_modulus, fracture_toughness and length_scale must be positive")
    if not 0.0 <= residual < 1.0:
        raise ValueError("residual_stiffness must lie in [0, 1)")

    driving_force = (1.0 - residual) * modulus * ell * epsilon**2 / toughness
    damage = driving_force / (1.0 + driving_force)
    degradation = residual + (1.0 - residual) / (1.0 + driving_force) ** 2
    stress = modulus * epsilon * degradation
    return HomogeneousAT2State(
        strain=epsilon,
        damage=damage,
        degradation=degradation,
        nominal_stress=stress,
        dimensionless_driving_force=driving_force,
    )


def _first_peak_driving_force(residual_stiffness: float) -> float:
    residual = residual_stiffness
    if residual == 0.0:
        return 1.0 / 3.0
    if residual >= 0.2:
        raise ValueError("residual_stiffness must be below 0.2 for a first local constitutive peak")

    intact = 1.0 - residual

    def stationarity(value: float) -> float:
        return residual * (1.0 + value) ** 3 + intact * (1.0 - 3.0 * value)

    lower, upper = 1.0 / 3.0, 1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if stationarity(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def homogeneous_at2_first_peak(
    *,
    young_modulus: float,
    poisson_ratio: float,
    fracture_toughness: float,
    length_scale: float,
    residual_stiffness: float,
) -> HomogeneousAT2Peak:
    """Return the first local peak of the current homogeneous tensile model."""
    modulus = plane_strain_uniaxial_modulus(young_modulus, poisson_ratio)
    toughness = _finite_real(fracture_toughness, "fracture_toughness")
    ell = _finite_real(length_scale, "length_scale")
    residual = _finite_real(residual_stiffness, "residual_stiffness")
    if toughness <= 0.0 or ell <= 0.0:
        raise ValueError("fracture_toughness and length_scale must be positive")
    if not 0.0 <= residual < 0.2:
        raise ValueError("residual_stiffness must lie in [0, 0.2) for the first peak")

    driving_force = _first_peak_driving_force(residual)
    strain = math.sqrt(driving_force * toughness / ((1.0 - residual) * modulus * ell))
    state = homogeneous_at2_state(
        strain,
        effective_modulus=modulus,
        fracture_toughness=toughness,
        length_scale=ell,
        residual_stiffness=residual,
    )
    return HomogeneousAT2Peak(
        strain=state.strain,
        stress=state.nominal_stress,
        damage=state.damage,
        degradation=state.degradation,
        dimensionless_driving_force=state.dimensionless_driving_force,
        effective_modulus=modulus,
        residual_stiffness=residual,
    )
