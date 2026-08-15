r"""One-dimensional AT2 pre-calibration for a Gaussian weak interface.

The centre-line toughness prescribed by a diffuse grain-boundary field is not
the fracture energy dissipated by an AT2 crack profile of finite width.  This
module isolates that distinction with the dimensionless half-line problem

.. math::

   \min_d \int_0^L g(y)\left(d^2 + |d'|^2\right)\,dy,
   \quad d(0)=1,\ d(L)=0,

where ``y=x/ell`` and
``g(y)=1-(1-r) exp(-y^2/rho^2)`` with ``rho=b/ell``.  The energy is normalised
by the discrete homogeneous solution on the same mesh/domain (the convergent
approximation to ``coth(L)``), so ``r=1`` is exactly one.

This is a one-dimensional input pre-calibration.  It does not validate crack
deflection, mixed mode loading, mesh orientation or a two-dimensional FEniCSx
fracture calculation.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_DOMAIN_HALF_WIDTH = 12.0
DEFAULT_ELEMENTS = 2048
DEFAULT_CENTERLINE_RATIOS = (0.2, 0.45, 0.6, 0.8)
DEFAULT_WIDTH_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_TARGET_EFFECTIVE_RATIOS = (0.5, 0.7, 0.8)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "effective_toughness_ratio",
    "invert_centerline_ratio",
    "main",
    "minimum_interface_profile",
    "scan_interface_calibration",
]


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str, *, minimum: int = 2) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validated_parameters(
    centerline_ratio: Any,
    width_over_length_scale: Any,
    domain_half_width: Any,
    elements: Any,
) -> tuple[float, float, float, int]:
    ratio = _finite_real(centerline_ratio, "centerline_ratio")
    width = _finite_real(width_over_length_scale, "width_over_length_scale")
    domain = _finite_real(domain_half_width, "domain_half_width")
    count = _positive_integer(elements, "elements")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("centerline_ratio must lie in [0, 1]")
    if width <= 0.0:
        raise ValueError("width_over_length_scale must be positive")
    if domain <= 0.0:
        raise ValueError("domain_half_width must be positive")
    return ratio, width, domain, count


def _element_matrices(
    centerline_ratio: float,
    width_over_length_scale: float,
    domain_half_width: float,
    elements: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorised two-point-Gauss element matrix entries."""
    h = domain_half_width / elements
    left = np.arange(elements, dtype=float) * h
    midpoint = left + 0.5 * h
    xi = 1.0 / math.sqrt(3.0)
    quadrature_x = np.stack((midpoint - 0.5 * h * xi, midpoint + 0.5 * h * xi))
    g = 1.0 - (1.0 - centerline_ratio) * np.exp(-((quadrature_x / width_over_length_scale) ** 2))
    weights = 0.5 * h
    n_left = np.array(((1.0 + xi) / 2.0, (1.0 - xi) / 2.0))[:, None]
    n_right = np.array(((1.0 - xi) / 2.0, (1.0 + xi) / 2.0))[:, None]
    derivative_product = 1.0 / h**2
    k00 = weights * np.sum(g * (n_left**2 + derivative_product), axis=0)
    k11 = weights * np.sum(g * (n_right**2 + derivative_product), axis=0)
    k01 = weights * np.sum(g * (n_left * n_right - derivative_product), axis=0)
    return k00, k01, k11


def _solve_symmetric_tridiagonal(
    diagonal: np.ndarray, off_diagonal: np.ndarray, right_hand_side: np.ndarray
) -> np.ndarray:
    """Thomas solve with explicit positive-pivot checks."""
    n = int(diagonal.size)
    if right_hand_side.shape != (n,) or off_diagonal.shape != (max(n - 1, 0),):
        raise ValueError("inconsistent tridiagonal system dimensions")
    modified_diagonal = np.array(diagonal, dtype=float, copy=True)
    modified_rhs = np.array(right_hand_side, dtype=float, copy=True)
    for index in range(1, n):
        pivot = modified_diagonal[index - 1]
        if not math.isfinite(float(pivot)) or pivot <= 0.0:
            raise RuntimeError("AT2 calibration matrix lost positive definiteness")
        multiplier = off_diagonal[index - 1] / pivot
        modified_diagonal[index] -= multiplier * off_diagonal[index - 1]
        modified_rhs[index] -= multiplier * modified_rhs[index - 1]
    if modified_diagonal[-1] <= 0.0:
        raise RuntimeError("AT2 calibration matrix lost positive definiteness")
    solution = np.empty(n, dtype=float)
    solution[-1] = modified_rhs[-1] / modified_diagonal[-1]
    for index in range(n - 2, -1, -1):
        solution[index] = (
            modified_rhs[index] - off_diagonal[index] * solution[index + 1]
        ) / modified_diagonal[index]
    return solution


def _minimum_discrete_profile_and_energy(
    k00: np.ndarray, k01: np.ndarray, k11: np.ndarray
) -> tuple[np.ndarray, float]:
    """Minimise the quadratic FE energy with d(0)=1 and d(L)=0."""
    count = int(k00.size)
    main = np.zeros(count + 1, dtype=float)
    main[:-1] += k00
    main[1:] += k11
    rhs = np.zeros(count - 1, dtype=float)
    rhs[0] = -k01[0]
    interior = _solve_symmetric_tridiagonal(main[1:-1], k01[1:-1], rhs)
    damage = np.empty(count + 1, dtype=float)
    damage[0] = 1.0
    damage[-1] = 0.0
    damage[1:-1] = interior
    energy = float(
        np.sum(
            k00 * damage[:-1] ** 2 + 2.0 * k01 * damage[:-1] * damage[1:] + k11 * damage[1:] ** 2
        )
    )
    return damage, energy


def _minimum_discrete_energy(k00: np.ndarray, k01: np.ndarray, k11: np.ndarray) -> float:
    return _minimum_discrete_profile_and_energy(k00, k01, k11)[1]


@functools.lru_cache(maxsize=32)
def _homogeneous_discrete_energy(domain_half_width: float, elements: int) -> float:
    matrices = _element_matrices(1.0, 1.0, domain_half_width, elements)
    return _minimum_discrete_energy(*matrices)


def effective_toughness_ratio(
    centerline_ratio: float,
    width_over_length_scale: float,
    *,
    domain_half_width: float = DEFAULT_DOMAIN_HALF_WIDTH,
    elements: int = DEFAULT_ELEMENTS,
) -> float:
    """Return the minimum one-dimensional AT2 surface-energy ratio.

    ``centerline_ratio`` is the input ``Gc(0)/Gc_bulk`` and
    ``width_over_length_scale`` is ``b/ell``.  The returned number is the
    effective crack-surface energy divided by the homogeneous value on the
    same truncated domain.
    """
    ratio, width, domain, count = _validated_parameters(
        centerline_ratio, width_over_length_scale, domain_half_width, elements
    )
    k00, k01, k11 = _element_matrices(ratio, width, domain, count)
    energy = _minimum_discrete_energy(k00, k01, k11)
    result = energy / _homogeneous_discrete_energy(domain, count)
    if result > 1.0 and result <= 1.0 + 5.0e-10:
        result = 1.0
    if not 0.0 < result <= 1.0 + 5.0e-10:
        raise RuntimeError(f"invalid effective toughness ratio {result!r}")
    return min(result, 1.0)


def minimum_interface_profile(
    centerline_ratio: float,
    width_over_length_scale: float,
    *,
    domain_half_width: float = DEFAULT_DOMAIN_HALF_WIDTH,
    elements: int = DEFAULT_ELEMENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dimensionless half-domain coordinates and the minimum FE profile."""
    ratio, width, domain, count = _validated_parameters(
        centerline_ratio, width_over_length_scale, domain_half_width, elements
    )
    damage, _ = _minimum_discrete_profile_and_energy(
        *_element_matrices(ratio, width, domain, count)
    )
    coordinates = np.linspace(0.0, domain, count + 1)
    return coordinates, damage


def invert_centerline_ratio(
    target_effective_ratio: float,
    width_over_length_scale: float,
    *,
    domain_half_width: float = DEFAULT_DOMAIN_HALF_WIDTH,
    elements: int = DEFAULT_ELEMENTS,
    tolerance: float = 1.0e-8,
    maximum_iterations: int = 80,
) -> float:
    """Invert the monotone calibration map for a requested effective ratio."""
    target = _finite_real(target_effective_ratio, "target_effective_ratio")
    width = _finite_real(width_over_length_scale, "width_over_length_scale")
    domain = _finite_real(domain_half_width, "domain_half_width")
    count = _positive_integer(elements, "elements")
    tol = _finite_real(tolerance, "tolerance")
    iterations = _positive_integer(maximum_iterations, "maximum_iterations", minimum=1)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_effective_ratio must lie in (0, 1]")
    if width <= 0.0 or domain <= 0.0 or tol <= 0.0:
        raise ValueError(
            "width_over_length_scale, domain_half_width and tolerance must be positive"
        )
    lower_value = effective_toughness_ratio(0.0, width, domain_half_width=domain, elements=count)
    upper_value = effective_toughness_ratio(1.0, width, domain_half_width=domain, elements=count)
    admissibility_tolerance = max(tol, 1.0e-12)
    if (
        target < lower_value - admissibility_tolerance
        or target > upper_value + admissibility_tolerance
    ):
        raise ValueError(
            "target_effective_ratio is unattainable for this b/ell: "
            f"admissible interval is [{lower_value:.8g}, {upper_value:.8g}]"
        )
    if abs(target - lower_value) <= tol:
        return 0.0
    if abs(target - upper_value) <= tol:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        value = effective_toughness_ratio(midpoint, width, domain_half_width=domain, elements=count)
        if abs(value - target) <= tol:
            return midpoint
        if value < target:
            lower = midpoint
        else:
            upper = midpoint
    result = 0.5 * (lower + upper)
    residual = abs(
        effective_toughness_ratio(result, width, domain_half_width=domain, elements=count) - target
    )
    if residual > tol:
        raise RuntimeError(
            f"centerline-ratio inversion did not reach tolerance: residual={residual:.6g}"
        )
    return result


def _real_sequence(
    values: Sequence[float] | None, default: Sequence[float], name: str
) -> list[float]:
    source = list(default if values is None else values)
    if not source:
        raise ValueError(f"at least one {name} is required")
    return [_finite_real(value, f"{name}[{index}]") for index, value in enumerate(source)]


def scan_interface_calibration(
    *,
    centerline_ratios: Sequence[float] | None = None,
    width_ratios: Sequence[float] | None = None,
    target_effective_ratios: Sequence[float] | None = None,
    domain_half_width: float = DEFAULT_DOMAIN_HALF_WIDTH,
    elements: int = DEFAULT_ELEMENTS,
) -> dict[str, Any]:
    """Build a machine-readable forward/inverse ``b/ell`` calibration table."""
    ratios = _real_sequence(centerline_ratios, DEFAULT_CENTERLINE_RATIOS, "centerline_ratio")
    widths = _real_sequence(width_ratios, DEFAULT_WIDTH_RATIOS, "width_over_ell")
    targets = _real_sequence(
        target_effective_ratios,
        DEFAULT_TARGET_EFFECTIVE_RATIOS,
        "target_effective_ratio",
    )
    domain = _finite_real(domain_half_width, "domain_half_width")
    count = _positive_integer(elements, "elements")
    for ratio in ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("centerline ratios must lie in [0, 1]")
    for width in widths:
        if width <= 0.0:
            raise ValueError("width ratios must be positive")
    for target in targets:
        if not 0.0 < target <= 1.0:
            raise ValueError("target effective ratios must lie in (0, 1]")

    forward = [
        {
            "centerline_toughness_ratio": ratio,
            "width_over_length_scale": width,
            "effective_toughness_ratio": effective_toughness_ratio(
                ratio, width, domain_half_width=domain, elements=count
            ),
        }
        for width in widths
        for ratio in ratios
    ]
    inverse: list[dict[str, Any]] = []
    for width in widths:
        minimum = effective_toughness_ratio(0.0, width, domain_half_width=domain, elements=count)
        for target in targets:
            record: dict[str, Any] = {
                "target_effective_toughness_ratio": target,
                "width_over_length_scale": width,
                "minimum_achievable_effective_ratio_at_zero_centerline_toughness": minimum,
            }
            if target < minimum - 1.0e-8:
                record.update(
                    {
                        "achievable": False,
                        "centerline_toughness_ratio": None,
                        "reason": "target lies below the finite-width weak-band minimum",
                    }
                )
            else:
                centerline = invert_centerline_ratio(
                    target,
                    width,
                    domain_half_width=domain,
                    elements=count,
                )
                achieved = effective_toughness_ratio(
                    centerline, width, domain_half_width=domain, elements=count
                )
                record.update(
                    {
                        "achievable": True,
                        "centerline_toughness_ratio": centerline,
                        "achieved_effective_toughness_ratio": achieved,
                        "absolute_inversion_residual": abs(achieved - target),
                    }
                )
            inverse.append(record)

    check_ratio = 0.45
    check_width = 1.0
    coarse = effective_toughness_ratio(
        check_ratio, check_width, domain_half_width=domain, elements=count
    )
    refined = effective_toughness_ratio(
        check_ratio, check_width, domain_half_width=domain, elements=2 * count
    )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "study": "one_dimensional_AT2_Gaussian_weak_interface_precalibration",
        "dimensionless_model": {
            "coordinate": "y = x/ell on the half-domain [0,L]",
            "toughness_ratio": "g(y)=1-(1-r)*exp(-y^2/rho^2)",
            "width_ratio": "rho=b/ell",
            "boundary_conditions": "d(0)=1, d(L)=0",
            "objective": "integral_0^L g(y)*(d^2+|d_y|^2) dy",
            "normalisation": ("discrete homogeneous energy on the same mesh; converges to coth(L)"),
        },
        "discretisation": {
            "method": "uniform linear finite elements with two-point Gauss quadrature",
            "domain_half_width_over_ell": domain,
            "elements": count,
            "element_size_over_ell": domain / count,
            "linear_solver": "symmetric tridiagonal Thomas algorithm",
        },
        "parameter_grid": {
            "centerline_toughness_ratio": ratios,
            "width_over_length_scale": widths,
            "target_effective_toughness_ratio": targets,
        },
        "forward_calibration": forward,
        "inverse_calibration": inverse,
        "refinement_crosscheck": {
            "centerline_toughness_ratio": check_ratio,
            "width_over_length_scale": check_width,
            "elements": count,
            "effective_ratio": coarse,
            "refined_elements": 2 * count,
            "refined_effective_ratio": refined,
            "absolute_difference": abs(refined - coarse),
        },
        "interpretation_limit": (
            "One-dimensional AT2 input pre-calibration only; this does not validate "
            "two-dimensional crack deflection, mesh orientation, mixed mode loading, "
            "measured-EBSD fracture, or a FEniCSx implementation."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m graphfracture.interface_calibration",
        description="Pre-calibrate a Gaussian weak-interface input for one-dimensional AT2.",
    )
    parser.add_argument(
        "--centerline-ratio",
        action="append",
        type=float,
        dest="centerline_ratios",
        help="Gc(0)/Gc_bulk; repeat to define a forward grid",
    )
    parser.add_argument(
        "--width-over-ell",
        action="append",
        type=float,
        dest="width_ratios",
        help="Gaussian b/ell; repeat to define a forward/inverse grid",
    )
    parser.add_argument(
        "--target-effective-ratio",
        action="append",
        type=float,
        dest="target_effective_ratios",
        help="requested effective ratio for inverse calibration; repeatable",
    )
    parser.add_argument("--domain-half-width", type=float, default=DEFAULT_DOMAIN_HALF_WIDTH)
    parser.add_argument("--elements", type=int, default=DEFAULT_ELEMENTS)
    parser.add_argument("--output", "-o", type=Path, help="exclusive-create JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = scan_interface_calibration(
            centerline_ratios=args.centerline_ratios,
            width_ratios=args.width_ratios,
            target_effective_ratios=args.target_effective_ratios,
            domain_half_width=args.domain_half_width,
            elements=args.elements,
        )
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"interface-calibration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
