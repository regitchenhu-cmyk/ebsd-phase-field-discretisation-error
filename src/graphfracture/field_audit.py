"""Mesh-free sensitivity audit for measured EBSD material fields.

The audit samples :class:`graphfracture.chain_field.ChainBoundaryField` at
uniform cell centres.  It is deliberately independent of DOLFINx: the purpose
is to quantify how the prescribed Gaussian kernel width and the EBSD geometry
confidence cutoff change the *input* fields before an expensive finite-element
run is launched.

This module is not a material-parameter calibration.  Its reported area
fractions are midpoint quadrature estimates on the requested regular sampling
grid, not finite-element solution statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .chain_field import ChainBoundaryField
from .config import RunConfig, load_config

FIELD_AUDIT_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_NX = 192
DEFAULT_SAMPLE_NY = 96
QUANTILE_LEVELS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)

__all__ = [
    "DEFAULT_SAMPLE_NX",
    "DEFAULT_SAMPLE_NY",
    "FIELD_AUDIT_SCHEMA_VERSION",
    "QUANTILE_LEVELS",
    "audit_chain_field",
    "main",
]


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _influence_radii(values: Sequence[float] | None, config: RunConfig) -> list[float]:
    source = [config.graph.influence_radius] if values is None else list(values)
    if not source:
        raise ValueError("at least one influence_radius is required")
    result = []
    for index, value in enumerate(source):
        radius = _finite_real(value, f"influence_radius[{index}]")
        if radius <= 0.0:
            raise ValueError("influence_radius must be positive")
        result.append(radius)
    return result


def _confidence_floors(values: Sequence[float] | None, config: RunConfig) -> list[float]:
    source = [config.graph.confidence_floor] if values is None else list(values)
    if not source:
        raise ValueError("at least one confidence_floor is required")
    result = []
    for index, value in enumerate(source):
        floor = _finite_real(value, f"confidence_floor[{index}]")
        if not 0.0 <= floor <= 1.0:
            raise ValueError("confidence_floor must lie in [0, 1]")
        result.append(floor)
    return result


def _sample_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _uniform_cell_centres(config: RunConfig, nx: int, ny: int) -> np.ndarray:
    geometry = config.geometry
    x = (np.arange(nx, dtype=float) + 0.5) * (geometry.length / nx)
    y = (np.arange(ny, dtype=float) + 0.5) * (geometry.height / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx.ravel(), yy.ravel()))


def _quantile_key(level: float) -> str:
    return f"{level:g}"


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("sampled field arrays must be non-empty and one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("sampled field values must be finite")
    quantiles = np.quantile(array, QUANTILE_LEVELS)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "quantiles": {
            _quantile_key(level): float(value)
            for level, value in zip(QUANTILE_LEVELS, quantiles, strict=True)
        },
    }


def _area_fraction(mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(mask, dtype=float)))


def _field_statistics(
    gc_ratio: np.ndarray, diffusivity_ratio: np.ndarray, trap_density: np.ndarray
) -> dict[str, Any]:
    gc = _summary(gc_ratio)
    gc["area_fraction_below"] = {
        f"{threshold:g}": _area_fraction(gc_ratio < threshold)
        for threshold in (0.6, 0.8, 0.9, 0.95)
    }

    diffusivity = _summary(diffusivity_ratio)
    diffusivity["area_fraction_above"] = {
        f"{threshold:g}": _area_fraction(diffusivity_ratio > threshold)
        for threshold in (2.0, 5.0, 10.0)
    }

    trap = _summary(trap_density)
    trap["area_fraction_above_zero"] = _area_fraction(trap_density > 0.0)
    return {"Gc_ratio": gc, "D_ratio": diffusivity, "Nt": trap}


def _require_chain_config(config: RunConfig) -> None:
    if not config.graph.enabled or not config.graph.chain_artifact.strip():
        raise ValueError(
            "field audit requires an enabled graph.chain_artifact configuration "
            "(synthetic node/edge and homogeneous configurations are unsupported)"
        )


def _case(
    config: RunConfig,
    points: np.ndarray,
    influence_radius: float,
    confidence_floor: float,
) -> dict[str, Any]:
    candidate = replace(
        config,
        graph=replace(
            config.graph,
            influence_radius=influence_radius,
            confidence_floor=confidence_floor,
        ),
    )
    candidate.validate()
    field = ChainBoundaryField.from_config(candidate)
    gc_ratio = field.toughness_ratio_at(points)
    diffusivity_ratio = field.diffusivity_ratio_at(points)
    trap_density = field.trap_density_at(points)
    description = field.describe()
    provenance = description.pop("provenance")
    return {
        "influence_radius": influence_radius,
        "confidence_floor": confidence_floor,
        "field": {"describe": description, "provenance": provenance},
        "statistics": _field_statistics(gc_ratio, diffusivity_ratio, trap_density),
    }


def audit_chain_field(
    config_path: str | Path,
    *,
    influence_radii: Sequence[float] | None = None,
    confidence_floors: Sequence[float] | None = None,
    sample_nx: int = DEFAULT_SAMPLE_NX,
    sample_ny: int = DEFAULT_SAMPLE_NY,
) -> dict[str, Any]:
    """Audit a Cartesian product of kernel widths and confidence cutoffs.

    When an option sequence is omitted, its value from the base TOML
    configuration is used.  Sampling points are the centres of ``sample_nx`` by
    ``sample_ny`` equal-area rectangles spanning the configured geometry.
    """

    path = Path(config_path).resolve()
    config = load_config(path)
    _require_chain_config(config)
    nx = _sample_count(sample_nx, "sample_nx")
    ny = _sample_count(sample_ny, "sample_ny")
    radii = _influence_radii(influence_radii, config)
    floors = _confidence_floors(confidence_floors, config)
    points = _uniform_cell_centres(config, nx, ny)

    geometry = config.geometry
    cell_area = geometry.length * geometry.height / (nx * ny)
    cases = [_case(config, points, radius, floor) for radius in radii for floor in floors]
    return {
        "schema_version": FIELD_AUDIT_SCHEMA_VERSION,
        "study": "ebsd_material_field_kernel_and_confidence_sensitivity",
        "base_config": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "geometry": {
            "length": geometry.length,
            "height": geometry.height,
            "area": geometry.length * geometry.height,
        },
        "sampling": {
            "nx": nx,
            "ny": ny,
            "sample_count": nx * ny,
            "cell_area": cell_area,
            "point_rule": "uniform rectangular cell centres",
            "area_fraction_rule": "equal-area midpoint sample fraction",
            "standard_deviation": "population (ddof=0)",
            "quantile_levels": list(QUANTILE_LEVELS),
        },
        "parameter_grid": {
            "influence_radius": radii,
            "confidence_floor": floors,
            "combination_rule": "Cartesian product; influence_radius outer order",
        },
        "cases": cases,
        "interpretation_limit": (
            "This is a Gaussian kernel-width and EBSD geometry-confidence sensitivity audit "
            "of prescribed input fields, not a physical material-parameter calibration."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m graphfracture.field_audit",
        description="Audit measured EBSD material-field kernel/confidence sensitivity.",
    )
    parser.add_argument("config", type=Path, help="TOML configuration with graph.chain_artifact")
    parser.add_argument(
        "--influence-radius",
        type=float,
        action="append",
        dest="influence_radii",
        help="Gaussian field width; repeat for a Cartesian sensitivity grid",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        action="append",
        dest="confidence_floors",
        help="minimum chain confidence in [0,1]; repeat for a Cartesian sensitivity grid",
    )
    parser.add_argument("--sample-nx", type=int, default=DEFAULT_SAMPLE_NX)
    parser.add_argument("--sample-ny", type=int, default=DEFAULT_SAMPLE_NY)
    parser.add_argument("--output", "-o", type=Path, help="exclusive-create JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = audit_chain_field(
            args.config,
            influence_radii=args.influence_radii,
            confidence_floors=args.confidence_floors,
            sample_nx=args.sample_nx,
            sample_ny=args.sample_ny,
        )
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
    except (OSError, ValueError, TypeError) as exc:
        print(f"field-audit error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
