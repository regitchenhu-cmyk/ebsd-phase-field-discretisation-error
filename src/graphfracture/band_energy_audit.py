"""Mesh-level surface-energy audit of the inclined DG0 weak band.

The one-dimensional pre-calibration in :mod:`graphfracture.interface_calibration`
maps a Gaussian centreline toughness ratio to an effective crack-surface-energy
ratio using the analytic profile.  The solver, however, interpolates the band
into a piecewise-constant DG0 field at triangle barycentres, so the profile a
crack actually feels on a given mesh is a staircase whose effective toughness
can drift from the continuum target.  This audit quantifies that drift:

1. sample the *project's own* boundary field at the barycentres of the
   structured ``nx x ny`` right-diagonal triangulation (the same call the
   solver makes when building ``Gc0``);
2. at stations along each interface edge, extract the DG0 staircase along the
   perpendicular transect by triangle-containment lookup;
3. minimise the same one-dimensional AT2 surface-energy functional on that
   staircase (two-point Gauss elements, Thomas solve, identical discretisation
   constants to the pre-calibration) with ``d = 1`` on the centreline;
4. report the per-station effective ratio against the continuum target.

Limitations, stated deliberately: the transect model is one-dimensional, so it
audits the material field as discretised, not a two-dimensional inclined crack
solution; transect tails that leave the physical domain evaluate the analytic
band extension (their Gaussian decay is negligible there); the lower-left /
upper-right triangle split assumed for ``diagonal = "right"`` shifts barycentres
by at most ``h/3`` relative to the mirrored convention, which is sub-cell
against a band width of ``ell``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, load_config
from .gb_graph import boundary_field_from_config
from .interface_calibration import (
    _homogeneous_discrete_energy,
    _minimum_discrete_profile_and_energy,
)

BAND_ENERGY_AUDIT_SCHEMA_VERSION = 1
DEFAULT_DOMAIN_HALF_WIDTH = 12.0
DEFAULT_ELEMENTS = 2048
DEFAULT_EDGE_STATIONS = 17
NEAR_IMPACT_PARAMETERS = (0.95, 0.98, 0.02, 0.05)

__all__ = ["BAND_ENERGY_AUDIT_SCHEMA_VERSION", "audit_band_energy", "main"]


class _Dg0Sampler:
    """Piecewise-constant lookup on the structured right-diagonal triangulation."""

    def __init__(self, config: RunConfig, values_at) -> None:
        g = config.geometry
        self.nx, self.ny = int(g.nx), int(g.ny)
        self.hx, self.hy = g.length / self.nx, g.height / self.ny
        self.length, self.height = float(g.length), float(g.height)
        cell_x = (np.arange(self.nx) + 0.0) * self.hx
        cell_y = (np.arange(self.ny) + 0.0) * self.hy
        x0, y0 = np.meshgrid(cell_x, cell_y, indexing="ij")
        lower = np.stack(
            (x0 + self.hx / 3.0, y0 + self.hy / 3.0), axis=0
        ).reshape(2, -1)
        upper = np.stack(
            (x0 + 2.0 * self.hx / 3.0, y0 + 2.0 * self.hy / 3.0), axis=0
        ).reshape(2, -1)
        self.lower_values = np.asarray(values_at(lower), dtype=float).reshape(
            self.nx, self.ny
        )
        self.upper_values = np.asarray(values_at(upper), dtype=float).reshape(
            self.nx, self.ny
        )
        self._values_at = values_at

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Return DG0 values at ``points`` with analytic fallback outside."""
        x = np.asarray(points[0], dtype=float)
        y = np.asarray(points[1], dtype=float)
        inside = (x >= 0.0) & (x <= self.length) & (y >= 0.0) & (y <= self.height)
        result = np.empty(x.shape, dtype=float)
        if np.any(~inside):
            outside = np.stack((x[~inside], y[~inside]))
            result[~inside] = np.asarray(self._values_at(outside), dtype=float)
        if np.any(inside):
            xi = np.clip((x[inside] / self.hx).astype(int), 0, self.nx - 1)
            yj = np.clip((y[inside] / self.hy).astype(int), 0, self.ny - 1)
            local_x = x[inside] / self.hx - xi
            local_y = y[inside] / self.hy - yj
            is_lower = local_x + local_y < 1.0
            picked = np.where(
                is_lower, self.lower_values[xi, yj], self.upper_values[xi, yj]
            )
            result[inside] = picked
        return result


def _transect_effective_ratio(
    sampler,
    station: np.ndarray,
    normal: np.ndarray,
    length_scale: float,
    domain_half_width: float,
    elements: int,
) -> float:
    """Minimise the 1D AT2 energy on the sampled staircase, both half-domains."""
    h = domain_half_width / elements
    left_edges = np.arange(elements, dtype=float) * h
    midpoint = left_edges + 0.5 * h
    xi = 1.0 / math.sqrt(3.0)
    quadrature_y = np.stack((midpoint - 0.5 * h * xi, midpoint + 0.5 * h * xi))
    weights = 0.5 * h
    n_left = np.array(((1.0 + xi) / 2.0, (1.0 - xi) / 2.0))[:, None]
    n_right = np.array(((1.0 - xi) / 2.0, (1.0 + xi) / 2.0))[:, None]
    derivative_product = 1.0 / h**2

    energy = 0.0
    for sign in (1.0, -1.0):
        physical = (
            station[:, None, None]
            + sign * normal[:, None, None] * quadrature_y[None, :, :] * length_scale
        )
        g = sampler(physical.reshape(2, -1)).reshape(quadrature_y.shape)
        k00 = weights * np.sum(g * (n_left**2 + derivative_product), axis=0)
        k11 = weights * np.sum(g * (n_right**2 + derivative_product), axis=0)
        k01 = weights * np.sum(g * (n_left * n_right - derivative_product), axis=0)
        energy += _minimum_discrete_profile_and_energy(k00, k01, k11)[1]
    return energy / (2.0 * _homogeneous_discrete_energy(domain_half_width, elements))


def audit_band_energy(
    config_path: Path,
    *,
    edge_stations: int = DEFAULT_EDGE_STATIONS,
    domain_half_width: float = DEFAULT_DOMAIN_HALF_WIDTH,
    elements: int = DEFAULT_ELEMENTS,
) -> dict[str, Any]:
    config = load_config(Path(config_path))
    if not config.graph.enabled or not config.graph_edges:
        raise ValueError("configuration does not define an interface graph")
    field = boundary_field_from_config(config)
    sampler = _Dg0Sampler(config, field.toughness_ratio_at)
    length_scale = config.material.length_scale

    nodes = {node.name: np.asarray(node.point, dtype=float) for node in config.graph_nodes}
    stations: list[dict[str, Any]] = []
    for edge in config.graph_edges:
        source, target = nodes[edge.source], nodes[edge.target]
        tangent = target - source
        edge_length = float(np.linalg.norm(tangent))
        tangent = tangent / edge_length
        normal = np.array((-tangent[1], tangent[0]))
        interior = np.linspace(0.1, 0.9, edge_stations)
        for kind, parameters in (
            ("interior", interior),
            ("near_impact", np.asarray(NEAR_IMPACT_PARAMETERS)),
        ):
            for t in parameters:
                point = source + float(t) * edge_length * tangent
                ratio = _transect_effective_ratio(
                    sampler, point, normal, length_scale, domain_half_width, elements
                )
                stations.append(
                    {
                        "edge": f"{edge.source}->{edge.target}",
                        "parameter": float(t),
                        "kind": kind,
                        "point": [float(point[0]), float(point[1])],
                        "effective_toughness_ratio": ratio,
                    }
                )

    interior_values = [
        s["effective_toughness_ratio"] for s in stations if s["kind"] == "interior"
    ]
    near_impact_values = [
        s["effective_toughness_ratio"] for s in stations if s["kind"] == "near_impact"
    ]
    geometry = config.geometry
    return {
        "schema_version": BAND_ENERGY_AUDIT_SCHEMA_VERSION,
        "study": "mesh_level_DG0_inclined_band_surface_energy_audit",
        "config": str(Path(config_path)),
        "mesh": {
            "nx": geometry.nx,
            "ny": geometry.ny,
            "diagonal_convention_assumed": "right: lower-left/upper-right split",
            "cell_size": [sampler.hx, sampler.hy],
            "ell_over_cell": length_scale / max(sampler.hx, sampler.hy),
        },
        "band": {
            "centerline_toughness_ratio": min(
                edge.toughness_ratio for edge in config.graph_edges
            ),
            "influence_radius": config.graph.influence_radius,
            "width_over_length_scale": config.graph.influence_radius / length_scale,
        },
        "one_dimensional_model": {
            "domain_half_width_over_ell": domain_half_width,
            "elements_per_half_domain": elements,
            "boundary_conditions": "d(centreline)=1, d(+/-L)=0, asymmetric halves",
            "normalisation": "two half-domain homogeneous energies on the same mesh",
        },
        "interior_stations": {
            "count": len(interior_values),
            "minimum": min(interior_values),
            "median": float(np.median(interior_values)),
            "maximum": max(interior_values),
        },
        "near_impact_stations": {
            "count": len(near_impact_values),
            "minimum": min(near_impact_values),
            "median": float(np.median(near_impact_values)),
            "maximum": max(near_impact_values),
        },
        "stations": stations,
        "interpretation_limit": (
            "One-dimensional transect audit of the DG0-projected band; it is not "
            "a two-dimensional inclined-crack computation and does not replace "
            "the mirror/refinement/increment gates."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m graphfracture.band_energy_audit",
        description="Audit the mesh-level surface energy of the inclined DG0 weak band.",
    )
    parser.add_argument("config", type=Path, help="TOML configuration with graph nodes/edges")
    parser.add_argument("--edge-stations", type=int, default=DEFAULT_EDGE_STATIONS)
    parser.add_argument("--domain-half-width", type=float, default=DEFAULT_DOMAIN_HALF_WIDTH)
    parser.add_argument("--elements", type=int, default=DEFAULT_ELEMENTS)
    parser.add_argument("--output", "-o", type=Path, help="exclusive-create JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = audit_band_energy(
            args.config,
            edge_stations=args.edge_stations,
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
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"band-energy-audit error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
