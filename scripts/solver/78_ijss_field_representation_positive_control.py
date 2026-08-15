"""DG0-versus-nodal graph-field comparison and strong positive control."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from graphfracture.config import (
    GeometryConfig,
    GraphConfig,
    GraphEdgeConfig,
    GraphNodeConfig,
    LoadingConfig,
    MaterialConfig,
    OutputConfig,
    RunConfig,
    SolverConfig,
)
from graphfracture.dolfinx_solver import AT2Solver


LENGTH = 0.09639
HEIGHT = 0.048195
PRECRACK = 0.0240975
YOUNG = 210_000.0
POISSON = 0.30
GC = 0.06163
ELL = 0.0015
BANDWIDTH = 0.0030
TOUGHNESS_RATIO = 0.20
TARGET_DISPLACEMENT = 4.0e-5


class NodalMaterialAT2Solver(AT2Solver):
    """Use continuous P1 nodal interpolation for graph-derived coefficients."""

    def _create_mesh_and_spaces(self) -> None:
        super()._create_mesh_and_spaces()
        self.V_0 = self.V_d


def parse_meshes(value: str) -> tuple[int, ...]:
    levels = tuple(sorted(int(item.strip()) for item in value.split(",") if item.strip()))
    if not levels or any(level < 8 or level % 4 for level in levels):
        raise argparse.ArgumentTypeError("mesh nx values must be unique multiples of four")
    if len(levels) != len(set(levels)):
        raise argparse.ArgumentTypeError("mesh nx values must be unique")
    return levels


def configuration(nx: int, output: Path) -> RunConfig:
    mid = 0.5 * HEIGHT
    config = RunConfig(
        geometry=GeometryConfig(
            length=LENGTH,
            height=HEIGHT,
            nx=nx,
            ny=nx // 2,
            precrack_length=PRECRACK,
            diagonal="right",
        ),
        material=MaterialConfig(
            young_modulus=YOUNG,
            poisson_ratio=POISSON,
            fracture_toughness=GC,
            length_scale=ELL,
            residual_stiffness=1.0e-8,
        ),
        loading=LoadingConfig(
            maximum_displacement=TARGET_DISPLACEMENT,
            steps=20,
            stagger_max_iterations=320,
            stagger_tolerance=1.0e-5,
            damage_kkt_tolerance=1.0e-6,
            adaptive=True,
            maximum_subdivisions=10,
            minimum_increment=1.0e-11,
        ),
        graph=GraphConfig(
            enabled=True,
            influence_radius=BANDWIDTH,
            crack_threshold=0.70,
        ),
        output=OutputConfig(directory=str(output), write_every=20),
        solver=SolverConfig(),
        graph_nodes=(
            GraphNodeConfig("left", (0.0, mid)),
            GraphNodeConfig("right", (LENGTH, mid)),
        ),
        graph_edges=(
            GraphEdgeConfig(
                "left",
                "right",
                toughness_ratio=TOUGHNESS_RATIO,
                hydrogen_diffusivity_ratio=1.0,
                trap_density=0.0,
            ),
        ),
    )
    config.validate()
    return config


def run_case(
    solver_type: type[AT2Solver], config: RunConfig, representation: str
) -> dict[str, float | int | str]:
    output = Path(config.output.directory)
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    solver = solver_type(config)
    area = LENGTH * HEIGHT
    mean_gc0 = solver._global_scalar(fem.form(solver.Gc0 * ufl.dx)) / area
    owned = solver.Gc0.function_space.dofmap.index_map.size_local
    local_min = float(np.min(solver.Gc0.x.array[:owned].real, initial=math.inf))
    local_max = float(np.max(solver.Gc0.x.array[:owned].real, initial=-math.inf))
    gc_min = float(solver.comm.allreduce(local_min, op=MPI.MIN))
    gc_max = float(solver.comm.allreduce(local_max, op=MPI.MAX))
    history = solver.run()
    first = history[0]
    last = history[-1]
    if not math.isclose(
        float(last["displacement"]), TARGET_DISPLACEMENT, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError(f"{representation} branch did not reach the common endpoint")
    discrete, traction = solver._elastic_reactions()
    result: dict[str, float | int | str] = {
        "representation": representation,
        "nx": config.geometry.nx,
        "ny": config.geometry.ny,
        "h_diagonal_mm": math.hypot(
            LENGTH / config.geometry.nx, HEIGHT / config.geometry.ny
        ),
        "ell_over_hK": ELL
        / math.hypot(LENGTH / config.geometry.nx, HEIGHT / config.geometry.ny),
        "b_over_hK": BANDWIDTH
        / math.hypot(LENGTH / config.geometry.nx, HEIGHT / config.geometry.ny),
        "ell_over_b": ELL / BANDWIDTH,
        "toughness_ratio": TOUGHNESS_RATIO,
        "mean_Gc0_N_per_mm": mean_gc0,
        "minimum_Gc0_N_per_mm": gc_min,
        "maximum_Gc0_N_per_mm": gc_max,
        "discrete_reaction_N_per_mm": discrete,
        "traction_reaction_N_per_mm": traction,
        "reaction_crosscheck_relative_pct": 100.0
        * abs(traction - discrete)
        / max(abs(discrete), 1.0e-30),
        "regularised_crack_length_mm": float(last["regularised_crack_length"]),
        "crack_measure_increase_pct": 100.0
        * (
            float(last["regularised_crack_length"])
            / float(first["regularised_crack_length"])
            - 1.0
        ),
        "maximum_damage_kkt_relative": max(
            float(row["damage_kkt_relative"]) for row in history
        ),
        "accepted_states": len(history),
        "wall_time_s": time.perf_counter() - start,
    }
    del solver
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", type=parse_meshes, default=parse_meshes("184,276,368,552"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ijss_m4_field_representation_positive_control_v1"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if MPI.COMM_WORLD.rank == 0:
        output.mkdir(parents=True)
    MPI.COMM_WORLD.barrier()
    rows: list[dict[str, float | int | str]] = []
    for nx in args.meshes:
        ny = nx // 2
        for representation, solver_type in (
            ("DG0_centroid", AT2Solver),
            ("CG1_nodal", NodalMaterialAT2Solver),
        ):
            if MPI.COMM_WORLD.rank == 0:
                print(f"positive control {nx}x{ny}: {representation}", flush=True)
            config = configuration(nx, output / f"{representation}_{nx}x{ny}")
            rows.append(run_case(solver_type, config, representation))
    if MPI.COMM_WORLD.rank == 0:
        fields = list(rows[0])
        with (output / "field_representation_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema_version": 1,
            "purpose": "IJSS M4 field representation and M5 positive control",
            "synthetic_control": {
                "geometry": "horizontal weak band through the precrack and ligament",
                "centerline_toughness_ratio": TOUGHNESS_RATIO,
                "bandwidth_mm": BANDWIDTH,
                "ell_over_b": ELL / BANDWIDTH,
            },
            "representations": {
                "DG0_centroid": "piecewise-constant cell-centroid interpolation",
                "CG1_nodal": "continuous piecewise-linear nodal interpolation",
            },
            "mesh_nx": list(args.meshes),
            "mpi_ranks": MPI.COMM_WORLD.size,
            "claim_limit": "synthetic numerical positive control; not a calibrated grain-boundary law",
        }
        (output / "field_representation_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(output / "field_representation_results.csv")


if __name__ == "__main__":
    main()
