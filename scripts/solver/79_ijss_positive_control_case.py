"""Run one resumable strong-contrast DG0 or CG1 positive-control case."""

from __future__ import annotations

import argparse
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
GC = 0.06163
ELL = 0.0015
BANDWIDTH = 0.0030
TOUGHNESS_RATIO = 0.20


class NodalMaterialAT2Solver(AT2Solver):
    def _create_mesh_and_spaces(self) -> None:
        super()._create_mesh_and_spaces()
        self.V_0 = self.V_d


def config(nx: int, target: float, output: Path) -> RunConfig:
    mid = 0.5 * HEIGHT
    increment = 2.0e-6
    steps = int(round(target / increment))
    if not math.isclose(steps * increment, target, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("target displacement must be an integer multiple of 2e-6 mm")
    result = RunConfig(
        geometry=GeometryConfig(
            length=LENGTH,
            height=HEIGHT,
            nx=nx,
            ny=nx // 2,
            precrack_length=PRECRACK,
            diagonal="right",
        ),
        material=MaterialConfig(
            young_modulus=210_000.0,
            poisson_ratio=0.30,
            fracture_toughness=GC,
            length_scale=ELL,
            residual_stiffness=1.0e-8,
        ),
        loading=LoadingConfig(
            maximum_displacement=target,
            steps=steps,
            stagger_max_iterations=320,
            stagger_tolerance=1.0e-5,
            damage_kkt_tolerance=1.0e-6,
            adaptive=True,
            maximum_subdivisions=10,
            minimum_increment=1.0e-11,
        ),
        graph=GraphConfig(enabled=True, influence_radius=BANDWIDTH, crack_threshold=0.7),
        output=OutputConfig(directory=str(output), write_every=steps),
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
    result.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--representation", choices=("DG0", "CG1"), required=True)
    parser.add_argument("--target", type=float, default=3.0e-5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.nx < 8 or args.nx % 4:
        raise ValueError("nx must be a multiple of four")
    output = args.output.resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(output)
    elif output.exists():
        raise FileExistsError(output)

    configuration = config(args.nx, args.target, output)
    solver_type = AT2Solver if args.representation == "DG0" else NodalMaterialAT2Solver
    start = time.perf_counter()
    solver = solver_type(configuration)
    area = LENGTH * HEIGHT
    mean_gc0 = solver._global_scalar(fem.form(solver.Gc0 * ufl.dx)) / area
    owned = solver.Gc0.function_space.dofmap.index_map.size_local
    local_min = float(np.min(solver.Gc0.x.array[:owned].real, initial=math.inf))
    minimum_gc0 = float(solver.comm.allreduce(local_min, op=MPI.MIN))
    history = solver.run(resume=args.resume)
    final = history[-1]
    if not math.isclose(
        float(final["displacement"]), args.target, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError("positive-control case did not reach the matched target")
    discrete, traction = solver._elastic_reactions()
    result = {
        "schema_version": 1,
        "representation": args.representation,
        "nx": args.nx,
        "ny": args.nx // 2,
        "target_displacement_mm": args.target,
        "h_diagonal_mm": math.hypot(LENGTH / args.nx, HEIGHT / (args.nx // 2)),
        "ell_over_hK": ELL / math.hypot(LENGTH / args.nx, HEIGHT / (args.nx // 2)),
        "b_over_hK": BANDWIDTH
        / math.hypot(LENGTH / args.nx, HEIGHT / (args.nx // 2)),
        "ell_over_b": ELL / BANDWIDTH,
        "centerline_toughness_ratio": TOUGHNESS_RATIO,
        "mean_Gc0_N_per_mm": mean_gc0,
        "minimum_Gc0_N_per_mm": minimum_gc0,
        "reaction_N_per_mm": discrete,
        "traction_reaction_N_per_mm": traction,
        "reaction_crosscheck_relative_pct": 100.0
        * abs(traction - discrete)
        / max(abs(discrete), 1.0e-30),
        "regularised_crack_length_mm": float(final["regularised_crack_length"]),
        "crack_measure_increase_pct": 100.0
        * (
            float(final["regularised_crack_length"])
            / float(history[0]["regularised_crack_length"])
            - 1.0
        ),
        "maximum_damage_kkt_relative": max(
            float(row["damage_kkt_relative"]) for row in history
        ),
        "accepted_states": len(history),
        "wall_time_s": time.perf_counter() - start,
        "status": "matched_endpoint_complete",
        "claim_limit": "synthetic numerical positive control; not a calibrated interface law",
    }
    if MPI.COMM_WORLD.rank == 0:
        (output / "positive_control_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
