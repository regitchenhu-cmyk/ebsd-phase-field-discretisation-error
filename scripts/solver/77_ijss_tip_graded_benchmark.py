"""Equal-element-count crack-tip grading comparison for IJSS M4."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from graphfracture.config import (
    GeometryConfig,
    GraphConfig,
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
TARGET_DISPLACEMENT = 4.0e-5


class TipGradedAT2Solver(AT2Solver):
    grading_ratio = 0.25
    grading_power = 2.0

    def _create_mesh_and_spaces(self) -> None:
        super()._create_mesh_and_spaces()
        coordinates = self.domain.geometry.x
        x = coordinates[:, 0].copy()
        y = coordinates[:, 1].copy()

        left = x <= PRECRACK
        q_left = np.clip(x[left] / PRECRACK, 0.0, 1.0)
        coordinates[left, 0] = PRECRACK * (
            self.grading_ratio * q_left
            + (1.0 - self.grading_ratio)
            * (1.0 - (1.0 - q_left) ** self.grading_power)
        )
        right = ~left
        q_right = np.clip(
            (x[right] - PRECRACK) / (LENGTH - PRECRACK), 0.0, 1.0
        )
        coordinates[right, 0] = PRECRACK + (LENGTH - PRECRACK) * (
            self.grading_ratio * q_right
            + (1.0 - self.grading_ratio) * q_right**self.grading_power
        )

        half = 0.5 * HEIGHT
        signed = y - half
        q_y = np.clip(np.abs(signed) / half, 0.0, 1.0)
        mapped = half * (
            self.grading_ratio * q_y
            + (1.0 - self.grading_ratio) * q_y**self.grading_power
        )
        coordinates[:, 1] = half + np.sign(signed) * mapped

    def mesh_edge_statistics(self) -> dict[str, float]:
        geometry_map = self.domain.geometry.dofmaps[0]
        coordinates = self.domain.geometry.x
        local_min = math.inf
        local_max = 0.0
        local_tip_max = 0.0
        local_tip_min = math.inf
        for cell_nodes in geometry_map:
            points = coordinates[cell_nodes, :2]
            edges = (
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[1]),
                np.linalg.norm(points[0] - points[2]),
            )
            cell_min = float(min(edges))
            cell_max = float(max(edges))
            local_min = min(local_min, cell_min)
            local_max = max(local_max, cell_max)
            centroid = np.mean(points, axis=0)
            if np.linalg.norm(centroid - np.array((PRECRACK, 0.5 * HEIGHT))) <= ELL:
                local_tip_min = min(local_tip_min, cell_min)
                local_tip_max = max(local_tip_max, cell_max)
        comm = self.comm
        return {
            "minimum_edge_mm": float(comm.allreduce(local_min, op=MPI.MIN)),
            "maximum_edge_mm": float(comm.allreduce(local_max, op=MPI.MAX)),
            "tip_minimum_edge_mm": float(comm.allreduce(local_tip_min, op=MPI.MIN)),
            "tip_maximum_edge_mm": float(comm.allreduce(local_tip_max, op=MPI.MAX)),
        }


def parse_meshes(value: str) -> tuple[int, ...]:
    levels = tuple(sorted(int(item.strip()) for item in value.split(",") if item.strip()))
    if not levels or any(level < 8 or level % 4 for level in levels):
        raise argparse.ArgumentTypeError("mesh nx values must be unique multiples of four")
    if len(levels) != len(set(levels)):
        raise argparse.ArgumentTypeError("mesh nx values must be unique")
    return levels


def configuration(nx: int, output: Path) -> RunConfig:
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
        graph=GraphConfig(enabled=False, influence_radius=ELL),
        output=OutputConfig(directory=str(output), write_every=20),
        solver=SolverConfig(),
    )
    config.validate()
    return config


def reactions(solver: AT2Solver) -> dict[str, float]:
    discrete, traction = solver._elastic_reactions()
    return {
        "discrete_reaction_N_per_mm": discrete,
        "traction_reaction_N_per_mm": traction,
        "reaction_crosscheck_relative_pct": 100.0
        * abs(traction - discrete)
        / max(abs(discrete), 1.0e-30),
    }


def frozen(config: RunConfig) -> dict[str, float]:
    start = time.perf_counter()
    solver = TipGradedAT2Solver(config)
    mesh_stats = solver.mesh_edge_statistics()
    solve = solver._solve_load_step(0.0)
    if not solve["converged"]:
        raise RuntimeError(f"zero-load graded crack failed: {solve}")
    initial_length = solver._global_scalar(solver.crack_length_form)
    solver.load.value = TARGET_DISPLACEMENT
    solver.elastic_problem.solve()
    solver.u.x.scatter_forward()
    result = {
        **mesh_stats,
        **reactions(solver),
        "regularised_crack_length_mm": solver._global_scalar(solver.crack_length_form),
        "damage_increment_pct": 0.0,
        "wall_time_s": time.perf_counter() - start,
        "zero_load_regularised_crack_length_mm": initial_length,
    }
    del solver
    gc.collect()
    return result


def coupled(config: RunConfig) -> dict[str, float]:
    output = Path(config.output.directory)
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    solver = TipGradedAT2Solver(config)
    mesh_stats = solver.mesh_edge_statistics()
    history = solver.run()
    first = history[0]
    last = history[-1]
    if not math.isclose(
        float(last["displacement"]), TARGET_DISPLACEMENT, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError("graded coupled branch did not reach the common endpoint")
    result = {
        **mesh_stats,
        **reactions(solver),
        "regularised_crack_length_mm": float(last["regularised_crack_length"]),
        "zero_load_regularised_crack_length_mm": float(
            first["regularised_crack_length"]
        ),
        "damage_increment_pct": 100.0
        * (
            float(last["regularised_crack_length"])
            / float(first["regularised_crack_length"])
            - 1.0
        ),
        "maximum_damage_kkt_relative": max(
            float(row["damage_kkt_relative"]) for row in history
        ),
        "wall_time_s": time.perf_counter() - start,
    }
    del solver
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", type=parse_meshes, default=parse_meshes("184,276,368"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ijss_m4_tip_graded_benchmark_v1"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if MPI.COMM_WORLD.rank == 0:
        output.mkdir(parents=True)
    MPI.COMM_WORLD.barrier()
    rows: list[dict[str, object]] = []
    for nx in args.meshes:
        ny = nx // 2
        config = configuration(nx, output / f"coupled_{nx}x{ny}")
        if MPI.COMM_WORLD.rank == 0:
            print(f"graded {nx}x{ny}: frozen", flush=True)
        rows.append(
            {
                "mesh_type": "tip_graded",
                "mode": "frozen_edge_crack",
                "nx": nx,
                "ny": ny,
                "grading_ratio": TipGradedAT2Solver.grading_ratio,
                "grading_power": TipGradedAT2Solver.grading_power,
                **frozen(config),
            }
        )
        if MPI.COMM_WORLD.rank == 0:
            print(f"graded {nx}x{ny}: coupled", flush=True)
        rows.append(
            {
                "mesh_type": "tip_graded",
                "mode": "coupled_at2",
                "nx": nx,
                "ny": ny,
                "grading_ratio": TipGradedAT2Solver.grading_ratio,
                "grading_power": TipGradedAT2Solver.grading_power,
                **coupled(config),
            }
        )
    if MPI.COMM_WORLD.rank == 0:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (output / "tip_graded_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema_version": 1,
            "purpose": "equal-element-count crack-tip grading comparison",
            "grading_ratio": TipGradedAT2Solver.grading_ratio,
            "grading_power": TipGradedAT2Solver.grading_power,
            "mesh_nx": list(args.meshes),
            "mpi_ranks": MPI.COMM_WORLD.size,
            "interpretation_limit": (
                "smooth coordinate grading, not topological h-adaptivity or a singular element"
            ),
        }
        (output / "tip_graded_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(output / "tip_graded_results.csv")


if __name__ == "__main__":
    main()
