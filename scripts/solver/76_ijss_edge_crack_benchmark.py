"""M4 edge-crack verification sequence for the IJSS revision.

The benchmark separates two mechanisms on the same structured meshes:

1. a zero-load equilibrated AT2 edge crack is frozen and only elasticity is
   solved at the common displacement;
2. the ordinary irreversible AT2 model is advanced incrementally to the same
   displacement.

The finest mesh is an over-resolved same-model numerical reference. No GCI is
reported unless an asymptotic sequence is independently demonstrated.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import socket
import time
from dataclasses import asdict
from pathlib import Path

import dolfinx
import mpi4py
import numpy as np
import petsc4py
import ufl
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


def parse_meshes(value: str) -> tuple[int, ...]:
    meshes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not meshes or any(nx < 8 or nx % 2 for nx in meshes):
        raise argparse.ArgumentTypeError("meshes must be comma-separated even integers >= 8")
    if len(set(meshes)) != len(meshes):
        raise argparse.ArgumentTypeError("mesh levels must be unique")
    return tuple(sorted(meshes))


def configuration(nx: int, output: Path, residual_stiffness: float) -> RunConfig:
    ny = nx // 2
    config = RunConfig(
        geometry=GeometryConfig(
            length=LENGTH,
            height=HEIGHT,
            nx=nx,
            ny=ny,
            precrack_length=PRECRACK,
            diagonal="right",
            x_pin_corner="bottom_left",
        ),
        material=MaterialConfig(
            young_modulus=YOUNG,
            poisson_ratio=POISSON,
            fracture_toughness=GC,
            length_scale=ELL,
            residual_stiffness=residual_stiffness,
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
        solver=SolverConfig(
            linear_ksp_type="preonly",
            linear_pc_type="lu",
            factor_solver="mumps",
            damage_snes_type="vinewtonrsls",
        ),
    )
    config.validate()
    return config


def scalar_metrics(solver: AT2Solver) -> dict[str, float]:
    discrete, traction = solver._elastic_reactions()
    return {
        "discrete_reaction_N_per_mm": discrete,
        "traction_reaction_N_per_mm": traction,
        "reaction_crosscheck_relative_pct": 100.0
        * abs(traction - discrete)
        / max(abs(discrete), 1.0e-30),
        "elastic_energy_N_mm": solver._global_scalar(solver.elastic_energy_form),
        "fracture_energy_N_mm": solver._global_scalar(solver.fracture_energy_form),
        "regularised_crack_length_mm": solver._global_scalar(solver.crack_length_form),
        "maximum_damage": solver._max_damage(),
    }


def frozen_edge_crack(config: RunConfig) -> dict[str, float]:
    start = time.perf_counter()
    solver = AT2Solver(config)
    zero_info = solver._solve_load_step(0.0)
    if not zero_info["converged"]:
        raise RuntimeError(f"zero-load crack equilibration failed: {zero_info}")
    zero_crack = solver._global_scalar(solver.crack_length_form)
    zero_damage = solver._max_damage()
    solver.load.value = TARGET_DISPLACEMENT
    solver.elastic_problem.solve()
    solver.u.x.scatter_forward()
    metrics = scalar_metrics(solver)
    metrics.update(
        {
            "zero_load_regularised_crack_length_mm": zero_crack,
            "zero_load_maximum_damage": zero_damage,
            "damage_increment_pct": 0.0,
            "wall_time_s": time.perf_counter() - start,
        }
    )
    del solver
    gc.collect()
    return metrics


def coupled_edge_crack(config: RunConfig) -> dict[str, float]:
    output = Path(config.output.directory)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing coupled benchmark output: {output}"
        )
    start = time.perf_counter()
    solver = AT2Solver(config)
    initial_crack = solver._global_scalar(solver.crack_length_form)
    history = solver.run()
    final = history[-1]
    if not math.isclose(
        float(final["displacement"]), TARGET_DISPLACEMENT, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError(f"coupled run did not reach the common endpoint: {final}")
    metrics = scalar_metrics(solver)
    metrics.update(
        {
            "zero_load_regularised_crack_length_mm": float(
                history[0]["regularised_crack_length"]
            ),
            "zero_load_maximum_damage": float(history[0]["maximum_damage"]),
            "damage_increment_pct": 100.0
            * (
                float(final["regularised_crack_length"])
                / max(float(history[0]["regularised_crack_length"]), 1.0e-30)
                - 1.0
            ),
            "wall_time_s": time.perf_counter() - start,
            "accepted_states": len(history),
            "initial_seed_crack_length_mm": initial_crack,
            "maximum_damage_kkt_relative": max(
                float(row["damage_kkt_relative"]) for row in history
            ),
            "maximum_energy_balance_relative": max(
                abs(float(row["energy_balance_relative"])) for row in history
            ),
        }
    )
    del solver
    gc.collect()
    return metrics


def reference_errors(rows: list[dict[str, object]], mode: str) -> None:
    selected = [row for row in rows if row["mode"] == mode]
    reference = min(selected, key=lambda row: float(row["h_diagonal_mm"]))
    reference_reaction = float(reference["discrete_reaction_N_per_mm"])
    for row in selected:
        row["relative_to_finest_reference_pct"] = 100.0 * abs(
            float(row["discrete_reaction_N_per_mm"]) - reference_reaction
        ) / max(abs(reference_reaction), 1.0e-30)
        row["reference_mesh"] = f"{reference['nx']}x{reference['ny']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meshes",
        type=parse_meshes,
        default=parse_meshes("184,276,368,460,552"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ijss_m4_edge_crack_benchmark"),
    )
    parser.add_argument("--residual-stiffness", type=float, default=1.0e-8)
    parser.add_argument(
        "--frozen-only",
        action="store_true",
        help="skip the incrementally coupled AT2 sequence",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark directory: {output}")
    if MPI.COMM_WORLD.rank == 0:
        output.mkdir(parents=True)
    MPI.COMM_WORLD.barrier()

    rows: list[dict[str, object]] = []
    for nx in args.meshes:
        ny = nx // 2
        h_diagonal = math.hypot(LENGTH / nx, HEIGHT / ny)
        coupled_output = output / f"coupled_{nx}x{ny}"
        config = configuration(nx, coupled_output, args.residual_stiffness)
        if MPI.COMM_WORLD.rank == 0:
            print(f"M4 benchmark mesh {nx}x{ny}: frozen edge crack", flush=True)
        frozen = frozen_edge_crack(config)
        row = {
            "mode": "frozen_edge_crack",
            "nx": nx,
            "ny": ny,
            "h_diagonal_mm": h_diagonal,
            "ell_over_hK": ELL / h_diagonal,
            "residual_stiffness": args.residual_stiffness,
            **frozen,
        }
        rows.append(row)
        if not args.frozen_only:
            if MPI.COMM_WORLD.rank == 0:
                print(f"M4 benchmark mesh {nx}x{ny}: coupled AT2", flush=True)
            coupled = coupled_edge_crack(config)
            rows.append(
                {
                    "mode": "coupled_at2",
                    "nx": nx,
                    "ny": ny,
                    "h_diagonal_mm": h_diagonal,
                    "ell_over_hK": ELL / h_diagonal,
                    "residual_stiffness": args.residual_stiffness,
                    **coupled,
                }
            )

    if MPI.COMM_WORLD.rank == 0:
        reference_errors(rows, "frozen_edge_crack")
        if not args.frozen_only:
            reference_errors(rows, "coupled_at2")
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (output / "benchmark_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema_version": 1,
            "purpose": "IJSS M4 homogeneous edge-crack error-source attribution",
            "interpretation": (
                "finest-level same-model reference; no Richardson/GCI claim without an "
                "independently demonstrated asymptotic sequence"
            ),
            "parameters": {
                "length_mm": LENGTH,
                "height_mm": HEIGHT,
                "precrack_length_mm": PRECRACK,
                "young_modulus_MPa": YOUNG,
                "poisson_ratio": POISSON,
                "fracture_toughness_N_per_mm": GC,
                "length_scale_mm": ELL,
                "target_displacement_mm": TARGET_DISPLACEMENT,
                "residual_stiffness": args.residual_stiffness,
                "mesh_nx": list(args.meshes),
            },
            "runtime": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "dolfinx": dolfinx.__version__,
                "petsc4py": petsc4py.__version__,
                "mpi4py": mpi4py.__version__,
                "ufl": ufl.__version__,
                "mpi_ranks": MPI.COMM_WORLD.size,
                "cpu_count_visible": os.cpu_count(),
            },
            "run_config_template": asdict(configuration(args.meshes[0], Path("OUTPUT"), args.residual_stiffness)),
        }
        (output / "benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(output / "benchmark_results.csv")
        print(output / "benchmark_manifest.json")


if __name__ == "__main__":
    main()
