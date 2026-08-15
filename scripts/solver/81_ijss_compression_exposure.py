"""Quantify compression exposure in an accepted plane-strain edge-crack state.

This postprocessor does not alter or re-solve the accepted AT2 state. It reads
the authenticated restart checkpoint and evaluates the fixed-state terms that
would be treated differently by an Amor-type volumetric-deviatoric split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
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
RESIDUAL_STIFFNESS = 1.0e-8


def configuration(nx: int, source: Path) -> RunConfig:
    config = RunConfig(
        geometry=GeometryConfig(
            length=LENGTH,
            height=HEIGHT,
            nx=nx,
            ny=nx // 2,
            precrack_length=PRECRACK,
            diagonal="right",
            x_pin_corner="bottom_left",
        ),
        material=MaterialConfig(
            young_modulus=YOUNG,
            poisson_ratio=POISSON,
            fracture_toughness=GC,
            length_scale=ELL,
            residual_stiffness=RESIDUAL_STIFFNESS,
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
        output=OutputConfig(directory=str(source), write_every=20),
        solver=SolverConfig(
            linear_ksp_type="preonly",
            linear_pc_type="lu",
            factor_solver="mumps",
            damage_snes_type="vinewtonrsls",
        ),
    )
    config.validate()
    return config


def ratio(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / max(abs(denominator), 1.0e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    solver = AT2Solver(configuration(args.nx, source))
    loaded = solver._load_restart_checkpoint(source)
    accepted_step = int(loaded[-2])
    accepted_displacement = float(loaded[-1])
    if not math.isclose(
        accepted_displacement,
        TARGET_DISPLACEMENT,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            f"checkpoint ends at {accepted_displacement:.6e} mm, not the matched endpoint"
        )

    domain = solver.domain
    damage = solver.d
    strain_2d = solver._epsilon(solver.u)
    trace_strain = ufl.tr(strain_2d)
    identity_3d = ufl.Identity(3)
    strain_3d = ufl.as_tensor(
        (
            (strain_2d[0, 0], strain_2d[0, 1], 0.0),
            (strain_2d[1, 0], strain_2d[1, 1], 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    deviatoric_strain = strain_3d - trace_strain * identity_3d / 3.0
    shear_modulus = YOUNG / (2.0 * (1.0 + POISSON))
    bulk_modulus = YOUNG / (3.0 * (1.0 - 2.0 * POISSON))
    lame_lambda = (
        YOUNG
        * POISSON
        / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
    )

    negative_trace = ufl.conditional(ufl.lt(trace_strain, 0.0), trace_strain, 0.0)
    negative_volumetric_energy = 0.5 * bulk_modulus * negative_trace**2
    strain_radius = ufl.sqrt(
        (0.5 * (strain_2d[0, 0] - strain_2d[1, 1])) ** 2
        + strain_2d[0, 1] ** 2
    )
    minimum_in_plane_strain = 0.5 * trace_strain - strain_radius
    maximum_in_plane_strain = 0.5 * trace_strain + strain_radius
    negative_minimum_strain = ufl.conditional(
        ufl.lt(minimum_in_plane_strain, 0.0), minimum_in_plane_strain, 0.0
    )
    negative_maximum_strain = ufl.conditional(
        ufl.lt(maximum_in_plane_strain, 0.0), maximum_in_plane_strain, 0.0
    )
    spectral_negative_energy = (
        0.5 * lame_lambda * negative_trace**2
        + shear_modulus
        * (negative_minimum_strain**2 + negative_maximum_strain**2)
    )
    deviatoric_energy = shear_modulus * ufl.inner(deviatoric_strain, deviatoric_strain)
    total_undegraded_energy_density = (
        0.5 * bulk_modulus * trace_strain**2 + deviatoric_energy
    )
    degradation = solver._degradation(damage)
    compression_indicator = ufl.conditional(ufl.lt(trace_strain, 0.0), 1.0, 0.0)
    damaged_indicator = ufl.conditional(ufl.gt(damage, 0.1), 1.0, 0.0)
    damaged_compression_indicator = ufl.conditional(
        ufl.And(ufl.lt(trace_strain, 0.0), ufl.gt(damage, 0.1)), 1.0, 0.0
    )
    damage_driving_weight = 1.0 - damage

    dx = ufl.Measure("dx", domain=domain)
    scalars = {
        "area_mm2": solver._global_scalar(fem.form(1.0 * dx)),
        "undegraded_elastic_energy_N_mm": solver._global_scalar(
            fem.form(total_undegraded_energy_density * dx)
        ),
        "full_degradation_elastic_energy_N_mm": solver._global_scalar(
            fem.form(degradation * total_undegraded_energy_density * dx)
        ),
        "negative_volumetric_energy_N_mm": solver._global_scalar(
            fem.form(negative_volumetric_energy * dx)
        ),
        "fixed_state_split_energy_correction_N_mm": solver._global_scalar(
            fem.form((1.0 - degradation) * negative_volumetric_energy * dx)
        ),
        "spectral_negative_energy_N_mm": solver._global_scalar(
            fem.form(spectral_negative_energy * dx)
        ),
        "fixed_state_spectral_split_energy_correction_N_mm": solver._global_scalar(
            fem.form((1.0 - degradation) * spectral_negative_energy * dx)
        ),
        "damage_driving_integral_N_mm": solver._global_scalar(
            fem.form(damage_driving_weight * total_undegraded_energy_density * dx)
        ),
        "compressive_damage_driving_integral_N_mm": solver._global_scalar(
            fem.form(damage_driving_weight * negative_volumetric_energy * dx)
        ),
        "spectral_compressive_damage_driving_integral_N_mm": solver._global_scalar(
            fem.form(damage_driving_weight * spectral_negative_energy * dx)
        ),
        "compression_area_mm2": solver._global_scalar(
            fem.form(compression_indicator * dx)
        ),
        "damaged_area_d_gt_0p1_mm2": solver._global_scalar(
            fem.form(damaged_indicator * dx)
        ),
        "damaged_compression_area_mm2": solver._global_scalar(
            fem.form(damaged_compression_indicator * dx)
        ),
    }

    stress_2d = solver._sigma0(solver.u)
    stress_zz = lame_lambda * trace_strain
    in_plane_minimum = 0.5 * (stress_2d[0, 0] + stress_2d[1, 1]) - ufl.sqrt(
        (0.5 * (stress_2d[0, 0] - stress_2d[1, 1])) ** 2
        + stress_2d[0, 1] ** 2
    )
    minimum_principal_stress = ufl.min_value(in_plane_minimum, stress_zz)
    V0 = fem.functionspace(domain, ("DG", 0))
    interpolation_points = V0.element.interpolation_points
    principal_field = fem.Function(V0)
    principal_field.interpolate(fem.Expression(minimum_principal_stress, interpolation_points))
    trace_field = fem.Function(V0)
    trace_field.interpolate(fem.Expression(trace_strain, interpolation_points))

    reaction, traction_reaction = solver._elastic_reactions()
    result = {
        "schema_version": 1,
        "nx": args.nx,
        "ny": args.nx // 2,
        "ell_over_hK": ELL
        / math.hypot(LENGTH / args.nx, HEIGHT / (args.nx // 2)),
        "accepted_step": accepted_step,
        "accepted_displacement_mm": accepted_displacement,
        "reaction_N_per_mm": reaction,
        "traction_reaction_N_per_mm": traction_reaction,
        **scalars,
        "negative_volumetric_fraction_of_undegraded_energy_pct": ratio(
            scalars["negative_volumetric_energy_N_mm"],
            scalars["undegraded_elastic_energy_N_mm"],
        ),
        "fixed_state_split_correction_relative_to_full_elastic_energy_pct": ratio(
            scalars["fixed_state_split_energy_correction_N_mm"],
            scalars["full_degradation_elastic_energy_N_mm"],
        ),
        "spectral_negative_fraction_of_undegraded_energy_pct": ratio(
            scalars["spectral_negative_energy_N_mm"],
            scalars["undegraded_elastic_energy_N_mm"],
        ),
        "fixed_state_spectral_split_correction_relative_to_full_elastic_energy_pct": ratio(
            scalars["fixed_state_spectral_split_energy_correction_N_mm"],
            scalars["full_degradation_elastic_energy_N_mm"],
        ),
        "compressive_fraction_of_damage_driving_integral_pct": ratio(
            scalars["compressive_damage_driving_integral_N_mm"],
            scalars["damage_driving_integral_N_mm"],
        ),
        "spectral_compressive_fraction_of_damage_driving_integral_pct": ratio(
            scalars["spectral_compressive_damage_driving_integral_N_mm"],
            scalars["damage_driving_integral_N_mm"],
        ),
        "compression_area_fraction_pct": ratio(
            scalars["compression_area_mm2"], scalars["area_mm2"]
        ),
        "damaged_compression_area_fraction_pct": ratio(
            scalars["damaged_compression_area_mm2"],
            scalars["damaged_area_d_gt_0p1_mm2"],
        ),
        "minimum_cell_centroid_principal_stress_MPa": solver._owned_min(
            principal_field
        ),
        "minimum_cell_centroid_trace_strain": solver._owned_min(trace_field),
        "maximum_cell_centroid_trace_strain": solver._owned_max(trace_field),
        "interpretation": (
            "Fixed-state compression exposure quantifies the energy term treated "
            "differently by full degradation and an Amor volumetric-deviatoric split; "
            "it is not a re-solved split-model response."
        ),
    }

    if MPI.COMM_WORLD.rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
