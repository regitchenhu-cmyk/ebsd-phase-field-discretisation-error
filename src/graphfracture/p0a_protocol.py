"""Prospective fixed design for the P0-A homogeneous SENT study.

This module deliberately contains no runnable P0-A load schedule.  Existing
repository evidence is right-censored before the peak, so a non-evidential
qualification study must first freeze the load window, sampling interval and
numerical tolerances.  Keeping that absence executable prevents a template
configuration from being mistaken for a confirmatory preregistration.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from graphfracture.config import (
    GraphConfig,
    HydrogenConfig,
    LoadingConfig,
    PathControlConfig,
    RunConfig,
    SolverConfig,
    load_config,
)

PROTOCOL_REVISION = 1
DESIGN_STATUS = "blocked_pending_window_and_tolerance_qualification"


@dataclass(frozen=True)
class P0APhysicalModel:
    length: float = 0.09639
    height: float = 0.048195
    precrack_length: float = 0.0240975
    young_modulus: float = 210_000.0
    poisson_ratio: float = 0.30
    fracture_toughness: float = 0.0616296296296
    length_scale: float = 0.0015
    residual_stiffness: float = 1.0e-8
    diagonal: str = "right"
    x_pin_corner: str = "bottom_left"
    formulation: str = "plane_strain_unsplit_isotropic_AT2"
    graph_enabled: bool = False
    hydrogen_enabled: bool = False


@dataclass(frozen=True)
class P0AMeshLevel:
    name: str
    nominal_ell_over_hk: int
    nx: int
    ny: int

    def triangle_diameter(self, model: P0APhysicalModel) -> float:
        return math.hypot(model.length / self.nx, model.height / self.ny)

    def actual_ell_over_hk(self, model: P0APhysicalModel) -> float:
        return model.length_scale / self.triangle_diameter(model)

    def precrack_node_index(self, model: P0APhysicalModel) -> int:
        index = model.precrack_length / (model.length / self.nx)
        rounded = round(index)
        if not math.isclose(index, rounded, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{self.name} precrack is not mesh aligned: {index}")
        return rounded


@dataclass(frozen=True)
class P0AStage:
    stage_id: str
    mesh: str | None
    mpi_ranks: int
    cold_replica: int | None
    role: str
    execution_mode: str
    depends_on: tuple[str, ...]
    always_run: bool = False


@dataclass(frozen=True)
class P0AQualificationStage:
    stage_id: str
    mesh: str | None
    mpi_ranks: int
    schedule: str
    role: str
    execution_mode: str
    depends_on: tuple[str, ...]
    always_run: bool = False


PHYSICAL_MODEL = P0APhysicalModel()
MESH_LEVELS = (
    P0AMeshLevel("q2", 2, 184, 92),
    P0AMeshLevel("q3", 3, 276, 138),
    P0AMeshLevel("q4", 4, 368, 184),
)
STAGES = (
    P0AStage("S01", "q2", 2, 1, "coarse_canonical", "mpiexec_hydra_mpi2", ()),
    P0AStage("S02", "q3", 2, 1, "middle_canonical", "mpiexec_hydra_mpi2", ("S01",)),
    P0AStage("S03", "q4", 2, 1, "finest_canonical", "mpiexec_hydra_mpi2", ("S02",)),
    P0AStage("S04", "q4", 2, 2, "finest_cold_repeat", "mpiexec_hydra_mpi2", ("S03",)),
    P0AStage("S05", "q4", 2, 3, "finest_cold_repeat", "mpiexec_hydra_mpi2", ("S04",)),
    P0AStage("S06", "q3", 1, 1, "middle_serial_pair", "direct_python_serial", ("S05",)),
    P0AStage(
        "S07",
        None,
        1,
        None,
        "terminal_summary",
        "direct_python_no_solver",
        ("S01", "S02", "S03", "S04", "S05", "S06"),
        True,
    ),
)
QUALIFICATION_STAGES = (
    P0AQualificationStage(
        "Q01", "q2", 2, "exploratory", "window_discovery_only", "mpiexec_hydra_mpi2", ()
    ),
    P0AQualificationStage(
        "Q02",
        "q2",
        2,
        "candidate_fixed",
        "cross_mesh_qualification",
        "mpiexec_hydra_mpi2",
        ("Q01",),
    ),
    P0AQualificationStage(
        "Q03",
        "q3",
        2,
        "candidate_fixed",
        "cross_mesh_qualification",
        "mpiexec_hydra_mpi2",
        ("Q02",),
    ),
    P0AQualificationStage(
        "Q04",
        "q4",
        2,
        "candidate_fixed",
        "cross_mesh_and_window",
        "mpiexec_hydra_mpi2",
        ("Q03",),
    ),
    P0AQualificationStage(
        "Q05",
        "q4",
        2,
        "candidate_half_step_check_only",
        "load_discretization_acceptance_only",
        "mpiexec_hydra_mpi2",
        ("Q04",),
    ),
    P0AQualificationStage(
        "Q06",
        "q4",
        2,
        "candidate_fixed",
        "cold_repeatability",
        "mpiexec_hydra_mpi2",
        ("Q05",),
    ),
    P0AQualificationStage(
        "Q07",
        "q3",
        1,
        "candidate_fixed",
        "serial_mpi_repeatability",
        "direct_python_serial",
        ("Q06",),
    ),
    P0AQualificationStage(
        "Q08",
        None,
        1,
        "none",
        "terminal_summary",
        "direct_python_no_solver",
        ("Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07"),
        True,
    ),
)

SOURCE_TEMPLATE_FILENAMES = {
    "q2": "ijss_recalc_s1GPa_hom_noH_184x92.toml",
    "q3": "ijss_recalc_s1GPa_hom_noH_276x138.toml",
    "q4": "ijss_recalc_s1GPa_hom_noH_368x184.toml",
}
SOURCE_TEMPLATE_IDENTITIES = {
    "q2": {
        "bytes": 1409,
        "sha256": "fa8b8ec71be79f7394d31eba4cc05763a48ca32395eff0add6f8fc19de762621",
    },
    "q3": {
        "bytes": 1411,
        "sha256": "3190c7c8af8d3122e6b4ed905f0e6dc9d820160e06c7cd134df6a1b6ab1e23d2",
    },
    "q4": {
        "bytes": 1411,
        "sha256": "47a75c4f9978682844647026b003cf1dda84f09f9ef40c9d22c46a20c7e5bc40",
    },
}
SOURCE_TEMPLATE_OUTPUT_DIRECTORIES = {
    "q2": "../results/ijss_recalc_s1GPa_hom_noH_184x92",
    "q3": "../results/ijss_recalc_s1GPa_hom_noH_276x138",
    "q4": "../results/ijss_recalc_s1GPa_hom_noH_368x184",
}
SOURCE_TEMPLATE_LOADING = LoadingConfig(
    maximum_displacement=1.2e-4,
    steps=60,
    stagger_max_iterations=320,
    stagger_tolerance=1.0e-5,
    damage_kkt_tolerance=1.0e-6,
    adaptive=True,
    maximum_subdivisions=10,
    minimum_increment=1.0e-11,
)
SOURCE_TEMPLATE_SOLVER = SolverConfig(
    linear_ksp_type="preonly",
    linear_pc_type="lu",
    factor_solver="mumps",
    damage_snes_type="vinewtonrsls",
    aitken_max_relaxation=1.25,
)
SOURCE_TEMPLATE_GRAPH = GraphConfig(
    enabled=False,
    influence_radius=0.0015,
    crack_threshold=0.70,
    chain_artifact="",
    confidence_floor=0.0,
    attribute_permutation_seed=-1,
    interface_start_node="",
    interface_impact_node="",
    interface_end_node="",
)
SOURCE_TEMPLATE_HYDROGEN = HydrogenConfig(
    enabled=False,
    diffusivity=1.0e-4,
    charging_concentration=0.05,
    charging_time=10.0,
    steps=20,
    trap_binding_constant=20.0,
    background_trap_density=0.0,
    toughness_degradation=0.8,
    minimum_toughness_ratio=0.2,
    charging_boundary="bottom",
)
SOURCE_TEMPLATE_PATH_CONTROL = PathControlConfig(
    enabled=False,
    functional="fracture_energy",
    switch_displacement=0.0,
    target_increment=1.0e-3,
    steps=1,
    adaptive=True,
    use_energy_predictor=True,
    energy_predictor_disable_after_nominal_path_step=-1,
    maximum_subdivisions=8,
    minimum_increment=1.0e-6,
    load_lower_bound=0.0,
    load_upper_bound=1.0,
    snes_max_iterations=100,
    residual_tolerance=1.0e-8,
    control_tolerance=1.0e-6,
    control_absolute_tolerance=1.0e-12,
)

RIGHT_CENSORED_LINEAGE_IDENTITIES = {
    "completion": {
        "relative_path": "results/ijss_recalc_s1GPa_hom_noH_184x92/completion.json",
        "bytes": 349,
        "sha256": "3f37a57895c8baf166f30fdbe7e491fc7744a6db6e614bf2ff11e6d278078d3c",
    },
    "history": {
        "relative_path": "results/ijss_recalc_s1GPa_hom_noH_184x92/history.csv",
        "bytes": 28259,
        "sha256": "ffb4cdaa677d211cb01348a82d61c90ccab20c538ed3144e1e905237786e9a8c",
    },
    "runtime": {
        "relative_path": "results/ijss_recalc_s1GPa_hom_noH_184x92/runtime.json",
        "bytes": 7566,
        "sha256": "418a7310982351313c2e3e7a49c542da269979c526b3f30bf005c0ad76f29891",
    },
    "resolved_config": {
        "relative_path": "results/ijss_recalc_s1GPa_hom_noH_184x92/config.resolved.json",
        "bytes": 2173,
        "sha256": "f2bf241395a1a2e756b5f3eeb8863cd331f040a05e8316a6d9ab70da6c1f9d99",
    },
}

PHASE_B_FIREWALL = {
    "phase_b_evidence_use": "local_same_target_software_diagnostic_only",
    "phase_b_tolerances_used": False,
    "phase_b_runs_count_toward_p0a": False,
    "cross_mesh_dof_array_comparison_allowed": False,
}
QUALIFICATION_SOURCE_POLICY = {
    "required_evidence_class": "p0a_homogeneous_sent_qualification",
    "qualification_source_sha256_required": True,
    "tolerance_derivation_record_required": True,
    "phase_b_artifact_references_allowed": False,
    "phase_b_tolerance_lineage_allowed": False,
    "pollution_negative_test_required": True,
}
FORMAL_FIXED_TABLE_POLICY = {
    "loading.adaptive": False,
    "loading.maximum_subdivisions": 0,
    "hybrid_if_enabled.path_control.adaptive": False,
    "hybrid_if_enabled.path_control.maximum_subdivisions": 0,
    "history.subdivision_level_required": 0,
    "augmented_table_requires_new_revision": True,
}

EXECUTION_BLOCKERS = (
    "qualification_preregistration_not_frozen",
    "common_postpeak_displacement_window_not_qualified",
    "nominal_displacement_spacing_not_qualified",
    "displacement_vs_hybrid_controller_not_qualified",
    "single_run_acceptance_literals_not_frozen",
    "cold_repeatability_tolerances_not_frozen",
    "serial_mpi_repeatability_tolerances_not_frozen",
    "cross_mesh_bands_floors_and_contraction_not_frozen",
    "qualification_source_whitelist_validator_not_frozen",
    "p0a_preflight_runner_extractor_and_aggregator_not_frozen",
)


def prospective_design() -> dict[str, Any]:
    """Return the fixed, non-executable portion of the prospective design."""

    return {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "evidence_class": "prospective_design_not_execution_authorization",
        "status": DESIGN_STATUS,
        "physical_model": asdict(PHYSICAL_MODEL),
        "mesh_levels": [
            {
                **asdict(level),
                "cells": 2 * level.nx * level.ny,
                "triangle_diameter": level.triangle_diameter(PHYSICAL_MODEL),
                "actual_ell_over_hk": level.actual_ell_over_hk(PHYSICAL_MODEL),
                "precrack_node_index": level.precrack_node_index(PHYSICAL_MODEL),
            }
            for level in MESH_LEVELS
        ],
        "stages": [asdict(stage) for stage in STAGES],
        "qualification_stages": [asdict(stage) for stage in QUALIFICATION_STAGES],
        "source_template_identities": dict(SOURCE_TEMPLATE_IDENTITIES),
        "right_censored_lineage_identities": dict(RIGHT_CENSORED_LINEAGE_IDENTITIES),
        "phase_b_firewall": dict(PHASE_B_FIREWALL),
        "qualification_source_policy": dict(QUALIFICATION_SOURCE_POLICY),
        "formal_fixed_table_policy": dict(FORMAL_FIXED_TABLE_POLICY),
        "execution_authorized": False,
        "execution_blockers": list(EXECUTION_BLOCKERS),
    }


def assert_execution_authorized() -> None:
    """Fail closed until a later, qualification-bound protocol revision exists."""

    blockers = ", ".join(EXECUTION_BLOCKERS)
    raise RuntimeError(f"P0-A execution is not authorized: {blockers}")


def source_template_paths(repository_root: Path) -> dict[str, Path]:
    return {
        name: repository_root / "examples" / filename
        for name, filename in SOURCE_TEMPLATE_FILENAMES.items()
    }


def _require_identity(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"identity-bound input must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected["bytes"] or observed_sha256 != expected["sha256"]:
        raise ValueError(f"identity mismatch: {path}")


def validate_right_censored_lineage(repository_root: Path) -> None:
    """Bind the old window only as evidence that qualification is required."""

    for record in RIGHT_CENSORED_LINEAGE_IDENTITIES.values():
        _require_identity(repository_root / record["relative_path"], record)


def load_and_validate_source_templates(repository_root: Path) -> dict[str, RunConfig]:
    """Authenticate the selected lineage as templates, not as formal cases."""

    paths = source_template_paths(repository_root)
    for name, path in paths.items():
        _require_identity(path, SOURCE_TEMPLATE_IDENTITIES[name])
    configs = {name: load_config(path) for name, path in paths.items()}
    validate_source_template_family(configs)
    return configs


def validate_source_template_family(configs: Mapping[str, RunConfig]) -> None:
    """Require the three existing templates to differ only by mesh and output."""

    expected_names = {level.name for level in MESH_LEVELS}
    if set(configs) != expected_names:
        raise ValueError(f"source template names must be exactly {sorted(expected_names)}")

    levels = {level.name: level for level in MESH_LEVELS}
    reference = configs["q2"]
    output_directories: set[str] = set()
    for name in sorted(expected_names):
        config = configs[name]
        config.validate()
        level = levels[name]
        geometry = config.geometry
        material = config.material

        if (geometry.nx, geometry.ny) != (level.nx, level.ny):
            raise ValueError(f"{name} mesh does not match the fixed design")
        expected_geometry = replace(
            reference.geometry,
            nx=level.nx,
            ny=level.ny,
        )
        if geometry != expected_geometry:
            raise ValueError(f"{name} changes a non-mesh geometry field")
        if (
            geometry.length,
            geometry.height,
            geometry.precrack_length,
            geometry.diagonal,
            geometry.x_pin_corner,
        ) != (
            PHYSICAL_MODEL.length,
            PHYSICAL_MODEL.height,
            PHYSICAL_MODEL.precrack_length,
            PHYSICAL_MODEL.diagonal,
            PHYSICAL_MODEL.x_pin_corner,
        ):
            raise ValueError(f"{name} does not match the fixed physical geometry")
        if (
            material.young_modulus,
            material.poisson_ratio,
            material.fracture_toughness,
            material.length_scale,
            material.residual_stiffness,
        ) != (
            PHYSICAL_MODEL.young_modulus,
            PHYSICAL_MODEL.poisson_ratio,
            PHYSICAL_MODEL.fracture_toughness,
            PHYSICAL_MODEL.length_scale,
            PHYSICAL_MODEL.residual_stiffness,
        ):
            raise ValueError(f"{name} does not match the fixed material")

        if config.material != reference.material:
            raise ValueError(f"{name} material differs from q2")
        if config.loading != SOURCE_TEMPLATE_LOADING:
            raise ValueError(f"{name} source template loading contract changed")
        if config.graph != SOURCE_TEMPLATE_GRAPH:
            raise ValueError(f"{name} source template graph contract changed")
        if config.hydrogen != SOURCE_TEMPLATE_HYDROGEN:
            raise ValueError(f"{name} source template hydrogen contract changed")
        if config.path_control != SOURCE_TEMPLATE_PATH_CONTROL:
            raise ValueError(f"{name} source template path-control contract changed")
        if config.solver != SOURCE_TEMPLATE_SOLVER:
            raise ValueError(f"{name} source template solver contract changed")
        if config.output.directory != SOURCE_TEMPLATE_OUTPUT_DIRECTORIES[name]:
            raise ValueError(f"{name} source template output directory changed")
        if config.output.write_every != 5:
            raise ValueError(f"{name} source template write_every must be 5")
        if config.output.directory in output_directories:
            raise ValueError("source template output directories must be unique")
        output_directories.add(config.output.directory)
        if config.graph_nodes or config.graph_edges:
            raise ValueError(f"{name} must not contain graph nodes or edges")
        if config.schema_version != 1:
            raise ValueError(f"{name} schema_version must be 1")
        level.precrack_node_index(PHYSICAL_MODEL)

    if len(output_directories) != len(MESH_LEVELS):
        raise ValueError("source template output directory count mismatch")
