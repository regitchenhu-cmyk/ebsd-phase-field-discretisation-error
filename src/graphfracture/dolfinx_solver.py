"""DOLFINx 0.11 implementation of an incremental AT2 fracture baseline.

The module is imported lazily by :mod:`graphfracture.cli`, allowing config and
graph tests to run on machines without DOLFINx.  DOLFINx itself is supplied by
the pinned official container rather than installed as a normal pip wheel.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dolfinx
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

from .config import (
    INTERFACE_DAMAGE_THRESHOLDS,
    RunConfig,
    continuation_control_increase,
)
from .crack_graph import (
    analyse_crack_graph,
    analyse_interface_interaction,
    extract_main_crack_geometry,
    interface_classification_consensus,
)
from .damage_control import (
    ControlPhase,
    fracture_energy_control_residual_certificate,
)
from .gb_graph import BoundaryField, BoundaryGraph, boundary_field_from_config
from .hybrid_state import HybridSchedulerState

if TYPE_CHECKING:
    from .path_control import FractureEnergyControlProblem

CONTAINER_IMAGE = (
    "dolfinx/dolfinx:v0.11.0@"
    "sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8"
)


class AT2Solver:
    """Plane-strain SENT solver with alternate minimisation and SNESVI damage.

    The first verification model is intentionally the strict isotropic AT2
    functional.  It degrades the complete elastic energy and therefore should
    only be used for tensile benchmarks.  A tension/compression split belongs
    in a separate constitutive model and is not silently approximated here.
    """

    TOP_TAG = 1
    HYDROGEN_TAG = 2
    RESTART_SCHEMA_VERSION = 2
    HYBRID_RESTART_SCHEMA_VERSION = 3
    LEGACY_RESTART_SCHEMA_VERSION = 1
    _RESTART_FUNCTION_NAMES = (
        "u",
        "d",
        "d_lower",
        "c_h",
        "c_h_old",
        "theta_h",
        "trapped_hydrogen",
        "Gc0",
        "Gc",
        "diffusivity",
        "trap_density",
    )
    _HYBRID_PATH_RESTART_ARRAY_NAMES = (
        "path_z",
        "path_alpha",
        "path_target_fracture_energy",
        "path_target_increment",
        "path_target_energy_density",
    )
    _CONTROL_HISTORY_FIELDS = (
        "control_phase",
        "phase_step",
        "load_factor",
        "reference_displacement",
        "path_coordinate",
        "path_increment",
        "control_target",
        "control_value",
        "control_residual_relative",
        "path_snes_iterations",
        "path_snes_reason",
        "path_ksp_reason",
        "mechanical_residual_relative",
        "load_factor_bound_status",
    )
    _HISTORY_RECORD_FIELDS = (
        "step",
        "scheduled_step",
        "subdivision_level",
        "displacement",
        "load_increment",
        *_CONTROL_HISTORY_FIELDS,
        "reaction_y",
        "traction_reaction_y",
        "elastic_energy",
        "fracture_energy",
        "total_internal_energy",
        "regularised_crack_length",
        "maximum_damage",
        "rightmost_damaged_x",
        "stagger_iterations",
        "stagger_error",
        "stagger_converged",
        "minimum_damage_increment",
        "damage_kkt_inf",
        "damage_kkt_relative",
        "damage_kkt_scale",
        "damage_snes_iterations",
        "damage_snes_reason",
        "elastic_ksp_reason",
        "aitken_accepted_iterations",
        "final_aitken_relaxation",
        "external_work",
        "energy_balance_residual",
        "energy_balance_relative",
    )

    def __init__(self, config: RunConfig, boundary_field: BoundaryField | None = None) -> None:
        if not dolfinx.__version__.startswith("0.11.0"):
            raise RuntimeError(
                f"this research baseline requires DOLFINx 0.11.0.x, found {dolfinx.__version__}"
            )
        self.config = config
        self.comm = MPI.COMM_WORLD
        if not config.graph.enabled and boundary_field is not None:
            raise ValueError("a boundary field cannot be supplied when graph.enabled is false")
        protocol_enabled = all(
            (
                config.graph.interface_start_node,
                config.graph.interface_impact_node,
                config.graph.interface_end_node,
            )
        )
        if (
            protocol_enabled
            and boundary_field is not None
            and not isinstance(boundary_field, BoundaryGraph)
        ):
            raise ValueError(
                "an inclined-interface protocol requires a synthetic BoundaryGraph field"
            )
        self.gb_graph = (
            (boundary_field if boundary_field is not None else boundary_field_from_config(config))
            if config.graph.enabled
            else None
        )
        self._crack_graph_local_keys: tuple[tuple[float, float], ...] | None = None
        self._crack_graph_local_edges: (
            tuple[tuple[tuple[float, float], tuple[float, float]], ...] | None
        ) = None
        self._restart_generation = -1
        self._restart_configuration_fingerprint_cache: str | None = None
        self._restart_partition_fingerprint_cache: str | None = None
        self._restart_implementation_fingerprint_cache: str | None = None
        self._restart_runtime_fingerprint_cache: str | None = None
        self._effective_loading = config.loading
        self._continuation_sessions: list[dict[str, Any]] = []
        self._legacy_schema1_implementation_identity_unavailable = False
        self._path_control_problem: FractureEnergyControlProblem | None = None
        self._create_mesh_and_spaces()
        self._create_fields()
        self._create_boundary_conditions()
        self._create_forms_and_problems()

    def _create_mesh_and_spaces(self) -> None:
        g = self.config.geometry
        diagonal = {
            "left": mesh.DiagonalType.left,
            "right": mesh.DiagonalType.right,
        }[g.diagonal]
        self.domain = mesh.create_rectangle(
            self.comm,
            points=((0.0, 0.0), (g.length, g.height)),
            n=(g.nx, g.ny),
            cell_type=mesh.CellType.triangle,
            ghost_mode=mesh.GhostMode.shared_facet,
            diagonal=diagonal,
        )
        self.V_u = fem.functionspace(self.domain, ("Lagrange", 1, (2,)))
        self.V_d = fem.functionspace(self.domain, ("Lagrange", 1))
        self.V_0 = fem.functionspace(self.domain, ("Discontinuous Lagrange", 0))

    def _create_fields(self) -> None:
        cfg = self.config
        self.u = fem.Function(self.V_u, name="displacement")
        self.d = fem.Function(self.V_d, name="damage")
        self.d_lower = fem.Function(self.V_d, name="damage_lower_bound")
        self.d_upper = fem.Function(self.V_d, name="damage_upper_bound")
        self.Gc0 = fem.Function(self.V_0, name="base_fracture_toughness")
        self.Gc = fem.Function(self.V_0, name="effective_fracture_toughness")
        self.diffusivity = fem.Function(self.V_0, name="hydrogen_diffusivity")
        self.trap_density = fem.Function(self.V_0, name="trap_density")
        self.theta_h = fem.Function(self.V_0, name="hydrogen_coverage")
        self.trapped_hydrogen = fem.Function(self.V_0, name="trapped_hydrogen")
        self.c_h = fem.Function(self.V_d, name="lattice_hydrogen")
        self.c_h_old = fem.Function(self.V_d, name="previous_lattice_hydrogen")
        self.c_h_lower = fem.Function(self.V_d, name="hydrogen_lower_bound")
        self.c_h_upper = fem.Function(self.V_d, name="hydrogen_upper_bound")

        bulk_gc = cfg.material.fracture_toughness
        if self.gb_graph is None:
            self.Gc0.interpolate(lambda x: np.full(x.shape[1], bulk_gc, dtype=PETSc.ScalarType))
        else:
            self.Gc0.interpolate(
                lambda x: np.asarray(
                    bulk_gc * self.gb_graph.toughness_ratio_at(x), dtype=PETSc.ScalarType
                )
            )
        self.Gc0.x.scatter_forward()
        self.Gc.x.array[:] = self.Gc0.x.array
        self.Gc.x.scatter_forward()

        hydrogen = cfg.hydrogen
        if self.gb_graph is None:
            self.diffusivity.interpolate(
                lambda x: np.full(x.shape[1], hydrogen.diffusivity, dtype=PETSc.ScalarType)
            )
            self.trap_density.interpolate(
                lambda x: np.full(
                    x.shape[1], hydrogen.background_trap_density, dtype=PETSc.ScalarType
                )
            )
        else:
            self.diffusivity.interpolate(
                lambda x: np.asarray(
                    hydrogen.diffusivity * self.gb_graph.diffusivity_ratio_at(x),
                    dtype=PETSc.ScalarType,
                )
            )
            self.trap_density.interpolate(
                lambda x: np.asarray(
                    hydrogen.background_trap_density + self.gb_graph.trap_density_at(x),
                    dtype=PETSc.ScalarType,
                )
            )
        self.diffusivity.x.scatter_forward()
        self.trap_density.x.scatter_forward()
        self.c_h_upper.x.array[:] = cfg.hydrogen.charging_concentration
        self.c_h_lower.x.scatter_forward()
        self.c_h_upper.x.scatter_forward()

        g = cfg.geometry
        hy = g.height / g.ny

        def initial_crack(x: np.ndarray) -> np.ndarray:
            # Select the single structured-mesh node row at y=H/2.  This is an
            # internal pre-crack line, not a finite-width permanently damaged
            # band.  The zero-load VI solve then generates its diffuse AT2
            # profile without artificially adding a/ell to the crack energy.
            precrack = (x[0] <= g.precrack_length) & np.isclose(
                x[1], 0.5 * g.height, rtol=0.0, atol=0.1 * hy
            )
            return precrack.astype(PETSc.ScalarType)

        self.d.interpolate(initial_crack)
        self.d.x.scatter_forward()
        self.d_lower.x.array[:] = self.d.x.array
        self.d_upper.x.array[:] = 1.0
        self.d_lower.x.scatter_forward()
        self.d_upper.x.scatter_forward()
        self._lower_vec = self.d_lower.x.petsc_vec
        self._upper_vec = self.d_upper.x.petsc_vec

    def _create_boundary_conditions(self) -> None:
        g = self.config.geometry
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        bottom_facets = mesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.isclose(x[1], 0.0)
        )
        top_facets = mesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.isclose(x[1], g.height)
        )
        bottom_y = fem.locate_dofs_topological(self.V_u.sub(1), fdim, bottom_facets)
        top_y = fem.locate_dofs_topological(self.V_u.sub(1), fdim, top_facets)
        self.top_y_dofs = np.asarray(top_y, dtype=np.int32)
        # DOLFINx cannot tabulate coordinates directly on an uncollapsed
        # subspace.  Locate the geometric vertex first, then map it to the
        # x-component dof topologically.
        pin_y = 0.0 if g.x_pin_corner == "bottom_left" else g.height
        corner_vertices = mesh.locate_entities_boundary(
            self.domain,
            0,
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], pin_y),
        )
        corner_x = fem.locate_dofs_topological(self.V_u.sub(0), 0, corner_vertices)

        zero = PETSc.ScalarType(0.0)
        self.load = fem.Constant(self.domain, zero)
        bottom_y_zero_bc = fem.dirichletbc(zero, bottom_y, self.V_u.sub(1))
        corner_x_zero_bc = fem.dirichletbc(zero, corner_x, self.V_u.sub(0))
        self.bcs_u = [
            bottom_y_zero_bc,
            fem.dirichletbc(self.load, top_y, self.V_u.sub(1)),
            corner_x_zero_bc,
        ]
        self.path_homogeneous_bcs = [
            bottom_y_zero_bc,
            fem.dirichletbc(zero, top_y, self.V_u.sub(1)),
            corner_x_zero_bc,
        ]

        top_facets = np.sort(top_facets)
        values = np.full(top_facets.size, self.TOP_TAG, dtype=np.int32)
        self.facet_tags = mesh.meshtags(self.domain, fdim, top_facets, values)

        self.hydrogen_bc = None
        if self.config.hydrogen.enabled:
            boundary = self.config.hydrogen.charging_boundary
            coordinate, value = {
                "left": (0, 0.0),
                "right": (0, g.length),
                "bottom": (1, 0.0),
                "top": (1, g.height),
            }[boundary]
            charging_facets = mesh.locate_entities_boundary(
                self.domain,
                fdim,
                lambda x: np.isclose(x[coordinate], value),
            )
            charging_dofs = fem.locate_dofs_topological(self.V_d, fdim, charging_facets)
            self.hydrogen_charging_dofs = np.asarray(charging_dofs, dtype=np.int32)
            self.hydrogen_bc = fem.dirichletbc(
                PETSc.ScalarType(self.config.hydrogen.charging_concentration),
                charging_dofs,
                self.V_d,
            )
            charging_facets = np.sort(charging_facets)
            self.hydrogen_facet_tags = mesh.meshtags(
                self.domain,
                fdim,
                charging_facets,
                np.full(charging_facets.size, self.HYDROGEN_TAG, dtype=np.int32),
            )

    def _epsilon(self, value: Any) -> Any:
        return ufl.sym(ufl.grad(value))

    def _sigma0(self, value: Any) -> Any:
        m = self.config.material
        mu = m.young_modulus / (2.0 * (1.0 + m.poisson_ratio))
        lam = (
            m.young_modulus
            * m.poisson_ratio
            / ((1.0 + m.poisson_ratio) * (1.0 - 2.0 * m.poisson_ratio))
        )
        eps = self._epsilon(value)
        return 2.0 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(2)

    def _psi0(self, value: Any) -> Any:
        return 0.5 * ufl.inner(self._sigma0(value), self._epsilon(value))

    def _degradation(self, damage: Any) -> Any:
        residual = self.config.material.residual_stiffness
        return (1.0 - residual) * (1.0 - damage) ** 2 + residual

    def _linear_options(self) -> dict[str, Any]:
        solver = self.config.solver
        options: dict[str, Any] = {
            "ksp_type": solver.linear_ksp_type,
            "pc_type": solver.linear_pc_type,
            "ksp_error_if_not_converged": True,
        }
        if solver.linear_pc_type == "lu" and solver.factor_solver:
            options["pc_factor_mat_solver_type"] = solver.factor_solver
        return options

    def _create_forms_and_problems(self) -> None:
        m = self.config.material
        if self.config.hydrogen.enabled:
            hcfg = self.config.hydrogen
            concentration_test = ufl.TestFunction(self.V_d)
            concentration_increment = ufl.TrialFunction(self.V_d)
            dt = hcfg.charging_time / hcfg.steps
            coverage_current = (
                hcfg.trap_binding_constant
                * self.c_h
                / (1.0 + hcfg.trap_binding_constant * self.c_h)
            )
            coverage_old = (
                hcfg.trap_binding_constant
                * self.c_h_old
                / (1.0 + hcfg.trap_binding_constant * self.c_h_old)
            )
            storage_current = self.c_h + self.trap_density * coverage_current
            storage_old = self.c_h_old + self.trap_density * coverage_old
            # Vertex quadrature mass-lumps the nonlinear storage term.  On the
            # non-obtuse structured mesh this preserves the diffusion
            # M-matrix structure and prevents the concentration VI bounds from
            # acting as artificial interior hydrogen sources.
            storage_dx = ufl.Measure(
                "dx",
                domain=self.domain,
                metadata={"quadrature_rule": "vertex", "quadrature_degree": 1},
            )
            residual_h = (
                (storage_current - storage_old) * concentration_test / dt
            ) * storage_dx + (
                self.diffusivity * ufl.dot(ufl.grad(self.c_h), ufl.grad(concentration_test))
            ) * ufl.dx
            jacobian_h = ufl.derivative(residual_h, self.c_h, concentration_increment)
            hydrogen_options = {
                "snes_type": "vinewtonrsls",
                "snes_rtol": 1.0e-10,
                "snes_atol": 1.0e-12,
                "snes_stol": 1.0e-12,
                "snes_max_it": 50,
                "snes_error_if_not_converged": True,
                **self._linear_options(),
            }
            self.hydrogen_problem = NonlinearProblem(
                residual_h,
                self.c_h,
                J=jacobian_h,
                bcs=[self.hydrogen_bc],
                petsc_options_prefix="graphfracture_hydrogen_",
                petsc_options=hydrogen_options,
            )
            self.hydrogen_problem.solver.setVariableBounds(
                self.c_h_lower.x.petsc_vec,
                self.c_h_upper.x.petsc_vec,
            )
            self.hydrogen_residual_form = fem.form(residual_h)
            self.hydrogen_storage_increment_form = fem.form(
                (storage_current - storage_old) * storage_dx
            )
            self._coverage_expression = fem.Expression(
                coverage_current, self.V_0.element.interpolation_points
            )
            self._trapped_expression = fem.Expression(
                self.trap_density * coverage_current,
                self.V_0.element.interpolation_points,
            )
            self.hydrogen_content_form = fem.form(storage_current * storage_dx)
            self.lattice_hydrogen_form = fem.form(self.c_h * storage_dx)
            hydrogen_ds = ufl.Measure(
                "ds",
                domain=self.domain,
                subdomain_data=self.hydrogen_facet_tags,
            )
            normal_h = ufl.FacetNormal(self.domain)
            self.hydrogen_inflow_form = fem.form(
                self.diffusivity
                * ufl.dot(ufl.grad(self.c_h), normal_h)
                * hydrogen_ds(self.HYDROGEN_TAG)
            )
        else:
            self.hydrogen_problem = None

        du = ufl.TrialFunction(self.V_u)
        v = ufl.TestFunction(self.V_u)
        zero = fem.Constant(self.domain, np.zeros(2, dtype=PETSc.ScalarType))
        a_u = self._degradation(self.d) * ufl.inner(self._sigma0(du), self._epsilon(v)) * ufl.dx
        L_u = ufl.inner(zero, v) * ufl.dx
        self.elastic_residual_form = fem.form(
            self._degradation(self.d) * ufl.inner(self._sigma0(self.u), self._epsilon(v)) * ufl.dx
            - L_u
        )
        self.elastic_problem = LinearProblem(
            a_u,
            L_u,
            bcs=self.bcs_u,
            u=self.u,
            petsc_options_prefix="graphfracture_elastic_",
            petsc_options=self._linear_options(),
        )

        q = ufl.TestFunction(self.V_d)
        dd = ufl.TrialFunction(self.V_d)
        psi = self._psi0(self.u)
        residual_stiffness = m.residual_stiffness
        self.elastic_damage_residual = (
            2.0 * (1.0 - residual_stiffness) * psi * (self.d - 1.0) * q * ufl.dx
        )
        self.fracture_damage_residual = (
            self.Gc
            * (
                (self.d / m.length_scale) * q
                + m.length_scale * ufl.dot(ufl.grad(self.d), ufl.grad(q))
            )
            * ufl.dx
        )
        self.damage_residual = self.elastic_damage_residual + self.fracture_damage_residual
        damage_jacobian = ufl.derivative(self.damage_residual, self.d, dd)
        damage_options = {
            "snes_type": self.config.solver.damage_snes_type,
            "snes_rtol": 1.0e-10,
            "snes_atol": 1.0e-10,
            "snes_stol": 1.0e-12,
            "snes_max_it": 50,
            "snes_error_if_not_converged": True,
            **self._linear_options(),
        }
        self.damage_problem = NonlinearProblem(
            self.damage_residual,
            self.d,
            J=damage_jacobian,
            petsc_options_prefix="graphfracture_damage_",
            petsc_options=damage_options,
        )
        self.damage_residual_form = fem.form(self.damage_residual)
        self.elastic_damage_residual_form = fem.form(self.elastic_damage_residual)
        self.fracture_damage_residual_form = fem.form(self.fracture_damage_residual)

        crack_density = self.d**2 / (2.0 * m.length_scale) + 0.5 * m.length_scale * ufl.dot(
            ufl.grad(self.d), ufl.grad(self.d)
        )
        self.elastic_energy_form = fem.form(self._degradation(self.d) * self._psi0(self.u) * ufl.dx)
        self.fracture_energy_form = fem.form(self.Gc * crack_density * ufl.dx)
        self.crack_length_form = fem.form(crack_density * ufl.dx)
        normal = ufl.FacetNormal(self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        traction = ufl.dot(self._degradation(self.d) * self._sigma0(self.u), normal)
        self.reaction_form = fem.form(traction[1] * ds(self.TOP_TAG))

    def _global_scalar(self, form: Any) -> float:
        local = fem.assemble_scalar(form)
        return float(self.comm.allreduce(local, op=MPI.SUM).real)

    def _owned_max(self, function: fem.Function) -> float:
        index_map = function.function_space.dofmap.index_map
        owned = index_map.size_local * function.function_space.dofmap.index_map_bs
        local = float(np.max(function.x.array[:owned].real, initial=-math.inf))
        return float(self.comm.allreduce(local, op=MPI.MAX))

    def _owned_min(self, function: fem.Function) -> float:
        index_map = function.function_space.dofmap.index_map
        owned = index_map.size_local * function.function_space.dofmap.index_map_bs
        local = float(np.min(function.x.array[:owned].real, initial=math.inf))
        return float(self.comm.allreduce(local, op=MPI.MIN))

    def _owned_statistics(self, function: fem.Function) -> dict[str, float | int]:
        """Return min/max/mean over owned degrees of freedom without ghost duplication."""
        index_map = function.function_space.dofmap.index_map
        owned = index_map.size_local * function.function_space.dofmap.index_map_bs
        values = np.asarray(function.x.array[:owned].real, dtype=float)
        local_count = int(values.size)
        global_count = int(self.comm.allreduce(local_count, op=MPI.SUM))
        local_sum = float(np.sum(values))
        global_sum = float(self.comm.allreduce(local_sum, op=MPI.SUM))
        local_min = float(np.min(values, initial=math.inf))
        local_max = float(np.max(values, initial=-math.inf))
        return {
            "minimum": float(self.comm.allreduce(local_min, op=MPI.MIN)),
            "maximum": float(self.comm.allreduce(local_max, op=MPI.MAX)),
            "mean": global_sum / max(global_count, 1),
            "owned_dofs": global_count,
        }

    def _run_hydrogen_precharge(self, output: Path) -> list[dict[str, float]]:
        """Solve transient lattice diffusion and update ``Gc`` once before loading.

        Trapping enters through the exact backward-Euler difference of the
        Oriani storage ``c + N_T Kc/(1+Kc)``.  The HEDE coverage is the physical
        Langmuir/Oriani occupancy ``K*c/(1+K*c)``; it is deliberately not
        normalised by the maximum value in the current simulation.
        """
        hcfg = self.config.hydrogen
        if not hcfg.enabled:
            return []

        dt = hcfg.charging_time / hcfg.steps
        volume = self.config.geometry.length * self.config.geometry.height
        history: list[dict[str, float]] = []
        previous_total = 0.0
        for step in range(1, hcfg.steps + 1):
            self.hydrogen_problem.solve()
            self.c_h.x.scatter_forward()

            # The backward-Euler diffusion operator obeys a maximum principle
            # up to solver round-off.  Treat a material violation as an error;
            # do not silently clip a non-physical transport solution.
            c_min = self._owned_min(self.c_h)
            c_max = self._owned_max(self.c_h)
            tolerance = 1.0e-10 * max(1.0, hcfg.charging_concentration)
            if c_min < -tolerance or c_max > hcfg.charging_concentration + tolerance:
                raise RuntimeError(
                    "hydrogen transport violated concentration bounds: "
                    f"min={c_min:.6e}, max={c_max:.6e}"
                )

            self.theta_h.interpolate(self._coverage_expression)
            self.trapped_hydrogen.interpolate(self._trapped_expression)
            self.theta_h.x.scatter_forward()
            self.trapped_hydrogen.x.scatter_forward()
            total_hydrogen = self._global_scalar(self.hydrogen_content_form)
            diffusive_inflow_rate = self._global_scalar(self.hydrogen_inflow_form)
            storage_increment = self._global_scalar(self.hydrogen_storage_increment_form)
            dirichlet_injection_rate, internal_bound_reaction_rate = self._hydrogen_reaction_rates()
            total_algebraic_source_rate = dirichlet_injection_rate + internal_bound_reaction_rate
            mass_increment = total_hydrogen - previous_total
            algebraic_balance_residual = storage_increment - dt * total_algebraic_source_rate
            physical_balance_residual = storage_increment - dt * dirichlet_injection_rate
            hydrogen_snes = self.hydrogen_problem.solver
            history.append(
                {
                    "step": step,
                    "time": step * dt,
                    "minimum_lattice_concentration": c_min,
                    "maximum_lattice_concentration": c_max,
                    "mean_lattice_concentration": self._global_scalar(self.lattice_hydrogen_form)
                    / volume,
                    "total_hydrogen": total_hydrogen,
                    "diffusive_boundary_inflow_rate": diffusive_inflow_rate,
                    "dirichlet_injection_rate": dirichlet_injection_rate,
                    "internal_bound_reaction_rate": internal_bound_reaction_rate,
                    "total_algebraic_source_rate": total_algebraic_source_rate,
                    "exact_storage_increment": storage_increment,
                    "algebraic_mass_balance_residual": algebraic_balance_residual,
                    "algebraic_mass_balance_relative": algebraic_balance_residual
                    / max(
                        abs(storage_increment),
                        abs(dt * total_algebraic_source_rate),
                        1.0e-30,
                    ),
                    "physical_mass_balance_residual": physical_balance_residual,
                    "physical_mass_balance_relative": physical_balance_residual
                    / max(
                        abs(storage_increment),
                        abs(dt * dirichlet_injection_rate),
                        1.0e-30,
                    ),
                    "total_storage_crosscheck_error": mass_increment - storage_increment,
                    "hydrogen_snes_iterations": hydrogen_snes.getIterationNumber(),
                    "hydrogen_snes_reason": int(hydrogen_snes.getConvergedReason()),
                    "hydrogen_ksp_reason": int(hydrogen_snes.getKSP().getConvergedReason()),
                }
            )
            previous_total = total_hydrogen
            self.c_h_old.x.array[:] = self.c_h.x.array
            self.c_h_old.x.scatter_forward()

        reduction = np.maximum(
            1.0 - hcfg.toughness_degradation * self.theta_h.x.array.real,
            hcfg.minimum_toughness_ratio,
        )
        self.Gc.x.array[:] = self.Gc0.x.array * reduction
        self.Gc.x.scatter_forward()

        if history:
            history[-1]["mean_total_hydrogen"] = (
                self._global_scalar(self.hydrogen_content_form) / volume
            )
            history[-1]["maximum_coverage"] = self._owned_max(self.theta_h)
            history[-1]["minimum_effective_toughness"] = self._owned_min(self.Gc)
        self._write_hydrogen_history(output, history)
        return history

    def _hydrogen_reaction_rates(self) -> tuple[float, float]:
        """Return Dirichlet injection and non-Dirichlet VI reaction rates.

        The residual is assembled before strong Dirichlet elimination.  Its
        sum on charging dofs is the discrete boundary injection.  Any remaining
        residual on free dofs is an artificial source/sink from active internal
        concentration bounds and must be reported separately.
        """
        residual = assemble_vector(self.hydrogen_residual_form)
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
        charging = self.hydrogen_charging_dofs
        charging = charging[charging < owned]
        local_dirichlet = float(np.sum(residual.array_r[charging].real))
        local_total = float(np.sum(residual.array_r[:owned].real))
        residual.destroy()
        dirichlet = float(self.comm.allreduce(local_dirichlet, op=MPI.SUM))
        total = float(self.comm.allreduce(local_total, op=MPI.SUM))
        return dirichlet, total - dirichlet

    def _relative_change(self, current: fem.Function, previous: np.ndarray) -> float:
        index_map = current.function_space.dofmap.index_map
        owned = index_map.size_local * current.function_space.dofmap.index_map_bs
        delta = current.x.array[:owned] - previous[:owned]
        numerator = float(np.vdot(delta, delta).real)
        denominator = float(np.vdot(current.x.array[:owned], current.x.array[:owned]).real)
        numerator = self.comm.allreduce(numerator, op=MPI.SUM)
        denominator = self.comm.allreduce(denominator, op=MPI.SUM)
        return math.sqrt(numerator / max(denominator, 1.0e-30))

    def _maximum_change(self, current: fem.Function, previous: np.ndarray) -> float:
        index_map = current.function_space.dofmap.index_map
        owned = index_map.size_local * current.function_space.dofmap.index_map_bs
        local = float(np.max(np.abs(current.x.array[:owned] - previous[:owned]), initial=0.0))
        return float(self.comm.allreduce(local, op=MPI.MAX))

    def _minimum_damage_increment(self) -> float:
        owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
        local = float(
            np.min(
                (self.d.x.array[:owned] - self.d_lower.x.array[:owned]).real,
                initial=math.inf,
            )
        )
        return float(self.comm.allreduce(local, op=MPI.MIN))

    def _owned_residual_values(self, form: Any, space: fem.FunctionSpace) -> np.ndarray:
        """Assemble a residual and return a copy of its owned entries."""
        residual = assemble_vector(form)
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        owned = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
        values = np.asarray(residual.array_r[:owned].real).copy()
        residual.destroy()
        return values

    def _damage_kkt_metrics(self) -> tuple[float, float, float]:
        """Return absolute/relative VI violation and its physical residual scale."""
        values = self._owned_residual_values(self.damage_residual_form, self.V_d)
        elastic_values = self._owned_residual_values(self.elastic_damage_residual_form, self.V_d)
        fracture_values = self._owned_residual_values(self.fracture_damage_residual_form, self.V_d)
        owned = values.size
        damage = self.d.x.array[:owned].real
        lower = self.d_lower.x.array[:owned].real
        upper = self.d_upper.x.array[:owned].real
        tolerance = 1.0e-8
        at_lower = damage <= lower + tolerance
        at_upper = damage >= upper - tolerance
        violation = np.abs(values)
        violation[at_lower] = np.maximum(-values[at_lower], 0.0)
        violation[at_upper] = np.maximum(values[at_upper], 0.0)
        violation[at_lower & at_upper] = 0.0
        local_violation = float(np.max(violation, initial=0.0))
        local_scale = max(
            float(np.max(np.abs(elastic_values), initial=0.0)),
            float(np.max(np.abs(fracture_values), initial=0.0)),
        )
        absolute = float(self.comm.allreduce(local_violation, op=MPI.MAX))
        scale = float(self.comm.allreduce(local_scale, op=MPI.MAX))
        return absolute, absolute / max(scale, 1.0e-30), scale

    def _mechanical_residual_metrics(self) -> tuple[float, float, float]:
        """Return free-DOF equilibrium violation for the physical displacement."""
        values = self._owned_residual_values(self.elastic_residual_form, self.V_u)
        constrained = np.zeros(values.size, dtype=bool)
        for bc in self.bcs_u:
            dofs, _ = bc.dof_indices()
            owned_dofs = np.asarray(dofs, dtype=np.int64)
            owned_dofs = owned_dofs[owned_dofs < values.size]
            constrained[owned_dofs] = True
        local_absolute = float(np.max(np.abs(values[~constrained]), initial=0.0))
        local_scale = float(np.max(np.abs(values), initial=0.0))
        absolute = float(self.comm.allreduce(local_absolute, op=MPI.MAX))
        scale = float(self.comm.allreduce(local_scale, op=MPI.MAX))
        return absolute, absolute / max(scale, 1.0e-30), scale

    def _damage_bound_violations(self) -> tuple[float, float]:
        owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
        damage = self.d.x.array[:owned].real
        lower = self.d_lower.x.array[:owned].real
        upper = self.d_upper.x.array[:owned].real
        local_lower = float(np.max(lower - damage, initial=0.0))
        local_upper = float(np.max(damage - upper, initial=0.0))
        return (
            float(self.comm.allreduce(local_lower, op=MPI.MAX)),
            float(self.comm.allreduce(local_upper, op=MPI.MAX)),
        )

    def _dirichlet_violation(
        self,
        function: fem.Function,
        bcs: list[Any],
    ) -> float:
        expected = np.array(function.x.array, copy=True)
        constrained = np.zeros(expected.size, dtype=bool)
        for bc in bcs:
            dofs, _ = bc.dof_indices()
            local_dofs = np.asarray(dofs, dtype=np.int64)
            local_dofs = local_dofs[local_dofs < expected.size]
            constrained[local_dofs] = True
            bc.set(expected)
        local = float(
            np.max(
                np.abs(function.x.array[constrained] - expected[constrained]),
                initial=0.0,
            )
        )
        return float(self.comm.allreduce(local, op=MPI.MAX))

    def _elastic_reactions(self) -> tuple[float, float]:
        """Return discrete top reaction and continuum traction cross-check."""
        values = self._owned_residual_values(self.elastic_residual_form, self.V_u)
        top = self.top_y_dofs
        top = top[top < values.size]
        local = float(np.sum(values[top]))
        discrete = float(self.comm.allreduce(local, op=MPI.SUM))
        return discrete, self._global_scalar(self.reaction_form)

    def _total_internal_energy(self) -> float:
        return self._global_scalar(self.elastic_energy_form) + self._global_scalar(
            self.fracture_energy_form
        )

    def _max_damage(self) -> float:
        owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
        local = float(np.max(self.d.x.array[:owned], initial=0.0).real)
        return float(self.comm.allreduce(local, op=MPI.MAX))

    def _tip_x(self, threshold: float) -> float | None:
        coords = self.V_d.tabulate_dof_coordinates()
        owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
        active = np.asarray(self.d.x.array[:owned].real >= threshold)
        local = float(np.max(coords[:owned, 0][active], initial=-math.inf))
        result = float(self.comm.allreduce(local, op=MPI.MAX))
        return result if math.isfinite(result) else None

    def _record(
        self,
        step: int,
        displacement: float,
        solve_info: dict[str, Any],
        *,
        scheduled_step: int,
        subdivision_level: int,
        load_increment: float,
        control_info: dict[str, Any] | None = None,
    ) -> dict:
        control_fields: dict[str, Any] = {
            "control_phase": "displacement",
            "phase_step": step,
            "load_factor": None,
            "reference_displacement": None,
            "path_coordinate": displacement,
            "path_increment": load_increment,
            "control_target": displacement,
            "control_value": displacement,
            "control_residual_relative": 0.0,
            "path_snes_iterations": None,
            "path_snes_reason": None,
            "path_ksp_reason": None,
            "mechanical_residual_relative": None,
            "load_factor_bound_status": None,
        }
        if control_info is not None:
            if not isinstance(control_info, dict):
                raise TypeError("control_info must be a mapping or None")
            unknown = set(control_info).difference(self._CONTROL_HISTORY_FIELDS)
            if unknown:
                raise ValueError(
                    "control_info contains unknown history fields: " + ", ".join(sorted(unknown))
                )
            control_fields.update(control_info)
        elastic = self._global_scalar(self.elastic_energy_form)
        fracture = self._global_scalar(self.fracture_energy_form)
        reaction, traction_reaction = self._elastic_reactions()
        if solve_info["converged"]:
            kkt_absolute, kkt_relative, kkt_scale = self._damage_kkt_metrics()
        else:
            kkt_absolute, kkt_relative, kkt_scale = None, None, None
        return {
            "step": step,
            "scheduled_step": scheduled_step,
            "subdivision_level": subdivision_level,
            "displacement": displacement,
            "load_increment": load_increment,
            **control_fields,
            "reaction_y": reaction,
            "traction_reaction_y": traction_reaction,
            "elastic_energy": elastic,
            "fracture_energy": fracture,
            "total_internal_energy": elastic + fracture,
            "regularised_crack_length": self._global_scalar(self.crack_length_form),
            "maximum_damage": self._max_damage(),
            # This inexpensive per-step quantity is the rightmost active node,
            # irrespective of connectivity.  The final connected pre-crack
            # component is analysed separately in graph_metrics.json.
            "rightmost_damaged_x": self._tip_x(self.config.graph.crack_threshold),
            "stagger_iterations": solve_info["iterations"],
            "stagger_error": solve_info["error"],
            "stagger_converged": solve_info["converged"],
            "minimum_damage_increment": self._minimum_damage_increment(),
            "damage_kkt_inf": kkt_absolute,
            "damage_kkt_relative": kkt_relative,
            "damage_kkt_scale": kkt_scale,
            "damage_snes_iterations": solve_info["damage_snes_iterations"],
            "damage_snes_reason": solve_info["damage_snes_reason"],
            "elastic_ksp_reason": solve_info["elastic_ksp_reason"],
            "aitken_accepted_iterations": solve_info["aitken_accepted_iterations"],
            "final_aitken_relaxation": solve_info["final_aitken_relaxation"],
            "external_work": 0.0,
            "energy_balance_residual": 0.0,
            "energy_balance_relative": 0.0,
        }

    def _path_solve_info_for_record(
        self,
        path_info: dict[str, Any],
        *,
        phase_step: int,
        path_increment: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Adapt one augmented solve without discarding its independent certificates."""
        if not isinstance(path_info, dict):
            raise TypeError("path_info must be a mapping")
        if type(phase_step) is not int or phase_step < 0:
            raise ValueError("phase_step must be a non-negative integer")
        if type(path_increment) not in {int, float}:
            raise TypeError("path_increment must be a real number")
        increment = float(path_increment)
        if not math.isfinite(increment) or increment <= 0.0:
            raise ValueError("path_increment must be finite and positive")
        if self._path_control_problem is None:
            raise RuntimeError("path-control record adaptation requires an initialised problem")

        required = (
            "certified",
            "iterations",
            "snes_reason",
            "ksp_reason",
            "load_factor",
            "fracture_energy",
            "target_fracture_energy",
            "control_residual_relative",
            "damage_kkt_relative",
            "mechanical_residual_relative",
            "load_factor_bound_status",
        )
        missing = [name for name in required if name not in path_info]
        if missing:
            raise ValueError("path_info is missing record fields: " + ", ".join(missing))

        iterations = int(path_info["iterations"])
        snes_reason = int(path_info["snes_reason"])
        ksp_reason = int(path_info["ksp_reason"])
        control_relative = float(path_info["control_residual_relative"])
        damage_relative = float(path_info["damage_kkt_relative"])
        mechanical_relative = float(path_info["mechanical_residual_relative"])
        load_factor = float(path_info["load_factor"])
        control_target = float(path_info["target_fracture_energy"])
        control_value = float(path_info["fracture_energy"])
        reference_displacement = float(self._path_control_problem.reference_displacement)
        finite_values = (
            control_relative,
            damage_relative,
            mechanical_relative,
            load_factor,
            control_target,
            control_value,
            reference_displacement,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("path_info record metrics must be finite")

        certified = bool(path_info["certified"])
        solve_info = {
            "iterations": iterations,
            "error": max(
                # This remains a transparent raw diagnostic aggregate. Path
                # acceptance is checked by the independent composite control,
                # KKT, and mechanical certificates.
                abs(control_relative),
                abs(damage_relative),
                abs(mechanical_relative),
            ),
            "converged": certified,
            # These legacy columns remain populated for existing history
            # consumers.  The explicit path_* columns below retain the fact
            # that this was one augmented SNES rather than staggered blocks.
            "damage_snes_iterations": iterations,
            "damage_snes_reason": snes_reason,
            "elastic_ksp_reason": ksp_reason,
            "aitken_accepted_iterations": 0,
            "final_aitken_relaxation": None,
        }
        control_info = {
            "control_phase": "fracture_energy",
            "phase_step": phase_step,
            "load_factor": load_factor,
            "reference_displacement": reference_displacement,
            "path_coordinate": control_target,
            "path_increment": increment,
            "control_target": control_target,
            "control_value": control_value,
            "control_residual_relative": control_relative,
            "path_snes_iterations": iterations,
            "path_snes_reason": snes_reason,
            "path_ksp_reason": ksp_reason,
            "mechanical_residual_relative": mechanical_relative,
            "load_factor_bound_status": path_info["load_factor_bound_status"],
        }
        return solve_info, control_info

    def _initialise_path_control_problem(self) -> FractureEnergyControlProblem:
        """Create the augmented controller from the current accepted switch state."""
        if not self.config.path_control.enabled:
            raise RuntimeError("path control is disabled in the run configuration")
        if self._path_control_problem is not None:
            return self._path_control_problem

        accepted_displacement = float(np.asarray(self.load.value).reshape(-1)[0].real)
        reference_displacement = float(self.config.path_control.switch_displacement)
        local_error = None
        if (
            not math.isfinite(accepted_displacement)
            or accepted_displacement <= 0.0
            or not math.isfinite(reference_displacement)
            or reference_displacement <= 0.0
        ):
            local_error = (
                f"rank {self.comm.rank}: path control requires positive finite "
                "accepted/reference displacements"
            )
        initialisation_errors = self.comm.allgather(local_error)
        if any(error is not None for error in initialisation_errors):
            raise RuntimeError(
                "path-control initialisation preflight failed: "
                + "; ".join(error for error in initialisation_errors if error is not None)
            )

        problem = self._new_hybrid_path_problem(reference_displacement)
        problem.initialize_from_physical_displacement(
            self.u,
            load_factor=accepted_displacement / reference_displacement,
        )
        self._path_control_problem = problem
        return problem

    def _solve_path_control_step(
        self,
        target_energy: float,
        *,
        use_energy_predictor: bool = True,
    ) -> dict[str, float | int | bool | str | None]:
        """Attempt one certified energy-controlled step from the accepted state.

        The augmented problem owns the nonlinear transaction for ``z``, damage
        and ``alpha``.  This outer snapshot additionally protects the physical
        displacement and load Constant, which are committed only after all
        independent certificates pass.
        """
        if type(use_energy_predictor) is not bool:
            raise TypeError("use_energy_predictor must be a boolean")
        problem = self._initialise_path_control_problem()
        state = self._snapshot_state()
        try:
            info = problem.solve(
                target_energy,
                use_energy_predictor=use_energy_predictor,
            )
            if not bool(info.get("certified", False)):
                self._restore_state(state)
                return info
            displacement = problem.copy_physical_displacement_to(self.u)
            self.load.value = PETSc.ScalarType(displacement)
        except Exception:
            self._restore_state(state)
            raise
        return info

    def _solve_load_step(self, displacement: float) -> dict[str, Any]:
        load_cfg = self._effective_loading
        self.load.value = PETSc.ScalarType(displacement)
        # The VI lower bound stays fixed throughout this load step.  Updating it
        # inside the stagger loop would over-constrain the incremental problem.
        self.d_lower.x.array[:] = self.d.x.array
        self.d_lower.x.scatter_forward()
        self.damage_problem.solver.setVariableBounds(self._lower_vec, self._upper_vec)

        # Start and finish every outer iteration with an equilibrated
        # displacement.  This makes the coupled KKT residual meaningful at
        # every iteration rather than only after a field-change pre-check.
        self.elastic_problem.solve()
        self.u.x.scatter_forward()
        current_energy = self._total_internal_energy()
        previous_raw_residual: np.ndarray | None = None
        raw_change_history: list[float] = []
        aitken_relaxation = 1.0
        aitken_accepted = 0
        error = math.inf
        for iteration in range(1, load_cfg.stagger_max_iterations + 1):
            u_previous = self.u.x.array.copy()
            d_previous = self.d.x.array.copy()

            # Exact bound-constrained minimisation of the damage block at the
            # current equilibrated displacement.
            self.damage_problem.solve()
            self.d.x.scatter_forward()
            d_star = self.d.x.array.copy()
            raw_residual = d_star - d_previous
            raw_change = self._maximum_change(self.d, d_previous)
            raw_change_history.append(raw_change)

            energy_tolerance = 1.0e-10 * max(abs(current_energy), 1.0)
            vi_energy = self._total_internal_energy()
            if vi_energy > current_energy + energy_tolerance:
                raise RuntimeError(
                    "damage block increased the fixed-displacement energy: "
                    f"{current_energy:.6e} -> {vi_energy:.6e}"
                )

            # When the fixed-point residual has not dropped by at least a
            # factor of four over eight iterations, try vector Aitken only on
            # damage.  Projection preserves the VI bounds, while the energy
            # guard falls back to the exact damage-block minimiser if the
            # extrapolated point is not a descent state.
            stalled = (
                iteration >= 9
                and raw_change_history[-1] > 0.25 * raw_change_history[-9]
                and previous_raw_residual is not None
            )
            if stalled:
                owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
                delta_residual = raw_residual[:owned] - previous_raw_residual[:owned]
                local_numerator = float(np.vdot(previous_raw_residual[:owned], delta_residual).real)
                local_denominator = float(np.vdot(delta_residual, delta_residual).real)
                numerator = float(self.comm.allreduce(local_numerator, op=MPI.SUM))
                denominator = float(self.comm.allreduce(local_denominator, op=MPI.SUM))
                if denominator > 1.0e-30:
                    candidate_relaxation = float(
                        np.clip(
                            -aitken_relaxation * numerator / denominator,
                            0.25,
                            self.config.solver.aitken_max_relaxation,
                        )
                    )
                    trial = np.clip(
                        d_previous + candidate_relaxation * raw_residual,
                        self.d_lower.x.array,
                        self.d_upper.x.array,
                    )
                    self.d.x.array[:] = trial
                    self.d.x.scatter_forward()
                    trial_energy = self._total_internal_energy()
                    if trial_energy <= current_energy + energy_tolerance:
                        aitken_relaxation = candidate_relaxation
                        aitken_accepted += 1
                    else:
                        self.d.x.array[:] = d_star
                        self.d.x.scatter_forward()
                        aitken_relaxation = 1.0
                else:
                    aitken_relaxation = 1.0
            else:
                aitken_relaxation = 1.0

            previous_raw_residual = raw_residual
            self.elastic_problem.solve()
            self.u.x.scatter_forward()
            new_energy = self._total_internal_energy()
            if new_energy > current_energy + energy_tolerance:
                raise RuntimeError(
                    "staggered iteration increased total energy: "
                    f"{current_energy:.6e} -> {new_energy:.6e}"
                )
            current_energy = new_energy
            field_error = max(
                self._relative_change(self.u, u_previous),
                self._maximum_change(self.d, d_previous),
            )
            _, kkt_relative, _ = self._damage_kkt_metrics()
            error = max(field_error, kkt_relative)
            if (
                field_error <= load_cfg.stagger_tolerance
                and kkt_relative <= load_cfg.damage_kkt_tolerance
            ):
                return {
                    "iterations": iteration,
                    "error": error,
                    "converged": True,
                    "damage_snes_iterations": self.damage_problem.solver.getIterationNumber(),
                    "damage_snes_reason": int(self.damage_problem.solver.getConvergedReason()),
                    "elastic_ksp_reason": int(self.elastic_problem.solver.getConvergedReason()),
                    "aitken_accepted_iterations": aitken_accepted,
                    "final_aitken_relaxation": aitken_relaxation,
                }
        return {
            "iterations": load_cfg.stagger_max_iterations,
            "error": error,
            "converged": False,
            "damage_snes_iterations": self.damage_problem.solver.getIterationNumber(),
            "damage_snes_reason": int(self.damage_problem.solver.getConvergedReason()),
            "elastic_ksp_reason": int(self.elastic_problem.solver.getConvergedReason()),
            "aitken_accepted_iterations": aitken_accepted,
            "final_aitken_relaxation": aitken_relaxation,
        }

    def _write_metadata(self, output: Path) -> None:
        diagnostics = self.config.diagnostics()
        diagnostics["G_GB"] = (
            self.gb_graph.describe() if self.gb_graph is not None else {"enabled": False}
        )
        diagnostics["material_fields"] = {
            "base_fracture_toughness": self._owned_statistics(self.Gc0),
            "hydrogen_diffusivity": self._owned_statistics(self.diffusivity),
            "trap_density": self._owned_statistics(self.trap_density),
        }
        if self.comm.rank != 0:
            return
        with (output / "config.resolved.json").open("w", encoding="utf-8") as stream:
            json.dump(self.config.to_dict(), stream, ensure_ascii=False, indent=2)
        config_sha256 = None
        if self.config.source_path is not None and self.config.source_path.is_file():
            config_sha256 = hashlib.sha256(self.config.source_path.read_bytes()).hexdigest()
        metadata = {
            "python": platform.python_version(),
            "dolfinx": dolfinx.__version__,
            "ufl": ufl.__version__,
            "numpy": np.__version__,
            "petsc": PETSc.Sys.getVersionInfo(),
            "mpi_library": MPI.Get_library_version(),
            "mpi_ranks": self.comm.size,
            "reference_container": CONTAINER_IMAGE,
            "config_sha256": config_sha256,
            "implementation_fingerprint": self._restart_implementation_fingerprint(),
            "runtime_fingerprint": self._restart_runtime_fingerprint(),
            "model": {
                "mechanics": "plane-strain isotropic AT2",
                "irreversibility": "SNESVI incremental lower bound",
                "graph_coupling": "embedded G_GB to DG0 material fields",
                "hydrogen_coupling": (
                    "one-way transient pre-charge with exact Oriani storage and HEDE softening"
                    if self.config.hydrogen.enabled
                    else "disabled"
                ),
            },
            "diagnostics": diagnostics,
        }
        with (output / "runtime.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)

    def _write_history(self, output: Path, history: list[dict]) -> None:
        if self.comm.rank != 0 or not history:
            return
        path = output / "history.csv"
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _write_hydrogen_history(self, output: Path, history: list[dict[str, float]]) -> None:
        if self.comm.rank != 0 or not history:
            return
        fieldnames = list(history[0])
        for record in history[1:]:
            for name in record:
                if name not in fieldnames:
                    fieldnames.append(name)
        with (output / "hydrogen_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    def _write_completion(self, output: Path, history: list[dict]) -> None:
        if self.comm.rank != 0:
            return
        final = history[-1]
        completion = {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "accepted_load_steps": len(history) - 1,
            "final_displacement": final["displacement"],
            "all_steps_converged": all(bool(record["stagger_converged"]) for record in history),
            "effective_continuation_controls": self._continuation_controls(self._effective_loading),
            "continuation_sessions": len(self._continuation_sessions),
        }
        if self.config.path_control.enabled:
            completion.update(
                {
                    "final_control_phase": final["control_phase"],
                    "final_control_target": final["control_target"],
                    "final_control_value": final["control_value"],
                }
            )
        self._atomic_write_json(output / "completion.json", completion)

    def _write_attempt_history(
        self,
        output: Path,
        records: list[dict[str, Any]],
        *,
        complete: bool,
    ) -> None:
        if self.comm.rank != 0:
            return
        self._atomic_write_json(
            output / "attempt_history.json",
            {
                "schema_version": 1,
                "status": "complete" if complete else "partial",
                "role": "unaccepted nonlinear load attempts",
                "records": records,
            },
        )

    def _interface_tracking_enabled(self) -> bool:
        graph = self.config.graph
        return all(
            (
                graph.interface_start_node,
                graph.interface_impact_node,
                graph.interface_end_node,
            )
        )

    def _interface_history_record(
        self,
        *,
        accepted_step: int,
        scheduled_step: int,
        displacement: float,
        subdivision_level: int,
    ) -> dict[str, Any] | None:
        """Collect one accepted-state interface screen when a protocol is configured."""
        if not self._interface_tracking_enabled():
            return None
        crack = self._gather_crack_graph()
        if self.comm.rank != 0:
            return None
        if crack is None or "inclined_interface_interaction" not in crack:
            raise RuntimeError("configured inclined-interface metrics were not generated")
        interaction = crack["inclined_interface_interaction"]
        return {
            "accepted_step": int(accepted_step),
            "scheduled_step": int(scheduled_step),
            "displacement": float(displacement),
            "subdivision_level": int(subdivision_level),
            "threshold_consensus": interaction["threshold_consensus"],
            "threshold_results": interaction["threshold_results"],
        }

    def _write_interface_history(
        self,
        output: Path,
        records: list[dict[str, Any]],
        *,
        complete: bool,
    ) -> None:
        if self.comm.rank != 0 or not self._interface_tracking_enabled():
            return
        protocol = None
        if records:
            final_crack = self._gathered_interface_protocol_without_collective()
            protocol = final_crack
        self._atomic_write_json(
            output / "interface_history.json",
            {
                "schema_version": 1,
                "status": "complete" if complete else "partial",
                "role": "accepted-load-state inclined-interface geometric screens",
                "protocol": protocol,
                "records": records,
                "interpretation_limit": (
                    "threshold/mesh-dependent candidate classifications; this history is not "
                    "by itself a validated penetration-deflection transition"
                ),
            },
        )

    def _gathered_interface_protocol_without_collective(self) -> dict[str, Any]:
        """Reconstruct the fixed protocol metadata already used by accepted-state screens."""
        if not isinstance(self.gb_graph, BoundaryGraph):
            raise RuntimeError("inclined-interface tracking requires BoundaryGraph")
        graph = self.config.graph
        names = (
            graph.interface_start_node,
            graph.interface_impact_node,
            graph.interface_end_node,
        )
        start, impact, end = (self.gb_graph.nodes[name] for name in names)
        geometry = self.config.geometry
        h = max(geometry.length / geometry.nx, geometry.height / geometry.ny)
        ell = self.config.material.length_scale
        corridor = max(2.0 * ell, graph.influence_radius + h)
        return {
            "start_node": names[0],
            "impact_node": names[1],
            "end_node": names[2],
            "start": [start.x, start.y],
            "impact": [impact.x, impact.y],
            "end": [end.x, end.y],
            "length_scale": ell,
            "influence_radius": graph.influence_radius,
            "mesh_h": h,
            "boundary_tolerance": 1.5 * h,
            "impact_tolerance": max(2.0 * ell, 1.5 * h),
            "interface_corridor_width": corridor,
            "impact_exclusion_radius": 3.0 * ell,
            "confirmation_distance": 4.0 * ell,
            "penetration_corridor_half_height": corridor,
            "thresholds": list(INTERFACE_DAMAGE_THRESHOLDS),
            "distance_rule": (
                "impact=max(2ell,1.5h); interface=max(2ell,b+h); exclusion=3ell; confirmation=4ell"
            ),
        }

    @staticmethod
    def _coordinate_key(point: np.ndarray) -> tuple[float, float]:
        return round(float(point[0]), 12), round(float(point[1]), 12)

    def _gather_crack_graph(self) -> dict[str, Any] | None:
        coords = self.V_d.tabulate_dof_coordinates()
        owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
        if self._crack_graph_local_keys is None or self._crack_graph_local_edges is None:
            self._crack_graph_local_keys = tuple(
                self._coordinate_key(coords[index]) for index in range(owned)
            )
            tdim = self.domain.topology.dim
            num_cells = self.domain.topology.index_map(tdim).size_local
            local_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
            for cell in range(num_cells):
                cell_dofs = self.V_d.dofmap.cell_dofs(cell)
                for first, second in itertools.combinations(cell_dofs, 2):
                    a = self._coordinate_key(coords[first])
                    b = self._coordinate_key(coords[second])
                    if a != b:
                        local_edges.add(tuple(sorted((a, b))))
            self._crack_graph_local_edges = tuple(local_edges)
        local_nodes = [
            (key, float(self.d.x.array[index].real))
            for index, key in enumerate(self._crack_graph_local_keys)
        ]

        all_nodes = self.comm.gather(local_nodes, root=0)
        all_edges = self.comm.gather(self._crack_graph_local_edges, root=0)
        if self.comm.rank != 0:
            return None

        damage_by_point: dict[tuple[float, float], float] = {}
        for rank_nodes in all_nodes:
            for key, value in rank_nodes:
                damage_by_point[key] = max(value, damage_by_point.get(key, 0.0))
        point_keys = sorted(damage_by_point)
        point_index = {key: index for index, key in enumerate(point_keys)}
        edges = {
            (point_index[a], point_index[b])
            for rank_edges in all_edges
            for a, b in rank_edges
            if a in point_index and b in point_index
        }
        g = self.config.geometry
        h = max(g.length / g.nx, g.height / g.ny)
        metrics = analyse_crack_graph(
            point_keys,
            edges,
            [damage_by_point[key] for key in point_keys],
            threshold=self.config.graph.crack_threshold,
            left_x=0.0,
            right_x=g.length,
            boundary_tolerance=1.5 * h,
        )
        main_geometry = extract_main_crack_geometry(
            point_keys,
            edges,
            [damage_by_point[key] for key in point_keys],
            threshold=self.config.graph.crack_threshold,
            left_x=0.0,
            boundary_tolerance=1.5 * h,
        )
        result: dict[str, Any] = {
            "kind": "G_crack",
            **metrics.to_dict(),
            "main_component_geometry": main_geometry.to_dict(),
            "mesh_nodes": len(point_keys),
            "mesh_edges": len(edges),
        }
        graph_config = self.config.graph
        protocol_names = (
            graph_config.interface_start_node.strip(),
            graph_config.interface_impact_node.strip(),
            graph_config.interface_end_node.strip(),
        )
        if all(protocol_names) and isinstance(self.gb_graph, BoundaryGraph):
            protocol = self._gathered_interface_protocol_without_collective()
            threshold_results = [
                analyse_interface_interaction(
                    point_keys,
                    edges,
                    [damage_by_point[key] for key in point_keys],
                    interface_start=protocol["start"],
                    interface_end=protocol["end"],
                    impact_point=protocol["impact"],
                    threshold=threshold,
                    impact_tolerance=protocol["impact_tolerance"],
                    interface_corridor_width=protocol["interface_corridor_width"],
                    impact_exclusion_radius=protocol["impact_exclusion_radius"],
                    confirmation_distance=protocol["confirmation_distance"],
                    penetration_corridor_half_height=protocol["penetration_corridor_half_height"],
                    left_x=0.0,
                    boundary_tolerance=1.5 * h,
                )
                for threshold in protocol["thresholds"]
            ]
            result["inclined_interface_interaction"] = {
                "schema_version": 1,
                "role": "inclined-interface thresholded-crack geometric screen",
                "protocol": protocol,
                "threshold_results": [item.to_dict() for item in threshold_results],
                "threshold_consensus": interface_classification_consensus(
                    item.geometric_classification for item in threshold_results
                ),
                "interpretation_limit": (
                    "same-material isotropic weak-plane numerical screen; this is not a "
                    "validated anisotropic or bimaterial bicrystal transition"
                ),
            }
        return result

    def _write_graph_metrics(self, output: Path) -> None:
        crack = self._gather_crack_graph()
        if self.comm.rank != 0:
            return
        gb: dict[str, Any] = {"enabled": False}
        if self.gb_graph is not None and not isinstance(self.gb_graph, BoundaryGraph):
            # Measured EBSD chain field: no named junction nodes, so the
            # synthetic reference shortest path is not defined.  The chain
            # summary and G_crack topology remain the reported diagnostics.
            gb = self.gb_graph.describe()
        elif self.gb_graph is not None:
            gb = self.gb_graph.describe()
            g = self.config.geometry
            nodes = tuple(self.gb_graph.nodes.values())
            source_radius = max(
                self.config.graph.influence_radius,
                1.5 * max(g.length / g.nx, g.height / g.ny),
            )
            source_candidates = tuple(
                node.name
                for node in nodes
                if math.hypot(
                    node.x - g.precrack_length,
                    node.y - 0.5 * g.height,
                )
                <= source_radius
            )
            if not source_candidates:
                nearest = min(
                    math.hypot(
                        node.x - g.precrack_length,
                        node.y - 0.5 * g.height,
                    )
                    for node in nodes
                )
                source_candidates = tuple(
                    node.name
                    for node in nodes
                    if math.isclose(
                        math.hypot(
                            node.x - g.precrack_length,
                            node.y - 0.5 * g.height,
                        ),
                        nearest,
                    )
                )

            target_candidates = tuple(
                node.name for node in nodes if node.x >= g.length - source_radius
            )
            if not target_candidates:
                rightmost = max(node.x for node in nodes)
                target_candidates = tuple(
                    node.name for node in nodes if math.isclose(node.x, rightmost)
                )

            path = self.gb_graph.shortest_path_between(source_candidates, target_candidates)
            path_coordinates = tuple(
                (self.gb_graph.nodes[name].x, self.gb_graph.nodes[name].y) for name in path.nodes
            )
            gb["reference_shortest_path"] = {
                "role": "pre-charge reference hypothesis; compare with G_crack",
                "cost_basis": "sum(toughness_ratio * physical_edge_length)",
                "uses_hydrogen_degraded_toughness": False,
                "source_candidates": source_candidates,
                "target_candidates": target_candidates,
                "source": path.nodes[0] if path.nodes else None,
                "target": path.nodes[-1] if path.nodes else None,
                "reachable": math.isfinite(path.cost),
                "nodes": path.nodes,
                "path_coordinates": path_coordinates,
                "path_segments": tuple(
                    (first, second)
                    for first, second in zip(path_coordinates, path_coordinates[1:], strict=False)
                ),
                "normalised_integrated_toughness_cost": (
                    path.cost if math.isfinite(path.cost) else None
                ),
                "geometric_length": (
                    path.geometric_length if math.isfinite(path.geometric_length) else None
                ),
            }
        with (output / "graph_metrics.json").open("w", encoding="utf-8") as stream:
            interaction = (
                crack.pop("inclined_interface_interaction", None) if crack is not None else None
            )
            json.dump(
                {
                    "schema_version": 1,
                    "G_GB": gb,
                    "G_crack": crack,
                    "interface_interaction": interaction,
                },
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    def _snapshot_state(self) -> dict[str, Any]:
        state = {
            "u": self.u.x.array.copy(),
            "d": self.d.x.array.copy(),
            "d_lower": self.d_lower.x.array.copy(),
            "load": np.array(self.load.value, copy=True),
        }
        if self._path_control_problem is not None:
            state["path_control"] = self._path_control_problem._snapshot_state()
        return state

    def _restore_state(self, state: dict[str, Any]) -> None:
        self.u.x.array[:] = state["u"]
        self.d.x.array[:] = state["d"]
        self.d_lower.x.array[:] = state["d_lower"]
        self.load.value = state["load"]
        self.u.x.scatter_forward()
        self.d.x.scatter_forward()
        self.d_lower.x.scatter_forward()
        path_state = state.get("path_control")
        if self._path_control_problem is not None and path_state is not None:
            self._path_control_problem._restore_state(path_state)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _continuation_controls(loading: Any) -> dict[str, float | int]:
        return {
            "stagger_max_iterations": int(loading.stagger_max_iterations),
            "maximum_subdivisions": int(loading.maximum_subdivisions),
            "minimum_increment": float(loading.minimum_increment),
        }

    def _runtime_identity(self) -> dict[str, Any]:
        return {
            "python": platform.python_version(),
            "dolfinx": dolfinx.__version__,
            "ufl": ufl.__version__,
            "numpy": np.__version__,
            "petsc": PETSc.Sys.getVersionInfo(),
            "mpi_library": MPI.Get_library_version(),
            "mpi_ranks": self.comm.size,
            "reference_container": CONTAINER_IMAGE,
        }

    def _restart_implementation_fingerprint(self) -> str:
        if self._restart_implementation_fingerprint_cache is not None:
            return self._restart_implementation_fingerprint_cache
        root = Path(__file__).resolve().parent
        names = (
            "chain_field.py",
            "config.py",
            "crack_graph.py",
            "damage_control.py",
            "dolfinx_solver.py",
            "gb_graph.py",
            "homogeneous_at2.py",
            "hybrid_runner.py",
            "hybrid_state.py",
            "path_control.py",
        )
        digest = hashlib.sha256()
        digest.update(CONTAINER_IMAGE.encode())
        for name in names:
            path = root / name
            digest.update(name.encode())
            digest.update(path.read_bytes())
        self._restart_implementation_fingerprint_cache = digest.hexdigest()
        return self._restart_implementation_fingerprint_cache

    def _restart_runtime_fingerprint(self) -> str:
        if self._restart_runtime_fingerprint_cache is not None:
            return self._restart_runtime_fingerprint_cache
        encoded = json.dumps(
            self._runtime_identity(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._restart_runtime_fingerprint_cache = hashlib.sha256(encoded).hexdigest()
        return self._restart_runtime_fingerprint_cache

    def _continuation_from_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        base = self._continuation_controls(self.config.loading)
        if manifest["schema_version"] == self.LEGACY_RESTART_SCHEMA_VERSION:
            return {
                "policy_version": 1,
                "base_controls": base,
                "effective_controls": base,
                "sessions": [],
                "legacy_schema1_implementation_identity_unavailable": True,
            }
        continuation = manifest.get("continuation")
        if not isinstance(continuation, dict) or continuation.get("policy_version") != 1:
            raise RuntimeError("restart continuation policy is invalid")
        stored_base = continuation.get("base_controls")
        effective = continuation.get("effective_controls")
        sessions = continuation.get("sessions")
        if stored_base != base:
            raise RuntimeError("restart continuation base controls disagree with the config")
        if not isinstance(effective, dict) or not isinstance(sessions, list):
            raise RuntimeError("restart continuation payload is invalid")
        try:
            continuation_control_increase(base, effective, allow_equal=True)
        except ValueError as exc:
            raise RuntimeError(
                f"restart effective continuation controls are invalid: {exc}"
            ) from exc

        previous = base
        for index, session in enumerate(sessions, start=1):
            if not isinstance(session, dict) or session.get("session") != index:
                raise RuntimeError("restart continuation session numbering is invalid")
            before = session.get("controls_before")
            after = session.get("controls_after")
            if before != previous or not isinstance(after, dict):
                raise RuntimeError("restart continuation session chain is invalid")
            try:
                continuation_control_increase(before, after, allow_equal=True)
            except ValueError as exc:
                raise RuntimeError(
                    f"restart continuation session {index} is invalid: {exc}"
                ) from exc
            previous = after
        if previous != effective:
            raise RuntimeError("restart continuation sessions disagree with effective controls")
        return {
            "policy_version": 1,
            "base_controls": base,
            "effective_controls": effective,
            "sessions": sessions,
            "legacy_schema1_implementation_identity_unavailable": bool(
                continuation.get("legacy_schema1_implementation_identity_unavailable", False)
            ),
        }

    def _apply_resume_continuation_controls(
        self,
        *,
        accepted_step: int,
        accepted_displacement: float,
        stagger_max_iterations: int | None,
        maximum_subdivisions: int | None,
        minimum_increment: float | None,
    ) -> None:
        before = self._continuation_controls(self._effective_loading)
        requested = dict(before)
        explicit = {
            "stagger_max_iterations": stagger_max_iterations,
            "maximum_subdivisions": maximum_subdivisions,
            "minimum_increment": minimum_increment,
        }
        for name, value in explicit.items():
            if value is not None:
                requested[name] = value
        try:
            continuation_control_increase(before, requested, allow_equal=True)
        except ValueError as exc:
            raise RuntimeError(f"invalid resume continuation controls: {exc}") from exc
        self._effective_loading = replace(
            self.config.loading,
            stagger_max_iterations=int(requested["stagger_max_iterations"]),
            maximum_subdivisions=int(requested["maximum_subdivisions"]),
            minimum_increment=float(requested["minimum_increment"]),
        )
        if self.comm.rank == 0:
            session = {
                "session": len(self._continuation_sessions) + 1,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "parent_generation": self._restart_generation,
                "accepted_state": {
                    "accepted_step": accepted_step,
                    "displacement": accepted_displacement,
                },
                "requested_overrides": {
                    name: value for name, value in explicit.items() if value is not None
                },
                "controls_before": before,
                "controls_after": requested,
                "mpi_ranks": self.comm.size,
                "runtime_identity": self._runtime_identity(),
            }
        else:
            session = None
        session = self.comm.bcast(session, root=0)
        self._continuation_sessions.append(session)

    def _write_continuation_history(self, output: Path) -> None:
        if self.comm.rank != 0:
            return
        self._atomic_write_json(
            output / "continuation_history.json",
            {
                "schema_version": 1,
                "policy": (
                    "exact base-config checkpoint authentication followed by monotone "
                    "runtime continuation-budget increases"
                ),
                "base_controls": self._continuation_controls(self.config.loading),
                "effective_controls": self._continuation_controls(self._effective_loading),
                "sessions": self._continuation_sessions,
                "legacy_schema1_implementation_identity_unavailable": (
                    self._legacy_schema1_implementation_identity_unavailable
                ),
            },
        )

    def _restart_configuration_fingerprint(self) -> str:
        if self._restart_configuration_fingerprint_cache is not None:
            return self._restart_configuration_fingerprint_cache
        input_hashes: dict[str, str] = {}
        if self.config.source_path is not None and self.config.source_path.is_file():
            input_hashes["config_source"] = self._file_sha256(self.config.source_path)
        if self.config.graph.chain_artifact.strip():
            artifact = self.config.resolve_path(self.config.graph.chain_artifact)
            input_hashes["chain_artifact"] = self._file_sha256(artifact)
            manifest = artifact.with_suffix(".json")
            if manifest.is_file():
                input_hashes["chain_manifest"] = self._file_sha256(manifest)
        encoded = json.dumps(
            {
                "config": self.config.to_dict(),
                "input_sha256": input_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._restart_configuration_fingerprint_cache = hashlib.sha256(encoded).hexdigest()
        return self._restart_configuration_fingerprint_cache

    def _restart_partition_fingerprint(self) -> str:
        """Fingerprint the exact local mesh/dof partition used by rank-local shards."""
        if self._restart_partition_fingerprint_cache is not None:
            return self._restart_partition_fingerprint_cache
        digest = hashlib.sha256()
        digest.update(f"rank={self.comm.rank};size={self.comm.size}".encode())
        geometry = np.ascontiguousarray(self.domain.geometry.x)
        digest.update(str(geometry.shape).encode())
        digest.update(geometry.tobytes())
        for name in self._RESTART_FUNCTION_NAMES:
            function = getattr(self, name)
            space = function.function_space
            index_map = space.dofmap.index_map
            metadata = (
                name,
                index_map.size_local,
                index_map.num_ghosts,
                index_map.size_global,
                space.dofmap.index_map_bs,
                function.x.array.shape,
            )
            digest.update(repr(metadata).encode())
            coordinates = np.ascontiguousarray(space.tabulate_dof_coordinates())
            digest.update(str(coordinates.shape).encode())
            digest.update(coordinates.tobytes())
        self._restart_partition_fingerprint_cache = digest.hexdigest()
        return self._restart_partition_fingerprint_cache

    def _hybrid_boundary_field_fingerprint(self) -> str:
        """Fingerprint the actual graph/field provider, including injected graphs."""
        digest = hashlib.sha256()
        digest.update(b"graphfracture.hybrid-boundary-field.v1\0")
        field = self.gb_graph
        if field is None:
            digest.update(b"disabled")
            return digest.hexdigest()
        digest.update(f"{type(field).__module__}.{type(field).__qualname__}".encode())
        if isinstance(field, BoundaryGraph):
            payload = {
                "influence_radius": field.influence_radius,
                "nodes": [
                    {
                        "name": node.name,
                        "x": node.x,
                        "y": node.y,
                    }
                    for node in sorted(field.nodes.values(), key=lambda item: item.name)
                ],
                "edges": [
                    {
                        "source": min(edge.source, edge.target),
                        "target": max(edge.source, edge.target),
                        "toughness_ratio": edge.toughness_ratio,
                        "hydrogen_diffusivity_ratio": (edge.hydrogen_diffusivity_ratio),
                        "trap_density": edge.trap_density,
                    }
                    for edge in sorted(
                        field.edges,
                        key=lambda item: (
                            min(item.source, item.target),
                            max(item.source, item.target),
                            item.toughness_ratio,
                            item.hydrogen_diffusivity_ratio,
                            item.trap_density,
                        ),
                    )
                ],
            }
        else:
            payload = dict(field.describe())
            # Runtime work counters do not define a continuum field.
            payload.pop("max_distance_pairs_observed", None)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(encoded)
        for name in (
            "_a",
            "_ab",
            "_tough",
            "_diff",
            "_trap",
            "_segment_chain",
            "_source_chain_ids",
        ):
            if hasattr(field, name):
                values = np.ascontiguousarray(getattr(field, name))
                digest.update(name.encode())
                digest.update(str(values.dtype).encode())
                digest.update(repr(values.shape).encode())
                digest.update(values.tobytes())
        return digest.hexdigest()

    def _hybrid_material_field_fingerprint(self) -> str:
        """Fingerprint static graph-derived FE fields on this exact partition."""
        digest = hashlib.sha256()
        digest.update(b"graphfracture.hybrid-material-fields.v1\0")
        digest.update(f"rank={self.comm.rank};size={self.comm.size}".encode())
        for name in ("Gc0", "diffusivity", "trap_density"):
            values = np.ascontiguousarray(getattr(self, name).x.array)
            digest.update(name.encode())
            digest.update(str(values.dtype).encode())
            digest.update(repr(values.shape).encode())
            digest.update(values.tobytes())
        return digest.hexdigest()

    def _restart_state_arrays(self) -> dict[str, np.ndarray]:
        state = {
            name: np.array(getattr(self, name).x.array, copy=True)
            for name in self._RESTART_FUNCTION_NAMES
        }
        state["load"] = np.array(self.load.value, copy=True)
        return state

    def _restore_restart_state_arrays(self, state: dict[str, np.ndarray]) -> None:
        for name in self._RESTART_FUNCTION_NAMES:
            function = getattr(self, name)
            values = state[name]
            if values.shape != function.x.array.shape:
                raise RuntimeError(
                    f"restart array {name!r} has shape {values.shape}, "
                    f"expected {function.x.array.shape}"
                )
            function.x.array[:] = values
            function.x.scatter_forward()
        expected_load_shape = np.asarray(self.load.value).shape
        if state["load"].shape != expected_load_shape:
            raise RuntimeError(
                f"restart load array has shape {state['load'].shape}, "
                f"expected {expected_load_shape}"
            )
        self.load.value = state["load"]

    @staticmethod
    def _pending_to_json(
        pending: list[tuple[float, int, int]],
    ) -> list[dict[str, float | int]]:
        return [
            {
                "displacement": float(displacement),
                "subdivision_level": int(subdivision_level),
                "scheduled_step": int(scheduled_step),
            }
            for displacement, subdivision_level, scheduled_step in pending
        ]

    def _validate_restart_manifest(self, manifest: Any) -> dict[str, Any]:
        if not isinstance(manifest, dict):
            raise RuntimeError("restart manifest must be a JSON object")
        if manifest.get("schema_version") not in {
            self.LEGACY_RESTART_SCHEMA_VERSION,
            self.RESTART_SCHEMA_VERSION,
        }:
            raise RuntimeError("restart manifest schema is unsupported")
        if manifest.get("status") != "partial":
            raise RuntimeError("restart manifest is not a partial accepted-state checkpoint")
        if manifest.get("config_fingerprint") != self._restart_configuration_fingerprint():
            raise RuntimeError("restart configuration does not exactly match the checkpoint")
        if (
            manifest["schema_version"] == self.RESTART_SCHEMA_VERSION
            and manifest.get("implementation_fingerprint")
            != self._restart_implementation_fingerprint()
        ):
            raise RuntimeError("restart implementation fingerprint does not match")
        if (
            manifest["schema_version"] == self.RESTART_SCHEMA_VERSION
            and manifest.get("runtime_fingerprint") != self._restart_runtime_fingerprint()
        ):
            raise RuntimeError("restart runtime fingerprint does not match")
        if manifest.get("mpi_ranks") != self.comm.size:
            raise RuntimeError(
                "restart requires the same MPI rank count as the checkpoint "
                f"({manifest.get('mpi_ranks')} != {self.comm.size})"
            )
        generation = manifest.get("generation")
        slot = manifest.get("slot")
        if type(generation) is not int or generation < 0:
            raise RuntimeError("restart manifest has an invalid generation")
        if type(slot) is not int or slot not in {0, 1} or slot != generation % 2:
            raise RuntimeError("restart manifest has an invalid checkpoint slot")
        if manifest["schema_version"] == self.RESTART_SCHEMA_VERSION:
            parent = manifest.get("parent_checkpoint")
            if generation == 0:
                if parent is not None:
                    raise RuntimeError("initial restart checkpoint must not have a parent")
            else:
                if not isinstance(parent, dict):
                    raise RuntimeError("restart parent checkpoint is missing")
                parent_generation = parent.get("generation")
                archive_name = parent.get("archived_manifest")
                parent_sha256 = parent.get("manifest_sha256")
                if (
                    type(parent_generation) is not int
                    or parent_generation != generation - 1
                    or not isinstance(archive_name, str)
                    or archive_name
                    != f"manifests/checkpoint_generation_{parent_generation:06d}.json"
                    or Path(archive_name).is_absolute()
                    or ".." in Path(archive_name).parts
                    or not isinstance(parent_sha256, str)
                    or len(parent_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in parent_sha256)
                    or not isinstance(parent.get("shards"), list)
                ):
                    raise RuntimeError("restart parent checkpoint metadata is invalid")

        history = manifest.get("history")
        interface_history = manifest.get("interface_history")
        attempt_history = manifest.get("attempt_history")
        accepted = manifest.get("accepted_state")
        pending = manifest.get("pending_loads")
        shards = manifest.get("shards")
        partitions = manifest.get("partition_fingerprints")
        if not isinstance(history, list) or not history:
            raise RuntimeError("restart manifest contains no accepted mechanical history")
        if not isinstance(interface_history, list):
            raise RuntimeError("restart manifest interface history is invalid")
        if not isinstance(attempt_history, list):
            raise RuntimeError("restart manifest attempt history is invalid")
        if not isinstance(accepted, dict) or not isinstance(pending, list):
            raise RuntimeError("restart manifest load-state payload is invalid")
        if not isinstance(shards, list) or len(shards) != self.comm.size:
            raise RuntimeError("restart manifest shard count disagrees with MPI size")
        if not isinstance(partitions, list) or len(partitions) != self.comm.size:
            raise RuntimeError("restart manifest partition count disagrees with MPI size")
        continuation = self._continuation_from_manifest(manifest)
        effective_controls = continuation["effective_controls"]

        accepted_step = accepted.get("accepted_step")
        accepted_displacement = accepted.get("displacement")
        if type(accepted_step) is not int or accepted_step != len(history) - 1:
            raise RuntimeError("restart accepted-step count disagrees with mechanical history")
        try:
            displacement = float(accepted_displacement)
            history_displacement = float(history[-1]["displacement"])
            history_step = int(history[-1]["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("restart accepted-state history payload is invalid") from exc
        if not math.isfinite(displacement) or not math.isclose(
            displacement,
            history_displacement,
            rel_tol=0.0,
            abs_tol=1.0e-14 * max(1.0, abs(displacement)),
        ):
            raise RuntimeError("restart displacement disagrees with mechanical history")
        if history_step != accepted_step:
            raise RuntimeError("restart accepted step disagrees with the final history row")
        previous_history_displacement = -math.inf
        for index, record in enumerate(history):
            if not isinstance(record, dict):
                raise RuntimeError(f"restart history record {index} is invalid")
            try:
                record_step = int(record["step"])
                record_displacement = float(record["displacement"])
                converged = record["stagger_converged"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"restart history record {index} is invalid") from exc
            if (
                record_step != index
                or not math.isfinite(record_displacement)
                or record_displacement < previous_history_displacement
                or type(converged) is not bool
                or converged is not True
            ):
                raise RuntimeError(f"restart history record {index} is inconsistent")
            previous_history_displacement = record_displacement
        if self._interface_tracking_enabled() and len(interface_history) != len(history):
            raise RuntimeError("restart interface history disagrees with mechanical history")
        if not self._interface_tracking_enabled() and interface_history:
            raise RuntimeError("restart contains interface records for an unconfigured protocol")

        previous = displacement
        for index, item in enumerate(pending):
            if not isinstance(item, dict):
                raise RuntimeError(f"restart pending load {index} is invalid")
            try:
                target = float(item["displacement"])
                level = int(item["subdivision_level"])
                scheduled = int(item["scheduled_step"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"restart pending load {index} is invalid") from exc
            if (
                not math.isfinite(target)
                or target <= previous
                or target
                > self.config.loading.maximum_displacement
                + 1.0e-14 * max(1.0, self.config.loading.maximum_displacement)
                or level < 0
                or level > effective_controls["maximum_subdivisions"]
                or scheduled < 1
                or scheduled > self.config.loading.steps
            ):
                raise RuntimeError(f"restart pending load {index} violates loading controls")
            previous = target
        maximum = self.config.loading.maximum_displacement
        tolerance = 1.0e-14 * max(1.0, abs(maximum))
        if pending:
            if not math.isclose(
                float(pending[-1]["displacement"]),
                maximum,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise RuntimeError("restart pending queue does not end at the configured maximum")
        elif displacement < maximum - tolerance:
            raise RuntimeError("restart pending queue is empty before the configured maximum")
        return continuation

    @staticmethod
    def _hybrid_close(left: float, right: float, *, scale: float = 1.0) -> bool:
        return math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=1.0e-12 * max(1.0, abs(left), abs(right), abs(scale)),
        )

    def _hybrid_energy_close(
        self,
        left: float,
        right: float,
        *,
        increment: float,
    ) -> bool:
        tolerance = max(
            1.0e-12 * max(1.0, abs(left), abs(right)),
            self.config.path_control.control_tolerance
            * max(abs(increment), self.config.path_control.minimum_increment),
        )
        return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)

    def _hybrid_control_residual_certificate(
        self,
        *,
        accepted_value: float,
        target_value: float,
        achieved_value: float,
    ):
        """Apply the online path-control certificate during persistence audits."""
        return fracture_energy_control_residual_certificate(
            achieved_value - target_value,
            accepted_value=accepted_value,
            target_value=target_value,
            relative_tolerance=self.config.path_control.control_tolerance,
            absolute_tolerance=(self.config.path_control.control_absolute_tolerance),
        )

    def _hybrid_control_coordinate_close(
        self,
        left: float,
        right: float,
        *,
        coordinate_scale: float | None = None,
    ) -> bool:
        """Compare control coordinates without losing subtraction-roundoff ULPs.

        Refined increments are differences of absolute fracture energies.  At
        level seven or eight their subtraction error is governed by the ULP of
        those absolute coordinates, not by the much smaller increment itself.
        Sixteen coordinate ULPs cover the short midpoint/subtraction chain but
        remain far below the existing relative tolerance for absolute targets.
        """
        scale = max(
            abs(left),
            abs(right),
            abs(coordinate_scale) if coordinate_scale is not None else 0.0,
        )
        ulp_tolerance = 16.0 * math.ulp(scale)
        return math.isclose(
            left,
            right,
            rel_tol=1.0e-12,
            abs_tol=max(
                ulp_tolerance,
                1.0e-14
                * max(
                    self.config.path_control.target_increment,
                    self.config.path_control.minimum_increment,
                    abs(left),
                    abs(right),
                ),
            ),
        )

    @staticmethod
    def _hybrid_control_increment_meets_minimum(
        interval: float,
        *,
        minimum_increment: float,
        coordinate_scale: float,
    ) -> bool:
        """Apply a lower increment bound with absolute-coordinate ULP slack."""
        tolerance = max(
            16.0 * math.ulp(abs(coordinate_scale)),
            1.0e-14
            * max(
                abs(interval),
                abs(minimum_increment),
                1.0e-30,
            ),
        )
        return interval >= minimum_increment - tolerance

    def _hybrid_dyadic_interval_valid(
        self,
        previous: float,
        target: float,
        *,
        origin: float,
        nominal_increment: float,
        subdivision_level: int,
        adaptive: bool,
        maximum_subdivisions: int,
    ) -> bool:
        """Return whether one interval is reachable by refining one nominal cell."""
        if (
            type(subdivision_level) is not int
            or subdivision_level < 0
            or subdivision_level > maximum_subdivisions
            or (subdivision_level > 0 and not adaptive)
        ):
            return False
        interval = target - previous
        dyadic_increment = nominal_increment / (2**subdivision_level)
        if not self._hybrid_control_coordinate_close(
            interval,
            dyadic_increment,
            coordinate_scale=max(abs(origin), abs(previous), abs(target)),
        ):
            return False
        scaled_previous = (previous - origin) / dyadic_increment
        scaled_target = (target - origin) / dyadic_increment
        if not math.isclose(
            scaled_previous,
            round(scaled_previous),
            rel_tol=0.0,
            abs_tol=1.0e-10 * max(1.0, abs(scaled_previous)),
        ) or not math.isclose(
            scaled_target,
            round(scaled_target),
            rel_tol=0.0,
            abs_tol=1.0e-10 * max(1.0, abs(scaled_target)),
        ):
            return False
        ratio = (target - origin) / nominal_increment
        nominal_index = math.ceil(ratio - 1.0e-12 * max(1.0, abs(ratio)))
        lower = origin + (nominal_index - 1) * nominal_increment
        upper = origin + nominal_index * nominal_increment
        tolerance = 1.0e-12 * max(
            1.0,
            abs(origin),
            abs(previous),
            abs(target),
            abs(nominal_increment),
        )
        return previous >= lower - tolerance and target <= upper + tolerance

    def _validate_hybrid_history_schema(
        self,
        history: list[dict[str, Any]],
    ) -> None:
        expected = set(self._HISTORY_RECORD_FIELDS)
        integer_fields = (
            "step",
            "scheduled_step",
            "subdivision_level",
            "phase_step",
            "stagger_iterations",
            "damage_snes_iterations",
            "damage_snes_reason",
            "elastic_ksp_reason",
            "aitken_accepted_iterations",
        )
        finite_fields = (
            "displacement",
            "load_increment",
            "path_coordinate",
            "path_increment",
            "control_target",
            "control_value",
            "control_residual_relative",
            "reaction_y",
            "traction_reaction_y",
            "elastic_energy",
            "fracture_energy",
            "total_internal_energy",
            "regularised_crack_length",
            "maximum_damage",
            "stagger_error",
            "minimum_damage_increment",
            "damage_kkt_inf",
            "damage_kkt_relative",
            "damage_kkt_scale",
            "external_work",
            "energy_balance_residual",
            "energy_balance_relative",
        )
        nullable_finite_fields = (
            "rightmost_damaged_x",
            "load_factor",
            "reference_displacement",
            "mechanical_residual_relative",
            "final_aitken_relaxation",
        )
        nullable_integer_fields = (
            "path_snes_iterations",
            "path_snes_reason",
            "path_ksp_reason",
        )
        for index, record in enumerate(history):
            if not isinstance(record, dict) or set(record) != expected:
                raise RuntimeError(
                    f"hybrid restart history record {index} does not use the fixed schema"
                )
            if any(type(record[name]) is not int for name in integer_fields):
                raise RuntimeError(
                    f"hybrid restart history record {index} has invalid integer provenance"
                )
            if any(
                type(record[name]) not in {int, float} or not math.isfinite(float(record[name]))
                for name in finite_fields
            ):
                raise RuntimeError(
                    f"hybrid restart history record {index} has non-finite physics fields"
                )
            for name in nullable_finite_fields:
                value = record[name]
                if value is not None and (
                    type(value) not in {int, float} or not math.isfinite(float(value))
                ):
                    raise RuntimeError(f"hybrid restart history record {index} has invalid {name}")
            for name in nullable_integer_fields:
                value = record[name]
                if value is not None and type(value) is not int:
                    raise RuntimeError(f"hybrid restart history record {index} has invalid {name}")
            if (
                record["step"] != index
                or record["scheduled_step"] < 0
                or record["subdivision_level"] < 0
                or record["phase_step"] < 0
                or record["stagger_iterations"] < 1
                or record["stagger_converged"] is not True
                or record["damage_snes_iterations"] < 0
                or record["damage_snes_reason"] <= 0
                or record["elastic_ksp_reason"] <= 0
                or record["aitken_accepted_iterations"] < 0
                or record["aitken_accepted_iterations"] > record["stagger_iterations"]
                or type(record["control_phase"]) is not str
                or record["load_factor_bound_status"] is not None
            ):
                raise RuntimeError(
                    f"hybrid restart history record {index} has invalid acceptance provenance"
                )
            expected_total = float(record["elastic_energy"]) + float(record["fracture_energy"])
            if not self._hybrid_energy_close(
                float(record["total_internal_energy"]),
                expected_total,
                increment=self.config.path_control.minimum_increment,
            ):
                raise RuntimeError(
                    f"hybrid restart history record {index} has inconsistent total energy"
                )
            kkt_absolute = float(record["damage_kkt_inf"])
            kkt_relative = float(record["damage_kkt_relative"])
            kkt_scale = float(record["damage_kkt_scale"])
            expected_kkt_relative = kkt_absolute / max(kkt_scale, 1.0e-30)
            nonnegative_energy_tolerance = 1.0e-12 * max(
                1.0,
                abs(float(record["elastic_energy"])),
                abs(float(record["fracture_energy"])),
                abs(float(record["total_internal_energy"])),
                abs(float(record["regularised_crack_length"])),
            )
            maximum_damage = float(record["maximum_damage"])
            if (
                float(record["stagger_error"]) < 0.0
                or float(record["elastic_energy"]) < -nonnegative_energy_tolerance
                or float(record["fracture_energy"]) < -nonnegative_energy_tolerance
                or float(record["total_internal_energy"]) < -nonnegative_energy_tolerance
                or float(record["regularised_crack_length"]) < -nonnegative_energy_tolerance
                or maximum_damage < -1.0e-10
                or maximum_damage > 1.0 + 1.0e-10
                or float(record["minimum_damage_increment"]) < -1.0e-10
                or kkt_absolute < 0.0
                or kkt_relative < 0.0
                or kkt_scale < 0.0
                or not math.isclose(
                    kkt_relative,
                    expected_kkt_relative,
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-14 * max(1.0, expected_kkt_relative),
                )
                or kkt_relative > self.config.loading.damage_kkt_tolerance
            ):
                raise RuntimeError(
                    f"hybrid restart history record {index} has an invalid physics certificate"
                )

        initial_internal = float(history[0]["total_internal_energy"])
        for index, record in enumerate(history):
            if index == 0:
                expected_work = 0.0
            else:
                previous = history[index - 1]
                displacement_increment = float(record["displacement"]) - float(
                    previous["displacement"]
                )
                work_increment = (
                    0.5
                    * (float(previous["reaction_y"]) + float(record["reaction_y"]))
                    * displacement_increment
                )
                expected_work = float(previous["external_work"]) + work_increment
            internal_change = float(record["total_internal_energy"]) - initial_internal
            expected_residual = internal_change - expected_work
            balance_scale = max(
                abs(internal_change),
                abs(expected_work),
                1.0e-30,
            )
            expected_relative = expected_residual / balance_scale
            if (
                not self._hybrid_close(
                    float(record["external_work"]),
                    expected_work,
                    scale=expected_work,
                )
                or not self._hybrid_close(
                    float(record["energy_balance_residual"]),
                    expected_residual,
                    scale=expected_residual,
                )
                or not self._hybrid_close(
                    float(record["energy_balance_relative"]),
                    expected_relative,
                    scale=expected_relative,
                )
            ):
                raise RuntimeError(
                    f"hybrid restart history record {index} has an inconsistent "
                    "energy-balance recurrence"
                )
        previous_tip = -math.inf
        tip_seen = False
        for index, record in enumerate(history):
            tip = record["rightmost_damaged_x"]
            if tip is None:
                if tip_seen:
                    raise RuntimeError("hybrid restart rightmost damage tip disappears in time")
                continue
            value = float(tip)
            if (
                value < -1.0e-12
                or value > self.config.geometry.length + 1.0e-12
                or value + 1.0e-12 < previous_tip
            ):
                raise RuntimeError(
                    f"hybrid restart history record {index} has a non-monotone rightmost damage tip"
                )
            tip_seen = True
            previous_tip = value

    def _validate_hybrid_displacement_history(
        self,
        records: list[dict[str, Any]],
        *,
        reference_displacement: float,
        description: str,
    ) -> None:
        """Authenticate accepted displacement steps and their dyadic provenance."""
        if not records:
            raise RuntimeError(f"hybrid {description} displacement history is empty")
        nominal = self.config.loading.maximum_displacement / self.config.loading.steps
        switch_index = round(reference_displacement / nominal)
        previous = 0.0
        for index, record in enumerate(records):
            displacement = float(record["displacement"])
            level = record["subdivision_level"]
            if index == 0:
                increment = 0.0
                expected_scheduled_step = 0
                zero_row_is_exact = all(
                    float(record[name]) == 0.0
                    for name in (
                        "displacement",
                        "load_increment",
                        "path_coordinate",
                        "path_increment",
                        "control_target",
                        "control_value",
                    )
                )
            else:
                increment = displacement - previous
                ratio = displacement / nominal
                expected_scheduled_step = math.ceil(ratio - 1.0e-12 * max(1.0, abs(ratio)))
                zero_row_is_exact = True
            if (
                record["control_phase"] != ControlPhase.DISPLACEMENT.value
                or record["phase_step"] != index
                or not zero_row_is_exact
                or (index > 0 and displacement <= previous)
                or displacement
                > reference_displacement + 1.0e-12 * max(1.0, abs(reference_displacement))
                or not self._hybrid_close(
                    float(record["load_increment"]),
                    increment,
                    scale=reference_displacement,
                )
                or not self._hybrid_close(
                    float(record["path_coordinate"]),
                    displacement,
                    scale=reference_displacement,
                )
                or not self._hybrid_close(
                    float(record["path_increment"]),
                    increment,
                    scale=reference_displacement,
                )
                or not self._hybrid_close(
                    float(record["control_target"]),
                    displacement,
                    scale=reference_displacement,
                )
                or not self._hybrid_close(
                    float(record["control_value"]),
                    displacement,
                    scale=reference_displacement,
                )
                or (
                    index > 0
                    and not self._hybrid_dyadic_interval_valid(
                        previous,
                        displacement,
                        origin=0.0,
                        nominal_increment=nominal,
                        subdivision_level=level,
                        adaptive=self.config.loading.adaptive,
                        maximum_subdivisions=(self.config.loading.maximum_subdivisions),
                    )
                )
                or float(record["control_residual_relative"]) != 0.0
                or record["load_factor"] is not None
                or record["reference_displacement"] is not None
                or record["path_snes_iterations"] is not None
                or record["path_snes_reason"] is not None
                or record["path_ksp_reason"] is not None
                or record["mechanical_residual_relative"] is not None
                or float(record["stagger_error"])
                > max(
                    self.config.loading.stagger_tolerance,
                    self.config.loading.damage_kkt_tolerance,
                )
                or record["final_aitken_relaxation"] is None
                or not 0.0
                < float(record["final_aitken_relaxation"])
                <= self.config.solver.aitken_max_relaxation
                or level > self.config.loading.maximum_subdivisions
                or (index == 0 and level != 0)
                or record["scheduled_step"] != expected_scheduled_step
                or expected_scheduled_step < 0
                or expected_scheduled_step > switch_index
            ):
                raise RuntimeError(
                    f"hybrid {description} displacement history must be strictly "
                    "increasing with valid dyadic provenance"
                )
            previous = displacement

    def _validate_hybrid_interface_history(
        self,
        history: list[dict[str, Any]],
        interface_history: list[dict[str, Any]],
    ) -> None:
        if self._interface_tracking_enabled():
            from .inclined_studies import _consensus, _validate_threshold_results

            if len(interface_history) != len(history):
                raise RuntimeError("hybrid restart interface history disagrees with history")
            protocol = self._gathered_interface_protocol_without_collective()
            protocol_scale_names = (
                "impact_tolerance",
                "interface_corridor_width",
                "impact_exclusion_radius",
                "confirmation_distance",
                "penetration_corridor_half_height",
            )
            interface_keys = {
                "accepted_step",
                "scheduled_step",
                "displacement",
                "subdivision_level",
                "threshold_consensus",
                "threshold_results",
            }
            threshold_keys = {
                "threshold",
                "impact_tolerance",
                "interface_corridor_width",
                "impact_exclusion_radius",
                "confirmation_distance",
                "penetration_corridor_half_height",
                "main_component_nodes",
                "reached_interface",
                "closest_main_node_to_impact",
                "interface_forward_advance",
                "penetration_forward_advance",
                "interface_active_edge_length",
                "penetration_active_edge_length",
                "geometric_classification",
                "role",
                "active_edge_measure_note",
                "interpretation_limit",
            }
            previous_thresholds: list[dict[str, Any]] | None = None
            for index, (record, interface) in enumerate(
                zip(history, interface_history, strict=True)
            ):
                if not isinstance(interface, dict) or set(interface) != interface_keys:
                    raise RuntimeError(
                        f"hybrid restart interface record {index} has an invalid schema"
                    )
                try:
                    interface_step = interface["accepted_step"]
                    scheduled_step = interface["scheduled_step"]
                    subdivision_level = interface["subdivision_level"]
                    displacement = float(interface["displacement"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"hybrid restart interface record {index} is invalid"
                    ) from exc
                if (
                    type(interface_step) is not int
                    or interface_step != record["step"]
                    or type(scheduled_step) is not int
                    or scheduled_step != record["scheduled_step"]
                    or type(subdivision_level) is not int
                    or subdivision_level != record["subdivision_level"]
                    or not math.isfinite(displacement)
                    or not self._hybrid_close(
                        displacement,
                        float(record["displacement"]),
                        scale=displacement,
                    )
                ):
                    raise RuntimeError(
                        f"hybrid restart interface record {index} disagrees with history"
                    )
                try:
                    threshold_results = _validate_threshold_results(
                        interface["threshold_results"],
                        f"hybrid restart interface record {index}",
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"hybrid restart interface record {index} has invalid thresholds"
                    ) from exc
                for threshold_index, result in enumerate(threshold_results):
                    raw = result["raw"]
                    if set(raw) != threshold_keys:
                        raise RuntimeError(
                            "hybrid restart interface threshold result uses an invalid fixed schema"
                        )
                    if (
                        type(raw["main_component_nodes"]) is not int
                        or raw["main_component_nodes"] < 0
                        or any(
                            type(raw[name]) not in {int, float}
                            or not math.isfinite(float(raw[name]))
                            or float(raw[name]) < 0.0
                            for name in (
                                "impact_tolerance",
                                "interface_corridor_width",
                                "impact_exclusion_radius",
                                "confirmation_distance",
                                "penetration_corridor_half_height",
                            )
                        )
                        or any(
                            type(raw[name]) is not str or not raw[name]
                            for name in (
                                "role",
                                "active_edge_measure_note",
                                "interpretation_limit",
                            )
                        )
                    ):
                        raise RuntimeError(
                            "hybrid restart interface threshold result has invalid "
                            "geometric provenance"
                        )
                    if float(raw["threshold"]) != float(
                        protocol["thresholds"][threshold_index]
                    ) or any(
                        float(raw[name]) != float(protocol[name]) for name in protocol_scale_names
                    ):
                        raise RuntimeError(
                            "hybrid restart interface threshold result disagrees "
                            "with the configured geometric protocol"
                        )
                    closest = result["closest_main_node_to_impact"]
                    reached_from_geometry = closest is not None and closest <= float(
                        raw["impact_tolerance"]
                    )
                    if result["reached_interface"] is not reached_from_geometry or (
                        raw["main_component_nodes"] == 0
                    ) != (closest is None):
                        raise RuntimeError(
                            "hybrid restart interface reachability disagrees with "
                            "the recorded closest-distance geometry"
                        )
                    confirmation = float(raw["confirmation_distance"])
                    deflection_confirmed = result["interface_forward_advance"] >= confirmation
                    penetration_confirmed = result["penetration_forward_advance"] >= confirmation
                    if not result["reached_interface"]:
                        expected_classification = "pre_impact_right_censored"
                    elif not deflection_confirmed and not penetration_confirmed:
                        expected_classification = "arrested_or_unresolved"
                    elif deflection_confirmed and not penetration_confirmed:
                        expected_classification = "deflection_candidate"
                    elif penetration_confirmed and not deflection_confirmed:
                        expected_classification = "penetration_candidate"
                    else:
                        expected_classification = "mixed_or_branched"
                    if result["classification"] != expected_classification:
                        raise RuntimeError(
                            "hybrid restart interface geometric classification "
                            "is inconsistent with its deterministic protocol"
                        )
                    if previous_thresholds is not None:
                        previous = previous_thresholds[threshold_index]
                        if previous["reached_interface"] and not result["reached_interface"]:
                            raise RuntimeError(
                                "hybrid restart interface reachability is not irreversible"
                            )
                        for name in (
                            "interface_forward_advance",
                            "penetration_forward_advance",
                        ):
                            if result[name] + 1.0e-12 < previous[name]:
                                raise RuntimeError(
                                    "hybrid restart interface advance decreases in time"
                                )
                        previous_closest = previous["closest_main_node_to_impact"]
                        current_closest = result["closest_main_node_to_impact"]
                        if (
                            previous_closest is not None
                            and current_closest is not None
                            and current_closest > previous_closest + 1.0e-12
                        ):
                            raise RuntimeError(
                                "hybrid restart closest interface distance increases"
                            )
                consensus = interface["threshold_consensus"]
                if type(consensus) is not str or consensus != _consensus(threshold_results):
                    raise RuntimeError(
                        f"hybrid restart interface record {index} has inconsistent consensus"
                    )
                previous_thresholds = threshold_results
        elif interface_history:
            raise RuntimeError(
                "hybrid restart contains interface records for an unconfigured protocol"
            )

    def _validate_hybrid_subdivision_ancestry(
        self,
        history: list[dict[str, Any]],
        attempt_history: list[dict[str, Any]],
        scheduler_state: HybridSchedulerState,
    ) -> None:
        """Require every refined interval to retain its complete failure ancestry."""
        nominal_displacement = self.config.loading.maximum_displacement / self.config.loading.steps
        requirements: list[tuple[str, float, float, int, int, str]] = []
        switch_energy: float | None = None

        previous_displacement = 0.0
        for index, record in enumerate(history):
            if record["control_phase"] != ControlPhase.DISPLACEMENT.value:
                break
            target = float(record["displacement"])
            level = int(record["subdivision_level"])
            if index > 0 and level > 0:
                requirements.append(
                    (
                        ControlPhase.DISPLACEMENT.value,
                        previous_displacement,
                        target,
                        level,
                        int(record["scheduled_step"]),
                        f"accepted displacement interval ending at history row {index}",
                    )
                )
            previous_displacement = target

        if scheduler_state.state.phase is ControlPhase.DISPLACEMENT:
            previous_displacement = float(history[-1]["displacement"])
            for index, target in enumerate(scheduler_state.pending_displacements):
                if target.subdivision_level > 0:
                    requirements.append(
                        (
                            ControlPhase.DISPLACEMENT.value,
                            previous_displacement,
                            target.displacement,
                            target.subdivision_level,
                            target.scheduled_step,
                            f"pending displacement interval {index}",
                        )
                    )
                previous_displacement = target.displacement

        switch_step = scheduler_state.switch_accepted_step
        if switch_step is not None:
            switch_energy = float(history[switch_step]["fracture_energy"])
            previous_energy = switch_energy
            for index, record in enumerate(history[switch_step + 1 :], start=1):
                target = float(record["control_target"])
                level = int(record["subdivision_level"])
                if level > 0:
                    requirements.append(
                        (
                            ControlPhase.FRACTURE_ENERGY.value,
                            previous_energy,
                            target,
                            level,
                            int(record["scheduled_step"]),
                            f"accepted fracture-energy interval {index}",
                        )
                    )
                previous_energy = target

            queue = scheduler_state.fracture_energy_queue
            if queue is not None:
                previous_energy = queue.accepted_value
                switch_index = round(scheduler_state.reference_displacement / nominal_displacement)
                for index, target in enumerate(queue.pending):
                    ratio = (target.value - switch_energy) / (
                        self.config.path_control.target_increment
                    )
                    scheduled_step = switch_index + math.ceil(
                        ratio - 1.0e-12 * max(1.0, abs(ratio))
                    )
                    if target.subdivision_level > 0:
                        requirements.append(
                            (
                                ControlPhase.FRACTURE_ENERGY.value,
                                previous_energy,
                                target.value,
                                target.subdivision_level,
                                scheduled_step,
                                f"pending fracture-energy interval {index}",
                            )
                        )
                    previous_energy = target.value

        attempt_entry = tuple[int, dict[str, Any]]
        coarse_index: dict[
            tuple[str, int, int],
            list[attempt_entry],
        ] = {}
        lattice_index: dict[
            tuple[str, int, int, int, int],
            list[attempt_entry],
        ] = {}
        for attempt_index, attempt in enumerate(attempt_history, start=1):
            if attempt["will_subdivide"] is not True:
                continue
            phase = str(attempt["control_phase"])
            level = int(attempt["subdivision_level"])
            scheduled_step = int(attempt["scheduled_step"])
            if phase == ControlPhase.DISPLACEMENT.value:
                origin = 0.0
                nominal_increment = nominal_displacement
            elif phase == ControlPhase.FRACTURE_ENERGY.value and switch_energy is not None:
                origin = switch_energy
                nominal_increment = self.config.path_control.target_increment
            else:  # Validated before this ancestry check; retained defensively.
                continue
            dyadic_increment = nominal_increment / (2**level)
            start_index = round((float(attempt["accepted_control"]) - origin) / dyadic_increment)
            endpoint_index = round((float(attempt["target_control"]) - origin) / dyadic_increment)
            entry = (attempt_index, attempt)
            coarse_index.setdefault(
                (phase, scheduled_step, level),
                [],
            ).append(entry)
            lattice_index.setdefault(
                (
                    phase,
                    scheduled_step,
                    level,
                    start_index,
                    endpoint_index,
                ),
                [],
            ).append(entry)

        maximum_displacement = self.config.loading.maximum_displacement
        displacement_tolerance = 1.0e-12 * max(
            1.0,
            abs(maximum_displacement),
            abs(nominal_displacement),
        )
        if switch_energy is None:
            energy_tolerance = math.inf
        else:
            terminal_energy = (
                switch_energy
                + self.config.path_control.steps * self.config.path_control.target_increment
            )
            energy_scale = max(abs(switch_energy), abs(terminal_energy))
            energy_tolerance = max(
                1.0e-12 * energy_scale,
                16.0 * math.ulp(energy_scale),
                1.0e-14
                * max(
                    self.config.path_control.target_increment,
                    self.config.path_control.minimum_increment,
                    energy_scale,
                ),
            )

        match_cache: dict[
            tuple[str, int, int, int, int],
            tuple[int, ...],
        ] = {}
        for phase, start, target, level, scheduled_step, description in requirements:
            if phase == ControlPhase.DISPLACEMENT.value:
                origin = 0.0
                nominal_increment = nominal_displacement
                matching_tolerance = displacement_tolerance
            else:
                if switch_energy is None:  # pragma: no cover - collected above only.
                    raise RuntimeError("hybrid energy ancestry has no switch state")
                origin = switch_energy
                nominal_increment = self.config.path_control.target_increment
                matching_tolerance = energy_tolerance

            current_start = start
            current_target = target
            current_level = level
            child_to_parent_attempts: list[int] = []
            while current_level > 0:
                child_increment = nominal_increment / (2**current_level)
                start_index = round((current_start - origin) / child_increment)
                endpoint_index = round((current_target - origin) / child_increment)
                if endpoint_index - start_index != 1:
                    raise RuntimeError(f"hybrid {description} is not one dyadic child interval")
                parent_start_index = ((endpoint_index - 1) // 2) * 2
                parent_start = origin + parent_start_index * child_increment
                parent_target = parent_start + 2.0 * child_increment
                parent_level = current_level - 1
                parent_increment = nominal_increment / (2**parent_level)
                expected_start_index = round((parent_start - origin) / parent_increment)
                expected_endpoint_index = round((parent_target - origin) / parent_increment)
                cache_key = (
                    phase,
                    scheduled_step,
                    parent_level,
                    expected_start_index,
                    expected_endpoint_index,
                )
                matches = match_cache.get(cache_key)
                if matches is None:
                    lattice_is_separated = parent_increment > 4.0 * matching_tolerance
                    candidates = (
                        lattice_index.get(cache_key, ())
                        if lattice_is_separated
                        else coarse_index.get(
                            (phase, scheduled_step, parent_level),
                            (),
                        )
                    )
                    matched_indices: list[int] = []
                    for attempt_index, attempt in candidates:
                        accepted_matches = (
                            self._hybrid_close(
                                float(attempt["accepted_control"]),
                                parent_start,
                                scale=nominal_displacement,
                            )
                            if phase == ControlPhase.DISPLACEMENT.value
                            else self._hybrid_control_coordinate_close(
                                float(attempt["accepted_control"]),
                                parent_start,
                            )
                        )
                        target_matches = (
                            self._hybrid_close(
                                float(attempt["target_control"]),
                                parent_target,
                                scale=nominal_displacement,
                            )
                            if phase == ControlPhase.DISPLACEMENT.value
                            else self._hybrid_control_coordinate_close(
                                float(attempt["target_control"]),
                                parent_target,
                            )
                        )
                        if accepted_matches and target_matches:
                            matched_indices.append(attempt_index)
                    matches = tuple(matched_indices)
                    match_cache[cache_key] = matches
                if len(matches) != 1:
                    raise RuntimeError(
                        f"hybrid {description} has no unique complete subdivision "
                        f"ancestor at level {parent_level}"
                    )
                child_to_parent_attempts.append(matches[0])
                current_start = parent_start
                current_target = parent_target
                current_level = parent_level

            root_to_child_attempts = list(reversed(child_to_parent_attempts))
            if any(child <= parent for parent, child in itertools.pairwise(root_to_child_attempts)):
                raise RuntimeError(f"hybrid {description} has an out-of-order subdivision ancestry")

    def _validate_hybrid_attempt_history(
        self,
        history: list[dict[str, Any]],
        attempt_history: list[dict[str, Any]],
        scheduler_state: HybridSchedulerState,
    ) -> None:
        """Authenticate failed continuation attempts and the retained queue head."""
        displacement_keys = {
            "attempt",
            "accepted_step_before_attempt",
            "control_phase",
            "accepted_control",
            "target_control",
            "trial_displacement",
            "accepted_displacement",
            "target_displacement",
            "load_increment",
            "scheduled_step",
            "subdivision_level",
            "failure_type",
            "failure_message",
            "iterations",
            "error",
            "damage_snes_iterations",
            "damage_snes_reason",
            "elastic_ksp_reason",
            "aitken_accepted_iterations",
            "final_aitken_relaxation",
            "will_subdivide",
        }
        energy_keys = {
            "attempt",
            "accepted_step_before_attempt",
            "control_phase",
            "accepted_control",
            "target_control",
            "trial_displacement",
            "accepted_displacement",
            "target_displacement",
            "load_increment",
            "control_increment",
            "scheduled_step",
            "subdivision_level",
            "failure_type",
            "failure_message",
            "iterations",
            "error",
            "path_snes_reason",
            "path_ksp_reason",
            "load_factor_bound_status",
            "will_subdivide",
        }
        nominal_displacement = self.config.loading.maximum_displacement / self.config.loading.steps
        switch_index = round(scheduler_state.reference_displacement / nominal_displacement)
        switch_step = scheduler_state.switch_accepted_step
        switch_energy = (
            float(history[switch_step]["fracture_energy"])
            if switch_step is not None and switch_step < len(history)
            else None
        )
        previous_accepted_step = -1
        for index, attempt in enumerate(attempt_history, start=1):
            if not isinstance(attempt, dict):
                raise RuntimeError(f"hybrid restart attempt {index} is invalid")
            phase = attempt.get("control_phase")
            expected_keys = (
                displacement_keys
                if phase == ControlPhase.DISPLACEMENT.value
                else energy_keys
                if phase == ControlPhase.FRACTURE_ENERGY.value
                else None
            )
            if expected_keys is None or set(attempt) != expected_keys:
                raise RuntimeError(
                    f"hybrid restart attempt {index} does not use a fixed phase schema"
                )
            accepted_step = attempt["accepted_step_before_attempt"]
            scheduled_step = attempt["scheduled_step"]
            level = attempt["subdivision_level"]
            integer_fields = (
                "attempt",
                "accepted_step_before_attempt",
                "scheduled_step",
                "subdivision_level",
                "iterations",
            )
            if phase == ControlPhase.DISPLACEMENT.value:
                integer_fields += (
                    "damage_snes_iterations",
                    "damage_snes_reason",
                    "elastic_ksp_reason",
                    "aitken_accepted_iterations",
                )
            else:
                integer_fields += ("path_snes_reason", "path_ksp_reason")
            if any(type(attempt[name]) is not int for name in integer_fields):
                raise RuntimeError(f"hybrid restart attempt {index} has invalid integer provenance")
            if (
                attempt["attempt"] != index
                or accepted_step < 0
                or accepted_step >= len(history)
                or accepted_step < previous_accepted_step
                or scheduled_step < 1
                or level < 0
                or type(attempt["will_subdivide"]) is not bool
                or type(attempt["failure_type"]) is not str
                or not attempt["failure_type"]
                or type(attempt["failure_message"]) is not str
                or not attempt["failure_message"]
            ):
                raise RuntimeError(f"hybrid restart attempt {index} has invalid failure provenance")
            previous_accepted_step = accepted_step
            accepted_record = history[accepted_step]
            try:
                accepted_control = float(attempt["accepted_control"])
                target_control = float(attempt["target_control"])
                accepted_displacement = float(attempt["accepted_displacement"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"hybrid restart attempt {index} has invalid control scalars"
                ) from exc
            if not all(
                math.isfinite(value)
                for value in (
                    accepted_control,
                    target_control,
                    accepted_displacement,
                )
            ):
                raise RuntimeError(f"hybrid restart attempt {index} has non-finite control scalars")
            error = attempt["error"]
            if error is not None and (
                type(error) not in {int, float}
                or not math.isfinite(float(error))
                or float(error) < 0.0
            ):
                raise RuntimeError(f"hybrid restart attempt {index} has an invalid error metric")
            if not self._hybrid_close(
                accepted_displacement,
                float(accepted_record["displacement"]),
                scale=scheduler_state.reference_displacement,
            ):
                raise RuntimeError(
                    f"hybrid restart attempt {index} disagrees with its accepted state"
                )

            maximum_level = (
                self.config.loading.maximum_subdivisions
                if phase == ControlPhase.DISPLACEMENT.value
                else self.config.path_control.maximum_subdivisions
            )
            if level > maximum_level:
                raise RuntimeError(f"hybrid restart attempt {index} exceeds subdivision controls")

            if phase == ControlPhase.DISPLACEMENT.value:
                expected_increment = nominal_displacement / (2**level)
                interval = target_control - accepted_control
                ratio = target_control / nominal_displacement
                expected_scheduled = math.ceil(ratio - 1.0e-12 * max(1.0, abs(ratio)))
                trial = float(attempt["trial_displacement"])
                target_displacement = float(attempt["target_displacement"])
                load_increment = float(attempt["load_increment"])
                relaxation = attempt["final_aitken_relaxation"]
                if (
                    level > self.config.loading.maximum_subdivisions
                    or not all(
                        math.isfinite(value)
                        for value in (trial, target_displacement, load_increment)
                    )
                    or not self._hybrid_close(
                        accepted_control,
                        accepted_displacement,
                        scale=scheduler_state.reference_displacement,
                    )
                    or not self._hybrid_close(
                        accepted_control,
                        float(accepted_record["displacement"]),
                        scale=scheduler_state.reference_displacement,
                    )
                    or not self._hybrid_close(
                        trial,
                        target_control,
                        scale=scheduler_state.reference_displacement,
                    )
                    or not self._hybrid_close(
                        target_displacement,
                        target_control,
                        scale=scheduler_state.reference_displacement,
                    )
                    or not self._hybrid_close(
                        load_increment,
                        interval,
                        scale=scheduler_state.reference_displacement,
                    )
                    or not self._hybrid_close(
                        interval,
                        expected_increment,
                        scale=nominal_displacement,
                    )
                    or not self._hybrid_dyadic_interval_valid(
                        accepted_control,
                        target_control,
                        origin=0.0,
                        nominal_increment=nominal_displacement,
                        subdivision_level=level,
                        adaptive=self.config.loading.adaptive,
                        maximum_subdivisions=(self.config.loading.maximum_subdivisions),
                    )
                    or scheduled_step != expected_scheduled
                    or expected_scheduled > switch_index
                    or relaxation is not None
                    and (
                        type(relaxation) not in {int, float} or not math.isfinite(float(relaxation))
                    )
                ):
                    raise RuntimeError(
                        f"hybrid displacement attempt {index} violates dyadic provenance"
                    )
                can_subdivide = (
                    self.config.loading.adaptive
                    and level < self.config.loading.maximum_subdivisions
                    and 0.5 * interval >= self.config.loading.minimum_increment
                )
            else:
                if switch_energy is None or switch_step is None:
                    raise RuntimeError("hybrid energy attempts require an accepted switch state")
                expected_increment = self.config.path_control.target_increment / (2**level)
                interval = target_control - accepted_control
                ratio = (target_control - switch_energy) / (
                    self.config.path_control.target_increment
                )
                expected_scheduled = switch_index + math.ceil(
                    ratio - 1.0e-12 * max(1.0, abs(ratio))
                )
                expected_accepted_control = (
                    float(accepted_record["control_target"])
                    if accepted_record["control_phase"] == ControlPhase.FRACTURE_ENERGY.value
                    else float(accepted_record["fracture_energy"])
                )
                trial = attempt["trial_displacement"]
                target_displacement = attempt["target_displacement"]
                load_increment = attempt["load_increment"]
                nullable_displacements = (
                    trial,
                    target_displacement,
                    load_increment,
                )
                if any(
                    value is not None
                    and (type(value) not in {int, float} or not math.isfinite(float(value)))
                    for value in nullable_displacements
                ):
                    raise RuntimeError(
                        f"hybrid energy attempt {index} has invalid trial displacement"
                    )
                if (trial is None) != (target_displacement is None) or (
                    (trial is None) != (load_increment is None)
                ):
                    raise RuntimeError(
                        f"hybrid energy attempt {index} has incomplete trial displacement"
                    )
                if trial is not None and (
                    not self._hybrid_close(
                        float(target_displacement),
                        float(trial),
                        scale=scheduler_state.reference_displacement,
                    )
                    or not self._hybrid_close(
                        float(load_increment),
                        float(trial) - accepted_displacement,
                        scale=scheduler_state.reference_displacement,
                    )
                ):
                    raise RuntimeError(
                        f"hybrid energy attempt {index} has inconsistent trial displacement"
                    )
                control_increment = attempt["control_increment"]
                bound_status = attempt["load_factor_bound_status"]
                if (
                    type(control_increment) not in {int, float}
                    or not math.isfinite(float(control_increment))
                    or level > self.config.path_control.maximum_subdivisions
                    or not self._hybrid_control_coordinate_close(
                        accepted_control,
                        expected_accepted_control,
                    )
                    or not self._hybrid_control_coordinate_close(
                        float(control_increment),
                        interval,
                        coordinate_scale=max(
                            abs(accepted_control),
                            abs(target_control),
                        ),
                    )
                    or not self._hybrid_control_coordinate_close(
                        interval,
                        expected_increment,
                        coordinate_scale=max(
                            abs(accepted_control),
                            abs(target_control),
                        ),
                    )
                    or not self._hybrid_dyadic_interval_valid(
                        accepted_control,
                        target_control,
                        origin=switch_energy,
                        nominal_increment=(self.config.path_control.target_increment),
                        subdivision_level=level,
                        adaptive=self.config.path_control.adaptive,
                        maximum_subdivisions=(self.config.path_control.maximum_subdivisions),
                    )
                    or scheduled_step != expected_scheduled
                    or expected_scheduled > switch_index + self.config.path_control.steps
                    or bound_status not in {None, "lower", "upper"}
                ):
                    raise RuntimeError(f"hybrid energy attempt {index} violates dyadic provenance")
                can_subdivide = (
                    self.config.path_control.adaptive
                    and level < self.config.path_control.maximum_subdivisions
                    and self._hybrid_control_increment_meets_minimum(
                        0.5 * interval,
                        minimum_increment=(self.config.path_control.minimum_increment),
                        coordinate_scale=max(
                            abs(accepted_control),
                            abs(target_control),
                        ),
                    )
                )
            if attempt["will_subdivide"] is not can_subdivide:
                raise RuntimeError(
                    f"hybrid restart attempt {index} has inconsistent subdivision intent"
                )

        self._validate_hybrid_subdivision_ancestry(
            history,
            attempt_history,
            scheduler_state,
        )
        if not attempt_history:
            return
        last = attempt_history[-1]
        if (
            last["accepted_step_before_attempt"] != len(history) - 1
            or last["control_phase"] != scheduler_state.state.phase.value
        ):
            return
        accepted = float(last["accepted_control"])
        target = float(last["target_control"])
        level = int(last["subdivision_level"])
        if last["will_subdivide"]:
            expected_head = 0.5 * (accepted + target)
            expected_level = level + 1
            expected_second = target
        else:
            expected_head = target
            expected_level = level
            expected_second = None
        if scheduler_state.state.phase is ControlPhase.DISPLACEMENT:
            pending = scheduler_state.pending_displacements
            if (
                not pending
                or not self._hybrid_close(
                    pending[0].displacement,
                    expected_head,
                    scale=scheduler_state.reference_displacement,
                )
                or pending[0].subdivision_level != expected_level
                or pending[0].scheduled_step != last["scheduled_step"]
                or expected_second is not None
                and (
                    len(pending) < 2
                    or not self._hybrid_close(
                        pending[1].displacement,
                        expected_second,
                        scale=scheduler_state.reference_displacement,
                    )
                    or pending[1].subdivision_level != expected_level
                    or pending[1].scheduled_step != last["scheduled_step"]
                )
            ):
                raise RuntimeError(
                    "hybrid displacement failure head is not retained by the scheduler"
                )
        else:
            queue = scheduler_state.fracture_energy_queue
            pending = queue.pending if queue is not None else ()
            if (
                not pending
                or not self._hybrid_control_coordinate_close(
                    pending[0].value,
                    expected_head,
                )
                or pending[0].subdivision_level != expected_level
                or expected_second is not None
                and (
                    len(pending) < 2
                    or not self._hybrid_control_coordinate_close(
                        pending[1].value,
                        expected_second,
                    )
                    or pending[1].subdivision_level != expected_level
                )
            ):
                raise RuntimeError(
                    "hybrid fracture-energy failure head is not retained by the scheduler"
                )

    def _validate_hybrid_restart_manifest(
        self,
        manifest: Any,
    ) -> HybridSchedulerState:
        """Validate the schema-3 envelope and its phase-aware scheduler payload."""
        if not isinstance(manifest, dict):
            raise RuntimeError("hybrid restart manifest must be a JSON object")
        if manifest.get("schema_version") != self.HYBRID_RESTART_SCHEMA_VERSION:
            raise RuntimeError(
                "hybrid restart requires schema 3; displacement schemas 1/2 are unsupported"
            )
        expected_keys = {
            "schema_version",
            "status",
            "checkpoint_kind",
            "committed_at_utc",
            "generation",
            "slot",
            "config_fingerprint",
            "implementation_fingerprint",
            "runtime_fingerprint",
            "mpi_ranks",
            "parent_checkpoint",
            "accepted_state",
            "scheduler_state",
            "history",
            "interface_history",
            "attempt_history",
            "boundary_field_fingerprint",
            "material_field_fingerprints",
            "partition_fingerprints",
            "path_partition_fingerprints",
            "shards",
            "field_state",
            "compatibility",
        }
        if set(manifest) != expected_keys:
            missing = sorted(expected_keys.difference(manifest))
            unknown = sorted(set(manifest).difference(expected_keys))
            raise RuntimeError(
                f"hybrid restart manifest key set is invalid: missing={missing}, unknown={unknown}"
            )
        if manifest["status"] != "partial":
            raise RuntimeError("hybrid restart is not a partial accepted-state checkpoint")
        if manifest["checkpoint_kind"] != "hybrid_displacement_fracture_energy":
            raise RuntimeError("hybrid restart checkpoint kind is invalid")
        if not self.config.path_control.enabled:
            raise RuntimeError("hybrid restart requires path_control.enabled=true")
        if manifest["config_fingerprint"] != self._restart_configuration_fingerprint():
            raise RuntimeError("hybrid restart configuration does not exactly match")
        if manifest["implementation_fingerprint"] != (self._restart_implementation_fingerprint()):
            raise RuntimeError("hybrid restart implementation fingerprint does not match")
        if manifest["runtime_fingerprint"] != self._restart_runtime_fingerprint():
            raise RuntimeError("hybrid restart runtime fingerprint does not match")
        if manifest["mpi_ranks"] != self.comm.size:
            raise RuntimeError("hybrid restart requires the same MPI rank count as the checkpoint")
        boundary_fingerprint = manifest["boundary_field_fingerprint"]
        if (
            not isinstance(boundary_fingerprint, str)
            or len(boundary_fingerprint) != 64
            or boundary_fingerprint != self._hybrid_boundary_field_fingerprint()
        ):
            raise RuntimeError("hybrid restart boundary-field/graph identity does not match")
        material_fingerprints = manifest["material_field_fingerprints"]
        if (
            not isinstance(material_fingerprints, list)
            or len(material_fingerprints) != self.comm.size
            or not all(
                isinstance(value, str) and len(value) == 64 for value in material_fingerprints
            )
        ):
            raise RuntimeError("hybrid restart material-field fingerprints are invalid")

        generation = manifest["generation"]
        slot = manifest["slot"]
        if type(generation) is not int or generation < 0:
            raise RuntimeError("hybrid restart generation is invalid")
        if type(slot) is not int or slot not in {0, 1} or slot != generation % 2:
            raise RuntimeError("hybrid restart checkpoint slot is invalid")
        parent = manifest["parent_checkpoint"]
        if generation == 0:
            if parent is not None:
                raise RuntimeError("initial hybrid restart checkpoint must not have a parent")
        else:
            if not isinstance(parent, dict) or set(parent) != {
                "generation",
                "manifest_sha256",
                "archived_manifest",
                "shards",
            }:
                raise RuntimeError("hybrid restart parent checkpoint metadata is invalid")
            parent_generation = parent["generation"]
            archive_name = parent["archived_manifest"]
            parent_sha256 = parent["manifest_sha256"]
            if (
                type(parent_generation) is not int
                or parent_generation != generation - 1
                or not isinstance(archive_name, str)
                or archive_name != f"manifests/checkpoint_generation_{parent_generation:06d}.json"
                or Path(archive_name).is_absolute()
                or ".." in Path(archive_name).parts
                or not isinstance(parent_sha256, str)
                or len(parent_sha256) != 64
                or any(character not in "0123456789abcdef" for character in parent_sha256)
                or not isinstance(parent["shards"], list)
            ):
                raise RuntimeError("hybrid restart parent checkpoint metadata is invalid")

        shards = manifest["shards"]
        partitions = manifest["partition_fingerprints"]
        if not isinstance(shards, list) or len(shards) != self.comm.size:
            raise RuntimeError("hybrid restart shard count disagrees with MPI size")
        if not isinstance(partitions, list) or len(partitions) != self.comm.size:
            raise RuntimeError("hybrid restart partition count disagrees with MPI size")
        if not isinstance(manifest["interface_history"], list):
            raise RuntimeError("hybrid restart interface history is invalid")
        if not isinstance(manifest["attempt_history"], list):
            raise RuntimeError("hybrid restart attempt history is invalid")
        expected_compatibility = {
            "requires_same_config": True,
            "requires_same_implementation": True,
            "requires_same_runtime": True,
            "requires_same_mpi_size_and_partition": True,
            "resume_wiring": "hybrid runner schema-3 resume is enabled",
        }
        if manifest["compatibility"] != expected_compatibility:
            raise RuntimeError("hybrid restart compatibility policy is invalid")

        try:
            scheduler_state = HybridSchedulerState.from_payload(manifest["scheduler_state"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"hybrid scheduler payload is invalid: {exc}") from exc
        reference = float(self.config.path_control.switch_displacement)
        if scheduler_state.reference_displacement != reference:
            raise RuntimeError(
                "hybrid scheduler reference displacement disagrees with the configured switch"
            )
        if scheduler_state.state.phase is ControlPhase.FRACTURE_ENERGY:
            expected_field_state = {
                "functions": list(self._RESTART_FUNCTION_NAMES),
                "includes_load_constant": True,
                "path_control_arrays": list(self._HYBRID_PATH_RESTART_ARRAY_NAMES),
                "real_element_global_owners": 1,
                "path_state_storage": (
                    "schema-3 authenticated path arrays; physical u/load remain the shared "
                    "source for reconstruction checks"
                ),
            }
            path_partitions = manifest["path_partition_fingerprints"]
            if (
                not isinstance(path_partitions, list)
                or len(path_partitions) != self.comm.size
                or not all(isinstance(value, str) for value in path_partitions)
            ):
                raise RuntimeError("hybrid restart path partition fingerprints are invalid")
        else:
            expected_field_state = {
                "functions": list(self._RESTART_FUNCTION_NAMES),
                "includes_load_constant": True,
                "path_control_arrays": [],
                "real_element_global_owners": 0,
                "path_state_storage": "not initialised in displacement phase",
            }
            if manifest["path_partition_fingerprints"] is not None:
                raise RuntimeError(
                    "hybrid displacement checkpoint must not contain path partition metadata"
                )
        if manifest["field_state"] != expected_field_state:
            raise RuntimeError("hybrid restart field-state codec is invalid")

        history = manifest["history"]
        accepted = manifest["accepted_state"]
        if not isinstance(history, list) or not history:
            raise RuntimeError("hybrid restart contains no accepted mechanical history")
        self._validate_hybrid_history_schema(history)
        if not isinstance(accepted, dict) or set(accepted) != {
            "accepted_step",
            "displacement",
            "control_phase",
            "phase_step",
            "control_value",
        }:
            raise RuntimeError("hybrid restart accepted-state payload is invalid")
        accepted_step = accepted["accepted_step"]
        accepted_displacement = accepted["displacement"]
        accepted_phase = accepted["control_phase"]
        accepted_phase_step = accepted["phase_step"]
        accepted_control = accepted["control_value"]
        if (
            type(accepted_step) is not int
            or accepted_step != len(history) - 1
            or type(accepted_phase) is not str
            or type(accepted_phase_step) is not int
            or type(accepted_displacement) not in {int, float}
            or type(accepted_control) not in {int, float}
        ):
            raise RuntimeError("hybrid restart accepted state has invalid scalar types")
        accepted_displacement = float(accepted_displacement)
        accepted_control = float(accepted_control)
        if not math.isfinite(accepted_displacement) or not math.isfinite(accepted_control):
            raise RuntimeError("hybrid restart accepted state is not finite")

        for index, record in enumerate(history):
            if not isinstance(record, dict):
                raise RuntimeError(f"hybrid restart history record {index} is invalid")
            try:
                record_step = record["step"]
                displacement = record["displacement"]
                fracture_energy = record["fracture_energy"]
                converged = record["stagger_converged"]
                control_phase = record["control_phase"]
                phase_step = record["phase_step"]
            except KeyError as exc:
                raise RuntimeError(
                    f"hybrid restart history record {index} is missing control fields"
                ) from exc
            if (
                type(record_step) is not int
                or record_step != index
                or type(displacement) not in {int, float}
                or not math.isfinite(float(displacement))
                or type(fracture_energy) not in {int, float}
                or not math.isfinite(float(fracture_energy))
                or converged is not True
                or type(control_phase) is not str
                or type(phase_step) is not int
                or phase_step < 0
            ):
                raise RuntimeError(f"hybrid restart history record {index} is inconsistent")
        if (
            int(history[-1]["step"]) != accepted_step
            or float(history[-1]["displacement"]) != accepted_displacement
        ):
            raise RuntimeError("hybrid restart accepted state disagrees with final history")
        interface_history = manifest["interface_history"]
        self._validate_hybrid_interface_history(history, interface_history)

        switch_index = round(
            reference / (self.config.loading.maximum_displacement / self.config.loading.steps)
        )
        phase = scheduler_state.state.phase
        if phase is ControlPhase.DISPLACEMENT:
            if accepted_phase != ControlPhase.DISPLACEMENT.value:
                raise RuntimeError("hybrid accepted phase disagrees with scheduler phase")
            if accepted_phase_step != scheduler_state.phase_step:
                raise RuntimeError("hybrid displacement phase_step disagrees with scheduler")
            if scheduler_state.phase_step != accepted_step:
                raise RuntimeError("hybrid displacement phase_step must equal accepted_step")
            self._validate_hybrid_displacement_history(
                history,
                reference_displacement=reference,
                description="displacement-phase",
            )
            if accepted_displacement >= reference:
                raise RuntimeError("hybrid displacement checkpoint must precede the switch target")
            if not self._hybrid_close(
                accepted_control,
                accepted_displacement,
                scale=reference,
            ):
                raise RuntimeError("hybrid displacement accepted control is inconsistent")
            pending = scheduler_state.pending_displacements
            if not pending:
                raise RuntimeError("hybrid displacement phase must retain a pending switch target")
            previous = accepted_displacement
            for index, target in enumerate(pending):
                interval = target.displacement - previous
                ratio = target.displacement / (
                    self.config.loading.maximum_displacement / self.config.loading.steps
                )
                expected_scheduled_step = math.ceil(ratio - 1.0e-12 * max(1.0, abs(ratio)))
                if (
                    target.displacement <= previous
                    or target.displacement > reference + 1.0e-12 * max(1.0, abs(reference))
                    or target.subdivision_level > self.config.loading.maximum_subdivisions
                    or not self._hybrid_dyadic_interval_valid(
                        previous,
                        target.displacement,
                        origin=0.0,
                        nominal_increment=(
                            self.config.loading.maximum_displacement / self.config.loading.steps
                        ),
                        subdivision_level=target.subdivision_level,
                        adaptive=self.config.loading.adaptive,
                        maximum_subdivisions=(self.config.loading.maximum_subdivisions),
                    )
                    or target.scheduled_step != expected_scheduled_step
                    or expected_scheduled_step < 1
                    or expected_scheduled_step > switch_index
                ):
                    raise RuntimeError(
                        f"hybrid pending displacement target {index} violates controls"
                    )
                previous = target.displacement
            if not self._hybrid_close(
                pending[-1].displacement,
                reference,
                scale=reference,
            ):
                raise RuntimeError("hybrid pending displacement queue does not end at the switch")
        elif phase is ControlPhase.FRACTURE_ENERGY:
            if accepted_phase != ControlPhase.FRACTURE_ENERGY.value:
                raise RuntimeError("hybrid accepted phase disagrees with scheduler phase")
            switch_step = scheduler_state.switch_accepted_step
            if switch_step is None or switch_step < 1 or switch_step >= len(history):
                raise RuntimeError("hybrid scheduler switch_accepted_step is invalid")
            if scheduler_state.phase_step != accepted_step - switch_step:
                raise RuntimeError("hybrid energy phase_step is not contiguous with accepted_step")
            if accepted_phase_step != scheduler_state.phase_step:
                raise RuntimeError("hybrid accepted energy phase_step disagrees with scheduler")
            self._validate_hybrid_displacement_history(
                history[: switch_step + 1],
                reference_displacement=reference,
                description="pre-switch",
            )
            if not self._hybrid_close(
                float(history[switch_step]["displacement"]),
                reference,
                scale=reference,
            ):
                raise RuntimeError("hybrid history switch displacement is inconsistent")
            switch_energy = float(history[switch_step]["fracture_energy"])
            previous_switch_energy = float(history[switch_step - 1]["fracture_energy"])
            observed_reference_increment = switch_energy - previous_switch_energy
            reference_increment = scheduler_state.state.reference_increment
            if reference_increment is None or not self._hybrid_energy_close(
                reference_increment,
                observed_reference_increment,
                increment=observed_reference_increment,
            ):
                raise RuntimeError(
                    "hybrid control-state reference increment disagrees with switch history"
                )

            lower = self.config.path_control.load_lower_bound
            upper = self.config.path_control.load_upper_bound
            previous_target = switch_energy
            previous_value = switch_energy
            path_records = history[switch_step + 1 :]
            for phase_step, record in enumerate(path_records, start=1):
                try:
                    target = float(record["control_target"])
                    value = float(record["control_value"])
                    path_coordinate = float(record["path_coordinate"])
                    path_increment = float(record["path_increment"])
                    displacement = float(record["displacement"])
                    load_increment = float(record["load_increment"])
                    load_factor = float(record["load_factor"])
                    record_reference = float(record["reference_displacement"])
                    control_relative = float(record["control_residual_relative"])
                    damage_relative = float(record["damage_kkt_relative"])
                    mechanical_relative = float(record["mechanical_residual_relative"])
                    stagger_error = float(record["stagger_error"])
                    if target <= previous_value:
                        raise RuntimeError(
                            "hybrid energy control target must exceed the previous accepted value"
                        )
                    control_certificate = self._hybrid_control_residual_certificate(
                        accepted_value=previous_value,
                        target_value=target,
                        achieved_value=value,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"hybrid energy history record {phase_step} is invalid"
                    ) from exc
                delta = target - previous_target
                level = record["subdivision_level"]
                if level > self.config.path_control.maximum_subdivisions:
                    raise RuntimeError("hybrid energy history exceeds subdivision controls")
                expected_delta = self.config.path_control.target_increment / (2**level)
                ratio = (target - switch_energy) / (self.config.path_control.target_increment)
                nominal_index = math.ceil(ratio - 1.0e-12 * max(1.0, abs(ratio)))
                previous_displacement = float(history[switch_step + phase_step - 1]["displacement"])
                if (
                    record["control_phase"] != ControlPhase.FRACTURE_ENERGY.value
                    or record["phase_step"] != phase_step
                    or not all(
                        math.isfinite(number)
                        for number in (
                            target,
                            value,
                            path_coordinate,
                            path_increment,
                            displacement,
                            load_increment,
                            load_factor,
                            record_reference,
                            control_relative,
                            control_certificate.absolute,
                            control_certificate.limit,
                            control_certificate.ratio,
                            damage_relative,
                            mechanical_relative,
                        )
                    )
                    or target <= previous_target
                    or not self._hybrid_control_increment_meets_minimum(
                        delta,
                        minimum_increment=(self.config.path_control.minimum_increment),
                        coordinate_scale=max(
                            abs(previous_target),
                            abs(target),
                        ),
                    )
                    or level > self.config.path_control.maximum_subdivisions
                    or not self._hybrid_control_coordinate_close(
                        path_increment,
                        delta,
                        coordinate_scale=max(
                            abs(previous_target),
                            abs(target),
                        ),
                    )
                    or not self._hybrid_control_coordinate_close(
                        delta,
                        expected_delta,
                        coordinate_scale=max(
                            abs(previous_target),
                            abs(target),
                        ),
                    )
                    or not self._hybrid_dyadic_interval_valid(
                        previous_target,
                        target,
                        origin=switch_energy,
                        nominal_increment=(self.config.path_control.target_increment),
                        subdivision_level=level,
                        adaptive=self.config.path_control.adaptive,
                        maximum_subdivisions=(self.config.path_control.maximum_subdivisions),
                    )
                    or not self._hybrid_control_coordinate_close(
                        path_coordinate,
                        target,
                    )
                    or not self._hybrid_energy_close(
                        float(record["fracture_energy"]),
                        value,
                        increment=delta,
                    )
                    or not control_certificate.certified
                    or control_relative < 0.0
                    or not self._hybrid_close(
                        control_relative,
                        control_certificate.relative,
                        scale=max(control_relative, control_certificate.relative),
                    )
                    or not 0.0 <= mechanical_relative <= self.config.path_control.residual_tolerance
                    or not 0.0 <= damage_relative <= self.config.loading.damage_kkt_tolerance
                    or not self._hybrid_close(
                        stagger_error,
                        max(
                            control_relative,
                            damage_relative,
                            mechanical_relative,
                        ),
                        scale=stagger_error,
                    )
                    or record_reference != reference
                    or not self._hybrid_close(
                        load_factor * reference,
                        displacement,
                        scale=reference,
                    )
                    or not self._hybrid_close(
                        load_increment,
                        displacement - previous_displacement,
                        scale=reference,
                    )
                    or displacement < lower - 1.0e-12 * max(1.0, abs(lower))
                    or displacement > upper + 1.0e-12 * max(1.0, abs(upper))
                    or type(record["path_snes_iterations"]) is not int
                    or record["path_snes_iterations"] < 0
                    or type(record["path_snes_reason"]) is not int
                    or record["path_snes_reason"] <= 0
                    or type(record["path_ksp_reason"]) is not int
                    or record["path_ksp_reason"] <= 0
                    or record["damage_snes_iterations"] != record["path_snes_iterations"]
                    or record["damage_snes_reason"] != record["path_snes_reason"]
                    or record["elastic_ksp_reason"] != record["path_ksp_reason"]
                    or record["stagger_iterations"] != record["path_snes_iterations"]
                    or record["aitken_accepted_iterations"] != 0
                    or record["final_aitken_relaxation"] is not None
                    or nominal_index < 1
                    or nominal_index > self.config.path_control.steps
                    or record["scheduled_step"] != switch_index + nominal_index
                ):
                    raise RuntimeError(
                        "hybrid energy control targets must be strictly increasing with "
                        "contiguous phase_step values"
                    )
                previous_target = target
                previous_value = value
            queue = scheduler_state.fracture_energy_queue
            if queue is None:
                raise RuntimeError("hybrid energy phase has no fracture-energy queue")
            expected_accepted = (
                float(path_records[-1]["control_target"]) if path_records else switch_energy
            )
            last_increment = (
                float(path_records[-1]["control_target"])
                - (
                    float(path_records[-2]["control_target"])
                    if len(path_records) > 1
                    else switch_energy
                )
                if path_records
                else self.config.path_control.target_increment
            )
            if not self._hybrid_control_coordinate_close(
                queue.accepted_value,
                expected_accepted,
            ):
                raise RuntimeError(
                    "hybrid fracture-energy queue accepted value disagrees with accepted control"
                )
            if not self._hybrid_control_coordinate_close(
                accepted_control,
                queue.accepted_value,
            ):
                raise RuntimeError("hybrid accepted control disagrees with energy queue")
            measured = float(path_records[-1]["control_value"]) if path_records else switch_energy
            measured_energy = float(history[-1]["fracture_energy"])
            measured_is_certified = self._hybrid_control_coordinate_close(
                measured,
                queue.accepted_value,
            )
            if path_records:
                previous_measured = (
                    float(path_records[-2]["control_value"])
                    if len(path_records) > 1
                    else switch_energy
                )
                measured_is_certified = self._hybrid_control_residual_certificate(
                    accepted_value=previous_measured,
                    target_value=queue.accepted_value,
                    achieved_value=measured,
                ).certified
            if not measured_is_certified or not self._hybrid_energy_close(
                measured_energy,
                measured,
                increment=last_increment,
            ):
                raise RuntimeError(
                    "hybrid accepted fracture energy disagrees with queue accepted value"
                )
            previous = queue.accepted_value
            for index, target in enumerate(queue.pending):
                interval = target.value - previous
                if target.subdivision_level > self.config.path_control.maximum_subdivisions:
                    raise RuntimeError(
                        f"hybrid fracture-energy queue target {index} exceeds controls"
                    )
                expected_interval = self.config.path_control.target_increment / (
                    2**target.subdivision_level
                )
                ratio = (target.value - switch_energy) / (self.config.path_control.target_increment)
                nominal_index = math.ceil(ratio - 1.0e-12 * max(1.0, abs(ratio)))
                if (
                    target.subdivision_level > self.config.path_control.maximum_subdivisions
                    or not self._hybrid_control_increment_meets_minimum(
                        interval,
                        minimum_increment=(self.config.path_control.minimum_increment),
                        coordinate_scale=max(abs(previous), abs(target.value)),
                    )
                    or not self._hybrid_control_coordinate_close(
                        interval,
                        expected_interval,
                        coordinate_scale=max(abs(previous), abs(target.value)),
                    )
                    or not self._hybrid_dyadic_interval_valid(
                        previous,
                        target.value,
                        origin=switch_energy,
                        nominal_increment=(self.config.path_control.target_increment),
                        subdivision_level=target.subdivision_level,
                        adaptive=self.config.path_control.adaptive,
                        maximum_subdivisions=(self.config.path_control.maximum_subdivisions),
                    )
                    or nominal_index < 1
                    or nominal_index > self.config.path_control.steps
                ):
                    raise RuntimeError(
                        f"hybrid fracture-energy queue target {index} violates controls"
                    )
                previous = target.value
            terminal = queue.pending[-1].value if queue.pending else queue.accepted_value
            expected_terminal = (
                switch_energy
                + self.config.path_control.steps * self.config.path_control.target_increment
            )
            if not self._hybrid_control_coordinate_close(
                terminal,
                expected_terminal,
            ):
                raise RuntimeError(
                    "hybrid fracture-energy queue does not end at the configured target"
                )
        else:  # pragma: no cover - ControlState closes this enum.
            raise RuntimeError("hybrid scheduler phase is unsupported")
        self._validate_hybrid_attempt_history(
            history,
            manifest["attempt_history"],
            scheduler_state,
        )
        return scheduler_state

    def _commit_restart_checkpoint(
        self,
        output: Path,
        *,
        schema_version: int,
        validator: Any,
        payload: dict[str, Any],
        state_arrays: dict[str, np.ndarray] | None = None,
        path_partition_fingerprint: str | None = None,
        boundary_field_fingerprint: str | None = None,
        material_field_fingerprint: str | None = None,
    ) -> None:
        """Commit one schema-specific payload through the shared two-slot journal."""
        restart_directory = output / "restart"
        if self.comm.rank == 0:
            try:
                restart_directory.mkdir(parents=False, exist_ok=True)
                manifest_path = restart_directory / "checkpoint.json"
                previous_bytes = manifest_path.read_bytes() if manifest_path.is_file() else None
                if previous_bytes is not None:
                    previous_manifest = json.loads(previous_bytes)
                    validator(previous_manifest)
                    if previous_manifest.get("generation") != self._restart_generation:
                        raise RuntimeError(
                            "previous restart manifest generation changed during this run"
                        )
                elif self._restart_generation != -1:
                    raise RuntimeError("previous restart manifest is missing during this run")
                directory_error = None
            except (
                ArithmeticError,
                OSError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                previous_bytes = None
                directory_error = f"{type(exc).__name__}: {exc}"
        else:
            previous_bytes = None
            directory_error = None
        directory_error = self.comm.bcast(directory_error, root=0)
        if directory_error is not None:
            raise RuntimeError(f"restart directory preparation failed: {directory_error}")
        self.comm.barrier()

        generation = self._restart_generation + 1
        slot = generation % 2
        shard_name = f"state_slot_{slot}_rank_{self.comm.rank:06d}.npz"
        shard_path = restart_directory / shard_name
        temporary = shard_path.with_name(f".{shard_path.name}.tmp")
        local_error = None
        local_metadata: dict[str, Any] | None = None
        try:
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    **(self._restart_state_arrays() if state_arrays is None else state_arrays),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, shard_path)
            local_metadata = {
                "rank": self.comm.rank,
                "file": shard_name,
                "bytes": shard_path.stat().st_size,
                "sha256": self._file_sha256(shard_path),
            }
        except (OSError, ValueError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        write_errors = self.comm.allgather(local_error)
        if any(error is not None for error in write_errors):
            raise RuntimeError(
                "restart shard write failed: "
                + "; ".join(error for error in write_errors if error is not None)
            )
        shard_metadata = self.comm.gather(local_metadata, root=0)
        partition_fingerprints = self.comm.gather(self._restart_partition_fingerprint(), root=0)
        path_partition_fingerprints = (
            self.comm.gather(path_partition_fingerprint, root=0)
            if path_partition_fingerprint is not None
            else None
        )
        material_field_fingerprints = (
            self.comm.gather(material_field_fingerprint, root=0)
            if material_field_fingerprint is not None
            else None
        )

        if self.comm.rank == 0:
            commit_stage = "initialise_manifest_commit"
            try:
                commit_stage = "verify_previous_manifest"
                manifest_path = restart_directory / "checkpoint.json"
                parent_checkpoint = None
                if previous_bytes is not None:
                    if manifest_path.read_bytes() != previous_bytes:
                        raise RuntimeError(
                            "previous restart manifest changed during checkpoint commit"
                        )
                    previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
                    previous_manifest = json.loads(previous_bytes)
                    previous_generation = previous_manifest["generation"]
                    commit_stage = "archive_previous_manifest"
                    archive_directory = restart_directory / "manifests"
                    archive_directory.mkdir(exist_ok=True)
                    archive_name = f"checkpoint_generation_{previous_generation:06d}.json"
                    archive_path = archive_directory / archive_name
                    if archive_path.is_file():
                        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != (
                            previous_sha256
                        ):
                            raise RuntimeError(
                                "archived parent restart manifest conflicts with checkpoint"
                            )
                    else:
                        archive_temporary = archive_path.with_name(f".{archive_path.name}.tmp")
                        with archive_temporary.open("wb") as stream:
                            stream.write(previous_bytes)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(archive_temporary, archive_path)
                    parent_checkpoint = {
                        "generation": previous_generation,
                        "manifest_sha256": previous_sha256,
                        "archived_manifest": f"manifests/{archive_name}",
                        "shards": previous_manifest["shards"],
                    }
                commit_stage = "build_manifest"
                manifest = {
                    "schema_version": schema_version,
                    "status": "partial",
                    "committed_at_utc": datetime.now(UTC).isoformat(),
                    "generation": generation,
                    "slot": slot,
                    "config_fingerprint": self._restart_configuration_fingerprint(),
                    "implementation_fingerprint": (self._restart_implementation_fingerprint()),
                    "runtime_fingerprint": self._restart_runtime_fingerprint(),
                    "mpi_ranks": self.comm.size,
                    "parent_checkpoint": parent_checkpoint,
                    **payload,
                    "partition_fingerprints": partition_fingerprints,
                    **(
                        {
                            "path_partition_fingerprints": path_partition_fingerprints,
                            "boundary_field_fingerprint": boundary_field_fingerprint,
                            "material_field_fingerprints": material_field_fingerprints,
                        }
                        if schema_version == self.HYBRID_RESTART_SCHEMA_VERSION
                        else {}
                    ),
                    "shards": shard_metadata,
                }
                commit_stage = "validate_manifest"
                validator(manifest)
                commit_stage = "publish_manifest"
                self._atomic_write_json(manifest_path, manifest)
                manifest_error = None
            except (
                ArithmeticError,
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                manifest_error = (
                    f"stage={commit_stage}; {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
        else:
            manifest_error = None
        manifest_error = self.comm.bcast(manifest_error, root=0)
        if manifest_error is not None:
            raise RuntimeError(f"restart manifest commit failed: {manifest_error}")
        self.comm.barrier()
        self._restart_generation = generation

    def _load_restart_manifest(
        self,
        output: Path,
        *,
        validator: Any,
    ) -> tuple[dict[str, Any], Any]:
        restart_directory = output / "restart"
        manifest_path = restart_directory / "checkpoint.json"
        if self.comm.rank == 0:
            try:
                with manifest_path.open(encoding="utf-8") as stream:
                    manifest = json.load(stream)
                validated = validator(manifest)
                parent = manifest.get("parent_checkpoint")
                if parent is not None:
                    archive_path = restart_directory / parent["archived_manifest"]
                    if not archive_path.is_file():
                        raise RuntimeError(
                            "restart parent manifest archive is missing or corrupted"
                        )
                    archive_bytes = archive_path.read_bytes()
                    if hashlib.sha256(archive_bytes).hexdigest() != parent["manifest_sha256"]:
                        raise RuntimeError(
                            "restart parent manifest archive is missing or corrupted"
                        )
                    parent_manifest = json.loads(archive_bytes)
                    validator(parent_manifest)
                    if (
                        parent_manifest.get("generation") != parent["generation"]
                        or parent_manifest.get("shards") != parent["shards"]
                    ):
                        raise RuntimeError(
                            "restart parent manifest archive disagrees with lineage metadata"
                        )
                manifest_error = None
            except (
                ArithmeticError,
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                manifest = None
                validated = None
                manifest_error = f"{type(exc).__name__}: {exc}"
        else:
            manifest = None
            validated = None
            manifest_error = None
        manifest, validated, manifest_error = self.comm.bcast(
            (manifest, validated, manifest_error), root=0
        )
        if manifest_error is not None:
            raise RuntimeError(f"restart checkpoint validation failed: {manifest_error}")
        return manifest, validated

    def _restore_restart_shards(
        self,
        output: Path,
        manifest: dict[str, Any],
        *,
        extra_names: tuple[str, ...] = (),
    ) -> dict[str, np.ndarray]:
        restart_directory = output / "restart"
        local_error = None
        state: dict[str, np.ndarray] | None = None
        try:
            partition_fingerprint = self._restart_partition_fingerprint()
            expected_partition = manifest["partition_fingerprints"][self.comm.rank]
            if partition_fingerprint != expected_partition:
                raise RuntimeError("local mesh/dof partition differs from the committed checkpoint")
            material_fingerprint = self._hybrid_material_field_fingerprint()
            expected_material = manifest.get("material_field_fingerprints")
            if expected_material is not None and (
                material_fingerprint != expected_material[self.comm.rank]
            ):
                raise RuntimeError("local graph-derived material fields differ from the checkpoint")
            shard = manifest["shards"][self.comm.rank]
            if not isinstance(shard, dict):
                raise RuntimeError("restart shard metadata is invalid")
            if shard.get("rank") != self.comm.rank:
                raise RuntimeError("restart shard rank ordering is invalid")
            shard_name = shard.get("file")
            if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
                raise RuntimeError("restart shard filename is invalid")
            shard_path = restart_directory / shard_name
            if shard_path.stat().st_size != shard.get("bytes"):
                raise RuntimeError("restart shard size differs from the manifest")
            if self._file_sha256(shard_path) != shard.get("sha256"):
                raise RuntimeError("restart shard checksum differs from the manifest")
            with np.load(shard_path, allow_pickle=False) as archive:
                expected_names = {*self._RESTART_FUNCTION_NAMES, "load", *extra_names}
                if set(archive.files) != expected_names:
                    raise RuntimeError("restart shard field set is invalid")
                state = {name: np.array(archive[name], copy=True) for name in archive.files}
            for name in self._RESTART_FUNCTION_NAMES:
                expected_shape = getattr(self, name).x.array.shape
                if state[name].shape != expected_shape:
                    raise RuntimeError(
                        f"restart array {name!r} shape differs from the current partition"
                    )
                if np.iscomplexobj(state[name]) or not np.all(np.isfinite(state[name])):
                    raise RuntimeError(f"restart array {name!r} is not finite real data")
            tolerance = 1.0e-10
            upper_values = np.asarray(self.d_upper.x.array)
            if (
                np.min(state["d"].real, initial=0.0) < -tolerance
                or np.max(state["d"].real, initial=0.0) > 1.0 + tolerance
                or np.min(state["d_lower"].real, initial=0.0) < -tolerance
                or np.max(state["d_lower"].real, initial=0.0) > 1.0 + tolerance
                or np.iscomplexobj(upper_values)
                or not np.all(np.isfinite(upper_values))
                or np.min(upper_values.real, initial=0.0) < -tolerance
                or np.max(upper_values.real, initial=0.0) > 1.0 + tolerance
                or np.min((state["d"] - state["d_lower"]).real, initial=0.0) < -tolerance
                or np.min((upper_values - state["d"]).real, initial=0.0) < -tolerance
            ):
                raise RuntimeError("restart damage state violates bounds or irreversibility")
            for name in ("Gc0", "Gc", "diffusivity"):
                if np.min(state[name].real, initial=math.inf) <= 0.0:
                    raise RuntimeError(f"restart array {name!r} must be positive")
            if np.min(state["trap_density"].real, initial=0.0) < -tolerance:
                raise RuntimeError("restart trap density must be non-negative")
            expected_load_shape = np.asarray(self.load.value).shape
            if state["load"].shape != expected_load_shape:
                raise RuntimeError(
                    "restart load array shape differs from the current scalar Constant"
                )
            if np.iscomplexobj(state["load"]) or not np.all(np.isfinite(state["load"])):
                raise RuntimeError("restart load Constant is not finite real data")
            restored_load = float(np.asarray(state["load"]).reshape(-1)[0].real)
            accepted_load = float(manifest["accepted_state"]["displacement"])
            if restored_load != accepted_load:
                raise RuntimeError("restart load Constant disagrees with the accepted displacement")
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        load_errors = self.comm.allgather(local_error)
        if any(error is not None for error in load_errors):
            raise RuntimeError(
                "restart shard validation failed: "
                + "; ".join(error for error in load_errors if error is not None)
            )
        if state is None:
            raise RuntimeError("restart shard state is unavailable after validation")
        return state

    def _hybrid_path_constructor_local_error(
        self,
        reference_displacement: float,
    ) -> str | None:
        """Check every local constructor precondition before collective setup."""
        try:
            if type(reference_displacement) not in {int, float}:
                raise TypeError("reference displacement must be a real number")
            if (
                not math.isfinite(reference_displacement)
                or reference_displacement <= 0.0
                or reference_displacement != self.config.path_control.switch_displacement
            ):
                raise ValueError("reference displacement must be the exact configured switch")
            if (
                type(self.config.path_control.snes_max_iterations) is not int
                or self.config.path_control.snes_max_iterations < 1
            ):
                raise ValueError("path SNES iteration limit is invalid")
            for name in ("d", "d_lower", "d_upper"):
                if getattr(self, name).function_space is not self.V_d:
                    raise ValueError(f"{name} does not use V_d")
            if self.Gc.function_space.mesh is not self.domain:
                raise ValueError("Gc does not use the solver domain")
            gc_owned = (
                self.Gc.function_space.dofmap.index_map.size_local
                * self.Gc.function_space.dofmap.index_map_bs
            )
            local_gc = self.Gc.x.array[:gc_owned].real
            if not np.all(np.isfinite(local_gc)):
                raise ValueError("local Gc data is not finite")
            if np.min(local_gc, initial=math.inf) <= 0.0:
                raise ValueError("local Gc data is not positive")
            lower = self.config.path_control.load_lower_bound / reference_displacement
            upper = self.config.path_control.load_upper_bound / reference_displacement
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("path load-factor bounds are invalid")
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            return f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        return None

    def _new_hybrid_path_problem(self, reference_displacement: float) -> Any:
        from .path_control import FractureEnergyControlProblem

        preflight_error = self._hybrid_path_constructor_local_error(reference_displacement)
        preflight_errors = self.comm.allgather(preflight_error)
        if any(error is not None for error in preflight_errors):
            raise RuntimeError(
                "hybrid path constructor preflight failed: "
                + "; ".join(error for error in preflight_errors if error is not None)
            )

        return FractureEnergyControlProblem(
            domain=self.domain,
            config=self.config,
            V_u=self.V_u,
            V_d=self.V_d,
            d=self.d,
            d_lower=self.d_lower,
            d_upper=self.d_upper,
            Gc=self.Gc,
            reference_displacement=reference_displacement,
            homogeneous_bcs_z=self.path_homogeneous_bcs,
        )

    def _hybrid_path_partition_fingerprint(self, problem: Any) -> str:
        local_error = None
        alpha_owned = 0
        try:
            alpha_map = problem.V_alpha.dofmap.index_map
            alpha_owned = alpha_map.size_local * problem.V_alpha.dofmap.index_map_bs
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        setup_errors = self.comm.allgather(local_error)
        if any(error is not None for error in setup_errors):
            raise RuntimeError(
                "hybrid path partition setup failed: "
                + "; ".join(error for error in setup_errors if error is not None)
            )
        global_owners = int(self.comm.allreduce(1 if alpha_owned else 0, op=MPI.SUM))
        if global_owners != 1:
            raise RuntimeError("hybrid path Real element must have exactly one global owner")
        fingerprint = None
        local_error = None
        try:
            digest = hashlib.sha256()
            digest.update(self._restart_partition_fingerprint().encode())
            digest.update(repr(float(problem.reference_displacement)).encode())
            for name, function in (
                ("path_z", problem.z),
                ("path_alpha", problem.alpha),
            ):
                space = function.function_space
                index_map = space.dofmap.index_map
                metadata = (
                    name,
                    self.comm.rank,
                    self.comm.size,
                    index_map.size_local,
                    index_map.num_ghosts,
                    index_map.size_global,
                    space.dofmap.index_map_bs,
                    function.x.array.shape,
                    alpha_owned if name == "path_alpha" else None,
                    global_owners if name == "path_alpha" else None,
                )
                digest.update(repr(metadata).encode())
                if name == "path_z":
                    coordinates = np.ascontiguousarray(space.tabulate_dof_coordinates())
                    digest.update(str(coordinates.shape).encode())
                    digest.update(coordinates.tobytes())
            fingerprint = digest.hexdigest()
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        fingerprint_errors = self.comm.allgather(local_error)
        if any(error is not None for error in fingerprint_errors):
            raise RuntimeError(
                "hybrid path partition fingerprint failed: "
                + "; ".join(error for error in fingerprint_errors if error is not None)
            )
        if fingerprint is None:  # pragma: no cover - collective gate invariant.
            raise RuntimeError("hybrid path partition fingerprint is unavailable")
        return fingerprint

    def _hybrid_restart_state_arrays(
        self,
        scheduler_state: HybridSchedulerState,
    ) -> tuple[dict[str, np.ndarray], str | None]:
        state = None
        local_error = None
        try:
            state = self._restart_state_arrays()
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        common_errors = self.comm.allgather(local_error)
        if any(error is not None for error in common_errors):
            raise RuntimeError(
                "hybrid common restart state capture failed: "
                + "; ".join(error for error in common_errors if error is not None)
            )
        if state is None:  # pragma: no cover - collective gate invariant.
            raise RuntimeError("hybrid common restart state is unavailable")
        local_error = None
        try:
            if np.iscomplexobj(state["load"]) or not np.all(np.isfinite(state["load"])):
                raise RuntimeError("hybrid restart load is not finite real data")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        load_errors = self.comm.allgather(local_error)
        if any(error is not None for error in load_errors):
            raise RuntimeError(
                "hybrid load-state capture failed: "
                + "; ".join(error for error in load_errors if error is not None)
            )
        canonical_load = self.comm.bcast(
            np.array(state["load"], copy=True) if self.comm.rank == 0 else None,
            root=0,
        )
        load_mismatch = (
            None
            if np.array_equal(state["load"], canonical_load)
            else f"rank {self.comm.rank}: hybrid restart load differs across MPI ranks"
        )
        load_mismatches = self.comm.allgather(load_mismatch)
        if any(error is not None for error in load_mismatches):
            raise RuntimeError(
                "hybrid load-state capture failed: "
                + "; ".join(error for error in load_mismatches if error is not None)
            )
        if scheduler_state.state.phase is ControlPhase.DISPLACEMENT:
            return state, None
        problem = self._path_control_problem
        reference = scheduler_state.reference_displacement
        local_error = None
        if problem is None:
            local_error = (
                f"rank {self.comm.rank}: hybrid fracture-energy checkpoint requires "
                "an initialised path problem"
            )
        else:
            try:
                problem_reference = float(problem.reference_displacement)
                if problem_reference != reference:
                    local_error = (
                        f"rank {self.comm.rank}: hybrid path problem reference "
                        "displacement disagrees with scheduler"
                    )
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        problem_errors = self.comm.allgather(local_error)
        if any(error is not None for error in problem_errors):
            raise RuntimeError(
                "hybrid path problem preflight failed: "
                + "; ".join(error for error in problem_errors if error is not None)
            )
        if problem is None:  # pragma: no cover - collective gate invariant.
            raise RuntimeError("hybrid path problem is unavailable after preflight")
        path_partition = self._hybrid_path_partition_fingerprint(problem)

        load_factor = problem.load_factor
        bound_status = problem.load_factor_bound_status()
        lower_factor, upper_factor = problem.load_factor_bounds
        stored_load = float(np.asarray(state["load"]).reshape(-1)[0].real)
        actual_state_error = None
        if bound_status is not None or not lower_factor < load_factor < upper_factor:
            actual_state_error = (
                f"rank {self.comm.rank}: hybrid path load factor is not strictly inside its bounds"
            )
        elif not self._hybrid_close(
            load_factor * reference,
            stored_load,
            scale=reference,
        ):
            actual_state_error = (
                f"rank {self.comm.rank}: hybrid path alpha/reference product "
                "disagrees with the physical load"
            )
        actual_state_errors = self.comm.allgather(actual_state_error)
        if any(error is not None for error in actual_state_errors):
            raise RuntimeError(
                "hybrid path state preflight failed: "
                + "; ".join(error for error in actual_state_errors if error is not None)
            )

        local_error = None
        try:
            state.update(
                {
                    "path_z": np.array(problem.z.x.array, copy=True),
                    "path_alpha": np.array(problem.alpha.x.array, copy=True),
                    "path_target_fracture_energy": np.array(
                        problem._target_fracture_energy,
                        copy=True,
                    ),
                    "path_target_increment": np.array(
                        problem._target_increment,
                        copy=True,
                    ),
                    "path_target_energy_density": np.array(
                        problem._target_energy_density.value,
                        copy=True,
                    ),
                }
            )
            for name in self._HYBRID_PATH_RESTART_ARRAY_NAMES:
                if np.iscomplexobj(state[name]) or not np.all(np.isfinite(state[name])):
                    raise RuntimeError(f"hybrid restart array {name!r} is not finite real data")
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        path_array_errors = self.comm.allgather(local_error)
        if any(error is not None for error in path_array_errors):
            raise RuntimeError(
                "hybrid path restart state capture failed: "
                + "; ".join(error for error in path_array_errors if error is not None)
            )
        scalar_errors: list[str] = []
        for name in (
            "path_target_fracture_energy",
            "path_target_increment",
            "path_target_energy_density",
        ):
            canonical = self.comm.bcast(
                np.array(state[name], copy=True) if self.comm.rank == 0 else None,
                root=0,
            )
            if not np.array_equal(state[name], canonical):
                scalar_errors.append(f"hybrid restart scalar {name!r} differs across MPI ranks")
        scalar_errors_by_rank = self.comm.allgather("; ".join(scalar_errors) or None)
        if any(error is not None for error in scalar_errors_by_rank):
            raise RuntimeError(
                "hybrid path restart scalar capture failed: "
                + "; ".join(error for error in scalar_errors_by_rank if error is not None)
            )
        return state, path_partition

    def _validate_hybrid_save_state(
        self,
        state: dict[str, np.ndarray],
        *,
        history: list[dict[str, Any]],
        interface_history: list[dict[str, Any]],
        accepted_step: int,
        accepted_displacement: float,
        scheduler_state: HybridSchedulerState,
    ) -> None:
        """Run the shard/manifest cross-checks before publishing a generation."""
        local_errors: list[str] = []
        try:
            for name in self._RESTART_FUNCTION_NAMES:
                values = state[name]
                expected_shape = getattr(self, name).x.array.shape
                if values.shape != expected_shape:
                    local_errors.append(
                        f"restart array {name!r} has shape {values.shape}, "
                        f"expected {expected_shape}"
                    )
                elif np.iscomplexobj(values) or not np.all(np.isfinite(values)):
                    local_errors.append(f"restart array {name!r} is not finite real data")
            tolerance = 1.0e-10
            upper_values = np.asarray(self.d_upper.x.array)
            if (
                np.min(state["d"].real, initial=0.0) < -tolerance
                or np.max(state["d"].real, initial=0.0) > 1.0 + tolerance
                or np.min(state["d_lower"].real, initial=0.0) < -tolerance
                or np.max(state["d_lower"].real, initial=0.0) > 1.0 + tolerance
                or np.iscomplexobj(upper_values)
                or not np.all(np.isfinite(upper_values))
                or np.min(upper_values.real, initial=0.0) < -tolerance
                or np.max(upper_values.real, initial=0.0) > 1.0 + tolerance
                or np.min(
                    (state["d"] - state["d_lower"]).real,
                    initial=0.0,
                )
                < -tolerance
                or np.min(
                    (upper_values - state["d"]).real,
                    initial=0.0,
                )
                < -tolerance
            ):
                local_errors.append("restart damage state violates bounds or irreversibility")
            for name in ("Gc0", "Gc", "diffusivity"):
                if np.min(state[name].real, initial=math.inf) <= 0.0:
                    local_errors.append(f"restart array {name!r} must be positive")
            if np.min(state["trap_density"].real, initial=0.0) < -tolerance:
                local_errors.append("restart trap density must be non-negative")
            load_values = state["load"]
            if load_values.shape != np.asarray(self.load.value).shape:
                local_errors.append("restart load shape is invalid")
            elif np.iscomplexobj(load_values) or not np.all(np.isfinite(load_values)):
                local_errors.append("restart load is not finite real data")
            else:
                stored_load = float(load_values.reshape(-1)[0].real)
                if stored_load != float(accepted_displacement):
                    local_errors.append("restart load disagrees with accepted displacement")
            if (
                type(accepted_step) is not int
                or accepted_step != len(history) - 1
                or not history
                or float(history[-1]["displacement"]) != float(accepted_displacement)
            ):
                local_errors.append("restart accepted arguments disagree with final history")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            local_errors.append(f"rank {self.comm.rank}: {type(exc).__name__}: {exc}")
        common_errors = self.comm.allgather("; ".join(local_errors) or None)
        if any(error is not None for error in common_errors):
            raise RuntimeError(
                "hybrid common save-state preflight failed: "
                + "; ".join(error for error in common_errors if error is not None)
            )

        phase = scheduler_state.state.phase
        path_problem = self._path_control_problem
        trusted_control_increment = self.config.path_control.target_increment
        expected_internal_increment = 0.0
        queue = None
        local_error = None
        try:
            final_history_energy = float(history[-1]["fracture_energy"])
            if phase is ControlPhase.FRACTURE_ENERGY:
                switch_step = scheduler_state.switch_accepted_step
                queue = scheduler_state.fracture_energy_queue
                if switch_step is None or queue is None or path_problem is None:
                    raise RuntimeError(
                        "energy save-state requires a switch, queue, and path problem"
                    )
                path_records = history[switch_step + 1 :]
                trusted_control_increment = (
                    float(path_records[-1]["control_target"])
                    - (
                        float(path_records[-2]["control_target"])
                        if len(path_records) > 1
                        else float(history[switch_step]["fracture_energy"])
                    )
                    if path_records
                    else self.config.path_control.target_increment
                )
                expected_internal_increment = (
                    float(path_records[-1]["control_target"])
                    - (
                        float(path_records[-2]["control_value"])
                        if len(path_records) > 1
                        else float(history[switch_step]["fracture_energy"])
                    )
                    if path_records
                    else 0.0
                )
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        provenance_errors = self.comm.allgather(local_error)
        if any(error is not None for error in provenance_errors):
            raise RuntimeError(
                "hybrid save-state provenance preflight failed: "
                + "; ".join(error for error in provenance_errors if error is not None)
            )

        certificates = self._hybrid_terminal_certificates(
            path_problem if phase is ControlPhase.FRACTURE_ENERGY else None
        )
        recomputed_energy = certificates["fracture_energy"]
        certification_errors: list[str] = []
        if not self._hybrid_energy_close(
            recomputed_energy,
            final_history_energy,
            increment=trusted_control_increment,
        ):
            certification_errors.append("current fracture energy disagrees with final history")
        if phase is ControlPhase.FRACTURE_ENERGY:
            if path_problem is None or queue is None:  # pragma: no cover - stage gate.
                certification_errors.append("path save-state is unavailable")
            else:
                reference = scheduler_state.reference_displacement
                load_factor = path_problem.load_factor
                bound_status = path_problem.load_factor_bound_status()
                lower_factor, upper_factor = path_problem.load_factor_bounds
                physical_displacement = load_factor * reference
                local_u_error = float(
                    np.max(
                        np.abs(
                            state["u"]
                            - (state["path_z"] + physical_displacement * path_problem.lift.x.array)
                        ),
                        initial=0.0,
                    )
                )
                u_error = float(self.comm.allreduce(local_u_error, op=MPI.MAX))
                if bound_status is not None or not lower_factor < load_factor < upper_factor:
                    certification_errors.append(
                        "path load factor is not strictly inside its bounds"
                    )
                if not self._hybrid_close(
                    physical_displacement,
                    float(accepted_displacement),
                    scale=reference,
                ):
                    certification_errors.append(
                        "path alpha/reference product disagrees with accepted load"
                    )
                if u_error > 1.0e-12 * max(1.0, abs(physical_displacement)):
                    certification_errors.append(
                        "physical displacement disagrees with path z/alpha/lift"
                    )
                stored_target = float(state["path_target_fracture_energy"])
                stored_increment = float(state["path_target_increment"])
                stored_density = float(
                    np.asarray(state["path_target_energy_density"]).reshape(-1)[0]
                )
                if not self._hybrid_control_coordinate_close(
                    stored_target,
                    queue.accepted_value,
                ):
                    certification_errors.append("path target disagrees with queue accepted value")
                if not self._hybrid_control_coordinate_close(
                    stored_increment,
                    expected_internal_increment,
                    coordinate_scale=queue.accepted_value,
                ):
                    certification_errors.append(
                        "path internal increment disagrees with previous measured energy"
                    )
                if not self._hybrid_control_coordinate_close(
                    stored_density * path_problem._volume,
                    queue.accepted_value,
                ):
                    certification_errors.append(
                        "path target density disagrees with queue accepted value"
                    )
                energy_is_certified = self._hybrid_control_coordinate_close(
                    recomputed_energy,
                    queue.accepted_value,
                )
                if expected_internal_increment > 0.0:
                    energy_is_certified = self._hybrid_control_residual_certificate(
                        accepted_value=(queue.accepted_value - expected_internal_increment),
                        target_value=queue.accepted_value,
                        achieved_value=recomputed_energy,
                    ).certified
                if not energy_is_certified:
                    certification_errors.append(
                        "current fracture energy disagrees with queue accepted value"
                    )
        certification_errors.extend(
            self._hybrid_terminal_certificate_errors(
                certificates,
                history[-1],
                phase=phase,
                control_increment=trusted_control_increment,
            )
        )
        certification_errors.extend(
            self._hybrid_terminal_interface_errors(history, interface_history)
        )
        certification_errors_by_rank = self.comm.allgather("; ".join(certification_errors) or None)
        if any(error is not None for error in certification_errors_by_rank):
            raise RuntimeError(
                "hybrid save-state certification failed: "
                + "; ".join(error for error in certification_errors_by_rank if error is not None)
            )

    def _hybrid_terminal_certificates(
        self,
        path_problem: Any | None,
    ) -> dict[str, float | None]:
        """Reassemble independent certificates for the currently restored FE state."""
        elastic_energy = self._global_scalar(self.elastic_energy_form)
        fracture_energy = self._global_scalar(self.fracture_energy_form)
        crack_length = self._global_scalar(self.crack_length_form)
        reaction, traction_reaction = self._elastic_reactions()
        kkt_absolute, kkt_relative, kkt_scale = self._damage_kkt_metrics()
        minimum_increment = self._minimum_damage_increment()
        maximum_damage = self._max_damage()
        rightmost_damaged_x = self._tip_x(self.config.graph.crack_threshold)
        lower_violation, upper_violation = self._damage_bound_violations()
        if path_problem is None:
            mechanical_absolute, mechanical_relative, mechanical_scale = (
                self._mechanical_residual_metrics()
            )
            path_bc_violation = 0.0
        else:
            mechanical_absolute, mechanical_relative, mechanical_scale = (
                path_problem.mechanical_residual_metrics()
            )
            path_bc_violation = self._dirichlet_violation(
                path_problem.z,
                self.path_homogeneous_bcs,
            )
        physical_bc_violation = self._dirichlet_violation(self.u, self.bcs_u)
        return {
            "elastic_energy": elastic_energy,
            "fracture_energy": fracture_energy,
            "total_internal_energy": elastic_energy + fracture_energy,
            "regularised_crack_length": crack_length,
            "reaction_y": reaction,
            "traction_reaction_y": traction_reaction,
            "damage_kkt_inf": kkt_absolute,
            "damage_kkt_relative": kkt_relative,
            "damage_kkt_scale": kkt_scale,
            "minimum_damage_increment": minimum_increment,
            "maximum_damage": maximum_damage,
            "rightmost_damaged_x": rightmost_damaged_x,
            "damage_lower_bound_violation": lower_violation,
            "damage_upper_bound_violation": upper_violation,
            "mechanical_residual_inf": mechanical_absolute,
            "mechanical_residual_relative": mechanical_relative,
            "mechanical_residual_scale": mechanical_scale,
            "physical_bc_violation": physical_bc_violation,
            "path_bc_violation": path_bc_violation,
        }

    def _hybrid_terminal_certificate_errors(
        self,
        certificates: dict[str, float | None],
        record: dict[str, Any],
        *,
        phase: ControlPhase,
        control_increment: float,
    ) -> list[str]:
        errors: list[str] = []
        for name in (
            "elastic_energy",
            "fracture_energy",
            "total_internal_energy",
            "regularised_crack_length",
        ):
            if not self._hybrid_energy_close(
                float(certificates[name]),
                float(record[name]),
                increment=control_increment,
            ):
                errors.append(f"recomputed {name} disagrees with final history")
        for name in (
            "reaction_y",
            "traction_reaction_y",
            "damage_kkt_inf",
            "damage_kkt_relative",
            "damage_kkt_scale",
            "minimum_damage_increment",
            "maximum_damage",
        ):
            if not self._hybrid_close(
                float(certificates[name]),
                float(record[name]),
                scale=float(record[name]),
            ):
                errors.append(f"recomputed {name} disagrees with final history")
        recomputed_tip = certificates["rightmost_damaged_x"]
        recorded_tip = record["rightmost_damaged_x"]
        if (recomputed_tip is None) != (recorded_tip is None) or (
            recomputed_tip is not None
            and recorded_tip is not None
            and not self._hybrid_close(
                recomputed_tip,
                float(recorded_tip),
                scale=self.config.geometry.length,
            )
        ):
            errors.append("recomputed rightmost damage tip disagrees with final history")
        if (
            certificates["damage_kkt_relative"] > self.config.loading.damage_kkt_tolerance
            or certificates["damage_lower_bound_violation"] > 1.0e-10
            or certificates["damage_upper_bound_violation"] > 1.0e-10
            or certificates["minimum_damage_increment"] < -1.0e-10
        ):
            errors.append("restored damage state fails KKT/bound certification")
        mechanical_relative = certificates["mechanical_residual_relative"]
        if mechanical_relative > self.config.path_control.residual_tolerance:
            errors.append("restored state fails mechanical equilibrium certification")
        if (
            phase is ControlPhase.FRACTURE_ENERGY
            and record["control_phase"] == ControlPhase.FRACTURE_ENERGY.value
        ):
            recorded_mechanical = record["mechanical_residual_relative"]
            if recorded_mechanical is None or not self._hybrid_close(
                mechanical_relative,
                float(recorded_mechanical),
                scale=float(recorded_mechanical),
            ):
                errors.append("recomputed mechanical residual disagrees with final history")
        displacement_scale = max(1.0, abs(float(record["displacement"])))
        if certificates["physical_bc_violation"] > 1.0e-12 * displacement_scale:
            errors.append("restored physical displacement violates Dirichlet data")
        if certificates["path_bc_violation"] > 1.0e-12 * displacement_scale:
            errors.append("restored path correction violates homogeneous Dirichlet data")
        return errors

    def _hybrid_terminal_interface_errors(
        self,
        history: list[dict[str, Any]],
        interface_history: list[dict[str, Any]],
    ) -> list[str]:
        """Recompute the terminal damage-derived interface record collectively."""
        if not self._interface_tracking_enabled():
            return []
        final = history[-1]
        recomputed = None
        local_error = None
        try:
            recomputed = self._interface_history_record(
                accepted_step=int(final["step"]),
                scheduled_step=int(final["scheduled_step"]),
                displacement=float(final["displacement"]),
                subdivision_level=int(final["subdivision_level"]),
            )
        except Exception as exc:  # noqa: BLE001 - gate root post-gather failures.
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        stage_errors = self.comm.allgather(local_error)
        if any(error is not None for error in stage_errors):
            return [
                "terminal interface recomputation failed: "
                + "; ".join(error for error in stage_errors if error is not None)
            ]

        if self.comm.rank == 0:
            if not interface_history:
                comparison_error = "terminal interface history record is missing"
            elif recomputed is None:
                comparison_error = "terminal interface recomputation returned no record"
            elif recomputed != interface_history[-1]:
                comparison_error = (
                    "terminal interface history disagrees with the restored damage field"
                )
            else:
                comparison_error = None
        else:
            comparison_error = None
        comparison_error = self.comm.bcast(comparison_error, root=0)
        return [comparison_error] if comparison_error is not None else []

    def _save_hybrid_restart_checkpoint(
        self,
        output: Path,
        *,
        history: list[dict],
        interface_history: list[dict[str, Any]],
        attempt_history: list[dict[str, Any]],
        accepted_step: int,
        accepted_displacement: float,
        scheduler_state: HybridSchedulerState,
    ) -> None:
        """Commit one schema-3 hybrid state without changing schema-2 semantics."""
        scheduler_payload = None
        local_error = None
        try:
            if type(scheduler_state) is not HybridSchedulerState:
                raise TypeError("scheduler_state must be a HybridSchedulerState")
            scheduler_payload = scheduler_state.to_payload()
        except (TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        scheduler_errors = self.comm.allgather(local_error)
        if any(error is not None for error in scheduler_errors):
            raise RuntimeError(
                "hybrid scheduler save preflight failed: "
                + "; ".join(error for error in scheduler_errors if error is not None)
            )
        canonical_scheduler_payload = self.comm.bcast(
            scheduler_payload if self.comm.rank == 0 else None,
            root=0,
        )
        scheduler_mismatch = (
            None
            if scheduler_payload == canonical_scheduler_payload
            else f"rank {self.comm.rank}: hybrid scheduler state differs across MPI ranks"
        )
        scheduler_mismatches = self.comm.allgather(scheduler_mismatch)
        if any(error is not None for error in scheduler_mismatches):
            raise RuntimeError(
                "hybrid scheduler save preflight failed: "
                + "; ".join(error for error in scheduler_mismatches if error is not None)
            )
        metadata_encoding = None
        local_error = None
        try:
            metadata_encoding = json.dumps(
                {
                    "history": history,
                    "attempt_history": attempt_history,
                    "accepted_step": accepted_step,
                    "accepted_displacement": accepted_displacement,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        metadata_errors = self.comm.allgather(local_error)
        if any(error is not None for error in metadata_errors):
            raise RuntimeError(
                "hybrid checkpoint metadata save preflight failed: "
                + "; ".join(error for error in metadata_errors if error is not None)
            )
        canonical_metadata = self.comm.bcast(
            metadata_encoding if self.comm.rank == 0 else None,
            root=0,
        )
        metadata_mismatch = (
            None
            if metadata_encoding == canonical_metadata
            else f"rank {self.comm.rank}: checkpoint metadata differs across MPI ranks"
        )
        metadata_mismatches = self.comm.allgather(metadata_mismatch)
        if any(error is not None for error in metadata_mismatches):
            raise RuntimeError(
                "hybrid checkpoint metadata save preflight failed: "
                + "; ".join(error for error in metadata_mismatches if error is not None)
            )
        boundary_field_fingerprint = None
        material_field_fingerprint = None
        local_error = None
        try:
            boundary_field_fingerprint = self._hybrid_boundary_field_fingerprint()
            material_field_fingerprint = self._hybrid_material_field_fingerprint()
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        field_identity_errors = self.comm.allgather(local_error)
        if any(error is not None for error in field_identity_errors):
            raise RuntimeError(
                "hybrid graph/material identity preflight failed: "
                + "; ".join(error for error in field_identity_errors if error is not None)
            )
        boundary_fingerprints = self.comm.allgather(boundary_field_fingerprint)
        if len(set(boundary_fingerprints)) != 1:
            raise RuntimeError("hybrid boundary-field identity differs across MPI ranks")
        state_arrays, path_partition = self._hybrid_restart_state_arrays(scheduler_state)
        self._validate_hybrid_save_state(
            state_arrays,
            history=history,
            interface_history=interface_history,
            accepted_step=accepted_step,
            accepted_displacement=accepted_displacement,
            scheduler_state=scheduler_state,
        )
        phase = scheduler_state.state.phase
        if phase is ControlPhase.FRACTURE_ENERGY:
            queue = scheduler_state.fracture_energy_queue
            if queue is None:  # pragma: no cover - dataclass invariant.
                raise RuntimeError("hybrid fracture-energy scheduler has no queue")
            accepted_control = queue.accepted_value
            field_state = {
                "functions": list(self._RESTART_FUNCTION_NAMES),
                "includes_load_constant": True,
                "path_control_arrays": list(self._HYBRID_PATH_RESTART_ARRAY_NAMES),
                "real_element_global_owners": 1,
                "path_state_storage": (
                    "schema-3 authenticated path arrays; physical u/load remain the shared "
                    "source for reconstruction checks"
                ),
            }
        else:
            accepted_control = float(accepted_displacement)
            field_state = {
                "functions": list(self._RESTART_FUNCTION_NAMES),
                "includes_load_constant": True,
                "path_control_arrays": [],
                "real_element_global_owners": 0,
                "path_state_storage": "not initialised in displacement phase",
            }
        payload = {
            "checkpoint_kind": "hybrid_displacement_fracture_energy",
            "accepted_state": {
                "accepted_step": int(accepted_step),
                "displacement": float(accepted_displacement),
                "control_phase": phase.value,
                "phase_step": int(scheduler_state.phase_step),
                "control_value": float(accepted_control),
            },
            "scheduler_state": scheduler_payload,
            "history": history,
            "interface_history": interface_history,
            "attempt_history": attempt_history,
            "field_state": field_state,
            "compatibility": {
                "requires_same_config": True,
                "requires_same_implementation": True,
                "requires_same_runtime": True,
                "requires_same_mpi_size_and_partition": True,
                "resume_wiring": "hybrid runner schema-3 resume is enabled",
            },
        }
        self._commit_restart_checkpoint(
            output,
            schema_version=self.HYBRID_RESTART_SCHEMA_VERSION,
            validator=self._validate_hybrid_restart_manifest,
            payload=payload,
            state_arrays=state_arrays,
            path_partition_fingerprint=path_partition,
            boundary_field_fingerprint=boundary_field_fingerprint,
            material_field_fingerprint=material_field_fingerprint,
        )

    def _load_hybrid_restart_checkpoint(
        self,
        output: Path,
    ) -> tuple[
        list[dict],
        list[dict[str, Any]],
        list[dict[str, Any]],
        HybridSchedulerState,
        int,
        float,
    ]:
        """Authenticate and restore schema 3; runner resume remains intentionally unwired."""
        freshness_error = (
            f"rank {self.comm.rank}: hybrid restart requires a fresh solver instance"
            if self._path_control_problem is not None
            else None
        )
        freshness_errors = self.comm.allgather(freshness_error)
        if any(error is not None for error in freshness_errors):
            raise RuntimeError(
                "hybrid restart must be loaded into a fresh solver instance: "
                + "; ".join(error for error in freshness_errors if error is not None)
            )
        manifest, scheduler_state = self._load_restart_manifest(
            output,
            validator=self._validate_hybrid_restart_manifest,
        )
        phase = scheduler_state.state.phase
        extra_names = (
            self._HYBRID_PATH_RESTART_ARRAY_NAMES if phase is ControlPhase.FRACTURE_ENERGY else ()
        )
        state = self._restore_restart_shards(
            output,
            manifest,
            extra_names=extra_names,
        )

        path_problem = None
        if phase is ControlPhase.FRACTURE_ENERGY:
            local_error = None
            try:
                reference = scheduler_state.reference_displacement
                path_problem = self._new_hybrid_path_problem(reference)
            except (TypeError, ValueError, RuntimeError) as exc:
                local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
            constructor_errors = self.comm.allgather(local_error)
            if any(error is not None for error in constructor_errors):
                raise RuntimeError(
                    "hybrid path problem construction failed: "
                    + "; ".join(error for error in constructor_errors if error is not None)
                )
            if path_problem is None:  # pragma: no cover - collective gate invariant.
                raise RuntimeError("hybrid path problem is unavailable after construction")

            observed_path_partition = self._hybrid_path_partition_fingerprint(path_problem)
            local_path_errors: list[str] = []
            try:
                expected_path_partition = manifest["path_partition_fingerprints"][self.comm.rank]
                if observed_path_partition != expected_path_partition:
                    local_path_errors.append(
                        "local path/Real partition differs from the hybrid checkpoint"
                    )
                expected_shapes = {
                    "path_z": path_problem.z.x.array.shape,
                    "path_alpha": path_problem.alpha.x.array.shape,
                    "path_target_fracture_energy": (),
                    "path_target_increment": (),
                    "path_target_energy_density": np.asarray(
                        path_problem._target_energy_density.value
                    ).shape,
                }
                for name, shape in expected_shapes.items():
                    values = state[name]
                    if values.shape != shape:
                        local_path_errors.append(
                            f"hybrid restart array {name!r} has shape {values.shape}, "
                            f"expected {shape}"
                        )
                    elif np.iscomplexobj(values) or not np.all(np.isfinite(values)):
                        local_path_errors.append(
                            f"hybrid restart array {name!r} is not finite real data"
                        )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                local_path_errors.append(f"rank {self.comm.rank}: {type(exc).__name__}: {exc}")
            path_errors = self.comm.allgather("; ".join(local_path_errors) or None)
            if any(error is not None for error in path_errors):
                raise RuntimeError(
                    "hybrid path shard validation failed: "
                    + "; ".join(error for error in path_errors if error is not None)
                )

        if phase is ControlPhase.FRACTURE_ENERGY:
            scalar_errors: list[str] = []
            for name in (
                "path_target_fracture_energy",
                "path_target_increment",
                "path_target_energy_density",
            ):
                canonical = self.comm.bcast(
                    np.array(state[name], copy=True) if self.comm.rank == 0 else None,
                    root=0,
                )
                if not np.array_equal(state[name], canonical):
                    scalar_errors.append(
                        f"hybrid restart scalar {name!r} differs across MPI shards"
                    )
            queue = scheduler_state.fracture_energy_queue
            switch_step = scheduler_state.switch_accepted_step
            if queue is None or switch_step is None or path_problem is None:
                scalar_errors.append("hybrid path scheduler/problem state is incomplete")
            else:
                path_records = manifest["history"][switch_step + 1 :]
                expected_internal_increment = (
                    float(path_records[-1]["control_target"])
                    - (
                        float(path_records[-2]["control_value"])
                        if len(path_records) > 1
                        else float(manifest["history"][switch_step]["fracture_energy"])
                    )
                    if path_records
                    else 0.0
                )
                stored_target = float(state["path_target_fracture_energy"])
                stored_increment = float(state["path_target_increment"])
                stored_density = float(
                    np.asarray(state["path_target_energy_density"]).reshape(-1)[0]
                )
                if not self._hybrid_control_coordinate_close(
                    stored_target,
                    queue.accepted_value,
                ):
                    scalar_errors.append(
                        "hybrid stored path target disagrees with queue accepted value"
                    )
                if not self._hybrid_control_coordinate_close(
                    stored_increment,
                    expected_internal_increment,
                    coordinate_scale=queue.accepted_value,
                ):
                    scalar_errors.append(
                        "hybrid stored path increment disagrees with the previous measured "
                        "fracture energy"
                    )
                if not self._hybrid_control_coordinate_close(
                    stored_density * path_problem._volume,
                    queue.accepted_value,
                ):
                    scalar_errors.append(
                        "hybrid stored target density disagrees with queue accepted value"
                    )
            scalar_errors_by_rank = self.comm.allgather("; ".join(scalar_errors) or None)
            if any(error is not None for error in scalar_errors_by_rank):
                raise RuntimeError(
                    "hybrid path scalar validation failed: "
                    + "; ".join(error for error in scalar_errors_by_rank if error is not None)
                )

        original_state = None
        original_generation = self._restart_generation
        local_error = None
        try:
            original_state = self._restart_state_arrays()
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        original_state_errors = self.comm.allgather(local_error)
        if any(error is not None for error in original_state_errors):
            raise RuntimeError(
                "hybrid restart rollback snapshot failed: "
                + "; ".join(error for error in original_state_errors if error is not None)
            )
        if original_state is None:  # pragma: no cover - collective gate invariant.
            raise RuntimeError("hybrid restart rollback snapshot is unavailable")

        def rollback_and_raise(stage: str, errors: list[str | None]) -> None:
            rollback_error = None
            try:
                self._restore_restart_state_arrays(original_state)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                rollback_error = f"rank {self.comm.rank}: rollback {type(exc).__name__}: {exc}"
            rollback_errors = self.comm.allgather(rollback_error)
            self._path_control_problem = None
            self._restart_generation = original_generation
            details = [error for error in errors if error is not None]
            details.extend(error for error in rollback_errors if error is not None)
            raise RuntimeError(
                f"hybrid restart state restoration failed during {stage}: " + "; ".join(details)
            )

        local_error = None
        try:
            self._restore_restart_state_arrays(state)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        restore_errors = self.comm.allgather(local_error)
        if any(error is not None for error in restore_errors):
            rollback_and_raise("common-state restore", restore_errors)

        if phase is ControlPhase.FRACTURE_ENERGY:
            local_error = None
            try:
                if path_problem is None:
                    raise RuntimeError("hybrid path problem is unavailable after validation")
                reference = scheduler_state.reference_displacement
                accepted_load = float(np.asarray(self.load.value).reshape(-1)[0].real)
                path_problem.initialize_from_physical_displacement(
                    self.u,
                    load_factor=accepted_load / reference,
                )
                path_problem.z.x.array[:] = state["path_z"]
                path_problem.alpha.x.array[:] = state["path_alpha"]
                path_problem.z.x.scatter_forward()
                path_problem.alpha.x.scatter_forward()
                path_problem._target_fracture_energy = float(state["path_target_fracture_energy"])
                path_problem._target_increment = float(state["path_target_increment"])
                path_problem._target_energy_density.value = state["path_target_energy_density"]
                path_problem._update_variable_bounds()
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
            path_initialisation_errors = self.comm.allgather(local_error)
            if any(error is not None for error in path_initialisation_errors):
                rollback_and_raise(
                    "path-state initialisation",
                    path_initialisation_errors,
                )

        recomputed_energy = None
        certificates = None
        local_error = None
        try:
            certificates = self._hybrid_terminal_certificates(
                path_problem if phase is ControlPhase.FRACTURE_ENERGY else None
            )
            recomputed_energy = certificates["fracture_energy"]
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        energy_assembly_errors = self.comm.allgather(local_error)
        if any(error is not None for error in energy_assembly_errors):
            rollback_and_raise("fracture-energy assembly", energy_assembly_errors)
        if recomputed_energy is None:  # pragma: no cover - collective gate invariant.
            rollback_and_raise(
                "fracture-energy assembly",
                ["recomputed fracture energy is unavailable"],
            )
        if certificates is None:  # pragma: no cover - collective gate invariant.
            rollback_and_raise(
                "terminal FE certification",
                ["terminal FE certificates are unavailable"],
            )

        validation_errors: list[str] = []
        final_history_energy = float(manifest["history"][-1]["fracture_energy"])
        if phase is ControlPhase.FRACTURE_ENERGY:
            switch_step = scheduler_state.switch_accepted_step
            path_records = manifest["history"][switch_step + 1 :] if switch_step is not None else []
            trusted_control_increment = (
                float(path_records[-1]["control_target"])
                - (
                    float(path_records[-2]["control_target"])
                    if len(path_records) > 1
                    else float(manifest["history"][switch_step]["fracture_energy"])
                )
                if path_records and switch_step is not None
                else self.config.path_control.target_increment
            )
        else:
            path_records = []
            trusted_control_increment = self.config.path_control.target_increment
        if not self._hybrid_energy_close(
            recomputed_energy,
            final_history_energy,
            increment=trusted_control_increment,
        ):
            validation_errors.append(
                "hybrid restored fracture energy disagrees with accepted history"
            )

        if phase is ControlPhase.FRACTURE_ENERGY:
            if path_problem is None:  # pragma: no cover - collective gate invariant.
                validation_errors.append("hybrid path problem is unavailable")
            else:
                reference = scheduler_state.reference_displacement
                accepted_load = float(np.asarray(self.load.value).reshape(-1)[0].real)
                load_factor = path_problem.load_factor
                bound_status = path_problem.load_factor_bound_status()
                displacement = load_factor * reference
                lower_factor, upper_factor = path_problem.load_factor_bounds
                local_u_error = float(
                    np.max(
                        np.abs(
                            self.u.x.array
                            - (path_problem.z.x.array + displacement * path_problem.lift.x.array)
                        ),
                        initial=0.0,
                    )
                )
                u_error = float(self.comm.allreduce(local_u_error, op=MPI.MAX))
                if not self._hybrid_close(
                    displacement,
                    accepted_load,
                    scale=reference,
                ):
                    validation_errors.append(
                        "hybrid path alpha/reference product disagrees with restored load"
                    )
                if bound_status is not None or not lower_factor < load_factor < upper_factor:
                    validation_errors.append(
                        "hybrid restored load factor is not strictly inside its bounds"
                    )
                if u_error > 1.0e-12 * max(1.0, abs(displacement)):
                    validation_errors.append(
                        "hybrid restored physical displacement disagrees with z/alpha/lift"
                    )
                if not np.array_equal(path_problem.z.x.array, state["path_z"]):
                    validation_errors.append("hybrid path z array was not restored bitwise")
                if not np.array_equal(
                    path_problem.alpha.x.array,
                    state["path_alpha"],
                ):
                    validation_errors.append("hybrid path alpha array was not restored bitwise")
                queue = scheduler_state.fracture_energy_queue
                switch_step = scheduler_state.switch_accepted_step
                if queue is None or switch_step is None:
                    validation_errors.append("hybrid restored energy scheduler has no queue/switch")
                else:
                    expected_internal_increment = (
                        float(path_records[-1]["control_target"])
                        - (
                            float(path_records[-2]["control_value"])
                            if len(path_records) > 1
                            else float(manifest["history"][switch_step]["fracture_energy"])
                        )
                        if path_records
                        else 0.0
                    )
                    if not self._hybrid_control_coordinate_close(
                        path_problem.target_fracture_energy,
                        queue.accepted_value,
                    ):
                        validation_errors.append(
                            "hybrid path target disagrees with queue accepted value"
                        )
                    if not self._hybrid_control_coordinate_close(
                        path_problem._target_increment,
                        expected_internal_increment,
                        coordinate_scale=queue.accepted_value,
                    ):
                        validation_errors.append(
                            "hybrid path internal increment disagrees with the previous "
                            "measured fracture energy"
                        )
                    energy_is_certified = self._hybrid_control_coordinate_close(
                        recomputed_energy,
                        queue.accepted_value,
                    )
                    if expected_internal_increment > 0.0:
                        energy_is_certified = self._hybrid_control_residual_certificate(
                            accepted_value=(queue.accepted_value - expected_internal_increment),
                            target_value=queue.accepted_value,
                            achieved_value=recomputed_energy,
                        ).certified
                    if not energy_is_certified:
                        validation_errors.append(
                            "hybrid recomputed fracture energy disagrees with queue accepted value"
                        )
                    target_density = float(
                        np.asarray(path_problem._target_energy_density.value).reshape(-1)[0]
                    )
                    if not self._hybrid_control_coordinate_close(
                        target_density * path_problem._volume,
                        queue.accepted_value,
                    ):
                        validation_errors.append(
                            "hybrid target energy density disagrees with queue accepted value"
                        )

        validation_errors.extend(
            self._hybrid_terminal_certificate_errors(
                certificates,
                manifest["history"][-1],
                phase=phase,
                control_increment=trusted_control_increment,
            )
        )
        validation_errors.extend(
            self._hybrid_terminal_interface_errors(
                manifest["history"],
                manifest["interface_history"],
            )
        )

        validation_errors_by_rank = self.comm.allgather("; ".join(validation_errors) or None)
        if any(error is not None for error in validation_errors_by_rank):
            rollback_and_raise(
                "post-restore certification",
                validation_errors_by_rank,
            )

        if phase is ControlPhase.FRACTURE_ENERGY:
            self._path_control_problem = path_problem
        self._restart_generation = int(manifest["generation"])

        accepted = manifest["accepted_state"]
        return (
            manifest["history"],
            manifest["interface_history"],
            manifest["attempt_history"],
            scheduler_state,
            int(accepted["accepted_step"]),
            float(accepted["displacement"]),
        )

    def _save_restart_checkpoint(
        self,
        output: Path,
        *,
        history: list[dict],
        interface_history: list[dict[str, Any]],
        attempt_history: list[dict[str, Any]],
        pending: list[tuple[float, int, int]],
        accepted_step: int,
        accepted_displacement: float,
    ) -> None:
        """Atomically commit one accepted state using alternating MPI shard slots."""
        restart_directory = output / "restart"
        if self.comm.rank == 0:
            try:
                restart_directory.mkdir(parents=False, exist_ok=True)
                directory_error = None
            except OSError as exc:
                directory_error = f"{type(exc).__name__}: {exc}"
        else:
            directory_error = None
        directory_error = self.comm.bcast(directory_error, root=0)
        if directory_error is not None:
            raise RuntimeError(f"restart directory preparation failed: {directory_error}")
        self.comm.barrier()

        generation = self._restart_generation + 1
        slot = generation % 2
        shard_name = f"state_slot_{slot}_rank_{self.comm.rank:06d}.npz"
        shard_path = restart_directory / shard_name
        temporary = shard_path.with_name(f".{shard_path.name}.tmp")
        local_error = None
        local_metadata: dict[str, Any] | None = None
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **self._restart_state_arrays())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, shard_path)
            local_metadata = {
                "rank": self.comm.rank,
                "file": shard_name,
                "bytes": shard_path.stat().st_size,
                "sha256": self._file_sha256(shard_path),
            }
        except (OSError, ValueError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        write_errors = self.comm.allgather(local_error)
        if any(error is not None for error in write_errors):
            raise RuntimeError(
                "restart shard write failed: "
                + "; ".join(error for error in write_errors if error is not None)
            )
        shard_metadata = self.comm.gather(local_metadata, root=0)
        partition_fingerprints = self.comm.gather(self._restart_partition_fingerprint(), root=0)

        if self.comm.rank == 0:
            try:
                manifest_path = restart_directory / "checkpoint.json"
                parent_checkpoint = None
                if manifest_path.is_file():
                    previous_bytes = manifest_path.read_bytes()
                    previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
                    previous_manifest = json.loads(previous_bytes)
                    self._validate_restart_manifest(previous_manifest)
                    previous_generation = previous_manifest.get("generation")
                    if previous_generation != self._restart_generation:
                        raise RuntimeError(
                            "previous restart manifest generation changed during this run"
                        )
                    archive_directory = restart_directory / "manifests"
                    archive_directory.mkdir(exist_ok=True)
                    archive_name = f"checkpoint_generation_{previous_generation:06d}.json"
                    archive_path = archive_directory / archive_name
                    if archive_path.is_file():
                        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != previous_sha256:
                            raise RuntimeError(
                                "archived parent restart manifest conflicts with checkpoint"
                            )
                    else:
                        archive_temporary = archive_path.with_name(f".{archive_path.name}.tmp")
                        with archive_temporary.open("wb") as stream:
                            stream.write(previous_bytes)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(archive_temporary, archive_path)
                    parent_checkpoint = {
                        "generation": previous_generation,
                        "manifest_sha256": previous_sha256,
                        "archived_manifest": f"manifests/{archive_name}",
                        "shards": previous_manifest.get("shards"),
                    }
                manifest = {
                    "schema_version": self.RESTART_SCHEMA_VERSION,
                    "status": "partial",
                    "committed_at_utc": datetime.now(UTC).isoformat(),
                    "generation": generation,
                    "slot": slot,
                    "config_fingerprint": self._restart_configuration_fingerprint(),
                    "implementation_fingerprint": (self._restart_implementation_fingerprint()),
                    "runtime_fingerprint": self._restart_runtime_fingerprint(),
                    "mpi_ranks": self.comm.size,
                    "parent_checkpoint": parent_checkpoint,
                    "accepted_state": {
                        "accepted_step": int(accepted_step),
                        "displacement": float(accepted_displacement),
                    },
                    "pending_loads": self._pending_to_json(pending),
                    "history": history,
                    "interface_history": interface_history,
                    "attempt_history": attempt_history,
                    "partition_fingerprints": partition_fingerprints,
                    "shards": shard_metadata,
                    "continuation": {
                        "policy_version": 1,
                        "base_controls": self._continuation_controls(self.config.loading),
                        "effective_controls": self._continuation_controls(self._effective_loading),
                        "sessions": self._continuation_sessions,
                        "legacy_schema1_implementation_identity_unavailable": (
                            self._legacy_schema1_implementation_identity_unavailable
                        ),
                    },
                    "field_state": {
                        "functions": list(self._RESTART_FUNCTION_NAMES),
                        "includes_load_constant": True,
                        "irreversibility_history": "d_lower",
                        "hydrogen_precharge_state": [
                            "c_h",
                            "c_h_old",
                            "theta_h",
                            "trapped_hydrogen",
                            "Gc",
                        ],
                        "material_field_state": ["Gc0", "diffusivity", "trap_density"],
                    },
                    "compatibility": {
                        "requires_same_config": True,
                        "requires_same_implementation_from_schema2": True,
                        "requires_same_runtime_from_schema2": True,
                        "requires_same_mpi_size_and_partition": True,
                        "xdmf_resume_policy": (
                            "new segment; existing field files are not overwritten"
                        ),
                    },
                }
                self._atomic_write_json(manifest_path, manifest)
                manifest_error = None
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                manifest_error = f"{type(exc).__name__}: {exc}"
        else:
            manifest_error = None
        manifest_error = self.comm.bcast(manifest_error, root=0)
        if manifest_error is not None:
            raise RuntimeError(f"restart manifest commit failed: {manifest_error}")
        self.comm.barrier()
        self._restart_generation = generation

    def _load_restart_checkpoint(
        self,
        output: Path,
    ) -> tuple[
        list[dict],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[tuple[float, int, int]],
        int,
        float,
    ]:
        restart_directory = output / "restart"
        manifest_path = restart_directory / "checkpoint.json"
        if self.comm.rank == 0:
            try:
                with manifest_path.open(encoding="utf-8") as stream:
                    manifest = json.load(stream)
                continuation = self._validate_restart_manifest(manifest)
                parent = manifest.get("parent_checkpoint")
                if parent is not None:
                    archive_path = restart_directory / parent["archived_manifest"]
                    if not archive_path.is_file():
                        raise RuntimeError(
                            "restart parent manifest archive is missing or corrupted"
                        )
                    archive_bytes = archive_path.read_bytes()
                    if hashlib.sha256(archive_bytes).hexdigest() != parent["manifest_sha256"]:
                        raise RuntimeError(
                            "restart parent manifest archive is missing or corrupted"
                        )
                    parent_manifest = json.loads(archive_bytes)
                    self._validate_restart_manifest(parent_manifest)
                    if (
                        parent_manifest.get("generation") != parent["generation"]
                        or parent_manifest.get("shards") != parent["shards"]
                    ):
                        raise RuntimeError(
                            "restart parent manifest archive disagrees with lineage metadata"
                        )
                manifest_error = None
            except (OSError, json.JSONDecodeError, RuntimeError) as exc:
                manifest = None
                continuation = None
                manifest_error = f"{type(exc).__name__}: {exc}"
        else:
            manifest = None
            continuation = None
            manifest_error = None
        manifest, continuation, manifest_error = self.comm.bcast(
            (manifest, continuation, manifest_error), root=0
        )
        if manifest_error is not None:
            raise RuntimeError(f"restart checkpoint validation failed: {manifest_error}")
        self._effective_loading = replace(
            self.config.loading,
            stagger_max_iterations=int(
                continuation["effective_controls"]["stagger_max_iterations"]
            ),
            maximum_subdivisions=int(continuation["effective_controls"]["maximum_subdivisions"]),
            minimum_increment=float(continuation["effective_controls"]["minimum_increment"]),
        )
        self._continuation_sessions = list(continuation["sessions"])
        self._legacy_schema1_implementation_identity_unavailable = bool(
            continuation["legacy_schema1_implementation_identity_unavailable"]
        )

        partition_fingerprint = self._restart_partition_fingerprint()
        expected_partition = manifest["partition_fingerprints"][self.comm.rank]
        local_error = None
        state: dict[str, np.ndarray] | None = None
        try:
            if partition_fingerprint != expected_partition:
                raise RuntimeError("local mesh/dof partition differs from the committed checkpoint")
            shard = manifest["shards"][self.comm.rank]
            if not isinstance(shard, dict):
                raise RuntimeError("restart shard metadata is invalid")
            if shard.get("rank") != self.comm.rank:
                raise RuntimeError("restart shard rank ordering is invalid")
            shard_name = shard.get("file")
            if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
                raise RuntimeError("restart shard filename is invalid")
            shard_path = restart_directory / shard_name
            if shard_path.stat().st_size != shard.get("bytes"):
                raise RuntimeError("restart shard size differs from the manifest")
            if self._file_sha256(shard_path) != shard.get("sha256"):
                raise RuntimeError("restart shard checksum differs from the manifest")
            with np.load(shard_path, allow_pickle=False) as archive:
                expected_names = {*self._RESTART_FUNCTION_NAMES, "load"}
                if set(archive.files) != expected_names:
                    raise RuntimeError("restart shard field set is invalid")
                state = {name: np.array(archive[name], copy=True) for name in archive.files}
            for name in self._RESTART_FUNCTION_NAMES:
                expected_shape = getattr(self, name).x.array.shape
                if state[name].shape != expected_shape:
                    raise RuntimeError(
                        f"restart array {name!r} shape differs from the current partition"
                    )
                if np.iscomplexobj(state[name]) or not np.all(np.isfinite(state[name])):
                    raise RuntimeError(f"restart array {name!r} is not finite real data")
            tolerance = 1.0e-10
            if (
                np.min(state["d"].real, initial=0.0) < -tolerance
                or np.max(state["d"].real, initial=0.0) > 1.0 + tolerance
                or np.min(state["d_lower"].real, initial=0.0) < -tolerance
                or np.max(state["d_lower"].real, initial=0.0) > 1.0 + tolerance
                or np.min((state["d"] - state["d_lower"]).real, initial=0.0) < -tolerance
            ):
                raise RuntimeError("restart damage state violates bounds or irreversibility")
            for name in ("Gc0", "Gc", "diffusivity"):
                if np.min(state[name].real, initial=math.inf) <= 0.0:
                    raise RuntimeError(f"restart array {name!r} must be positive")
            if np.min(state["trap_density"].real, initial=0.0) < -tolerance:
                raise RuntimeError("restart trap density must be non-negative")
            expected_load_shape = np.asarray(self.load.value).shape
            if state["load"].shape != expected_load_shape:
                raise RuntimeError(
                    "restart load array shape differs from the current scalar Constant"
                )
            if np.iscomplexobj(state["load"]) or not np.all(np.isfinite(state["load"])):
                raise RuntimeError("restart load Constant is not finite real data")
            restored_load = float(np.asarray(state["load"]).reshape(-1)[0].real)
            accepted_load = float(manifest["accepted_state"]["displacement"])
            if not math.isclose(
                restored_load,
                accepted_load,
                rel_tol=0.0,
                abs_tol=1.0e-14 * max(1.0, abs(accepted_load)),
            ):
                raise RuntimeError("restart load Constant disagrees with the accepted displacement")
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            local_error = f"rank {self.comm.rank}: {type(exc).__name__}: {exc}"
        load_errors = self.comm.allgather(local_error)
        if any(error is not None for error in load_errors):
            raise RuntimeError(
                "restart shard validation failed: "
                + "; ".join(error for error in load_errors if error is not None)
            )
        if state is None:
            raise RuntimeError("restart shard state is unavailable after validation")
        self._restore_restart_state_arrays(state)
        self._restart_generation = int(manifest["generation"])

        history = manifest["history"]
        interface_history = manifest["interface_history"]
        attempt_history = manifest["attempt_history"]
        pending = [
            (
                float(item["displacement"]),
                int(item["subdivision_level"]),
                int(item["scheduled_step"]),
            )
            for item in manifest["pending_loads"]
        ]
        accepted = manifest["accepted_state"]
        return (
            history,
            interface_history,
            attempt_history,
            pending,
            int(accepted["accepted_step"]),
            float(accepted["displacement"]),
        )

    def _field_output_paths(self, output: Path, *, resume: bool) -> dict[str, Path]:
        if not resume:
            return {
                "damage": output / "damage.xdmf",
                "displacement": output / "displacement.xdmf",
                "material": output / "material.xdmf",
                "hydrogen": output / "hydrogen.xdmf",
            }
        if self.comm.rank == 0:
            try:
                segment = 1
                while any(
                    (output / f"{stem}.resume_{segment:03d}{suffix}").exists()
                    for stem in ("damage", "displacement", "material", "hydrogen")
                    for suffix in (".xdmf", ".h5")
                ):
                    segment += 1
                segment_error = None
            except OSError as exc:
                segment = None
                segment_error = f"{type(exc).__name__}: {exc}"
        else:
            segment = None
            segment_error = None
        segment, segment_error = self.comm.bcast(
            (segment, segment_error),
            root=0,
        )
        if segment_error is not None:
            raise RuntimeError(f"resume field-output segment discovery failed: {segment_error}")
        return {
            stem: output / f"{stem}.resume_{segment:03d}.xdmf"
            for stem in ("damage", "displacement", "material", "hydrogen")
        }

    @staticmethod
    def _update_energy_balance(history: list[dict], record: dict) -> None:
        previous = history[-1]
        increment = record["displacement"] - previous["displacement"]
        work_increment = 0.5 * (previous["reaction_y"] + record["reaction_y"]) * increment
        record["external_work"] = previous["external_work"] + work_increment
        internal_change = record["total_internal_energy"] - history[0]["total_internal_energy"]
        residual = internal_change - record["external_work"]
        record["energy_balance_residual"] = residual
        scale = max(abs(internal_change), abs(record["external_work"]), 1.0e-30)
        record["energy_balance_relative"] = residual / scale

    def run(
        self,
        *,
        resume: bool = False,
        resume_stagger_max_iterations: int | None = None,
        resume_maximum_subdivisions: int | None = None,
        resume_minimum_increment: float | None = None,
    ) -> list[dict]:
        """Run or explicitly resume all load increments with accepted-state checkpoints."""
        if self.config.path_control.enabled:
            from .hybrid_runner import run_fresh_hybrid

            return run_fresh_hybrid(
                self,
                resume=resume,
                resume_stagger_max_iterations=resume_stagger_max_iterations,
                resume_maximum_subdivisions=resume_maximum_subdivisions,
                resume_minimum_increment=resume_minimum_increment,
            )
        continuation_overrides = (
            resume_stagger_max_iterations,
            resume_maximum_subdivisions,
            resume_minimum_increment,
        )
        if not resume and any(value is not None for value in continuation_overrides):
            raise ValueError("continuation-control overrides require resume=True")
        output = self.config.output_directory
        if self.comm.rank == 0:
            try:
                if output.exists() and not output.is_dir():
                    raise NotADirectoryError(f"output path is not a directory: {output}")
                nonempty = output.exists() and any(output.iterdir())
                if resume:
                    if not nonempty:
                        raise FileNotFoundError(
                            f"resume requested but the output directory is empty: {output}"
                        )
                    if (output / "completion.json").exists():
                        raise FileExistsError(
                            f"refusing to resume an already completed result: {output}"
                        )
                    if not (output / "restart" / "checkpoint.json").is_file():
                        raise FileNotFoundError(
                            "resume requested but no committed restart/checkpoint.json exists"
                        )
                elif not nonempty:
                    output.mkdir(parents=True, exist_ok=True)
                output_error = None
            except OSError as exc:
                nonempty = False
                output_error = f"{type(exc).__name__}: {exc}"
        else:
            nonempty = None
            output_error = None
        nonempty, output_error = self.comm.bcast((nonempty, output_error), root=0)
        if output_error is not None:
            raise RuntimeError(f"output directory preparation failed: {output_error}")
        if nonempty and not resume:
            raise FileExistsError(f"refusing to mix a new run with existing output files: {output}")
        self.comm.barrier()

        if resume:
            (
                history,
                interface_history,
                attempt_history,
                pending,
                accepted_step,
                accepted_displacement,
            ) = self._load_restart_checkpoint(output)
            self._apply_resume_continuation_controls(
                accepted_step=accepted_step,
                accepted_displacement=accepted_displacement,
                stagger_max_iterations=resume_stagger_max_iterations,
                maximum_subdivisions=resume_maximum_subdivisions,
                minimum_increment=resume_minimum_increment,
            )
            # Commit the authenticated state and the new effective controls
            # before attempting another nonlinear solve.
            self._save_restart_checkpoint(
                output,
                history=history,
                interface_history=interface_history,
                attempt_history=attempt_history,
                pending=pending,
                accepted_step=accepted_step,
                accepted_displacement=accepted_displacement,
            )
            self._write_history(output, history)
            self._write_interface_history(output, interface_history, complete=False)
            self._write_attempt_history(output, attempt_history, complete=False)
            self._write_continuation_history(output)
            if self.comm.rank == 0:
                controls = self._continuation_controls(self._effective_loading)
                print(
                    f"resume: accepted_step={accepted_step:03d} "
                    f"u={accepted_displacement:.6e} pending={len(pending)} "
                    f"stagger_max={controls['stagger_max_iterations']} "
                    f"subdivisions={controls['maximum_subdivisions']} "
                    f"minimum_increment={controls['minimum_increment']:.3e}"
                )
        else:
            self._write_metadata(output)
            self._run_hydrogen_precharge(output)

            # Equilibrate the diffuse profile around the sharp geometric pre-crack
            # before applying load.  Otherwise the first non-zero step mixes crack
            # initialisation energy with the mechanical response.
            initial_info = self._solve_load_step(0.0)
            interface_history = []
            attempt_history = []
            history = [
                self._record(
                    0,
                    0.0,
                    initial_info,
                    scheduled_step=0,
                    subdivision_level=0,
                    load_increment=0.0,
                )
            ]
            if not initial_info["converged"]:
                self._write_history(output, history)
                raise RuntimeError("pre-crack phase-field equilibration did not converge")
            initial_interface = self._interface_history_record(
                accepted_step=0,
                scheduled_step=0,
                displacement=0.0,
                subdivision_level=0,
            )
            if initial_interface is not None:
                interface_history.append(initial_interface)

            nominal_displacements = np.linspace(
                0.0,
                self.config.loading.maximum_displacement,
                self.config.loading.steps + 1,
            )[1:]
            pending = [
                (float(displacement), 0, scheduled_step)
                for scheduled_step, displacement in enumerate(nominal_displacements, start=1)
            ]
            accepted_displacement = 0.0
            accepted_step = 0
            self._save_restart_checkpoint(
                output,
                history=history,
                interface_history=interface_history,
                attempt_history=attempt_history,
                pending=pending,
                accepted_step=accepted_step,
                accepted_displacement=accepted_displacement,
            )
            self._write_history(output, history)
            self._write_interface_history(output, interface_history, complete=False)
            self._write_attempt_history(output, attempt_history, complete=False)
            self._write_continuation_history(output)

        field_paths = self._field_output_paths(output, resume=resume)
        damage_file = io.XDMFFile(self.comm, field_paths["damage"], "w")
        displacement_file = io.XDMFFile(self.comm, field_paths["displacement"], "w")
        try:
            damage_file.write_mesh(self.domain)
            displacement_file.write_mesh(self.domain)
            damage_file.write_function(self.d, accepted_displacement)
            displacement_file.write_function(self.u, accepted_displacement)
            with io.XDMFFile(self.comm, field_paths["material"], "w") as material_file:
                material_file.write_mesh(self.domain)
                material_file.write_function(self.Gc0, accepted_displacement)
                material_file.write_function(self.Gc, accepted_displacement)
                material_file.write_function(self.diffusivity, accepted_displacement)
                material_file.write_function(self.trap_density, accepted_displacement)
            if self.config.hydrogen.enabled:
                with io.XDMFFile(self.comm, field_paths["hydrogen"], "w") as hydrogen_file:
                    hydrogen_file.write_mesh(self.domain)
                    hydrogen_file.write_function(self.c_h, self.config.hydrogen.charging_time)
                    hydrogen_file.write_function(self.theta_h, self.config.hydrogen.charging_time)
                    hydrogen_file.write_function(
                        self.trapped_hydrogen, self.config.hydrogen.charging_time
                    )

            while pending:
                displacement, subdivision_level, scheduled_step = pending.pop(0)
                load_increment = displacement - accepted_displacement
                state = self._snapshot_state()
                failure: Exception | None = None
                try:
                    solve_info = self._solve_load_step(displacement)
                except (PETSc.Error, RuntimeError) as exc:
                    failure = exc
                    solve_info = {
                        "iterations": self._effective_loading.stagger_max_iterations,
                        "error": math.inf,
                        "converged": False,
                        "damage_snes_iterations": -1,
                        "damage_snes_reason": -1,
                        "elastic_ksp_reason": -1,
                        "aitken_accepted_iterations": -1,
                        "final_aitken_relaxation": None,
                    }

                if not solve_info["converged"]:
                    self._restore_state(state)
                    can_subdivide = (
                        self._effective_loading.adaptive
                        and subdivision_level < self._effective_loading.maximum_subdivisions
                        and 0.5 * load_increment >= self._effective_loading.minimum_increment
                    )
                    if can_subdivide:
                        midpoint = accepted_displacement + 0.5 * load_increment
                        next_level = subdivision_level + 1
                        pending.insert(0, (displacement, next_level, scheduled_step))
                        pending.insert(0, (midpoint, next_level, scheduled_step))
                        checkpoint_pending = pending
                    else:
                        # A committed queue must never omit the failed target:
                        # otherwise a later resume could incorrectly finalise
                        # the result without retrying the unaccepted interval.
                        checkpoint_pending = [
                            (displacement, subdivision_level, scheduled_step),
                            *pending,
                        ]
                    raw_error = float(solve_info["error"])
                    attempt_history.append(
                        {
                            "attempt": len(attempt_history) + 1,
                            "accepted_step_before_attempt": accepted_step,
                            "accepted_displacement": accepted_displacement,
                            "target_displacement": displacement,
                            "load_increment": load_increment,
                            "scheduled_step": scheduled_step,
                            "subdivision_level": subdivision_level,
                            "failure_type": (
                                type(failure).__name__ if failure is not None else "nonconverged"
                            ),
                            "failure_message": (
                                str(failure)
                                if failure is not None
                                else "alternate minimisation did not satisfy convergence gates"
                            ),
                            "iterations": int(solve_info["iterations"]),
                            "error": raw_error if math.isfinite(raw_error) else None,
                            "damage_snes_iterations": int(
                                solve_info.get("damage_snes_iterations", -1)
                            ),
                            "damage_snes_reason": int(solve_info.get("damage_snes_reason", -1)),
                            "elastic_ksp_reason": int(solve_info.get("elastic_ksp_reason", -1)),
                            "aitken_accepted_iterations": int(
                                solve_info.get("aitken_accepted_iterations", -1)
                            ),
                            "final_aitken_relaxation": solve_info.get("final_aitken_relaxation"),
                            "will_subdivide": can_subdivide,
                            "effective_continuation_controls": (
                                self._continuation_controls(self._effective_loading)
                            ),
                        }
                    )
                    self._save_restart_checkpoint(
                        output,
                        history=history,
                        interface_history=interface_history,
                        attempt_history=attempt_history,
                        pending=checkpoint_pending,
                        accepted_step=accepted_step,
                        accepted_displacement=accepted_displacement,
                    )
                    self._write_attempt_history(output, attempt_history, complete=False)
                    if can_subdivide:
                        if self.comm.rank == 0:
                            print(
                                "retry: subdividing load interval "
                                f"[{accepted_displacement:.6e}, {displacement:.6e}] "
                                f"at level {next_level}; "
                                f"last_error={solve_info['error']:.3e}"
                            )
                        continue
                    message = (
                        "alternate minimisation failed at displacement "
                        f"{displacement:.6e}; minimum attempted increment "
                        f"was {load_increment:.6e}"
                    )
                    if failure is not None:
                        raise RuntimeError(message) from failure
                    raise RuntimeError(message)

                accepted_step += 1
                record = self._record(
                    accepted_step,
                    displacement,
                    solve_info,
                    scheduled_step=scheduled_step,
                    subdivision_level=subdivision_level,
                    load_increment=load_increment,
                )
                self._update_energy_balance(history, record)
                history.append(record)
                interface_record = self._interface_history_record(
                    accepted_step=accepted_step,
                    scheduled_step=scheduled_step,
                    displacement=displacement,
                    subdivision_level=subdivision_level,
                )
                if interface_record is not None:
                    interface_history.append(interface_record)
                accepted_displacement = displacement
                self._save_restart_checkpoint(
                    output,
                    history=history,
                    interface_history=interface_history,
                    attempt_history=attempt_history,
                    pending=pending,
                    accepted_step=accepted_step,
                    accepted_displacement=accepted_displacement,
                )
                self._write_interface_history(output, interface_history, complete=False)
                self._write_history(output, history)
                self._write_attempt_history(output, attempt_history, complete=False)
                if accepted_step % self.config.output.write_every == 0 or not pending:
                    damage_file.write_function(self.d, displacement)
                    displacement_file.write_function(self.u, displacement)
                if self.comm.rank == 0:
                    print(
                        f"step={accepted_step:03d} target={scheduled_step:03d} "
                        f"u={displacement:.6e} "
                        f"R={record['reaction_y']:.6e} dmax={record['maximum_damage']:.4f} "
                        f"stagger={solve_info['iterations']} "
                        f"kkt={record['damage_kkt_inf']:.3e}"
                    )
        finally:
            damage_file.close()
            displacement_file.close()
            self._write_history(output, history)
            self._write_interface_history(output, interface_history, complete=False)
            self._write_attempt_history(output, attempt_history, complete=False)
            self._write_continuation_history(output)

        self._write_graph_metrics(output)
        self._write_interface_history(output, interface_history, complete=True)
        self._write_attempt_history(output, attempt_history, complete=True)
        self._write_continuation_history(output)
        self._write_completion(output, history)
        self.comm.barrier()
        return history
