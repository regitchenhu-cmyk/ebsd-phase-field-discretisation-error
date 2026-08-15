"""Monolithic fracture-energy path control for the DOLFINx research solver.

The load factor is a genuine unknown of the augmented nonlinear system.  The
implementation deliberately does not search for a displacement in an outer
loop: displacement equilibrium, the damage variational inequality and the
fracture-energy control equation are solved by one PETSc SNESVI problem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector, assign
from mpi4py import MPI
from petsc4py import PETSc

from .config import RunConfig
from .damage_control import fracture_energy_control_residual_certificate


class FractureEnergyControlProblem:
    """Augmented ``(z, d, alpha)`` AT2 fracture-energy control problem.

    The physical displacement is represented as

    ``u = z + alpha * reference_displacement * lift``

    where ``lift=(0, y/H)`` and ``z`` has homogeneous versions of the
    displacement Dirichlet conditions.  ``alpha`` is a DOLFINx Real-element
    degree of freedom, so the scalar control equation participates in the same
    automatically differentiated block Jacobian as mechanics and damage.
    """

    def __init__(
        self,
        *,
        domain: mesh.Mesh,
        config: RunConfig,
        V_u: fem.FunctionSpace,
        V_d: fem.FunctionSpace,
        d: fem.Function,
        d_lower: fem.Function,
        d_upper: fem.Function,
        Gc: fem.Function,
        reference_displacement: float,
        homogeneous_bcs_z: Sequence[fem.DirichletBC],
        load_factor_bounds: tuple[float, float] | None = None,
        snes_max_iterations: int | None = None,
        petsc_options_prefix: str = "graphfracture_path_control_",
    ) -> None:
        if type(reference_displacement) not in {int, float}:
            raise TypeError("reference_displacement must be a real number")
        if not math.isfinite(reference_displacement) or reference_displacement <= 0.0:
            raise ValueError("reference_displacement must be finite and positive")
        if snes_max_iterations is None:
            snes_max_iterations = config.path_control.snes_max_iterations
        if type(snes_max_iterations) is not int or snes_max_iterations < 1:
            raise ValueError("snes_max_iterations must be a positive integer")
        if not petsc_options_prefix:
            raise ValueError("petsc_options_prefix must not be empty")
        for name, function, space in (
            ("d", d, V_d),
            ("d_lower", d_lower, V_d),
            ("d_upper", d_upper, V_d),
        ):
            if function.function_space is not space:
                raise ValueError(f"{name} must use the supplied V_d space")
        if Gc.function_space.mesh is not domain:
            raise ValueError("Gc must be defined on the supplied domain")
        gc_owned = (
            Gc.function_space.dofmap.index_map.size_local
            * Gc.function_space.dofmap.index_map_bs
        )
        local_gc_finite = bool(np.all(np.isfinite(Gc.x.array[:gc_owned].real)))
        local_gc_minimum = float(np.min(Gc.x.array[:gc_owned].real, initial=math.inf))
        if not self._allreduce_bool(domain.comm, local_gc_finite):
            raise ValueError("Gc must contain finite real values")
        if float(domain.comm.allreduce(local_gc_minimum, op=MPI.MIN)) <= 0.0:
            raise ValueError("Gc must be strictly positive")

        self.domain = domain
        self.comm = domain.comm
        self.config = config
        self.V_u = V_u
        self.V_d = V_d
        self.d = d
        self.d_lower = d_lower
        self.d_upper = d_upper
        self.Gc = Gc
        self.reference_displacement = float(reference_displacement)
        self.homogeneous_bcs_z = list(homogeneous_bcs_z)

        if load_factor_bounds is None:
            if config.path_control.enabled:
                load_factor_bounds = (
                    config.path_control.load_lower_bound / self.reference_displacement,
                    config.path_control.load_upper_bound / self.reference_displacement,
                )
            else:
                load_factor_bounds = (
                    0.0,
                    config.loading.maximum_displacement / self.reference_displacement,
                )
        if len(load_factor_bounds) != 2:
            raise ValueError("load_factor_bounds must contain exactly two values")
        if any(type(value) not in {int, float} for value in load_factor_bounds):
            raise TypeError("load_factor_bounds must contain real numbers")
        alpha_min, alpha_max = (float(value) for value in load_factor_bounds)
        if (
            not math.isfinite(alpha_min)
            or not math.isfinite(alpha_max)
            or alpha_min >= alpha_max
        ):
            raise ValueError("load_factor_bounds must be finite and strictly ordered")
        self.load_factor_bounds = (alpha_min, alpha_max)

        self.z = fem.Function(V_u, name="path_displacement_correction")
        self.lift = fem.Function(V_u, name="path_displacement_lift")
        height = float(config.geometry.height)
        self.lift.interpolate(
            lambda x: np.vstack(
                (
                    np.zeros(x.shape[1], dtype=PETSc.ScalarType),
                    np.asarray(x[1] / height, dtype=PETSc.ScalarType),
                )
            )
        )
        self.lift.x.scatter_forward()

        self.V_alpha = fem.functionspace(
            domain,
            basix.ufl.real_element(domain.basix_cell(), dtype=PETSc.ScalarType),
        )
        self.alpha = fem.Function(self.V_alpha, name="path_load_factor")
        self._set_real_function(self.alpha, 1.0)
        self._target_energy_density = fem.Constant(domain, PETSc.ScalarType(0.0))
        self._volume = self._assemble_volume()
        self._target_fracture_energy = 0.0
        self._target_increment = 0.0
        self._initialised = False

        v = ufl.TestFunction(V_u)
        q = ufl.TestFunction(V_d)
        eta = ufl.TestFunction(self.V_alpha)
        m = config.material
        mu = m.young_modulus / (2.0 * (1.0 + m.poisson_ratio))
        lam = (
            m.young_modulus
            * m.poisson_ratio
            / ((1.0 + m.poisson_ratio) * (1.0 - 2.0 * m.poisson_ratio))
        )

        self.physical_displacement_expression = (
            self.z + self.alpha * self.reference_displacement * self.lift
        )
        eps = ufl.sym(ufl.grad(self.physical_displacement_expression))
        sigma = 2.0 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(2)
        psi = 0.5 * ufl.inner(sigma, eps)
        degradation = (
            (1.0 - m.residual_stiffness) * (1.0 - self.d) ** 2
            + m.residual_stiffness
        )

        elastic_residual = degradation * ufl.inner(
            sigma, ufl.sym(ufl.grad(v))
        ) * ufl.dx
        elastic_damage_residual = (
            2.0
            * (1.0 - m.residual_stiffness)
            * psi
            * (self.d - 1.0)
            * q
            * ufl.dx
        )
        fracture_damage_residual = (
            self.Gc
            * (
                (self.d / m.length_scale) * q
                + m.length_scale * ufl.dot(ufl.grad(self.d), ufl.grad(q))
            )
            * ufl.dx
        )
        damage_residual = elastic_damage_residual + fracture_damage_residual
        crack_density = self.d**2 / (2.0 * m.length_scale) + (
            0.5 * m.length_scale * ufl.dot(ufl.grad(self.d), ufl.grad(self.d))
        )
        fracture_density = self.Gc * crack_density
        control_residual = (
            fracture_density - self._target_energy_density
        ) * eta * ufl.dx

        # Public compiled forms are intentionally exposed for the parent
        # solver's post-solve equilibrium and KKT certification.
        self.elastic_residual_form = fem.form(elastic_residual)
        self.elastic_damage_residual_form = fem.form(elastic_damage_residual)
        self.fracture_damage_residual_form = fem.form(fracture_damage_residual)
        self.damage_residual_form = fem.form(damage_residual)
        self.fracture_energy_form = fem.form(fracture_density * ufl.dx)
        self.control_residual_form = fem.form(control_residual)

        options: dict[str, Any] = {
            "snes_type": config.solver.damage_snes_type,
            "snes_rtol": config.path_control.residual_tolerance,
            "snes_atol": 1.0e-10,
            "snes_stol": 1.0e-12,
            "snes_max_it": snes_max_iterations,
            "snes_linesearch_type": "bt",
            "snes_error_if_not_converged": True,
            "ksp_type": config.solver.linear_ksp_type,
            "pc_type": config.solver.linear_pc_type,
            "ksp_error_if_not_converged": True,
        }
        if config.solver.linear_pc_type == "lu" and config.solver.factor_solver:
            options["pc_factor_mat_solver_type"] = config.solver.factor_solver

        # Omitting J is deliberate: DOLFINx differentiates the complete
        # three-by-three block residual, including the alpha column and the
        # scalar fracture-energy row.  kind="mpi" gives MUMPS one monolithic
        # AIJ operator rather than a nested matrix.
        self.problem = NonlinearProblem(
            [elastic_residual, damage_residual, control_residual],
            [self.z, self.d, self.alpha],
            bcs=self.homogeneous_bcs_z,
            kind="mpi",
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=options,
        )
        self._create_variable_bounds()

    @staticmethod
    def _allreduce_bool(comm: MPI.Intracomm, value: bool) -> bool:
        return bool(comm.allreduce(value, op=MPI.LAND))

    def _assemble_volume(self) -> float:
        one = fem.Constant(self.domain, PETSc.ScalarType(1.0))
        volume = self._global_scalar(fem.form(one * ufl.dx))
        if not math.isfinite(volume) or volume <= 0.0:
            raise RuntimeError("path-control domain volume must be finite and positive")
        return volume

    def _global_scalar(self, form: Any) -> float:
        local = fem.assemble_scalar(form)
        return float(self.comm.allreduce(local, op=MPI.SUM).real)

    def _set_real_function(self, function: fem.Function, value: float) -> None:
        function.x.array[:] = PETSc.ScalarType(value)
        function.x.scatter_forward()

    def _real_function_value(self, function: fem.Function) -> float:
        index_map = function.function_space.dofmap.index_map
        owned = index_map.size_local * function.function_space.dofmap.index_map_bs
        local_count = 1 if owned else 0
        local_value = float(function.x.array[0].real) if owned else 0.0
        count = int(self.comm.allreduce(local_count, op=MPI.SUM))
        total = float(self.comm.allreduce(local_value, op=MPI.SUM))
        if count != 1:
            raise RuntimeError("Real-element load factor must have exactly one global owner")
        return total

    def _create_variable_bounds(self) -> None:
        self._z_lower = fem.Function(self.V_u)
        self._z_upper = fem.Function(self.V_u)
        self._alpha_lower = fem.Function(self.V_alpha)
        self._alpha_upper = fem.Function(self.V_alpha)
        self._z_lower.x.array[:] = -PETSc.INFINITY
        self._z_upper.x.array[:] = PETSc.INFINITY
        self._set_real_function(self._alpha_lower, self.load_factor_bounds[0])
        self._set_real_function(self._alpha_upper, self.load_factor_bounds[1])
        self._lower_vector = self.problem.x.duplicate()
        self._upper_vector = self.problem.x.duplicate()
        self._update_variable_bounds()
        self.problem.solver.setVariableBounds(self._lower_vector, self._upper_vector)

    def _update_variable_bounds(self) -> None:
        self.d_lower.x.scatter_forward()
        self.d_upper.x.scatter_forward()
        assign(
            [self._z_lower, self.d_lower, self._alpha_lower],
            self._lower_vector,
        )
        assign(
            [self._z_upper, self.d_upper, self._alpha_upper],
            self._upper_vector,
        )

    def initialize_from_physical_displacement(
        self,
        physical_u: fem.Function,
        *,
        load_factor: float = 1.0,
    ) -> None:
        """Initialise ``z`` and ``alpha`` from an accepted physical state."""
        if physical_u.function_space is not self.V_u:
            raise ValueError("physical_u must use the supplied V_u space")
        value = float(load_factor)
        if not math.isfinite(value):
            raise ValueError("load_factor must be finite")
        lower, upper = self.load_factor_bounds
        if value < lower or value > upper:
            raise ValueError("load_factor lies outside the configured bounds")
        self._set_real_function(self.alpha, value)
        self.z.x.array[:] = physical_u.x.array - (
            value * self.reference_displacement * self.lift.x.array
        )
        self.z.x.scatter_forward()
        self._target_fracture_energy = self.fracture_energy
        self._target_energy_density.value = PETSc.ScalarType(
            self._target_fracture_energy / self._volume
        )
        self._target_increment = 0.0
        self._initialised = True

    @property
    def load_factor(self) -> float:
        return self._real_function_value(self.alpha)

    @property
    def displacement(self) -> float:
        return self.load_factor * self.reference_displacement

    @property
    def fracture_energy(self) -> float:
        return self._global_scalar(self.fracture_energy_form)

    @property
    def target_fracture_energy(self) -> float:
        return self._target_fracture_energy

    @property
    def control_residual(self) -> float:
        return self.fracture_energy - self._target_fracture_energy

    @property
    def assembled_control_residual(self) -> float:
        """Return the directly assembled global Real-equation residual."""
        values = self._owned_residual_values(self.control_residual_form, self.V_alpha)
        local = float(np.sum(values))
        return float(self.comm.allreduce(local, op=MPI.SUM))

    @property
    def control_residual_relative(self) -> float:
        return abs(self.control_residual) / max(abs(self._target_increment), 1.0e-30)

    def freeze_damage_lower_bound(self) -> None:
        """Set the next control step's irreversibility bound to current damage."""
        self.d_lower.x.array[:] = self.d.x.array
        self.d_lower.x.scatter_forward()

    def _apply_energy_predictor(self, target: float, current: float) -> None:
        """Move off the active lower bound without selecting a displacement.

        AT2 fracture energy is quadratic under a uniform scaling of ``d``.
        Scaling the accepted field therefore gives a deterministic initial
        damage predictor; ``alpha`` remains an unknown of the augmented solve.
        """
        if current > 1.0e-30:
            scale = math.sqrt(target / current)
            trial = scale * self.d.x.array.real
        else:
            mean_gc = self._global_scalar(fem.form(self.Gc * ufl.dx)) / self._volume
            amplitude = math.sqrt(
                2.0 * self.config.material.length_scale * target / (mean_gc * self._volume)
            )
            trial = self.d.x.array.real + amplitude
        self.d.x.array[:] = np.clip(
            trial,
            self.d_lower.x.array.real,
            self.d_upper.x.array.real,
        )
        self.d.x.scatter_forward()

    def _snapshot_state(self) -> dict[str, Any]:
        """Capture every mutable quantity touched by one path-control attempt."""
        return {
            "z": self.z.x.array.copy(),
            "d": self.d.x.array.copy(),
            "d_lower": self.d_lower.x.array.copy(),
            "alpha": self.alpha.x.array.copy(),
            "target_fracture_energy": self._target_fracture_energy,
            "target_increment": self._target_increment,
            "target_energy_density": np.array(
                self._target_energy_density.value,
                copy=True,
            ),
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        """Restore an accepted state after an exception or failed certificate."""
        for function, name in (
            (self.z, "z"),
            (self.d, "d"),
            (self.d_lower, "d_lower"),
            (self.alpha, "alpha"),
        ):
            function.x.array[:] = state[name]
            function.x.scatter_forward()
        self._target_fracture_energy = float(state["target_fracture_energy"])
        self._target_increment = float(state["target_increment"])
        self._target_energy_density.value = state["target_energy_density"]
        self._update_variable_bounds()

    def solve(
        self,
        target_fracture_energy: float,
        *,
        freeze_damage_lower_bound: bool = True,
        use_energy_predictor: bool = True,
    ) -> dict[str, float | int | bool | str | None]:
        """Solve and certify one absolute fracture-energy target transactionally.

        An exception or a state that fails any independent physical certificate
        restores ``z``, ``d``, ``d_lower``, ``alpha`` and the accepted target
        exactly.  The caller can therefore refine the fracture-energy interval
        without inheriting a rejected nonlinear iterate.
        """
        if not self._initialised:
            raise RuntimeError("path control must be initialised from an accepted physical state")
        if type(target_fracture_energy) not in {int, float}:
            raise TypeError("target_fracture_energy must be a real number")
        if type(use_energy_predictor) is not bool:
            raise TypeError("use_energy_predictor must be a boolean")
        target = float(target_fracture_energy)
        if not math.isfinite(target) or target < 0.0:
            raise ValueError("target_fracture_energy must be finite and non-negative")
        current = self.fracture_energy
        tolerance = 1.0e-14 * max(1.0, abs(current), abs(target))
        if target <= current + tolerance:
            raise ValueError("target_fracture_energy must exceed the accepted current value")
        state = self._snapshot_state()
        try:
            if freeze_damage_lower_bound:
                self.freeze_damage_lower_bound()
            self._target_fracture_energy = target
            self._target_increment = target - current
            self._target_energy_density.value = PETSc.ScalarType(target / self._volume)
            self._update_variable_bounds()
            if use_energy_predictor:
                self._apply_energy_predictor(target, current)

            self.problem.solve()
            self.z.x.scatter_forward()
            self.d.x.scatter_forward()
            self.alpha.x.scatter_forward()
            reason = int(self.problem.solver.getConvergedReason())
            ksp_reason = int(self.problem.solver.getKSP().getConvergedReason())
            control_residual = self.control_residual
            assembled_control_residual = self.assembled_control_residual
            control_certificate = fracture_energy_control_residual_certificate(
                control_residual,
                accepted_value=current,
                target_value=target,
                relative_tolerance=self.config.path_control.control_tolerance,
                absolute_tolerance=(
                    self.config.path_control.control_absolute_tolerance
                ),
            )
            control_relative = control_certificate.relative
            kkt_absolute, kkt_relative, kkt_scale = self.damage_kkt_metrics()
            mechanical_absolute, mechanical_relative, mechanical_scale = (
                self.mechanical_residual_metrics()
            )
            lower_violation, upper_violation = self.damage_bound_violations()
            minimum_increment = self.minimum_damage_increment()
            bound_status = self.load_factor_bound_status()
            load_factor = self.load_factor
            displacement = self.displacement
            fracture_energy = self.fracture_energy
            finite_metrics = all(
                math.isfinite(value)
                for value in (
                    control_residual,
                    assembled_control_residual,
                    control_certificate.absolute,
                    control_certificate.limit,
                    control_certificate.ratio,
                    control_relative,
                    kkt_absolute,
                    kkt_relative,
                    mechanical_absolute,
                    mechanical_relative,
                    lower_violation,
                    upper_violation,
                    minimum_increment,
                    load_factor,
                    displacement,
                    fracture_energy,
                )
            )
            certified = (
                reason > 0
                and ksp_reason > 0
                and finite_metrics
                and control_certificate.certified
                and math.isclose(
                    assembled_control_residual,
                    control_residual,
                    rel_tol=1.0e-8,
                    abs_tol=1.0e-12,
                )
                and kkt_relative <= self.config.loading.damage_kkt_tolerance
                and mechanical_relative <= self.config.path_control.residual_tolerance
                and lower_violation <= 1.0e-10
                and upper_violation <= 1.0e-10
                and minimum_increment >= -1.0e-10
                and bound_status is None
            )
            info: dict[str, float | int | bool | str | None] = {
                "converged": certified,
                "snes_converged": reason > 0,
                "certified": certified,
                "iterations": int(self.problem.solver.getIterationNumber()),
                "snes_reason": reason,
                "ksp_reason": ksp_reason,
                "load_factor": load_factor,
                "displacement": displacement,
                "fracture_energy": fracture_energy,
                "target_fracture_energy": target,
                "control_residual": control_residual,
                "assembled_control_residual": assembled_control_residual,
                "control_residual_absolute": control_certificate.absolute,
                "control_residual_relative": control_relative,
                "control_residual_limit": control_certificate.limit,
                "control_residual_ratio": control_certificate.ratio,
                "control_certificate_branch": (
                    "relative"
                    if self.config.path_control.control_tolerance
                    * control_certificate.increment
                    >= self.config.path_control.control_absolute_tolerance
                    * control_certificate.scale
                    else "absolute"
                ),
                "damage_kkt_inf": kkt_absolute,
                "damage_kkt_relative": kkt_relative,
                "damage_kkt_scale": kkt_scale,
                "mechanical_residual_inf": mechanical_absolute,
                "mechanical_residual_relative": mechanical_relative,
                "mechanical_residual_scale": mechanical_scale,
                "damage_lower_bound_violation": lower_violation,
                "damage_upper_bound_violation": upper_violation,
                "minimum_damage_increment": minimum_increment,
                "load_factor_bound_status": bound_status,
                "energy_predictor_used": use_energy_predictor,
            }
        except Exception:
            self._restore_state(state)
            raise
        if not certified:
            self._restore_state(state)
        return info

    def copy_physical_displacement_to(self, physical_u: fem.Function) -> float:
        """Copy ``z + alpha*Uref*lift`` into an existing physical field."""
        if physical_u.function_space is not self.V_u:
            raise ValueError("physical_u must use the supplied V_u space")
        displacement = self.displacement
        physical_u.x.array[:] = self.z.x.array + displacement * self.lift.x.array
        physical_u.x.scatter_forward()
        return displacement

    def _owned_residual_values(self, form: Any, space: fem.FunctionSpace) -> np.ndarray:
        residual = assemble_vector(form)
        residual.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        owned = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
        values = np.asarray(residual.array_r[:owned].real).copy()
        residual.destroy()
        return values

    def damage_kkt_metrics(self) -> tuple[float, float, float]:
        """Return physical damage VI violation, relative violation and scale."""
        values = self._owned_residual_values(self.damage_residual_form, self.V_d)
        elastic = self._owned_residual_values(self.elastic_damage_residual_form, self.V_d)
        fracture = self._owned_residual_values(self.fracture_damage_residual_form, self.V_d)
        damage = self.d.x.array[: values.size].real
        lower = self.d_lower.x.array[: values.size].real
        upper = self.d_upper.x.array[: values.size].real
        bound_tolerance = 1.0e-8
        at_lower = damage <= lower + bound_tolerance
        at_upper = damage >= upper - bound_tolerance
        fixed = upper - lower <= 1.0e-14
        violation = np.abs(values)
        violation[at_lower] = np.maximum(-values[at_lower], 0.0)
        violation[at_upper] = np.maximum(values[at_upper], 0.0)
        narrow = at_lower & at_upper & ~fixed
        violation[narrow] = np.abs(values[narrow])
        violation[fixed] = 0.0
        local_violation = float(np.max(violation, initial=0.0))
        local_scale = max(
            float(np.max(np.abs(elastic), initial=0.0)),
            float(np.max(np.abs(fracture), initial=0.0)),
        )
        absolute = float(self.comm.allreduce(local_violation, op=MPI.MAX))
        scale = float(self.comm.allreduce(local_scale, op=MPI.MAX))
        return absolute, absolute / max(scale, 1.0e-30), scale

    def mechanical_residual_metrics(self) -> tuple[float, float, float]:
        """Return free-DOF equilibrium violation and the full force scale."""
        values = self._owned_residual_values(self.elastic_residual_form, self.V_u)
        constrained = np.zeros(values.size, dtype=bool)
        for bc in self.homogeneous_bcs_z:
            dofs, _ = bc.dof_indices()
            owned_dofs = np.asarray(dofs, dtype=np.int64)
            owned_dofs = owned_dofs[owned_dofs < values.size]
            constrained[owned_dofs] = True
        local_absolute = float(np.max(np.abs(values[~constrained]), initial=0.0))
        local_scale = float(np.max(np.abs(values), initial=0.0))
        absolute = float(self.comm.allreduce(local_absolute, op=MPI.MAX))
        scale = float(self.comm.allreduce(local_scale, op=MPI.MAX))
        return absolute, absolute / max(scale, 1.0e-30), scale

    def damage_bound_violations(self) -> tuple[float, float]:
        """Return maximum violations of the lower and upper damage bounds."""
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

    def load_factor_bound_status(self, *, tolerance: float = 1.0e-10) -> str | None:
        """Report an active load-window bound; such a state is right-censored."""
        if type(tolerance) not in {int, float}:
            raise TypeError("tolerance must be a real number")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        value = self.load_factor
        lower, upper = self.load_factor_bounds
        scale = max(abs(lower), abs(upper), abs(value), 1.0)
        if value <= lower + float(tolerance) * scale:
            return "lower"
        if value >= upper - float(tolerance) * scale:
            return "upper"
        return None

    def minimum_damage_increment(self) -> float:
        owned = self.V_d.dofmap.index_map.size_local * self.V_d.dofmap.index_map_bs
        local = float(
            np.min(
                (self.d.x.array[:owned] - self.d_lower.x.array[:owned]).real,
                initial=math.inf,
            )
        )
        return float(self.comm.allreduce(local, op=MPI.MIN))
