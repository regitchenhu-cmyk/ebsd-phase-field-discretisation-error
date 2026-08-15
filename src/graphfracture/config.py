"""Typed TOML configuration for the DOLFINx research track."""

from __future__ import annotations

import math
import tomllib
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from .homogeneous_at2 import homogeneous_at2_first_peak

INTERFACE_DAMAGE_THRESHOLDS = (0.6, 0.7, 0.8)
CONTINUATION_CONTROL_FIELDS = (
    "stagger_max_iterations",
    "maximum_subdivisions",
    "minimum_increment",
)


def continuation_control_increase(
    checkpoint_controls: dict[str, Any],
    requested_controls: dict[str, Any],
    *,
    allow_equal: bool = False,
) -> dict[str, dict[str, float | int]]:
    """Validate a monotone solver-budget increase after exact checkpoint auth.

    Physical/discretisation/loading-grid invariance is established separately
    by the existing full configuration fingerprint.  This helper accepts only
    the three runtime continuation controls.
    """
    if not isinstance(checkpoint_controls, dict) or not isinstance(requested_controls, dict):
        raise ValueError("continuation migration requires two control mappings")
    expected = set(CONTINUATION_CONTROL_FIELDS)
    if set(checkpoint_controls) != expected or set(requested_controls) != expected:
        raise ValueError("continuation migration contains unknown or missing controls")

    changes: dict[str, dict[str, float | int]] = {}
    for name in CONTINUATION_CONTROL_FIELDS:
        old_value = checkpoint_controls[name]
        new_value = requested_controls[name]
        if name == "minimum_increment":
            if type(old_value) not in {int, float} or type(new_value) not in {int, float}:
                raise ValueError("continuation minimum_increment values must be finite numbers")
            old_number = float(old_value)
            new_number = float(new_value)
            if not math.isfinite(old_number) or not math.isfinite(new_number):
                raise ValueError("continuation minimum_increment values must be finite")
            if old_number <= 0.0 or new_number <= 0.0:
                raise ValueError("continuation minimum_increment values must be positive")
            if new_number > old_number:
                raise ValueError("continuation minimum_increment may only decrease")
            changes[name] = {"checkpoint": old_number, "requested": new_number}
        else:
            if type(old_value) is not int or type(new_value) is not int:
                raise ValueError(f"continuation {name} values must be integers")
            minimum = 1 if name == "stagger_max_iterations" else 0
            if old_value < minimum or new_value < minimum:
                raise ValueError(f"continuation {name} values are outside their valid range")
            if new_value < old_value:
                raise ValueError(f"continuation {name} may only increase")
            changes[name] = {"checkpoint": old_value, "requested": new_value}

    if not allow_equal and not any(
        values["checkpoint"] != values["requested"] for values in changes.values()
    ):
        raise ValueError("continuation migration does not increase any solver budget")
    return changes


@dataclass(frozen=True)
class GeometryConfig:
    length: float = 1.0
    height: float = 0.5
    nx: int = 192
    ny: int = 96
    precrack_length: float = 0.16145833333333334
    diagonal: str = "right"
    x_pin_corner: str = "bottom_left"


@dataclass(frozen=True)
class MaterialConfig:
    young_modulus: float = 210_000.0
    poisson_ratio: float = 0.30
    fracture_toughness: float = 2.7
    length_scale: float = 0.015
    residual_stiffness: float = 1.0e-8


@dataclass(frozen=True)
class LoadingConfig:
    maximum_displacement: float = 0.018
    steps: int = 36
    stagger_max_iterations: int = 120
    stagger_tolerance: float = 1.0e-5
    damage_kkt_tolerance: float = 1.0e-6
    adaptive: bool = True
    maximum_subdivisions: int = 6
    minimum_increment: float = 1.0e-6


@dataclass(frozen=True)
class PathControlConfig:
    """Hybrid displacement/fracture-energy path-following controls.

    The initial branch is advanced with the ordinary displacement schedule.
    When ``enabled`` is true, the solver stops that schedule at
    ``switch_displacement`` and subsequently prescribes increments of the
    regularised AT2 fracture energy while solving the boundary load factor as
    an additional unknown.
    """

    enabled: bool = False
    functional: str = "fracture_energy"
    switch_displacement: float = 0.0
    target_increment: float = 1.0e-3
    steps: int = 1
    adaptive: bool = True
    use_energy_predictor: bool = True
    energy_predictor_disable_after_nominal_path_step: int = -1
    maximum_subdivisions: int = 8
    minimum_increment: float = 1.0e-6
    load_lower_bound: float = 0.0
    load_upper_bound: float = 1.0
    snes_max_iterations: int = 100
    residual_tolerance: float = 1.0e-8
    control_tolerance: float = 1.0e-6
    control_absolute_tolerance: float = 1.0e-12

    def energy_predictor_enabled_for_step(self, nominal_path_step: int) -> bool:
        """Resolve the deterministic predictor policy for one nominal target.

        ``energy_predictor_disable_after_nominal_path_step=-1`` preserves the
        original always-on policy.  A non-negative value keeps the predictor
        through that nominal fracture-energy target and disables it for every
        later target, including all adaptive children of those later targets.
        """
        if type(nominal_path_step) is not int or nominal_path_step < 1:
            raise ValueError("nominal_path_step must be an integer >= 1")
        return self.use_energy_predictor and (
            self.energy_predictor_disable_after_nominal_path_step < 0
            or nominal_path_step
            <= self.energy_predictor_disable_after_nominal_path_step
        )


@dataclass(frozen=True)
class HydrogenConfig:
    """One-way transient pre-charging parameters.

    Length and time units must be consistent with ``geometry`` and
    ``diffusivity``.  ``charging_concentration`` is a dimensionless lattice
    site fraction, while ``background_trap_density`` and edge trap densities
    are trap sites per lattice site.
    """

    enabled: bool = False
    diffusivity: float = 1.0e-4
    charging_concentration: float = 0.05
    charging_time: float = 50.0
    steps: int = 20
    trap_binding_constant: float = 20.0
    background_trap_density: float = 0.0
    toughness_degradation: float = 0.80
    minimum_toughness_ratio: float = 0.20
    charging_boundary: str = "bottom"


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool = True
    influence_radius: float = 0.012
    crack_threshold: float = 0.70
    # Measured-microstructure mode: a grain-boundary chain artifact (.npz) is
    # used instead of hand-written [[graph.nodes]]/[[graph.edges]].  Empty means
    # the synthetic node/edge graph is used.
    chain_artifact: str = ""
    # Drop chains whose geometry confidence is below this floor (filled labels).
    confidence_floor: float = 0.0
    # Fixed-topology attribute-location association permutation; -1 disables it.
    attribute_permutation_seed: int = -1
    # Optional explicit protocol for a synthetic inclined-interface study.
    interface_start_node: str = ""
    interface_impact_node: str = ""
    interface_end_node: str = ""


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "results/sent_graph"
    write_every: int = 1


@dataclass(frozen=True)
class SolverConfig:
    linear_ksp_type: str = "preonly"
    linear_pc_type: str = "lu"
    factor_solver: str = "mumps"
    damage_snes_type: str = "vinewtonrsls"
    aitken_max_relaxation: float = 1.25


@dataclass(frozen=True)
class GraphNodeConfig:
    name: str
    point: tuple[float, float]


@dataclass(frozen=True)
class GraphEdgeConfig:
    source: str
    target: str
    toughness_ratio: float = 0.5
    hydrogen_diffusivity_ratio: float = 1.0
    trap_density: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    geometry: GeometryConfig
    material: MaterialConfig
    loading: LoadingConfig
    graph: GraphConfig
    output: OutputConfig
    solver: SolverConfig
    hydrogen: HydrogenConfig = field(default_factory=HydrogenConfig)
    path_control: PathControlConfig = field(default_factory=PathControlConfig)
    graph_nodes: tuple[GraphNodeConfig, ...] = ()
    graph_edges: tuple[GraphEdgeConfig, ...] = ()
    schema_version: int = 1
    source_path: Path | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` for a configuration that cannot be simulated."""
        g, m, load, graph, hydrogen, path_control = (
            self.geometry,
            self.material,
            self.loading,
            self.graph,
            self.hydrogen,
            self.path_control,
        )
        errors = _runtime_type_errors(self)
        if errors:
            raise ValueError("invalid configuration:\n- " + "\n- ".join(errors))

        if self.schema_version != 1:
            errors.append(f"unsupported schema_version={self.schema_version}; expected 1")
        if g.length <= 0 or g.height <= 0:
            errors.append("geometry length and height must be positive")
        if g.nx < 4 or g.ny < 4:
            errors.append("geometry nx and ny must both be at least 4")
        if not 0.0 < g.precrack_length < g.length:
            errors.append("precrack_length must lie strictly inside the specimen")
        if g.diagonal not in {"left", "right"}:
            errors.append("geometry.diagonal must be one of: left, right")
        if g.x_pin_corner not in {"bottom_left", "top_left"}:
            errors.append("geometry.x_pin_corner must be one of: bottom_left, top_left")
        if g.ny % 2:
            errors.append(
                "geometry ny must be even so the structured mesh contains the pre-crack line"
            )
        if m.young_modulus <= 0 or m.fracture_toughness <= 0 or m.length_scale <= 0:
            errors.append("E, Gc, and ell must be positive")
        if not -1.0 < m.poisson_ratio < 0.5:
            errors.append("poisson_ratio must lie in (-1, 0.5) for isotropic elasticity")
        if not 0.0 < m.residual_stiffness <= 1.0e-2:
            errors.append("residual_stiffness must lie in (0, 1e-2]")
        if load.maximum_displacement <= 0 or load.steps < 2:
            errors.append("maximum_displacement must be positive and steps >= 2")
        if (
            load.stagger_max_iterations < 1
            or load.stagger_tolerance <= 0
            or load.damage_kkt_tolerance <= 0
        ):
            errors.append("invalid stagger iteration controls")
        if load.maximum_subdivisions < 0 or load.minimum_increment <= 0:
            errors.append("adaptive loading requires subdivisions >= 0 and minimum_increment > 0")
        if path_control.functional != "fracture_energy":
            errors.append("path_control.functional must be fracture_energy")
        if path_control.energy_predictor_disable_after_nominal_path_step < -1:
            errors.append(
                "path_control.energy_predictor_disable_after_nominal_path_step "
                "must be -1 or non-negative"
            )
        if path_control.enabled:
            nominal_increment = load.maximum_displacement / load.steps
            switch_index = round(path_control.switch_displacement / nominal_increment)
            aligned_switch = switch_index * nominal_increment
            alignment_tolerance = 1.0e-12 * max(load.maximum_displacement, 1.0)
            if not 0.0 < path_control.switch_displacement < load.maximum_displacement:
                errors.append(
                    "path_control.switch_displacement must lie strictly inside the loading window"
                )
            elif switch_index < 1 or switch_index >= load.steps or not math.isclose(
                path_control.switch_displacement,
                aligned_switch,
                rel_tol=0.0,
                abs_tol=alignment_tolerance,
            ):
                errors.append(
                    "path_control.switch_displacement must coincide with a nominal loading node"
                )
            if path_control.target_increment <= 0.0 or path_control.steps < 1:
                errors.append(
                    "path_control target_increment must be positive and steps must be at least 1"
                )
            if (
                path_control.maximum_subdivisions < 0
                or path_control.minimum_increment <= 0.0
                or path_control.minimum_increment > path_control.target_increment
            ):
                errors.append(
                    "path_control adaptive increments require subdivisions >= 0 and "
                    "0 < minimum_increment <= target_increment"
                )
            if not (
                0.0
                <= path_control.load_lower_bound
                < path_control.switch_displacement
                < path_control.load_upper_bound
                <= load.maximum_displacement
            ):
                errors.append(
                    "path_control load bounds must bracket the switch inside the loading window"
                )
            if (
                path_control.snes_max_iterations < 1
                or path_control.residual_tolerance <= 0.0
                or path_control.control_tolerance <= 0.0
                or not math.isfinite(
                    path_control.control_absolute_tolerance
                )
                or path_control.control_absolute_tolerance <= 0.0
            ):
                errors.append("invalid path_control nonlinear solver controls")
            if hydrogen.enabled:
                errors.append(
                    "path_control with hydrogen is not supported by the first "
                    "verified implementation"
                )
        elif path_control.energy_predictor_disable_after_nominal_path_step != -1:
            errors.append(
                "path_control.energy_predictor_disable_after_nominal_path_step "
                "requires path_control.enabled"
            )
        if graph.influence_radius <= 0:
            errors.append("graph influence_radius must be positive")
        if not 0.0 < graph.crack_threshold <= 1.0:
            errors.append("graph crack_threshold must lie in (0, 1]")
        if not 0.0 <= graph.confidence_floor <= 1.0:
            errors.append("graph confidence_floor must lie in [0, 1]")
        if graph.attribute_permutation_seed < -1:
            errors.append("graph attribute_permutation_seed must be -1 (disabled) or >= 0")
        if self.output.write_every < 1:
            errors.append("output write_every must be at least 1")
        if not self.output.directory.strip():
            errors.append("output directory must not be empty")
        if hydrogen.diffusivity <= 0 or hydrogen.charging_time <= 0:
            errors.append("hydrogen diffusivity and charging_time must be positive")
        if not 0.0 < hydrogen.charging_concentration <= 1.0:
            errors.append("hydrogen charging_concentration must lie in (0, 1]")
        if hydrogen.steps < 1:
            errors.append("hydrogen steps must be at least 1")
        if hydrogen.trap_binding_constant < 0 or hydrogen.background_trap_density < 0:
            errors.append("hydrogen trap parameters must be non-negative")
        if not 0.0 <= hydrogen.toughness_degradation < 1.0:
            errors.append("hydrogen toughness_degradation must lie in [0, 1)")
        if not 0.0 < hydrogen.minimum_toughness_ratio <= 1.0:
            errors.append("hydrogen minimum_toughness_ratio must lie in (0, 1]")
        if hydrogen.charging_boundary not in {"left", "right", "bottom", "top"}:
            errors.append("hydrogen charging_boundary must be left, right, bottom, or top")
        solver_values = (
            self.solver.linear_ksp_type,
            self.solver.linear_pc_type,
            self.solver.damage_snes_type,
        )
        if any(not value.strip() for value in solver_values):
            errors.append("PETSc solver type names must not be empty")
        if self.solver.damage_snes_type not in {"vinewtonrsls", "vinewtonssls"}:
            errors.append("damage_snes_type must be a PETSc VI solver")
        if not 1.0 <= self.solver.aitken_max_relaxation <= 2.0:
            errors.append("aitken_max_relaxation must lie in [1, 2]")

        names = [node.name for node in self.graph_nodes]
        if any(not name.strip() for name in names):
            errors.append("graph node names must not be empty")
        if len(names) != len(set(names)):
            errors.append("graph node names must be unique")
        known = set(names)
        for edge in self.graph_edges:
            if edge.source not in known or edge.target not in known:
                errors.append(f"graph edge {edge.source!r}->{edge.target!r} has an unknown node")
            if edge.source == edge.target:
                errors.append(f"graph edge {edge.source!r}->{edge.target!r} is a self-loop")
            if not 0.0 < edge.toughness_ratio <= 1.0:
                errors.append("every graph edge toughness_ratio must lie in (0, 1]")
            if edge.hydrogen_diffusivity_ratio < 1.0:
                errors.append("every graph edge hydrogen_diffusivity_ratio must be >= 1")
            if edge.trap_density < 0.0:
                errors.append("every graph edge trap_density must be non-negative")

        for node in self.graph_nodes:
            x, y = node.point
            if not (0.0 <= x <= g.length and 0.0 <= y <= g.height):
                errors.append(f"graph node {node.name!r} lies outside the specimen")

        coordinates = [(node.point[0], node.point[1]) for node in self.graph_nodes]
        if len(coordinates) != len(set(coordinates)):
            errors.append("graph node coordinates must be unique")
        undirected_edges = [frozenset((edge.source, edge.target)) for edge in self.graph_edges]
        if len(undirected_edges) != len(set(undirected_edges)):
            errors.append("duplicate undirected graph edges are not allowed")

        using_chains = bool(graph.chain_artifact.strip())
        raw_interface_protocol = (
            graph.interface_start_node,
            graph.interface_impact_node,
            graph.interface_end_node,
        )
        interface_protocol = tuple(value.strip() for value in raw_interface_protocol)
        if any(
            raw != stripped
            for raw, stripped in zip(raw_interface_protocol, interface_protocol, strict=True)
        ):
            errors.append("inclined-interface protocol node names must not contain whitespace")
        protocol_enabled = all(interface_protocol)
        if any(interface_protocol) and not protocol_enabled:
            errors.append("inclined-interface protocol requires start, impact and end nodes")
        if protocol_enabled and using_chains:
            errors.append("inclined-interface protocol is valid only for a synthetic graph")
        if protocol_enabled and not graph.enabled:
            errors.append("inclined-interface protocol requires graph.enabled = true")
        if protocol_enabled and len(set(interface_protocol)) != 3:
            errors.append("inclined-interface protocol nodes must be distinct")
        if protocol_enabled and any(name not in known for name in interface_protocol):
            errors.append("inclined-interface protocol references an unknown graph node")
        if protocol_enabled and all(name in known for name in interface_protocol):
            node_by_name = {node.name: node for node in self.graph_nodes}
            start = node_by_name[interface_protocol[0]].point
            impact = node_by_name[interface_protocol[1]].point
            end = node_by_name[interface_protocol[2]].point
            segment_x, segment_y = end[0] - start[0], end[1] - start[1]
            relative_x, relative_y = impact[0] - start[0], impact[1] - start[1]
            segment_length2 = segment_x**2 + segment_y**2
            cross = segment_x * relative_y - segment_y * relative_x
            projection = relative_x * segment_x + relative_y * segment_y
            tolerance = 1.0e-10 * max(g.length, g.height, 1.0)
            projection_fraction = (
                projection / segment_length2 if segment_length2 > 0.0 else math.nan
            )
            if (
                segment_length2 <= 0.0
                or abs(cross) > tolerance * math.sqrt(max(segment_length2, 1.0e-30))
                or projection_fraction <= 1.0e-10
                or projection_fraction >= 1.0 - 1.0e-10
            ):
                errors.append("interface impact node must lie strictly inside the endpoint segment")
            endpoints_span_height = (
                math.isclose(start[1], 0.0, rel_tol=0.0, abs_tol=tolerance)
                and math.isclose(end[1], g.height, rel_tol=0.0, abs_tol=tolerance)
            ) or (
                math.isclose(end[1], 0.0, rel_tol=0.0, abs_tol=tolerance)
                and math.isclose(start[1], g.height, rel_tol=0.0, abs_tol=tolerance)
            )
            if not endpoints_span_height:
                errors.append(
                    "inclined-interface endpoints must span the bottom and top boundaries"
                )
            discrete_tip = math.floor(g.precrack_length / (g.length / g.nx) + 1.0e-12) * (
                g.length / g.nx
            )
            if (
                not math.isclose(
                    impact[1],
                    0.5 * g.height,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                or impact[0] <= discrete_tip + tolerance
            ):
                errors.append(
                    "interface impact node must lie on the precrack line ahead of the discrete tip"
                )
            edge_by_pair = {
                frozenset((edge.source, edge.target)): edge for edge in self.graph_edges
            }
            first = edge_by_pair.get(frozenset((interface_protocol[0], interface_protocol[1])))
            second = edge_by_pair.get(frozenset((interface_protocol[1], interface_protocol[2])))
            if first is None or second is None:
                errors.append(
                    "inclined-interface protocol requires start-impact and impact-end edges"
                )
            elif (
                first.toughness_ratio,
                first.hydrogen_diffusivity_ratio,
                first.trap_density,
            ) != (
                second.toughness_ratio,
                second.hydrogen_diffusivity_ratio,
                second.trap_density,
            ):
                errors.append("inclined-interface protocol edges must have identical attributes")
        if using_chains and (self.graph_nodes or self.graph_edges):
            errors.append(
                "graph.chain_artifact and [[graph.nodes]]/[[graph.edges]] are mutually exclusive"
            )
        if not graph.enabled and using_chains:
            errors.append("graph.chain_artifact requires graph.enabled = true")
        if graph.enabled and not using_chains and (not self.graph_nodes or not self.graph_edges):
            errors.append(
                "graph.enabled requires either a chain_artifact or at least one node and one edge"
            )
        if not using_chains and graph.confidence_floor != 0.0:
            errors.append("graph.confidence_floor is valid only with graph.chain_artifact")
        if not using_chains and graph.attribute_permutation_seed >= 0:
            errors.append(
                "graph.attribute_permutation_seed is valid only with graph.chain_artifact"
            )

        if errors:
            raise ValueError("invalid configuration:\n- " + "\n- ".join(errors))

    @property
    def output_directory(self) -> Path:
        return self.resolve_path(self.output.directory)

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the TOML source directory."""
        path = Path(relative)
        if path.is_absolute() or self.source_path is None:
            return path
        return (self.source_path.parent / path).resolve()

    def diagnostics(self) -> dict[str, Any]:
        """Return mesh/model scales useful for a pre-run sanity check."""
        g, m, load, graph, hydrogen, path_control = (
            self.geometry,
            self.material,
            self.loading,
            self.graph,
            self.hydrogen,
            self.path_control,
        )
        hx, hy = g.length / g.nx, g.height / g.ny
        h_diameter = math.hypot(hx, hy)
        discrete_tip = math.floor(g.precrack_length / hx + 1.0e-12) * hx
        # Closed-form first local peak of the same homogeneous tensile AT2
        # functional and plane-strain/free-lateral boundary condition used by
        # the DOLFINx patch test.  This remains a constitutive scale diagnostic;
        # the notched two-dimensional specimen is not this homogeneous test.
        homogeneous_peak = homogeneous_at2_first_peak(
            young_modulus=m.young_modulus,
            poisson_ratio=m.poisson_ratio,
            fracture_toughness=m.fracture_toughness,
            length_scale=m.length_scale,
            residual_stiffness=m.residual_stiffness,
        )
        # Engineering LEFM screen for a finite-width single-edge crack under
        # remote tension.  It uses the common polynomial K_I correction with
        # a/W = precrack_length/specimen_length and the *bulk* Gc.  The
        # heterogeneous diffuse-field problem is not reduced to this estimate;
        # it is recorded only to expose obviously under/over-scaled load windows.
        crack_fraction = g.precrack_length / g.length
        lefm_geometry_factor = (
            1.12
            - 0.231 * crack_fraction
            + 10.55 * crack_fraction**2
            - 21.72 * crack_fraction**3
            + 30.39 * crack_fraction**4
        )
        plane_strain_modulus = homogeneous_peak.effective_modulus
        lefm_critical_stress = math.sqrt(
            plane_strain_modulus
            * m.fracture_toughness
            / (math.pi * g.precrack_length * lefm_geometry_factor**2)
        )
        lefm_critical_strain = lefm_critical_stress / plane_strain_modulus
        lefm_critical_displacement = lefm_critical_strain * g.height
        maximum_nominal_strain = load.maximum_displacement / g.height
        small_strain_screening_limit = 1.0e-2
        using_chains = bool(graph.chain_artifact.strip())
        protocol_enabled = all(
            (
                graph.interface_start_node.strip(),
                graph.interface_impact_node.strip(),
                graph.interface_end_node.strip(),
            )
        )
        graph_source_mode = (
            "disabled" if not graph.enabled else ("chains" if using_chains else "segments")
        )
        return {
            "schema_version": self.schema_version,
            "mesh": {
                "cells": 2 * g.nx * g.ny,
                "hx": hx,
                "hy": hy,
                "triangle_diameter": h_diameter,
                "diagonal": g.diagonal,
            },
            "resolution": {
                "ell_over_triangle_diameter": m.length_scale / h_diameter,
                "gb_width_over_triangle_diameter": graph.influence_radius / h_diameter,
                "recommended_minimum": 2.0,
            },
            "precrack": {
                "requested_tip_x": g.precrack_length,
                "discrete_tip_x": discrete_tip,
                "tip_error": g.precrack_length - discrete_tip,
                "mesh_aligned": math.isclose(
                    g.precrack_length, discrete_tip, rel_tol=0.0, abs_tol=1.0e-12
                ),
            },
            "quality_gates": {
                "phase_field_resolution_pass": m.length_scale / h_diameter >= 2.0,
                "precrack_mesh_alignment_pass": math.isclose(
                    g.precrack_length, discrete_tip, rel_tol=0.0, abs_tol=1.0e-12
                ),
                "grain_boundary_band_has_multiple_cells": (
                    graph.influence_radius / h_diameter >= 1.5
                ),
                "small_strain_load_screening_pass": (
                    maximum_nominal_strain <= small_strain_screening_limit
                ),
            },
            "loading": {
                "maximum_nominal_strain": maximum_nominal_strain,
                "small_strain_screening_limit": small_strain_screening_limit,
                "bulk_gc_edge_crack_lefm_screen": {
                    "a_over_W": crack_fraction,
                    "finite_width_geometry_factor": lefm_geometry_factor,
                    "plane_strain_modulus": plane_strain_modulus,
                    "estimated_critical_nominal_stress": lefm_critical_stress,
                    "estimated_critical_nominal_strain": lefm_critical_strain,
                    "estimated_critical_displacement": lefm_critical_displacement,
                    "configured_to_estimated_displacement_ratio": (
                        load.maximum_displacement / lefm_critical_displacement
                    ),
                    "basis": "bulk Gc finite-width single-edge-crack LEFM engineering screen",
                    "interpretation_limit": (
                        "not a phase-field onset prediction and ignores the heterogeneous "
                        "G_GB field"
                    ),
                },
                "screening_note": (
                    "heuristic applicability screen for infinitesimal-strain kinematics; "
                    "not a material validation"
                ),
            },
            "model": {
                "formulation": "plane-strain isotropic AT2; no tension/compression split",
                "irreversibility": "SNESVI bound d_n >= d_(n-1)",
                "horizontal_rigid_body_pin": g.x_pin_corner,
                "homogeneous_plane_strain_AT2_first_peak": homogeneous_peak.to_dict(),
            },
            "graphs": {
                "gb_nodes": len(self.graph_nodes),
                "gb_edges": len(self.graph_edges),
                "enabled": self.graph.enabled,
                "source_mode": graph_source_mode,
                "chain_artifact": self.graph.chain_artifact or None,
                "resolved_chain_artifact": (
                    str(self.resolve_path(self.graph.chain_artifact)) if using_chains else None
                ),
                "confidence_floor": self.graph.confidence_floor,
                "attribute_permutation_seed": (
                    self.graph.attribute_permutation_seed
                    if self.graph.attribute_permutation_seed >= 0
                    else None
                ),
                "inclined_interface_protocol": (
                    {
                        "start_node": graph.interface_start_node,
                        "impact_node": graph.interface_impact_node,
                        "end_node": graph.interface_end_node,
                        "thresholds": list(INTERFACE_DAMAGE_THRESHOLDS),
                    }
                    if protocol_enabled
                    else None
                ),
            },
            "hydrogen": {
                "enabled": hydrogen.enabled,
                "charging_boundary": hydrogen.charging_boundary,
                "diffusion_length": math.sqrt(hydrogen.diffusivity * hydrogen.charging_time),
            },
            "path_control": {
                "enabled": path_control.enabled,
                "functional": path_control.functional,
                "switch_displacement": (
                    path_control.switch_displacement if path_control.enabled else None
                ),
                "switch_nominal_step": (
                    round(
                        path_control.switch_displacement
                        / (load.maximum_displacement / load.steps)
                    )
                    if path_control.enabled
                    else None
                ),
                "target_increment": (
                    path_control.target_increment if path_control.enabled else None
                ),
                "steps": path_control.steps if path_control.enabled else 0,
                "control_certificate_mode": (
                    "composite_relative_absolute"
                    if path_control.enabled
                    else None
                ),
                "control_relative_tolerance": (
                    path_control.control_tolerance
                    if path_control.enabled
                    else None
                ),
                "control_absolute_tolerance": (
                    path_control.control_absolute_tolerance
                    if path_control.enabled
                    else None
                ),
                "use_energy_predictor": path_control.use_energy_predictor,
                "energy_predictor_disable_after_nominal_path_step": (
                    path_control.energy_predictor_disable_after_nominal_path_step
                    if (
                        path_control.enabled
                        and path_control.energy_predictor_disable_after_nominal_path_step
                        >= 0
                    )
                    else None
                ),
                "load_bounds": (
                    [path_control.load_lower_bound, path_control.load_upper_bound]
                    if path_control.enabled
                    else None
                ),
                "interpretation": (
                    "Gc-weighted regularised AT2 fracture-energy path coordinate; "
                    "boundary displacement is an augmented unknown"
                    if path_control.enabled
                    else "ordinary displacement control"
                ),
            },
            "output_directory": str(self.output_directory),
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path) if self.source_path else None
        return data


T = TypeVar("T")


_REAL_FIELDS: dict[type[Any], frozenset[str]] = {
    GeometryConfig: frozenset({"length", "height", "precrack_length"}),
    MaterialConfig: frozenset(
        {
            "young_modulus",
            "poisson_ratio",
            "fracture_toughness",
            "length_scale",
            "residual_stiffness",
        }
    ),
    LoadingConfig: frozenset(
        {
            "maximum_displacement",
            "stagger_tolerance",
            "damage_kkt_tolerance",
            "minimum_increment",
        }
    ),
    PathControlConfig: frozenset(
        {
            "switch_displacement",
            "target_increment",
            "minimum_increment",
            "load_lower_bound",
            "load_upper_bound",
            "residual_tolerance",
            "control_tolerance",
            "control_absolute_tolerance",
        }
    ),
    HydrogenConfig: frozenset(
        {
            "diffusivity",
            "charging_concentration",
            "charging_time",
            "trap_binding_constant",
            "background_trap_density",
            "toughness_degradation",
            "minimum_toughness_ratio",
        }
    ),
    GraphConfig: frozenset({"influence_radius", "crack_threshold", "confidence_floor"}),
    SolverConfig: frozenset({"aitken_max_relaxation"}),
    GraphEdgeConfig: frozenset({"toughness_ratio", "hydrogen_diffusivity_ratio", "trap_density"}),
}
_INTEGER_FIELDS: dict[type[Any], frozenset[str]] = {
    GeometryConfig: frozenset({"nx", "ny"}),
    LoadingConfig: frozenset({"steps", "stagger_max_iterations", "maximum_subdivisions"}),
    PathControlConfig: frozenset(
        {
            "steps",
            "energy_predictor_disable_after_nominal_path_step",
            "maximum_subdivisions",
            "snes_max_iterations",
        }
    ),
    HydrogenConfig: frozenset({"steps"}),
    OutputConfig: frozenset({"write_every"}),
    GraphConfig: frozenset({"attribute_permutation_seed"}),
}
_BOOLEAN_FIELDS: dict[type[Any], frozenset[str]] = {
    GraphConfig: frozenset({"enabled"}),
    LoadingConfig: frozenset({"adaptive"}),
    PathControlConfig: frozenset(
        {"enabled", "adaptive", "use_energy_predictor"}
    ),
    HydrogenConfig: frozenset({"enabled"}),
}
_STRING_FIELDS: dict[type[Any], frozenset[str]] = {
    GeometryConfig: frozenset({"diagonal", "x_pin_corner"}),
    PathControlConfig: frozenset({"functional"}),
    OutputConfig: frozenset({"directory"}),
    SolverConfig: frozenset(
        {"linear_ksp_type", "linear_pc_type", "factor_solver", "damage_snes_type"}
    ),
    HydrogenConfig: frozenset({"charging_boundary"}),
    GraphConfig: frozenset(
        {
            "chain_artifact",
            "interface_start_node",
            "interface_impact_node",
            "interface_end_node",
        }
    ),
    GraphNodeConfig: frozenset({"name"}),
    GraphEdgeConfig: frozenset({"source", "target"}),
}


def _finite_real(value: Any, context: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{context} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer(value: Any, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _string(value: Any, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} must be a string")
    return value


def _normalise_scalar_fields(cls: type[T], values: dict[str, Any], section: str) -> dict[str, Any]:
    result = dict(values)
    for name in _REAL_FIELDS.get(cls, ()):
        if name in result:
            result[name] = _finite_real(result[name], f"{section}.{name}")
    for name in _INTEGER_FIELDS.get(cls, ()):
        if name in result:
            result[name] = _integer(result[name], f"{section}.{name}")
    for name in _BOOLEAN_FIELDS.get(cls, ()):
        if name in result:
            result[name] = _boolean(result[name], f"{section}.{name}")
    for name in _STRING_FIELDS.get(cls, ()):
        if name in result:
            result[name] = _string(result[name], f"{section}.{name}")
    return result


def _strict_dataclass(cls: type[T], values: dict[str, Any], section: str) -> T:
    if type(values) is not dict:
        raise ValueError(f"[{section}] must be a table")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {', '.join(unknown)}")
    required = {
        f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"missing keys in [{section}]: {', '.join(missing)}")
    return cls(**_normalise_scalar_fields(cls, values, section))


def _point(value: Any, context: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{context}.point must contain exactly two numbers")
    return (
        _finite_real(value[0], f"{context}.point[0]"),
        _finite_real(value[1], f"{context}.point[1]"),
    )


def _runtime_type_errors(config: RunConfig) -> list[str]:
    """Return type/finite-value errors for configs built without ``load_config``."""
    errors: list[str] = []
    sections = (
        ("geometry", config.geometry, GeometryConfig),
        ("material", config.material, MaterialConfig),
        ("loading", config.loading, LoadingConfig),
        ("graph", config.graph, GraphConfig),
        ("output", config.output, OutputConfig),
        ("solver", config.solver, SolverConfig),
        ("hydrogen", config.hydrogen, HydrogenConfig),
        ("path_control", config.path_control, PathControlConfig),
    )
    for section, value, cls in sections:
        if type(value) is not cls:
            errors.append(f"{section} must be a {cls.__name__}")
            continue
        for config_field in fields(cls):
            try:
                _normalise_scalar_fields(
                    cls,
                    {config_field.name: getattr(value, config_field.name)},
                    section,
                )
            except ValueError as exc:
                errors.append(str(exc))

    try:
        _integer(config.schema_version, "schema_version")
    except ValueError as exc:
        errors.append(str(exc))

    if type(config.graph_nodes) is not tuple:
        errors.append("graph_nodes must be a tuple")
    else:
        for index, node in enumerate(config.graph_nodes):
            context = f"graph.nodes[{index}]"
            if type(node) is not GraphNodeConfig:
                errors.append(f"{context} must be a GraphNodeConfig")
                continue
            try:
                _string(node.name, f"{context}.name")
                if type(node.point) is not tuple:
                    raise ValueError(f"{context}.point must be a tuple")
                _point(node.point, context)
            except ValueError as exc:
                errors.append(str(exc))

    if type(config.graph_edges) is not tuple:
        errors.append("graph_edges must be a tuple")
    else:
        for index, edge in enumerate(config.graph_edges):
            context = f"graph.edges[{index}]"
            if type(edge) is not GraphEdgeConfig:
                errors.append(f"{context} must be a GraphEdgeConfig")
                continue
            try:
                _normalise_scalar_fields(
                    GraphEdgeConfig,
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "toughness_ratio": edge.toughness_ratio,
                        "hydrogen_diffusivity_ratio": edge.hydrogen_diffusivity_ratio,
                        "trap_density": edge.trap_density,
                    },
                    context,
                )
            except ValueError as exc:
                errors.append(str(exc))

    if config.source_path is not None and not isinstance(config.source_path, Path):
        errors.append("source_path must be a pathlib.Path or None")
    return errors


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if type(value) is not dict:
        raise ValueError(f"[{name}] must be a table")
    return value


def _array_of_tables(value: Any, context: str) -> list[dict[str, Any]]:
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"{context} must be an array of tables")
    return value


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a versioned TOML run configuration."""
    source = Path(path).resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)

    top_allowed = {
        "schema_version",
        "geometry",
        "material",
        "loading",
        "graph",
        "output",
        "solver",
        "hydrogen",
        "path_control",
    }
    unknown = sorted(set(raw) - top_allowed)
    if unknown:
        raise ValueError(f"unknown top-level keys: {', '.join(unknown)}")

    graph_raw = dict(_section(raw, "graph"))
    nodes_raw = _array_of_tables(graph_raw.pop("nodes", []), "graph.nodes")
    edges_raw = _array_of_tables(graph_raw.pop("edges", []), "graph.edges")
    nodes_list: list[GraphNodeConfig] = []
    for index, item in enumerate(nodes_raw):
        context = f"graph.nodes[{index}]"
        node_values = dict(item)
        if "point" in node_values:
            node_values["point"] = _point(node_values["point"], context)
        nodes_list.append(_strict_dataclass(GraphNodeConfig, node_values, context))
    nodes = tuple(nodes_list)
    edges = tuple(
        _strict_dataclass(GraphEdgeConfig, item, f"graph.edges[{index}]")
        for index, item in enumerate(edges_raw)
    )

    schema_version = raw.get("schema_version", 1)
    schema_version = _integer(schema_version, "schema_version")

    config = RunConfig(
        geometry=_strict_dataclass(GeometryConfig, _section(raw, "geometry"), "geometry"),
        material=_strict_dataclass(MaterialConfig, _section(raw, "material"), "material"),
        loading=_strict_dataclass(LoadingConfig, _section(raw, "loading"), "loading"),
        graph=_strict_dataclass(GraphConfig, graph_raw, "graph"),
        output=_strict_dataclass(OutputConfig, _section(raw, "output"), "output"),
        solver=_strict_dataclass(SolverConfig, _section(raw, "solver"), "solver"),
        hydrogen=_strict_dataclass(HydrogenConfig, _section(raw, "hydrogen"), "hydrogen"),
        path_control=_strict_dataclass(
            PathControlConfig,
            _section(raw, "path_control"),
            "path_control",
        ),
        graph_nodes=nodes,
        graph_edges=edges,
        schema_version=schema_version,
        source_path=source,
    )
    config.validate()
    return config
