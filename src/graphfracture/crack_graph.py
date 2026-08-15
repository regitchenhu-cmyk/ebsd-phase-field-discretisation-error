"""Extract graph-theoretic crack metrics from a finite-element damage field."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

Point2D = tuple[float, float]


@dataclass(frozen=True)
class CrackGraphMetrics:
    threshold: float
    active_nodes: int
    active_edges: int
    components: int
    largest_component_nodes: int
    main_component_nodes: int
    tip_x: float | None
    spans_left_to_right: bool
    path_length: float | None
    tortuosity: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "thresholded_active_graph_shortest_edge_path_length": self.path_length,
            "path_length_role": "thresholded active-FE-graph shortest edge path",
            "path_length_interpretation_limit": (
                "mesh- and threshold-dependent topology diagnostic; not a cross-mesh physical "
                "crack-length measure"
            ),
        }


@dataclass(frozen=True)
class InterfaceInteractionMetrics:
    threshold: float
    impact_tolerance: float
    interface_corridor_width: float
    impact_exclusion_radius: float
    confirmation_distance: float
    penetration_corridor_half_height: float
    main_component_nodes: int
    reached_interface: bool
    closest_main_node_to_impact: float | None
    interface_forward_advance: float
    penetration_forward_advance: float
    interface_active_edge_length: float
    penetration_active_edge_length: float
    geometric_classification: str

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "role": "thresholded-crack geometric screen",
            "active_edge_measure_note": (
                "edge measures sum all active finite-element mesh edges inside each corridor; "
                "they are descriptive on one mesh and are not crack-length convergence metrics"
            ),
            "interpretation_limit": (
                "classification is mesh/threshold dependent and is not by itself a "
                "validated penetration-deflection transition"
            ),
        }


_INTERFACE_CLASSIFICATIONS = frozenset(
    {
        "pre_impact_right_censored",
        "arrested_or_unresolved",
        "deflection_candidate",
        "penetration_candidate",
        "mixed_or_branched",
    }
)


def interface_classification_consensus(classifications: Iterable[str]) -> str:
    """Return an exact threshold consensus or the explicit ambiguity label."""
    values = tuple(str(value) for value in classifications)
    if not values:
        raise ValueError("at least one interface classification is required")
    invalid = sorted(set(values) - _INTERFACE_CLASSIFICATIONS)
    if invalid:
        raise ValueError(f"unknown interface classification(s): {invalid}")
    return values[0] if all(value == values[0] for value in values[1:]) else "threshold_ambiguous"


@dataclass(frozen=True)
class EmbeddedCrackGeometry:
    """Deterministic geometry of the precrack-connected active component.

    ``points`` and ``edges`` preserve the complete selected component for
    topology auditing.  ``backbone`` is the deterministic shortest embedded
    path from the precrack boundary to the furthest-x active node and is the
    ordered polyline used for cross-mesh path comparisons.
    """

    status: str
    threshold: float
    points: tuple[Point2D, ...]
    edges: tuple[tuple[int, int], ...]
    backbone: tuple[Point2D, ...]
    backbone_length: float | None
    branched: bool
    source: Point2D | None
    tip: Point2D | None

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **asdict(self)}


@dataclass(frozen=True)
class PathComparisonMetrics:
    """Mesh-density-resistant comparison of two ordered embedded crack paths."""

    status: str
    observed_geometry_status: str
    reference_geometry_status: str
    observed_branched: bool | None
    reference_branched: bool | None
    sample_spacing: float
    coverage_radius: float
    directed_mean_observed_to_reference: float | None
    directed_mean_reference_to_observed: float | None
    symmetric_chamfer: float | None
    sampled_directed_hausdorff_observed_to_reference: float | None
    sampled_directed_hausdorff_reference_to_observed: float | None
    sampled_symmetric_hausdorff: float | None
    reference_coverage: float | None
    observed_precision: float | None
    exit_status: str
    observed_exit_coordinates: tuple[float, ...]
    reference_exit_coordinates: tuple[float, ...]
    exit_error: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "path_measure_note": (
                "distances use fixed-physical-arclength quadrature points and point-to-segment "
                "distance, not raw active-node density; sampled Hausdorff values require a "
                "sample-spacing convergence check; all distance values are primary-backbone-only"
            ),
            "coverage_note": (
                "reference_coverage is reference-backbone length recall; observed_precision "
                "penalizes excursions of the selected observed backbone; neither includes "
                "off-backbone branches"
            ),
            "topology_note": (
                "distance, coverage, and exit comparisons are primary-backbone-only; branched "
                "flags are reported separately, off-backbone branches are excluded, and exit "
                "error is not evaluated when either geometry is branched"
            ),
        }


def _validated_planar_inputs(
    points: Sequence[Sequence[float]], damage: Sequence[float]
) -> tuple[tuple[Point2D, ...], tuple[float, ...]]:
    if len(points) != len(damage):
        raise ValueError("points and damage must have equal length")
    planar_points: list[Point2D] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"point {index} must contain exactly two coordinates")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"point {index} contains a non-finite coordinate")
        planar_points.append((x, y))
    planar_damage: list[float] = []
    for index, value in enumerate(damage):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"damage value {index} is not finite")
        planar_damage.append(scalar)
    return tuple(planar_points), tuple(planar_damage)


def _validated_point(value: Sequence[float], *, name: str) -> Point2D:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    point = float(value[0]), float(value[1])
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{name} contains a non-finite coordinate")
    return point


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalise_edges(edges: Iterable[tuple[int, int]], n: int) -> tuple[tuple[int, int], ...]:
    result: set[tuple[int, int]] = set()
    for i, j in edges:
        i, j = int(i), int(j)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("edge endpoint lies outside the point array")
        if i != j:
            result.add((min(i, j), max(i, j)))
    return tuple(sorted(result))


def _components(adjacency: dict[int, list[int]], active: set[int]) -> list[set[int]]:
    unseen = set(active)
    groups: list[set[int]] = []
    while unseen:
        seed = unseen.pop()
        group = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    group.add(neighbour)
                    stack.append(neighbour)
        groups.append(group)
    return groups


def _select_main_component(
    groups: Sequence[set[int]], left: set[int], points: Sequence[Point2D]
) -> set[int]:
    """Select the furthest-reaching precrack-connected component deterministically."""
    connected = [group for group in groups if group & left]
    if not connected:
        return set()
    return min(
        connected,
        key=lambda group: (
            -max(points[index][0] for index in group),
            -len(group),
            tuple(sorted(points[index] for index in group)),
            tuple(sorted(group)),
        ),
    )


def _dijkstra_length(
    source: int,
    targets: set[int],
    adjacency: dict[int, list[int]],
    points: Sequence[Sequence[float]],
) -> tuple[float, int | None]:
    queue = [(0.0, source)]
    distances = {source: 0.0}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        if current in targets:
            return distance, current
        x0, y0 = points[current][:2]
        for neighbour in adjacency[current]:
            x1, y1 = points[neighbour][:2]
            candidate = distance + math.hypot(x1 - x0, y1 - y0)
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return math.inf, None


def _dijkstra_path(
    source: int,
    targets: set[int],
    adjacency: dict[int, list[int]],
    points: Sequence[Point2D],
) -> tuple[float, tuple[int, ...]]:
    """Return a shortest path in ``O((V + E) log V)`` time.

    Queue and predecessor state are ``O(V + E)`` in the worst case and
    ``O(V)`` for the bounded-degree sparse finite-element graphs used here.
    """
    queue = [(0.0, points[source], source)]
    distances = {source: 0.0}
    previous: dict[int, int] = {}
    while queue:
        distance, _, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        x0, y0 = points[current]
        for neighbour in sorted(adjacency[current], key=lambda index: (points[index], index)):
            x1, y1 = points[neighbour]
            candidate_distance = distance + math.hypot(x1 - x0, y1 - y0)
            known_distance = distances.get(neighbour, math.inf)
            if candidate_distance < known_distance:
                distances[neighbour] = candidate_distance
                previous[neighbour] = current
                heapq.heappush(queue, (candidate_distance, points[neighbour], neighbour))
            elif candidate_distance == known_distance and neighbour != source:
                old_parent = previous.get(neighbour)
                candidate_parent_key = distance, points[current], current
                old_parent_key = (
                    (distances[old_parent], points[old_parent], old_parent)
                    if old_parent is not None
                    else (math.inf, (math.inf, math.inf), math.inf)
                )
                if candidate_parent_key < old_parent_key:
                    previous[neighbour] = current

    reachable_targets = [target for target in targets if target in distances]
    if not reachable_targets:
        return math.inf, ()
    target = min(
        reachable_targets,
        key=lambda index: (distances[index], points[index], index),
    )
    reverse_path = [target]
    seen = {target}
    while reverse_path[-1] != source:
        parent = previous.get(reverse_path[-1])
        if parent is None or parent in seen:
            return math.inf, ()
        reverse_path.append(parent)
        seen.add(parent)
    return distances[target], tuple(reversed(reverse_path))


def extract_main_crack_geometry(
    points: Sequence[Sequence[float]],
    edges: Iterable[tuple[int, int]],
    damage: Sequence[float],
    *,
    threshold: float = 0.7,
    left_x: float | None = None,
    boundary_tolerance: float = 0.0,
) -> EmbeddedCrackGeometry:
    """Extract a deterministic precrack-connected component and ordered backbone.

    A disconnected active nucleus is deliberately *not* promoted to the main
    crack.  In that case the function returns ``status='precrack_disconnected'``
    and structured empty geometry.
    """
    planar_points, planar_damage = _validated_planar_inputs(points, damage)
    threshold = _finite_float(threshold, name="threshold")
    boundary_tolerance = _finite_float(boundary_tolerance, name="boundary_tolerance")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    if boundary_tolerance < 0.0:
        raise ValueError("boundary_tolerance must be non-negative")
    if left_x is None:
        left_x = min((point[0] for point in planar_points), default=0.0)
    left_x = _finite_float(left_x, name="left_x")

    graph_edges = _normalise_edges(edges, len(planar_points))
    active = {index for index, value in enumerate(planar_damage) if value >= threshold}
    if not active:
        return EmbeddedCrackGeometry(
            status="empty",
            threshold=threshold,
            points=(),
            edges=(),
            backbone=(),
            backbone_length=None,
            branched=False,
            source=None,
            tip=None,
        )

    adjacency = {index: [] for index in active}
    for first, second in graph_edges:
        if first in active and second in active:
            adjacency[first].append(second)
            adjacency[second].append(first)
    groups = _components(adjacency, active)
    left = {index for index in active if planar_points[index][0] <= left_x + boundary_tolerance}
    main = _select_main_component(groups, left, planar_points)
    if not main:
        return EmbeddedCrackGeometry(
            status="precrack_disconnected",
            threshold=threshold,
            points=(),
            edges=(),
            backbone=(),
            backbone_length=None,
            branched=False,
            source=None,
            tip=None,
        )

    sources = left & main
    farthest_x = max(planar_points[index][0] for index in main)
    targets = {
        index
        for index in main
        if math.isclose(planar_points[index][0], farthest_x, rel_tol=0.0, abs_tol=1.0e-12)
    }
    candidates: list[tuple[float, tuple[Point2D, ...], tuple[int, ...]]] = []
    for source in sorted(sources, key=lambda index: (planar_points[index], index)):
        length, path = _dijkstra_path(source, targets, adjacency, planar_points)
        if path and math.isfinite(length):
            candidates.append((length, tuple(planar_points[index] for index in path), path))
    if not candidates:
        status = "backbone_unavailable"
        backbone: tuple[Point2D, ...] = ()
        backbone_length: float | None = None
        source_point: Point2D | None = None
        tip_point: Point2D | None = None
    else:
        backbone_length, backbone, path = min(candidates)
        source_point = planar_points[path[0]]
        tip_point = planar_points[path[-1]]
        status = "ok" if len(backbone) >= 2 and backbone_length > 0.0 else "degenerate_backbone"

    ordered_nodes = sorted(main, key=lambda index: (planar_points[index], index))
    local_index = {global_index: index for index, global_index in enumerate(ordered_nodes)}
    main_edges = tuple(
        sorted(
            (
                min(local_index[first], local_index[second]),
                max(local_index[first], local_index[second]),
            )
            for first, second in graph_edges
            if first in main and second in main
        )
    )
    branched = any(sum(neighbour in main for neighbour in adjacency[index]) > 2 for index in main)
    return EmbeddedCrackGeometry(
        status=status,
        threshold=threshold,
        points=tuple(planar_points[index] for index in ordered_nodes),
        edges=main_edges,
        backbone=backbone,
        backbone_length=backbone_length,
        branched=branched,
        source=source_point,
        tip=tip_point,
    )


def _normalise_polyline(path: Sequence[Sequence[float]], *, name: str) -> tuple[Point2D, ...]:
    result: list[Point2D] = []
    for index, value in enumerate(path):
        point = _validated_point(value, name=f"{name}[{index}]")
        if not result or point != result[-1]:
            result.append(point)
    if len(result) >= 2 and result[-1] < result[0]:
        result.reverse()
    return tuple(result)


def _polyline_length(path: Sequence[Point2D]) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(path, path[1:], strict=False)
    )


def _sample_polyline(
    path: Sequence[Point2D], spacing: float
) -> tuple[tuple[Point2D, ...], tuple[float, ...]]:
    segment_lengths = [
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(path, path[1:], strict=False)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 0.0:
        return (), ()
    if total_length / spacing > 1_000_000:
        raise ValueError("sample_spacing would create more than one million path samples")
    positions = [0.0]
    position = spacing
    while position < total_length:
        positions.append(position)
        position += spacing
    if math.isclose(positions[-1], total_length, rel_tol=0.0, abs_tol=1.0e-14 * total_length):
        positions[-1] = total_length
    else:
        positions.append(total_length)

    samples: list[Point2D] = []
    segment = 0
    segment_start = 0.0
    for position in positions:
        while (
            segment + 1 < len(segment_lengths)
            and position > segment_start + segment_lengths[segment]
        ):
            segment_start += segment_lengths[segment]
            segment += 1
        first, second = path[segment], path[segment + 1]
        fraction = (position - segment_start) / segment_lengths[segment]
        samples.append(
            (
                first[0] + fraction * (second[0] - first[0]),
                first[1] + fraction * (second[1] - first[1]),
            )
        )

    weights: list[float] = []
    for index, position in enumerate(positions):
        if index == 0:
            weight = (positions[1] - position) / 2.0
        elif index == len(positions) - 1:
            weight = (position - positions[index - 1]) / 2.0
        else:
            weight = (positions[index + 1] - positions[index - 1]) / 2.0
        weights.append(weight)
    return tuple(samples), tuple(weights)


def _point_to_polyline_distance(point: Point2D, path: Sequence[Point2D]) -> float:
    best = math.inf
    for first, second in zip(path, path[1:], strict=False):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            candidate = math.hypot(point[0] - first[0], point[1] - first[1])
        else:
            fraction = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / length_squared
            fraction = min(1.0, max(0.0, fraction))
            candidate = math.hypot(
                point[0] - (first[0] + fraction * dx),
                point[1] - (first[1] + fraction * dy),
            )
        best = min(best, candidate)
    return best


def _cross(first: Point2D, second: Point2D) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _boundary_parameters_for_segment(
    first: Point2D, second: Point2D, boundary_start: Point2D, boundary_end: Point2D
) -> tuple[float, ...]:
    path_vector = second[0] - first[0], second[1] - first[1]
    boundary_vector = (
        boundary_end[0] - boundary_start[0],
        boundary_end[1] - boundary_start[1],
    )
    offset = boundary_start[0] - first[0], boundary_start[1] - first[1]
    denominator = _cross(path_vector, boundary_vector)
    tolerance = 1.0e-12
    if abs(denominator) > tolerance * max(
        1.0,
        math.hypot(*path_vector) * math.hypot(*boundary_vector),
    ):
        path_fraction = _cross(offset, boundary_vector) / denominator
        boundary_fraction = _cross(offset, path_vector) / denominator
        if (
            -tolerance <= path_fraction <= 1.0 + tolerance
            and -tolerance <= boundary_fraction <= 1.0 + tolerance
        ):
            return (min(1.0, max(0.0, boundary_fraction)),)
        return ()

    if abs(_cross(offset, path_vector)) > tolerance * max(1.0, math.hypot(*path_vector)):
        return ()
    boundary_length_squared = (
        boundary_vector[0] * boundary_vector[0] + boundary_vector[1] * boundary_vector[1]
    )
    first_parameter = (
        (first[0] - boundary_start[0]) * boundary_vector[0]
        + (first[1] - boundary_start[1]) * boundary_vector[1]
    ) / boundary_length_squared
    second_parameter = (
        (second[0] - boundary_start[0]) * boundary_vector[0]
        + (second[1] - boundary_start[1]) * boundary_vector[1]
    ) / boundary_length_squared
    lower = max(0.0, min(first_parameter, second_parameter))
    upper = min(1.0, max(first_parameter, second_parameter))
    if lower > upper + tolerance:
        return ()
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=tolerance):
        return (min(1.0, max(0.0, (lower + upper) / 2.0)),)
    return min(1.0, max(0.0, lower)), min(1.0, max(0.0, upper))


def _boundary_exit_coordinates(
    path: Sequence[Point2D], boundary_start: Point2D, boundary_end: Point2D
) -> tuple[float, ...]:
    boundary_length = math.hypot(
        boundary_end[0] - boundary_start[0], boundary_end[1] - boundary_start[1]
    )
    parameters = sorted(
        parameter
        for first, second in zip(path, path[1:], strict=False)
        for parameter in _boundary_parameters_for_segment(
            first, second, boundary_start, boundary_end
        )
    )
    unique: list[float] = []
    for parameter in parameters:
        coordinate = parameter * boundary_length
        if not unique or not math.isclose(
            coordinate, unique[-1], rel_tol=0.0, abs_tol=1.0e-12 * max(1.0, boundary_length)
        ):
            unique.append(coordinate)
    return tuple(unique)


def _scalar_set_hausdorff(first: Sequence[float], second: Sequence[float]) -> float:
    return max(
        max(min(abs(left - right) for right in second) for left in first),
        max(min(abs(right - left) for left in first) for right in second),
    )


def _null_path_comparison(
    *,
    status: str,
    observed_status: str,
    reference_status: str,
    observed_branched: bool | None,
    reference_branched: bool | None,
    sample_spacing: float,
    coverage_radius: float,
) -> PathComparisonMetrics:
    return PathComparisonMetrics(
        status=status,
        observed_geometry_status=observed_status,
        reference_geometry_status=reference_status,
        observed_branched=observed_branched,
        reference_branched=reference_branched,
        sample_spacing=sample_spacing,
        coverage_radius=coverage_radius,
        directed_mean_observed_to_reference=None,
        directed_mean_reference_to_observed=None,
        symmetric_chamfer=None,
        sampled_directed_hausdorff_observed_to_reference=None,
        sampled_directed_hausdorff_reference_to_observed=None,
        sampled_symmetric_hausdorff=None,
        reference_coverage=None,
        observed_precision=None,
        exit_status="not_evaluated",
        observed_exit_coordinates=(),
        reference_exit_coordinates=(),
        exit_error=None,
    )


def compare_crack_paths(
    observed: EmbeddedCrackGeometry | Sequence[Sequence[float]],
    reference: EmbeddedCrackGeometry | Sequence[Sequence[float]],
    *,
    sample_spacing: float,
    coverage_radius: float,
    exit_boundary: tuple[Sequence[float], Sequence[float]],
) -> PathComparisonMetrics:
    """Compare ordered crack paths without weighting dense meshes more heavily.

    Symmetric Chamfer is the mean of the two arclength-weighted directed mean
    distances.  Sampled symmetric Hausdorff is the maximum
    sampled-source-to-segment distance in either direction and must be checked
    for convergence under a smaller spacing.  Coverage and precision use the frozen
    physical ``coverage_radius`` rather than a mesh-dependent nodal radius.
    """
    sample_spacing = _finite_float(sample_spacing, name="sample_spacing")
    coverage_radius = _finite_float(coverage_radius, name="coverage_radius")
    if sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive")
    if coverage_radius < 0.0:
        raise ValueError("coverage_radius must be non-negative")
    if len(exit_boundary) != 2:
        raise ValueError("exit_boundary must contain exactly two endpoints")
    boundary_start = _validated_point(exit_boundary[0], name="exit_boundary[0]")
    boundary_end = _validated_point(exit_boundary[1], name="exit_boundary[1]")
    if boundary_start == boundary_end:
        raise ValueError("exit_boundary endpoints must be distinct")

    def unpack(
        value: EmbeddedCrackGeometry | Sequence[Sequence[float]], *, name: str
    ) -> tuple[tuple[Point2D, ...], str, bool | None]:
        if isinstance(value, EmbeddedCrackGeometry):
            path = _normalise_polyline(value.backbone, name=name)
            status = value.status
            if status == "ok" and not path:
                status = "empty"
            elif status == "ok" and (len(path) < 2 or _polyline_length(path) <= 0.0):
                status = "degenerate_backbone"
            return path, status, value.branched
        path = _normalise_polyline(value, name=name)
        if not path:
            return path, "empty", None
        if len(path) < 2 or _polyline_length(path) <= 0.0:
            return path, "degenerate_backbone", None
        return path, "ok", None

    observed_path, observed_status, observed_branched = unpack(observed, name="observed")
    reference_path, reference_status, reference_branched = unpack(reference, name="reference")
    if observed_status != "ok":
        return _null_path_comparison(
            status=f"observed_{observed_status}",
            observed_status=observed_status,
            reference_status=reference_status,
            observed_branched=observed_branched,
            reference_branched=reference_branched,
            sample_spacing=sample_spacing,
            coverage_radius=coverage_radius,
        )
    if reference_status != "ok":
        return _null_path_comparison(
            status=f"reference_{reference_status}",
            observed_status=observed_status,
            reference_status=reference_status,
            observed_branched=observed_branched,
            reference_branched=reference_branched,
            sample_spacing=sample_spacing,
            coverage_radius=coverage_radius,
        )

    observed_samples, observed_weights = _sample_polyline(observed_path, sample_spacing)
    reference_samples, reference_weights = _sample_polyline(reference_path, sample_spacing)
    observed_distances = tuple(
        _point_to_polyline_distance(point, reference_path) for point in observed_samples
    )
    reference_distances = tuple(
        _point_to_polyline_distance(point, observed_path) for point in reference_samples
    )
    observed_weight = sum(observed_weights)
    reference_weight = sum(reference_weights)
    mean_observed_to_reference = (
        sum(
            weight * distance
            for weight, distance in zip(observed_weights, observed_distances, strict=True)
        )
        / observed_weight
    )
    mean_reference_to_observed = (
        sum(
            weight * distance
            for weight, distance in zip(reference_weights, reference_distances, strict=True)
        )
        / reference_weight
    )
    coordinate_scale = max(
        1.0,
        coverage_radius,
        *(abs(coordinate) for point in (*observed_path, *reference_path) for coordinate in point),
    )
    coverage_slack = 64.0 * math.ulp(coordinate_scale)
    observed_precision = (
        sum(
            weight
            for weight, distance in zip(observed_weights, observed_distances, strict=True)
            if distance <= coverage_radius + coverage_slack
        )
        / observed_weight
    )
    reference_coverage = (
        sum(
            weight
            for weight, distance in zip(reference_weights, reference_distances, strict=True)
            if distance <= coverage_radius + coverage_slack
        )
        / reference_weight
    )

    observed_exits = _boundary_exit_coordinates(observed_path, boundary_start, boundary_end)
    reference_exits = _boundary_exit_coordinates(reference_path, boundary_start, boundary_end)
    if observed_branched is True or reference_branched is True:
        exit_status, exit_error = "branched_not_evaluated", None
    elif not reference_exits and not observed_exits:
        exit_status, exit_error = "both_right_censored", None
    elif reference_exits and not observed_exits:
        exit_status, exit_error = "observed_right_censored", None
    elif observed_exits and not reference_exits:
        exit_status, exit_error = "reference_right_censored", None
    elif len(observed_exits) == len(reference_exits) == 1:
        exit_status = "matched_unique_exit"
        exit_error = abs(observed_exits[0] - reference_exits[0])
    else:
        exit_status = "multiple_or_branched_exit"
        exit_error = _scalar_set_hausdorff(observed_exits, reference_exits)

    return PathComparisonMetrics(
        status="ok",
        observed_geometry_status=observed_status,
        reference_geometry_status=reference_status,
        observed_branched=observed_branched,
        reference_branched=reference_branched,
        sample_spacing=sample_spacing,
        coverage_radius=coverage_radius,
        directed_mean_observed_to_reference=mean_observed_to_reference,
        directed_mean_reference_to_observed=mean_reference_to_observed,
        symmetric_chamfer=(mean_observed_to_reference + mean_reference_to_observed) / 2.0,
        sampled_directed_hausdorff_observed_to_reference=max(observed_distances),
        sampled_directed_hausdorff_reference_to_observed=max(reference_distances),
        sampled_symmetric_hausdorff=max(max(observed_distances), max(reference_distances)),
        reference_coverage=reference_coverage,
        observed_precision=observed_precision,
        exit_status=exit_status,
        observed_exit_coordinates=observed_exits,
        reference_exit_coordinates=reference_exits,
        exit_error=exit_error,
    )


def analyse_crack_graph(
    points: Sequence[Sequence[float]],
    edges: Iterable[tuple[int, int]],
    damage: Sequence[float],
    *,
    threshold: float = 0.7,
    left_x: float | None = None,
    right_x: float | None = None,
    boundary_tolerance: float = 0.0,
) -> CrackGraphMetrics:
    """Threshold nodal damage and analyse the induced active subgraph ``G_crack``.

    Unlike a column-wise ``y(x)`` reduction, this representation supports
    branching, merging and disconnected crack nuclei.
    """
    planar_points, planar_damage = _validated_planar_inputs(points, damage)
    threshold = _finite_float(threshold, name="threshold")
    boundary_tolerance = _finite_float(boundary_tolerance, name="boundary_tolerance")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    if boundary_tolerance < 0.0:
        raise ValueError("boundary_tolerance must be non-negative")
    n = len(planar_points)
    graph_edges = _normalise_edges(edges, n)
    active = {i for i, value in enumerate(planar_damage) if value >= threshold}
    adjacency = {i: [] for i in active}
    active_edges = 0
    for i, j in graph_edges:
        if i in active and j in active:
            adjacency[i].append(j)
            adjacency[j].append(i)
            active_edges += 1

    groups = _components(adjacency, active)
    largest = max(groups, key=len, default=set())

    if left_x is None:
        left_x = min((point[0] for point in planar_points), default=0.0)
    left_x = _finite_float(left_x, name="left_x")
    if right_x is None:
        right_x = max((point[0] for point in planar_points), default=0.0)
    right_x = _finite_float(right_x, name="right_x")
    left = {i for i in active if planar_points[i][0] <= left_x + boundary_tolerance}
    right = {i for i in active if planar_points[i][0] >= right_x - boundary_tolerance}

    main = _select_main_component(groups, left, planar_points)
    tip_x = max((planar_points[i][0] for i in main), default=None)

    spans = bool(main & right)
    best_length = math.inf
    best_pair: tuple[int, int] | None = None
    sources = left & main
    if sources:
        farthest_x = max(planar_points[i][0] for i in main)
        targets = {
            i
            for i in main
            if math.isclose(planar_points[i][0], farthest_x, rel_tol=0.0, abs_tol=1.0e-12)
        }
        for source in sorted(sources, key=lambda index: (planar_points[index], index)):
            length, target = _dijkstra_length(source, targets, adjacency, planar_points)
            if length < best_length and target is not None:
                best_length = length
                best_pair = source, target

    path_length: float | None = None
    tortuosity: float | None = None
    if best_pair is not None and math.isfinite(best_length):
        path_length = best_length
        a, b = best_pair
        direct = math.hypot(
            planar_points[b][0] - planar_points[a][0],
            planar_points[b][1] - planar_points[a][1],
        )
        tortuosity = best_length / direct if direct > 0.0 else None

    return CrackGraphMetrics(
        threshold=threshold,
        active_nodes=len(active),
        active_edges=active_edges,
        components=len(groups),
        largest_component_nodes=len(largest),
        main_component_nodes=len(main),
        tip_x=tip_x,
        spans_left_to_right=spans,
        path_length=path_length,
        tortuosity=tortuosity,
    )


def analyse_interface_interaction(
    points: Sequence[Sequence[float]],
    edges: Iterable[tuple[int, int]],
    damage: Sequence[float],
    *,
    interface_start: Sequence[float],
    interface_end: Sequence[float],
    impact_point: Sequence[float],
    threshold: float = 0.7,
    impact_tolerance: float,
    interface_corridor_width: float,
    impact_exclusion_radius: float,
    confirmation_distance: float,
    penetration_corridor_half_height: float | None = None,
    left_x: float | None = None,
    boundary_tolerance: float = 0.0,
) -> InterfaceInteractionMetrics:
    """Classify post-impact advances of the left-connected thresholded crack.

    The classification deliberately uses cautious ``candidate`` language.  A
    publishable penetration/deflection statement still requires mesh-diagonal,
    angle, threshold and refinement controls.
    """
    planar_points, planar_damage = _validated_planar_inputs(points, damage)
    threshold = _finite_float(threshold, name="threshold")
    impact_tolerance = _finite_float(impact_tolerance, name="impact_tolerance")
    interface_corridor_width = _finite_float(
        interface_corridor_width, name="interface_corridor_width"
    )
    impact_exclusion_radius = _finite_float(impact_exclusion_radius, name="impact_exclusion_radius")
    confirmation_distance = _finite_float(confirmation_distance, name="confirmation_distance")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    if (
        impact_tolerance <= 0.0
        or interface_corridor_width <= 0.0
        or impact_exclusion_radius < 0.0
        or confirmation_distance <= 0.0
    ):
        raise ValueError("interface protocol distances are invalid")
    if penetration_corridor_half_height is None:
        penetration_corridor_half_height = interface_corridor_width
    penetration_corridor_half_height = _finite_float(
        penetration_corridor_half_height, name="penetration_corridor_half_height"
    )
    if penetration_corridor_half_height <= 0.0:
        raise ValueError("penetration_corridor_half_height must be positive")

    ax, ay = _validated_point(interface_start, name="interface_start")
    bx, by = _validated_point(interface_end, name="interface_end")
    tangent_x, tangent_y = bx - ax, by - ay
    interface_length = math.hypot(tangent_x, tangent_y)
    if interface_length <= 0.0:
        raise ValueError("interface endpoints must be distinct")
    tangent_x /= interface_length
    tangent_y /= interface_length
    if tangent_x < 0.0 or (math.isclose(tangent_x, 0.0) and tangent_y < 0.0):
        tangent_x, tangent_y = -tangent_x, -tangent_y
    normal_x, normal_y = -tangent_y, tangent_x
    if normal_x < 0.0:
        normal_x, normal_y = -normal_x, -normal_y
    impact_x, impact_y = _validated_point(impact_point, name="impact_point")

    graph_edges = _normalise_edges(edges, len(planar_points))
    active = {index for index, value in enumerate(planar_damage) if value >= threshold}
    adjacency = {index: [] for index in active}
    for first, second in graph_edges:
        if first in active and second in active:
            adjacency[first].append(second)
            adjacency[second].append(first)
    groups = _components(adjacency, active)
    if left_x is None:
        left_x = min((point[0] for point in planar_points), default=0.0)
    left_x = _finite_float(left_x, name="left_x")
    boundary_tolerance = _finite_float(boundary_tolerance, name="boundary_tolerance")
    if boundary_tolerance < 0.0:
        raise ValueError("boundary_tolerance must be non-negative")
    left = {index for index in active if planar_points[index][0] <= left_x + boundary_tolerance}
    main = _select_main_component(groups, left, planar_points)

    closest_to_impact = min(
        (
            math.hypot(
                planar_points[index][0] - impact_x,
                planar_points[index][1] - impact_y,
            )
            for index in main
        ),
        default=None,
    )
    reached_interface = closest_to_impact is not None and closest_to_impact <= impact_tolerance
    interface_advance = 0.0
    penetration_advance = 0.0
    for index in main:
        dx = planar_points[index][0] - impact_x
        dy = planar_points[index][1] - impact_y
        tangent_coordinate = dx * tangent_x + dy * tangent_y
        normal_coordinate = dx * normal_x + dy * normal_y
        if (
            tangent_coordinate > impact_exclusion_radius
            and abs(normal_coordinate) <= interface_corridor_width
        ):
            interface_advance = max(interface_advance, tangent_coordinate)
        if (
            dx > impact_exclusion_radius
            and normal_coordinate > interface_corridor_width
            and abs(dy) <= penetration_corridor_half_height
        ):
            penetration_advance = max(penetration_advance, dx)

    interface_edge_length = 0.0
    penetration_edge_length = 0.0
    for first, second in graph_edges:
        if first not in main or second not in main:
            continue
        x0, y0 = planar_points[first]
        x1, y1 = planar_points[second]
        midpoint_x, midpoint_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = midpoint_x - impact_x, midpoint_y - impact_y
        tangent_coordinate = dx * tangent_x + dy * tangent_y
        normal_coordinate = dx * normal_x + dy * normal_y
        edge_length = math.hypot(x1 - x0, y1 - y0)
        if (
            tangent_coordinate > impact_exclusion_radius
            and abs(normal_coordinate) <= interface_corridor_width
        ):
            interface_edge_length += edge_length
        if (
            dx > impact_exclusion_radius
            and normal_coordinate > interface_corridor_width
            and abs(dy) <= penetration_corridor_half_height
        ):
            penetration_edge_length += edge_length

    deflection_confirmed = interface_advance >= confirmation_distance
    penetration_confirmed = penetration_advance >= confirmation_distance
    if not reached_interface:
        classification = "pre_impact_right_censored"
    elif not deflection_confirmed and not penetration_confirmed:
        classification = "arrested_or_unresolved"
    elif deflection_confirmed and not penetration_confirmed:
        classification = "deflection_candidate"
    elif penetration_confirmed and not deflection_confirmed:
        classification = "penetration_candidate"
    else:
        classification = "mixed_or_branched"

    return InterfaceInteractionMetrics(
        threshold=threshold,
        impact_tolerance=impact_tolerance,
        interface_corridor_width=interface_corridor_width,
        impact_exclusion_radius=impact_exclusion_radius,
        confirmation_distance=confirmation_distance,
        penetration_corridor_half_height=penetration_corridor_half_height,
        main_component_nodes=len(main),
        reached_interface=reached_interface,
        closest_main_node_to_impact=closest_to_impact,
        interface_forward_advance=interface_advance,
        penetration_forward_advance=penetration_advance,
        interface_active_edge_length=interface_edge_length,
        penetration_active_edge_length=penetration_edge_length,
        geometric_classification=classification,
    )
