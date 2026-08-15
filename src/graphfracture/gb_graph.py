"""Grain-boundary segment graph and its continuum-field mapping.

This is deliberately a junction/segment graph, not the grain-centroid graph
used by the legacy path predictor.  Its embedded edges represent physical
boundary segments, so their distance to a finite-element integration point has
a clear geometric meaning.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from .config import RunConfig


@dataclass(frozen=True)
class BoundaryNode:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class BoundaryEdge:
    source: str
    target: str
    toughness_ratio: float = 0.5
    hydrogen_diffusivity_ratio: float = 1.0
    trap_density: float = 0.0


@dataclass(frozen=True)
class PathResult:
    nodes: tuple[str, ...]
    edges: tuple[BoundaryEdge, ...]
    cost: float
    geometric_length: float


class BoundaryField(Protocol):
    """Structural interface consumed by the DOLFINx material-field mapper."""

    influence_radius: float

    def toughness_ratio_at(self, x: np.ndarray) -> np.ndarray: ...

    def diffusivity_ratio_at(self, x: np.ndarray) -> np.ndarray: ...

    def trap_density_at(self, x: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict[str, object]: ...


class BoundaryGraph:
    """An attributed, spatially embedded grain-boundary graph ``G_GB``."""

    def __init__(
        self,
        nodes: Iterable[BoundaryNode],
        edges: Iterable[BoundaryEdge],
        influence_radius: float,
    ) -> None:
        self.nodes = {node.name: node for node in nodes}
        self.edges = tuple(edges)
        self.influence_radius = float(influence_radius)
        self._validate()

    @classmethod
    def from_config(cls, config: RunConfig) -> BoundaryGraph:
        nodes = (BoundaryNode(node.name, *node.point) for node in config.graph_nodes)
        edges = (
            BoundaryEdge(
                source=edge.source,
                target=edge.target,
                toughness_ratio=edge.toughness_ratio,
                hydrogen_diffusivity_ratio=edge.hydrogen_diffusivity_ratio,
                trap_density=edge.trap_density,
            )
            for edge in config.graph_edges
        )
        return cls(nodes, edges, config.graph.influence_radius)

    def _validate(self) -> None:
        if self.influence_radius <= 0:
            raise ValueError("influence_radius must be positive")
        if not self.nodes:
            raise ValueError("a boundary graph needs at least one node")
        seen_edges: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"edge {edge.source!r}->{edge.target!r} has an unknown node")
            if edge.source == edge.target:
                raise ValueError("self-loops are not supported")
            edge_key = tuple(sorted((edge.source, edge.target)))
            if edge_key in seen_edges:
                raise ValueError(f"duplicate undirected edge {edge.source!r}<->{edge.target!r}")
            seen_edges.add(edge_key)
            if not self.edge_length(edge) > 0.0:
                raise ValueError("edge length must be positive")
            if not 0.0 < edge.toughness_ratio <= 1.0:
                raise ValueError("toughness_ratio must lie in (0, 1]")
            if not edge.hydrogen_diffusivity_ratio >= 1.0:
                raise ValueError("hydrogen_diffusivity_ratio must be at least 1")
            if not edge.trap_density >= 0.0:
                raise ValueError("trap_density must be non-negative")

    def edge_length(self, edge: BoundaryEdge) -> float:
        a, b = self.nodes[edge.source], self.nodes[edge.target]
        return math.hypot(a.x - b.x, a.y - b.y)

    def edge_cost(self, edge: BoundaryEdge) -> float:
        """Normalised crack-path cost with the necessary physical-length factor.

        The omitted bulk ``Gc`` is common to every edge and therefore does not
        change the shortest path. Multiply by bulk ``Gc`` when a dimensional
        energy-per-thickness proxy is required.
        """
        return self.edge_length(edge) * edge.toughness_ratio

    def shortest_path(self, source: str, target: str) -> PathResult:
        """Return the minimum normalised integrated-toughness route in ``G_GB``."""
        return self.shortest_path_between((source,), (target,))

    def shortest_path_between(self, sources: Iterable[str], targets: Iterable[str]) -> PathResult:
        """Return the cheapest route between two non-empty node sets.

        A multi-source/multi-target search avoids making a path diagnostic
        depend on an arbitrary single extreme node when a boundary has several
        entrances or exits.
        """
        source_names = tuple(dict.fromkeys(sources))
        target_names = frozenset(targets)
        if not source_names or not target_names:
            raise ValueError("sources and targets must both be non-empty")
        unknown = (set(source_names) | target_names) - self.nodes.keys()
        if unknown:
            raise KeyError(f"unknown graph nodes: {', '.join(sorted(unknown))}")

        adjacency: dict[str, list[tuple[str, BoundaryEdge, float]]] = {
            name: [] for name in self.nodes
        }
        for edge in self.edges:
            cost = self.edge_cost(edge)
            adjacency[edge.source].append((edge.target, edge, cost))
            adjacency[edge.target].append((edge.source, edge, cost))

        queue: list[tuple[float, str]] = [(0.0, source) for source in source_names]
        heapq.heapify(queue)
        distance = {source: 0.0 for source in source_names}
        previous: dict[str, tuple[str, BoundaryEdge]] = {}
        reached_target: str | None = None
        while queue:
            cost, current = heapq.heappop(queue)
            if cost != distance.get(current):
                continue
            if current in target_names:
                reached_target = current
                break
            for neighbour, edge, edge_cost in adjacency[current]:
                candidate = cost + edge_cost
                if candidate < distance.get(neighbour, math.inf):
                    distance[neighbour] = candidate
                    previous[neighbour] = (current, edge)
                    heapq.heappush(queue, (candidate, neighbour))

        if reached_target is None:
            return PathResult((), (), math.inf, math.inf)

        node_path = [reached_target]
        edge_path: list[BoundaryEdge] = []
        current = reached_target
        source_set = set(source_names)
        while current not in source_set:
            parent, edge = previous[current]
            node_path.append(parent)
            edge_path.append(edge)
            current = parent
        node_path.reverse()
        edge_path.reverse()
        return PathResult(
            tuple(node_path),
            tuple(edge_path),
            distance[reached_target],
            sum(self.edge_length(edge) for edge in edge_path),
        )

    def _gaussian_decay_at(self, points: np.ndarray, edge: BoundaryEdge) -> np.ndarray:
        """Return the edge-centred Gaussian decay at physical points."""
        import numpy as np

        a_node, b_node = self.nodes[edge.source], self.nodes[edge.target]
        a = np.array([a_node.x, a_node.y])
        b = np.array([b_node.x, b_node.y])
        segment = b - a
        length2 = float(segment @ segment)
        projection = np.clip(((points - a) @ segment) / length2, 0.0, 1.0)
        closest = a + projection[:, None] * segment
        distance2 = np.sum((points - closest) ** 2, axis=1)
        return np.exp(-distance2 / self.influence_radius**2)

    def toughness_ratio_at(self, x: np.ndarray) -> np.ndarray:
        """Map embedded boundary segments to a mesh-independent toughness band.

        Parameters
        ----------
        x:
            Coordinate array with DOLFINx interpolation layout ``(gdim, n)``.

        Returns
        -------
        numpy.ndarray
            A ratio in ``(0, 1]``.  Every edge has a fixed *physical* influence
            radius; refinement therefore does not turn a boundary into an
            arbitrarily thinner one-cell strip.
        """
        import numpy as np

        points = np.asarray(x[:2], dtype=float).T
        ratio = np.ones(points.shape[0], dtype=float)
        for edge in self.edges:
            # Smooth Gaussian band: edge value is exact at its centreline and
            # approaches the bulk value continuously away from the segment.
            decay = self._gaussian_decay_at(points, edge)
            candidate = 1.0 - (1.0 - edge.toughness_ratio) * decay
            ratio = np.minimum(ratio, candidate)
        return ratio

    def diffusivity_ratio_at(self, x: np.ndarray) -> np.ndarray:
        """Map edge diffusivity ratios to Gaussian bands with a bulk value of one."""
        import numpy as np

        points = np.asarray(x[:2], dtype=float).T
        ratio = np.ones(points.shape[0], dtype=float)
        for edge in self.edges:
            decay = self._gaussian_decay_at(points, edge)
            candidate = 1.0 + (edge.hydrogen_diffusivity_ratio - 1.0) * decay
            ratio = np.maximum(ratio, candidate)
        return ratio

    def trap_density_at(self, x: np.ndarray) -> np.ndarray:
        """Map additive edge trap densities to Gaussian boundary bands."""
        import numpy as np

        points = np.asarray(x[:2], dtype=float).T
        density = np.zeros(points.shape[0], dtype=float)
        for edge in self.edges:
            density += edge.trap_density * self._gaussian_decay_at(points, edge)
        return density

    def describe(self) -> dict[str, object]:
        return {
            "kind": "G_GB",
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "influence_radius": self.influence_radius,
            "minimum_edge_toughness_ratio": min(
                (edge.toughness_ratio for edge in self.edges), default=1.0
            ),
            "maximum_edge_hydrogen_diffusivity_ratio": max(
                (edge.hydrogen_diffusivity_ratio for edge in self.edges), default=1.0
            ),
            "sum_edge_trap_density_parameters": sum(edge.trap_density for edge in self.edges),
            "field_combination_rules": {
                "toughness": "minimum Gaussian edge ratio",
                "diffusivity": "maximum Gaussian edge ratio",
                "trap_density": "sum of Gaussian edge contributions",
            },
        }


def boundary_field_from_config(config: RunConfig) -> BoundaryField:
    """Return the configured boundary field provider.

    A ``graph.chain_artifact`` selects the measured EBSD polyline field
    (:class:`graphfracture.chain_field.ChainBoundaryField`); otherwise the
    synthetic node/edge :class:`BoundaryGraph` is built.  Both expose the same
    ``toughness_ratio_at`` / ``diffusivity_ratio_at`` / ``trap_density_at`` /
    ``describe`` interface consumed by the solver.
    """
    if config.graph.chain_artifact.strip():
        from .chain_field import ChainBoundaryField

        return ChainBoundaryField.from_config(config)
    return BoundaryGraph.from_config(config)
