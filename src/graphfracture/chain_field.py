"""Map an embedded EBSD grain-boundary polyline artifact to continuum fields.

This is the finite-element counterpart of :class:`graphfracture.gb_graph.BoundaryGraph`
for a *measured* microstructure rather than a hand-written synthetic graph.  It
consumes the auditable :class:`inverse_pfm.gb_chains.GrainBoundaryChains`
artifact (pixel-dual boundary polylines with per-chain toughness / diffusivity /
trap attributes) and evaluates the same fixed-physical-width band maps

.. math::

    G_{c0}(x)/G_c^{bulk} = \\min_c [1-(1-r_c)\\exp(-\\operatorname{dist}(x,\\Gamma_c)^2/b^2)],\\\\
    D_H(x)/D_L           = \\max_c [1+(m_c-1)\\exp(-\\operatorname{dist}(x,\\Gamma_c)^2/b^2)],\\\\
    N_T(x)               = \\sum_c N_{T,c}\\exp(-\\operatorname{dist}(x,\\Gamma_c)^2/b^2),

where :math:`c` indexes a parent polyline and its distance is the minimum true
point-to-segment distance over that polyline.  Redividing the same polyline into
more atomic segments therefore cannot increase its trap contribution.  The real
``ban`` ROI has ~15k atomic segments, so a naive all-pairs evaluation does not
scale.  A pure-``numpy`` uniform-grid segment index and bounded point/segment
chunks restrict work to the explicit Gaussian cutoff
``exp(-d^2/b^2) >= eps``.  The omitted-field bound is ``eps`` times the relevant
attribute amplitude (or the sum of parent-chain trap parameters).

The band width ``b`` is a *fixed physical length*: refining the mesh never turns
a boundary into a thinner one-cell strip.  ``numpy`` only; no SciPy.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .config import RunConfig

# A boundary segment farther than ``b * sqrt(-ln TRUNCATION)`` contributes less
# than ``TRUNCATION`` to any Gaussian band and is safely ignored by the index.
_TRUNCATION = 1.0e-9
_DEFAULT_MAX_DISTANCE_PAIRS = 262_144
_MAX_POINTS_PER_CHUNK = 256


class ChainBoundaryField:
    """Continuum band fields from measured grain-boundary polyline segments.

    Parameters
    ----------
    segment_a, segment_b:
        ``(m, 2)`` physical endpoints of the ``m`` atomic boundary segments.
    toughness_ratio, diffusivity_ratio, trap_density:
        Per-segment attributes (each ``(m,)``) inherited from the parent chain.
        Toughness lies in ``(0, 1]``, diffusivity is ``>= 1`` and trap density is
        ``>= 0``.
    influence_radius:
        Fixed physical Gaussian band width ``b`` (> 0).
    metadata:
        Optional provenance carried into :meth:`describe`.
    segment_chain_id:
        Parent-polyline id of every segment.  If omitted, every segment is
        treated as its own parent chain.  Attributes must be constant within a
        parent chain.
    max_distance_pairs:
        Hard upper bound on the point--segment pairs in one distance work array.
    """

    def __init__(
        self,
        segment_a: np.ndarray,
        segment_b: np.ndarray,
        toughness_ratio: np.ndarray,
        diffusivity_ratio: np.ndarray,
        trap_density: np.ndarray,
        influence_radius: float,
        *,
        segment_chain_id: np.ndarray | None = None,
        max_distance_pairs: int = _DEFAULT_MAX_DISTANCE_PAIRS,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        a = np.ascontiguousarray(segment_a, dtype=float)
        b = np.ascontiguousarray(segment_b, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape:
            raise ValueError("segment endpoints must both have shape (m, 2)")
        m = a.shape[0]
        if m == 0:
            raise ValueError("a boundary field needs at least one segment")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("segment endpoints must be finite")
        tough = np.ascontiguousarray(toughness_ratio, dtype=float)
        diff = np.ascontiguousarray(diffusivity_ratio, dtype=float)
        trap = np.ascontiguousarray(trap_density, dtype=float)
        for name, values in (
            ("toughness_ratio", tough),
            ("diffusivity_ratio", diff),
            ("trap_density", trap),
        ):
            if values.shape != (m,):
                raise ValueError(f"{name} must have shape ({m},)")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
        if np.any((tough <= 0.0) | (tough > 1.0)):
            raise ValueError("toughness_ratio must lie in (0, 1]")
        if np.any(diff < 1.0):
            raise ValueError("diffusivity_ratio must be at least 1")
        if np.any(trap < 0.0):
            raise ValueError("trap_density must be non-negative")
        if (
            isinstance(influence_radius, (bool, np.bool_))
            or not isinstance(influence_radius, Real)
            or not math.isfinite(float(influence_radius))
            or influence_radius <= 0
        ):
            raise ValueError("influence_radius must be a positive finite real number")
        if (
            isinstance(max_distance_pairs, (bool, np.bool_))
            or not isinstance(max_distance_pairs, Integral)
            or max_distance_pairs < 1
        ):
            raise ValueError("max_distance_pairs must be a positive integer")
        seg_lengths = np.linalg.norm(b - a, axis=1)
        if np.any(seg_lengths <= 0.0):
            raise ValueError("segments must have positive length")

        if segment_chain_id is None:
            parent = np.arange(m, dtype=np.int64)
        else:
            parent_raw = np.asarray(segment_chain_id)
            if parent_raw.dtype.hasobject or not np.issubdtype(parent_raw.dtype, np.integer):
                raise ValueError("segment_chain_id must have an integer dtype")
            if parent_raw.shape != (m,):
                raise ValueError(f"segment_chain_id must have shape ({m},)")
            if np.any(parent_raw < 0):
                raise ValueError("segment_chain_id must be non-negative")
            parent = np.asarray(parent_raw, dtype=np.int64)
        source_chain_ids, first, dense_parent = np.unique(
            parent, return_index=True, return_inverse=True
        )
        for name, values in (
            ("toughness_ratio", tough),
            ("diffusivity_ratio", diff),
            ("trap_density", trap),
        ):
            chain_values = values[first]
            if not np.array_equal(values, chain_values[dense_parent]):
                raise ValueError(f"{name} must be constant within each parent chain")

        self._a = a
        self._ab = b - a
        self._len2 = np.einsum("ij,ij->i", self._ab, self._ab)
        self._tough = tough
        self._diff = diff
        self._trap = trap
        self._segment_chain = np.ascontiguousarray(dense_parent, dtype=np.int64)
        self._source_chain_ids = np.ascontiguousarray(source_chain_ids, dtype=np.int64)
        self._chain_tough = np.ascontiguousarray(tough[first])
        self._chain_diff = np.ascontiguousarray(diff[first])
        self._chain_trap = np.ascontiguousarray(trap[first])
        self.influence_radius = float(influence_radius)
        self.metadata = dict(metadata or {})
        self._segment_lengths = seg_lengths
        self.max_distance_pairs = int(max_distance_pairs)
        self._max_distance_pairs_observed = 0

        # Gaussian truncation radius plus the longest segment half-length: any
        # segment whose *midpoint* is farther than this from a query point is
        # below the truncation tolerance everywhere on the segment.
        self._trunc_radius = self.influence_radius * math.sqrt(-math.log(_TRUNCATION))
        self._query_radius = self._trunc_radius + 0.5 * float(np.max(seg_lengths))
        self._build_index()
        self._cache: tuple[bytes, np.ndarray, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_chains(
        cls,
        chains: Any,
        influence_radius: float,
        *,
        confidence_floor: float = 0.0,
        attribute_permutation_seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChainBoundaryField:
        """Flatten a :class:`GrainBoundaryChains` artifact to a segment field.

        ``confidence_floor`` drops chains whose geometry confidence (fraction of
        the chain with both adjacent EBSD pixels indexed) is below the floor, so
        filled-label boundaries can be excluded from a trusted-geometry study.

        ``attribute_permutation_seed`` performs an *attribute--location
        permutation control*.  After confidence filtering, the joint per-chain
        attribute rows of the retained chains are permuted with a fixed PCG64
        seed.  This preserves their exact joint distribution while destroying
        its assignment to retained boundary locations.  It does not reconnect
        or otherwise alter the segment topology or geometry.
        """
        n_chains = int(chains.n_chains)
        tough_chain = _chain_attribute(chains.toughness_ratio, n_chains, 1.0, "toughness_ratio")
        diff_chain = _chain_attribute(chains.diffusivity_ratio, n_chains, 1.0, "diffusivity_ratio")
        trap_chain = _chain_attribute(chains.trap_density, n_chains, 0.0, "trap_density")

        confidence = np.asarray(chains.confidence_fraction, dtype=float)
        if (
            isinstance(confidence_floor, (bool, np.bool_))
            or not isinstance(confidence_floor, Real)
            or not math.isfinite(float(confidence_floor))
            or not 0.0 <= confidence_floor <= 1.0
        ):
            raise ValueError("confidence_floor must be a finite real number in [0, 1]")
        keep_chain = confidence >= confidence_floor
        if not np.any(keep_chain):
            raise ValueError("confidence_floor excluded every chain")

        if attribute_permutation_seed is not None and (
            isinstance(attribute_permutation_seed, (bool, np.bool_))
            or not isinstance(attribute_permutation_seed, Integral)
            or attribute_permutation_seed < 0
        ):
            raise ValueError("attribute_permutation_seed must be a non-negative integer or None")
        kept_source_ids = np.flatnonzero(keep_chain).astype(np.int64)
        permuted = attribute_permutation_seed is not None
        if permuted:
            generator = np.random.Generator(np.random.PCG64(int(attribute_permutation_seed)))
            order = generator.permutation(kept_source_ids.size)
        else:
            order = np.arange(kept_source_ids.size, dtype=np.int64)
        assigned_source_ids = kept_source_ids[order]
        tough_kept = tough_chain[assigned_source_ids]
        diff_kept = diff_chain[assigned_source_ids]
        trap_kept = trap_chain[assigned_source_ids]
        permutation_hash = _permutation_sha256(assigned_source_ids)

        points = np.asarray(chains.points, dtype=float)
        offsets = np.asarray(chains.chain_offsets)
        point_ids = np.asarray(chains.chain_point_ids)
        a_list, b_list, ti, di, ni, parent_ids = [], [], [], [], [], []
        for local_chain, source_chain in enumerate(kept_source_ids):
            ids = point_ids[offsets[source_chain] : offsets[source_chain + 1]]
            a_list.append(points[ids[:-1]])
            b_list.append(points[ids[1:]])
            count = ids.size - 1
            ti.append(np.full(count, tough_kept[local_chain]))
            di.append(np.full(count, diff_kept[local_chain]))
            ni.append(np.full(count, trap_kept[local_chain]))
            parent_ids.append(np.full(count, local_chain, dtype=np.int64))

        meta = {
            "kind": "G_GB_chains",
            "source_chains": n_chains,
            "kept_chains": int(kept_source_ids.size),
            "excluded_low_confidence_chains": int(np.count_nonzero(~keep_chain)),
            "confidence_floor": float(confidence_floor),
            "attribute_location_permutation": permuted,
            "attribute_location_permutation_seed": (
                int(attribute_permutation_seed) if permuted else None
            ),
            "attribute_location_permutation_sha256": permutation_hash,
            "artifact_semantic_sha256": chains.semantic_sha256,
        }
        extra_metadata = dict(metadata or {})
        overlap = sorted(set(meta) & set(extra_metadata))
        if overlap:
            raise ValueError("metadata must not override generated keys: " + ", ".join(overlap))
        meta.update(extra_metadata)
        return cls(
            np.concatenate(a_list),
            np.concatenate(b_list),
            np.concatenate(ti),
            np.concatenate(di),
            np.concatenate(ni),
            influence_radius,
            segment_chain_id=np.concatenate(parent_ids),
            metadata=meta,
        )

    @classmethod
    def from_config(cls, config: RunConfig) -> ChainBoundaryField:
        from inverse_pfm.gb_chains import load_grain_boundary_chains

        artifact = config.graph.chain_artifact
        if not artifact:
            raise ValueError("graph.chain_artifact is not set")
        path = config.resolve_path(artifact)
        chains = load_grain_boundary_chains(path)
        manifest = _verified_manifest(path, chains)

        # The artifact coordinates are physical (mm); refuse to silently place a
        # microstructure whose extent disagrees with the simulated specimen.
        g = config.geometry
        pts = np.asarray(chains.points, dtype=float)
        if chains.n_segments == 0:
            raise ValueError("chain_artifact contains no boundary segments")
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        tol = 1.0e-6 * math.hypot(g.length, g.height)
        if lo[0] < -tol or lo[1] < -tol or hi[0] > g.length + tol or hi[1] > g.height + tol:
            raise ValueError(
                "chain_artifact extent "
                f"[{lo[0]:.6g},{hi[0]:.6g}]x[{lo[1]:.6g},{hi[1]:.6g}] "
                f"does not fit geometry {g.length:.6g}x{g.height:.6g}"
            )
        seed = config.graph.attribute_permutation_seed
        return cls.from_chains(
            chains,
            config.graph.influence_radius,
            confidence_floor=config.graph.confidence_floor,
            attribute_permutation_seed=None if seed < 0 else seed,
            metadata={
                "chain_artifact": str(path),
                "chain_artifact_manifest": str(path.with_suffix(".json")),
                "artifact_sha256": manifest["artifact_sha256"],
                "ctf_sha256": chains.metadata.get("ctf_sha256"),
                "coordinate_units": chains.metadata.get("coordinate_units"),
                "coordinate_convention": chains.metadata.get("coordinate_convention"),
                "coordinate_min": [float(lo[0]), float(lo[1])],
                "coordinate_max": [float(hi[0]), float(hi[1])],
            },
        )

    # ------------------------------------------------------------------ #
    # uniform-grid segment index
    # ------------------------------------------------------------------ #
    def _build_index(self) -> None:
        midpoints = self._a + 0.5 * self._ab
        self._origin = midpoints.min(axis=0) - self._query_radius
        upper = midpoints.max(axis=0) + self._query_radius
        cell = self._query_radius
        span = np.maximum(upper - self._origin, cell)
        self._n_cells = np.maximum(np.ceil(span / cell).astype(np.int64), 1)
        self._cell_size = cell
        ncx = int(self._n_cells[0])
        cx = np.clip((midpoints[:, 0] - self._origin[0]) / cell, 0, self._n_cells[0] - 1)
        cy = np.clip((midpoints[:, 1] - self._origin[1]) / cell, 0, self._n_cells[1] - 1)
        flat = cy.astype(np.int64) * ncx + cx.astype(np.int64)
        order = np.argsort(flat, kind="stable")
        self._seg_order = order
        sorted_flat = flat[order]
        n_total = int(self._n_cells[0] * self._n_cells[1])
        self._bucket_start = np.searchsorted(sorted_flat, np.arange(n_total + 1))

    def _candidates(self, cell_x: int, cell_y: int) -> np.ndarray:
        ncx, ncy = int(self._n_cells[0]), int(self._n_cells[1])
        pieces: list[np.ndarray] = []
        for jy in range(max(0, cell_y - 1), min(ncy, cell_y + 2)):
            for jx in range(max(0, cell_x - 1), min(ncx, cell_x + 2)):
                flat = jy * ncx + jx
                lo, hi = self._bucket_start[flat], self._bucket_start[flat + 1]
                if hi > lo:
                    pieces.append(self._seg_order[lo:hi])
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(pieces)

    def _distance2(self, points: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """Squared point-to-segment distance, ``(n_points, n_seg)``."""
        pair_count = int(points.shape[0] * seg.size)
        if pair_count > self.max_distance_pairs:
            raise RuntimeError(
                "internal distance chunk exceeded max_distance_pairs: "
                f"{pair_count} > {self.max_distance_pairs}"
            )
        self._max_distance_pairs_observed = max(self._max_distance_pairs_observed, pair_count)
        pa = points[:, None, :] - self._a[None, seg, :]
        ab = self._ab[None, seg, :]
        t = np.einsum("psd,psd->ps", pa, ab) / self._len2[None, seg]
        t = np.clip(t, 0.0, 1.0)
        closest = self._a[None, seg, :] + t[:, :, None] * ab
        delta = points[:, None, :] - closest
        return np.einsum("psd,psd->ps", delta, delta)

    def _decay(self, distance2: np.ndarray) -> np.ndarray:
        """Gaussian decay with an explicit, index-independent cutoff."""
        result = np.zeros_like(distance2)
        inside = distance2 < self._trunc_radius**2
        np.exp(
            -distance2 / self.influence_radius**2,
            out=result,
            where=inside,
        )
        return result

    def _evaluate_group(
        self, points: np.ndarray, candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate one spatial-index point group with bounded work arrays.

        Candidate segments are first put in deterministic ``(parent chain,
        segment id)`` order.  Point chunks and whole-parent-chain segment blocks
        are then chosen so every call to :meth:`_distance2` contains at most
        ``max_distance_pairs`` point--segment pairs.  A parent chain larger than
        one segment block is reduced incrementally before its single field
        contribution is applied.
        """
        parent = self._segment_chain[candidates]
        order = np.lexsort((candidates, parent))
        segments = candidates[order]
        parent = parent[order]
        chain_starts = np.r_[0, np.flatnonzero(np.diff(parent)) + 1]
        chain_stops = np.r_[chain_starts[1:], segments.size]
        chain_ids = parent[chain_starts]

        n_points = points.shape[0]
        tough = np.ones(n_points)
        diff = np.ones(n_points)
        trap = np.zeros(n_points)
        point_capacity = max(
            1,
            min(
                _MAX_POINTS_PER_CHUNK,
                self.max_distance_pairs // min(segments.size, self.max_distance_pairs),
            ),
        )

        for point_start in range(0, n_points, point_capacity):
            point_stop = min(point_start + point_capacity, n_points)
            point_block = points[point_start:point_stop]
            segment_capacity = max(1, self.max_distance_pairs // point_block.shape[0])
            weakness = np.zeros(point_block.shape[0])
            enhancement = np.zeros(point_block.shape[0])
            trap_block = np.zeros(point_block.shape[0])

            chain = 0
            while chain < chain_ids.size:
                chain_size = int(chain_stops[chain] - chain_starts[chain])
                if chain_size > segment_capacity:
                    # A single long polyline spans several segment chunks.  It
                    # still contributes exactly once, using its nearest segment.
                    minimum = np.full(point_block.shape[0], math.inf)
                    first_segment = int(chain_starts[chain])
                    final_segment = int(chain_stops[chain])
                    for segment_start in range(first_segment, final_segment, segment_capacity):
                        segment_stop = min(segment_start + segment_capacity, final_segment)
                        distance2 = self._distance2(
                            point_block, segments[segment_start:segment_stop]
                        )
                        minimum = np.minimum(minimum, np.min(distance2, axis=1))
                    decay = self._decay(minimum)
                    chain_id = int(chain_ids[chain])
                    weakness = np.maximum(weakness, (1.0 - self._chain_tough[chain_id]) * decay)
                    enhancement = np.maximum(
                        enhancement, (self._chain_diff[chain_id] - 1.0) * decay
                    )
                    trap_block += self._chain_trap[chain_id] * decay
                    chain += 1
                    continue

                # Pack as many complete parent chains as possible into one
                # segment block.  Complete chains allow a vectorised reduceat.
                block_stop_chain = chain + 1
                while block_stop_chain < chain_ids.size:
                    proposed = int(chain_stops[block_stop_chain] - chain_starts[chain])
                    if proposed > segment_capacity:
                        break
                    block_stop_chain += 1
                segment_start = int(chain_starts[chain])
                segment_stop = int(chain_stops[block_stop_chain - 1])
                distance2 = self._distance2(point_block, segments[segment_start:segment_stop])
                local_starts = chain_starts[chain:block_stop_chain] - segment_start
                minimum = np.minimum.reduceat(distance2, local_starts, axis=1)
                decay = self._decay(minimum)
                ids = chain_ids[chain:block_stop_chain]
                weakness = np.maximum(
                    weakness,
                    np.max((1.0 - self._chain_tough[ids])[None, :] * decay, axis=1),
                )
                enhancement = np.maximum(
                    enhancement,
                    np.max((self._chain_diff[ids] - 1.0)[None, :] * decay, axis=1),
                )
                # Canonical chain order and fixed chunking make this reduction
                # deterministic on every independently evaluating MPI rank.
                trap_block += np.sum(self._chain_trap[ids][None, :] * decay, axis=1)
                chain = block_stop_chain

            tough[point_start:point_stop] = 1.0 - weakness
            diff[point_start:point_stop] = 1.0 + enhancement
            trap[point_start:point_stop] = trap_block
        return tough, diff, trap

    def _evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (toughness_ratio, diffusivity_ratio, trap_density) at ``x``."""
        raw = np.asarray(x)
        if raw.ndim != 2 or raw.shape[0] < 2:
            raise ValueError("query coordinates must have shape (gdim, n) with gdim >= 2")
        if np.issubdtype(raw.dtype, np.complexfloating):
            raise ValueError("query coordinates must be real")
        points = np.ascontiguousarray(np.asarray(raw[:2], dtype=float).T)
        if not np.all(np.isfinite(points)):
            raise ValueError("query coordinates must be finite")
        key = points.tobytes()
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1], self._cache[2], self._cache[3]

        n = points.shape[0]
        tough = np.ones(n)
        diff = np.ones(n)
        trap = np.zeros(n)
        if n == 0:
            self._cache = (key, tough, diff, trap)
            return tough, diff, trap

        cell = self._cell_size
        cx = np.clip(
            ((points[:, 0] - self._origin[0]) / cell).astype(np.int64), 0, int(self._n_cells[0]) - 1
        )
        cy = np.clip(
            ((points[:, 1] - self._origin[1]) / cell).astype(np.int64), 0, int(self._n_cells[1]) - 1
        )
        flat = cy * int(self._n_cells[0]) + cx
        order = np.argsort(flat, kind="stable")
        flat_sorted = flat[order]
        boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
        groups = np.split(order, boundaries)
        for group in groups:
            gx, gy = int(cx[group[0]]), int(cy[group[0]])
            seg = self._candidates(gx, gy)
            if seg.size == 0:
                continue
            group_tough, group_diff, group_trap = self._evaluate_group(points[group], seg)
            tough[group] = group_tough
            diff[group] = group_diff
            trap[group] = group_trap
        self._cache = (key, tough, diff, trap)
        return tough, diff, trap

    # ------------------------------------------------------------------ #
    # public field interface (mirrors BoundaryGraph)
    # ------------------------------------------------------------------ #
    def toughness_ratio_at(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x)[0]

    def diffusivity_ratio_at(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x)[1]

    def trap_density_at(self, x: np.ndarray) -> np.ndarray:
        return self._evaluate(x)[2]

    @property
    def n_segments(self) -> int:
        return int(self._a.shape[0])

    def describe(self) -> dict[str, object]:
        return {
            "kind": "G_GB_chains",
            "segments": self.n_segments,
            "parent_chains": int(self._chain_tough.size),
            "influence_radius": self.influence_radius,
            "truncation_radius": self._trunc_radius,
            "gaussian_decay_cutoff": _TRUNCATION,
            "max_distance_pairs_per_chunk": self.max_distance_pairs,
            "max_distance_pairs_observed": self._max_distance_pairs_observed,
            "total_boundary_length": float(np.sum(self._segment_lengths)),
            "minimum_segment_toughness_ratio": float(np.min(self._tough)),
            "maximum_segment_diffusivity_ratio": float(np.max(self._diff)),
            "sum_parent_chain_trap_density": float(np.sum(self._chain_trap)),
            "index_cells": [int(self._n_cells[0]), int(self._n_cells[1])],
            "field_combination_rules": {
                "toughness": "minimum Gaussian parent-chain ratio",
                "diffusivity": "maximum Gaussian parent-chain ratio",
                "trap_density": (
                    "sum of parent-chain contributions, each using its nearest segment"
                ),
            },
            "provenance": dict(self.metadata),
        }


def _chain_attribute(
    values: np.ndarray | None, n_chains: int, default: float, name: str
) -> np.ndarray:
    if values is None:
        return np.full(n_chains, default, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape != (n_chains,):
        raise ValueError(f"{name} must have shape ({n_chains},)")
    return array


def _permutation_sha256(assigned_source_chain_ids: np.ndarray) -> str:
    values = np.ascontiguousarray(assigned_source_chain_ids, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"graphfracture.attribute-location-permutation.v1\0")
    digest.update(values.shape[0].to_bytes(8, "little"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _verified_manifest(path: Any, chains: Any) -> dict[str, Any]:
    """Read provenance from the manifest already integrity-checked by loader."""
    manifest_path = path.with_suffix(".json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read verified chain_artifact manifest {manifest_path}") from exc
    if type(manifest) is not dict:
        raise ValueError("chain_artifact manifest must be a JSON object")
    if manifest.get("semantic_sha256") != chains.semantic_sha256:
        raise ValueError("chain_artifact manifest changed after artifact validation")
    artifact_hash = manifest.get("artifact_sha256")
    if (
        type(artifact_hash) is not str
        or len(artifact_hash) != 64
        or any(character not in "0123456789abcdef" for character in artifact_hash)
    ):
        raise ValueError("chain_artifact manifest has an invalid artifact_sha256")
    if manifest.get("metadata") != dict(chains.metadata):
        raise ValueError("chain_artifact metadata changed after artifact validation")
    return manifest
