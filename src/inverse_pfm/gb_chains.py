"""Extract embedded grain-boundary chains from a regular grain-label image.

The grain adjacency graph in :mod:`inverse_pfm.micrograph` carries useful
attributes, but a line between two grain centroids is not the physical grain
boundary.  This module constructs the geometric counterpart directly on the
dual of the regular EBSD pixel grid:

* unequal left/right labels create a vertical atomic segment;
* unequal top/bottom labels create a horizontal atomic segment;
* atomic segments with the same canonical grain pair are followed through a
  vertex only when that vertex has global degree two.

Consequently, chains stop at specimen edges and junctions, while isolated
closed boundaries remain closed polylines.  No skeletonisation or geometric
smoothing is applied, so atomic length and topology are exactly auditable.

The artifact format is deliberately numeric-only.  NPZ files are read with
``allow_pickle=False`` and accompanied by a JSON manifest containing a
name-sorted semantic SHA256 over array names, dtypes, shapes and values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_NAME = "inverse_pfm.gb_chains"
FORMAT_VERSION = 1

_REQUIRED_ARRAYS = frozenset(
    {
        "points",
        "chain_offsets",
        "chain_point_ids",
        "grain_pairs",
        "phase_pairs",
        "confidence_fraction",
    }
)
_OPTIONAL_ARRAYS = frozenset({"toughness_ratio", "diffusivity_ratio", "trap_density"})
_ALLOWED_ARRAYS = _REQUIRED_ARRAYS | _OPTIONAL_ARRAYS


def _as_array(
    value: Any,
    name: str,
    *,
    dtype: np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"{name} must not have object dtype")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError(f"{name} must have a numeric dtype")
    if np.issubdtype(dtype, np.integer):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must have an integer dtype")
        limits = np.iinfo(dtype)
        if array.size and (np.any(array < limits.min) or np.any(array > limits.max)):
            raise ValueError(f"{name} contains values outside the {dtype} range")
    elif np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real values")
    try:
        result = np.asarray(array, dtype=dtype).copy(order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as {np.dtype(dtype)}") from exc
    result.setflags(write=False)
    return result


def _optional_float_array(value: Any, name: str, size: int) -> np.ndarray | None:
    if value is None:
        return None
    result = _as_array(value, name, dtype=np.dtype(np.float64), ndim=1)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        result: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise ValueError("metadata must be a mapping")
    try:
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be a finite JSON-serialisable mapping") from exc
    decoded = json.loads(encoded)
    if decoded != result:
        raise ValueError(
            "metadata must use stable JSON types with string object keys and list sequences"
        )
    return decoded


def _canonical_numeric_bytes(array: np.ndarray) -> tuple[str, bytes]:
    """Return a platform-independent dtype label and C-order byte payload."""
    dtype = array.dtype
    if dtype.hasobject:
        raise ValueError("semantic hashes do not support object dtype")
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
        raise ValueError(f"semantic hashes require numeric arrays, got {dtype}")
    if np.issubdtype(dtype, np.inexact) and not np.all(np.isfinite(array)):
        raise ValueError("semantic hashes require finite arrays")

    if dtype.byteorder == ">" or (dtype.byteorder == "=" and os.sys.byteorder == "big"):
        canonical = array.byteswap().view(dtype.newbyteorder("<"))
    else:
        canonical = array.view(dtype.newbyteorder("<")) if dtype.byteorder == "=" else array
    canonical = np.ascontiguousarray(canonical)
    return canonical.dtype.str, canonical.tobytes(order="C")


def semantic_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash numeric arrays independently of mapping insertion order.

    Array names are sorted and each contribution includes its canonical dtype,
    shape and C-order values.  The function intentionally rejects object arrays
    rather than allowing an implicit pickle-based representation.
    """
    digest = hashlib.sha256()
    digest.update(b"inverse_pfm.gb_chains.semantic.v1\0")
    names = list(arrays)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("semantic hash array names must be non-empty strings")
    for name in sorted(names):
        array = np.asarray(arrays[name])
        dtype_label, payload = _canonical_numeric_bytes(array)
        name_bytes = name.encode("utf-8")
        shape = json.dumps(array.shape, separators=(",", ":")).encode("ascii")
        for part in (name_bytes, dtype_label.encode("ascii"), shape, payload):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


def _artifact_sha256(
    array_hash: str,
    ambiguous_checkerboards: int,
    metadata: Mapping[str, Any],
) -> str:
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "array_semantic_sha256": array_hash,
        "ambiguous_checkerboards": ambiguous_checkerboards,
        "metadata": metadata,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GrainBoundaryChains:
    """Numeric representation of embedded grain-boundary polylines.

    ``phase_pairs`` follow the order of each canonical ``grain_pairs`` row;
    the two phase ids are not themselves sorted.  ``trap_density`` is an
    optional caller-supplied chain parameter.  This module never interprets a
    legacy trap *ratio* as an absolute trap density.
    """

    points: np.ndarray
    chain_offsets: np.ndarray
    chain_point_ids: np.ndarray
    grain_pairs: np.ndarray
    phase_pairs: np.ndarray
    confidence_fraction: np.ndarray
    toughness_ratio: np.ndarray | None = None
    diffusivity_ratio: np.ndarray | None = None
    trap_density: np.ndarray | None = None
    ambiguous_checkerboards: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _as_array(self.points, "points", dtype=np.dtype(np.float64), ndim=2)
        offsets = _as_array(self.chain_offsets, "chain_offsets", dtype=np.dtype(np.int64), ndim=1)
        point_ids = _as_array(
            self.chain_point_ids, "chain_point_ids", dtype=np.dtype(np.int64), ndim=1
        )
        grain_pairs = _as_array(self.grain_pairs, "grain_pairs", dtype=np.dtype(np.int64), ndim=2)
        phase_pairs = _as_array(self.phase_pairs, "phase_pairs", dtype=np.dtype(np.int64), ndim=2)
        confidence = _as_array(
            self.confidence_fraction,
            "confidence_fraction",
            dtype=np.dtype(np.float64),
            ndim=1,
        )

        if points.shape[1:] != (2,):
            raise ValueError("points must have shape (n_points, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")
        if points.shape[0] and np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("points must have unique coordinates")
        if offsets.size == 0 or offsets[0] != 0:
            raise ValueError("chain_offsets must start at zero")
        if np.any(np.diff(offsets) < 2):
            raise ValueError("every chain must contain at least two point ids")
        if offsets[-1] != point_ids.size:
            raise ValueError("the final chain offset must equal len(chain_point_ids)")

        n_chains = offsets.size - 1
        if grain_pairs.shape != (n_chains, 2):
            raise ValueError(f"grain_pairs must have shape ({n_chains}, 2)")
        if phase_pairs.shape != (n_chains, 2):
            raise ValueError(f"phase_pairs must have shape ({n_chains}, 2)")
        if confidence.shape != (n_chains,):
            raise ValueError(f"confidence_fraction must have shape ({n_chains},)")
        if point_ids.size and (np.min(point_ids) < 0 or np.max(point_ids) >= points.shape[0]):
            raise ValueError("chain_point_ids contains an out-of-range point id")
        if np.any(grain_pairs[:, 0] >= grain_pairs[:, 1]):
            raise ValueError("grain_pairs rows must be strictly increasing")
        if np.any(grain_pairs < 0):
            raise ValueError("grain ids must be non-negative")
        if np.any(phase_pairs < 0):
            raise ValueError("phase ids must be non-negative")
        if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence_fraction must lie in [0, 1]")
        if type(self.ambiguous_checkerboards) is not int or self.ambiguous_checkerboards < 0:
            raise ValueError("ambiguous_checkerboards must be a non-negative integer")

        segments: list[tuple[int, int]] = []
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            ids = point_ids[start:stop]
            unique_ids = ids[:-1] if ids[0] == ids[-1] else ids
            if np.unique(unique_ids).size != unique_ids.size:
                raise ValueError(
                    "an open chain must be simple and a closed chain may repeat only its endpoint"
                )
            if np.any(ids[:-1] == ids[1:]):
                raise ValueError("chains must not contain zero-length segments")
            delta = points[ids[1:]] - points[ids[:-1]]
            if np.any(np.linalg.norm(delta, axis=1) <= 0.0):
                raise ValueError("chains must not contain zero-length segments")
            segments.extend(
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in zip(ids[:-1], ids[1:], strict=True)
            )
        if len(set(segments)) != len(segments):
            raise ValueError("duplicate undirected atomic segment")

        toughness = _optional_float_array(self.toughness_ratio, "toughness_ratio", n_chains)
        diffusivity = _optional_float_array(self.diffusivity_ratio, "diffusivity_ratio", n_chains)
        traps = _optional_float_array(self.trap_density, "trap_density", n_chains)
        if toughness is not None and np.any((toughness <= 0) | (toughness > 1)):
            raise ValueError("toughness_ratio must lie in (0, 1]")
        if diffusivity is not None and np.any(diffusivity < 1):
            raise ValueError("diffusivity_ratio must be at least 1")
        if traps is not None and np.any(traps < 0):
            raise ValueError("trap_density must be non-negative")

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "chain_offsets", offsets)
        object.__setattr__(self, "chain_point_ids", point_ids)
        object.__setattr__(self, "grain_pairs", grain_pairs)
        object.__setattr__(self, "phase_pairs", phase_pairs)
        object.__setattr__(self, "confidence_fraction", confidence)
        object.__setattr__(self, "toughness_ratio", toughness)
        object.__setattr__(self, "diffusivity_ratio", diffusivity)
        object.__setattr__(self, "trap_density", traps)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_chains(self) -> int:
        return int(self.chain_offsets.size - 1)

    @property
    def n_segments(self) -> int:
        return int(self.chain_point_ids.size - self.n_chains)

    @property
    def closed(self) -> np.ndarray:
        result = np.zeros(self.n_chains, dtype=bool)
        for index, (start, stop) in enumerate(
            zip(self.chain_offsets[:-1], self.chain_offsets[1:], strict=True)
        ):
            ids = self.chain_point_ids[start:stop]
            result[index] = ids[0] == ids[-1]
        return result

    @property
    def chain_lengths(self) -> np.ndarray:
        result = np.zeros(self.n_chains, dtype=float)
        for index, (start, stop) in enumerate(
            zip(self.chain_offsets[:-1], self.chain_offsets[1:], strict=True)
        ):
            ids = self.chain_point_ids[start:stop]
            result[index] = np.linalg.norm(
                self.points[ids[1:]] - self.points[ids[:-1]], axis=1
            ).sum()
        return result

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the numeric artifact arrays, omitting absent attributes."""
        result = {
            "points": self.points,
            "chain_offsets": self.chain_offsets,
            "chain_point_ids": self.chain_point_ids,
            "grain_pairs": self.grain_pairs,
            "phase_pairs": self.phase_pairs,
            "confidence_fraction": self.confidence_fraction,
        }
        for name in sorted(_OPTIONAL_ARRAYS):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @property
    def semantic_sha256(self) -> str:
        return semantic_sha256(self.arrays())

    def with_attributes(
        self,
        *,
        toughness_ratio: Sequence[float] | np.ndarray | None = None,
        diffusivity_ratio: Sequence[float] | np.ndarray | None = None,
        trap_density: Sequence[float] | np.ndarray | None = None,
    ) -> GrainBoundaryChains:
        """Return a validated copy with caller-defined chain attributes.

        All three arguments are interpreted literally.  In particular,
        ``trap_density`` is not populated from a legacy dimensionless trap
        ratio by this method.
        """
        return replace(
            self,
            toughness_ratio=toughness_ratio,
            diffusivity_ratio=diffusivity_ratio,
            trap_density=trap_density,
        )


@dataclass(frozen=True)
class _AtomicSegment:
    first: int
    second: int
    grain_pair: tuple[int, int]
    length: float
    trusted: bool


def _canonical_chain(point_ids: list[int]) -> tuple[int, ...]:
    """Choose a deterministic orientation and, for cycles, starting point."""
    if point_ids[0] != point_ids[-1]:
        forward = tuple(point_ids)
        reverse = tuple(reversed(point_ids))
        return min(forward, reverse)

    core = point_ids[:-1]
    if not core:
        raise ValueError("closed chains require at least one non-repeated point")
    minimum = min(core)
    start = core.index(minimum)
    forward_core = core[start:] + core[:start]
    reversed_core = list(reversed(core))
    reverse_start = reversed_core.index(minimum)
    reverse_core = reversed_core[reverse_start:] + reversed_core[:reverse_start]
    canonical = min(tuple(forward_core), tuple(reverse_core))
    return (*canonical, canonical[0])


def _majority_phase_by_grain(grain_id: np.ndarray, phase: np.ndarray) -> dict[int, int]:
    result: dict[int, int] = {}
    for grain in np.unique(grain_id).tolist():
        values = phase[grain_id == grain]
        indexed_values = values[values > 0]
        if indexed_values.size == 0:
            result[int(grain)] = 0
            continue
        labels, counts = np.unique(indexed_values, return_counts=True)
        maximum = int(np.max(counts))
        result[int(grain)] = int(np.min(labels[counts == maximum]))
    return result


def extract_grain_boundary_chains(
    grain_id: np.ndarray,
    *,
    phase: np.ndarray | None = None,
    indexed: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    roi_origin: tuple[float, float] = (0.0, 0.0),
    unit_scale: float = 1.0,
    flip_y: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> GrainBoundaryChains:
    """Extract maximal attributed boundary chains from a 2-D grain label map.

    ``roi_origin`` is expressed in the same source length units as ``dx`` and
    ``dy``.  Without a y flip it is the top/first-row dual-grid origin.  With a
    y flip it is the lower-left origin of the transformed ROI.  ``unit_scale``
    is then applied uniformly to both axes, preventing silent anisotropic
    distortion.

    Chain confidence is the physical-length-weighted fraction of atomic
    segments for which both adjacent EBSD pixels are marked ``indexed``.
    """
    labels_raw = np.asarray(grain_id)
    if labels_raw.dtype.hasobject:
        raise ValueError("grain_id must not have object dtype")
    if labels_raw.ndim != 2 or not np.issubdtype(labels_raw.dtype, np.integer):
        raise ValueError("grain_id must be a two-dimensional integer array")
    labels = np.asarray(labels_raw, dtype=np.int64)
    if labels.size == 0 or min(labels.shape) == 0:
        raise ValueError("grain_id must contain at least one pixel")
    if np.any(labels < 0):
        raise ValueError("grain_id must be non-negative")
    ny, nx = labels.shape

    if phase is None:
        phase_array = np.zeros_like(labels)
    else:
        phase_raw = np.asarray(phase)
        if phase_raw.dtype.hasobject:
            raise ValueError("phase must not have object dtype")
        if phase_raw.shape != labels.shape or not np.issubdtype(phase_raw.dtype, np.integer):
            raise ValueError("phase must be an integer array matching grain_id")
        phase_array = np.asarray(phase_raw, dtype=np.int64)
        if np.any(phase_array < 0):
            raise ValueError("phase ids must be non-negative")

    if indexed is None:
        indexed_array = np.ones(labels.shape, dtype=bool)
    else:
        indexed_raw = np.asarray(indexed)
        if indexed_raw.dtype != np.dtype(bool) or indexed_raw.shape != labels.shape:
            raise ValueError("indexed must be a boolean array matching grain_id")
        indexed_array = indexed_raw

    for name, value in {"dx": dx, "dy": dy, "unit_scale": unit_scale}.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite real number")
    if (
        not isinstance(roi_origin, tuple | list)
        or len(roi_origin) != 2
        or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in roi_origin
        )
        or not all(math.isfinite(float(value)) for value in roi_origin)
    ):
        raise ValueError("roi_origin must contain two finite real numbers")
    if type(flip_y) is not bool:
        raise ValueError("flip_y must be a boolean")

    # Each entry keeps integer dual-grid endpoints until the unique point table
    # has been constructed.  The endpoint coordinates are exact integers at
    # this stage, avoiding floating-point equality as a topology operation.
    raw_segments: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], float, bool]] = []
    vertical_y, vertical_x = np.where(labels[:, :-1] != labels[:, 1:])
    for iy, ix in zip(vertical_y.tolist(), vertical_x.tolist(), strict=True):
        pair = tuple(sorted((int(labels[iy, ix]), int(labels[iy, ix + 1]))))
        raw_segments.append(
            (
                (ix + 1, iy),
                (ix + 1, iy + 1),
                pair,
                float(dy * unit_scale),
                bool(indexed_array[iy, ix] and indexed_array[iy, ix + 1]),
            )
        )

    horizontal_y, horizontal_x = np.where(labels[:-1, :] != labels[1:, :])
    for iy, ix in zip(horizontal_y.tolist(), horizontal_x.tolist(), strict=True):
        pair = tuple(sorted((int(labels[iy, ix]), int(labels[iy + 1, ix]))))
        raw_segments.append(
            (
                (ix, iy + 1),
                (ix + 1, iy + 1),
                pair,
                float(dx * unit_scale),
                bool(indexed_array[iy, ix] and indexed_array[iy + 1, ix]),
            )
        )

    lattice_points = sorted(
        {point for first, second, *_ in raw_segments for point in (first, second)}
    )
    point_lookup = {point: index for index, point in enumerate(lattice_points)}

    ox, oy = float(roi_origin[0]), float(roi_origin[1])
    coordinates = np.empty((len(lattice_points), 2), dtype=np.float64)
    for index, (ix, iy) in enumerate(lattice_points):
        coordinates[index, 0] = (ox + ix * dx) * unit_scale
        y_index = ny - iy if flip_y else iy
        coordinates[index, 1] = (oy + y_index * dy) * unit_scale

    atomic = [
        _AtomicSegment(point_lookup[first], point_lookup[second], pair, length, trusted)
        for first, second, pair, length, trusted in raw_segments
    ]
    incidence: dict[int, list[int]] = defaultdict(list)
    for segment_id, segment in enumerate(atomic):
        incidence[segment.first].append(segment_id)
        incidence[segment.second].append(segment_id)
    for segment_ids in incidence.values():
        segment_ids.sort()

    unused = set(range(len(atomic)))

    def pass_through(point_id: int, pair: tuple[int, int]) -> bool:
        incident = incidence[point_id]
        return len(incident) == 2 and all(atomic[index].grain_pair == pair for index in incident)

    def trace(segment_id: int, start_point: int) -> tuple[tuple[int, ...], tuple[int, int], float]:
        points = [start_point]
        current_point = start_point
        current_segment = segment_id
        pair = atomic[segment_id].grain_pair
        trusted_length = 0.0
        total_length = 0.0
        while True:
            unused.remove(current_segment)
            segment = atomic[current_segment]
            total_length += segment.length
            if segment.trusted:
                trusted_length += segment.length
            next_point = segment.second if current_point == segment.first else segment.first
            points.append(next_point)
            if not pass_through(next_point, pair):
                break
            candidates = [index for index in incidence[next_point] if index in unused]
            if not candidates:  # completed a closed cycle
                break
            if len(candidates) != 1:
                raise RuntimeError("degree-two chain tracing found ambiguous continuation")
            current_point = next_point
            current_segment = candidates[0]
        confidence = trusted_length / total_length
        return _canonical_chain(points), pair, confidence

    starts: list[tuple[int, int]] = []
    for segment_id, segment in enumerate(atomic):
        for point_id in sorted((segment.first, segment.second)):
            if not pass_through(point_id, segment.grain_pair):
                starts.append((segment_id, point_id))

    traced: list[tuple[tuple[int, ...], tuple[int, int], float]] = []
    for segment_id, point_id in sorted(starts):
        if segment_id in unused:
            traced.append(trace(segment_id, point_id))
    while unused:  # components without a non-degree-two start are closed cycles
        segment_id = min(unused)
        segment = atomic[segment_id]
        traced.append(trace(segment_id, min(segment.first, segment.second)))

    traced.sort(key=lambda item: (item[1], item[0]))
    offsets = [0]
    flat_point_ids: list[int] = []
    grain_pairs: list[tuple[int, int]] = []
    confidence: list[float] = []
    for point_ids, pair, trusted_fraction in traced:
        flat_point_ids.extend(point_ids)
        offsets.append(len(flat_point_ids))
        grain_pairs.append(pair)
        confidence.append(trusted_fraction)

    majority_phase = _majority_phase_by_grain(labels, phase_array)
    phase_pairs = [(majority_phase[first], majority_phase[second]) for first, second in grain_pairs]

    if ny > 1 and nx > 1:
        top_left = labels[:-1, :-1]
        top_right = labels[:-1, 1:]
        bottom_left = labels[1:, :-1]
        bottom_right = labels[1:, 1:]
        checkerboard = (
            (top_left == bottom_right) & (top_right == bottom_left) & (top_left != top_right)
        )
        ambiguous_checkerboards = int(np.count_nonzero(checkerboard))
    else:
        ambiguous_checkerboards = 0

    automatic_metadata = {
        "grid_shape": [ny, nx],
        "pixel_size_source_units": [float(dx), float(dy)],
        "roi_origin_source_units": [ox, oy],
        "unit_scale": float(unit_scale),
        "flip_y": flip_y,
        "coordinate_convention": "pixel-dual vertices",
        "chain_rule": "same grain pair through global degree-two vertices",
        "confidence_rule": "length fraction with both adjacent pixels indexed",
    }
    supplied_metadata = _json_mapping(metadata)
    overlap = sorted(set(automatic_metadata) & set(supplied_metadata))
    if overlap:
        raise ValueError("metadata must not override generated keys: " + ", ".join(overlap))
    automatic_metadata.update(supplied_metadata)

    return GrainBoundaryChains(
        points=coordinates.reshape(-1, 2),
        chain_offsets=np.asarray(offsets, dtype=np.int64),
        chain_point_ids=np.asarray(flat_point_ids, dtype=np.int64),
        grain_pairs=np.asarray(grain_pairs, dtype=np.int64).reshape(-1, 2),
        phase_pairs=np.asarray(phase_pairs, dtype=np.int64).reshape(-1, 2),
        confidence_fraction=np.asarray(confidence, dtype=np.float64),
        ambiguous_checkerboards=ambiguous_checkerboards,
        metadata=automatic_metadata,
    )


def _json_path(path: Path) -> Path:
    if path.suffix.lower() != ".npz":
        raise ValueError("grain-boundary chain artifacts must use a .npz path")
    return path.with_suffix(".json")


def save_grain_boundary_chains(chains: GrainBoundaryChains, path: str | Path) -> tuple[Path, Path]:
    """Write a numeric NPZ artifact and its JSON integrity manifest."""
    if not isinstance(chains, GrainBoundaryChains):
        raise TypeError("chains must be a GrainBoundaryChains instance")
    npz_path = Path(path)
    json_path = _json_path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = chains.arrays()
    semantic_hash = semantic_sha256(arrays)
    artifact_hash = _artifact_sha256(semantic_hash, chains.ambiguous_checkerboards, chains.metadata)
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "semantic_sha256": semantic_hash,
        "artifact_sha256": artifact_hash,
        "ambiguous_checkerboards": chains.ambiguous_checkerboards,
        "arrays": {
            name: {"dtype": array.dtype.str, "shape": list(array.shape)}
            for name, array in sorted(arrays.items())
        },
        "metadata": dict(chains.metadata),
    }

    npz_tmp = npz_path.with_name(npz_path.name + ".tmp")
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    try:
        with npz_tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        json_tmp.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        npz_tmp.replace(npz_path)
        json_tmp.replace(json_path)
    finally:
        npz_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
    return npz_path, json_path


def load_grain_boundary_chains(path: str | Path) -> GrainBoundaryChains:
    """Load and integrity-check a numeric grain-boundary chain artifact."""
    npz_path = Path(path)
    json_path = _json_path(npz_path)
    try:
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read grain-boundary chain manifest {json_path}") from exc
    if type(manifest) is not dict:
        raise ValueError("grain-boundary chain manifest must be a JSON object")
    if (
        type(manifest.get("format")) is not str
        or manifest.get("format") != FORMAT_NAME
        or type(manifest.get("format_version")) is not int
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("unsupported grain-boundary chain artifact format")

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            names = set(archive.files)
            unknown = sorted(names - _ALLOWED_ARRAYS)
            missing = sorted(_REQUIRED_ARRAYS - names)
            if unknown:
                raise ValueError("unknown artifact arrays: " + ", ".join(unknown))
            if missing:
                raise ValueError("missing artifact arrays: " + ", ".join(missing))
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            ("unknown artifact arrays", "missing artifact arrays")
        ):
            raise
        raise ValueError(f"cannot read numeric artifact {npz_path} without pickle") from exc

    manifest_arrays = manifest.get("arrays")
    if type(manifest_arrays) is not dict or set(manifest_arrays) != set(arrays):
        raise ValueError("artifact array manifest does not match NPZ contents")
    for name, array in arrays.items():
        description = manifest_arrays[name]
        if type(description) is not dict:
            raise ValueError(f"invalid array manifest for {name}")
        if description.get("dtype") != array.dtype.str or description.get("shape") != list(
            array.shape
        ):
            raise ValueError(f"artifact array metadata mismatch for {name}")

    actual_hash = semantic_sha256(arrays)
    if manifest.get("semantic_sha256") != actual_hash:
        raise ValueError("grain-boundary chain artifact semantic SHA256 mismatch")

    ambiguous = manifest.get("ambiguous_checkerboards", 0)
    metadata = _json_mapping(manifest.get("metadata", {}))
    expected_artifact_hash = _artifact_sha256(actual_hash, ambiguous, metadata)
    if manifest.get("artifact_sha256") != expected_artifact_hash:
        raise ValueError("grain-boundary chain artifact payload SHA256 mismatch")
    return GrainBoundaryChains(
        points=arrays["points"],
        chain_offsets=arrays["chain_offsets"],
        chain_point_ids=arrays["chain_point_ids"],
        grain_pairs=arrays["grain_pairs"],
        phase_pairs=arrays["phase_pairs"],
        confidence_fraction=arrays["confidence_fraction"],
        toughness_ratio=arrays.get("toughness_ratio"),
        diffusivity_ratio=arrays.get("diffusivity_ratio"),
        trap_density=arrays.get("trap_density"),
        ambiguous_checkerboards=ambiguous,
        metadata=metadata,
    )
