"""Export measured EBSD labels as an embedded grain-boundary-chain artifact.

The full EBSD map is reconstructed first so ROI boundaries do not change grain
segmentation or small-grain absorption.  Only then are ``grain_id``, ``phase``
and the original indexed mask cropped and converted to pixel-dual polylines.

Examples
--------
Export the default centred 2:1 ``ban`` ROI, treating every non-bcc boundary as
neutral in this first geometry-only baseline::

    python scripts/76_export_ebsd_gb_chains.py ban \
        --heterophase-policy neutral

The output is numeric NPZ plus a JSON integrity/provenance manifest.  Existing
outputs are never overwritten unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_pfm.ebsd import (  # noqa: E402
    SECTION_AXES,
    input_sha256,
    resolve_ctf_path,
)
from inverse_pfm.ebsd import (  # noqa: E402
    load as load_ebsd,
)
from inverse_pfm.gb_chains import (  # noqa: E402
    GrainBoundaryChains,
    extract_grain_boundary_chains,
    save_grain_boundary_chains,
)
from inverse_pfm.micrograph import GraphModelParams, build_graph  # noqa: E402

DEFAULT_ROI = (0, 238, 54, 173)  # x0, x1, y0, y1; 238 x 119 pixels = 2:1
UM_TO_MM = 1.0e-3
HETEROPHASE_POLICIES = ("reject", "neutral")


def _output_path(section: str, output: str | Path | None) -> Path:
    path = (
        ROOT / "data" / "ebsd_gb_chains" / f"{section}_roi.npz" if output is None else Path(output)
    )
    if path.suffix == "":
        path = path.with_suffix(".npz")
    if path.suffix.lower() != ".npz":
        raise ValueError("output must be a .npz path or a suffix-free artifact base")
    return path.resolve()


def _validate_roi(
    roi: tuple[int, int, int, int],
    *,
    nx: int,
    ny: int,
) -> tuple[int, int, int, int]:
    if len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError("ROI must contain four integer pixel bounds: x0 x1 y0 y1")
    x0, x1, y0, y1 = roi
    if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
        raise ValueError(f"ROI {roi} lies outside the EBSD grid (nx={nx}, ny={ny})")
    return x0, x1, y0, y1


def _pair_attribute_lookup(graph: Any) -> dict[tuple[int, int], tuple[float, float]]:
    edge_i = np.asarray(graph.edge_i)
    edge_j = np.asarray(graph.edge_j)
    if (
        edge_i.dtype.hasobject
        or edge_j.dtype.hasobject
        or not np.issubdtype(edge_i.dtype, np.integer)
        or not np.issubdtype(edge_j.dtype, np.integer)
    ):
        raise ValueError("grain-graph edge ids must use an integer dtype")
    edge_gc_raw = np.asarray(graph.edge_Gc)
    edge_dh_raw = np.asarray(graph.edge_DH)
    if edge_gc_raw.dtype.hasobject or edge_dh_raw.dtype.hasobject:
        raise ValueError("grain-graph edge attributes must not use object dtype")
    arrays = {
        "edge_i": edge_i,
        "edge_j": edge_j,
        "edge_Gc": np.asarray(edge_gc_raw, dtype=float),
        "edge_DH": np.asarray(edge_dh_raw, dtype=float),
    }
    if any(array.ndim != 1 for array in arrays.values()):
        raise ValueError("grain-graph edge attribute arrays must be one-dimensional")
    sizes = {array.size for array in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("grain-graph edge attribute arrays have inconsistent lengths")

    lookup: dict[tuple[int, int], tuple[float, float]] = {}
    for first, second, toughness, diffusivity in zip(
        arrays["edge_i"],
        arrays["edge_j"],
        arrays["edge_Gc"],
        arrays["edge_DH"],
        strict=True,
    ):
        pair = tuple(sorted((int(first), int(second))))
        if pair[0] < 0:
            raise ValueError(f"grain graph contains a negative grain id in pair {pair}")
        if pair[0] == pair[1]:
            raise ValueError(f"grain graph contains a self-pair {pair}")
        if pair in lookup:
            raise ValueError(f"grain graph contains duplicate pair {pair}")
        if not np.isfinite(toughness) or not 0.0 < toughness <= 1.0:
            raise ValueError(f"invalid toughness ratio for grain pair {pair}")
        if not np.isfinite(diffusivity) or diffusivity < 1.0:
            raise ValueError(f"invalid diffusivity ratio for grain pair {pair}")
        lookup[pair] = (float(toughness), float(diffusivity))
    return lookup


def _attribute_chains(
    chains: GrainBoundaryChains,
    graph: Any,
    *,
    matrix_phase_id: int,
    heterophase_policy: str,
) -> tuple[GrainBoundaryChains, dict[str, Any]]:
    if heterophase_policy not in HETEROPHASE_POLICIES:
        raise ValueError("heterophase_policy must be one of " + ", ".join(HETEROPHASE_POLICIES))
    lookup = _pair_attribute_lookup(graph)
    toughness = np.empty(chains.n_chains, dtype=float)
    diffusivity = np.empty(chains.n_chains, dtype=float)
    non_matrix = np.any(chains.phase_pairs != matrix_phase_id, axis=1)

    missing: list[tuple[int, int]] = []
    for index, pair_array in enumerate(chains.grain_pairs):
        pair = (int(pair_array[0]), int(pair_array[1]))
        attributes = lookup.get(pair)
        if attributes is None:
            missing.append(pair)
            continue
        if non_matrix[index]:
            if heterophase_policy == "reject":
                phase_pair = tuple(int(value) for value in chains.phase_pairs[index])
                raise ValueError(
                    f"grain pair {pair} has non-bcc phase pair {phase_pair}; "
                    "rerun with --heterophase-policy neutral to mark it neutral"
                )
            toughness[index] = 1.0
            diffusivity[index] = 1.0
        else:
            toughness[index], diffusivity[index] = attributes
    if missing:
        unique = ", ".join(str(pair) for pair in sorted(set(missing)))
        raise ValueError(f"ROI boundary chains have no grain-graph attributes for: {unique}")

    neutral_length = float(chains.chain_lengths[non_matrix].sum())
    attribute_metadata = {
        "matrix_phase_id": int(matrix_phase_id),
        "heterophase_policy": heterophase_policy,
        "non_matrix_chain_count": int(np.count_nonzero(non_matrix)),
        "non_matrix_chain_length_mm": neutral_length,
        "toughness_source": "legacy GrainGraph.edge_Gc dimensionless ratio",
        "diffusivity_source": "legacy GrainGraph.edge_DH dimensionless ratio",
        "trap_density": {
            "status": "omitted",
            "reason": (
                "legacy GrainGraph.edge_trap is a dimensionless ratio and is not "
                "silently reinterpreted as an absolute additive trap density"
            ),
        },
    }
    return (
        chains.with_attributes(
            toughness_ratio=toughness,
            diffusivity_ratio=diffusivity,
            trap_density=None,
        ),
        attribute_metadata,
    )


def export_ebsd_gb_chains(
    section: str,
    *,
    output: str | Path | None = None,
    roi: tuple[int, int, int, int] = DEFAULT_ROI,
    heterophase_policy: str = "reject",
    y_flip: bool = True,
    overwrite: bool = False,
    graph_params: GraphModelParams | None = None,
) -> dict[str, Any]:
    """Build and save one EBSD grain-boundary-chain artifact."""
    if section not in SECTION_AXES:
        raise ValueError(f"unknown EBSD section {section!r}; choose ban or zhu")
    if heterophase_policy not in HETEROPHASE_POLICIES:
        raise ValueError("heterophase_policy must be one of " + ", ".join(HETEROPHASE_POLICIES))
    if type(overwrite) is not bool or type(y_flip) is not bool:
        raise ValueError("overwrite and y_flip must be booleans")
    output_path = _output_path(section, output)
    json_path = output_path.with_suffix(".json")
    existing = [path for path in (output_path, json_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing artifact files: "
            + ", ".join(str(path) for path in existing)
        )

    ctf_path = resolve_ctf_path(section)
    ctf_hash = input_sha256(ctf_path)
    ebsd = load_ebsd(section)
    params = GraphModelParams() if graph_params is None else graph_params
    if not isinstance(params, GraphModelParams):
        raise TypeError("graph_params must be a GraphModelParams instance")

    # Deliberately reconstruct the complete measured map before applying the
    # ROI.  Cropping first would alter connected components and small-grain
    # absorption at the ROI edges.
    graph = build_graph(ebsd, params)
    x0, x1, y0, y1 = _validate_roi(roi, nx=ebsd.nx, ny=ebsd.ny)
    full_grain_id = np.asarray(graph.grain_id)
    if full_grain_id.shape != (ebsd.ny, ebsd.nx):
        raise ValueError("full reconstructed grain_id shape does not match the EBSD source grid")
    grain_roi = full_grain_id[y0:y1, x0:x1]
    phase_roi = np.asarray(ebsd.phase)[y0:y1, x0:x1]
    indexed_roi = np.asarray(ebsd.indexed)[y0:y1, x0:x1]

    parameter_values = asdict(params)
    reconstruction_names = ("theta_seg", "min_grain_px", "kam_cap")
    reconstruction_parameters = {name: parameter_values[name] for name in reconstruction_names}
    attribute_parameters = {
        name: value for name, value in parameter_values.items() if name not in reconstruction_names
    }
    matrix_phase_id = int(ebsd.matrix_phase_id())
    matrix_phase = next(
        (phase.name for phase in ebsd.phases if phase.number == matrix_phase_id),
        f"phase-{matrix_phase_id}",
    )
    if "bcc" not in matrix_phase.lower():
        raise ValueError(f"the most abundant indexed phase {matrix_phase!r} is not explicitly bcc")
    provenance = {
        "section": section,
        "section_axes": list(SECTION_AXES[section]),
        "ctf_path": str(ctf_path.resolve()),
        "ctf_sha256": ctf_hash,
        "source_grid_shape": [int(ebsd.ny), int(ebsd.nx)],
        "source_step_um": [float(ebsd.dx), float(ebsd.dy)],
        "roi_pixel_bounds": [x0, x1, y0, y1],
        "roi_shape": [y1 - y0, x1 - x0],
        "coordinate_units": "mm",
        "source_length_units": "um",
        "reconstruction_scope": "full EBSD map before ROI crop",
        "reconstruction_parameters": reconstruction_parameters,
        "attribute_parameters": attribute_parameters,
        "full_graph_counts": {
            "grains": int(graph.n_grains),
            "adjacency_edges": int(graph.n_edges),
        },
        "matrix_phase": {"id": matrix_phase_id, "name": matrix_phase},
        "unindexed_policy": {
            "label_fill": (
                "legacy reconstruction: indexed components followed by iterative "
                "four-neighbour fill"
            ),
            "original_roi_index_rate": float(np.mean(indexed_roi)),
            "geometry_confidence": (
                "chain length fraction whose two adjacent source pixels were indexed"
            ),
        },
    }
    chains = extract_grain_boundary_chains(
        grain_roi,
        phase=phase_roi,
        indexed=indexed_roi,
        dx=float(ebsd.dx),
        dy=float(ebsd.dy),
        roi_origin=(0.0, 0.0),
        unit_scale=UM_TO_MM,
        flip_y=y_flip,
        metadata=provenance,
    )
    attributed, attribute_metadata = _attribute_chains(
        chains,
        graph,
        matrix_phase_id=matrix_phase_id,
        heterophase_policy=heterophase_policy,
    )
    combined_metadata = dict(attributed.metadata)
    combined_metadata["chain_attribute_mapping"] = attribute_metadata
    attributed = replace(attributed, metadata=combined_metadata)

    npz_path, manifest_path = save_grain_boundary_chains(attributed, output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "section": section,
        "output_npz": str(npz_path),
        "output_json": str(manifest_path),
        "ctf_sha256": ctf_hash,
        "semantic_sha256": attributed.semantic_sha256,
        "artifact_sha256": manifest["artifact_sha256"],
        "roi_pixel_bounds": [x0, x1, y0, y1],
        "roi_shape": [y1 - y0, x1 - x0],
        "points": attributed.n_points,
        "chains": attributed.n_chains,
        "atomic_segments": attributed.n_segments,
        "closed_chains": int(np.count_nonzero(attributed.closed)),
        "ambiguous_checkerboards": attributed.ambiguous_checkerboards,
        "minimum_confidence_fraction": (
            float(np.min(attributed.confidence_fraction)) if attributed.n_chains else None
        ),
        "non_matrix_chain_count": attribute_metadata["non_matrix_chain_count"],
        "heterophase_policy": heterophase_policy,
        "trap_density_exported": False,
    }
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=sorted(SECTION_AXES))
    parser.add_argument(
        "--output",
        help=(
            "NPZ path or suffix-free artifact base; defaults to "
            "data/ebsd_gb_chains/<section>_roi.npz"
        ),
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X0", "X1", "Y0", "Y1"),
        default=DEFAULT_ROI,
        help="half-open pixel ROI; default: 0 238 54 173",
    )
    parser.add_argument(
        "--heterophase-policy",
        choices=HETEROPHASE_POLICIES,
        default="reject",
        help="reject non-bcc chains by default, or export them as neutral ratios",
    )
    parser.add_argument(
        "--y-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flip EBSD row coordinates to an upward FE y-axis (default: true)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing NPZ/JSON artifact pair",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = export_ebsd_gb_chains(
        args.section,
        output=args.output,
        roi=tuple(args.roi),
        heterophase_policy=args.heterophase_policy,
        y_flip=args.y_flip,
        overwrite=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
