"""Graph-based phase-field fracture for microstructure-sensitive HAC (paper 7).

This package turns measured EBSD microstructure into an attributed graph and
couples that graph to a hydrogen-aware phase-field fracture solver:

* :mod:`inverse_pfm.crystallography` -- cubic orientations, disorientation,
  Schmid factors and IPF colouring;
* :mod:`inverse_pfm.ebsd` -- Channel Text File (``.ctf``) reader;
* :mod:`inverse_pfm.micrograph` -- grain reconstruction and the attributed grain
  graph ``G = (V, E)`` with the ``F_G / F_D / F_alpha`` attribute maps;
* :mod:`inverse_pfm.graph_pf` -- the graph-constrained phase-field solver;
* :mod:`inverse_pfm.hydrogen_network` -- hydrogen diffusion on the boundary
  network with the ``Gc_eff`` softening coupling;
* :mod:`inverse_pfm.path_search` -- weighted-graph crack-path prediction and the
  microstructure path-regulation design loop.
"""

from . import crystallography, graph_pf, hydrogen_network, path_search
from .crystallography import (
    disorientation_angle,
    euler_to_matrix,
    euler_to_quaternion,
    ipf_rgb,
    schmid_factor,
)
from .ebsd import EBSDMap, Phase, read_ctf
from .ebsd import load as load_ebsd
from .gb_chains import (
    GrainBoundaryChains,
    extract_grain_boundary_chains,
    load_grain_boundary_chains,
    save_grain_boundary_chains,
    semantic_sha256,
)
from .graph_pf import GraphPFModel, GraphPFParams, coarsen, crack_metrics, crack_path
from .hydrogen_network import HydrogenNetwork, HydrogenParams
from .micrograph import GrainGraph, GraphModelParams, build_graph, reconstruct_grains
from .path_search import (
    PathWeightParams,
    edge_weights,
    notch_and_far_grains,
    predict_path,
    regulate_path,
)

__all__ = [
    "crystallography",
    "graph_pf",
    "hydrogen_network",
    "path_search",
    "disorientation_angle",
    "euler_to_matrix",
    "euler_to_quaternion",
    "ipf_rgb",
    "schmid_factor",
    "EBSDMap",
    "Phase",
    "load_ebsd",
    "read_ctf",
    "GrainBoundaryChains",
    "extract_grain_boundary_chains",
    "load_grain_boundary_chains",
    "save_grain_boundary_chains",
    "semantic_sha256",
    "GrainGraph",
    "GraphModelParams",
    "build_graph",
    "reconstruct_grains",
    "GraphPFModel",
    "GraphPFParams",
    "coarsen",
    "crack_metrics",
    "crack_path",
    "HydrogenNetwork",
    "HydrogenParams",
    "PathWeightParams",
    "edge_weights",
    "notch_and_far_grains",
    "predict_path",
    "regulate_path",
]
