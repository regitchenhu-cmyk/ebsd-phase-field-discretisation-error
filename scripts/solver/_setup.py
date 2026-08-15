"""Shared setup and plotting style for the seventh-paper case studies.

All numbered scripts (70-74) import the EBSD sections, the graph model
parameters, the baseline phase-field / hydrogen magnitudes and the figure style
from here so the figures form one coherent story.

Unit convention
---------------
Lengths are in micrometres (the native EBSD step is ~0.4 um).  Stresses and the
applied load are normalised (the case studies compare crack *paths* and
*deflection* across models, not absolute load levels), and the fracture energy
``Gc`` is given in the same normalised energy-per-area units.  Hydrogen is
tracked as a normalised lattice occupancy in ``[0, 1]``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inverse_pfm import (  # noqa: E402
    GraphModelParams,
    GraphPFModel,
    GraphPFParams,
    build_graph,
    graph_pf,
    load_ebsd,
)

# Output locations -------------------------------------------------------- #
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures", "pdf")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# EBSD sections ----------------------------------------------------------- #
SECTIONS = ("ban", "zhu")
SECTION_LABEL = {"ban": "Rolling plane (RD-TD)", "zhu": "Columnar section (ND-RD)"}

# Microstructure-graph model --------------------------------------------- #
# Fine graph (10 deg) for characterisation (script 70); block/packet-scale
# graph (15 deg, larger minimum grain) for the mechanical case studies, where
# the high-angle block-boundary network is the crack-relevant topology.
GRAPH_PARAMS = GraphModelParams()
GRAPH_PARAMS_PF = GraphModelParams(theta_seg=15.0, min_grain_px=25)

# Baseline (bulk) phase-field / hydrogen magnitudes ----------------------- #
# The grain-boundary attribute ratios in the graph multiply these.
GC_BULK = 1.0  # bulk fracture energy (normalised)
ELL = 1.2  # phase-field regularisation length (um) ~ 3 px
E_MOD = 1.0  # normalised Young's modulus
NU = 0.3  # Poisson ratio
DH_BULK = 1.0  # bulk hydrogen diffusivity (normalised)
TRAP_BULK = 1.0  # bulk trap density (normalised)
CHI_H = 0.90  # hydrogen toughness-degradation coefficient (HEDE)

# Graph crack-path edge weight  w_e = Gc_e - a*cH_e - b*psi+_e + c*R_e ---- #
PATH_ALPHA = 0.6  # hydrogen susceptibility weight
PATH_BETA = 0.4  # mechanical driving-force weight
PATH_GAMMA = 0.3  # boundary-resistance weight
PATH_LIGAMENT = 1.2  # mode-I ligament-deviation penalty (per unit y/Ly)

# Phase-field solver resolution and the graph-coupling strength ----------- #
PF_COARSEN = 2  # EBSD -> solver downsampling factor
LAMBDA_GRAPH = 0.6  # graph-regulariser strength for the "graph" variant
N_LOAD_STEPS = 32  # incremental load steps per phase-field run


def pf_params(lambda_graph: float = 0.0) -> GraphPFParams:
    """A fresh phase-field parameter bundle (avoids dataclass aliasing)."""
    return GraphPFParams(
        E=E_MOD,
        nu=NU,
        ell=ELL,
        lambda_graph=lambda_graph,
        max_stagger=15,
        stagger_tol=4e-3,
    )


@dataclass
class Prep:
    """Everything a case study needs from one EBSD section at solver resolution."""

    sec: str
    ebsd: object
    graph: object
    factor: int
    shape: tuple
    step: tuple
    grain_c: object  # coarsened grain-id map
    kam_c: object
    Gc_c: object  # rasterised toughness field (no hydrogen)
    DH_c: object
    trap_c: object
    boundary_c: object


def prepare_section(
    sec: str, factor: int = PF_COARSEN, graph_params: GraphModelParams = GRAPH_PARAMS_PF
) -> Prep:
    """Load a section, build its graph and rasterise fields at solver resolution."""
    ebsd = load_ebsd(sec)
    G = build_graph(ebsd, graph_params)
    grain_c = graph_pf.coarsen(G.grain_id, factor, "mode")
    kam_c = graph_pf.coarsen(G.kam_field, factor, "mean")
    fields = G.rasterize_fields(GC_BULK, DH_BULK, TRAP_BULK, grain_id=grain_c, kam_field=kam_c)
    return Prep(
        sec=sec,
        ebsd=ebsd,
        graph=G,
        factor=factor,
        shape=grain_c.shape,
        step=(ebsd.dx * factor, ebsd.dy * factor),
        grain_c=grain_c,
        kam_c=kam_c,
        Gc_c=fields["Gc"],
        DH_c=fields["D_H"],
        trap_c=fields["rho_trap"],
        boundary_c=fields["boundary"],
    )


def build_model(
    prep: Prep,
    gc_field=None,
    lambda_graph: float = 0.0,
    notch_frac: float = 0.20,
    notch_half: int = 1,
) -> GraphPFModel:
    """Configure a :class:`GraphPFModel` for a section (notch + mode-I tension)."""
    ny, nx = prep.shape
    model = GraphPFModel(ny, nx, prep.step[0], prep.step[1], pf_params(lambda_graph))
    Gc = prep.Gc_c if gc_field is None else gc_field
    weak = 1.0 - prep.graph.edge_Gc  # boundary weakness drives the coupling
    W = prep.graph.adjacency(weight=weak)
    model.set_fields(Gc=Gc, grain_id=prep.grain_c, W=W)
    model.set_bc_tension()
    mid = ny // 2
    model.set_notch(mid - notch_half, mid + notch_half + 1, 0, int(notch_frac * nx))
    return model


# --------------------------------------------------------------------------- #
# Plotting style (shared rcParams and helpers)
# --------------------------------------------------------------------------- #
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def style_axes(ax, grid_axis="y"):
    if grid_axis:
        ax.grid(True, axis=grid_axis, ls=":", lw=0.5, color="#9A9A9A", alpha=0.6)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both", top=True, right=True)


def style_map(ax):
    """Axis styling for spatial micrographs (equal aspect, micron labels)."""
    ax.set_aspect("equal")
    ax.set_xlabel(r"x ($\mu$m)")
    ax.set_ylabel(r"y ($\mu$m)")
    ax.tick_params(direction="out", which="both")


def panel_tags(axes, tags=None, dx=-0.14, dy=1.04):
    import numpy as np

    tags = tags or ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for tag, ax in zip(tags, np.ravel(axes), strict=False):
        ax.text(dx, dy, tag, transform=ax.transAxes, fontweight="bold", fontsize=10)
