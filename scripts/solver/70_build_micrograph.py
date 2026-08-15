"""Case study 0 (foundation): EBSD -> attributed microstructure graph.

Reconstructs grains for both sample sections, builds the attributed grain graph
``G = (V, E)``, writes the statistics the manuscript quotes, and renders the
microstructure-graph figures: orientation (IPF), phase, grains, the grain-graph
overlay, the KAM (trap-density proxy) field, and the weak-boundary network that
previews the crack-susceptible paths.

Run:  python scripts/70_build_micrograph.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import _setup as S
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

from inverse_pfm import build_graph, load_ebsd
from inverse_pfm import crystallography as xtal
from inverse_pfm.ebsd import SECTION_AXES

plt.rcParams.update(S.RCPARAMS)

PHASE_COLORS = ListedColormap(["#1b1b1b", "#e4b34a", "#9bb7d4", "#c0504d"])
# 0 non-indexed (dark), 1 Fe3C (gold), 2 Iron bcc (blue-grey), 3 Iron fcc (red)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _ipf_image(ebsd):
    rgb = np.zeros((ebsd.ny, ebsd.nx, 3))
    idx = ebsd.indexed
    rgb[idx] = xtal.ipf_rgb(ebsd.euler[idx], sample_dir=(0, 0, 1))
    rgb[~idx] = 0.12
    return rgb


def _grain_image(G, seed=3):
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 0.95, size=(G.n_grains, 3))
    return colors[G.grain_id]


def _network_segments(G):
    p0 = G.centroid[G.edge_i]
    p1 = G.centroid[G.edge_j]
    return np.stack([p0, p1], axis=1)


def figure_section(sec, ebsd, G):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.2))
    ext = ebsd.extent

    # (a) IPF-Z orientation map
    ax = axes[0, 0]
    ax.imshow(_ipf_image(ebsd), extent=ext, origin="upper")
    ax.set_title("Orientation (IPF$_z$)")
    S.style_map(ax)

    # (b) phase map
    ax = axes[0, 1]
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], PHASE_COLORS.N)
    ax.imshow(ebsd.phase, extent=ext, origin="upper", cmap=PHASE_COLORS, norm=norm)
    ax.set_title("Phases")
    S.style_map(ax)
    from matplotlib.patches import Patch

    names = {p.number: p.name for p in ebsd.phases}
    handles = [Patch(facecolor="#1b1b1b", label="non-indexed")] + [
        Patch(facecolor=PHASE_COLORS(i), label=names.get(i, str(i))) for i in sorted(names)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.9)

    # (c) reconstructed grains
    ax = axes[0, 2]
    ax.imshow(_grain_image(G), extent=ext, origin="upper")
    ax.set_title(f"Grains (N = {G.n_grains})")
    S.style_map(ax)

    # (d) grain-graph overlay: nodes at centroids, edges coloured by misorientation
    ax = axes[1, 0]
    ax.imshow(np.ones((ebsd.ny, ebsd.nx)), extent=ext, origin="upper", cmap="Greys", vmin=0, vmax=1)
    segs = _network_segments(G)
    lc = LineCollection(segs, cmap="viridis", array=G.edge_theta, lw=0.45)
    lc.set_clim(0, 62.8)
    ax.add_collection(lc)
    sizes = 6.0 * G.equiv_diam / G.equiv_diam.mean()
    ax.scatter(
        G.centroid[:, 0],
        G.centroid[:, 1],
        s=sizes,
        c="#b2182b",
        edgecolors="none",
        alpha=0.7,
        zorder=3,
    )
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"misorientation $\theta$ (deg)")
    ax.set_title("Grain graph $G=(V,E)$")
    S.style_map(ax)

    # (e) KAM field (GND / trap-density proxy)
    ax = axes[1, 1]
    im = ax.imshow(
        G.kam_field, extent=ext, origin="upper", cmap="magma", vmin=0, vmax=S.GRAPH_PARAMS.kam_cap
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("KAM (deg)")
    ax.set_title("Stored dislocations (KAM)")
    S.style_map(ax)

    # (f) weak-boundary network: edges coloured by Gc ratio (crack-susceptible)
    ax = axes[1, 2]
    ax.imshow(np.ones((ebsd.ny, ebsd.nx)), extent=ext, origin="upper", cmap="Greys", vmin=0, vmax=1)
    order = np.argsort(-G.edge_Gc)  # draw weak (low-Gc) boundaries on top
    lc = LineCollection(segs[order], cmap="RdYlGn", array=G.edge_Gc[order], lw=0.6)
    lc.set_clim(1.0 - S.GRAPH_PARAMS.kgb_Gc, 1.0)
    ax.add_collection(lc)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$G_{c,ij}/G_{c,0}$")
    ax.set_title("Weak-boundary network")
    S.style_map(ax)

    S.panel_tags(axes)
    fig.suptitle(f"{sec.upper()} -- {S.SECTION_LABEL[sec]}", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(S.FIG_DIR, f"micrograph_graph_{sec}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_statistics(graphs):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    colors = {"ban": "#2166ac", "zhu": "#b2182b"}

    # grain-size distribution
    ax = axes[0, 0]
    for sec, G in graphs.items():
        ax.hist(
            G.equiv_diam,
            bins=np.linspace(0, 12, 40),
            density=True,
            histtype="step",
            lw=1.4,
            color=colors[sec],
            label=sec,
        )
    ax.set_xlabel(r"equivalent grain diameter ($\mu$m)")
    ax.set_ylabel("pdf")
    ax.legend(title="section")
    S.style_axes(ax)

    # misorientation distribution
    ax = axes[0, 1]
    for sec, G in graphs.items():
        ax.hist(
            G.edge_theta,
            bins=np.linspace(0, 62.8, 40),
            density=True,
            histtype="step",
            lw=1.4,
            color=colors[sec],
            label=sec,
        )
    ax.axvline(15.0, ls="--", lw=0.8, color="#666")
    ax.axvline(60.0, ls=":", lw=0.8, color="#666")
    ax.set_xlabel(r"boundary misorientation $\theta$ (deg)")
    ax.set_ylabel("pdf")
    S.style_axes(ax)

    # boundary-type fractions
    ax = axes[1, 0]
    width = 0.35
    labels = ["LAGB", "HAGB", r"special ($\Sigma$3)"]
    for k, (sec, G) in enumerate(graphs.items()):
        fr = [np.mean(G.edge_type == t) for t in (0, 1, 2)]
        ax.bar(
            np.arange(3) + (k - 0.5) * width, fr, width, color=colors[sec], label=sec, alpha=0.85
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("edge fraction")
    ax.legend()
    S.style_axes(ax)

    # boundary GB-H diffusivity ratio distribution
    ax = axes[1, 1]
    for sec, G in graphs.items():
        ax.hist(
            G.edge_DH,
            bins=np.linspace(1, S.GRAPH_PARAMS.beta_DH, 30),
            density=True,
            histtype="step",
            lw=1.4,
            color=colors[sec],
            label=sec,
        )
    ax.set_xlabel(r"GB hydrogen diffusivity ratio $D_{H,ij}/D_{H,0}$")
    ax.set_ylabel("pdf")
    S.style_axes(ax)

    S.panel_tags(axes)
    fig.tight_layout()
    path = os.path.join(S.FIG_DIR, "micrograph_statistics.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Statistics dump
# --------------------------------------------------------------------------- #


def section_stats(sec, ebsd, G):
    def frac(mask):
        return float(np.mean(mask))

    hist, edges = np.histogram(G.edge_theta, bins=np.linspace(0, 62.8, 32))
    return {
        "section": sec,
        "axes": list(SECTION_AXES[sec]),
        "grid": list(ebsd.shape),
        "step_um": [ebsd.dx, ebsd.dy],
        "index_rate": float(np.mean(ebsd.indexed)),
        "phase_fractions": ebsd.phase_fractions(),
        "n_grains": int(G.n_grains),
        "n_edges": int(G.n_edges),
        "grain_diam_um": {
            "mean": float(G.equiv_diam.mean()),
            "median": float(np.median(G.equiv_diam)),
            "p90": float(np.percentile(G.equiv_diam, 90)),
            "max": float(G.equiv_diam.max()),
        },
        "misorientation_deg": {
            "mean": float(G.edge_theta.mean()),
            "median": float(np.median(G.edge_theta)),
        },
        "boundary_fraction": {
            "LAGB": frac(G.edge_type == 0),
            "HAGB": frac(G.edge_type == 1),
            "special_sigma3": frac(G.edge_type == 2),
        },
        "schmid_mean": float(np.nanmean(G.schmid)),
        "kam_mean_deg": float(np.nanmean(G.node_kam)),
        "edge_Gc_ratio": {
            "mean": float(G.edge_Gc.mean()),
            "min": float(G.edge_Gc.min()),
        },
        "edge_DH_ratio_mean": float(G.edge_DH.mean()),
        "misorientation_histogram": {
            "counts": hist.astype(int).tolist(),
            "bin_edges_deg": edges.round(2).tolist(),
        },
    }


def main():
    graphs = {}
    summary = {}
    for sec in S.SECTIONS:
        ebsd = load_ebsd(sec)
        G = build_graph(ebsd, S.GRAPH_PARAMS)
        graphs[sec] = G
        summary[sec] = section_stats(sec, ebsd, G)
        path = figure_section(sec, ebsd, G)
        print(f"[fig] {path}")

    stat_path = figure_statistics(graphs)
    print(f"[fig] {stat_path}")

    out = os.path.join(S.DATA_DIR, "micrograph_stats.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[data] {out}")

    for sec in S.SECTIONS:
        s = summary[sec]
        print(
            f"\n{sec}: {s['n_grains']} grains, {s['n_edges']} edges | "
            f"d50={s['grain_diam_um']['median']:.2f} um | "
            f"HAGB={s['boundary_fraction']['HAGB']:.2f} | "
            f"Sigma3={s['boundary_fraction']['special_sigma3']:.2f}"
        )


if __name__ == "__main__":
    main()
