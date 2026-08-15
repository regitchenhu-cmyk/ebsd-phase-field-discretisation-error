from pathlib import Path
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


ROOT = Path(r"E:\models\PFM_EBSD\IJSS_reconstruction")
FIG_DIR = ROOT / "figures_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "source_data_v2" / "spatial_fields_export.npz"
MEDIA = ROOT / "original_media"
GRAPH_PIVOT = Path(r"E:\models\PFM_EBSD\IJSS_graph_pivot")
MODEL = next(
    Path("D:/models").glob(
        "PFM*/PFM_H_GraphFracture/PFM_H_GraphFracture"
    )
)
CTF = next(Path("D:/models").glob("PFM*/EBSD data/EBSD data/ban/ban.ctf"))
CHAIN = MODEL / "data" / "ebsd_gb_chains" / "ban_roi.npz"
RESULTS = MODEL / "results"

INK = "#17324d"
BLUE = "#2f6f91"
TEAL = "#2c8c84"
GOLD = "#d29a3a"
RED = "#b8534f"
VIOLET = "#655a9e"
GREY = "#687986"
LIGHT = "#eef2f1"
PAPER = "#fbfaf6"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.7,
        "axes.titlesize": 9.4,
        "axes.labelsize": 8.8,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)

FIELD_CMAP = LinearSegmentedColormap.from_list(
    "field", ["#102f4a", "#2c7f91", "#8bc3ae", "#f1d77b", "#bd5547"]
)
DAMAGE_CMAP = LinearSegmentedColormap.from_list(
    "damage", ["#f7f5ee", "#e7c96a", "#d96b45", "#7f2339"]
)
PHASE_CMAP = ListedColormap(["#d9d9d9", "#d99045", "#2e7fa6", "#63a76e"])


def panel_label(ax, label, x=0.015, y=0.985):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
        zorder=20,
    )


def clean_axes(ax, grid=True):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.75)
    ax.tick_params(
        direction="in",
        length=3,
        width=0.7,
        color="black",
        top=True,
        right=True,
    )
    if grid:
        ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.65, zorder=0)
        ax.set_axisbelow(True)


def add_colorbar(fig, artist, ax, label, fraction=0.048):
    cb = fig.colorbar(artist, ax=ax, fraction=fraction, pad=0.025)
    cb.set_label(label, fontsize=8.2)
    cb.ax.tick_params(labelsize=7.5, length=2.5)
    cb.outline.set_linewidth(0.55)
    return cb


def save_figure(fig, stem):
    for ext, kwargs in {
        "png": {"dpi": 600},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_ctf():
    lines = CTF.read_text(encoding="utf-8", errors="replace").splitlines()
    xcells = int(next(line.split("\t")[1] for line in lines if line.startswith("XCells")))
    ycells = int(next(line.split("\t")[1] for line in lines if line.startswith("YCells")))
    xstep = float(next(line.split("\t")[1] for line in lines if line.startswith("XStep")))
    ystep = float(next(line.split("\t")[1] for line in lines if line.startswith("YStep")))
    header = next(i for i, line in enumerate(lines) if line.startswith("Phase\tX\tY"))
    data = np.loadtxt(CTF, skiprows=header + 1)
    fields = {
        "phase": data[:, 0].reshape(ycells, xcells),
        "euler1": data[:, 5].reshape(ycells, xcells),
        "euler2": data[:, 6].reshape(ycells, xcells),
        "euler3": data[:, 7].reshape(ycells, xcells),
        "mad": data[:, 8].reshape(ycells, xcells),
        "bc": data[:, 9].reshape(ycells, xcells),
        "bs": data[:, 10].reshape(ycells, xcells),
    }
    return fields, xcells, ycells, xstep, ystep


def ipf_nd_rgb(phi1, Phi, phi2, indexed):
    del phi1
    P = np.deg2rad(Phi)
    p2 = np.deg2rad(phi2)
    direction = np.stack(
        [np.sin(p2) * np.sin(P), np.cos(p2) * np.sin(P), np.cos(P)],
        axis=-1,
    )
    direction = np.sort(np.abs(direction), axis=-1)[..., ::-1]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / np.maximum(norm, 1.0e-12)
    h, k, l = direction[..., 0], direction[..., 1], direction[..., 2]
    sx = k / np.maximum(1.0 + h, 1.0e-12)
    sy = l / np.maximum(1.0 + h, 1.0e-12)
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0 / (1.0 + math.sqrt(2.0)), 0.0],
            [
                (1.0 / math.sqrt(3.0)) / (1.0 + 1.0 / math.sqrt(3.0)),
                (1.0 / math.sqrt(3.0)) / (1.0 + 1.0 / math.sqrt(3.0)),
            ],
        ]
    )
    matrix = np.array(
        [
            [vertices[0, 0], vertices[1, 0], vertices[2, 0]],
            [vertices[0, 1], vertices[1, 1], vertices[2, 1]],
            [1.0, 1.0, 1.0],
        ]
    )
    rhs = np.stack([sx, sy, np.ones_like(sx)], axis=0).reshape(3, -1)
    bary = np.linalg.solve(matrix, rhs).T.reshape(sx.shape + (3,))
    bary = np.clip(bary, 0.0, 1.0)
    rgb = np.sqrt(bary)
    rgb = rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1.0e-12)
    rgb[~indexed] = np.array([0.78, 0.78, 0.78])
    return rgb


def load_chain_lines():
    with np.load(CHAIN, allow_pickle=False) as z:
        points = z["points"]
        offsets = z["chain_offsets"]
        point_ids = z["chain_point_ids"]
        confidence = z["confidence_fraction"]
    lines = []
    retained = []
    for i in range(len(offsets) - 1):
        ids = point_ids[offsets[i] : offsets[i + 1]]
        if len(ids) < 2:
            continue
        line = points[ids] * 1000.0
        lines.append(line)
        if confidence[i] >= 1.0 - 1.0e-12:
            retained.append(line)
    return lines, retained


def triangulation(geometry, topology):
    xy = geometry * 1000.0
    return mtri.Triangulation(xy[:, 0], xy[:, 1], topology)


def add_precrack(ax):
    ax.plot([0, 24.0975], [24.0975, 24.0975], color="black", linewidth=1.4, zorder=15)
    ax.plot(24.0975, 24.0975, marker=">", color="black", markersize=4.5, zorder=16)


def figure_1_workflow():
    fig, ax = plt.subplots(figsize=(7.25, 6.60))
    ax.set_xlim(0, 12)
    ax.set_ylim(-3.95, 6.9)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.add_patch(
        Rectangle(
            (0.22, 3.45),
            11.55,
            3.10,
            fill=False,
            edgecolor="black",
            linewidth=0.9,
            linestyle=(0, (4, 3)),
        )
    )
    ax.text(0.35, 6.38, "A", color=RED, fontsize=9.5, fontweight="bold")
    ax.text(
        0.67,
        6.38,
        "Forward transformation",
        color="black",
        fontsize=8.6,
        fontweight="bold",
    )

    stages = [
        ("A1", "EBSD observation", "Phase, BC, MAD\nIPF-ND and ROI"),
        ("A2", "Boundary polylines", "Pixel-derived geometry\nand indexed-segment fraction"),
        ("A3", "Continuum fields", r"$G_{c0}$, $D_H$, $N_T$" + "\nfixed physical width"),
        ("A4", "Transient precharge", r"$c_L$, $\theta$" + "\nfrozen terminal state"),
        ("A5", "AT2 fracture", "SNESVI bounds\nadaptive continuation"),
    ]
    xs = [0.46, 2.72, 4.98, 7.24, 9.50]
    for x, (num, title, body) in zip(xs, stages):
        box = FancyBboxPatch(
            (x, 4.03),
            1.82,
            1.78,
            boxstyle="square,pad=0.025",
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
        )
        ax.add_patch(box)
        ax.text(x + 0.11, 5.58, num, color=RED, fontsize=7.8, fontweight="bold")
        ax.text(x + 0.11, 5.25, title, color="black", fontsize=7.5, fontweight="bold")
        ax.text(
            x + 0.11,
            4.82,
            body,
            color="black",
            fontsize=6.7,
            va="top",
            linespacing=1.35,
        )
    for x1, x2 in zip([2.28, 4.54, 6.80, 9.06], [2.72, 4.98, 7.24, 9.50]):
        ax.add_patch(
            FancyArrowPatch(
                (x1, 4.92),
                (x2, 4.92),
                arrowstyle="-|>",
                mutation_scale=9,
                color="black",
                linewidth=0.9,
            )
        )

    ax.add_patch(
        Rectangle(
            (0.22, 0.45),
            11.55,
            2.48,
            fill=False,
            edgecolor="black",
            linewidth=0.9,
            linestyle=(0, (4, 3)),
        )
    )
    ax.text(0.35, 2.75, "B", color=RED, fontsize=9.5, fontweight="bold")
    ax.text(
        0.67,
        2.75,
        "Verification and interpretation checks",
        color="black",
        fontsize=8.6,
        fontweight="bold",
    )
    gates = [
        ("B1", "Source data", "CTF map and extracted\nboundary polylines"),
        ("B2", "Numerical checks", "mesh, load step,\nenergy and KKT"),
        ("B3", "Interpretation", "numerical error,\nconvergence and scope"),
    ]
    widths = [3.20, 3.20, 3.20]
    x = 0.70
    gate_centres = []
    for (code, title, body), width in zip(gates, widths):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.85),
                width,
                1.35,
                boxstyle="square,pad=0.025",
                facecolor="white",
                edgecolor="black",
                linewidth=0.85,
            )
        )
        ax.text(x + 0.13, 1.95, code, color=RED, fontsize=7.7, fontweight="bold")
        ax.text(x + 0.52, 1.95, title, color="black", fontsize=7.5, fontweight="bold")
        ax.text(
            x + 0.13,
            1.48,
            body,
            color="black",
            fontsize=6.8,
            va="center",
            linespacing=1.3,
        )
        gate_centres.append(x + width / 2)
        x += width + 0.44
    for x0, x1 in zip(gate_centres, [1.37, 6.00, 10.41]):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 2.20),
                (x1, 4.02),
                arrowstyle="-|>",
                mutation_scale=8,
                color="black",
                linewidth=0.75,
                linestyle=(0, (3, 2)),
            )
        )
    ax.text(
        11.55,
        0.12,
        "Mechanical interpretation requires a common converged state and an effect larger than numerical error.",
        ha="right",
        color="black",
        fontsize=7.3,
    )
    ax.text(
        -0.004,
        0.985,
        "a",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="top",
        clip_on=False,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
    )

    bvp = ax.inset_axes([0.12, 0.015, 0.76, 0.285])
    bvp.set_xlim(-0.22, 2.30)
    bvp.set_ylim(-0.30, 1.38)
    bvp.add_patch(Rectangle((0, 0), 2.0, 1.0, facecolor="#F7F4EC", edgecolor="#16324F", lw=1.2))
    bvp.plot([0, 0.50], [0.50, 0.50], color="#B6403A", lw=3.0, solid_capstyle="butt")
    bvp.plot(0.50, 0.50, marker=">", ms=5.5, color="#B6403A")
    for xpos in np.linspace(0.18, 1.82, 5):
        bvp.add_patch(
            FancyArrowPatch(
                (xpos, 1.02),
                (xpos, 1.28),
                arrowstyle="-|>",
                mutation_scale=8,
                lw=0.8,
                color="#2C7FB8",
            )
        )
    bvp.plot([0.0, 2.0], [0.0, 0.0], color="#16324F", lw=1.0)
    for xpos in np.linspace(0.10, 1.90, 7):
        bvp.plot(
            [xpos - 0.055, xpos + 0.055],
            [-0.08, 0.0],
            color="#16324F",
            lw=0.65,
            marker="",
        )
    bvp.plot(0.0, 0.0, marker="s", ms=4.2, color="#16324F")
    bvp.text(1.00, 1.30, r"prescribed $u_y$", ha="center", va="bottom")
    bvp.text(1.00, -0.19, r"$u_y=0$; lower-left $u_x=0$", ha="center", va="top")
    bvp.text(0.24, 0.57, r"$a_0=H/2=24.10~\mu$m", color="#8F2D2A", ha="center", va="bottom")
    bvp.text(
        2.06,
        0.50,
        r"$H=48.195~\mu$m",
        rotation=270,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )
    bvp.text(1.00, 0.78, r"$L=96.39~\mu$m; plane strain; unit thickness", ha="center", va="center")
    bvp.text(-0.19, 1.31, "b", fontsize=10, fontweight="bold", ha="left", va="top")
    bvp.axis("off")

    save_figure(fig, "figure_1_workflow_v2")


def figure_2_ebsd_observation():
    fields, nx, ny, dx, dy = load_ctf()
    x0, x1, y0, y1 = 0, 238, 54, 173
    indexed = fields["phase"] > 0
    ipf = ipf_nd_rgb(fields["euler1"], fields["euler2"], fields["euler3"], indexed)
    full_extent = [0, nx * dx, 0, ny * dy]
    roi_extent = [0, (x1 - x0) * dx, 0, (y1 - y0) * dy]
    phase_roi = np.flipud(fields["phase"][y0:y1, x0:x1])
    bc_roi = np.flipud(fields["bc"][y0:y1, x0:x1])
    mad_roi = np.flipud(fields["mad"][y0:y1, x0:x1])
    ipf_roi = np.flipud(ipf[y0:y1, x0:x1])
    _, retained = load_chain_lines()

    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.15), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(np.flipud(fields["phase"]), extent=full_extent, cmap=PHASE_CMAP, vmin=0, vmax=3, interpolation="nearest")
    roi_y = (ny - y1) * dy
    ax.add_patch(Rectangle((x0 * dx, roi_y), (x1 - x0) * dx, (y1 - y0) * dy, fill=False, edgecolor=RED, linewidth=1.2))
    ax.text(2, roi_y + 2, "mechanics ROI", color=RED, fontsize=7, fontweight="bold")
    ax.set_title("Full measured phase map")
    ax.set_xlabel("RD (micrometres)")
    ax.set_ylabel("TD (micrometres)")
    panel_label(ax, "a")

    ax = axes[0, 1]
    im = ax.imshow(phase_roi, extent=roi_extent, cmap=PHASE_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("ROI phase ID")
    ax.set_xlabel("RD (micrometres)")
    ax.set_ylabel("TD (micrometres)")
    panel_label(ax, "b")

    ax = axes[1, 0]
    im = ax.imshow(bc_roi, extent=roi_extent, cmap="gray", interpolation="nearest")
    ax.set_title("Band contrast")
    ax.set_xlabel("RD (micrometres)")
    ax.set_ylabel("TD (micrometres)")
    add_colorbar(fig, im, ax, "BC")
    panel_label(ax, "c")

    ax = axes[1, 1]
    im = ax.imshow(mad_roi, extent=roi_extent, cmap="magma_r", vmin=0, vmax=np.nanpercentile(mad_roi, 99), interpolation="nearest")
    ax.set_title("Mean angular deviation")
    ax.set_xlabel("RD (micrometres)")
    ax.set_ylabel("TD (micrometres)")
    add_colorbar(fig, im, ax, "MAD (deg)")
    panel_label(ax, "d")

    ax = axes[2, 0]
    ax.imshow(ipf_roi, extent=roi_extent, interpolation="nearest")
    ax.set_title("Cubic IPF-ND reconstruction")
    ax.set_xlabel("RD (micrometres)")
    ax.set_ylabel("TD (micrometres)")
    panel_label(ax, "e")

    ax = axes[2, 1]
    ax.imshow(ipf_roi, extent=roi_extent, interpolation="nearest", alpha=0.72)
    ax.add_collection(LineCollection(retained, colors=INK, linewidths=0.24, alpha=0.82))
    ax.set_xlim(0, 96.39)
    ax.set_ylim(0, 48.195)
    ax.set_title("Fully indexed boundary polylines")
    ax.set_xlabel("RD (micrometres)")
    ax.set_ylabel("TD (micrometres)")
    ax.text(
        94,
        3.0,
        "568 polylines\n1,825 pixel-edge segments",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=INK,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=2.0),
    )
    panel_label(ax, "f")
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.tick_params(length=2.5)
    save_figure(fig, "figure_2_ebsd_observation")


def figure_3_graph_fields_hydrogen(spatial):
    all_lines, retained = load_chain_lines()
    geometry = spatial["geometry"]
    topology = spatial["topology"]
    tri = triangulation(geometry, topology)
    g0 = spatial["base_fracture_toughness"] / 2.7
    geff = spatial["effective_fracture_toughness"] / 2.7
    dh = spatial["hydrogen_diffusivity"] / 1.0e-4
    htri = triangulation(spatial["hydrogen_geometry"], spatial["hydrogen_topology"])
    c_l = spatial["lattice_hydrogen"]

    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.15), constrained_layout=True)
    ax = axes[0, 0]
    ax.add_collection(LineCollection(all_lines, colors="#b8c0c4", linewidths=0.18, alpha=0.72))
    ax.set_xlim(0, 96.39)
    ax.set_ylim(0, 48.195)
    ax.set_title("All source boundary polylines")
    ax.text(93, 3, "3,771 polylines", ha="right", fontsize=7, color=INK)
    ax.text(
        3,
        45,
        r"92.55% area below $0.6G_c^{bulk}$" + "\n" + r"at $b=1.5$ micrometres",
        va="top",
        fontsize=6.3,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.8),
    )
    panel_label(ax, "a")

    ax = axes[0, 1]
    ax.add_collection(LineCollection(retained, colors=TEAL, linewidths=0.32, alpha=0.9))
    ax.set_xlim(0, 96.39)
    ax.set_ylim(0, 48.195)
    ax.set_title("Fully indexed boundary polylines")
    ax.text(93, 3, "568 polylines", ha="right", fontsize=7, color=INK)
    ax.text(
        3,
        45,
        r"27.27% area below $0.6G_c^{bulk}$" + "\n" + "fully indexed selection",
        va="top",
        fontsize=6.3,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.8),
    )
    panel_label(ax, "b")

    ax = axes[1, 0]
    im = ax.tripcolor(tri, facecolors=g0, shading="flat", cmap=FIELD_CMAP, vmin=0.45, vmax=1.0)
    add_precrack(ax)
    ax.set_title("Mapped base toughness")
    add_colorbar(fig, im, ax, r"$G_{c0}/G_c^{bulk}$")
    panel_label(ax, "c")

    ax = axes[1, 1]
    im = ax.tripcolor(tri, facecolors=dh, shading="flat", cmap="viridis", vmin=1, vmax=12)
    add_precrack(ax)
    ax.set_title("Mapped diffusivity")
    add_colorbar(fig, im, ax, r"$D_H/D_{bulk}$")
    panel_label(ax, "d")

    ax = axes[2, 0]
    c_l_mean = float(np.mean(c_l))
    delta_c_l = (c_l - c_l_mean) * 1.0e9
    limit = max(abs(float(np.min(delta_c_l))), abs(float(np.max(delta_c_l))), 1.0e-12)
    im = ax.tripcolor(htri, delta_c_l, shading="gouraud", cmap="coolwarm", vmin=-limit, vmax=limit)
    add_precrack(ax)
    ax.set_title("500 s terminal deviation (numerical zero)")
    add_colorbar(fig, im, ax, r"$10^{9}(c_L-\overline{c_L})$")
    ax.text(
        92,
        45,
        fr"$\overline{{c_L}}={c_l_mean:.6f}$",
        ha="right",
        va="top",
        fontsize=6.5,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.8),
    )
    panel_label(ax, "e")

    ax = axes[2, 1]
    im = ax.tripcolor(tri, facecolors=geff, shading="flat", cmap=FIELD_CMAP, vmin=0.2, vmax=1.0)
    add_precrack(ax)
    ax.set_title("Effective toughness")
    add_colorbar(fig, im, ax, r"$G_c^{eff}/G_c^{bulk}$")
    ax.text(
        92,
        3.0,
        r"$N_T=0$ in measured dataset",
        ha="right",
        fontsize=8.0,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
    )
    panel_label(ax, "f")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xlabel("x (micrometres)")
        ax.set_ylabel("y (micrometres)")
        ax.tick_params(length=2.4)
    save_figure(fig, "figure_3_graph_fields_hydrogen")


def find_column(frame, names):
    for name in names:
        if name in frame.columns:
            return name
    lowered = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"none of {names} found in {list(frame.columns)}")


def figure_4_pilot():
    history = pd.read_csv(RESULTS / "ebsd_ban_confidence1_onset_pilot" / "history.csv")
    xcol = find_column(history, ["displacement", "prescribed_displacement"])
    rcol = find_column(history, ["reaction_y", "reaction"])
    lcol = find_column(history, ["regularised_crack_length", "regularized_crack_length"])
    kcol = find_column(history, ["damage_kkt_inf", "damage_kkt_relative"])
    ecol = find_column(history, ["energy_balance_relative"])
    x = history[xcol].to_numpy(float) * 1000.0
    reaction = history[rcol].to_numpy(float)
    lreg = history[lcol].to_numpy(float)
    kkt = np.maximum(history[kcol].to_numpy(float), 1.0e-14)
    energy = np.maximum(np.abs(history[ecol].to_numpy(float)), 1.0e-14)
    positive = x > 0
    secant = reaction[positive] / x[positive]
    secant = secant / secant[0]
    tangent = np.gradient(reaction, x, edge_order=1)
    tangent = tangent[positive] / tangent[positive][0]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.45))
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.13,
        top=0.94,
        hspace=0.58,
        wspace=0.34,
    )
    ax = axes[0, 0]
    ax.plot(x, reaction, color=BLUE, marker="o", markevery=max(1, len(x) // 8), label="reaction")
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel(r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    ax.set_title("Boundary-property-field reference response")
    clean_axes(ax)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(x[positive], 100 * (secant - 1), color=TEAL, marker="o", label="secant")
    ax.plot(x[positive], 100 * (tangent - 1), color=GOLD, marker="s", label="incremental")
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel("stiffness change (%)")
    ax.set_title("Distributed softening")
    ax.legend(fontsize=8.4, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, borderaxespad=0.0)
    clean_axes(ax)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.plot(x, 100 * (lreg / lreg[0] - 1), color=VIOLET, marker="o", markevery=max(1, len(x) // 8))
    ax.axhline(3.60, color=VIOLET, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel(r"$\Delta L_{reg}$ (%)")
    ax.set_title("Regularized crack measure")
    clean_axes(ax)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    ax.semilogy(x, kkt, color=RED, marker="o", markevery=max(1, len(x) // 8), label="damage KKT")
    ax.semilogy(x, energy, color=BLUE, marker="s", markevery=max(1, len(x) // 8), label="energy balance")
    ax.axhline(1.0e-6, color=RED, linestyle=":", linewidth=0.9)
    ax.axhline(1.0e-3, color=BLUE, linestyle=":", linewidth=0.9)
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel("relative residual")
    ax.set_title("Accepted-state diagnostics")
    ax.legend(fontsize=8.8, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20), borderaxespad=0.0)
    clean_axes(ax)
    panel_label(ax, "(d)")
    save_figure(fig, "figure_4_pilot_diagnostics")


def figure_5_synthetic():
    labels = ["uniform\nno H", "uniform\nH", "boundary field\nno H", "boundary field\nH"]
    reactions = np.array([1024.752, 991.392, 1019.815, 991.213])
    colors = [GREY, BLUE, GOLD, RED]
    h_effect = np.array([-3.255, -2.805])
    g_effect = np.array([-0.482, -0.018])
    crack = np.array([0, 0, 0, 1])

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.75), constrained_layout=True)
    ax = axes[0, 0]
    bars = ax.bar(np.arange(4), reactions, color=colors, width=0.68)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel(r"endpoint reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    ax.set_title("Common endpoint response")
    ax.set_ylim(970, 1032)
    ax.bar_label(bars, fmt="%.1f", fontsize=6.7, padding=2)
    clean_axes(ax, grid=False)
    panel_label(ax, "a")

    ax = axes[0, 1]
    bars = ax.bar(["uniform", "boundary field"], h_effect, color=[BLUE, RED], width=0.62)
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_ylabel("hydrogen contrast (%)")
    ax.set_title("Hydrogen effect")
    ax.bar_label(bars, fmt="%.3f%%", fontsize=7, padding=2)
    ax.set_ylim(-3.7, 0.2)
    clean_axes(ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    bars = ax.bar(["H off", "H on"], g_effect, color=[GOLD, RED], width=0.62)
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_ylabel("boundary-field contrast (%)")
    ax.set_title("Boundary-property-field effect")
    ax.bar_label(bars, fmt="%.3f%%", fontsize=7, padding=2)
    ax.set_ylim(-0.60, 0.08)
    clean_axes(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    x_indicator = np.arange(4)
    ax.plot(x_indicator, crack, color=GREY, linewidth=1.25, zorder=1)
    for xi, yi, color, marker in zip(
        x_indicator,
        crack,
        colors,
        ["o", "s", "^", "D"],
    ):
        ax.scatter(
            xi,
            yi,
            s=88,
            c=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.set_xticks(np.arange(4), labels)
    ax.set_yticks([0, 1], ["no", "yes"])
    ax.set_ylim(-0.25, 1.25)
    ax.set_ylabel(r"connected $d\geq0.7$ advance")
    ax.set_title("Threshold-front indicator")
    clean_axes(ax)
    panel_label(ax, "d")
    save_figure(fig, "figure_5_synthetic_factor")


def contains(frame, pattern):
    return frame[frame["case_id"].astype(str).str.contains(pattern, case=False, regex=True)].copy()


def figure_6_verification():
    raw = pd.read_csv(GRAPH_PIVOT / "source_data" / "figure_4_5_numerical_source_data.csv")
    summary = pd.read_csv(GRAPH_PIVOT / "source_data" / "numerical_case_summary.csv")
    mesh_ids = ["mesh_h1p54", "mesh_h2p03", "mesh_h2p55", "mesh_h3p04"]
    mesh_colors = [GREY, BLUE, TEAL, RED]
    ratios = [1.54, 2.03, 2.55, 3.04]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.45))
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.11,
        top=0.94,
        hspace=0.60,
        wspace=0.34,
    )
    ax = axes[0, 0]
    for case_id, color, ratio in zip(mesh_ids, mesh_colors, ratios):
        frame = raw[raw["case_id"] == case_id].sort_values("displacement")
        ax.plot(frame["displacement_um"], frame["reaction_y"], color=color, label=fr"$\ell/h_K={ratio:.2f}$")
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel(r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    ax.set_title("Low-load mesh series")
    fig.legend(
        fontsize=8.4,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        loc="center",
        bbox_to_anchor=(0.5, 0.52),
        ncol=4,
        borderaxespad=0.0,
    )
    clean_axes(ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    mesh_summary = summary[summary["case_id"].isin(mesh_ids)].set_index("case_id").loc[mesh_ids]
    reaction = mesh_summary["endpoint_reaction_N_per_mm"].to_numpy(float)
    lreg = mesh_summary["endpoint_regularised_crack_length_mm"].to_numpy(float)
    ax.plot(ratios, reaction, color=BLUE, marker="o", label="reaction")
    ax.set_xlabel(r"$\ell/h_K$")
    ax.set_ylabel(r"endpoint reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx()
    ax2.plot(ratios, lreg, color=RED, marker="s", label=r"$L_{reg}$")
    ax2.set_ylabel(r"$L_{reg}$ (mm)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax.set_title("Endpoint convergence")
    clean_axes(ax)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.75)
    ax2.tick_params(direction="in", top=True, right=True)
    panel_label(ax, "b")

    ax = axes[1, 0]
    step_rows = []
    for token, fallback in [
        ("steps15|dt_steps15", 174.9718),
        ("steps30|dt_steps30", 174.9768),
        ("steps60|dt_steps60", 174.9789),
    ]:
        frame = contains(summary, token)
        step_rows.append(float(frame.iloc[0]["endpoint_reaction_N_per_mm"]) if not frame.empty else fallback)
    bars = ax.bar(["15", "30", "60"], step_rows, color=[GOLD, BLUE, TEAL], width=0.62)
    ax.set_ylim(min(step_rows) - 0.015, max(step_rows) + 0.015)
    ax.set_xlabel("requested load steps")
    ax.set_ylabel(r"endpoint reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    ax.set_title("Load-step sensitivity")
    ax.bar_label(bars, fmt="%.4f", fontsize=6.8, padding=2)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    clean_axes(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    metrics = ["reaction", r"$L_{reg}$", "tip position"]
    coarse = [287.1466, 0.03325275, 0.0]
    fine = [285.0939, 0.03291074, 0.41547]
    changes = [0.7148, 1.0285, 1.0]
    x = np.arange(3)
    ax.bar(x, changes, color=[BLUE, TEAL, RED], width=0.62)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("reported difference")
    ax.set_title("Extended baseline: two-mesh difference")
    ax.text(0, changes[0] + 0.035, "-0.715%", ha="center", fontsize=7)
    ax.text(1, changes[1] + 0.035, "-1.029%", ha="center", fontsize=7)
    ax.text(2, changes[2] + 0.035, "1.0 fine cell\n(0.415 micrometres)", ha="center", fontsize=6.7)
    ax.set_ylim(0, 1.22)
    clean_axes(ax)
    panel_label(ax, "d")
    save_figure(fig, "figure_6_numerical_verification")


def factorial_frames(raw):
    no_h = contains(raw, "factorial_homogeneous_noH|homogeneous_noH")
    h = contains(raw, "factorial_homogeneous_H|homogeneous_H")
    graph_h = contains(raw, "factorial_graph_H|graph_H")
    return [
        ("homogeneous, no H", no_h, GREY),
        ("homogeneous, H", h, BLUE),
        ("boundary-property field, H", graph_h, RED),
    ]


def figure_7_measured_factorial():
    raw = pd.read_csv(GRAPH_PIVOT / "source_data" / "figure_4_5_numerical_source_data.csv")
    frames = factorial_frames(raw)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.65))
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.11,
        top=0.94,
        hspace=0.60,
        wspace=0.34,
    )

    ax = axes[0, 0]
    for label, frame, color in frames:
        if frame.empty:
            continue
        frame = frame.sort_values("displacement").drop_duplicates("displacement", keep="last")
        ax.plot(frame["displacement_um"], frame["reaction_y"], color=color, label=label)
    ax.axvline(0.48195, color=GOLD, linestyle="--", linewidth=0.9, label="1% nominal-strain limit")
    ax.axvline(0.7811426, color=RED, linestyle=":", linewidth=0.9)
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel(r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    ax.set_title("Measured-domain response")
    fig.legend(
        fontsize=8.4,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        loc="center",
        bbox_to_anchor=(0.5, 0.52),
        ncol=4,
        borderaxespad=0.0,
    )
    clean_axes(ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    for label, frame, color in frames:
        if frame.empty:
            continue
        frame = frame.sort_values("displacement").drop_duplicates("displacement", keep="last")
        ax.plot(frame["displacement_um"], frame["regularised_crack_length"], color=color, label=label)
    ax.axvline(0.48195, color=GOLD, linestyle="--", linewidth=0.9)
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel(r"$L_{reg}$ (mm)")
    ax.set_title("Regularized crack evolution")
    clean_axes(ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    for label, frame, color in frames:
        if frame.empty:
            continue
        frame = frame.sort_values("displacement").drop_duplicates("displacement", keep="last")
        ax.plot(frame["displacement_um"], frame["crack_front_extension_um"], color=color, label=label)
    ax.axvline(0.48195, color=GOLD, linestyle="--", linewidth=0.9)
    ax.set_xlabel("top displacement (micrometres)")
    ax.set_ylabel(r"connected $d\geq0.7$ extension (micrometres)")
    ax.set_title("Threshold-front metric")
    clean_axes(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    labels = [
        "homogeneous H\nat 0.9 micrometres",
        "boundary field + H vs homogeneous + H\nmatched displacement",
        "boundary field + H vs no H\nmatched displacement",
    ]
    values = [-7.017, -6.357, -11.772]
    bars = ax.barh(np.arange(3), values, color=[BLUE, TEAL, RED], height=0.62)
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_yticks(np.arange(3), labels)
    ax.set_xlabel("reaction contrast (%)")
    ax.set_title("Controlled contrasts")
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}%",
            ha="left",
            va="center",
            color="white",
            fontsize=6.8,
            fontweight="bold",
        )
    ax.tick_params(axis="y", labelsize=6.6)
    ax.set_xlim(-13.2, 0.5)
    clean_axes(ax)
    panel_label(ax, "d")
    save_figure(fig, "figure_7_measured_factorial")


def figure_8_damage_continuation(spatial):
    noh_tri = triangulation(spatial["homogeneous_noh_geometry"], spatial["homogeneous_noh_topology"])
    h_tri = triangulation(spatial["homogeneous_h_geometry"], spatial["homogeneous_h_topology"])
    graph_tri = triangulation(spatial["graph_damage_geometry"], spatial["graph_damage_topology"])
    graph_damage = spatial["graph_damage"][-1]
    graph_time = float(spatial["graph_damage_times"][-1]) * 1000.0

    attempt_path = RESULTS / "ijss_ban_factorial_graph_H" / "attempt_history.json"
    attempts = json.loads(attempt_path.read_text(encoding="utf-8"))["records"]
    target = np.array([record["target_displacement"] for record in attempts]) * 1000.0
    depth = np.array([record["subdivision_level"] for record in attempts])
    iterations = np.array([record["iterations"] for record in attempts])
    error = np.maximum(np.array([record["error"] for record in attempts]), 1.0e-12)

    fig = plt.figure(figsize=(7.25, 5.15))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 0.045, 1.0, 0.045],
        left=0.08,
        right=0.96,
        bottom=0.14,
        top=0.94,
        hspace=0.34,
        wspace=0.22,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 2])],
        ],
        dtype=object,
    )
    damage_cax = fig.add_subplot(grid[1, 1])
    error_cax = fig.add_subplot(grid[1, 3])
    spatial_panels = [
        (
            axes[0, 0],
            noh_tri,
            spatial["homogeneous_noh_damage"],
            f"Homogeneous, no H: u={float(spatial['homogeneous_noh_time'][0])*1000:.3f} micrometres",
            "a",
        ),
        (
            axes[0, 1],
            h_tri,
            spatial["homogeneous_h_damage"],
            f"Homogeneous, H: u={float(spatial['homogeneous_h_time'][0])*1000:.3f} micrometres",
            "b",
        ),
        (
            axes[1, 0],
            graph_tri,
            graph_damage,
            f"Boundary-property field, H: checkpoint u={graph_time:.6f} micrometres",
            "c",
        ),
    ]
    last_artist = None
    for ax, tri, damage, title, label in spatial_panels:
        last_artist = ax.tripcolor(tri, damage, shading="gouraud", cmap=DAMAGE_CMAP, vmin=0, vmax=1)
        add_precrack(ax)
        ax.set_xlim(18.0, 38.0)
        ax.set_ylim(17.0, 31.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (micrometres)")
        ax.set_ylabel("y (micrometres)")
        ax.set_title(title, fontsize=8.2)
        panel_label(ax, label)
    damage_cb = fig.colorbar(last_artist, cax=damage_cax)
    damage_cb.ax.set_title(r"$d$", fontsize=8.2, pad=3)
    damage_cb.ax.yaxis.set_ticks_position("left")
    damage_cb.ax.tick_params(labelsize=7.4, length=2.5, pad=2)

    ax = axes[1, 1]
    scatter = ax.scatter(
        target,
        depth,
        s=24 + 0.018 * iterations,
        c=np.log10(error),
        cmap="viridis",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.axvline(0.781142578125, color=RED, linestyle="--", linewidth=1.0, label="last converged state")
    ax.plot(
        target,
        depth,
        color="#9aa8ad",
        linewidth=0.9,
        marker="s",
        markersize=3.0,
        markerfacecolor="white",
        markeredgecolor="#7b898f",
        zorder=1,
    )
    ax.set_xlabel("rejected target displacement (micrometres)")
    ax.set_ylabel("subdivision level")
    ax.set_title("Rejected continuation attempts")
    ax.legend(fontsize=8.4, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, borderaxespad=0.0)
    clean_axes(ax)
    panel_label(ax, "d")
    cb = fig.colorbar(scatter, cax=error_cax)
    cb.set_label(r"$\log_{10}$ stagger error", fontsize=8.2)
    cb.ax.tick_params(labelsize=7.4)
    ax.text(
        0.98,
        0.05,
        "15 rejected attempts\n240 -> 960 iterations\ndepth 0 -> 10",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=INK,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
    )
    save_figure(fig, "figure_8_damage_continuation")


def main():
    plt.rcParams["axes.prop_cycle"] = (
        cycler(color=[BLUE, GOLD, TEAL, RED])
        + cycler(marker=["o", "s", "^", "D"])
        + cycler(linestyle=["-", "--", "-.", ":"])
        + cycler(markevery=[6, 6, 6, 6])
    )
    with np.load(SOURCE, allow_pickle=False) as spatial_file:
        spatial = {key: spatial_file[key] for key in spatial_file.files}
    figure_1_workflow()
    figure_2_ebsd_observation()
    figure_3_graph_fields_hydrogen(spatial)
    figure_4_pilot()
    figure_5_synthetic()
    figure_6_verification()
    figure_7_measured_factorial()
    figure_8_damage_continuation(spatial)
    print(FIG_DIR)


if __name__ == "__main__":
    main()
