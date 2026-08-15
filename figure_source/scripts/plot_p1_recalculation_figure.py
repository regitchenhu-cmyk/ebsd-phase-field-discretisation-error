from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures_v2"
DATA_DIR = ROOT / "source_data_v2"
STEM = FIGURE_DIR / "figure_10_p1_mesh_convergence"

resolution = np.array([2.0247066, 3.0370599, 4.0494133, 5.0617666])
mesh_labels = ["184×92", "276×138", "368×184", "460×230"]

reaction = {
    "Homogeneous, no H": np.array([16.366286, 15.976432, 15.755944, 15.535596]),
    "Homogeneous, 5 s H": np.array([16.298818, 15.897874, 15.670141, 15.444393]),
    "EBSD graph, no H": np.array([16.299160, 15.894989, 15.660028, 15.424492]),
    "EBSD graph, 5 s H": np.array([16.132085, 15.672571, 15.344990, 15.064457]),
}

crack_change = {
    "Homogeneous, no H": np.array([0.120983, 0.138053, 0.152657, 0.161269]),
    "Homogeneous, 5 s H": np.array([0.227901, 0.260138, 0.287471, 0.303060]),
    "EBSD graph, no H": np.array([0.252600, 0.300814, 0.349121, 0.391295]),
    "EBSD graph, 5 s H": np.array([0.681935, 0.901781, 1.265580, 1.405509]),
}

contrasts = {
    "H effect, graph off": np.array([-0.412238, -0.491709, -0.544580, -0.587060]),
    "H effect, graph on": np.array([-1.025054, -1.399298, -2.011730, -2.334180]),
    "Graph effect, H off": np.array([-0.410148, -0.509772, -0.608764, -0.715153]),
    "Graph effect, H on": np.array([-1.022977, -1.417195, -2.074967, -2.460023]),
}

gci_labels = [
    "Hom., no H",
    "Hom., 5 s H",
    "Graph, no H",
    "Graph, 5 s H",
    "H effect,\ngraph off",
    "H effect,\ngraph on",
    "Graph effect,\nH off",
    "Graph effect,\nH on",
]
observed_order = np.array([0.106, 0.109, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100])
fine_pair = np.array([1.418, 1.462, 1.527, 1.862, 7.236, 13.814, 14.876, 15.653])
gci = np.array([74.1, 74.2, 84.6, 103.2, 400.8, 765.2, 824.1, 867.1])

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.6,
        "axes.labelsize": 8.7,
        "axes.titlesize": 9.4,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

colors = ["#334E68", "#D18B28", "#4C8C74", "#B64B4B"]
markers = ["o", "s", "^", "D"]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.30))
fig.subplots_adjust(
    left=0.11,
    right=0.98,
    bottom=0.14,
    top=0.94,
    hspace=0.28,
    wspace=0.32,
)

ax = axes[0, 0]
for (label, values), color, marker in zip(reaction.items(), colors, markers):
    ax.plot(resolution, values, marker=marker, ms=4.2, lw=1.35, color=color, label=label)
ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
ax.set_ylabel(r"reaction at $u=4\times10^{-5}$ mm ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
ax.set_title("Reaction decreases monotonically without a plateau", loc="left", fontweight="bold")

ax = axes[0, 1]
for (label, values), color, marker in zip(contrasts.items(), colors, markers):
    ax.plot(resolution, np.abs(values), marker=marker, ms=4.2, lw=1.35, color=color, label=label)
ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
ax.set_ylabel("absolute same-mesh contrast (%)")
ax.set_title("Graph and hydrogen contrasts grow with refinement", loc="left", fontweight="bold")
effect_legend_handles, effect_legend_labels = ax.get_legend_handles_labels()

ax = axes[1, 0]
for (label, values), color, marker in zip(crack_change.items(), colors, markers):
    ax.plot(resolution, values, marker=marker, ms=4.2, lw=1.35, color=color, label=label)
ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
ax.set_ylabel("regularised crack-measure increase (%)")
ax.set_title("Diffuse damage is strongly mesh sensitive", loc="left", fontweight="bold")
ax.legend(fontsize=8.5, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0)

ax = axes[1, 1]
x = np.arange(len(gci))
ax.bar(x, gci, color=["#A8B6C3"] * 4 + ["#D9A15B"] * 4, width=0.72)
ax.set_yscale("log")
ax.axhline(2.0, color="#B64B4B", lw=1.0, ls="--")
ax.set_ylim(1, 1500)
ax.set_xticks(x, gci_labels, rotation=35, ha="right")
ax.set_ylabel("formal GCI warning metric (%)")
ax.set_title("Near-zero observed order invalidates asymptotic GCI", loc="left", fontweight="bold")
for xi, value, order in zip(x, gci, observed_order):
    ax.text(xi, value * 1.12, f"{value:.0f}\n$p$={order:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.text(7.45, 2.2, "2%", color="#B64B4B", ha="right", va="bottom")
ax.legend(
    effect_legend_handles,
    effect_legend_labels,
    fontsize=8.5,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    framealpha=1.0,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    borderaxespad=0.0,
)

for label, ax in zip(("a", "b", "c", "d"), axes.flat):
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(axis="y", color="#D9E0E6", lw=0.45, alpha=0.7)
    ax.set_axisbelow(True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(
    STEM.with_suffix(".tiff"),
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)
plt.close(fig)

with (DATA_DIR / "figure_10_p1_mesh_convergence.csv").open(
    "w", newline="", encoding="utf-8"
) as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "mesh",
            "ell_over_hK",
            *[f"reaction_{key}" for key in reaction],
            *[f"crack_change_pct_{key}" for key in crack_change],
            *[f"contrast_pct_{key}" for key in contrasts],
        ]
    )
    for index, mesh in enumerate(mesh_labels):
        writer.writerow(
            [
                mesh,
                resolution[index],
                *[values[index] for values in reaction.values()],
                *[values[index] for values in crack_change.values()],
                *[values[index] for values in contrasts.values()],
            ]
        )
