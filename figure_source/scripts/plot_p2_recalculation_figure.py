from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures_v2"
DATA_DIR = ROOT / "source_data_v2"
STEM = FIGURE_DIR / "figure_11_p2_b_confidence_matrix"

b_um = np.array([0.405, 0.8, 1.5])
confidence = np.array([0.0, 0.5, 1.0])
reaction = np.array(
    [
        [15.629304315, 15.611306563, 15.717235414],
        [15.532225005, 15.541216190, 15.697885627],
        [15.483290799, 15.501250496, 15.660027716],
    ]
)
crack_change = np.array(
    [
        [0.358416404, 0.390246660, 0.215598294],
        [0.585600573, 0.566805763, 0.263269359],
        [0.717505313, 0.694074929, 0.349121251],
    ]
)
b_effect = np.array([0.934229, 0.704977, 0.363981])
confidence_span = np.array([0.673966, 1.055305, 1.128586])

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
        "legend.fontsize": 7.0,
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

navy = "#314E68"
teal = "#4D927E"
amber = "#D28A25"
red = "#B84A45"
grey = "#778492"

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
fig.subplots_adjust(
    left=0.10,
    right=0.95,
    bottom=0.14,
    top=0.93,
    hspace=0.34,
    wspace=0.36,
)


def heatmap(ax, values, title, cmap, fmt, cbar_label):
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(3), ["0", "0.5", "1"])
    ax.set_yticks(np.arange(3), ["0.405", "0.800", "1.500"])
    ax.set_xlabel("confidence floor")
    ax.set_ylabel(r"graph width $b$ ($\mu$m)")
    ax.set_title(title, loc="left", fontweight="bold")
    for row in range(3):
        for column in range(3):
            red_channel, green_channel, blue_channel, _ = image.cmap(
                image.norm(values[row, column])
            )
            luminance = (
                0.2126 * red_channel + 0.7152 * green_channel + 0.0722 * blue_channel
            )
            color = "white" if luminance < 0.52 else "#17324D"
            ax.text(
                column,
                row,
                format(values[row, column], fmt),
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label(cbar_label)


heatmap(
    axes[0, 0],
    reaction,
    "Reaction varies weakly across graph constructions",
    "Blues_r",
    ".3f",
    r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)",
)
heatmap(
    axes[0, 1],
    crack_change,
    "Diffuse damage grows with graph-field coverage",
    "YlOrRd",
    ".3f",
    "crack-measure increase (%)",
)

ax = axes[1, 0]
for index, (label, color, marker) in enumerate(
    zip(["confidence 0", "confidence 0.5", "confidence 1"], [navy, amber, teal], ["o", "s", "^"])
):
    ax.plot(b_um, reaction[:, index], marker=marker, ms=4.5, lw=1.4, color=color, label=label)
ax.set_xlabel(r"graph width $b$ ($\mu$m)")
ax.set_ylabel(r"reaction at $u=4\times10^{-5}$ mm ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
ax.set_title("Width effect depends on retained-chain set", loc="left", fontweight="bold")
ax.legend(fontsize=8.4, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), borderaxespad=0.0)

ax = axes[1, 1]
labels = [
    r"$b$ effect, conf. 0",
    r"$b$ effect, conf. 0.5",
    r"$b$ effect, conf. 1",
    r"conf. span, $b=0.405$",
    r"conf. span, $b=0.8$",
    r"conf. span, $b=1.5$",
    "full matrix span",
    "minimum formal P1 metric",
]
values = np.concatenate([b_effect, confidence_span, [1.488459, 74.1]])
colors = [teal] * 3 + [amber] * 3 + [navy, red]
y = np.arange(len(values))
ax.barh(y, values, color=colors, height=0.62)
ax.set_xscale("log")
ax.set_xlim(0.25, 120)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("reaction-change magnitude (%)")
ax.set_title("Construction changes versus a non-asymptotic warning metric", loc="left", fontweight="bold")
for yi, value in zip(y, values):
    ax.text(value * 1.08, yi, f"{value:.2f}", va="center")

for label, ax in zip(("a", "b", "c", "d"), axes.flat):
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)
    ax.tick_params(direction="in", top=True, right=True)
    if ax not in axes[0, :]:
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

with (DATA_DIR / "figure_11_p2_b_confidence_matrix.csv").open(
    "w", newline="", encoding="utf-8"
) as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "b_um",
            "confidence_floor",
            "reaction_N_per_mm",
            "crack_measure_change_pct",
        ]
    )
    for row, b_value in enumerate(b_um):
        for column, confidence_value in enumerate(confidence):
            writer.writerow(
                [
                    b_value,
                    confidence_value,
                    reaction[row, column],
                    crack_change[row, column],
                ]
            )
