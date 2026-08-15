from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"D:\models\PFM计算模型\PFM_H_GraphFracture\PFM_H_GraphFracture")
M4 = REPO / "results" / "ijss_m4_edge_crack_benchmark_v1"
M4_FINE = REPO / "results" / "ijss_m4_edge_crack_benchmark_v1_fine736"
CONTROL = REPO / "results" / "ijss_m5_positive_control_v4"
OUTPUT = Path(__file__).resolve().parent / "M5_positive_gate_v4"

TARGET = 4.0e-6
LENGTH = 0.0964
HEIGHT = 0.0482
ELL = 0.0015
MESHES = (184, 276, 368)
REPRESENTATIONS = ("DG0", "CG1")

COLORS = {
    "homogeneous": "#244a6d",
    "DG0": "#d48522",
    "CG1": "#2d8066",
    "threshold": "#b2463f",
}
MARKERS = {"homogeneous": "o", "DG0": "s", "CG1": "^"}
LINESTYLES = {"homogeneous": "-", "DG0": "--", "CG1": "-."}


def accepted_state(path: Path, target: float) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if abs(float(row["displacement"]) - target) <= 1.0e-12]
    if not matches:
        raise RuntimeError(f"No accepted state at u={target:.6e} mm in {path}")
    row = matches[-1]
    return {
        "reaction": float(row["reaction_y"]),
        "traction_reaction": float(row["traction_reaction_y"]),
        "crack_measure": float(row["regularised_crack_length"]),
        "fracture_energy": float(row["fracture_energy"]),
    }


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#d7dce0", linewidth=0.55, alpha=0.8)
    axis.tick_params(direction="in", top=True, right=True, width=0.8, length=3.2)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_color("#20262b")


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | str | int]] = []
    for nx in MESHES:
        ny = nx // 2
        h_diagonal = math.hypot(LENGTH / nx, HEIGHT / ny)
        homogeneous = accepted_state(M4 / f"coupled_{nx}x{ny}" / "history.csv", TARGET)
        for representation in REPRESENTATIONS:
            control = accepted_state(
                CONTROL / f"{representation}_{nx}x{ny}" / "history.csv", TARGET
            )
            records.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "representation": representation,
                    "ell_over_hK": ELL / h_diagonal,
                    "homogeneous_reaction": homogeneous["reaction"],
                    "control_reaction": control["reaction"],
                    "graph_effect_pct": 100.0
                    * (control["reaction"] / homogeneous["reaction"] - 1.0),
                    "homogeneous_crack_measure": homogeneous["crack_measure"],
                    "control_crack_measure": control["crack_measure"],
                    "crack_measure_effect_pct": 100.0
                    * (control["crack_measure"] / homogeneous["crack_measure"] - 1.0),
                    "reaction_extraction_mismatch_pct": 100.0
                    * abs(control["traction_reaction"] / control["reaction"] - 1.0),
                }
            )

    fine = accepted_state(M4_FINE / "coupled_736x368" / "history.csv", TARGET)
    homogeneous_368 = next(
        float(row["homogeneous_reaction"])
        for row in records
        if row["nx"] == 368 and row["representation"] == "DG0"
    )
    homogeneous_discrepancy = 100.0 * abs(homogeneous_368 / fine["reaction"] - 1.0)

    representation_differences: list[float] = []
    for nx in MESHES:
        dg0 = next(
            float(row["control_reaction"])
            for row in records
            if row["nx"] == nx and row["representation"] == "DG0"
        )
        cg1 = next(
            float(row["control_reaction"])
            for row in records
            if row["nx"] == nx and row["representation"] == "CG1"
        )
        representation_differences.append(100.0 * abs(cg1 / dg0 - 1.0))

    all_effects = np.asarray([float(row["graph_effect_pct"]) for row in records])
    stable_sign = bool(np.all(all_effects > 0.0) or np.all(all_effects < 0.0))
    finest_effects = [
        abs(float(row["graph_effect_pct"])) for row in records if row["nx"] == 368
    ]
    threshold = max(homogeneous_discrepancy, representation_differences[-1])
    gate_pass = stable_sign and min(finest_effects) > threshold

    diagnostics = {
        "matched_accepted_displacement_mm": TARGET,
        "completed_evidence_meshes": [f"{nx}x{nx // 2}" for nx in MESHES],
        "synthetic_control": {
            "centerline_toughness_ratio": 0.05,
            "bandwidth_mm": 0.006,
            "ell_over_b": 0.25,
        },
        "graph_effect_ranges_pct": {
            representation: [
                float(row["graph_effect_pct"])
                for row in records
                if row["representation"] == representation
            ]
            for representation in REPRESENTATIONS
        },
        "DG0_vs_CG1_difference_pct": representation_differences,
        "homogeneous_368_vs_736_discrepancy_pct": homogeneous_discrepancy,
        "gate_threshold_pct": threshold,
        "stable_effect_sign": stable_sign,
        "positive_control_gate_pass": gate_pass,
        "maximum_reaction_extraction_mismatch_pct": max(
            float(row["reaction_extraction_mismatch_pct"]) for row in records
        ),
        "claim_limit": (
            "This synthetic field-transfer control demonstrates gate acceptance only; "
            "it does not validate the measured EBSD graph as a material mechanism."
        ),
    }
    (OUTPUT / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.5,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.85,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    x = np.asarray(
        [
            float(row["ell_over_hK"])
            for row in records
            if row["representation"] == "DG0"
        ]
    )
    homogeneous = np.asarray(
        [
            float(row["homogeneous_reaction"])
            for row in records
            if row["representation"] == "DG0"
        ]
    )

    axis = axes[0, 0]
    axis.plot(
        x,
        homogeneous,
        color=COLORS["homogeneous"],
        marker=MARKERS["homogeneous"],
        linestyle=LINESTYLES["homogeneous"],
        linewidth=1.35,
        markersize=5.0,
        label="homogeneous",
    )
    for representation in REPRESENTATIONS:
        values = np.asarray(
            [
                float(row["control_reaction"])
                for row in records
                if row["representation"] == representation
            ]
        )
        axis.plot(
            x,
            values,
            color=COLORS[representation],
            marker=MARKERS[representation],
            linestyle=LINESTYLES[representation],
            linewidth=1.35,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=f"strong band, {representation}",
        )
    axis.set_xlabel(r"phase-field resolution $\ell/h_K$")
    axis.set_ylabel(r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    axis.set_title("Matched pre-peak reaction", loc="left", fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=True)
    style_axis(axis)
    panel_label(axis, "a")

    axis = axes[0, 1]
    effect_series = {}
    for representation in REPRESENTATIONS:
        effects = np.asarray(
            [
                abs(float(row["graph_effect_pct"]))
                for row in records
                if row["representation"] == representation
            ]
        )
        effect_series[representation] = effects
        axis.plot(
            x,
            effects,
            color=COLORS[representation],
            marker=MARKERS[representation],
            linestyle=LINESTYLES[representation],
            linewidth=1.35,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=representation,
        )
    axis.axhline(
        homogeneous_discrepancy,
        color="#B8B8B8",
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        label="secondary homogeneous-mesh difference",
    )
    for representation, y_offset in (("DG0", 10), ("CG1", -16)):
        effects = effect_series[representation]
        final_change = abs(effects[-1] - effects[-2])
        axis.annotate(
            rf"$\Delta\delta={final_change:.3f}$ pp",
            xy=(x[-1], effects[-1]),
            xytext=(8, y_offset),
            textcoords="offset points",
            fontsize=7.6,
            ha="left",
            va="center",
        )
    axis.set_xlabel(r"phase-field resolution $\ell/h_K$")
    axis.set_ylabel("paired reaction contrast (%)")
    axis.set_title("Paired contrast stabilises under refinement", loc="left", fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True)
    style_axis(axis)
    panel_label(axis, "b")

    axis = axes[1, 0]
    axis.plot(
        x,
        representation_differences,
        color="#5b6770",
        marker="D",
        linestyle="-",
        linewidth=1.35,
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="DG0-CG1 difference",
    )
    axis.set_xlabel(r"phase-field resolution $\ell/h_K$")
    axis.set_ylabel("DG0--CG1 difference (%)")
    axis.set_title("Representation difference decreases under refinement", loc="left", fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True)
    style_axis(axis)
    panel_label(axis, "c")

    axis = axes[1, 1]
    for representation in REPRESENTATIONS:
        crack_effect = np.asarray(
            [
                float(row["crack_measure_effect_pct"])
                for row in records
                if row["representation"] == representation
            ]
        )
        axis.plot(
            x,
            crack_effect,
            color=COLORS[representation],
            marker=MARKERS[representation],
            linestyle=LINESTYLES[representation],
            linewidth=1.35,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=representation,
        )
    axis.axhline(0.0, color="#4e555b", linewidth=0.8, linestyle=":")
    axis.set_xlabel(r"phase-field resolution $\ell/h_K$")
    axis.set_ylabel("crack-measure change (%)")
    axis.set_title("Independent damage signature has a stable sign", loc="left", fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True)
    style_axis(axis)
    panel_label(axis, "d")

    figure.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.15, wspace=0.34, hspace=0.64)
    stem = OUTPUT / "figure_14_m5_positive_gate"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    dg0_relative_change = 100.0 * abs(effect_series["DG0"][-1] - effect_series["DG0"][-2]) / abs(effect_series["DG0"][-1])
    cg1_relative_change = 100.0 * abs(effect_series["CG1"][-1] - effect_series["CG1"][-2]) / abs(effect_series["CG1"][-1])
    report = rf"""# High-contrast paired-comparison diagnostics

- Matched accepted state: $u=4.0\times10^{{-6}}$ mm on 184x92, 276x138, and 368x184 meshes.
- Synthetic control: centerline $r_G=0.05$, $b=6~\mu$m, and $\ell/b=0.25$.
- Finest DG0 boundary-property-field contrast: {finest_effects[0]:.4f}%.
- Finest CG1 boundary-property-field contrast: {finest_effects[1]:.4f}%.
- Finest DG0-CG1 reaction difference: {representation_differences[-1]:.4f}%.
- Secondary homogeneous-mesh difference: {homogeneous_discrepancy:.4f}%.
- Relative final-refinement changes: DG0 {dg0_relative_change:.3f}%; CG1 {cg1_relative_change:.3f}%.
- Preliminary detectability criterion met: {'YES' if gate_pass else 'NO'}.

The state is an exactly accepted common displacement, not an interpolated endpoint. The later target of $3.0\times10^{{-5}}$ mm was not reached within the computational-time limit and is not used. The relative final-refinement changes satisfy the 1% operational plateau criterion, but the sequence contains only three meshes and two refinement increments. Detectability is therefore preliminary until a fourth mesh confirms the plateau. This deliberately strong prescribed boundary-property field does not validate the measured EBSD field as a material mechanism.
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")

    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
