from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = Path(r"D:\models\PFM计算模型\PFM_H_GraphFracture\PFM_H_GraphFracture")
RESULT_ROOT = REPO / "results" / "ijss_m5_positive_control_v2"
BENCHMARK_ROOT = REPO / "results" / "ijss_m4_edge_crack_benchmark_v1"
BENCHMARK_FINE = REPO / "results" / "ijss_m4_edge_crack_benchmark_v1_fine736"
OUTPUT = ROOT / "M4_M5_positive_control"
TARGET = 3.0e-5
PARTIAL_TARGET = 2.0e-5
MESHES = (184, 276, 368)


def history_row(path: Path, target: float) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row
        for row in rows
        if math.isclose(float(row["displacement"]), target, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if not matches:
        raise RuntimeError(f"No accepted state at {target:.8g} in {path}")
    return matches[-1]


def homogeneous_reaction(nx: int, target: float) -> float:
    ny = nx // 2
    root = BENCHMARK_FINE if nx == 736 else BENCHMARK_ROOT
    row = history_row(root / f"coupled_{nx}x{ny}" / "history.csv", target)
    return float(row["reaction_y"])


def completed_control(representation: str, nx: int) -> dict[str, object]:
    path = RESULT_ROOT / f"{representation}_{nx}x{nx // 2}" / "positive_control_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_homogeneous = homogeneous_reaction(736, TARGET)
    records: list[dict[str, object]] = []
    for nx in MESHES:
        homogeneous = homogeneous_reaction(nx, TARGET)
        controls = {
            representation: completed_control(representation, nx)
            for representation in ("DG0", "CG1")
        }
        representation_difference = 100.0 * (
            float(controls["CG1"]["reaction_N_per_mm"])
            / float(controls["DG0"]["reaction_N_per_mm"])
            - 1.0
        )
        homogeneous_reference_difference = 100.0 * abs(
            homogeneous - reference_homogeneous
        ) / abs(reference_homogeneous)
        for representation, control in controls.items():
            control_reaction = float(control["reaction_N_per_mm"])
            graph_effect = 100.0 * (control_reaction / homogeneous - 1.0)
            records.append(
                {
                    "nx": nx,
                    "ny": nx // 2,
                    "representation": representation,
                    "ell_over_hK": float(control["ell_over_hK"]),
                    "b_over_hK": float(control["b_over_hK"]),
                    "homogeneous_reaction_N_per_mm": homogeneous,
                    "control_reaction_N_per_mm": control_reaction,
                    "graph_effect_pct": graph_effect,
                    "DG0_vs_CG1_difference_pct": representation_difference,
                    "homogeneous_difference_from_736_reference_pct": homogeneous_reference_difference,
                    "mean_Gc0_N_per_mm": float(control["mean_Gc0_N_per_mm"]),
                    "minimum_Gc0_N_per_mm": float(control["minimum_Gc0_N_per_mm"]),
                    "crack_measure_increase_pct": float(
                        control["crack_measure_increase_pct"]
                    ),
                    "reaction_crosscheck_relative_pct": float(
                        control["reaction_crosscheck_relative_pct"]
                    ),
                    "wall_time_s": float(control["wall_time_s"]),
                    "status": control["status"],
                }
            )

    partial_552: list[dict[str, object]] = []
    for representation in ("DG0", "CG1"):
        path = RESULT_ROOT / f"{representation}_552x276" / "history.csv"
        row = history_row(path, PARTIAL_TARGET)
        partial_552.append(
            {
                "representation": representation,
                "matched_displacement_mm": PARTIAL_TARGET,
                "reaction_N_per_mm": float(row["reaction_y"]),
                "status": "right-censored continuation; common accepted state only",
            }
        )

    finest_records = [record for record in records if record["nx"] == max(MESHES)]
    finest_graph_effects = [abs(float(record["graph_effect_pct"])) for record in finest_records]
    finest_representation_difference = abs(
        float(finest_records[0]["DG0_vs_CG1_difference_pct"])
    )
    finest_homogeneous_reference_difference = float(
        finest_records[0]["homogeneous_difference_from_736_reference_pct"]
    )
    stable_sign = all(float(record["graph_effect_pct"]) < 0.0 for record in records)
    gate_threshold = max(
        finest_representation_difference, finest_homogeneous_reference_difference
    )
    gate_pass = stable_sign and min(finest_graph_effects) > gate_threshold
    partial_difference = 100.0 * (
        float(partial_552[1]["reaction_N_per_mm"])
        / float(partial_552[0]["reaction_N_per_mm"])
        - 1.0
    )
    diagnostics = {
        "matched_endpoint_mm": TARGET,
        "completed_meshes": [f"{nx}x{nx // 2}" for nx in MESHES],
        "synthetic_control": "r_G=0.2, b=3 micrometres, ell/b=0.5",
        "finest_DG0_graph_effect_pct": float(
            next(
                record["graph_effect_pct"]
                for record in finest_records
                if record["representation"] == "DG0"
            )
        ),
        "finest_CG1_graph_effect_pct": float(
            next(
                record["graph_effect_pct"]
                for record in finest_records
                if record["representation"] == "CG1"
            )
        ),
        "finest_DG0_vs_CG1_difference_pct": finest_representation_difference,
        "homogeneous_368_vs_736_difference_pct": finest_homogeneous_reference_difference,
        "positive_control_gate_threshold_pct": gate_threshold,
        "positive_control_gate_pass": gate_pass,
        "right_censored_552_common_state_mm": PARTIAL_TARGET,
        "right_censored_552_CG1_vs_DG0_difference_pct": partial_difference,
        "claim_limit": (
            "The deliberately strong synthetic effect passes the conservative gate; "
            "this is not validation of a calibrated measured-graph mechanism."
            if gate_pass
            else "The deliberately strong synthetic effect remains below the conservative "
            "homogeneous-mesh discrepancy and is not accepted."
        ),
    }
    (OUTPUT / "m4_m5_positive_control_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUTPUT / "m4_m5_positive_control_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.3,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    colors = {"homogeneous": "#16324F", "DG0": "#2C7FB8", "CG1": "#C44E3B"}
    markers = {"homogeneous": "o", "DG0": "s", "CG1": "^"}
    x = np.asarray(
        [float(record["ell_over_hK"]) for record in records if record["representation"] == "DG0"]
    )
    homogeneous = np.asarray(
        [float(record["homogeneous_reaction_N_per_mm"]) for record in records if record["representation"] == "DG0"]
    )
    axes[0, 0].plot(x, homogeneous, color=colors["homogeneous"], marker=markers["homogeneous"], mfc="white", lw=1.4, label="homogeneous")
    for representation in ("DG0", "CG1"):
        selected = [record for record in records if record["representation"] == representation]
        axes[0, 0].plot(
            x,
            [float(record["control_reaction_N_per_mm"]) for record in selected],
            color=colors[representation],
            marker=markers[representation],
            mfc="white",
            lw=1.4,
            label=f"strong band, {representation}",
        )
    axes[0, 0].set_xlabel(r"phase-field resolution $\ell/h_K$")
    axes[0, 0].set_ylabel(r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    axes[0, 0].set_title("Strong control separates from homogeneous response", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=True, loc="best")

    ax = axes[0, 1]
    for representation in ("DG0", "CG1"):
        selected = [record for record in records if record["representation"] == representation]
        ax.plot(
            x,
            [abs(float(record["graph_effect_pct"])) for record in selected],
            color=colors[representation],
            marker=markers[representation],
            mfc="white",
            lw=1.4,
            label=f"graph effect, {representation}",
        )
    ax.plot(
        x,
        [abs(float(record["DG0_vs_CG1_difference_pct"])) for record in records if record["representation"] == "DG0"],
        color="#D98E04",
        marker="D",
        mfc="white",
        lw=1.25,
        ls="--",
        label="DG0-CG1 difference",
    )
    ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
    ax.set_ylabel("absolute reaction contrast (%)")
    ax.set_title("Positive effect exceeds representation ambiguity", loc="left", fontweight="bold")
    ax.legend(frameon=True, loc="best")

    ax = axes[1, 0]
    for representation in ("DG0", "CG1"):
        selected = [record for record in records if record["representation"] == representation]
        ax.plot(
            x,
            [float(record["mean_Gc0_N_per_mm"]) for record in selected],
            color=colors[representation],
            marker=markers[representation],
            mfc="white",
            lw=1.4,
            label=representation,
        )
    ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
    ax.set_ylabel(r"area-mean $G_{c0}$ ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    ax.set_title("DG0 and CG1 field integrals converge together", loc="left", fontweight="bold")
    ax.legend(frameon=True, loc="best")

    ax = axes[1, 1]
    for representation in ("DG0", "CG1"):
        selected = [record for record in records if record["representation"] == representation]
        ax.plot(
            x,
            [float(record["crack_measure_increase_pct"]) for record in selected],
            color=colors[representation],
            marker=markers[representation],
            mfc="white",
            lw=1.4,
            label=representation,
        )
    ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
    ax.set_ylabel("regularised crack-measure increase (%)")
    ax.set_title("Accepted control has a consistent damage signature", loc="left", fontweight="bold")
    ax.legend(frameon=True, loc="best")

    for label, ax in zip(("a", "b", "c", "d"), axes.flat):
        ax.text(0.018, 0.982, label, transform=ax.transAxes, va="top", fontweight="bold", fontsize=9.8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(color="#D9E0E6", lw=0.45, alpha=0.65)
        for spine in ax.spines.values():
            spine.set_visible(True)

    stem = OUTPUT / "figure_14_m4_m5_positive_control"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    report = rf"""# M4/M5 field-representation and positive-control result

- Completed matched endpoint: $u=3.0\times10^{{-5}}$ mm on 184x92, 276x138, and 368x184 meshes for both DG0 and CG1.
- Synthetic control: $r_G=0.2$, $b=3~\mu$m, and $\ell/b=0.5$.
- Finest DG0 graph effect: {diagnostics['finest_DG0_graph_effect_pct']:.4f}%.
- Finest CG1 graph effect: {diagnostics['finest_CG1_graph_effect_pct']:.4f}%.
- Finest DG0-CG1 reaction difference: {diagnostics['finest_DG0_vs_CG1_difference_pct']:.4f}%.
- Homogeneous 368x184-to-736x368 reference difference: {diagnostics['homogeneous_368_vs_736_difference_pct']:.4f}%.
- Positive-control gate: {'PASS' if gate_pass else 'FAIL'}.
- The 552x276 branches are right-censored; at their common accepted $u=2.0\times10^{{-5}}$ mm state, CG1-DG0 differs by {partial_difference:.4f}%.

{diagnostics['claim_limit']} The 552x276 branches remain right-censored.
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
