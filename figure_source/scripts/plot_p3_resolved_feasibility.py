"""Create the submission-grade P3 resolved-scale feasibility figure and report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RECONSTRUCTION = Path(r"E:\models\PFM_EBSD\IJSS_reconstruction")
RESULTS = Path(
    r"D:\models\PFM计算模型\PFM_H_GraphFracture"
    r"\PFM_H_GraphFracture\results"
)
FIGURES = RECONSTRUCTION / "figures_v2"
SOURCE_DATA = RECONSTRUCTION / "source_data_v2"

HOM_DISP = RESULTS / "ijss_recalc_resolved_ell0p5um_b1p5um_hom_u40_1096x548"
HOM_PATH = RESULTS / "ijss_recalc_resolved_ell0p5um_b1p5um_hom_path_1096x548"
GRAPH_PATH = RESULTS / "ijss_recalc_resolved_ell0p5um_b1p5um_graph_path_1096x548"


def read_history(directory: Path, case: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / "history.csv")
    numeric = [
        "step",
        "scheduled_step",
        "subdivision_level",
        "displacement",
        "load_increment",
        "reaction_y",
        "fracture_energy",
        "regularised_crack_length",
        "control_residual_relative",
        "mechanical_residual_relative",
        "energy_balance_relative",
        "path_increment",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["case"] = case
    frame["displacement_um"] = 1.0e3 * frame["displacement"]
    initial_crack = float(frame.iloc[0]["regularised_crack_length"])
    frame["relative_crack_change_percent"] = (
        100.0 * (frame["regularised_crack_length"] - initial_crack) / initial_crack
    )
    return frame


def checkpoint(directory: Path) -> dict:
    return json.loads((directory / "restart" / "checkpoint.json").read_text("utf-8"))


def method_status(directory: Path) -> dict:
    return json.loads((directory / "method_status.json").read_text("utf-8"))


def first_deep_subdivision(frame: pd.DataFrame) -> pd.Series:
    candidates = frame[
        (frame["control_phase"] == "displacement")
        & (frame["subdivision_level"] >= 4)
    ]
    return candidates.iloc[0]


def metric_row(
    label: str,
    frame: pd.DataFrame,
    status: str,
    generation: int,
    minimum_accepted_increment: float,
    increment_type: str,
) -> dict:
    terminal = frame.iloc[-1]
    peak = frame.loc[frame["reaction_y"].idxmax()]
    return {
        "case": label,
        "status": status,
        "checkpoint_generation": generation,
        "accepted_states": len(frame),
        "terminal_displacement_mm": terminal["displacement"],
        "terminal_reaction_N": terminal["reaction_y"],
        "terminal_fracture_energy_N_mm": terminal["fracture_energy"],
        "terminal_regularised_crack_length_mm": terminal[
            "regularised_crack_length"
        ],
        "terminal_relative_crack_change_percent": terminal[
            "relative_crack_change_percent"
        ],
        "maximum_computed_reaction_N": peak["reaction_y"],
        "displacement_at_maximum_computed_reaction_mm": peak["displacement"],
        "minimum_accepted_increment": minimum_accepted_increment,
        "increment_type": increment_type,
        "terminal_control_residual_relative": terminal.get(
            "control_residual_relative", np.nan
        ),
        "terminal_mechanical_residual_relative": terminal.get(
            "mechanical_residual_relative", np.nan
        ),
        "terminal_energy_balance_relative": terminal.get(
            "energy_balance_relative", np.nan
        ),
    }


def write_markdown_report(
    summary: pd.DataFrame,
    hom_disp: pd.DataFrame,
    hom_path: pd.DataFrame,
    graph_path: pd.DataFrame,
    hom_path_status: dict,
    hom_path_cp: dict,
    graph_cp: dict,
) -> None:
    hd = summary.loc[summary["case"] == "Homogeneous displacement"].iloc[0]
    hp = summary.loc[summary["case"] == "Homogeneous hybrid path"].iloc[0]
    gp = summary.loc[summary["case"] == "EBSD graph hybrid path"].iloc[0]
    hd_deep = first_deep_subdivision(hom_disp)
    gp_deep = first_deep_subdivision(graph_path)

    common_u = 1.5e-5
    hom_r = float(np.interp(common_u, hom_path["displacement"], hom_path["reaction_y"]))
    graph_r = float(
        np.interp(common_u, graph_path["displacement"], graph_path["reaction_y"])
    )
    graph_effect = 100.0 * (graph_r / hom_r - 1.0)

    failed_path_increments = [
        float(record["control_increment"])
        for record in hom_path_cp.get("attempt_history", [])
        if record.get("control_phase") == "fracture_energy"
        and record.get("control_increment") is not None
    ]
    minimum_failed_path_increment = min(failed_path_increments)
    graph_attempts = graph_cp.get("attempt_history", [])

    report = f"""# P3 resolved-scale feasibility result

## Decision

P3 resolves both regularisation scales on the full EBSD domain
(`nx=1096`, `ny=548`, 1,201,216 cells, `ell/h_K=4.020`,
`b/h_K=12.060`, and `ell/b=1/3`). It does **not**, however, provide a
completed homogeneous-versus-graph benchmark at the planned common
`u=4e-5 mm` state. The accepted calculations instead quantify a
solver/path feasibility boundary and must not be used for a mechanistic
grain-boundary comparison.

## Accepted high-resolution evidence

| Run | Last accepted displacement (mm) | Last reaction (N) | Crack-measure change (%) | Smallest accepted increment | Status |
|---|---:|---:|---:|---:|---|
| Homogeneous, displacement control | {hd.terminal_displacement_mm:.9e} | {hd.terminal_reaction_N:.6f} | {hd.terminal_relative_crack_change_percent:.6f} | {hd.minimum_accepted_increment:.9e} mm | partial, deliberately stopped |
| Homogeneous, hybrid fracture-energy path | {hp.terminal_displacement_mm:.9e} | {hp.terminal_reaction_N:.6f} | {hp.terminal_relative_crack_change_percent:.6f} | {hp.minimum_accepted_increment:.9e} N mm | {hom_path_status["status"]} |
| EBSD graph, hybrid path setup | {gp.terminal_displacement_mm:.9e} | {gp.terminal_reaction_N:.6f} | {gp.terminal_relative_crack_change_percent:.6f} | {gp.minimum_accepted_increment:.9e} mm | partial, deliberately stopped |

The homogeneous displacement calculation first required subdivision level
4 at `u={hd_deep.displacement:.9e} mm`. Fracture-energy path control extended
the accepted branch to `u={hp.terminal_displacement_mm:.9e} mm`, with a
relative control residual of {hp.terminal_control_residual_relative:.3e}
and a mechanical residual of {hp.terminal_mechanical_residual_relative:.3e}.
The next path target failed even at
`dE={minimum_failed_path_increment:.9e} N mm`, which is the configured
level-10 minimum.

The EBSD graph calculation entered deep displacement subdivision at
`u={gp_deep.displacement:.9e} mm`, approximately
{gp_deep.displacement / hd_deep.displacement:.3f} of the homogeneous onset
displacement. At the last common accepted state `u=1.5e-5 mm`, its reaction
differs from the homogeneous path by {graph_effect:+.4f}%. This common-state
difference is small, whereas the subsequent continuation cost is very
different. Because the graph branch never reached the common path-switch
state, that contrast cannot yet be attributed uniquely to grain-boundary
fracture physics.

## Numerical interpretation

1. The resolved mesh passes the geometric quality gates, so P3 is not a
   repeat of the under-resolved `b=ell` problem identified in P1.
2. Both controls encounter sharply curved branches. Displacement control
   needs increments down to `{hd.minimum_accepted_increment:.3e} mm`;
   fracture-energy control accepts `{hp.minimum_accepted_increment:.3e} N mm`
   but rejects the next target at the same scale.
3. The graph field shifts the severe subdivision region to much lower load,
   but the unmatched terminal states prevent a quantitative graph effect.
4. The computed maximum reaction
   ({hp.maximum_computed_reaction_N:.6f} N) is only the maximum on the
   accepted branch, not a certified global peak or post-peak result.
5. A complete P3 benchmark requires a more robust monolithic or arc-length
   continuation formulation, plus a matched endpoint criterion. Relaxing
   residual tolerances would not be an acceptable substitute.

## Manuscript use

Use Figure 12 and this result as a numerical-verification/limitations result,
preferably in the Supplementary Material. Retain P1 and P2 as screening
evidence, state that the fully resolved full-domain benchmark exposed a
continuation limitation, and avoid claims about resolved grain-boundary
crack-path selection or peak-load reduction.

## Audit

- Homogeneous hybrid checkpoint generation: {int(hom_path_cp["generation"])}
- Graph hybrid checkpoint generation: {int(graph_cp["generation"])}
- Homogeneous path accepted states: {len(hom_path)}
- Graph accepted states: {len(graph_path)}
- Graph unaccepted nonlinear attempts: {len(graph_attempts)}
- Figure source data: `source_data_v2/figure_12_p3_path_histories.csv`
- Adaptivity source data: `source_data_v2/figure_12_p3_adaptivity.csv`
- Summary source data: `source_data_v2/figure_12_p3_summary.csv`
"""
    (RECONSTRUCTION / "IJSS_P3_resolved_feasibility_results.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)

    hom_disp = read_history(HOM_DISP, "Homogeneous displacement")
    hom_path = read_history(HOM_PATH, "Homogeneous hybrid path")
    graph_path = read_history(GRAPH_PATH, "EBSD graph hybrid path")

    hom_disp_cp = checkpoint(HOM_DISP)
    hom_path_cp = checkpoint(HOM_PATH)
    graph_cp = checkpoint(GRAPH_PATH)
    hom_path_status = method_status(HOM_PATH)

    path_source = pd.concat(
        [
            hom_path[
                [
                    "case",
                    "step",
                    "control_phase",
                    "subdivision_level",
                    "displacement",
                    "displacement_um",
                    "reaction_y",
                    "fracture_energy",
                    "regularised_crack_length",
                    "relative_crack_change_percent",
                    "path_increment",
                    "control_residual_relative",
                    "mechanical_residual_relative",
                    "energy_balance_relative",
                ]
            ],
            graph_path[
                [
                    "case",
                    "step",
                    "control_phase",
                    "subdivision_level",
                    "displacement",
                    "displacement_um",
                    "reaction_y",
                    "fracture_energy",
                    "regularised_crack_length",
                    "relative_crack_change_percent",
                    "path_increment",
                    "control_residual_relative",
                    "mechanical_residual_relative",
                    "energy_balance_relative",
                ]
            ],
        ],
        ignore_index=True,
    )
    path_source.to_csv(
        SOURCE_DATA / "figure_12_p3_path_histories.csv", index=False
    )

    adaptivity = pd.concat(
        [
            hom_disp.loc[
                (hom_disp["control_phase"] == "displacement")
                & (hom_disp["load_increment"] > 0),
                [
                    "case",
                    "step",
                    "subdivision_level",
                    "displacement",
                    "displacement_um",
                    "load_increment",
                    "reaction_y",
                ],
            ],
            graph_path.loc[
                (graph_path["control_phase"] == "displacement")
                & (graph_path["load_increment"] > 0),
                [
                    "case",
                    "step",
                    "subdivision_level",
                    "displacement",
                    "displacement_um",
                    "load_increment",
                    "reaction_y",
                ],
            ],
        ],
        ignore_index=True,
    )
    adaptivity.to_csv(SOURCE_DATA / "figure_12_p3_adaptivity.csv", index=False)

    hom_disp_min = float(
        hom_disp.loc[hom_disp["load_increment"] > 0, "load_increment"].min()
    )
    hom_path_min = float(
        hom_path.loc[
            (hom_path["control_phase"] == "fracture_energy")
            & (hom_path["path_increment"] > 0),
            "path_increment",
        ].min()
    )
    graph_min = float(
        graph_path.loc[graph_path["load_increment"] > 0, "load_increment"].min()
    )
    summary = pd.DataFrame(
        [
            metric_row(
                "Homogeneous displacement",
                hom_disp,
                hom_disp_cp["status"],
                int(hom_disp_cp["generation"]),
                hom_disp_min,
                "displacement_mm",
            ),
            metric_row(
                "Homogeneous hybrid path",
                hom_path,
                hom_path_status["status"],
                int(hom_path_cp["generation"]),
                hom_path_min,
                "fracture_energy_N_mm",
            ),
            metric_row(
                "EBSD graph hybrid path",
                graph_path,
                graph_cp["status"],
                int(graph_cp["generation"]),
                graph_min,
                "displacement_mm",
            ),
        ]
    )
    summary.to_csv(SOURCE_DATA / "figure_12_p3_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.facecolor": "white",
        }
    )

    navy = "#123B5D"
    vermillion = "#C4472D"
    charcoal = "#363B40"

    figure = plt.figure(figsize=(7.2, 6.40))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.12, 1.0), height_ratios=(1.0, 0.94),
        hspace=0.42, wspace=0.31
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])

    ax_a.plot(
        hom_path["displacement_um"],
        hom_path["reaction_y"],
        color=navy,
        lw=1.8,
        marker="o",
        ms=3.1,
        markevery=max(1, len(hom_path) // 9),
        label="Homogeneous: hybrid path",
        zorder=3,
    )
    ax_a.plot(
        graph_path["displacement_um"],
        graph_path["reaction_y"],
        color=vermillion,
        lw=1.8,
        marker="s",
        ms=2.8,
        markevery=max(1, len(graph_path) // 7),
        label="EBSD graph: hybrid setup",
        zorder=3,
    )
    hom_disp_last = hom_disp.iloc[-1]
    ax_a.scatter(
        [hom_disp_last["displacement_um"]],
        [hom_disp_last["reaction_y"]],
        marker="x",
        s=48,
        linewidths=1.5,
        color=charcoal,
        label="Displacement-control limit",
        zorder=5,
    )
    ax_a.axvline(0.028, color=charcoal, lw=0.85, ls=(0, (4, 3)), alpha=0.8)
    ax_a.text(
        0.0283,
        1.25,
        "path switch",
        rotation=90,
        va="bottom",
        ha="left",
        color=charcoal,
        fontsize=7.6,
    )
    ax_a.annotate(
        "hybrid-path limit",
        xy=(hom_path.iloc[-1]["displacement_um"], hom_path.iloc[-1]["reaction_y"]),
        xytext=(0.0308, 13.15),
        arrowprops=dict(arrowstyle="-", color=navy, lw=0.8),
        color=navy,
        fontsize=7.6,
    )
    ax_a.annotate(
        "graph deep subdivision",
        xy=(
            graph_path.iloc[-1]["displacement_um"],
            graph_path.iloc[-1]["reaction_y"],
        ),
        xytext=(0.0182, 5.1),
        arrowprops=dict(arrowstyle="-", color=vermillion, lw=0.8),
        color=vermillion,
        fontsize=7.6,
    )
    ax_a.set_xlabel(r"Top-edge displacement, $u$ ($\mu$m)")
    ax_a.set_ylabel(r"Reaction force, $R_y$ (N)")
    ax_a.set_xlim(0.0, 0.038)
    ax_a.set_ylim(bottom=0.0)
    ax_a.grid(True, color="#D7DADD", lw=0.55, alpha=0.7)
    ax_a.legend(fontsize=8.7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, borderaxespad=0.0)

    ax_b.plot(
        hom_path["displacement_um"],
        hom_path["relative_crack_change_percent"],
        color=navy,
        lw=1.7,
        marker="o",
        ms=2.8,
        label="Homogeneous",
    )
    ax_b.plot(
        graph_path["displacement_um"],
        graph_path["relative_crack_change_percent"],
        color=vermillion,
        lw=1.7,
        marker="s",
        ms=2.6,
        label="EBSD graph",
    )
    ax_b.axvline(0.028, color=charcoal, lw=0.85, ls=(0, (4, 3)))
    ax_b.set_xlabel(r"Top-edge displacement, $u$ ($\mu$m)")
    ax_b.set_ylabel(r"Relative crack-measure change (\%)")
    ax_b.set_xlim(0.0, 0.038)
    ax_b.set_ylim(bottom=0.0)
    ax_b.grid(True, color="#D7DADD", lw=0.55, alpha=0.7)
    ax_b.legend(fontsize=8.7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, borderaxespad=0.0)

    for case, color, marker, label in [
        ("Homogeneous displacement", navy, "o", "Homogeneous"),
        ("EBSD graph hybrid path", vermillion, "s", "EBSD graph"),
    ]:
        subset = adaptivity[adaptivity["case"] == case]
        ax_c.semilogy(
            subset["displacement_um"],
            subset["load_increment"],
            color=color,
            lw=1.15,
            marker=marker,
            ms=3.0,
            label=label,
        )
    ax_c.axhline(
        7.8125e-9,
        color=charcoal,
        lw=0.8,
        ls=(0, (3, 2)),
    )
    ax_c.set_xlabel(r"Accepted displacement, $u$ ($\mu$m)")
    ax_c.set_ylabel(r"Accepted increment, $\Delta u$ (mm)")
    ax_c.set_xlim(0.007, 0.031)
    ax_c.set_ylim(4.0e-9, 3.5e-6)
    ax_c.grid(True, which="both", color="#D7DADD", lw=0.55, alpha=0.7)
    ax_c.legend(fontsize=8.7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, borderaxespad=0.0)

    for axis, label in zip((ax_a, ax_b, ax_c), ("a", "b", "c")):
        axis.text(
            -0.085,
            1.045,
            f"({label})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )

    base = FIGURES / "figure_12_p3_resolved_feasibility"
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    write_markdown_report(
        summary,
        hom_disp,
        hom_path,
        graph_path,
        hom_path_status,
        hom_path_cp,
        graph_cp,
    )

    print(summary.to_string(index=False))
    print(f"Wrote {base}.[pdf|svg|png|tiff]")


if __name__ == "__main__":
    main()
