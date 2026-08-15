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
OUTPUT = ROOT / "M4_edge_crack_benchmark"
INPUTS = (
    REPO / "results" / "ijss_m4_edge_crack_benchmark_v1" / "benchmark_results.csv",
    REPO
    / "results"
    / "ijss_m4_edge_crack_benchmark_v1_fine736"
    / "benchmark_results.csv",
)


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric = {
        "nx",
        "ny",
        "h_diagonal_mm",
        "ell_over_hK",
        "residual_stiffness",
        "discrete_reaction_N_per_mm",
        "traction_reaction_N_per_mm",
        "reaction_crosscheck_relative_pct",
        "elastic_energy_N_mm",
        "fracture_energy_N_mm",
        "regularised_crack_length_mm",
        "maximum_damage",
        "zero_load_regularised_crack_length_mm",
        "zero_load_maximum_damage",
        "damage_increment_pct",
        "wall_time_s",
        "accepted_states",
        "initial_seed_crack_length_mm",
        "maximum_damage_kkt_relative",
        "maximum_energy_balance_relative",
    }
    for path in INPUTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row: dict[str, object] = dict(raw)
                for key in numeric:
                    value = raw.get(key, "")
                    if value not in {"", None}:
                        row[key] = float(value)
                rows.append(row)
    unique = {(str(row["mode"]), int(float(row["nx"]))): row for row in rows}
    return sorted(unique.values(), key=lambda row: (str(row["mode"]), float(row["nx"])))


def power_fit(rows: list[dict[str, object]]) -> dict[str, float | str]:
    h = np.asarray([float(row["h_diagonal_mm"]) for row in rows])
    reaction = np.asarray([float(row["discrete_reaction_N_per_mm"]) for row in rows])
    best: tuple[float, float, float, float] | None = None
    for order in np.linspace(0.01, 4.0, 3991):
        design = np.column_stack((np.ones_like(h), h**order))
        limit, coefficient = np.linalg.lstsq(design, reaction, rcond=None)[0]
        error = float(np.sum((reaction - design @ np.array((limit, coefficient))) ** 2))
        if best is None or error < best[0]:
            best = (error, order, float(limit), float(coefficient))
    assert best is not None
    error, order, limit, coefficient = best
    return {
        "formal_joint_order": order,
        "formal_limit": limit,
        "coefficient": coefficient,
        "residual_sum_squares": error,
        "status": "diagnostic only; asymptotic behavior not assumed",
    }


def mode_rows(rows: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row["mode"] == mode]
    return sorted(selected, key=lambda row: float(row["ell_over_hK"]))


def augment_reference(rows: list[dict[str, object]], mode: str) -> dict[str, float | str]:
    selected = mode_rows(rows, mode)
    reference = selected[-1]
    reference_reaction = float(reference["discrete_reaction_N_per_mm"])
    for row in selected:
        row["relative_to_736_reference_pct"] = 100.0 * abs(
            float(row["discrete_reaction_N_per_mm"]) - reference_reaction
        ) / abs(reference_reaction)
    previous = selected[-2]
    finest_pair_change = 100.0 * abs(
        float(previous["discrete_reaction_N_per_mm"]) - reference_reaction
    ) / abs(reference_reaction)
    return {
        "reference_mesh": f"{int(float(reference['nx']))}x{int(float(reference['ny']))}",
        "reference_reaction_N_per_mm": reference_reaction,
        "finest_pair_change_pct": finest_pair_change,
        **power_fit(selected),
    }


def main() -> None:
    rows = load_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_summary = augment_reference(rows, "frozen_edge_crack")
    coupled_summary = augment_reference(rows, "coupled_at2")

    frozen_by_nx = {
        int(float(row["nx"])): row for row in mode_rows(rows, "frozen_edge_crack")
    }
    coupled = mode_rows(rows, "coupled_at2")
    for row in coupled:
        frozen = frozen_by_nx[int(float(row["nx"]))]
        row["coupled_reaction_reduction_from_frozen_pct"] = 100.0 * (
            1.0
            - float(row["discrete_reaction_N_per_mm"])
            / float(frozen["discrete_reaction_N_per_mm"])
        )

    diagnostics = {
        "frozen_edge_crack": frozen_summary,
        "coupled_at2": coupled_summary,
        "maximum_reaction_extraction_mismatch_pct": max(
            float(row["reaction_crosscheck_relative_pct"]) for row in rows
        ),
        "maximum_coupled_reaction_reduction_from_frozen_pct": max(
            float(row["coupled_reaction_reduction_from_frozen_pct"]) for row in coupled
        ),
        "finest_coupled_damage_increment_pct": float(coupled[-1]["damage_increment_pct"]),
        "interpretation": (
            "The frozen and evolving sequences are compared against the same 736x368 "
            "same-model reference. Their formal power fits are diagnostics only."
        ),
    }
    (OUTPUT / "m4_edge_crack_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUTPUT / "m4_edge_crack_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

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
            "legend.fontsize": 8.3,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {"frozen_edge_crack": "#16324F", "coupled_at2": "#C44E3B"}
    labels = {
        "frozen_edge_crack": "frozen equilibrated crack",
        "coupled_at2": "evolving AT2 crack",
    }
    markers = {"frozen_edge_crack": "s", "coupled_at2": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)

    ax = axes[0, 0]
    for mode in ("frozen_edge_crack", "coupled_at2"):
        selected = mode_rows(rows, mode)
        x = np.asarray([float(row["ell_over_hK"]) for row in selected])
        y = np.asarray([float(row["discrete_reaction_N_per_mm"]) for row in selected])
        ax.plot(
            x,
            y,
            color=colors[mode],
            marker=markers[mode],
            mfc="white",
            lw=1.4,
            label=labels[mode],
        )
    ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
    ax.set_ylabel(r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    ax.set_title("Both edge-crack sequences retain mesh drift", loc="left", fontweight="bold")
    ax.legend(frameon=True, loc="best")

    ax = axes[0, 1]
    for mode in ("frozen_edge_crack", "coupled_at2"):
        selected = mode_rows(rows, mode)[:-1]
        x = np.asarray([float(row["ell_over_hK"]) for row in selected])
        y = np.asarray([float(row["relative_to_736_reference_pct"]) for row in selected])
        ax.plot(
            x,
            y,
            color=colors[mode],
            marker=markers[mode],
            mfc="white",
            lw=1.4,
            label=labels[mode],
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
    ax.set_ylabel("difference from 736x368 reference (%)")
    ax.set_title("Finest-level differences remain non-zero", loc="left", fontweight="bold")

    ax = axes[1, 0]
    x = np.asarray([float(row["ell_over_hK"]) for row in coupled])
    reduction = np.asarray(
        [float(row["coupled_reaction_reduction_from_frozen_pct"]) for row in coupled]
    )
    damage = np.asarray([float(row["damage_increment_pct"]) for row in coupled])
    ax.plot(x, reduction, color="#2C7FB8", marker="o", mfc="white", lw=1.4, label="reaction reduction")
    ax.plot(x, damage, color="#D98E04", marker="D", mfc="white", lw=1.3, ls="--", label="crack-measure increase")
    ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
    ax.set_ylabel("coupled evolution relative change (%)")
    ax.set_title("Damage evolution adds a smaller mesh-dependent shift", loc="left", fontweight="bold")
    ax.legend(frameon=True, loc="best")

    ax = axes[1, 1]
    for mode in ("frozen_edge_crack", "coupled_at2"):
        selected = mode_rows(rows, mode)
        x = np.asarray([float(row["ell_over_hK"]) for row in selected])
        y = np.asarray([float(row["reaction_crosscheck_relative_pct"]) for row in selected])
        ax.plot(
            x,
            y,
            color=colors[mode],
            marker=markers[mode],
            mfc="white",
            lw=1.4,
            label=labels[mode],
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"phase-field resolution $\ell/h_K$")
    ax.set_ylabel("traction/residual reaction mismatch (%)")
    ax.set_title("Reaction extraction is not the dominant error", loc="left", fontweight="bold")

    for label, ax in zip(("a", "b", "c", "d"), axes.flat):
        ax.text(
            0.018,
            0.982,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            fontsize=9.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8),
        )
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(color="#D9E0E6", lw=0.45, alpha=0.65)
        for spine in ax.spines.values():
            spine.set_visible(True)

    stem = OUTPUT / "figure_13_m4_edge_crack_benchmark"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    report = rf"""# M4 edge-crack benchmark result

- Runtime: DOLFINx 0.11.0, PETSc 3.25.4, four MPI ranks.
- Meshes: 184x92 through 736x368; the finest level resolves $\ell$ with approximately 8.1 element diagonals.
- Frozen-crack finest-pair change: {float(frozen_summary['finest_pair_change_pct']):.4f}%.
- Coupled-AT2 finest-pair change: {float(coupled_summary['finest_pair_change_pct']):.4f}%.
- Formal all-level orders: frozen {float(frozen_summary['formal_joint_order']):.3f}, coupled {float(coupled_summary['formal_joint_order']):.3f}; these remain non-asymptotic diagnostics, not GCI inputs.
- Maximum traction-versus-residual reaction mismatch: {float(diagnostics['maximum_reaction_extraction_mismatch_pct']):.4f}%.
- Finest coupled crack-measure increase: {float(diagnostics['finest_coupled_damage_increment_pct']):.4f}%.

The frozen edge-crack sequence retains substantial mesh drift, so crack representation and the tip field are primary contributors before graph projection is introduced. Allowing AT2 evolution adds a smaller mesh-dependent change. Reaction extraction is not the dominant source because the two independent reactions remain much closer to each other than either sequence is to the finest reference. The 736x368 result is a same-model numerical reference, not an exact sharp-crack solution; local refinement and field-representation comparisons remain required.
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
