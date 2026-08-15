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
OUT = ROOT / "M4_reaction_extraction_audit"
TARGET_DISPLACEMENT = 4.0e-5
LENGTH = 0.09639
HEIGHT = 0.048195
ELL = 0.0015

CASES = (
    (184, 92, "ijss_recalc_s1GPa_hom_noH_184x92"),
    (276, 138, "ijss_recalc_s1GPa_hom_noH_u70_276x138"),
    (368, 184, "ijss_recalc_s1GPa_hom_noH_u40_368x184"),
    (460, 230, "ijss_recalc_s1GPa_hom_noH_u40_460x230"),
)


def load_target_row(directory: str) -> dict[str, str]:
    path = REPO / "results" / directory / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row
        for row in rows
        if math.isclose(
            float(row["displacement"]),
            TARGET_DISPLACEMENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    if not matches:
        raise RuntimeError(f"No accepted state at u={TARGET_DISPLACEMENT:.8g} in {path}")
    return matches[-1]


def joint_power_fit(h: np.ndarray, values: np.ndarray) -> dict[str, float]:
    best: tuple[float, float, float, float] | None = None
    for order in np.linspace(0.01, 4.0, 3991):
        design = np.column_stack((np.ones_like(h), h**order))
        limit, coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = float(np.sum((values - design @ np.array([limit, coefficient])) ** 2))
        if best is None or residual < best[0]:
            best = (residual, order, float(limit), float(coefficient))
    assert best is not None
    residual, order, limit, coefficient = best
    return {
        "residual_sum_squares": residual,
        "observed_order": order,
        "formal_limit": limit,
        "coefficient": coefficient,
        "interpretation": "non-asymptotic diagnostic; not a certified Richardson extrapolation",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []
    for nx, ny, directory in CASES:
        row = load_target_row(directory)
        discrete = float(row["reaction_y"])
        traction = float(row["traction_reaction_y"])
        h_diagonal = math.hypot(LENGTH / nx, HEIGHT / ny)
        records.append(
            {
                "nx": nx,
                "ny": ny,
                "result_directory": directory,
                "h_diagonal_mm": h_diagonal,
                "h_over_ell": h_diagonal / ELL,
                "displacement_mm": float(row["displacement"]),
                "discrete_reaction_N_per_mm": discrete,
                "traction_reaction_N_per_mm": traction,
                "traction_minus_discrete_N_per_mm": traction - discrete,
                "reaction_crosscheck_relative_pct": 100.0
                * abs(traction - discrete)
                / max(abs(discrete), 1.0e-30),
                "maximum_damage": float(row.get("maximum_damage", "nan")),
                "regularised_crack_length_mm": float(
                    row.get("regularised_crack_length", "nan")
                ),
                "damage_kkt_relative": float(row.get("damage_kkt_relative", "nan")),
            }
        )

    h = np.asarray([float(record["h_diagonal_mm"]) for record in records])
    discrete = np.asarray(
        [float(record["discrete_reaction_N_per_mm"]) for record in records]
    )
    traction = np.asarray(
        [float(record["traction_reaction_N_per_mm"]) for record in records]
    )
    diagnostics = {
        "target_displacement_mm": TARGET_DISPLACEMENT,
        "discrete_reaction_four_level_fit": joint_power_fit(h, discrete),
        "traction_reaction_four_level_fit": joint_power_fit(h, traction),
        "maximum_reaction_crosscheck_relative_pct": max(
            float(record["reaction_crosscheck_relative_pct"]) for record in records
        ),
        "interpretation_gate": (
            "Reaction extraction is exonerated only if the discrete/traction mismatch is "
            "small relative to the mesh trend on every level. This audit does not replace "
            "the required over-resolved cracked reference solution."
        ),
    }

    with (OUT / "reaction_extraction_crosscheck.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "reaction_extraction_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    order = np.argsort(h / ELL)
    x = h[order] / ELL
    axes[0].plot(
        x,
        discrete[order],
        color="#16324F",
        marker="o",
        mfc="white",
        lw=1.4,
        label="constrained-DOF residual",
    )
    axes[0].plot(
        x,
        traction[order],
        color="#C44E3B",
        marker="^",
        mfc="white",
        lw=1.25,
        ls="--",
        label="top-boundary traction",
    )
    axes[0].set_xlabel(r"element diagonal $h_K/\ell$")
    axes[0].set_ylabel(r"reaction ($\mathrm{N}\!\cdot\!\mathrm{mm}^{-1}$)")
    axes[0].set_title("Archived homogeneous edge-crack sequence", loc="left", fontweight="bold")
    axes[0].legend(frameon=True, loc="best")

    mismatch = np.asarray(
        [float(record["reaction_crosscheck_relative_pct"]) for record in records]
    )[order]
    axes[1].plot(
        x,
        mismatch,
        color="#2C7FB8",
        marker="s",
        mfc="white",
        lw=1.4,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"element diagonal $h_K/\ell$")
    axes[1].set_ylabel("reaction cross-check mismatch (%)")
    axes[1].set_title("Extraction mismatch versus mesh trend", loc="left", fontweight="bold")
    for label, axis in zip(("a", "b"), axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(color="#D9E0E6", lw=0.45, alpha=0.65)
        for spine in axis.spines.values():
            spine.set_visible(True)
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        fig.savefig(OUT / f"reaction_extraction_crosscheck.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    summary = [
        "# M4 archived reaction-extraction audit",
        "",
        f"- Common accepted displacement: `{TARGET_DISPLACEMENT:.8g} mm`.",
        f"- Maximum discrete-versus-traction mismatch: `{diagnostics['maximum_reaction_crosscheck_relative_pct']:.6g}%`.",
        f"- Four-level discrete-reaction fit: `p={diagnostics['discrete_reaction_four_level_fit']['observed_order']:.3f}` (diagnostic only).",
        f"- Four-level traction-reaction fit: `p={diagnostics['traction_reaction_four_level_fit']['observed_order']:.3f}` (diagnostic only).",
        "- This comparison can exclude reaction extraction as the dominant source only when its mismatch is much smaller than the mesh trend.",
        "- It does not complete M4: an over-resolved cracked reference and representation/local-refinement comparisons are still required.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(summary), encoding="utf-8")
    print(OUT)
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
