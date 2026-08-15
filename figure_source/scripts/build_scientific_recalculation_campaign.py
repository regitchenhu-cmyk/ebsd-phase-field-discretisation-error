from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = Path(r"D:\models\PFM计算模型\PFM_H_GraphFracture\PFM_H_GraphFracture")
CAMPAIGN_CSV = HERE / "IJSS_scientific_recalculation_matrix.csv"
PROTOCOL = HERE / "IJSS_scientific_recalculation_protocol.md"
STATUS = HERE / "IJSS_scientific_recalculation_status.md"
SUMMARY_CSV = HERE / "IJSS_scientific_recalculation_results.csv"
CONVERGENCE_JSON = HERE / "IJSS_scientific_recalculation_convergence.json"

LENGTH = 0.09639
HEIGHT = 0.048195
YOUNG = 210_000.0
POISSON = 0.30
TARGET_STRENGTH = 1_000.0
REFERENCE_ELL = 0.0015
REFERENCE_B = 0.0015
P1_H_CHARGE = 5.0
P1_COMPARISON_DISPLACEMENT = 0.00004


def gc_for_plane_strain_at2_strength(strength: float, ell: float) -> float:
    prefactor = 3.0 * math.sqrt(3.0) / 16.0
    plane_strain_modulus = YOUNG / (1.0 - POISSON**2)
    return (strength / prefactor) ** 2 * ell / plane_strain_modulus


REFERENCE_GC = gc_for_plane_strain_at2_strength(TARGET_STRENGTH, REFERENCE_ELL)


@dataclass(frozen=True)
class Case:
    name: str
    stage: str
    priority: str
    purpose: str
    nx: int
    ny: int
    ell: float = REFERENCE_ELL
    gc: float = REFERENCE_GC
    influence_radius: float = REFERENCE_B
    confidence_floor: float = 1.0
    graph_enabled: bool = True
    hydrogen_enabled: bool = False
    charging_time: float = 10.0
    maximum_displacement: float = 0.00012
    steps: int = 60
    path_control: bool = False

    @property
    def h_diagonal(self) -> float:
        return math.hypot(LENGTH / self.nx, HEIGHT / self.ny)

    @property
    def ell_over_h(self) -> float:
        return self.ell / self.h_diagonal

    @property
    def b_over_h(self) -> float:
        return self.influence_radius / self.h_diagonal

    @property
    def ell_over_b(self) -> float:
        return self.ell / self.influence_radius


def mesh_cases() -> list[Case]:
    cases: list[Case] = []
    meshes = [(184, 92), (276, 138), (368, 184), (460, 230)]
    factors = [
        ("hom_noH", False, False, 10.0),
        ("hom_H5s", False, True, P1_H_CHARGE),
        ("graph_noH", True, False, 10.0),
        ("graph_H5s", True, True, P1_H_CHARGE),
    ]
    for nx, ny in meshes:
        if nx < 368:
            window_tag = "u70"
            maximum_displacement = 0.00007
            steps = 35
        else:
            window_tag = "u40"
            maximum_displacement = P1_COMPARISON_DISPLACEMENT
            steps = 20
        for label, graph, hydrogen, charging_time in factors:
            cases.append(
                Case(
                    name=f"ijss_recalc_s1GPa_{label}_{window_tag}_{nx}x{ny}",
                    stage="B_contrast_convergence",
                    priority="P1",
                    purpose=(
                        "stable-window same-load-node contrast convergence at "
                        "u=4e-5 mm; 5 s hydrogen cases are transient lattice-only precharge"
                    ),
                    nx=nx,
                    ny=ny,
                    graph_enabled=graph,
                    hydrogen_enabled=hydrogen,
                    charging_time=charging_time,
                    maximum_displacement=maximum_displacement,
                    steps=steps,
                )
            )
    return cases


def campaign_cases() -> list[Case]:
    cases = [
        Case(
            name="ijss_recalc_s1GPa_graph_noH_path_184x92",
            stage="A_path_pilot",
            priority="P0",
            purpose="fracture-energy path-control pilot; hydrogen disabled by verified solver contract",
            nx=184,
            ny=92,
            graph_enabled=True,
            hydrogen_enabled=False,
            path_control=True,
        )
    ]
    for time in (5.0, 10.0, 20.0):
        cases.append(
            Case(
                name=f"ijss_recalc_s1GPa_graph_H{int(time)}s_field_184x92",
                stage="A_transport_screen",
                priority="P0",
                purpose="short transient lattice-precharge screen; no trapping claim because Nt=0",
                nx=184,
                ny=92,
                graph_enabled=True,
                hydrogen_enabled=True,
                charging_time=time,
                maximum_displacement=0.00004,
                steps=20,
            )
        )
    cases.extend(mesh_cases())

    for radius_um, radius in ((0.405, 0.000405), (0.8, 0.0008), (1.5, 0.0015)):
        for confidence in (0.0, 0.5, 1.0):
            confidence_tag = str(confidence).replace(".", "p")
            radius_tag = str(radius_um).replace(".", "p")
            cases.append(
                Case(
                    name=(
                        f"ijss_recalc_s1GPa_b{radius_tag}um_"
                        f"conf{confidence_tag}_u40_368x184"
                    ),
                    stage="C_mapping_mechanics_matrix",
                    priority="P2",
                    purpose=(
                        "mechanics-level b-confidence sensitivity; interpretation remains "
                        "resolution-limited when ell/b is not small"
                    ),
                    nx=368,
                    ny=184,
                    influence_radius=radius,
                    confidence_floor=confidence,
                    graph_enabled=True,
                    hydrogen_enabled=False,
                    maximum_displacement=P1_COMPARISON_DISPLACEMENT,
                    steps=20,
                )
            )

    resolved_ell = 0.0005
    resolved_gc = gc_for_plane_strain_at2_strength(TARGET_STRENGTH, resolved_ell)
    for graph, label in ((False, "hom"), (True, "graph")):
        cases.append(
            Case(
                name=(
                    f"ijss_recalc_resolved_ell0p5um_b1p5um_"
                    f"{label}_u40_1096x548"
                ),
                stage="D_resolved_length_scale",
                priority="P3",
                purpose=(
                    "high-cost ell/b=1/3 full-domain benchmark with approximately "
                    "four elements per ell at the common u=4e-5 mm state"
                ),
                nx=1096,
                ny=548,
                ell=resolved_ell,
                gc=resolved_gc,
                influence_radius=REFERENCE_B,
                graph_enabled=graph,
                hydrogen_enabled=False,
                maximum_displacement=P1_COMPARISON_DISPLACEMENT,
                steps=20,
            )
        )
    return cases


def bool_toml(value: bool) -> str:
    return "true" if value else "false"


def render(case: Case) -> str:
    chain_artifact = (
        '"../data/ebsd_gb_chains/ban_roi.npz"' if case.graph_enabled else '""'
    )
    confidence_floor = case.confidence_floor if case.graph_enabled else 0.0
    path_block = ""
    if case.path_control:
        path_block = """
[path_control]
enabled = true
functional = "fracture_energy"
switch_displacement = 0.00004
target_increment = 0.000006
steps = 20
adaptive = true
use_energy_predictor = true
maximum_subdivisions = 8
minimum_increment = 2.34375e-8
load_lower_bound = 0.000005
load_upper_bound = 0.00012
snes_max_iterations = 140
residual_tolerance = 1e-8
control_tolerance = 1e-6
"""
    return f"""# IJSS scientific recalculation: {case.stage}, {case.priority}
# Purpose: {case.purpose}
# Strength calibration: plane-strain homogeneous AT2 first peak = {TARGET_STRENGTH:.1f} MPa.
# Resolution: ell/h_K={case.ell_over_h:.6f}, b/h_K={case.b_over_h:.6f}, ell/b={case.ell_over_b:.6f}.

[geometry]
length = {LENGTH}
height = {HEIGHT}
nx = {case.nx}
ny = {case.ny}
precrack_length = 0.0240975

[material]
young_modulus = {YOUNG}
poisson_ratio = {POISSON}
fracture_toughness = {case.gc:.12g}
length_scale = {case.ell}
residual_stiffness = 1e-8

[loading]
maximum_displacement = {case.maximum_displacement}
steps = {case.steps}
stagger_max_iterations = 320
stagger_tolerance = 1e-5
damage_kkt_tolerance = 1e-6
adaptive = true
maximum_subdivisions = 10
minimum_increment = 1e-11

[graph]
enabled = {bool_toml(case.graph_enabled)}
influence_radius = {case.influence_radius}
crack_threshold = 0.7
chain_artifact = {chain_artifact}
confidence_floor = {confidence_floor}
attribute_permutation_seed = -1
{path_block}
[output]
directory = "../results/{case.name}"
write_every = {max(1, case.steps // 12)}

[solver]
linear_ksp_type = "preonly"
linear_pc_type = "lu"
factor_solver = "mumps"
damage_snes_type = "vinewtonrsls"

[hydrogen]
enabled = {bool_toml(case.hydrogen_enabled)}
diffusivity = 0.0001
charging_concentration = 0.05
charging_time = {case.charging_time}
steps = 20
trap_binding_constant = 20.0
background_trap_density = 0.0
toughness_degradation = 0.8
minimum_toughness_ratio = 0.2
charging_boundary = "bottom"
"""


def matrix_row(case: Case) -> dict[str, object]:
    row = asdict(case)
    row.update(
        {
            "h_diagonal_mm": case.h_diagonal,
            "ell_over_hK": case.ell_over_h,
            "b_over_hK": case.b_over_h,
            "ell_over_b": case.ell_over_b,
            "config": f"examples/{case.name}.toml",
            "result": f"results/{case.name}",
        }
    )
    return row


def write_protocol(cases: list[Case]) -> None:
    stage_counts: dict[str, int] = {}
    for case in cases:
        stage_counts[case.stage] = stage_counts.get(case.stage, 0) + 1
    counts = "\n".join(f"- `{stage}`: {count} cases" for stage, count in stage_counts.items())
    min_strength = TARGET_STRENGTH * math.sqrt(0.45 * 0.60)
    PROTOCOL.write_text(
        f"""# IJSS scientific recalculation protocol

## Constitutive calibration

The recalculation uses a transparent AT2 bridge calibration, not a claim that AT2 is the final constitutive choice. For plane strain,

\\[
\\sigma_c=\\frac{{3\\sqrt{{3}}}}{{16}}
\\sqrt{{\\frac{{E/(1-\\nu^2)\\,G_c}}{{\\ell}}}}.
\\]

Setting \\(\\sigma_c=1.00\\) GPa at \\(\\ell=1.5~\\mu\\)m gives
\\(G_c={REFERENCE_GC:.5f}\\) N mm\\(^{-1}\\). The combined uniform-hydrogen and
graph minimum has an intrinsic strength of {min_strength/1000:.3f} GPa. This is
a physically screened elastic baseline; PF-CZM or elasto-plastic calibration
remains preferable for final material interpretation.

## Campaign order

{counts}

Run `P0` first. `P1` is the minimum convergence evidence. `P2` may be interpreted
only as a mechanics-level field-construction sensitivity because the narrow
bands are not independent of the current fracture regularisation. `P3` is the
only campaign with \\(\\ell/b=1/3\\), but it is intentionally high cost.

## Transport boundary

The 5, 10 and 20 s cases test a non-terminal lattice-occupancy field. Trap density
remains zero because the measured artifact contains no calibrated trap density.
These cases must not be described as trapping simulations.

The P1 four-factor convergence comparison is evaluated at
\(u=4.0\times10^{-5}\) mm. Coarse histories may continue beyond this point,
but only the exact common node enters the Richardson/GCI analysis. This
is a common stable displacement node for all factors. It is intentionally
separate from the post-instability path-control evidence and must not be
reported as a global peak or terminal resistance.

## Path-control boundary

The verified path-control implementation currently excludes hydrogen. The P0
path pilot therefore uses the measured graph without hydrogen. Hydrogen cases
remain displacement controlled until the coupled implementation is verified.

## Convergence decision

Endpoint and peak reactions, regularised crack-length change, connected tip
position and paired contrasts must each be assessed. A GCI below 2% is not by
itself sufficient if the contrast or crack path remains mesh sensitive. Report
the observed order, refinement ratios, extrapolated limit and fine-grid GCI.
""",
        encoding="utf-8",
    )


def generate(repo: Path) -> None:
    repo = repo.resolve()
    examples = repo / "examples"
    if not examples.is_dir():
        raise FileNotFoundError(f"examples directory not found: {examples}")
    cases = campaign_cases()
    rows = [matrix_row(case) for case in cases]
    fieldnames = list(rows[0])
    for case in cases:
        target = examples / f"{case.name}.toml"
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            proposed = render(case)
            if existing != proposed:
                if not target.name.startswith("ijss_recalc_"):
                    raise FileExistsError(f"refusing to replace changed config: {target}")
                target.write_text(proposed, encoding="utf-8")
        else:
            target.write_text(render(case), encoding="utf-8")
    with CAMPAIGN_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_protocol(cases)
    print(f"Generated {len(cases)} configurations in {examples}")
    print(CAMPAIGN_CSV)
    print(PROTOCOL)


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else math.nan


def result_row(case: Case, repo: Path) -> dict[str, object]:
    result_dir = repo / "results" / case.name
    history_path = result_dir / "history.csv"
    base = {
        "case": case.name,
        "stage": case.stage,
        "priority": case.priority,
        "nx": case.nx,
        "ny": case.ny,
        "h_diagonal_mm": case.h_diagonal,
        "ell_over_hK": case.ell_over_h,
        "graph_enabled": case.graph_enabled,
        "hydrogen_enabled": case.hydrogen_enabled,
        "charging_time_s": case.charging_time if case.hydrogen_enabled else 0.0,
        "path_control": case.path_control,
        "status": "not_run",
        "source_status": "not_run",
        "comparison_displacement_mm": (
            P1_COMPARISON_DISPLACEMENT
            if case.stage == "B_contrast_convergence"
            else math.nan
        ),
        "accepted_states": 0,
        "last_displacement_mm": math.nan,
        "last_reaction_N_per_mm": math.nan,
        "peak_reaction_N_per_mm": math.nan,
        "peak_displacement_mm": math.nan,
        "last_regularised_crack_length_mm": math.nan,
        "crack_length_change_pct": math.nan,
        "last_rightmost_damaged_x_mm": math.nan,
        "max_damage_kkt_relative": math.nan,
        "max_energy_balance_relative": math.nan,
        "path_states": 0,
    }
    if not history_path.exists():
        return base
    history = read_history(history_path)
    if not history:
        base["status"] = "empty_history"
        base["source_status"] = "empty_history"
        return base
    full_displacements = np.asarray([number(row, "displacement") for row in history])
    source_complete = (
        math.isclose(
            full_displacements[-1],
            case.maximum_displacement,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or case.path_control
    )
    selected_history = history
    comparison_complete = True
    if case.stage == "B_contrast_convergence":
        matching = [
            index
            for index, displacement in enumerate(full_displacements)
            if math.isclose(
                displacement,
                P1_COMPARISON_DISPLACEMENT,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if matching:
            selected_history = history[: matching[-1] + 1]
        else:
            comparison_complete = False
    analysis_complete = (
        comparison_complete if case.stage == "B_contrast_convergence" else source_complete
    )
    reactions = np.asarray([number(row, "reaction_y") for row in selected_history])
    displacements = np.asarray([number(row, "displacement") for row in selected_history])
    peak_index = int(np.nanargmax(reactions))
    first_length = number(selected_history[0], "regularised_crack_length")
    last_length = number(selected_history[-1], "regularised_crack_length")
    base.update(
        {
            "status": "complete" if analysis_complete else "partial",
            "source_status": "complete" if source_complete else "partial",
            "accepted_states": len(selected_history),
            "last_displacement_mm": displacements[-1],
            "last_reaction_N_per_mm": reactions[-1],
            "peak_reaction_N_per_mm": reactions[peak_index],
            "peak_displacement_mm": displacements[peak_index],
            "last_regularised_crack_length_mm": last_length,
            "crack_length_change_pct": 100.0 * (last_length / first_length - 1.0),
            "last_rightmost_damaged_x_mm": number(
                selected_history[-1], "rightmost_damaged_x"
            ),
            "max_damage_kkt_relative": float(
                np.nanmax(
                    [number(row, "damage_kkt_relative") for row in selected_history]
                )
            ),
            "max_energy_balance_relative": float(
                np.nanmax(
                    [
                        abs(number(row, "energy_balance_relative"))
                        for row in selected_history
                    ]
                )
            ),
            "path_states": sum(
                row.get("control_phase", "") == "fracture_energy"
                for row in selected_history
            ),
        }
    )
    return base


def fit_richardson(rows: list[dict[str, object]], metric: str) -> dict[str, float] | None:
    usable = [
        row
        for row in rows
        if row["status"] == "complete" and math.isfinite(float(row[metric]))
    ]
    if len(usable) < 3:
        return None
    usable.sort(key=lambda row: float(row["h_diagonal_mm"]), reverse=True)
    h = np.asarray([float(row["h_diagonal_mm"]) for row in usable])
    values = np.asarray([float(row[metric]) for row in usable])
    best: tuple[float, float, float, float] | None = None
    for p in np.linspace(0.1, 6.0, 5901):
        design = np.column_stack([np.ones_like(h), h**p])
        limit, coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
        error = float(np.sum((values - design @ np.array([limit, coefficient])) ** 2))
        if best is None or error < best[0]:
            best = (error, p, float(limit), float(coefficient))
    assert best is not None
    _, p, limit, coefficient = best
    fine, finer = usable[-2], usable[-1]
    h_fine = float(fine["h_diagonal_mm"])
    h_finer = float(finer["h_diagonal_mm"])
    value_fine = float(fine[metric])
    value_finer = float(finer[metric])
    ratio = h_fine / h_finer
    approximate_error = abs(value_finer - value_fine) / max(abs(value_finer), 1e-30)
    gci = 1.25 * approximate_error / max(ratio**p - 1.0, 1e-30) * 100.0
    return {
        "observed_order": p,
        "extrapolated_limit": limit,
        "coefficient": coefficient,
        "fine_grid_gci_pct": gci,
        "fine_pair_relative_difference_pct": approximate_error * 100.0,
        "fine_refinement_ratio": ratio,
        "n_levels": len(usable),
    }


def factor_label(row: dict[str, object]) -> str | None:
    name = str(row["case"])
    for label in ("hom_noH", "hom_H5s", "graph_noH", "graph_H5s"):
        if f"_{label}_" in name:
            return label
    return None


def contrast_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_mesh: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        if row["stage"] != "B_contrast_convergence":
            continue
        label = factor_label(row)
        if label and row["status"] == "complete":
            by_mesh.setdefault((int(row["nx"]), int(row["ny"])), {})[label] = row
    output: list[dict[str, object]] = []
    comparisons = [
        ("H_effect_graph_off", "hom_H5s", "hom_noH"),
        ("H_effect_graph_on", "graph_H5s", "graph_noH"),
        ("graph_effect_H_off", "graph_noH", "hom_noH"),
        ("graph_effect_H_on", "graph_H5s", "hom_H5s"),
    ]
    for (nx, ny), factors in sorted(by_mesh.items()):
        for label, treatment, baseline in comparisons:
            if treatment not in factors or baseline not in factors:
                continue
            treatment_value = float(factors[treatment]["last_reaction_N_per_mm"])
            baseline_value = float(factors[baseline]["last_reaction_N_per_mm"])
            output.append(
                {
                    "case": f"contrast_{label}_{nx}x{ny}",
                    "stage": "B_contrast_convergence",
                    "priority": "derived",
                    "nx": nx,
                    "ny": ny,
                    "h_diagonal_mm": float(factors[baseline]["h_diagonal_mm"]),
                    "ell_over_hK": float(factors[baseline]["ell_over_hK"]),
                    "graph_enabled": "",
                    "hydrogen_enabled": "",
                    "charging_time_s": P1_H_CHARGE,
                    "path_control": False,
                    "status": "complete",
                    "accepted_states": "",
                    "last_displacement_mm": factors[baseline]["last_displacement_mm"],
                    "last_reaction_N_per_mm": treatment_value - baseline_value,
                    "peak_reaction_N_per_mm": math.nan,
                    "peak_displacement_mm": math.nan,
                    "last_regularised_crack_length_mm": math.nan,
                    "crack_length_change_pct": math.nan,
                    "last_rightmost_damaged_x_mm": math.nan,
                    "max_damage_kkt_relative": math.nan,
                    "max_energy_balance_relative": math.nan,
                    "path_states": "",
                    "contrast_pct": 100.0 * (treatment_value / baseline_value - 1.0),
                }
            )
    return output


def analyse(repo: Path) -> None:
    repo = repo.resolve()
    cases = campaign_cases()
    rows = [result_row(case, repo) for case in cases]
    derived = contrast_rows(rows)
    all_rows = rows + derived
    fieldnames: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    convergence: dict[str, object] = {}
    for factor in ("hom_noH", "hom_H5s", "graph_noH", "graph_H5s"):
        factor_rows = [
            row
            for row in rows
            if row["stage"] == "B_contrast_convergence"
            and factor_label(row) == factor
        ]
        convergence[factor] = {
            metric: fit_richardson(factor_rows, metric)
            for metric in (
                "last_reaction_N_per_mm",
                "peak_reaction_N_per_mm",
                "crack_length_change_pct",
            )
        }
    for contrast_name in (
        "H_effect_graph_off",
        "H_effect_graph_on",
        "graph_effect_H_off",
        "graph_effect_H_on",
    ):
        selected = [row for row in derived if contrast_name in str(row["case"])]
        convergence[contrast_name] = {
            "absolute_reaction_contrast": fit_richardson(
                selected, "last_reaction_N_per_mm"
            ),
            "relative_reaction_contrast": fit_richardson(selected, "contrast_pct"),
        }
    CONVERGENCE_JSON.write_text(
        json.dumps(convergence, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    completed = sum(row["status"] == "complete" for row in rows)
    partial = sum(row["status"] == "partial" for row in rows)
    not_run = sum(row["status"] == "not_run" for row in rows)
    path_rows = [row for row in rows if row["path_control"]]
    path_summary = (
        f"{path_rows[0]['path_states']} accepted fracture-energy states"
        if path_rows and path_rows[0]["status"] != "not_run"
        else "not run"
    )
    STATUS.write_text(
        f"""# IJSS scientific recalculation status

- Complete cases: {completed}
- Partial cases: {partial}
- Not-run cases: {not_run}
- Path-control pilot: {path_summary}

The results table is `{SUMMARY_CSV.name}`. Richardson/GCI fits are stored in
`{CONVERGENCE_JSON.name}`. A fit is emitted only when at least three complete
mesh levels exist. No absent case is imputed and no partial endpoint is treated
as a completed common-load comparison.
""",
        encoding="utf-8",
    )
    print(SUMMARY_CSV)
    print(CONVERGENCE_JSON)
    print(STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "analyse"))
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.repo)
    else:
        analyse(args.repo)


if __name__ == "__main__":
    main()
