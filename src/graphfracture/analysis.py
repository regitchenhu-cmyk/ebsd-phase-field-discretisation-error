"""Post-process one or more GraphFracture result directories.

The module intentionally has no DOLFINx dependency.  It reduces the tabular
solver diagnostics and final graph state to a compact, JSON-serialisable
research summary suitable for regression comparisons and manuscript tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA_VERSION = 1

__all__ = [
    "SUMMARY_SCHEMA_VERSION",
    "main",
    "summarise_case",
    "summarise_cases",
]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no CSV header")
            fieldnames = list(reader.fieldnames)
            if len(fieldnames) != len(set(fieldnames)):
                raise ValueError(f"{path} has duplicate CSV columns")
            rows = list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required result file is missing: {path}") from exc
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return fieldnames, rows


def _read_json_object(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required result file is missing: {path}")
        return None
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _number(
    row: dict[str, str],
    name: str,
    path: Path,
    *,
    required: bool = False,
) -> float | None:
    value = row.get(name)
    if value is None or not value.strip():
        if required:
            raise ValueError(f"{path}: missing numeric value for {name!r}")
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}: {name!r} must be numeric, found {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: {name!r} must be finite, found {value!r}")
    return result


def _required_number(row: dict[str, str], name: str, path: Path) -> float:
    result = _number(row, name, path, required=True)
    assert result is not None
    return result


def _optional_extreme(
    rows: Sequence[dict[str, str]],
    name: str,
    path: Path,
    *,
    mode: str,
) -> float | None:
    values = [value for row in rows if (value := _number(row, name, path)) is not None]
    if not values:
        return None
    if mode == "min":
        return min(values)
    if mode == "max":
        return max(values)
    if mode == "max_abs":
        return max(abs(value) for value in values)
    raise ValueError(f"unsupported extreme mode: {mode}")


def _optional_sum(
    rows: Sequence[dict[str, str]],
    name: str,
    path: Path,
) -> float | None:
    values = [value for row in rows if (value := _number(row, name, path)) is not None]
    return sum(values) if values else None


def _cumulative_external_work(rows: Sequence[dict[str, str]], history_path: Path) -> list[float]:
    displacements = [_required_number(row, "displacement", history_path) for row in rows]
    reactions = [_required_number(row, "reaction_y", history_path) for row in rows]
    work = [0.0]
    for index in range(1, len(rows)):
        increment = displacements[index] - displacements[index - 1]
        average_reaction = 0.5 * (reactions[index] + reactions[index - 1])
        work.append(work[-1] + average_reaction * increment)
    return work


def _internal_energies(
    rows: Sequence[dict[str, str]], fieldnames: Sequence[str], history_path: Path
) -> list[float] | None:
    if "total_internal_energy" in fieldnames:
        return [_required_number(row, "total_internal_energy", history_path) for row in rows]
    if "elastic_energy" in fieldnames and "fracture_energy" in fieldnames:
        result: list[float] = []
        for row in rows:
            elastic = _required_number(row, "elastic_energy", history_path)
            fracture = _required_number(row, "fracture_energy", history_path)
            result.append(elastic + fracture)
        return result
    return None


def _energy_balance_error(
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
    history_path: Path,
    work: Sequence[float],
) -> float | None:
    if "energy_balance_relative" in fieldnames:
        return _optional_extreme(
            rows,
            "energy_balance_relative",
            history_path,
            mode="max_abs",
        )

    energies = _internal_energies(rows, fieldnames, history_path)
    if energies is None:
        return None
    initial = energies[0]
    errors: list[float] = []
    for energy, external_work in zip(energies, work, strict=True):
        internal_change = energy - initial
        residual = internal_change - external_work
        scale = max(abs(internal_change), abs(external_work), 1.0e-30)
        errors.append(abs(residual / scale))
    return max(errors, default=0.0)


def _coerce_csv_scalar(value: str, path: Path, name: str) -> Any:
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        result = float(text)
    except ValueError:
        return text
    if not math.isfinite(result):
        raise ValueError(f"{path}: {name!r} must be finite, found {value!r}")
    return result


def _hydrogen_metrics(
    case_directory: Path,
) -> tuple[dict[str, Any] | None, dict[str, float | None] | None]:
    path = case_directory / "hydrogen_history.csv"
    if not path.is_file():
        return None, None
    fieldnames, rows = _read_csv(path)
    final = rows[-1]
    final_metrics = {
        name: _coerce_csv_scalar(final.get(name, ""), path, name) for name in fieldnames
    }
    history_metrics = {
        "maximum_absolute_physical_mass_balance_relative": (
            _optional_extreme(rows, "physical_mass_balance_relative", path, mode="max_abs")
            if "physical_mass_balance_relative" in fieldnames
            else None
        ),
        "maximum_absolute_algebraic_mass_balance_relative": (
            _optional_extreme(rows, "algebraic_mass_balance_relative", path, mode="max_abs")
            if "algebraic_mass_balance_relative" in fieldnames
            else None
        ),
        "maximum_absolute_internal_bound_reaction_rate": (
            _optional_extreme(rows, "internal_bound_reaction_rate", path, mode="max_abs")
            if "internal_bound_reaction_rate" in fieldnames
            else None
        ),
    }
    return final_metrics, history_metrics


def summarise_case(case_directory: str | Path) -> dict[str, Any]:
    """Return a compact research summary for one solver result directory."""
    case = Path(case_directory).resolve()
    if not case.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {case}")

    history_path = case / "history.csv"
    fieldnames, rows = _read_csv(history_path)
    required_columns = {"displacement", "reaction_y", "regularised_crack_length"}
    missing = sorted(required_columns - set(fieldnames))
    if missing:
        raise ValueError(f"{history_path} is missing columns: {', '.join(missing)}")

    displacement_reaction = [
        (
            _required_number(row, "displacement", history_path),
            _required_number(row, "reaction_y", history_path),
        )
        for row in rows
    ]
    peak_index, (peak_displacement, peak_reaction) = max(
        enumerate(displacement_reaction),
        key=lambda item: abs(item[1][1]),
    )

    work_history = _cumulative_external_work(rows, history_path)
    recorded_work = (
        _number(rows[-1], "external_work", history_path) if "external_work" in fieldnames else None
    )
    external_work = work_history[-1] if recorded_work is None else recorded_work

    graph_metrics = _read_json_object(case / "graph_metrics.json", required=True)
    assert graph_metrics is not None
    if "G_crack" not in graph_metrics:
        raise ValueError(f"{case / 'graph_metrics.json'} has no 'G_crack' entry")
    final_crack_graph = graph_metrics["G_crack"]
    if final_crack_graph is not None and not isinstance(final_crack_graph, dict):
        raise ValueError("graph_metrics.json 'G_crack' entry must be an object or null")

    front_column = "rightmost_damaged_x" if "rightmost_damaged_x" in fieldnames else "tip_x"
    final_rightmost_damaged = _number(rows[-1], front_column, history_path)
    maximum_rightmost_damaged = _optional_extreme(rows, front_column, history_path, mode="max")
    final_main_crack_tip = (
        final_crack_graph.get("tip_x") if isinstance(final_crack_graph, dict) else None
    )
    final_length = _required_number(rows[-1], "regularised_crack_length", history_path)
    final_step = (
        _coerce_csv_scalar(rows[-1].get("step", ""), history_path, "step")
        if "step" in fieldnames
        else len(rows) - 1
    )
    final_hydrogen, hydrogen_history = _hydrogen_metrics(case)
    completion = _read_json_object(case / "completion.json", required=False)
    complete_result = bool(
        completion is not None
        and completion.get("status") == "complete"
        and completion.get("all_steps_converged") is True
    )
    maximum_stagger_iterations = _optional_extreme(
        rows, "stagger_iterations", history_path, mode="max"
    )
    total_aitken_accepted_iterations = _optional_sum(
        rows, "aitken_accepted_iterations", history_path
    )

    return {
        "case": case.name,
        "directory": str(case),
        "history_records": len(rows),
        "final_step": final_step,
        "peak_reaction_y": peak_reaction,
        "peak_reaction_displacement": peak_displacement,
        "maximum_observed_reaction_y": peak_reaction,
        "maximum_observed_reaction_displacement": peak_displacement,
        "maximum_observed_reaction_at_final_record": peak_index == len(rows) - 1,
        "final_rightmost_damaged_x": final_rightmost_damaged,
        "maximum_rightmost_damaged_x": maximum_rightmost_damaged,
        "final_main_crack_tip_x": final_main_crack_tip,
        "final_regularised_crack_length": final_length,
        "external_work": external_work,
        "maximum_absolute_energy_balance_relative_error": _energy_balance_error(
            rows, fieldnames, history_path, work_history
        ),
        "minimum_damage_increment": _optional_extreme(
            rows, "minimum_damage_increment", history_path, mode="min"
        ),
        "maximum_damage_kkt_inf": _optional_extreme(
            rows, "damage_kkt_inf", history_path, mode="max"
        ),
        "maximum_damage_kkt_relative": _optional_extreme(
            rows, "damage_kkt_relative", history_path, mode="max"
        ),
        "maximum_stagger_iterations": (
            int(maximum_stagger_iterations) if maximum_stagger_iterations is not None else None
        ),
        "total_aitken_accepted_iterations": (
            int(total_aitken_accepted_iterations)
            if total_aitken_accepted_iterations is not None
            else None
        ),
        "grain_boundary_graph": graph_metrics.get("G_GB"),
        "final_crack_graph": final_crack_graph,
        "final_hydrogen": final_hydrogen,
        "hydrogen_history_diagnostics": hydrogen_history,
        "runtime": _read_json_object(case / "runtime.json", required=False),
        "complete_result": complete_result,
        "completion": completion,
    }


def summarise_cases(case_directories: Sequence[str | Path]) -> dict[str, Any]:
    """Return a versioned summary payload for cases in caller-supplied order."""
    if not case_directories:
        raise ValueError("at least one result directory is required")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "cases": [summarise_case(case) for case in case_directories],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphfracture-analysis",
        description="Summarise GraphFracture history, graph and hydrogen diagnostics.",
    )
    parser.add_argument("cases", nargs="+", type=Path, help="result case directories")
    parser.add_argument("--output", "-o", type=Path, help="write the summary to JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = summarise_cases(args.cases)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is None:
            print(encoded)
        else:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
