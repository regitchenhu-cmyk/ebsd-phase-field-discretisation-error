"""Command-line entry point for reproducible DOLFINx studies."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path

from .config import load_config
from .gb_graph import boundary_field_from_config


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphfracture",
        description="Run a graph-informed AT2 phase-field fracture case with DOLFINx.",
    )
    parser.add_argument("config", type=Path, help="versioned TOML run configuration")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print scales without importing DOLFINx",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact accepted state in output/restart/checkpoint.json",
    )
    parser.add_argument(
        "--resume-stagger-max-iterations",
        type=_positive_int,
        help="monotonically increase the committed outer-iteration budget",
    )
    parser.add_argument(
        "--resume-maximum-subdivisions",
        type=_positive_int,
        help="monotonically increase the committed adaptive subdivision limit",
    )
    parser.add_argument(
        "--resume-minimum-increment",
        type=_positive_float,
        help="monotonically decrease the committed minimum adaptive increment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    continuation_overrides = (
        args.resume_stagger_max_iterations,
        args.resume_maximum_subdivisions,
        args.resume_minimum_increment,
    )
    if any(value is not None for value in continuation_overrides) and not args.resume:
        parser.error("resume continuation controls require --resume")
    try:
        config = load_config(args.config)
    except (OSError, ValueError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    summary = config.diagnostics()
    field = None
    if config.graph.enabled:
        try:
            field = boundary_field_from_config(config)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"boundary field error: {exc}", file=sys.stderr)
            return 2
        summary["G_GB"] = field.describe()
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    try:
        from .dolfinx_solver import AT2Solver
    except ModuleNotFoundError as exc:
        if exc.name in {"dolfinx", "mpi4py", "petsc4py", "ufl"}:
            print(
                "DOLFINx runtime not found. Use the pinned container from compose.yaml, "
                "or install DOLFINx 0.11.0.post0 in WSL/Linux.",
                file=sys.stderr,
            )
            return 3
        raise

    AT2Solver(config, boundary_field=field).run(
        resume=args.resume,
        resume_stagger_max_iterations=args.resume_stagger_max_iterations,
        resume_maximum_subdivisions=args.resume_maximum_subdivisions,
        resume_minimum_increment=args.resume_minimum_increment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
