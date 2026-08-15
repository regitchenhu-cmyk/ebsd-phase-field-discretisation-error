"""Channel Text File (``.ctf``) reader for the seventh-paper EBSD maps.

The HKL/AZtec ``.ctf`` export is a tab-delimited ASCII grid: a short header that
declares the grid size (``XCells``/``YCells``), the step (``XStep``/``YStep``,
microns) and the indexed phases, followed by one row per measurement point with
columns ``Phase X Y Bands Error Euler1 Euler2 Euler3 MAD BC BS``.

``Phase == 0`` marks a non-indexed (zero-solution) point.  The data are stored
row-major with ``X`` varying fastest, so a straight reshape to ``(YCells,
XCells)`` recovers the spatial map with ``[iy, ix]`` indexing.

The two maps used by the paper are the rolling-plane section ``ban`` (RD-TD) and
the columnar section ``zhu`` (ND-RD) of a hydrogen-charged lath-martensitic
steel; the b.c.c. matrix is phase ``Iron bcc (old)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from os import environ
from pathlib import Path

import numpy as np

from . import crystallography as xtal

EBSD_ROOT_ENV = "PFM_EBSD_ROOT"
_EBSD_RELATIVE_ROOT = Path("EBSD data") / "EBSD data"


def _find_project_root() -> Path:
    """Return the nearest source ancestor containing ``pyproject.toml``."""
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Source-tree fallback for an unpackaged checkout: <root>/src/inverse_pfm/ebsd.py.
    return module_path.parents[2]


_PROJECT_ROOT = _find_project_root()


def _normalise_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def ebsd_root_candidates() -> tuple[Path, ...]:
    """Candidate directories containing the ``ban`` and ``zhu`` subdirectories.

    ``PFM_EBSD_ROOT`` is an explicit override.  Otherwise the search starts at
    the project root and walks towards the filesystem root, looking for the
    shared ``EBSD data/EBSD data`` export at each level.
    """
    configured = environ.get(EBSD_ROOT_ENV)
    if configured:
        return (_normalise_path(configured),)

    bases = (_PROJECT_ROOT, *_PROJECT_ROOT.parents)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        candidate = _normalise_path(base / _EBSD_RELATIVE_ROOT)
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    return tuple(candidates)


def _preferred_ebsd_root() -> Path:
    candidates = ebsd_root_candidates()
    if not candidates:
        return _normalise_path(_PROJECT_ROOT / _EBSD_RELATIVE_ROOT)
    for root in candidates:
        if all((root / section / f"{section}.ctf").is_file() for section in ("ban", "zhu")):
            return root
    for root in candidates:
        if root.is_dir():
            return root
    return candidates[0]


# Compatibility mapping retained for existing scripts and callers.  ``load``
# resolves the environment/default search dynamically unless a caller mutates a
# mapping entry explicitly.
_DEFAULT_EBSD_ROOT = _preferred_ebsd_root()
CTF_PATHS = {
    "ban": str(_DEFAULT_EBSD_ROOT / "ban" / "ban.ctf"),
    "zhu": str(_DEFAULT_EBSD_ROOT / "zhu" / "zhu.ctf"),
}
_INITIAL_CTF_PATHS = CTF_PATHS.copy()


def _section_candidates(section: str) -> tuple[Path, ...]:
    configured = CTF_PATHS[section]
    if configured != _INITIAL_CTF_PATHS[section]:
        return (_normalise_path(configured),)
    return tuple(root / section / f"{section}.ctf" for root in ebsd_root_candidates())


def resolve_ctf_path(section: str) -> Path:
    """Resolve a named section to an existing CTF input path.

    Raises a diagnostic :class:`FileNotFoundError` rather than letting a later
    ``open`` call report only the final candidate.
    """
    if section not in CTF_PATHS:
        raise KeyError(f"unknown section {section!r}; choose from {list(CTF_PATHS)}")

    candidates = _section_candidates(section)
    for path in candidates:
        if path.is_file():
            return path

    checked = "\n".join(f"  - {path}" for path in candidates) or "  - <none>"
    raise FileNotFoundError(
        f"EBSD CTF input for section {section!r} was not found.\n"
        f"Checked candidate paths:\n{checked}\n"
        f"Set {EBSD_ROOT_ENV} to the directory containing "
        "ban/ban.ctf and zhu/zhu.ctf."
    )


def input_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 digest of an input file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ctf_sha256(section: str) -> str:
    """Resolve a named EBSD section and return its input SHA256 digest."""
    return input_sha256(resolve_ctf_path(section))


# Human-readable axis labels for the two sample sections.
SECTION_AXES = {
    "ban": ("RD", "TD"),  # rolling plane
    "zhu": ("ND", "RD"),  # columnar section
}


@dataclass
class Phase:
    """A single indexed phase from the ``.ctf`` header."""

    number: int  # 1-based id as written in the data column
    name: str
    lattice: tuple  # (a, b, c) in angstrom
    laue_group: int


@dataclass
class EBSDMap:
    """A parsed EBSD map on a regular grid.

    All 2-D fields are ``(ny, nx)`` with ``[iy, ix]`` indexing; ``euler`` is
    ``(ny, nx, 3)`` Bunge angles in degrees.
    """

    name: str
    nx: int
    ny: int
    dx: float
    dy: float
    phase: np.ndarray  # (ny, nx) int, 0 = non-indexed
    euler: np.ndarray  # (ny, nx, 3) deg
    bc: np.ndarray  # (ny, nx) band contrast (pattern quality proxy)
    mad: np.ndarray  # (ny, nx) mean angular deviation (fit quality)
    bands: np.ndarray  # (ny, nx) number of detected bands
    phases: list  # list[Phase]

    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple:
        return (self.ny, self.nx)

    @property
    def indexed(self) -> np.ndarray:
        """Boolean mask of successfully indexed points."""
        return self.phase > 0

    @property
    def extent(self) -> tuple:
        """``(x0, x1, y0, y1)`` in microns for ``imshow(origin='upper')``."""
        return (0.0, self.nx * self.dx, self.ny * self.dy, 0.0)

    def phase_mask(self, phase_id: int) -> np.ndarray:
        return self.phase == phase_id

    def matrix_phase_id(self) -> int:
        """Id of the b.c.c. matrix (most abundant indexed phase, expected bcc)."""
        ids = [p.number for p in self.phases]
        counts = {pid: int(np.sum(self.phase == pid)) for pid in ids}
        return max(counts, key=counts.get)

    def quaternions(self) -> np.ndarray:
        """Orientation quaternions ``(ny, nx, 4)`` (non-indexed points -> identity)."""
        q = xtal.euler_to_quaternion(self.euler.reshape(-1, 3))
        q = q.reshape(self.ny, self.nx, 4)
        bad = ~self.indexed
        q[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        return q

    def phase_fractions(self) -> dict:
        n_idx = max(int(np.sum(self.indexed)), 1)
        out = {}
        for p in self.phases:
            out[p.name] = float(np.sum(self.phase == p.number) / n_idx)
        return out


def _parse_phase_line(line: str) -> Phase | None:
    """Parse one phase declaration row of the header, or ``None``."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4 or ";" not in parts[0]:
        return None
    try:
        a, b, c = (float(v) for v in parts[0].split(";"))
    except ValueError:
        return None
    name = parts[2].strip()
    try:
        laue = int(parts[3])
    except (ValueError, IndexError):
        laue = 0
    return Phase(number=0, name=name, lattice=(a, b, c), laue_group=laue)


def read_ctf(path: str | Path, name: str | None = None) -> EBSDMap:
    """Read a Channel Text File into an :class:`EBSDMap`."""
    path = Path(path)
    if name is None:
        name = path.stem

    with path.open("r", encoding="latin-1") as fh:
        lines = fh.readlines()

    nx = ny = None
    dx = dy = None
    phases: list[Phase] = []
    data_start = None
    for i, line in enumerate(lines):
        key = line.split("\t", 1)[0].strip()
        if key == "XCells":
            nx = int(line.split("\t")[1])
        elif key == "YCells":
            ny = int(line.split("\t")[1])
        elif key == "XStep":
            dx = float(line.split("\t")[1])
        elif key == "YStep":
            dy = float(line.split("\t")[1])
        elif key == "Phase" and line.split("\t")[1].strip() == "X":
            data_start = i + 1
            break
        else:
            ph = _parse_phase_line(line)
            if ph is not None:
                ph.number = len(phases) + 1
                phases.append(ph)

    if None in (nx, ny, dx, dy) or data_start is None:
        raise ValueError(f"malformed CTF header in {path}")

    raw = np.loadtxt(lines[data_start:], dtype=float)
    expected = nx * ny
    if raw.shape[0] != expected:
        raise ValueError(f"{path}: got {raw.shape[0]} rows, expected {expected} (= {nx}*{ny})")

    phase = raw[:, 0].astype(int).reshape(ny, nx)
    euler = raw[:, 5:8].reshape(ny, nx, 3)
    bands = raw[:, 3].astype(int).reshape(ny, nx)
    mad = raw[:, 8].reshape(ny, nx)
    bc = raw[:, 9].reshape(ny, nx)

    return EBSDMap(
        name=name,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        phase=phase,
        euler=euler,
        bc=bc,
        mad=mad,
        bands=bands,
        phases=phases,
    )


def load(section: str) -> EBSDMap:
    """Load a named section (``"ban"`` or ``"zhu"``) from the shared export."""
    return read_ctf(resolve_ctf_path(section), name=section)


def self_test() -> dict:
    """Read both sections and report basic grid/phase sanity."""
    out: dict = {}
    ok = True
    for sec in CTF_PATHS:
        try:
            m = load(sec)
        except FileNotFoundError:
            out[sec] = "missing"
            ok = False
            continue
        frac = m.phase_fractions()
        matrix = m.matrix_phase_id()
        info = {
            "shape": m.shape,
            "step_um": (m.dx, m.dy),
            "n_indexed": int(np.sum(m.indexed)),
            "index_rate": float(np.mean(m.indexed)),
            "matrix_phase": next(p.name for p in m.phases if p.number == matrix),
            "phase_fractions": {k: round(v, 4) for k, v in frac.items()},
        }
        out[sec] = info
        ok = ok and m.nx > 0 and m.ny > 0 and info["n_indexed"] > 0
    out["ok"] = ok
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2, ensure_ascii=False))
