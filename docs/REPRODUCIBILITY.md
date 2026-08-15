# Reproducibility guide

## 1. Verify the release

Compare local file hashes with `SHA256SUMS.txt`. `MANIFEST.tsv` also records the byte size of each archived file. These checks establish file identity; they are not a substitute for numerical verification.

## 2. Install the lightweight Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[legacy]"
```

This environment supports EBSD processing, compact field audits, and figure generation. DOLFINx is intentionally not installed from PyPI by this command.

## 3. Inspect or regenerate the boundary-polyline data

The raw map is `data/ebsd/raw/ban.ctf`. The relevant export and construction scripts are under `scripts/solver/`. Run a script with `--help` before supplying a manuscript configuration, because the resolved JSON beside each archived result is the authoritative parameter record.

## 4. Run DOLFINx calculations

Use the supplied `Dockerfile` and `compose.yaml`, or a compatible DOLFINx/PETSc environment. The solver entry point is implemented in `src/graphfracture/dolfinx_solver.py`. Manuscript configurations are stored under `examples/`.

The archived histories should not be overwritten. Write reproduced results to a new directory and compare reaction histories, accepted endpoints, and paired contrasts with `evidence/runs/` and `evidence/summaries/`.

## 5. Recreate figures

Figure scripts are in `figure_source/scripts/`; their compact numerical inputs are in `figure_source/data/` and `evidence/summaries/`. Output graphics should be written outside the repository or to a temporary directory so that the archived source package remains unchanged.

## 6. Interpretation rule

The main quantity is the heterogeneous-minus-homogeneous reaction contrast evaluated on the same mesh and at the same accepted loading state. Its change under refinement is assessed directly. The absolute reaction difference between two meshes is retained only as a secondary reference and is not treated as a rigorous error bound for the paired contrast.

The deliberately high-contrast boundary field is a positive numerical control. It does not represent a calibrated material law.