# Distinguishing grain-boundary effects from discretisation error in EBSD-based phase-field fracture simulations

This repository is the reproducibility package for the associated manuscript. It contains the measured electron backscatter diffraction (EBSD) map, the processed boundary-polyline data, the phase-field source code and input files, compact run histories, figure-source data, and the LaTeX manuscript source.

The central numerical question is whether the reaction-force difference between a heterogeneous calculation and its homogeneous counterpart on the same mesh stabilises under mesh refinement. The repository therefore preserves both accepted states and unsuccessful continuation attempts. An unsuccessful attempt is recorded as numerical evidence and is not interpreted as a fracture event.

## Main contents

- `data/ebsd/raw/ban.ctf`: raw EBSD map used in the study (239 x 227 measurement cells, 0.405 micrometre step size).
- `data/ebsd/processed/`: processed boundary-polyline artifact used by the continuum calculations.
- `data/fields/`: compact spatial-field export used for plotting and cross-checks.
- `src/`: Python source for EBSD processing, field construction, audits, and DOLFINx phase-field calculations.
- `examples/`: manuscript-related TOML configurations.
- `evidence/runs/`: resolved configurations and compact accepted/rejected histories from the reported calculations.
- `evidence/summaries/`: benchmark, positive-control, and reaction-extraction tables and diagnostics.
- `figure_source/`: numerical source data and scripts for the manuscript figures.
- `paper/`: manuscript and supplementary-material LaTeX sources with the referenced figures.
- `MANIFEST.tsv` and `SHA256SUMS.txt`: file sizes and SHA-256 fingerprints.

## Terminology and historical identifiers

The manuscript uses `boundary polyline` for the segmented geometry and `boundary field` for its continuum projection. Some archived file names, configuration keys, and run-directory names retain earlier identifiers such as `graph`, `screen`, or `pilot`. They are preserved because changing archival identifiers would break links between configurations, histories, and source data. They should not be read as additional mechanics terminology.

## Data lineage

The principal path is:

```text
ban.ctf
  -> EBSD segmentation and boundary-polyline export
  -> ban_roi.json / ban_roi.npz
  -> continuum toughness and diffusivity fields
  -> homogeneous and heterogeneous phase-field calculations
  -> paired reaction contrasts and mesh-refinement checks
```

The raw CTF file and every derived file in this release can be checked against `SHA256SUMS.txt`.

## Quick start

For preprocessing, plotting, and non-DOLFINx audits:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[legacy]"
graphfracture-field-audit --help
```

For the finite-element calculations, use the supplied container definition or an existing DOLFINx/PETSc installation:

```powershell
docker compose build
docker compose run --rm dolfinx
```

The manuscript configurations are under `examples/`. The resolved configuration stored beside each history is the authoritative record of the parameters used for that run.

## Scope of this GitHub package

GitHub-sized source data, configurations, compact histories, and the raw EBSD map are included directly. Large volumetric XDMF/HDF5 time histories are intentionally excluded from this repository. They can be regenerated from the resolved configurations and should be deposited separately in a DOI-bearing data archive if the final journal data policy requires direct download of every field state.

No calibrated grain-boundary fracture properties are claimed. The high-contrast boundary field is a numerical positive control used to test whether the comparison procedure can resolve an imposed response change.

## Citation

Use the metadata in `CITATION.cff`. If a journal article or data-repository DOI becomes available, add it to that file without changing the checksums of the archived numerical evidence.

## Licence

See `LICENSE`. The current release retains copyright while permitting inspection and reproduction of the reported academic results. The authors may replace it with an open-source and open-data licence before public release if broader reuse is intended.