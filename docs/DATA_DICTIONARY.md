# Data dictionary

## Raw EBSD map

`data/ebsd/raw/ban.ctf` is the measured map used in the manuscript. Its header records 239 x 227 measurement cells and an in-plane step of 0.405 micrometre in both directions. The file is retained byte-for-byte; use `SHA256SUMS.txt` to verify it.

The separate `zhu.ctf` map present in the working directory was not used for the calculations reported in the manuscript and is deliberately excluded from this release.

## Processed EBSD data

`ban_roi.json` stores human-readable metadata for the selected region and boundary-polyline construction. `ban_roi.npz` stores the corresponding numerical arrays. Together they are the compact processed input to the continuum field construction.

## Compact spatial fields

`data/fields/spatial_fields_export.npz` contains plotting and cross-check arrays for the mapped material fields. It is not a substitute for every time-dependent finite-element field.

## Configurations and run histories

Each directory under `evidence/runs/` preserves the original run identifier. The key files are:

- `config.resolved.json`: fully resolved parameters used by the run.
- `history.csv`: accepted-state history and reported quantities of interest.
- `attempt_history.json`: attempted increments, including rejected attempts when available.
- `continuation_history.json`: continuation-control history when available.

The precrack definition is retained through the source implementation and the resolved run configurations. No reconstructed image or undocumented hand-edited crack mask is substituted for that definition.

## Figure-source tables

CSV files under `figure_source/data/` contain the values used to draw the convergence, parameter, and continuation figures. Some file names retain numbering from an earlier internal layout; the manuscript source is authoritative for the final figure number.

## Units

The solver configurations use the unit system stated in the manuscript. Lengths reported in the EBSD input are in micrometres. Reaction resultants in the manuscript are reported per unit out-of-plane thickness.