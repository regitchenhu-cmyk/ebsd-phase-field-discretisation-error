# M4 edge-crack benchmark result

- Runtime: DOLFINx 0.11.0, PETSc 3.25.4, four MPI ranks.
- Meshes: 184x92 through 736x368; the finest level resolves $\ell$ with approximately 8.1 element diagonals.
- Frozen-crack finest-pair change: 1.3661%.
- Coupled-AT2 finest-pair change: 1.4406%.
- Formal all-level orders: frozen 0.203, coupled 0.235; these remain non-asymptotic diagnostics, not GCI inputs.
- Maximum traction-versus-residual reaction mismatch: 0.1950%.
- Finest coupled crack-measure increase: 0.1719%.

The frozen edge-crack sequence retains substantial mesh drift, so crack representation and the tip field are primary contributors before graph projection is introduced. Allowing AT2 evolution adds a smaller mesh-dependent change. Reaction extraction is not the dominant source because the two independent reactions remain much closer to each other than either sequence is to the finest reference. The 736x368 result is a same-model numerical reference, not an exact sharp-crack solution; local refinement and field-representation comparisons remain required.
