"""Hydrogen transport on the grain-boundary network and the Gc coupling.

Hydrogen reaches the microstructure from a charging surface, diffuses slowly
through the b.c.c. lattice but races along the connected high-angle boundary
network (the rasterised ``D_H(x)`` field from the graph), and is trapped at
boundaries and stored-dislocation sites (the rasterised ``rho_trap(x)`` field).
The resulting occupancy ``theta_H(x)`` then softens the fracture energy through
the HEDE-style coupling used across this paper series,

    Gc_eff(x) = Gc(x) * (1 - chi * theta_H(x)).

Transport model
---------------
* Steady lattice diffusion with the variable-coefficient operator
  ``-div(D_H grad c_L) = 0`` (Neumann walls, Dirichlet ``c_L = c_charge`` on the
  charging surface), optionally with stress drift
  ``-div(D_H (grad c_L - (Vh/RT) c_L grad sigma_h))`` when a hydrostatic-stress
  field is supplied (central differences).
* Local Oriani trapping equilibrium: ``c_T = rho_trap * K c_L / (1 + K c_L)``;
  the occupancy that enters the softening is the normalised total content
  ``theta_H = (c_L + c_T) / max(c_L + c_T)``, so it peaks along the trapping
  boundary network.

Reuses :func:`inverse_pfm.graph_pf.variable_laplacian`.  Pure numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .graph_pf import apply_dirichlet as _apply_dirichlet
from .graph_pf import variable_laplacian


@dataclass
class HydrogenParams:
    D_bulk: float = 1.0  # bulk lattice diffusivity (normalised)
    c_charge: float = 1.0  # lattice concentration at the charging surface
    K_trap: float = 20.0  # Oriani trap binding constant
    chi: float = 0.90  # HEDE toughness-degradation coefficient
    Vh_over_RT: float = 0.0  # stress-drift coefficient (0 = diffusion only)


class HydrogenNetwork:
    """Steady hydrogen field on a rasterised diffusivity / trap network."""

    def __init__(self, shape, step, params: HydrogenParams | None = None):
        self.ny, self.nx = shape
        self.dx, self.dy = step
        self.params = params or HydrogenParams()
        self.D_field = np.full(shape, self.params.D_bulk)
        self.trap_field = np.zeros(shape)
        self._charge = np.zeros(shape, dtype=bool)
        self._sink = np.zeros(shape, dtype=bool)

    # ------------------------------------------------------------------ #
    def set_fields(self, D_field=None, trap_field=None):
        if D_field is not None:
            self.D_field = np.asarray(D_field, dtype=float).reshape(self.ny, self.nx)
        if trap_field is not None:
            self.trap_field = np.asarray(trap_field, dtype=float).reshape(self.ny, self.nx)

    @staticmethod
    def _edge_mask(target, edge):
        if edge == "left":
            target[:, 0] = True
        elif edge == "right":
            target[:, -1] = True
        elif edge == "top":
            target[0, :] = True
        elif edge == "bottom":
            target[-1, :] = True

    def set_charging(self, mask=None, edge=None):
        """Dirichlet charging surface (``c = c_charge``): a ``mask`` or an ``edge``."""
        self._charge[:] = False
        if mask is not None:
            self._charge |= np.asarray(mask, dtype=bool)
        self._edge_mask(self._charge, edge)

    def set_sink(self, mask=None, edge=None):
        """Dirichlet absorbing surface (``c = 0``): a ``mask`` or an ``edge``."""
        self._sink[:] = False
        if mask is not None:
            self._sink |= np.asarray(mask, dtype=bool)
        self._edge_mask(self._sink, edge)

    # ------------------------------------------------------------------ #
    def _drift_operator(self, sigma_h):
        """Central-difference ``-div(D (Vh/RT) c grad sigma_h)`` operator."""
        ny, nx = self.ny, self.nx
        mu = self.params.Vh_over_RT
        idx = np.arange(ny * nx).reshape(ny, nx)
        D = self.D_field
        rows, cols, vals = [], [], []

        def add(a, b, h):
            # advection velocity on the face from the stress gradient
            v = (
                mu
                * 0.5
                * (D.ravel()[a] + D.ravel()[b])
                * (sigma_h.ravel()[b] - sigma_h.ravel()[a])
                / h
            )
            # central split of the flux onto the two cells
            coef = v / (2.0 * h)
            rows.extend([a, a, b, b])
            cols.extend([a, b, a, b])
            vals.extend([coef, coef, -coef, -coef])

        add(idx[:, :-1].ravel(), idx[:, 1:].ravel(), self.dx)
        add(idx[:-1, :].ravel(), idx[1:, :].ravel(), self.dy)
        n = ny * nx
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )

    def solve_steady(self, sigma_h=None) -> np.ndarray:
        """Steady lattice concentration ``c_L`` on the network."""
        n = self.ny * self.nx
        A = variable_laplacian(self.D_field, self.dx, self.dy)
        if sigma_h is not None and self.params.Vh_over_RT != 0.0:
            A = A + self._drift_operator(np.asarray(sigma_h).reshape(self.ny, self.nx))
        b = np.zeros(n)
        charge = self._charge.ravel()
        sink = self._sink.ravel() & ~charge
        if not charge.any():
            raise ValueError("no charging surface set")
        dir_mask = charge | sink
        dir_vals = np.where(charge, self.params.c_charge, 0.0)
        A, b = _apply_dirichlet(A, b, dir_mask, dir_vals)
        c = spsolve(A.tocsc(), b)
        return np.clip(c, 0.0, None).reshape(self.ny, self.nx)

    def solve_transient(self, total_time, n_steps, sigma_h=None, snapshots=None):
        """Transient charging with trap-modified capacity (implicit Euler).

        Solves ``(1 + dcT/dcL) dcL/dt = div(D grad cL) (+ drift)`` from a
        hydrogen-free start, pinning the charging surface to ``c_charge`` (and any
        sink to 0).  Fast-diffusivity boundaries advance the front ahead of the
        bulk; dense traps retard the local front but accumulate hydrogen.

        Returns the final ``c_L``; if ``snapshots`` (list of step indices) is
        given, also returns ``{step: c_L}``.
        """
        n = self.ny * self.nx
        L = variable_laplacian(self.D_field, self.dx, self.dy)
        if sigma_h is not None and self.params.Vh_over_RT != 0.0:
            L = L + self._drift_operator(np.asarray(sigma_h).reshape(self.ny, self.nx))
        dt = total_time / n_steps
        K = self.params.K_trap
        NT = self.trap_field.ravel()

        c = np.zeros(n)
        charge = self._charge.ravel()
        sink = self._sink.ravel() & ~charge
        c[charge] = self.params.c_charge
        dir_mask = charge | sink
        dir_vals = np.where(charge, self.params.c_charge, 0.0)

        snaps = {}
        want = set(snapshots or [])
        for step in range(n_steps):
            beta = NT * K / (1.0 + K * c) ** 2  # dcT/dcL at c^n (lagged)
            cap = (1.0 + beta) / dt
            A = L + sparse.diags(cap)
            b = cap * c
            A, b = _apply_dirichlet(A, b, dir_mask, dir_vals)
            c = spsolve(A.tocsc(), b)
            c = np.clip(c, 0.0, None)
            if step in want:
                snaps[step] = c.reshape(self.ny, self.nx).copy()
        c2d = c.reshape(self.ny, self.nx)
        return (c2d, snaps) if snapshots else c2d

    # ------------------------------------------------------------------ #
    def occupancy(self, c_L: np.ndarray) -> np.ndarray:
        """Normalised total hydrogen content (lattice + trapped) in [0, 1]."""
        K = self.params.K_trap
        c_T = self.trap_field * (K * c_L) / (1.0 + K * c_L)
        total = c_L + c_T
        peak = total.max()
        return total / peak if peak > 0 else total

    def gc_effective(self, Gc: np.ndarray, theta_H: np.ndarray) -> np.ndarray:
        """HEDE softening ``Gc_eff = Gc (1 - chi theta_H)`` (floored positive)."""
        return Gc * np.maximum(1.0 - self.params.chi * theta_H, 1e-3)

    def solve(self, Gc=None, sigma_h=None) -> dict:
        """Convenience: steady ``c_L``, occupancy and (optionally) ``Gc_eff``."""
        c_L = self.solve_steady(sigma_h=sigma_h)
        theta = self.occupancy(c_L)
        out = {"c_L": c_L, "theta_H": theta}
        if Gc is not None:
            out["Gc_eff"] = self.gc_effective(np.asarray(Gc), theta)
        return out


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def self_test() -> dict:
    out: dict = {}

    # 1. Uniform-D, left source + right sink -> linear steady profile.
    ny, nx = 20, 40
    hn = HydrogenNetwork((ny, nx), (1.0, 1.0), HydrogenParams(K_trap=0.0))
    hn.set_charging(edge="left")
    hn.set_sink(edge="right")
    hn.set_fields(D_field=np.ones((ny, nx)))
    c = hn.solve_steady()
    expected = np.linspace(1, 0, nx)[None, :] * np.ones((ny, 1))
    out["linear_profile_err"] = float(np.max(np.abs(c - expected)))

    # 2. Transient charging: a fast-diffusivity boundary advances the hydrogen
    #    front far deeper than the surrounding bulk over the same time.
    hn2 = HydrogenNetwork((ny, nx), (1.0, 1.0), HydrogenParams(K_trap=0.0))
    hn2.set_charging(edge="left")
    D = np.ones((ny, nx))
    D[ny // 2, :] = 50.0  # fast boundary channel
    hn2.set_fields(D_field=D)
    c2 = hn2.solve_transient(total_time=2.0, n_steps=40)

    def penetration(row):
        cols = np.where(c2[row] > 0.3)[0]
        return cols.max() if cols.size else 0

    out["channel_penetration"] = int(penetration(ny // 2))
    out["bulk_penetration"] = int(penetration(2))

    # 3. Trapping raises total content where rho_trap is high.
    hn3 = HydrogenNetwork((ny, nx), (1.0, 1.0), HydrogenParams(K_trap=20.0))
    hn3.set_charging(edge="left")
    trap = np.zeros((ny, nx))
    trap[ny // 2, :] = 5.0
    hn3.set_fields(D_field=np.ones((ny, nx)), trap_field=trap)
    res = hn3.solve(Gc=np.ones((ny, nx)))
    out["theta_peak_on_trap"] = bool(res["theta_H"][ny // 2, nx // 2] > res["theta_H"][2, nx // 2])
    out["gc_eff_min"] = float(res["Gc_eff"].min())

    out["ok"] = bool(
        out["linear_profile_err"] < 1e-9
        and out["channel_penetration"] > 2 * max(out["bulk_penetration"], 1)
        and out["theta_peak_on_trap"]
        and out["gc_eff_min"] < 1.0
    )
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
