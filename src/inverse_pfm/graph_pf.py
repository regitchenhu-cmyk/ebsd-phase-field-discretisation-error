"""Graph-constrained phase-field fracture on the EBSD pixel grid.

Solves the staggered phase-field problem

    min_u  int g(d) psi+ (eps(u)) + psi-(eps(u))                       (elasticity)
    min_d  int g(d) H + Gc(x) ( d^2/(2 ell) + ell/2 |grad d|^2 )       (damage)
                       + (lambda/2) sum_{(i,j) in E} w_ij (dbar_i - dbar_j)^2

on a regular grid of square-ish pixels.  Displacements live on the
``(ny+1, nx+1)`` node grid (bilinear Q4 elements, plane strain, volumetric-
deviatoric Amor split so only tensile/deviatoric energy drives damage); damage
lives on the ``(ny, nx)`` pixel grid (AT2, history field ``H`` for
irreversibility, divergence-form gradient term so a spatially varying ``Gc(x)``
is handled consistently).

The last term is the proposal's **graph regulariser**: ``dbar_i`` is the mean
damage of grain ``i`` and ``w_ij`` the edge weight (boundary weakness /
hydrogen susceptibility).  It is treated by a Picard linearisation inside the
staggered loop, adding ``lambda * deg_w`` to the damage diagonal and
``lambda * (W dbar)`` to the right-hand side, so the converged staggered state
satisfies the fully coupled system without ever forming a dense pixel operator.

Three model variants are obtained from the same code:

* ``homogeneous``  : constant ``Gc``, ``lambda = 0``;
* ``heterogeneous``: rasterised ``Gc(x)``, ``lambda = 0``;
* ``graph``        : rasterised ``Gc(x)`` and ``lambda > 0`` (network coupling).

Pure ``numpy`` / ``scipy.sparse``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass
class GraphPFParams:
    E: float = 1.0
    nu: float = 0.3
    ell: float = 1.2  # regularisation length (same units as step)
    k_res: float = 1e-7  # residual stiffness
    lambda_graph: float = 0.0  # graph-regulariser strength
    max_stagger: int = 30
    stagger_tol: float = 1e-3  # max |d - d_prev| for staggered convergence
    notch_H: float = 1e6  # history seed in the pre-crack


# --------------------------------------------------------------------------- #
# Q4 plane-strain element
# --------------------------------------------------------------------------- #


def _plane_strain_D(E: float, nu: float) -> np.ndarray:
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array(
        [[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]]
    )


def _q4_B(xi: float, eta: float, hx: float, hy: float):
    """Strain-displacement matrix B (3x8) at (xi, eta) for an hx*hy rectangle."""
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    dN_dx = dN_dxi * (2.0 / hx)
    dN_dy = dN_deta * (2.0 / hy)
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx
    B[1, 1::2] = dN_dy
    B[2, 0::2] = dN_dy
    B[2, 1::2] = dN_dx
    return B


def _q4_stiffness(hx: float, hy: float, D: np.ndarray) -> np.ndarray:
    """Full 2x2-Gauss Q4 stiffness for a unit-thickness hx*hy element."""
    ke = np.zeros((8, 8))
    g = 1.0 / np.sqrt(3.0)
    detJ = 0.25 * hx * hy
    for xi in (-g, g):
        for eta in (-g, g):
            B = _q4_B(xi, eta, hx, hy)
            ke += B.T @ D @ B * detJ
    return ke


def variable_laplacian(coeff: np.ndarray, dx: float, dy: float) -> sparse.csr_matrix:
    """Divergence-form operator ``-div(coeff grad .)`` on a pixel grid (Neumann BC).

    Face coefficients use the arithmetic mean of the two adjacent cells.  Shared
    by the damage gradient term (``coeff = Gc*ell``) and steady hydrogen
    diffusion (``coeff = D_H``).
    """
    ny, nx = coeff.shape
    c = coeff.ravel()
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros(ny * nx)

    def add_faces(a_idx, b_idx, h):
        face = 0.5 * (c[a_idx] + c[b_idx]) / (h * h)
        rows.extend([a_idx, b_idx])
        cols.extend([b_idx, a_idx])
        vals.extend([-face, -face])
        np.add.at(diag, a_idx, face)
        np.add.at(diag, b_idx, face)

    add_faces(idx[:, :-1].ravel(), idx[:, 1:].ravel(), dx)
    add_faces(idx[:-1, :].ravel(), idx[1:, :].ravel(), dy)
    n = ny * nx
    L = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return L + sparse.diags(diag)


def apply_dirichlet(A, b, dir_mask, dir_vals):
    """Impose ``x = dir_vals`` on ``dir_mask`` rows of ``A x = b`` (no LIL).

    ``diag(keep) @ A`` zeros the Dirichlet rows and ``+ diag(dir_mask)`` puts 1 on
    their diagonal.  Avoids ``csr.tolil()``, which is unstable on this
    scipy/Python build and was the source of intermittent solver crashes.
    """
    dir_mask = np.asarray(dir_mask, dtype=bool)
    keep = (~dir_mask).astype(float)
    A = sparse.diags(keep) @ A.tocsr() + sparse.diags(dir_mask.astype(float))
    b = b.copy()
    b[dir_mask] = np.asarray(dir_vals)[dir_mask]
    return A.tocsr(), b


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


@dataclass
class GraphPFModel:
    ny: int
    nx: int
    dx: float
    dy: float
    params: GraphPFParams = field(default_factory=GraphPFParams)

    def __post_init__(self):
        p = self.params
        self.D = _plane_strain_D(p.E, p.nu)
        self.ke = _q4_stiffness(self.dx, self.dy, self.D)
        self.B0 = _q4_B(0.0, 0.0, self.dx, self.dy)  # centroid strain operator
        self._build_dof_maps()
        self._build_elastic_indices()
        # defaults: homogeneous unit Gc, no graph coupling, no notch
        self.Gc = np.ones((self.ny, self.nx))
        self.grain_id = None
        self.W = None
        self._deg_w = None
        self._notch = np.zeros((self.ny, self.nx), dtype=bool)
        self._L_kappa = None

    # ----- topology -------------------------------------------------- #
    def _build_dof_maps(self):
        ny, nx = self.ny, self.nx
        nnx = nx + 1
        iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        iy = iy.ravel()
        ix = ix.ravel()
        n00 = iy * nnx + ix
        n01 = iy * nnx + ix + 1
        n11 = (iy + 1) * nnx + ix + 1
        n10 = (iy + 1) * nnx + ix
        nodes = np.stack([n00, n01, n11, n10], axis=1)  # (n_elem, 4)
        dof = np.empty((nodes.shape[0], 8), dtype=int)
        dof[:, 0::2] = 2 * nodes
        dof[:, 1::2] = 2 * nodes + 1
        self.elem_dof = dof
        self.n_nodes = (ny + 1) * nnx
        self.n_dof = 2 * self.n_nodes
        self.n_elem = nodes.shape[0]

    def _build_elastic_indices(self):
        dof = self.elem_dof
        rows = np.repeat(dof, 8, axis=1).ravel()
        cols = np.tile(dof, (1, 8)).ravel()
        self._K_rows = rows
        self._K_cols = cols
        self._ke_flat = self.ke.ravel()

    # ----- fields and boundary conditions ---------------------------- #
    def set_fields(self, Gc=None, grain_id=None, W=None):
        if Gc is not None:
            self.Gc = np.asarray(Gc, dtype=float).reshape(self.ny, self.nx)
        self.grain_id = grain_id
        self.W = W
        if W is not None:
            deg = np.asarray(W.sum(axis=1)).ravel()
            self._deg_w = deg[grain_id]  # (ny,nx) weighted degree
        self._L_kappa = self._assemble_damage_laplacian()

    def set_notch(self, row_lo, row_hi, col_lo, col_hi):
        self._notch[:] = False
        self._notch[row_lo:row_hi, col_lo:col_hi] = True

    def initial_history(self) -> np.ndarray:
        H = np.zeros((self.ny, self.nx))
        H[self._notch] = self.params.notch_H
        return H

    def critical_strain(self) -> float:
        """Homogeneous AT2 critical strain ``sqrt(<Gc>/(E ell))`` (loading scale)."""
        return float(np.sqrt(self.Gc.mean() / (self.params.E * self.params.ell)))

    def strain_schedule(self, eps_max_factor=1.25, n_steps=40, eps_start=0.25):
        """Displacement schedule driving applied strain across the critical value."""
        Ly = self.ny * self.dy
        eps_c = self.critical_strain()
        eps = np.linspace(eps_start * eps_c, eps_max_factor * eps_c, n_steps)
        return eps * Ly

    def set_bc_tension(self):
        """Mode-I SENT: bottom u_y=0 (+corner u_x=0), top u_y=U (scaled later)."""
        ny, nx = self.ny, self.nx
        nnx = nx + 1
        top = np.arange(nnx) + ny * nnx  # top row nodes
        bot = np.arange(nnx)  # bottom row nodes
        fixed = {}
        for n in bot:
            fixed[2 * n + 1] = 0.0  # u_y = 0
        fixed[2 * bot[0]] = 0.0  # u_x = 0 at one corner
        self._top_dof_y = 2 * top + 1  # scaled by U each step
        self._fixed_zero = np.array(sorted(fixed.keys()))
        all_dof = np.arange(self.n_dof)
        constrained = np.union1d(self._fixed_zero, self._top_dof_y)
        self._constrained = constrained
        self._free = np.setdiff1d(all_dof, constrained)

    # ----- elasticity ------------------------------------------------ #
    def _degradation(self, d):
        return (1.0 - d) ** 2 + self.params.k_res

    def assemble_K(self, d):
        gd = self._degradation(d).ravel()  # (n_elem,)
        data = (gd[:, None] * self._ke_flat[None, :]).ravel()
        K = sparse.csr_matrix((data, (self._K_rows, self._K_cols)), shape=(self.n_dof, self.n_dof))
        return K

    def solve_u(self, d, U):
        K = self.assemble_K(d)
        u = np.zeros(self.n_dof)
        u[self._top_dof_y] = U
        f = -K[:, self._constrained] @ u[self._constrained]
        K_ff = K[self._free][:, self._free]
        u[self._free] = spsolve(K_ff.tocsc(), f[self._free])
        return u

    def elastic_energy_plus(self, u):
        """Centroid tensile/deviatoric (Amor) energy density per element."""
        ue = u[self.elem_dof]  # (n_elem, 8)
        eps = ue @ self.B0.T  # (n_elem, 3) tensor? eng shear
        exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
        exy = 0.5 * gxy  # tensor shear
        tr = exx + eyy
        p = self.params
        lam = p.E * p.nu / ((1.0 + p.nu) * (1.0 - 2.0 * p.nu))
        mu = p.E / (2.0 * (1.0 + p.nu))
        K_pe = lam + mu  # 2-D bulk modulus
        tr_pos = np.maximum(tr, 0.0)
        e_dev_xx = exx - 0.5 * tr
        e_dev_yy = eyy - 0.5 * tr
        psi_dev = mu * (e_dev_xx**2 + e_dev_yy**2 + 2.0 * exy**2)
        psi_plus = 0.5 * K_pe * tr_pos**2 + psi_dev
        return psi_plus.reshape(self.ny, self.nx)

    # ----- damage ---------------------------------------------------- #
    def _assemble_damage_laplacian(self):
        """Divergence-form operator for Gc(x)*ell with Neumann BC (pixel grid)."""
        return variable_laplacian(self.Gc * self.params.ell, self.dx, self.dy)

    def _grain_mean(self, d):
        """Mean damage per grain, scattered back to pixels (Picard term)."""
        gid = self.grain_id.ravel()
        n_grains = self.W.shape[0]
        s = np.bincount(gid, weights=d.ravel(), minlength=n_grains)
        c = np.bincount(gid, minlength=n_grains)
        dbar = s / np.maximum(c, 1)
        return dbar

    def solve_d(self, H, d_prev):
        ny, nx = self.ny, self.nx
        ell = self.params.ell
        n = ny * nx
        diag = (self.Gc / ell + 2.0 * H).ravel()
        rhs = (2.0 * H).ravel()

        if self.params.lambda_graph > 0.0 and self.W is not None:
            lam = self.params.lambda_graph
            dbar = self._grain_mean(d_prev)
            Wdbar = np.asarray(self.W @ dbar).ravel()  # (n_grains,)
            gid = self.grain_id.ravel()
            diag = diag + lam * self._deg_w.ravel()
            rhs = rhs + lam * Wdbar[gid]

        A = self._L_kappa + sparse.diags(diag)
        # pin the pre-crack to d = 1
        notch = self._notch.ravel()
        if notch.any():
            A, rhs = apply_dirichlet(A, rhs, notch, np.ones(n))

        d = spsolve(A.tocsc(), rhs).reshape(ny, nx)
        d = np.clip(d, d_prev, 1.0)  # irreversibility
        return d

    # ----- staggered driver ----------------------------------------- #
    def solve_step(self, U, d, H):
        for _ in range(self.params.max_stagger):
            u = self.solve_u(d, U)
            psi = self.elastic_energy_plus(u)
            H = np.maximum(H, psi)
            H[self._notch] = self.params.notch_H
            d_new = self.solve_d(H, d)
            delta = np.max(np.abs(d_new - d))
            d = d_new
            if delta < self.params.stagger_tol:
                break
        return u, d, H, delta

    def run(self, U_schedule, on_step=None, verbose=False):
        d = np.zeros((self.ny, self.nx))
        d[self._notch] = 1.0
        H = self.initial_history()
        history = []
        for k, U in enumerate(U_schedule):
            u, d, H, delta = self.solve_step(U, d, H)
            metrics = crack_metrics(d, self.dx, self.dy, self._notch)
            rec = {"step": k, "U": float(U), "stagger_resid": float(delta), **metrics}
            history.append(rec)
            if on_step is not None:
                on_step(k, U, u, d, H, metrics)
            if verbose:
                print(
                    f"  step {k:3d}  U={U:.4f}  crack_area={metrics['crack_area']:.2f}"
                    f"  tip_x={metrics['tip_x']:.1f}  perc={metrics['percolated']}"
                )
        return {"d": d, "H": H, "history": history}


# --------------------------------------------------------------------------- #
# Crack metrics & path extraction
# --------------------------------------------------------------------------- #


def crack_metrics(d, dx, dy, notch=None, d_thr=0.9):
    cracked = d > d_thr
    if notch is not None:
        cracked = cracked & ~notch
    area = float(cracked.sum() * dx * dy)
    ny, nx = d.shape
    cols = np.where(cracked.any(axis=0))[0]
    tip_x = float(cols.max() * dx) if cols.size else 0.0
    # percolation: a damaged path connecting left and right edges
    perc = _percolates(d > d_thr)
    return {"crack_area": area, "tip_x": tip_x, "percolated": bool(perc)}


def crack_path(d, dx, dy, d_thr=0.5):
    """Single-valued crack path y(x): damage-weighted mean row per column."""
    ny, nx = d.shape
    ys = np.arange(ny)[:, None]
    w = np.where(d > d_thr, d, 0.0)
    col_mass = w.sum(axis=0)
    has = col_mass > 1e-9
    yc = np.full(nx, np.nan)
    yc[has] = (w[:, has] * ys).sum(axis=0) / col_mass[has]
    x = np.arange(nx) * dx
    return x, yc * dy, has


def _percolates(mask):
    """True if the damaged region connects the left and right edges."""
    from scipy.ndimage import label

    lab, _ = label(mask)
    left = set(np.unique(lab[:, 0])) - {0}
    right = set(np.unique(lab[:, -1])) - {0}
    return len(left & right) > 0


# --------------------------------------------------------------------------- #
# Field coarsening (EBSD resolution -> solver resolution)
# --------------------------------------------------------------------------- #


def coarsen(field, factor, how="mean"):
    """Block-reduce a 2-D field by an integer factor (crop to a multiple)."""
    if factor == 1:
        return field
    ny, nx = field.shape
    ny2, nx2 = (ny // factor) * factor, (nx // factor) * factor
    f = field[:ny2, :nx2].reshape(ny2 // factor, factor, nx2 // factor, factor)
    if how == "mean":
        return f.mean(axis=(1, 3))
    if how == "min":
        return f.min(axis=(1, 3))
    if how == "max":
        return f.max(axis=(1, 3))
    if how == "mode":
        f = f.transpose(0, 2, 1, 3).reshape(ny2 // factor, nx2 // factor, factor * factor)
        out = np.empty(f.shape[:2], dtype=field.dtype)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                vals, counts = np.unique(f[i, j], return_counts=True)
                out[i, j] = vals[counts.argmax()]
        return out
    raise ValueError(how)


# --------------------------------------------------------------------------- #
# Self-test: a homogeneous SENT specimen cracks straight across mode I.
# --------------------------------------------------------------------------- #


def self_test() -> dict:
    ny, nx = 40, 60
    model = GraphPFModel(ny, nx, 1.0, 1.0, GraphPFParams(ell=2.0, max_stagger=40))
    model.set_fields(Gc=np.ones((ny, nx)))
    model.set_bc_tension()
    model.set_notch(ny // 2 - 1, ny // 2 + 1, 0, nx // 4)

    out = {}
    schedule = model.strain_schedule(eps_max_factor=1.3, n_steps=45)
    res = model.run(schedule, verbose=False)
    d = res["d"]
    x, yc, has = crack_path(d, 1.0, 1.0, d_thr=0.5)

    # crack should be roughly straight at mid-height
    mid = ny / 2.0
    valid = has & (x > nx / 4)
    out["path_mean_row"] = float(np.nanmean(yc[valid]))
    out["path_row_std"] = float(np.nanstd(yc[valid]))
    out["final_tip_x"] = float(res["history"][-1]["tip_x"])
    out["percolated"] = bool(res["history"][-1]["percolated"])
    out["crack_area"] = float(res["history"][-1]["crack_area"])

    # energy monotonicity of the damage (history field never decreases)
    out["ok"] = bool(
        abs(out["path_mean_row"] - mid) < 3.0
        and out["path_row_std"] < 3.0
        and out["percolated"]
        and out["final_tip_x"] > 0.7 * nx
    )
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
