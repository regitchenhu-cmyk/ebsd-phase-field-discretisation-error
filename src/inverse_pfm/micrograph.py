"""Microstructure-as-graph layer: EBSD map -> attributed grain graph.

This is the representational core of the seventh paper.  It turns a measured
EBSD map into the attributed graph ``G = (V, E)`` of the proposal:

* **Nodes** ``V`` are reconstructed grains, decorated with
  ``X_v = [orientation, equivalent diameter, area, phase, max Schmid factor,
  KAM]`` (KAM = kernel average misorientation, a geometrically-necessary
  dislocation / trap-density proxy).
* **Edges** ``E`` are grain boundaries, decorated with
  ``X_e = [misorientation theta, boundary length, boundary type, Read-Shockley
  energy gamma_gb, hydrogen diffusivity D_H, trap density rho_trap, fracture
  energy Gc]``.

The boundary attributes are the explicit maps ``F_G, F_D, F_alpha`` from the
proposal that push graph topology into local phase-field parameters; the
:meth:`GrainGraph.rasterize_fields` method realises them as ``(ny, nx)`` fields
``Gc(x), D_H(x), rho_trap(x)`` that the continuum phase-field solver consumes.

Grain reconstruction follows the EBSD-export recipe (critical misorientation
10 deg, minimum grain size 3 px): a union-find flood fill over same-phase,
sub-threshold neighbour pairs, small-grain absorption, then a nearest-grain fill
of the non-indexed points so the field is space-filling for the solver.

Graph algorithms (connectivity, shortest path, Laplacian) are taken from
``scipy.sparse.csgraph`` -- no networkx dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from . import crystallography as xtal
from .ebsd import EBSDMap

# --------------------------------------------------------------------------- #
# Parameters of the graph -> phase-field attribute maps
# --------------------------------------------------------------------------- #


@dataclass
class GraphModelParams:
    """Tunable parameters for grain reconstruction and the attribute maps."""

    # --- reconstruction --------------------------------------------------- #
    theta_seg: float = 10.0  # grain-segmentation misorientation (deg)
    min_grain_px: int = 3  # absorb grains smaller than this
    kam_cap: float = 5.0  # KAM cap per neighbour pair (deg)

    # --- boundary classification ----------------------------------------- #
    theta_hagb: float = 15.0  # LAGB / HAGB cut (deg)
    sigma3_angle: float = 60.0  # Sigma3 / K-S special-boundary angle (deg)
    sigma3_tol: float = 8.66  # Brandon criterion for Sigma3 (deg)

    # --- Read-Shockley grain-boundary energy (normalised to HAGB = 1) ----- #
    theta_rs: float = 15.0  # Read-Shockley saturation angle (deg)
    sigma3_energy_factor: float = 0.30  # coherent-twin energy relief

    # --- attribute -> phase-field maps ----------------------------------- #
    beta_DH: float = 12.0  # max GB/bulk H-diffusivity ratio (HAGB)
    beta_trap: float = 8.0  # max GB/bulk trap-density ratio (HAGB)
    kam_trap_gain: float = 6.0  # trap density gain per unit normalised KAM
    kgb_Gc: float = 0.55  # max GB toughness reduction fraction (HAGB)
    sigma3_Gc_bonus: float = 0.25  # toughness restored on special boundaries


# --------------------------------------------------------------------------- #
# Grain graph container
# --------------------------------------------------------------------------- #


@dataclass
class GrainGraph:
    """An attributed grain graph reconstructed from an EBSD map."""

    name: str
    shape: tuple
    step: tuple  # (dx, dy) microns
    grain_id: np.ndarray  # (ny, nx) int label per pixel, 0..G-1
    n_grains: int

    # node attributes (length n_grains)
    centroid: np.ndarray  # (G, 2) (x, y) microns
    area_um2: np.ndarray  # (G,)
    equiv_diam: np.ndarray  # (G,)
    node_phase: np.ndarray  # (G,) majority phase id
    node_quat: np.ndarray  # (G, 4) representative orientation
    schmid: np.ndarray  # (G,) max Schmid factor (mean orientation)
    node_kam: np.ndarray  # (G,) mean KAM (deg)

    # edge attributes (length E), edges are undirected i<j
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_theta: np.ndarray  # misorientation (deg)
    edge_len: np.ndarray  # boundary length (microns)
    edge_type: np.ndarray  # 0=LAGB, 1=HAGB, 2=special(Sigma3)
    edge_gamma: np.ndarray  # Read-Shockley energy (normalised)
    edge_DH: np.ndarray  # GB/bulk H-diffusivity ratio
    edge_trap: np.ndarray  # GB/bulk trap-density ratio
    edge_Gc: np.ndarray  # GB/bulk fracture-energy ratio

    # pixel-level fields
    kam_field: np.ndarray  # (ny, nx) KAM (deg)
    params: GraphModelParams = field(default_factory=GraphModelParams)

    # ------------------------------------------------------------------ #
    @property
    def n_edges(self) -> int:
        return int(self.edge_i.size)

    def adjacency(self, weight: np.ndarray | None = None) -> sparse.csr_matrix:
        """Symmetric sparse adjacency; ``weight`` defaults to boundary length."""
        if weight is None:
            weight = self.edge_len
        i = np.concatenate([self.edge_i, self.edge_j])
        j = np.concatenate([self.edge_j, self.edge_i])
        w = np.concatenate([weight, weight])
        return sparse.csr_matrix((w, (i, j)), shape=(self.n_grains, self.n_grains))

    def laplacian(self, weight: np.ndarray | None = None) -> sparse.csr_matrix:
        """Graph Laplacian ``L = D - W`` for the chosen edge weight."""
        return sparse.csgraph.laplacian(self.adjacency(weight))

    def node_field_means(self, field: np.ndarray) -> np.ndarray:
        """Mean of a pixel ``field`` over each grain (length ``n_grains``)."""
        gid = self.grain_id.ravel()
        f = np.asarray(field).ravel()
        s = np.bincount(gid, weights=f, minlength=self.n_grains)
        c = np.bincount(gid, minlength=self.n_grains)
        return s / np.maximum(c, 1)

    def edge_field_means(self, field: np.ndarray) -> np.ndarray:
        """Mean of a pixel ``field`` over each edge's boundary segments.

        Returned in the same order as ``edge_i`` / ``edge_j``; edges whose
        boundary is not sampled fall back to the mean of their two grains.
        """
        f = np.asarray(field)
        gid = self.grain_id
        key_lut = {
            (int(a), int(b)): k
            for k, (a, b) in enumerate(zip(self.edge_i, self.edge_j, strict=True))
        }
        acc = np.zeros(self.n_edges)
        cnt = np.zeros(self.n_edges)
        for sl_a, sl_b in (
            ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
            ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
        ):
            ga, gb = gid[sl_a], gid[sl_b]
            diff = ga != gb
            a = np.minimum(ga[diff], gb[diff])
            b = np.maximum(ga[diff], gb[diff])
            # value on the boundary = mean of the two adjacent pixels
            fv = 0.5 * (f[sl_a][diff] + f[sl_b][diff])
            for aa, bb, vv in zip(a.tolist(), b.tolist(), fv.tolist(), strict=True):
                k = key_lut[(aa, bb)]
                acc[k] += vv
                cnt[k] += 1.0
        out = acc / np.maximum(cnt, 1)
        if np.any(cnt == 0):  # fall back to the two-grain mean for unsampled edges
            nf = self.node_field_means(field)
            fb = 0.5 * (nf[self.edge_i] + nf[self.edge_j])
            out = np.where(cnt > 0, out, fb)
        return out

    # ------------------------------------------------------------------ #
    def rasterize_fields(
        self,
        gc_bulk: float,
        dh_bulk: float,
        trap_bulk: float,
        grain_id: np.ndarray | None = None,
        kam_field: np.ndarray | None = None,
    ) -> dict:
        """Realise the graph attribute maps as ``(ny, nx)`` pixel fields.

        Returns a dict with ``Gc``, ``D_H``, ``rho_trap`` and the boolean
        ``boundary`` mask.  Boundary pixels take the *weakest* adjacent edge for
        toughness and the *fastest / most-trapping* adjacent edge for hydrogen,
        so a crack and the hydrogen network both see the easy paths first.

        Pass a coarsened ``grain_id`` (and matching ``kam_field``) to rasterise
        directly on a solver-resolution grain map -- this keeps the boundary
        network one cell wide instead of saturating under block-pooling.
        """
        gid = self.grain_id if grain_id is None else np.asarray(grain_id)
        ny, nx = gid.shape
        kam = self.kam_field if kam_field is None else np.asarray(kam_field)
        boundary = np.zeros((ny, nx), dtype=bool)
        gc = np.full((ny, nx), gc_bulk)
        dh = np.full((ny, nx), dh_bulk)
        # bulk trap density already rises with stored dislocation content (KAM)
        kam_n = np.clip(kam / self.params.kam_cap, 0.0, 1.0)
        trap = trap_bulk * (1.0 + self.params.kam_trap_gain * kam_n)

        # edge attribute lookup keyed by ordered grain pair
        pair_key = self.edge_i.astype(np.int64) * self.n_grains + self.edge_j
        order = np.argsort(pair_key)
        pair_key = pair_key[order]
        e_gc = self.edge_Gc[order]
        e_dh = self.edge_DH[order]
        e_trap = self.edge_trap[order]

        def lookup(a, b):
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            key = lo.astype(np.int64) * self.n_grains + hi
            idx = np.searchsorted(pair_key, key)
            return idx

        # accumulate per-pixel weakest/fastest adjacent boundary
        for _axis, sl_a, sl_b in (
            (1, (slice(None), slice(0, -1)), (slice(None), slice(1, None))),
            (0, (slice(0, -1), slice(None)), (slice(1, None), slice(None))),
        ):
            ga = gid[sl_a]
            gb = gid[sl_b]
            diff = ga != gb
            if not np.any(diff):
                continue
            aa = ga[diff]
            bb = gb[diff]
            idx = lookup(aa, bb)
            valid = idx < pair_key.size
            key = np.minimum(aa, bb).astype(np.int64) * self.n_grains + np.maximum(aa, bb)
            valid &= pair_key[np.clip(idx, 0, pair_key.size - 1)] == key
            idx = idx[valid]

            for sl, mask_full in ((sl_a, diff), (sl_b, diff)):
                ys, xs = np.where(mask_full)
                ys = ys[valid]
                xs = xs[valid]
                yy = ys + (sl[0].start or 0)
                xx = xs + (sl[1].start or 0)
                boundary[yy, xx] = True
                gc[yy, xx] = np.minimum(gc[yy, xx], gc_bulk * e_gc[idx])
                dh[yy, xx] = np.maximum(dh[yy, xx], dh_bulk * e_dh[idx])
                trap[yy, xx] = np.maximum(trap[yy, xx], trap_bulk * e_trap[idx])

        return {"Gc": gc, "D_H": dh, "rho_trap": trap, "boundary": boundary}


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #


def _neighbor_disorientation(quat: np.ndarray):
    """Return horizontal and vertical neighbour disorientation maps (deg)."""
    ny, nx, _ = quat.shape
    qh_a = quat[:, :-1].reshape(-1, 4)
    qh_b = quat[:, 1:].reshape(-1, 4)
    dh = xtal.disorientation_angle(qh_a, qh_b).reshape(ny, nx - 1)
    qv_a = quat[:-1, :].reshape(-1, 4)
    qv_b = quat[1:, :].reshape(-1, 4)
    dv = xtal.disorientation_angle(qv_a, qv_b).reshape(ny - 1, nx)
    return dh, dv


def _kam_field(quat: np.ndarray, indexed: np.ndarray, cap: float) -> np.ndarray:
    """First-neighbour kernel average misorientation (deg), capped per pair."""
    ny, nx, _ = quat.shape
    dh, dv = _neighbor_disorientation(quat)
    acc = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx))

    def add(d, sl_a, sl_b):
        valid = indexed[sl_a] & indexed[sl_b]
        dd = np.minimum(d, cap)
        for sl in (sl_a, sl_b):
            acc[sl] += np.where(valid, dd, 0.0)
            cnt[sl] += valid

    add(dh, (slice(None), slice(0, -1)), (slice(None), slice(1, None)))
    add(dv, (slice(0, -1), slice(None)), (slice(1, None), slice(None)))
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)


def reconstruct_grains(ebsd: EBSDMap, params: GraphModelParams) -> np.ndarray:
    """Flood-fill grains; returns an ``(ny, nx)`` 0-based grain-id map.

    Grains are the connected components of the pixel graph whose edges join
    indexed, same-phase neighbours below the segmentation misorientation
    (``scipy.sparse.csgraph.connected_components``).
    """
    ny, nx = ebsd.shape
    quat = ebsd.quaternions()
    indexed = ebsd.indexed
    phase = ebsd.phase
    dh, dv = _neighbor_disorientation(quat)
    idx = np.arange(ny * nx).reshape(ny, nx)

    def links(d, sl_a, sl_b):
        same = indexed[sl_a] & indexed[sl_b] & (phase[sl_a] == phase[sl_b]) & (d < params.theta_seg)
        return idx[sl_a][same], idx[sl_b][same]

    ai, aj = links(dh, (slice(None), slice(0, -1)), (slice(None), slice(1, None)))
    bi, bj = links(dv, (slice(0, -1), slice(None)), (slice(1, None), slice(None)))
    ei = np.concatenate([ai, bi])
    ej = np.concatenate([aj, bj])
    n = ny * nx
    A = sparse.csr_matrix((np.ones(ei.size), (ei, ej)), shape=(n, n))
    _, labels = sparse.csgraph.connected_components(A, directed=False)
    labels = labels.reshape(ny, nx)

    # keep only indexed pixels, relabel their components contiguously
    roots = np.where(indexed, labels, -1)
    uniq = np.unique(roots[indexed])
    remap = np.full(labels.max() + 1, -1)
    remap[uniq] = np.arange(uniq.size)
    grain = np.where(indexed, remap[labels], -1)

    grain = _absorb_small_grains(grain, params)
    grain = _fill_unindexed(grain)
    # final relabel to contiguous 0..G-1
    uniq = np.unique(grain)
    remap = {r: g for g, r in enumerate(uniq)}
    out = np.zeros_like(grain)
    for r, g in remap.items():
        out[grain == r] = g
    return out


def _grain_mean_quat(quat: np.ndarray, grain: np.ndarray, g: int) -> np.ndarray:
    """Representative orientation = orientation nearest the grain centroid."""
    ys, xs = np.where(grain == g)
    cy, cx = ys.mean(), xs.mean()
    k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    return quat[ys[k], xs[k]]


def _absorb_small_grains(grain, params):
    """Merge grains below ``min_grain_px`` into their dominant neighbour.

    Fully vectorised: each sub-threshold grain is merged into the adjacent grain
    with which it shares the most boundary, preferring an above-threshold
    neighbour; merge chains (small -> small -> large) are then resolved so
    clustered fragments coalesce and grow past the threshold.  Adjacency-based
    (no per-pixel orientation comparison) so it is fast and allocation-light.
    """
    for _ in range(30):
        valid = grain >= 0
        if not valid.any():
            break
        n_lab = int(grain.max()) + 1
        counts = np.bincount(grain[valid], minlength=n_lab)
        is_small = (counts > 0) & (counts < params.min_grain_px)
        if not is_small.any():
            break

        src_list, dst_list = [], []
        for a, b in (
            (grain[:, :-1], grain[:, 1:]),
            (grain[:, 1:], grain[:, :-1]),
            (grain[:-1, :], grain[1:, :]),
            (grain[1:, :], grain[:-1, :]),
        ):
            a = a.ravel()
            b = b.ravel()
            m = (a >= 0) & (b >= 0) & (a != b) & is_small[np.where(a >= 0, a, 0)]
            src_list.append(a[m])
            dst_list.append(b[m])
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        if src.size == 0:
            break

        key = src.astype(np.int64) * n_lab + dst
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        usrc = uk // n_lab
        udst = uk % n_lab
        # prefer an above-threshold target, then the most-shared boundary
        score = cnt + np.where(~is_small[udst], 1e6, 0.0)
        merge = np.arange(n_lab)
        order = np.lexsort((-score, usrc))  # group by src, best score first
        first = np.ones(usrc.size, dtype=bool)
        first[1:] = usrc[order][1:] != usrc[order][:-1]
        chosen = order[first]
        merge[usrc[chosen]] = udst[chosen]
        for _ in range(20):  # resolve small->small->large chains
            nm = merge[merge]
            if np.array_equal(nm, merge):
                break
            merge = nm

        out = grain.copy()
        out[valid] = merge[grain[valid]]
        if np.array_equal(out, grain):
            break
        grain = out
    return grain


def _fill_unindexed(grain: np.ndarray) -> np.ndarray:
    """Dilate labelled grains into non-indexed (-1) pixels by nearest neighbour."""
    ny, nx = grain.shape
    out = grain.copy()
    for _ in range(max(ny, nx)):
        holes = np.argwhere(out < 0)
        if holes.size == 0:
            break
        filled = False
        for y, x in holes:
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < ny and 0 <= xx < nx and out[yy, xx] >= 0:
                    out[y, x] = out[yy, xx]
                    filled = True
                    break
        if not filled:
            break
    out[out < 0] = 0
    return out


# --------------------------------------------------------------------------- #
# Attribute maps F_G / F_D / F_alpha
# --------------------------------------------------------------------------- #


def _read_shockley(theta_deg: np.ndarray, params: GraphModelParams) -> np.ndarray:
    """Normalised Read-Shockley boundary energy (HAGB -> 1)."""
    t = np.clip(theta_deg / params.theta_rs, 1e-6, 1.0)
    low = t * (1.0 - np.log(t))  # rises to 1 at theta_rs
    return np.where(theta_deg < params.theta_rs, low, 1.0)


def _classify_edges(theta: np.ndarray, params: GraphModelParams) -> np.ndarray:
    etype = np.where(theta < params.theta_hagb, 0, 1)  # LAGB / HAGB
    special = np.abs(theta - params.sigma3_angle) < params.sigma3_tol
    etype = np.where(special, 2, etype)
    return etype


def _edge_attribute_maps(theta, etype, params):
    """Map (theta, type) -> (gamma_gb, D_H ratio, trap ratio, Gc ratio)."""
    gamma = _read_shockley(theta, params)
    gamma = np.where(etype == 2, gamma * params.sigma3_energy_factor, gamma)

    # Hydrogen short-circuit diffusion and trapping scale with boundary energy.
    dh = 1.0 + (params.beta_DH - 1.0) * gamma
    trap = 1.0 + (params.beta_trap - 1.0) * gamma

    # Fracture energy drops with boundary energy; special boundaries recover.
    gc = 1.0 - params.kgb_Gc * gamma
    gc = np.where(etype == 2, np.minimum(1.0, gc + params.sigma3_Gc_bonus), gc)
    return gamma, dh, trap, gc


def build_graph(ebsd: EBSDMap, params: GraphModelParams | None = None) -> GrainGraph:
    """Reconstruct grains and build the attributed grain graph."""
    params = params or GraphModelParams()
    ny, nx = ebsd.shape
    dx, dy = ebsd.dx, ebsd.dy
    quat = ebsd.quaternions()
    grain = reconstruct_grains(ebsd, params)
    n_grains = int(grain.max()) + 1

    kam = _kam_field(quat, ebsd.indexed, params.kam_cap)

    # ---- node attributes ---- #
    centroid = np.zeros((n_grains, 2))
    area_um2 = np.zeros(n_grains)
    node_phase = np.zeros(n_grains, dtype=int)
    node_quat = np.zeros((n_grains, 4))
    node_kam = np.zeros(n_grains)
    px_area = dx * dy
    for g in range(n_grains):
        ys, xs = np.where(grain == g)
        centroid[g] = [xs.mean() * dx, ys.mean() * dy]
        area_um2[g] = ys.size * px_area
        ph = ebsd.phase[ys, xs]
        ph = ph[ph > 0]
        node_phase[g] = np.bincount(ph).argmax() if ph.size else 0
        node_quat[g] = _grain_mean_quat(quat, grain, g)
        node_kam[g] = kam[ys, xs].mean()
    equiv_diam = 2.0 * np.sqrt(area_um2 / np.pi)
    eulers = _quat_to_euler(node_quat)
    schmid = xtal.schmid_factor(eulers, load_axis=(0.0, 1.0, 0.0))

    # ---- edges: accumulate over boundary pixel pairs ---- #
    ei, ej, theta = _accumulate_edges(grain, quat, n_grains)
    elen = _edge_lengths(grain, n_grains, ei, ej, dx, dy)
    etype = _classify_edges(theta, params)
    gamma, edh, etrap, egc = _edge_attribute_maps(theta, etype, params)

    return GrainGraph(
        name=ebsd.name,
        shape=(ny, nx),
        step=(dx, dy),
        grain_id=grain,
        n_grains=n_grains,
        centroid=centroid,
        area_um2=area_um2,
        equiv_diam=equiv_diam,
        node_phase=node_phase,
        node_quat=node_quat,
        schmid=schmid,
        node_kam=node_kam,
        edge_i=ei,
        edge_j=ej,
        edge_theta=theta,
        edge_len=elen,
        edge_type=etype,
        edge_gamma=gamma,
        edge_DH=edh,
        edge_trap=etrap,
        edge_Gc=egc,
        kam_field=kam,
        params=params,
    )


def _quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> Bunge Euler angles (deg). Vectorised, (N,3)."""
    q = np.atleast_2d(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # rotation matrix entries needed for the Bunge extraction
    g22 = w * w - x * x - y * y + z * z
    g22 = np.clip(g22, -1.0, 1.0)
    Phi = np.arccos(g22)
    g20 = 2.0 * (x * z + w * y)
    g21 = 2.0 * (y * z - w * x)
    g02 = 2.0 * (x * z - w * y)
    g12 = 2.0 * (y * z + w * x)
    sP = np.sin(Phi)
    near = sP < 1e-7
    phi1 = np.where(
        near,
        np.arctan2(2.0 * (x * y + w * z), w * w + x * x - y * y - z * z),
        np.arctan2(g20, -g21),
    )
    phi2 = np.where(near, 0.0, np.arctan2(g02, g12))
    out = np.degrees(np.vstack([phi1, Phi, phi2]).T) % 360.0
    return out


def _accumulate_edges(grain, quat, n_grains):
    """Collect undirected edges with mean boundary disorientation."""
    dh, dv = _neighbor_disorientation(quat)
    keys = []
    vals = []
    for d, sl_a, sl_b in (
        (dh, (slice(None), slice(0, -1)), (slice(None), slice(1, None))),
        (dv, (slice(0, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        ga = grain[sl_a]
        gb = grain[sl_b]
        diff = ga != gb
        a = np.minimum(ga[diff], gb[diff]).astype(np.int64)
        b = np.maximum(ga[diff], gb[diff]).astype(np.int64)
        keys.append(a * n_grains + b)
        vals.append(d[diff])
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)
    uniq, inv = np.unique(keys, return_inverse=True)
    theta_sum = np.bincount(inv, weights=vals)
    cnt = np.bincount(inv)
    theta = theta_sum / cnt
    ei = (uniq // n_grains).astype(int)
    ej = (uniq % n_grains).astype(int)
    return ei, ej, theta


def _edge_lengths(grain, n_grains, ei, ej, dx, dy):
    """Boundary length per edge (microns) from segment counts."""
    key_index = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ei, ej, strict=True))}
    length = np.zeros(ei.size)
    for seg_len, sl_a, sl_b in (
        (dy, (slice(None), slice(0, -1)), (slice(None), slice(1, None))),
        (dx, (slice(0, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        ga = grain[sl_a]
        gb = grain[sl_b]
        diff = ga != gb
        a = np.minimum(ga[diff], gb[diff])
        b = np.maximum(ga[diff], gb[diff])
        for aa, bb in zip(a.tolist(), b.tolist(), strict=True):
            length[key_index[(aa, bb)]] += seg_len
    return length


# --------------------------------------------------------------------------- #
# Self-test on a synthetic bicrystal
# --------------------------------------------------------------------------- #


def _synthetic_bicrystal(theta_deg: float = 30.0) -> EBSDMap:
    ny, nx = 20, 40
    euler = np.zeros((ny, nx, 3))
    euler[:, nx // 2 :, 0] = theta_deg  # right half rotated about z by theta
    phase = np.full((ny, nx), 2, dtype=int)  # all bcc matrix
    return EBSDMap(
        name="bicrystal",
        nx=nx,
        ny=ny,
        dx=1.0,
        dy=1.0,
        phase=phase,
        euler=euler,
        bc=np.full((ny, nx), 150.0),
        mad=np.zeros((ny, nx)),
        bands=np.full((ny, nx), 8),
        phases=[],
    )


def self_test() -> dict:
    """Reconstruct a 30 deg bicrystal and check the single boundary edge."""
    params = GraphModelParams()
    ebsd = _synthetic_bicrystal(30.0)
    G = build_graph(ebsd, params)

    out = {
        "n_grains": G.n_grains,
        "n_edges": G.n_edges,
        "edge_theta_deg": float(G.edge_theta[0]) if G.n_edges else None,
        "edge_type": int(G.edge_type[0]) if G.n_edges else None,
        "edge_len_um": float(G.edge_len[0]) if G.n_edges else None,
        "edge_Gc_ratio": float(G.edge_Gc[0]) if G.n_edges else None,
        "edge_DH_ratio": float(G.edge_DH[0]) if G.n_edges else None,
    }
    fields = G.rasterize_fields(gc_bulk=1.0, dh_bulk=1.0, trap_bulk=1.0)
    out["raster_Gc_min"] = float(fields["Gc"].min())
    out["raster_DH_max"] = float(fields["D_H"].max())
    out["raster_boundary_px"] = int(fields["boundary"].sum())

    # Laplacian of a 2-node graph has eigenvalues {0, 2w}
    L = G.laplacian(weight=np.ones(G.n_edges)).toarray()
    out["laplacian_eigs"] = sorted(np.linalg.eigvalsh(L).round(6).tolist())

    out["ok"] = bool(
        G.n_grains == 2
        and G.n_edges == 1
        and abs(out["edge_theta_deg"] - 30.0) < 1.0
        and out["edge_type"] == 1
        and abs(out["edge_len_um"] - 20.0) < 1e-9
        and out["raster_Gc_min"] < 1.0
        and out["raster_DH_max"] > 1.0
        and out["laplacian_eigs"][0] < 1e-9
    )
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
