"""Cubic crystallography for the seventh-paper microstructure-graph layer.

Everything the graph builder needs to turn raw EBSD Euler angles into the
quantities that decorate the microstructure graph: orientations as quaternions
and matrices, the cubic (m-3m) rotational point group, vectorised
*disorientation* angles between neighbouring measurements, and Schmid factors
for the dominant b.c.c. slip families.

Conventions
-----------
* Euler angles are **Bunge** ``(phi1, Phi, phi2)`` in **degrees** (the CTF/HKL
  convention), describing the passive rotation that carries *sample* axes onto
  *crystal* axes, ``v_crystal = g . v_sample`` with
  ``g = Rz(phi2) . Rx(Phi) . Rz(phi1)``.
* Quaternions are stored as ``(w, x, y, z)`` with ``w`` the scalar part and are
  kept on the unit sphere with a non-negative scalar part (``q ~ -q``).

The one non-obvious result used below: for two crystals of the *same* point
group the disorientation angle (minimum over the full two-sided symmetry
double-coset **and** the grain-switching ambiguity) collapses to a *one-sided*
minimum over the 24 rotation operators, because the group is closed under
products and inverses and ``angle(PQ) == angle(QP)``.  That turns the per-pixel
disorientation into a single ``(24, N)`` matrix product -- fast enough to run on
every measurement pair in the map.

Pure ``numpy``; no external crystallography dependency.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Euler <-> rotation matrix / quaternion (Bunge ZXZ, degrees)
# --------------------------------------------------------------------------- #


def euler_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """Bunge Euler angles (degrees) -> orientation matrices ``g`` (sample->crystal).

    Accepts a single ``(3,)`` triple or an ``(N, 3)`` array and returns ``(3, 3)``
    or ``(N, 3, 3)`` correspondingly.
    """
    e = np.atleast_2d(np.asarray(euler_deg, dtype=float))
    phi1, Phi, phi2 = np.radians(e[:, 0]), np.radians(e[:, 1]), np.radians(e[:, 2])
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    g = np.empty((e.shape[0], 3, 3))
    g[:, 0, 0] = c1 * c2 - s1 * s2 * c
    g[:, 0, 1] = s1 * c2 + c1 * s2 * c
    g[:, 0, 2] = s2 * s
    g[:, 1, 0] = -c1 * s2 - s1 * c2 * c
    g[:, 1, 1] = -s1 * s2 + c1 * c2 * c
    g[:, 1, 2] = c2 * s
    g[:, 2, 0] = s1 * s
    g[:, 2, 1] = -c1 * s
    g[:, 2, 2] = c
    return g[0] if np.ndim(euler_deg) == 1 else g


def euler_to_quaternion(euler_deg: np.ndarray) -> np.ndarray:
    """Bunge Euler angles (degrees) -> unit quaternions ``(w, x, y, z)``.

    Vectorised; returns ``(4,)`` for a single triple, ``(N, 4)`` otherwise.
    """
    e = np.atleast_2d(np.asarray(euler_deg, dtype=float))
    phi1, Phi, phi2 = np.radians(e[:, 0]), np.radians(e[:, 1]), np.radians(e[:, 2])
    sigma = 0.5 * (phi1 + phi2)
    delta = 0.5 * (phi1 - phi2)
    cP, sP = np.cos(0.5 * Phi), np.sin(0.5 * Phi)

    q = np.empty((e.shape[0], 4))
    q[:, 0] = cP * np.cos(sigma)
    q[:, 1] = sP * np.cos(delta)
    q[:, 2] = sP * np.sin(delta)
    q[:, 3] = cP * np.sin(sigma)
    q = _normalize_quat(q)
    return q[0] if np.ndim(euler_deg) == 1 else q


def axis_angle_to_quaternion(axis, angle_deg: float) -> np.ndarray:
    """Unit quaternion for a rotation of ``angle_deg`` about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    h = np.radians(angle_deg) / 2.0
    return _normalize_quat(np.array([np.cos(h), *(np.sin(h) * axis)]))


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(q)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    # canonical sign: non-negative scalar part
    flip = q[:, 0] < 0
    q[flip] *= -1.0
    return q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a (x) b`` (broadcasting over leading axis)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty(np.broadcast(aw, bw).shape + (4,))
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


# --------------------------------------------------------------------------- #
# Cubic (m-3m) proper rotation group: 24 operators
# --------------------------------------------------------------------------- #


def _cubic_symmetry_quaternions() -> np.ndarray:
    """The 24 proper rotations of the cubic point group as quaternions."""
    ops = [axis_angle_to_quaternion([0, 0, 1], 0.0)]  # identity
    # 90/180/270 about <100>
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        for ang in (90.0, 180.0, 270.0):
            ops.append(axis_angle_to_quaternion(axis, ang))
    # 180 about <110>
    for axis in ([1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]):
        ops.append(axis_angle_to_quaternion(axis, 180.0))
    # 120/240 about <111>
    for axis in ([1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]):
        for ang in (120.0, 240.0):
            ops.append(axis_angle_to_quaternion(axis, ang))
    q = np.vstack(ops)
    assert q.shape == (24, 4), q.shape
    return q


CUBIC_SYMMETRY = _cubic_symmetry_quaternions()


def disorientation_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Cubic disorientation angle (degrees) between orientations ``qa`` and ``qb``.

    ``qa``/``qb`` are ``(N, 4)`` (or ``(4,)``) unit quaternions.  Returns the
    per-pair disorientation angle in degrees, i.e. the minimum rotation angle
    over the cubic point group and the grain-switching ambiguity.
    """
    qa = np.atleast_2d(qa)
    qb = np.atleast_2d(qb)
    # misorientation m = qb (x) conj(qa)
    qa_conj = qa.copy()
    qa_conj[:, 1:] *= -1.0
    m = quat_multiply(qb, qa_conj)  # (N, 4)

    # For each symmetry s, scalar part of (s (x) m):
    #   (s (x) m)_w = s0 m0 - s1 m1 - s2 m2 - s3 m3
    # so with m' = (m0, -m1, -m2, -m3),  W = S @ m'^T  is (24, N).
    m_mod = m.copy()
    m_mod[:, 1:] *= -1.0
    w = CUBIC_SYMMETRY @ m_mod.T  # (24, N)
    max_w = np.clip(np.max(np.abs(w), axis=0), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(max_w))


# --------------------------------------------------------------------------- #
# Schmid factors for b.c.c. slip families
# --------------------------------------------------------------------------- #


def _bcc_slip_systems() -> tuple[np.ndarray, np.ndarray]:
    """Return unit (plane-normal, slip-direction) pairs for {110}<111>+{112}<111>."""
    burgers = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=float)
    planes_110 = np.array(
        [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]],
        dtype=float,
    )
    planes_112 = np.array(
        [
            [1, 1, 2],
            [1, 1, -2],
            [1, -1, 2],
            [1, -1, -2],
            [1, 2, 1],
            [1, 2, -1],
            [1, -2, 1],
            [1, -2, -1],
            [2, 1, 1],
            [2, 1, -1],
            [2, -1, 1],
            [2, -1, -1],
        ],
        dtype=float,
    )
    n_list, b_list = [], []
    for planes in (planes_110, planes_112):
        for n in planes:
            for b in burgers:
                if abs(np.dot(n, b)) < 1e-9:  # slip direction must lie in plane
                    n_list.append(n / np.linalg.norm(n))
                    b_list.append(b / np.linalg.norm(b))
    return np.array(n_list), np.array(b_list)


BCC_PLANES, BCC_DIRS = _bcc_slip_systems()


def schmid_factor(euler_deg: np.ndarray, load_axis=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Maximum Schmid factor over b.c.c. slip systems for uniaxial ``load_axis``.

    ``load_axis`` is given in the *sample* frame (default: tension along +y, the
    loading direction used by the phase-field case studies).  Returns a scalar
    for a single Euler triple, otherwise an ``(N,)`` array.
    """
    g = euler_to_matrix(euler_deg)  # sample->crystal
    g = g[None] if g.ndim == 2 else g
    load = np.asarray(load_axis, dtype=float)
    load = load / np.linalg.norm(load)
    load_c = np.einsum("nij,j->ni", g, load)  # load axis in crystal frame, (N,3)

    cos_lambda = load_c @ BCC_DIRS.T  # (N, S)
    cos_phi = load_c @ BCC_PLANES.T  # (N, S)
    m = np.abs(cos_lambda * cos_phi)
    out = np.max(m, axis=1)
    return out[0] if np.ndim(euler_deg) == 1 else out


# --------------------------------------------------------------------------- #
# Inverse pole figure (IPF) colouring
# --------------------------------------------------------------------------- #


def ipf_rgb(euler_deg: np.ndarray, sample_dir=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Standard cubic IPF colours for ``sample_dir`` (red-001 / green-011 / blue-111).

    For the full cubic Laue group m-3m the fundamental sector reduction is just
    ``sort(abs(.))`` of the crystal-frame direction, because the group acts on
    directions as all signed axis permutations.  Returns ``(N, 3)`` (or ``(3,)``)
    RGB in ``[0, 1]``.
    """
    g = euler_to_matrix(euler_deg)
    g = g[None] if g.ndim == 2 else g
    sd = np.asarray(sample_dir, dtype=float)
    sd = sd / np.linalg.norm(sd)
    t = np.einsum("nij,j->ni", g, sd)  # direction in crystal frame
    a = np.sort(np.abs(t), axis=1)  # 0 <= a0 <= a1 <= a2 (SST)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    r = a[:, 2] - a[:, 1]
    gr = a[:, 1] - a[:, 0]
    b = a[:, 0]
    rgb = np.stack([r, gr, b], axis=1)
    rgb /= np.maximum(rgb.max(axis=1, keepdims=True), 1e-12)
    rgb = np.sqrt(np.clip(rgb, 0.0, 1.0))  # perceptual brightening
    return rgb[0] if np.ndim(euler_deg) == 1 else rgb


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def self_test() -> dict:
    """Sanity gates for the crystallography primitives."""
    out: dict = {}

    # 1. Orientation matrices are proper rotations.
    rng = np.random.default_rng(0)
    e = rng.uniform([0, 0, 0], [360, 180, 360], size=(50, 3))
    g = euler_to_matrix(e)
    det_err = float(np.max(np.abs(np.linalg.det(g) - 1.0)))
    orth_err = float(np.max(np.abs(np.einsum("nij,nkj->nik", g, g) - np.eye(3))))
    out["matrix_det_err"] = det_err
    out["matrix_orth_err"] = orth_err

    # 2. Identity disorientation is zero.
    q = euler_to_quaternion(e)
    out["self_disorientation_max_deg"] = float(np.max(disorientation_angle(q, q)))

    # 3. A 90 deg rotation about <001> is a cubic symmetry -> disorientation 0.
    q0 = euler_to_quaternion(np.array([0.0, 0.0, 0.0]))
    q90 = euler_to_quaternion(np.array([90.0, 0.0, 0.0]))
    out["cubic_90deg_disorientation"] = float(disorientation_angle(q0, q90)[0])

    # 4. A 45 deg rotation about <001> -> disorientation 45 deg (below the
    #    cubic maximum of ~62.8 deg, so it survives symmetry reduction).
    q45 = euler_to_quaternion(np.array([45.0, 0.0, 0.0]))
    out["disorientation_45deg"] = float(disorientation_angle(q0, q45)[0])

    # 5. Sigma3 (60 deg about <111>) is the K-S / twin disorientation.
    q60_111 = quat_multiply(axis_angle_to_quaternion([1, 1, 1], 60.0), q0)[0]
    out["disorientation_sigma3"] = float(disorientation_angle(q0, q60_111)[0])

    # 6. Schmid factor for <001> b.c.c. tension: 0.408 on {110}<111>, but 0.471
    #    once {112}<111> is included (martensite slips on both families).
    out["schmid_001"] = float(schmid_factor(np.array([0.0, 0.0, 0.0]), (0, 0, 1)))
    out["schmid_max_theoretical_ok"] = bool(0.0 < out["schmid_001"] <= 0.5 + 1e-9)

    # 7. IPF colours: [001]->red, [011]->green, [111]->blue for the cube axis.
    c001 = ipf_rgb(np.array([0.0, 0.0, 0.0]), (0, 0, 1))
    c011 = ipf_rgb(np.array([0.0, 45.0, 0.0]), (0, 0, 1))  # Z -> <011> in crystal
    c111 = ipf_rgb(np.array([0.0, 54.7356, 45.0]), (0, 0, 1))  # Z -> <111>
    out["ipf_001"] = c001.round(3).tolist()
    out["ipf_011"] = c011.round(3).tolist()
    out["ipf_111"] = c111.round(3).tolist()
    ipf_ok = c001[0] > 0.9 and c011[1] > 0.9 and c111[2] > 0.9

    out["ok"] = bool(
        det_err < 1e-10
        and orth_err < 1e-10
        and out["self_disorientation_max_deg"] < 1e-6
        and out["cubic_90deg_disorientation"] < 1e-6
        and abs(out["disorientation_45deg"] - 45.0) < 1e-6
        and abs(out["disorientation_sigma3"] - 60.0) < 1.0
        and abs(out["schmid_001"] - 0.4714) < 1e-3
        and ipf_ok
    )
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
