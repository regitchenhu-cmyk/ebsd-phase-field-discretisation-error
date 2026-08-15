"""Weighted-graph crack-path prediction and microstructure path regulation.

A candidate crack path is the minimum-cost path across the grain graph,

    P* = argmin_P  sum_{e in P} w_e ,
    w_e = Gc_e - alpha * cH_e - beta * psi+_e + gamma * R_e ,

where ``Gc_e`` is the (normalised) boundary toughness, ``cH_e`` the boundary
hydrogen occupancy, ``psi+_e`` the boundary mechanical driving force and ``R_e``
a boundary-resistance term (low-energy / special boundaries resist cracking).
The weight is the *cost to crack along an edge*: hydrogen and driving force make
it cheaper, toughness and special-boundary resistance make it dearer.  Weights
are floored positive so Dijkstra (``scipy.sparse.csgraph``) applies.

The path is computed from the grains at the notch to the grains on the far edge
via a virtual super-source / super-sink.  Path *regulation* (case study 3) then
designs a local toughening / de-charging patch that reroutes ``P*`` away from a
protected zone, the discrete analogue of the proposal's control
``u = {Gc(x), D_H(x), rho_trap(x)}``.

No networkx dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .micrograph import GrainGraph


@dataclass
class PathWeightParams:
    alpha: float = 0.6  # hydrogen susceptibility
    beta: float = 0.4  # mechanical driving force
    gamma: float = 0.3  # boundary-resistance penalty
    w_floor: float = 1e-3  # positivity floor for Dijkstra


def edge_weights(
    graph: GrainGraph,
    params: PathWeightParams,
    c_H=None,
    psi=None,
    R=None,
) -> np.ndarray:
    """Per-edge crack-path cost ``w_e`` (length ``n_edges``, strictly positive)."""
    gc = graph.edge_Gc.copy()  # normalised boundary toughness
    cH = np.zeros(graph.n_edges) if c_H is None else np.asarray(c_H, float)
    ps = np.zeros(graph.n_edges) if psi is None else np.asarray(psi, float)
    if R is None:
        # default resistance: special / low-energy boundaries resist cracking
        R = (graph.edge_type == 2).astype(float)
    R = np.asarray(R, float)

    # normalise driving force to [0, 1] so the weights stay commensurate
    if ps.max() > ps.min():
        ps = (ps - ps.min()) / (ps.max() - ps.min())

    w = gc - params.alpha * cH - params.beta * ps + params.gamma * R
    return np.maximum(w, params.w_floor)


# --------------------------------------------------------------------------- #
# Source / target selection
# --------------------------------------------------------------------------- #


def grains_in_box(graph: GrainGraph, x_range=None, y_range=None) -> np.ndarray:
    """Grain ids whose centroid lies in the given (x, y) ranges (microns)."""
    cx, cy = graph.centroid[:, 0], graph.centroid[:, 1]
    mask = np.ones(graph.n_grains, dtype=bool)
    if x_range is not None:
        mask &= (cx >= x_range[0]) & (cx <= x_range[1])
    if y_range is not None:
        mask &= (cy >= y_range[0]) & (cy <= y_range[1])
    return np.where(mask)[0]


def notch_and_far_grains(graph: GrainGraph, notch_frac=0.2, band=0.18, target_band=0.25):
    """Source grains at the notch (mid-left) and target grains on the far edge.

    The far grains are restricted to a mid-height band so the predicted path
    exits near the mode-I ligament rather than diving to a cheaper corner.
    """
    Lx = graph.shape[1] * graph.step[0]
    Ly = graph.shape[0] * graph.step[1]
    ymid = 0.5 * Ly
    src = grains_in_box(
        graph,
        x_range=(0, notch_frac * Lx),
        y_range=(ymid - band * Ly, ymid + band * Ly),
    )
    tgt = grains_in_box(
        graph,
        x_range=(0.92 * Lx, Lx),
        y_range=(ymid - target_band * Ly, ymid + target_band * Ly),
    )
    return src, tgt


# --------------------------------------------------------------------------- #
# Shortest path via virtual super-source / super-sink
# --------------------------------------------------------------------------- #


def predict_path(graph: GrainGraph, weights, source_nodes, target_nodes):
    """Minimum-cost crack path; returns dict with node path, cost and polyline."""
    n = graph.n_grains
    src_v, snk_v = n, n + 1
    i = np.concatenate(
        [graph.edge_i, graph.edge_j, np.full(len(source_nodes), src_v), np.asarray(target_nodes)]
    )
    j = np.concatenate(
        [graph.edge_j, graph.edge_i, np.asarray(source_nodes), np.full(len(target_nodes), snk_v)]
    )
    w = np.concatenate([weights, weights, np.zeros(len(source_nodes)), np.zeros(len(target_nodes))])
    A = sparse.csr_matrix((w, (i, j)), shape=(n + 2, n + 2))

    dist, pred = csgraph.dijkstra(A, directed=True, indices=src_v, return_predecessors=True)
    if not np.isfinite(dist[snk_v]):
        return {"nodes": [], "cost": np.inf, "polyline": np.empty((0, 2))}

    # backtrack super-sink -> super-source
    path = []
    k = snk_v
    while k != src_v and k >= 0:
        path.append(k)
        k = pred[k]
    path.append(src_v)
    path = [p for p in reversed(path) if p < n]  # drop virtual nodes

    poly = graph.centroid[path]
    return {"nodes": path, "cost": float(dist[snk_v]), "polyline": poly}


def path_edges(graph: GrainGraph, nodes) -> list:
    """Edge indices traversed by a node path (for highlighting / regulation)."""
    lut = {
        (int(min(a, b)), int(max(a, b))): k
        for k, (a, b) in enumerate(zip(graph.edge_i, graph.edge_j, strict=True))
    }
    out = []
    for a, b in zip(nodes[:-1], nodes[1:], strict=True):
        key = (int(min(a, b)), int(max(a, b)))
        if key in lut:
            out.append(lut[key])
    return out


# --------------------------------------------------------------------------- #
# Path regulation (case study 3)
# --------------------------------------------------------------------------- #


def path_crosses(graph: GrainGraph, nodes, zone_nodes) -> bool:
    return len(set(nodes) & set(map(int, zone_nodes))) > 0


def regulate_path(
    graph: GrainGraph,
    weights: np.ndarray,
    source_nodes,
    target_nodes,
    protected_nodes,
    toughen: float = 2.0,
    max_iter: int = 40,
):
    """Greedily toughen boundaries to steer ``P*`` out of a protected zone.

    Each iteration raises the weight (``Gc`` design variable) of the cheapest
    protected-zone edge on the current path by ``toughen``, then re-routes, until
    the path no longer enters the protected zone or the budget is exhausted.
    Returns the baseline and regulated paths and the design effort.
    """
    protected = set(map(int, protected_nodes))
    w = weights.copy()
    base = predict_path(graph, w, source_nodes, target_nodes)
    history = [
        {
            "iter": 0,
            "cost": base["cost"],
            "crosses": path_crosses(graph, base["nodes"], protected),
            "design_effort": 0.0,
        }
    ]

    # candidate edges: those with at least one endpoint inside the protected zone
    in_zone = np.array(
        [
            (graph.edge_i[k] in protected) or (graph.edge_j[k] in protected)
            for k in range(graph.n_edges)
        ]
    )

    effort = 0.0
    cur = base
    for it in range(1, max_iter + 1):
        if not path_crosses(graph, cur["nodes"], protected):
            break
        pe = path_edges(graph, cur["nodes"])
        zone_pe = [k for k in pe if in_zone[k]]
        if not zone_pe:
            break
        k = min(zone_pe, key=lambda e: w[e])  # cheapest protected edge on path
        w[k] += toughen
        effort += toughen
        cur = predict_path(graph, w, source_nodes, target_nodes)
        history.append(
            {
                "iter": it,
                "cost": cur["cost"],
                "crosses": path_crosses(graph, cur["nodes"], protected),
                "design_effort": effort,
            }
        )

    return {
        "baseline": base,
        "regulated": cur,
        "design_weights": w,
        "design_effort": effort,
        "history": history,
        "diverted": not path_crosses(graph, cur["nodes"], protected),
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _grid_graph(n=6):
    """A small synthetic n*n grid grain graph for exact-path checks."""
    from .micrograph import GraphModelParams

    cells = n * n
    centroid = np.array([[c % n, c // n] for c in range(cells)], dtype=float)
    ei, ej = [], []
    for r in range(n):
        for c in range(n):
            a = r * n + c
            if c + 1 < n:
                ei.append(a)
                ej.append(a + 1)
            if r + 1 < n:
                ei.append(a)
                ej.append(a + n)
    ei = np.array(ei)
    ej = np.array(ej)
    z = np.zeros(len(ei))
    G = GrainGraph(
        name="grid",
        shape=(n, n),
        step=(1.0, 1.0),
        grain_id=np.arange(cells).reshape(n, n),
        n_grains=cells,
        centroid=centroid,
        area_um2=np.ones(cells),
        equiv_diam=np.ones(cells),
        node_phase=np.full(cells, 2),
        node_quat=np.tile([1, 0, 0, 0], (cells, 1)),
        schmid=np.full(cells, 0.45),
        node_kam=np.zeros(cells),
        edge_i=ei,
        edge_j=ej,
        edge_theta=np.full(len(ei), 30.0),
        edge_len=np.ones(len(ei)),
        edge_type=np.ones(len(ei), int),
        edge_gamma=z + 1,
        edge_DH=z + 1,
        edge_trap=z + 1,
        edge_Gc=z + 1.0,
        kam_field=np.zeros((n, n)),
        params=GraphModelParams(),
    )
    return G


def self_test() -> dict:
    out: dict = {}
    n = 6
    G = _grid_graph(n)

    # uniform weights: cost from corner 0 to corner n*n-1 is the Manhattan length
    w = np.ones(G.n_edges)
    res = predict_path(G, w, [0], [n * n - 1])
    out["uniform_cost"] = res["cost"]
    out["uniform_expected"] = float(2 * (n - 1))
    out["dijkstra_ok"] = abs(res["cost"] - 2 * (n - 1)) < 1e-9

    # cheap diagonal corridor should be preferred and lower the cost
    w2 = np.ones(G.n_edges) * 5.0
    lut = {
        (int(min(a, b)), int(max(a, b))): k
        for k, (a, b) in enumerate(zip(G.edge_i, G.edge_j, strict=True))
    }
    for d in range(n - 1):
        a = d * n + d
        if (a, a + 1) in lut:
            w2[lut[(a, a + 1)]] = 0.1
        if (min(a + 1, a + 1 + n), max(a + 1, a + 1 + n)) in lut:
            w2[lut[(a + 1, a + 1 + n)]] = 0.1
    res2 = predict_path(G, w2, [0], [n * n - 1])
    out["corridor_cost"] = res2["cost"]
    out["corridor_cheaper"] = res2["cost"] < out["uniform_cost"] * 5.0

    # regulation: a mid-left -> mid-right path runs straight through the centre;
    # protecting the centre must force a diversion at non-zero design effort.
    src = [2 * n + 0, 3 * n + 0]
    tgt = [2 * n + (n - 1), 3 * n + (n - 1)]
    centre = [r * n + c for r in (2, 3) for c in (2, 3)]
    base = predict_path(G, np.ones(G.n_edges), src, tgt)
    out["reg_baseline_crosses"] = path_crosses(G, base["nodes"], centre)
    reg = regulate_path(G, np.ones(G.n_edges), src, tgt, centre, toughen=3.0, max_iter=30)
    out["reg_diverted"] = bool(reg["diverted"])
    out["reg_effort"] = reg["design_effort"]
    out["reg_cost_increase"] = reg["regulated"]["cost"] - reg["baseline"]["cost"]

    out["ok"] = bool(
        out["dijkstra_ok"]
        and out["corridor_cheaper"]
        and out["reg_baseline_crosses"]
        and out["reg_diverted"]
        and out["reg_effort"] > 0.0
    )
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2))
