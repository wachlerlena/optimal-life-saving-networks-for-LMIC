"""
Step 3 - Demand-Weighted Edge Betweenness Centrality (DWEBC).

For every population node i with weight w_i > 0:
  1. Find the nearest hospital by shortest travel time (Dijkstra).
  2. Add w_i to every edge along that shortest path:
        C(e) := C(e) + w_i

Implementation note
-------------------
Running 400k+ individual single-source Dijkstras is wasteful: every population
node reaches the same hospital set, and most of the work duplicates. Instead we
build the shortest-path tree rooted at the hospital set by running ONE Dijkstra
from a virtual super-source connected to every hospital node with a 0-weight
edge. That single pass yields:

  - dist[v]  -- travel time from v to its nearest hospital, in minutes
  - pred[v]  -- the parent of v in the shortest-path tree (the next node on
                the path toward the hospital)
  - root[v]  -- the hospital each node ultimately routes to

For every populated graph node we then walk pred[v] -> ... -> hospital and
increment edge counters. This reduces the cost from O(N * (E log V)) to
O((E + V) log V) for the shortest-path step plus a single tree walk.

Also applies the Step 2 hard cap: routes whose TOTAL distance (snap_origin +
network_distance_km + snap_hospital) exceeds 300 km are excluded from the
centrality sum and flagged as unreachable in the per-population output.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from analysis import config


SUPER_SOURCE = ("__super_source__", None)  # sentinel; not a real coordinate


@dataclass
class DWEBCResult:
    edge_centrality: pd.DataFrame      # one row per edge with C(e) + metadata
    pop_assignment: pd.DataFrame       # one row per population: nearest hospital, time, dist
    unreachable: pd.DataFrame          # population rows with no path or total dist > 300 km


def _log(msg: str) -> None:
    print(f"[dwebc] {msg}", flush=True)


def _build_super_source_graph(G: nx.Graph, hospital_nodes: list) -> nx.Graph:
    """
    Return a shallow copy of G with a sentinel super-source node attached by
    zero-weight edges to every hospital node.
    """
    H = G.copy()
    H.add_node(SUPER_SOURCE)
    seen = set()
    for n in hospital_nodes:
        if n in H and n not in seen:
            H.add_edge(SUPER_SOURCE, n, travel_time_min=0.0, length_km=0.0)
            seen.add(n)
    _log(f"  super-source attached to {len(seen)} unique hospital nodes "
         f"(of {len(hospital_nodes)} requested)")
    return H


def _shortest_path_tree(H: nx.Graph) -> tuple[dict, dict]:
    """
    Single Dijkstra from the super-source. Returns:
      pred: dict[node] -> immediate parent on path BACK to super-source (i.e. toward a hospital)
      dist: dict[node] -> travel time in minutes to the nearest hospital (0 at super-source)
    """
    _log("running single-source Dijkstra from super-source (this is the big one)...")
    t0 = time.time()
    pred, dist = nx.dijkstra_predecessor_and_distance(
        H, source=SUPER_SOURCE, weight="travel_time_min"
    )
    _log(f"  Dijkstra done in {time.time()-t0:.1f}s; covered {len(dist)} nodes")
    return pred, dist


def _resolve_root(node, pred: dict, root_cache: dict) -> object:
    """
    Walk pred[node] back until we hit the super-source's direct child (a hospital
    node). Cache the result for each node we touch so subsequent calls are O(1).
    """
    if node in root_cache:
        return root_cache[node]

    chain = []
    cur = node
    while cur is not None and cur != SUPER_SOURCE:
        if cur in root_cache:
            root = root_cache[cur]
            for c in chain:
                root_cache[c] = root
            return root
        chain.append(cur)
        parents = pred.get(cur, [])
        if not parents:
            for c in chain:
                root_cache[c] = None
            return None
        nxt = parents[0]
        if nxt == SUPER_SOURCE:
            # `cur` is itself the hospital node
            root = cur
            for c in chain:
                root_cache[c] = root
            return root
        cur = nxt

    for c in chain:
        root_cache[c] = None
    return None


def compute_dwebc(
    G: nx.Graph,
    pop_df: pd.DataFrame,
    hosp_df: pd.DataFrame,
    max_route_km: float = config.MAX_ROUTE_DISTANCE_KM,
) -> DWEBCResult:
    """
    Parameters
    ----------
    G       : NetworkX graph with edge attrs `travel_time_min` and `length_km`.
    pop_df  : population dataframe with columns
                ID, lat, lon, household_count,
                population_node, population_to_graph_km
    hosp_df : hospital dataframe with columns
                ID, hospital_node, hospital_to_graph_km, specialized_equipment

    Returns
    -------
    DWEBCResult
    """
    # ----- Hospital lookup tables -----
    hosp_df = hosp_df.copy()
    hosp_df["ID"] = hosp_df["ID"].astype(int)

    # Map graph node -> hospital ID. If multiple hospitals snap to the same
    # node, prefer the comprehensive one (specialized_equipment=True) since the
    # tier metrics in Step 5 care about access to comprehensive centres.
    hosp_df_sorted = hosp_df.sort_values("specialized_equipment", ascending=False)
    node_to_hosp = {}
    node_to_hosp_snap_km = {}
    for _, r in hosp_df_sorted.iterrows():
        node = r["hospital_node"]
        if node not in node_to_hosp:
            node_to_hosp[node] = int(r["ID"])
            node_to_hosp_snap_km[node] = float(r["hospital_to_graph_km"])

    hospital_nodes = list(node_to_hosp.keys())
    _log(f"hospital nodes (unique): {len(hospital_nodes)}")

    # ----- Single Dijkstra from a virtual super-source -----
    H = _build_super_source_graph(G, hospital_nodes)
    pred, dist = _shortest_path_tree(H)

    # ----- Edge counter -----
    # Use a frozenset-keyed dict so undirected lookups are symmetric.
    edge_counts: dict[frozenset, float] = {}

    root_cache: dict = {SUPER_SOURCE: None}
    for n in hospital_nodes:
        root_cache[n] = n  # hospital nodes are their own root

    # ----- Per-population path walk -----
    t0 = time.time()
    n_total = len(pop_df)
    n_zero = 0
    n_no_path = 0
    n_too_far = 0
    n_used = 0

    nearest_hosp_id_arr = np.full(n_total, -1, dtype=np.int64)
    travel_time_arr = np.full(n_total, np.nan, dtype=float)
    network_km_arr = np.full(n_total, np.nan, dtype=float)
    total_km_arr = np.full(n_total, np.nan, dtype=float)
    path_edges_arr = np.zeros(n_total, dtype=np.int32)
    reachable_arr = np.zeros(n_total, dtype=bool)

    pop_nodes = pop_df["population_node"].to_numpy()
    pop_weights = pop_df["household_count"].to_numpy(dtype=float)
    pop_snap_km = pop_df["population_to_graph_km"].to_numpy(dtype=float)

    for i in range(n_total):
        w = pop_weights[i]
        node = pop_nodes[i]

        if w <= 0:
            n_zero += 1
            continue

        if node not in dist or dist[node] == np.inf:
            n_no_path += 1
            continue

        # Walk pred chain, summing edge length_km, until we reach the hospital.
        cur = node
        network_km = 0.0
        path_edges = []
        while True:
            parents = pred.get(cur, [])
            if not parents:
                break
            nxt = parents[0]
            if nxt == SUPER_SOURCE:
                break  # cur is the hospital node
            edge_data = G.get_edge_data(cur, nxt)
            if edge_data is None:
                # Should not happen except for the super-source edges (which we
                # already short-circuited above).
                break
            network_km += float(edge_data["length_km"])
            path_edges.append(frozenset((cur, nxt)))
            cur = nxt

        hosp_node = cur
        hosp_id = node_to_hosp.get(hosp_node, -1)
        if hosp_id < 0:
            n_no_path += 1
            continue

        hosp_snap_km = node_to_hosp_snap_km.get(hosp_node, 0.0)
        total_km = float(pop_snap_km[i]) + network_km + hosp_snap_km

        if total_km > max_route_km:
            n_too_far += 1
            # Still record nearest hospital for diagnostics, but do not contribute to C(e).
            nearest_hosp_id_arr[i] = hosp_id
            travel_time_arr[i] = float(dist[node])
            network_km_arr[i] = network_km
            total_km_arr[i] = total_km
            path_edges_arr[i] = len(path_edges)
            reachable_arr[i] = False
            continue

        # Reachable: accumulate w_i onto each edge along the path.
        for e in path_edges:
            edge_counts[e] = edge_counts.get(e, 0.0) + w

        nearest_hosp_id_arr[i] = hosp_id
        travel_time_arr[i] = float(dist[node])
        network_km_arr[i] = network_km
        total_km_arr[i] = total_km
        path_edges_arr[i] = len(path_edges)
        reachable_arr[i] = True
        n_used += 1

        if (i + 1) % 50000 == 0:
            _log(f"  processed {i+1}/{n_total} population rows "
                 f"({n_used} used, {n_too_far} too-far, {n_no_path} no-path) "
                 f"elapsed {time.time()-t0:.1f}s")

    _log(f"path walk done in {time.time()-t0:.1f}s")
    _log(f"  used: {n_used}   zero-weight: {n_zero}   no-path: {n_no_path}   "
         f">300km: {n_too_far}")
    _log(f"  distinct edges touched: {len(edge_counts)}")

    # ----- Edge centrality dataframe -----
    rows = []
    for u, v, data in G.edges(data=True):
        key = frozenset((u, v))
        c = edge_counts.get(key, 0.0)
        rows.append({
            "u_lon": u[0], "u_lat": u[1],
            "v_lon": v[0], "v_lat": v[1],
            "length_km": float(data.get("length_km", np.nan)),
            "travel_time_min": float(data.get("travel_time_min", np.nan)),
            "speed_kmh": float(data.get("speed_kmh", np.nan)) if data.get("speed_kmh") is not None else np.nan,
            "rtt": data.get("rtt"),
            "rst": data.get("rst"),
            "centrality": c,
        })
    edge_centrality = pd.DataFrame(rows).sort_values("centrality", ascending=False).reset_index(drop=True)

    # ----- Per-population assignment -----
    pop_assignment = pd.DataFrame({
        "pop_id": pop_df["ID"].to_numpy(),
        "lat": pop_df["lat"].to_numpy(),
        "lon": pop_df["lon"].to_numpy(),
        "household_count": pop_weights,
        "pop_snap_km": pop_snap_km,
        "nearest_hospital_id": nearest_hosp_id_arr,
        "travel_time_min": travel_time_arr,
        "network_km": network_km_arr,
        "total_distance_km": total_km_arr,
        "path_edges": path_edges_arr,
        "reachable_300km": reachable_arr,
    })

    unreachable = pop_assignment[
        (pop_weights > 0) & (~reachable_arr)
    ].copy()

    return DWEBCResult(
        edge_centrality=edge_centrality,
        pop_assignment=pop_assignment,
        unreachable=unreachable,
    )
