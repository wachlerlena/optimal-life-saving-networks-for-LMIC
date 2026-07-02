"""
Road-adjusted travel-time matrices (minutes).

Mirrors the precomputed road-DISTANCE matrices, but stores travel TIME computed
with per-road-type speeds instead of one flat average. Edge speeds come from
build_population_map.infer_speed_from_properties (motorway 70 / primary 55 /
secondary 40 / local 25 km/h, unpaved 30% slower); the shortest-TIME path is
found on that weighted graph.

For every (pop, facility) pair that already exists in the road-distance matrices
and is reachable within CUTOFF_MIN by road-adjusted time, we record:

    total_dist  =  pop_to_road_min  +  network_min  +  facility_to_road_min

(the column is still called total_dist, in MINUTES, so the optimisation pipeline
consumes it unchanged). The short off-network snap legs are converted with a
single connector speed CONNECTOR_KMH.

Outputs (to data/):
    existing_hospital_times.pkl   pop -> 130 existing hospitals
    greenfield_times.pkl          pop -> 3,555 greenfield candidate sites

Run from code/:  python3 build_travel_time_matrices.py
"""

import pickle
import time as _time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import build_population_map as bp

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
ROADS = DATA / "road_osm_preprocessed.geojson"

CONNECTOR_KMH = 40.0   # speed for the short off-network leg from a point to the road graph
CUTOFF_MIN = 250.0     # only pairs reachable within ~the largest time radius (240) matter


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def snap(tree, node_list, lons, lats):
    """Nearest graph node + offset (km) for arrays of lon/lat."""
    _, idx = tree.query(np.column_stack([lons, lats]), k=1)
    nodes = [node_list[i] for i in idx]
    nlat = np.array([n[1] for n in nodes], dtype=float)
    nlon = np.array([n[0] for n in nodes], dtype=float)
    off_km = bp.haversine_km(np.asarray(lats, float), np.asarray(lons, float), nlat, nlon)
    return nodes, off_km


def main():
    G = bp.build_graph_from_roads_geojson(bp.load_roads_geojson(str(ROADS)))
    tree, node_list, _ = bp.build_graph_kdtree(G)

    ex = load_pickle(DATA / "existing_hospital_distances.pkl")[["pop_id", "hosp_id"]].astype(int)
    gf = load_pickle(DATA / "greenfield_distances.pkl")[["pop_id", "hosp_id"]].astype(int)

    # --- snap population points in scope ---
    pop = load_pickle(DATA / "population.pkl").copy()
    pop["pop_id"] = pop["ID"].astype(int)
    scope = sorted((set(ex.pop_id) | set(gf.pop_id)) & set(pop["pop_id"]))
    popc = pop.set_index("pop_id").loc[scope, ["lat", "lon"]]
    pnodes, poff_km = snap(tree, node_list, popc["lon"].to_numpy(float), popc["lat"].to_numpy(float))
    pop_node = {pid: n for pid, n in zip(popc.index, pnodes)}
    pop_off_min = {pid: float(k) / CONNECTOR_KMH * 60.0 for pid, k in zip(popc.index, poff_km)}
    print(f"Snapped {len(scope):,} population points.")

    # --- facility coordinates ---
    allh = load_pickle(DATA / "all_hospitals.pkl"); allh["ID"] = allh["ID"].astype(int)
    exist_coords = allh[allh["ID"] < 130].set_index("ID")[["Latitude", "Longitude"]]
    newh = load_pickle(DATA / "new_hospitals.pkl"); newh["hosp_id"] = newh["Cluster_ID"].astype(int)
    gf_coords = newh.set_index("hosp_id")[["Latitude", "Longitude"]]

    def compute(pairs, coords_df, label):
        by_fac = pairs.groupby("hosp_id")["pop_id"].apply(lambda s: s.to_numpy())
        fids = list(by_fac.index)
        fc = coords_df.loc[fids]
        fnodes, foff_km = snap(tree, node_list, fc["Longitude"].to_numpy(float), fc["Latitude"].to_numpy(float))
        rows = []
        t0 = _time.time()
        for n, fid in enumerate(fids):
            fnode = fnodes[n]
            if fnode not in G:
                continue
            foff_min = float(foff_km[n]) / CONNECTOR_KMH * 60.0
            lengths = nx.single_source_dijkstra_path_length(
                G, fnode, weight="travel_time_min", cutoff=CUTOFF_MIN)
            for pid in by_fac.iloc[n]:
                pid = int(pid)
                nm = lengths.get(pop_node.get(pid))
                if nm is None:
                    continue
                tot = pop_off_min[pid] + float(nm) + foff_min
                if tot <= CUTOFF_MIN:
                    rows.append((pid, int(fid), round(tot, 3)))
            if (n + 1) % 200 == 0:
                print(f"  {label}: {n + 1}/{len(fids)} facilities | {_time.time() - t0:.0f}s | rows={len(rows):,}")
        df = pd.DataFrame(rows, columns=["pop_id", "hosp_id", "total_dist"])
        print(f"{label} matrix: {len(df):,} rows | {df.pop_id.nunique():,} pop | {df.hosp_id.nunique()} facilities")
        return df

    print("\n=== existing hospitals ===")
    ex_t = compute(ex, exist_coords, "existing")
    ex_t.to_pickle(DATA / "existing_hospital_times.pkl")
    print("\n=== greenfield sites ===")
    gf_t = compute(gf, gf_coords, "greenfield")
    gf_t.to_pickle(DATA / "greenfield_times.pkl")
    print("\nSaved existing_hospital_times.pkl + greenfield_times.pkl to data/")


if __name__ == "__main__":
    main()
