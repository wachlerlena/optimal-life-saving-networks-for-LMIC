"""
TRUE road-distance coverage of the 55 existing ADVANCED hospitals.

Unlike the provided nearest-facility matrix (which links each population point to
only its ~1 nearest facility among all 3,685 candidates), this rebuilds coverage
the way a maximum-covering model actually needs it: from EACH advanced hospital,
run single-source Dijkstra over the real road graph and keep EVERY population
point within range. A point is covered at radius R if ANY advanced hospital is
within R by road. Distance convention matches the pipeline:

    total_dist = pop_to_road_km + network_km + hosp_to_road_km   (all km)

Writes outputs/true_advanced_baseline.csv and prints coverage at several radii.
"""
import pickle, time as _time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

import build_population_map as bp

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
ROADS = DATA / "road_osm_preprocessed.geojson"
CUTOFF_KM = 300.0


def L(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def snap(tree, node_list, lons, lats):
    _, idx = tree.query(np.column_stack([lons, lats]), k=1)
    nodes = [node_list[i] for i in idx]
    nlat = np.array([n[1] for n in nodes], dtype=float)
    nlon = np.array([n[0] for n in nodes], dtype=float)
    off_km = bp.haversine_km(np.asarray(lats, float), np.asarray(lons, float), nlat, nlon)
    return nodes, off_km


def main():
    # --- advanced hospital ids (same logic as the model) ---
    sf = pd.read_csv(DATA / "stroke-facs-100-en.csv")
    sf["hosp_id"] = sf["TT"].astype(int) - 1
    adv_col = next(c for c in sf.columns if "can" in c.lower() and "thiệp" in c.lower())

    def is_adv(v):
        if pd.isna(v):
            return False
        t = str(v).strip().lower()
        return ("khong" not in t and "không" not in t) and ("co" in t or "có" in t)

    adv_ids = set(sf.loc[sf[adv_col].apply(is_adv), "hosp_id"].astype(int))
    print(f"advanced hospitals: {len(adv_ids)}", flush=True)

    # --- road graph ---
    t0 = _time.time()
    print("building road graph from geojson ...", flush=True)
    G = bp.build_graph_from_roads_geojson(bp.load_roads_geojson(str(ROADS)))
    tree, node_list, _ = bp.build_graph_kdtree(G)
    print(f"graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges "
          f"({_time.time()-t0:.0f}s)", flush=True)

    # --- population: snap ALL points, build node -> [(pop_id, offset_km)] ---
    pop = L(DATA / "population.pkl").copy()
    pop["pop_id"] = pop["ID"].astype(int)
    pop["demand"] = pop["household_count"].fillna(0).astype(float)
    TOT = float(pop["demand"].sum())
    pnodes, poff = snap(tree, node_list, pop["lon"].to_numpy(float), pop["lat"].to_numpy(float))
    node_pops = defaultdict(list)
    for pid, node, off in zip(pop["pop_id"].to_numpy(), pnodes, poff):
        node_pops[node].append((int(pid), float(off)))
    demand = dict(zip(pop["pop_id"].to_numpy(), pop["demand"].to_numpy()))
    print(f"snapped {len(pop):,} population points; total demand {TOT:,.0f}", flush=True)

    # --- advanced hospital coords, snapped ---
    allh = L(DATA / "all_hospitals.pkl")
    allh["ID"] = allh["ID"].astype(int)
    adv = allh[allh.ID.isin(adv_ids)]
    anodes, aoff = snap(tree, node_list, adv["Longitude"].to_numpy(float), adv["Latitude"].to_numpy(float))

    # --- Dijkstra from each advanced hospital; keep min total_dist per pop point ---
    best = {}
    t1 = _time.time()
    for k, (fnode, foff) in enumerate(zip(anodes, aoff)):
        if fnode not in G:
            continue
        lengths = nx.single_source_dijkstra_path_length(G, fnode, weight="length_km", cutoff=CUTOFF_KM)
        for node, road_km in lengths.items():
            pl = node_pops.get(node)
            if not pl:
                continue
            base = road_km + float(foff)
            for pid, poff_km in pl:
                tot = poff_km + base
                if tot < best.get(pid, 1e18):
                    best[pid] = tot
        print(f"  adv {k+1:2d}/{len(anodes)} done | reached pop pts so far={len(best):,} "
              f"| {_time.time()-t1:.0f}s", flush=True)

    # --- coverage at several radii ---
    bd = np.array([best.get(p, np.inf) for p in pop["pop_id"].to_numpy()])
    dem = pop["demand"].to_numpy()
    rows = []
    print("\n=== TRUE road-distance coverage by existing advanced hospitals (0 builds) ===", flush=True)
    for r in (15, 30, 50, 100, 150, 200, 300):
        cov = float(dem[bd <= r].sum())
        rows.append((r, cov, cov / TOT * 100))
        print(f"  within {r:>3} km road: {cov/TOT*100:5.1f}%  ({cov:,.0f} people)", flush=True)
    out = BASE / "outputs" / "true_advanced_baseline.csv"
    pd.DataFrame(rows, columns=["radius_km", "covered_demand", "coverage_pct"]).to_csv(out, index=False)
    print(f"\nsaved {out}", flush=True)
    print("FOR REFERENCE — model's sparse-matrix baseline at R=300km: 18.7%", flush=True)


if __name__ == "__main__":
    main()
