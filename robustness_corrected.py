"""
Robustness checks on the CORRECTED model (proper within-R road coverage).
Designed to run ALONGSIDE the main recompute: builds the road graph once, keeps a
small memory footprint, writes independent CSVs.

(A) Service-radius / time-threshold sensitivity  -> robustness_radius.csv, robustness_time.csv
(B) Travel-speed sensitivity (flat 40 km/h vs per-road-type) -> robustness_speed*.csv
(C) Facility-cost sensitivity (build = k upgrades, break-even k) -> robustness_cost.csv

(A)+(B) come from Dijkstra off the 55 advanced hospitals on the road graph
(length_km and travel_time_min). (C) rebuilds within-R matrices at the 30-min
radius and re-solves the aggregated MCLP with a cost-weighted budget.
"""
import pickle, time as _time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, vstack as svstack

import build_population_map as bp

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs" / "corrected_clinical"
ROADS = DATA / "road_osm_preprocessed.geojson"
CONNECTOR_KMH = 40.0
COST_RADIUS_KM = 20.0          # 30 min: light radius for the cost re-solve
COST_KS = [1, 2, 3, 4, 5]
COST_BUDGETS = [10, 20]
RADII_KM = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300]
TIME_MIN = [15, 30, 45, 60, 90, 120, 180, 240]


def L(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def snap(tree, node_list, lons, lats):
    _, idx = tree.query(np.column_stack([lons, lats]), k=1)
    nodes = [node_list[i] for i in idx]
    nlat = np.array([n[1] for n in nodes], float); nlon = np.array([n[0] for n in nodes], float)
    return nodes, bp.haversine_km(np.asarray(lats, float), np.asarray(lons, float), nlat, nlon)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sf = pd.read_csv(DATA / "stroke-facs-100-en.csv"); sf["hosp_id"] = sf["TT"].astype(int) - 1
    ac = next(c for c in sf.columns if "can" in c.lower() and "thiệp" in c.lower())
    def is_adv(v):
        if pd.isna(v): return False
        t = str(v).strip().lower(); return ("khong" not in t and "không" not in t) and ("co" in t or "có" in t)
    sf["adv"] = sf[ac].apply(is_adv); adv_map = dict(zip(sf.hosp_id, sf.adv))
    allh = L(DATA / "all_hospitals.pkl"); allh["ID"] = allh["ID"].astype(int)
    exist = allh[allh.ID < 130].copy()
    advanced_ids = [int(i) for i in exist.ID if adv_map.get(int(i), False)]
    basic_ids = set(int(i) for i in exist.ID if not adv_map.get(int(i), False))
    newh = L(DATA / "new_hospitals.pkl"); newh["hosp_id"] = newh["Cluster_ID"].astype(int)

    t0 = _time.time(); print("building graph ...", flush=True)
    G = bp.build_graph_from_roads_geojson(bp.load_roads_geojson(str(ROADS)))
    tree, node_list, _ = bp.build_graph_kdtree(G)
    print(f"graph {G.number_of_nodes():,} nodes ({_time.time()-t0:.0f}s)", flush=True)

    pop = L(DATA / "population.pkl").copy()
    pop["pop_id"] = pop["ID"].astype(int); pop["demand"] = pop["household_count"].fillna(0).astype(float)
    TOT = float(pop["demand"].sum())
    pnodes, poff = snap(tree, node_list, pop["lon"].to_numpy(float), pop["lat"].to_numpy(float))
    node_pops = defaultdict(list)
    for pid, node, off in zip(pop.pop_id.to_numpy(), pnodes, poff):
        node_pops[node].append((int(pid), float(off)))
    demand = dict(zip(pop.pop_id.to_numpy(), pop.demand.to_numpy()))

    adv = exist[exist.ID.isin(advanced_ids)]
    anodes, aoff = snap(tree, node_list, adv["Longitude"].to_numpy(float), adv["Latitude"].to_numpy(float))

    # ---- Dijkstra off advanced: min distance (km) and min time (min) per point ----
    INF = 1e18
    dist_to = defaultdict(lambda: INF)      # shortest road-km to any advanced
    time_to = defaultdict(lambda: INF)      # fastest road-time to any advanced
    best_t = {}                             # which advanced gives min time
    print("Dijkstra: time pass ...", flush=True)
    for k, (fn, fo) in enumerate(zip(anodes, aoff)):
        if fn not in G: continue
        fo_t = float(fo) / CONNECTOR_KMH * 60.0
        Lt = nx.single_source_dijkstra_path_length(G, fn, weight="travel_time_min", cutoff=360.0)
        for node, tt in Lt.items():
            pl = node_pops.get(node)
            if not pl: continue
            base = tt + fo_t
            for pid, po in pl:
                v = po / CONNECTOR_KMH * 60.0 + base
                if v < time_to[pid]:
                    time_to[pid] = v; best_t[pid] = k
    print("Dijkstra: length pass ...", flush=True)
    dist_same = defaultdict(lambda: INF)    # km to the time-nearest advanced (same facility)
    for k, (fn, fo) in enumerate(zip(anodes, aoff)):
        if fn not in G: continue
        Ld = nx.single_source_dijkstra_path_length(G, fn, weight="length_km", cutoff=300.0)
        for node, dd in Ld.items():
            pl = node_pops.get(node)
            if not pl: continue
            base = dd + float(fo)
            for pid, po in pl:
                v = po + base
                if v < dist_to[pid]: dist_to[pid] = v
                if best_t.get(pid) == k and v < dist_same[pid]: dist_same[pid] = v

    ids = pop.pop_id.to_numpy()
    dem = pop.demand.to_numpy()
    dvec = np.array([dist_to[p] for p in ids]); tvec = np.array([time_to[p] for p in ids])
    dsame = np.array([dist_same[p] for p in ids])

    # (A) radius + time-threshold sensitivity (baseline = existing advanced, 0 builds)
    pd.DataFrame([(r, float(dem[dvec <= r].sum()) / TOT * 100) for r in RADII_KM],
                 columns=["radius_km", "baseline_coverage_pct"]).to_csv(OUT / "robustness_radius.csv", index=False)
    rowsT = []
    for T in TIME_MIN:
        flat = float(dem[dvec <= T * CONNECTOR_KMH / 60.0].sum()) / TOT * 100   # flat: dist <= 40*T/60
        road = float(dem[tvec <= T].sum()) / TOT * 100                          # road: actual time
        rowsT.append((T, flat, road))
    pd.DataFrame(rowsT, columns=["time_min", "flat_coverage_pct", "road_coverage_pct"]).to_csv(OUT / "robustness_time.csv", index=False)
    print("(A) wrote radius + time sensitivity", flush=True)

    # (B) speed: effective road speed to the (time-)nearest advanced, vs flat 40
    ok = (tvec < INF) & (dsame < INF) & (tvec > 0)
    eff = dsame[ok] / (tvec[ok] / 60.0)            # km/h on the realistic route
    flat_time = dsame[ok] / CONNECTOR_KMH * 60.0   # flat model's time for that same trip
    road_time = tvec[ok]
    w = dem[ok]
    pct_faster = float((w[road_time < flat_time].sum()) / w.sum() * 100)
    # rural vs urban by household-count quintile (population-weighted thirds of demand by density)
    dens = pop.demand.to_numpy()[ok]
    order = np.argsort(dens)
    cum = np.cumsum(dens[order]); q1 = cum <= cum[-1] * 0.2; q5 = cum >= cum[-1] * 0.8
    eff_o = eff[order]
    summary = dict(pct_trips_faster_than_flat=pct_faster,
                   median_eff_speed=float(np.median(eff)),
                   mean_eff_speed=float(np.average(eff, weights=w)),
                   median_eff_speed_rural_q1=float(np.median(eff_o[q1])),
                   median_eff_speed_urban_q5=float(np.median(eff_o[q5])))
    pd.DataFrame([summary]).to_csv(OUT / "robustness_speed_summary.csv", index=False)
    pd.DataFrame({"pctile": list(range(0, 101, 10)),
                  "eff_speed_kmh": [float(np.percentile(eff, p)) for p in range(0, 101, 10)]}
                 ).to_csv(OUT / "robustness_speed_deciles.csv", index=False)
    print(f"(B) speed: {pct_faster:.0f}% trips faster than flat; rural q1 median "
          f"{summary['median_eff_speed_rural_q1']:.0f} vs urban q5 {summary['median_eff_speed_urban_q5']:.0f} km/h", flush=True)

    # (C) cost sensitivity at 30 min (20 km): build within-R matrices, aggregate, cost-weighted solve
    def build_pairs(ids_, fnodes, foff, cap):
        rows = []
        for fid, fn, fo in zip(ids_, fnodes, foff):
            if fn not in G: continue
            Ld = nx.single_source_dijkstra_path_length(G, fn, weight="length_km", cutoff=cap)
            bo = float(fo)
            for node, dd in Ld.items():
                pl = node_pops.get(node)
                if not pl: continue
                base = dd + bo
                for pid, po in pl:
                    if po + base <= cap: rows.append((pid, int(fid), po + base))
        return pd.DataFrame(rows, columns=["pop_id", "hosp_id", "total_dist"])

    print("(C) building 20 km matrices ...", flush=True)
    enodes, eoff = snap(tree, node_list, exist["Longitude"].to_numpy(float), exist["Latitude"].to_numpy(float))
    gnodes, goff = snap(tree, node_list, newh["Longitude"].to_numpy(float), newh["Latitude"].to_numpy(float))
    exP = build_pairs(exist.ID.astype(int).tolist(), enodes, eoff, COST_RADIUS_KM)
    gfP = build_pairs(newh.hosp_id.astype(int).tolist(), gnodes, goff, COST_RADIUS_KM)
    already = set(exP[exP.hosp_id.isin(advanced_ids)].pop_id.astype(int))
    adv_demand = sum(demand.get(p, 0.0) for p in already)
    bas = exP[exP.hosp_id.isin(basic_ids) & ~exP.pop_id.isin(already)][["pop_id", "hosp_id"]]
    gfu = gfP[~gfP.pop_id.isin(already)][["pop_id", "hosp_id"]]
    cov = pd.concat([bas, gfu], ignore_index=True)
    ptsig = cov.sort_values("hosp_id").groupby("pop_id")["hosp_id"].apply(tuple)
    sigmap = defaultdict(float)
    for pid, fs in ptsig.items(): sigmap[fs] += demand.get(int(pid), 0.0)
    sigs = list(sigmap.keys()); weights = [sigmap[s] for s in sigs]
    active = sorted({f for s in sigs for f in s})
    fidx = {f: i for i, f in enumerate(active)}; S, F = len(sigs), len(active)
    print(f"(C) {len(ptsig):,} pts -> {S:,} signatures, {F} active fac; baseline {adv_demand/TOT*100:.2f}%", flush=True)
    # coverage rows (fixed across k); budget row coefficients change with k
    rows, cols, vals = [], [], []
    for s, fs in enumerate(sigs):
        rows.append(s); cols.append(s); vals.append(1.0)
        for f in fs:
            rows.append(s); cols.append(S + fidx[f]); vals.append(-1.0)
    Acov = coo_matrix((vals, (rows, cols)), shape=(S, S + F)).tocsr()
    c = np.concatenate([-np.asarray(weights, float), np.zeros(F)])
    integ = np.concatenate([np.zeros(S), np.ones(F)])
    bnds = Bounds(np.zeros(S + F), np.ones(S + F))
    is_build = np.array([1 if f >= 130 else 0 for f in active])
    crows = []
    for k in COST_KS:
        cost = np.where(is_build == 1, float(k), 1.0)
        budrow = coo_matrix((cost, (np.zeros(F), np.arange(S, S + F))), shape=(1, S + F)).tocsr()
        Afull = svstack([Acov, budrow]).tocsr()
        for B in COST_BUDGETS:
            con = LinearConstraint(Afull, np.full(S + 1, -np.inf), np.concatenate([np.zeros(S), [float(B)]]))
            res = milp(c=c, integrality=integ, bounds=bnds, constraints=con,
                       options={"disp": False, "mip_rel_gap": 0.005, "time_limit": 900})
            if res.x is None:
                crows.append((k, B, np.nan, -1, -1)); continue
            sel = [active[i] for i in range(F) if res.x[S + i] > 0.5]
            selset = set(sel)
            new_cov = sum(weights[i] for i, s in enumerate(sigs) if selset.intersection(s))
            nb = len([f for f in sel if f >= 130]); nu = len([f for f in sel if f < 130])
            crows.append((k, B, (adv_demand + new_cov) / TOT * 100, nu, nb))
            print(f"  [cost k={k} B={B}] cov={(adv_demand+new_cov)/TOT*100:.2f}%  upg={nu} bld={nb}", flush=True)
    pd.DataFrame(crows, columns=["build_cost_k", "budget", "coverage_pct", "n_upgrades", "n_builds"]).to_csv(
        OUT / "robustness_cost.csv", index=False)
    print("(C) wrote cost sensitivity. DONE.", flush=True)


if __name__ == "__main__":
    main()
