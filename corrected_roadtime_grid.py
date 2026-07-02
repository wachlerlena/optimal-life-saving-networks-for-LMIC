"""
CORRECTED ROAD-TIME grid — the single foundation for the corrected thesis
evaluation AND the rebuilt time-only artifact.

Builds proper within-threshold travel-time matrices from the road graph
(weight = travel_time_min, per-road-type speeds; connector legs at 40 km/h),
then aggregate-solves the combined model and the upgrade-only benchmark for
budgets 1..50 at thresholds 15/30/60/120 min.

Outputs -> outputs/corrected_clinical/
  roadtime_scenarios.csv                 coverage curve per (threshold, variant, budget)
  combined_scenarios_roadtime_corrected.json   artifact payload (budgets 1-50 x thresholds)
  covered_rt_{T}min_b{B}.csv             per-point access (min) for equity + access-time
  selected_rt_{T}min_b{B}.csv            chosen upgrades + builds (coords)
"""
import json, pickle, time as _time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

import build_population_map as bp
import greenfield_existing_advanced_model as M

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs" / "corrected_clinical"
ROADS = DATA / "road_osm_preprocessed.geojson"
CONNECTOR_KMH = 40.0
THRESH = [15, 30, 60, 120]      # minutes
CAP = 120.0                     # build all pairs within the largest threshold
BUDGETS = list(range(1, 51)) + [60, 75, 100, 125, 150, 175, 200]   # thesis curves: full range
ARTIFACT_BUDGETS = set(range(1, 51))                               # artifact slider: 1..50 only
SAVE_DETAIL = {10, 20, 50}
TIME_LIMIT = 1200
GAP = 0.005


def L(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def snap(tree, node_list, lons, lats):
    _, idx = tree.query(np.column_stack([lons, lats]), k=1)
    nodes = [node_list[i] for i in idx]
    nlat = np.array([n[1] for n in nodes], float); nlon = np.array([n[0] for n in nodes], float)
    return nodes, bp.haversine_km(np.asarray(lats, float), np.asarray(lons, float), nlat, nlon)


def build_time_pairs(G, node_pops, ids, fnodes, foff_km, cap, label):
    """Pairs (pop_id, hosp_id, total_min) within `cap` minutes; total = connector+network+connector."""
    rows = []
    t0 = _time.time()
    for fid, fn, fo in zip(ids, fnodes, foff_km):
        if fn not in G:
            continue
        fo_min = float(fo) / CONNECTOR_KMH * 60.0
        Lt = nx.single_source_dijkstra_path_length(G, fn, weight="travel_time_min", cutoff=cap)
        for node, net in Lt.items():
            pl = node_pops.get(node)
            if not pl:
                continue
            base = net + fo_min
            for pid, po_km in pl:
                tot = po_km / CONNECTOR_KMH * 60.0 + base
                if tot <= cap:
                    rows.append((pid, int(fid), tot))
    df = pd.DataFrame(rows, columns=["pop_id", "hosp_id", "total_dist"])
    print(f"  [{label}] {len(df):,} rows | {df.pop_id.nunique():,} pop | {df.hosp_id.nunique()} fac | {_time.time()-t0:.0f}s", flush=True)
    return df


def build_agg_model(weights, facsets, active):
    S, F = len(weights), len(active)
    if S == 0 or F == 0:
        return None
    fidx = {f: i for i, f in enumerate(active)}
    rows, cols, vals = [], [], []
    for s, fs in enumerate(facsets):
        rows.append(s); cols.append(s); vals.append(1.0)
        for f in fs:
            rows.append(s); cols.append(S + fidx[f]); vals.append(-1.0)
    rows += [S] * F; cols += list(range(S, S + F)); vals += [1.0] * F
    A = coo_matrix((vals, (rows, cols)), shape=(S + 1, S + F)).tocsr()
    return dict(A=A, S=S, F=F, active=active,
                c=np.concatenate([-np.asarray(weights, float), np.zeros(F)]),
                integrality=np.concatenate([np.zeros(S), np.ones(F)]),
                bounds=Bounds(np.zeros(S + F), np.ones(S + F)))


def solve_budget(model, budget):
    S, F = model["S"], model["F"]
    con = LinearConstraint(model["A"], np.full(S + 1, -np.inf),
                           np.concatenate([np.zeros(S), [float(budget)]]))
    res = milp(c=model["c"], integrality=model["integrality"], bounds=model["bounds"],
               constraints=con, options={"disp": False, "mip_rel_gap": GAP, "time_limit": TIME_LIMIT})
    if res.x is None:
        return None
    return [model["active"][i] for i in range(F) if res.x[S + i] > 0.5]


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
    advanced_ids = set(int(i) for i in exist.ID if adv_map.get(int(i), False))
    basic_ids = set(int(i) for i in exist.ID if not adv_map.get(int(i), False))
    newh = L(DATA / "new_hospitals.pkl"); newh["hosp_id"] = newh["Cluster_ID"].astype(int)
    ex_ll = exist.set_index("ID")[["Latitude", "Longitude"]]
    new_ll = newh.set_index("hosp_id")[["Latitude", "Longitude"]]
    alat = exist[exist.ID.isin(advanced_ids)].Latitude.to_numpy(float)
    alon = exist[exist.ID.isin(advanced_ids)].Longitude.to_numpy(float)
    def hav(a, b, c, d):
        R = 6371.0; p = np.pi / 180
        h = np.sin((c-a)*p/2)**2 + np.cos(a*p)*np.cos(c*p)*np.sin((d-b)*p/2)**2
        return 2*R*np.arcsin(np.sqrt(h))
    def recs(ids, ll):
        return [{"id": int(i), "lat": float(ll.loc[i].Latitude), "lon": float(ll.loc[i].Longitude)}
                for i in ids if i in ll.index]

    t0 = _time.time(); print("building graph ...", flush=True)
    G = bp.build_graph_from_roads_geojson(bp.load_roads_geojson(str(ROADS)))
    tree, node_list, _ = bp.build_graph_kdtree(G)
    print(f"graph {G.number_of_nodes():,} nodes ({_time.time()-t0:.0f}s)", flush=True)
    pop = L(DATA / "population.pkl").copy()
    pop["pop_id"] = pop["ID"].astype(int); pop["demand"] = pop["household_count"].fillna(0).astype(float)
    TOT = float(pop["demand"].sum()); demand = dict(zip(pop.pop_id.to_numpy(), pop.demand.to_numpy()))
    pnodes, poff = snap(tree, node_list, pop["lon"].to_numpy(float), pop["lat"].to_numpy(float))
    node_pops = defaultdict(list)
    for pid, node, off in zip(pop.pop_id.to_numpy(), pnodes, poff):
        node_pops[node].append((int(pid), float(off)))
    print(f"snapped {len(pop):,} pop pts | demand {TOT:,.0f}", flush=True)

    enodes, eoff = snap(tree, node_list, exist["Longitude"].to_numpy(float), exist["Latitude"].to_numpy(float))
    gnodes, goff = snap(tree, node_list, newh["Longitude"].to_numpy(float), newh["Latitude"].to_numpy(float))
    print("building within-120min road-time matrices (heavy, once) ...", flush=True)
    ex_all = build_time_pairs(G, node_pops, exist.ID.astype(int).tolist(), enodes, eoff, CAP, "existing 120min")
    gf_all = build_time_pairs(G, node_pops, newh.hosp_id.astype(int).tolist(), gnodes, goff, CAP, "greenfield 120min")

    rows = []
    scenarios = {str(b): {} for b in sorted(ARTIFACT_BUDGETS)}
    for T in THRESH:
        print(f"\n{'#'*55}\n# threshold {T} min\n{'#'*55}", flush=True)
        exT = ex_all[ex_all.total_dist <= T]; gfT = gf_all[gf_all.total_dist <= T]
        already = set(exT[exT.hosp_id.isin(advanced_ids)].pop_id.astype(int))
        adv_demand = sum(demand.get(p, 0.0) for p in already)
        basic_u = exT[exT.hosp_id.isin(basic_ids) & ~exT.pop_id.isin(already)][["pop_id", "hosp_id"]]
        gf_u = gfT[~gfT.pop_id.isin(already)][["pop_id", "hosp_id"]]
        basic_r = exT[exT.hosp_id.isin(basic_ids)].copy()
        print(f"  baseline {adv_demand/TOT*100:.2f}%", flush=True)
        for variant in ("combined", "upgrade_only"):
            cov = basic_u if variant == "upgrade_only" else pd.concat([basic_u, gf_u], ignore_index=True)
            ptsig = cov.sort_values("hosp_id").groupby("pop_id")["hosp_id"].apply(tuple)
            sigmap = defaultdict(float)
            for pid, fs in ptsig.items():
                sigmap[fs] += demand.get(int(pid), 0.0)
            sigs = list(sigmap.keys()); weights = [sigmap[s] for s in sigs]
            active = sorted({f for s in sigs for f in s})
            model = build_agg_model(weights, sigs, active)
            print(f"  [{variant}] {len(ptsig):,} pts -> {len(sigs):,} sigs | {len(active)} fac", flush=True)
            gf_r = gfT.copy() if variant == "combined" else gfT.iloc[0:0].copy()
            for B in BUDGETS:
                t1 = _time.time()
                sel = [] if model is None else solve_budget(model, B)
                if sel is None:
                    print(f"  [{variant} B={B}] FAILED", flush=True); continue
                ss = set(sel)
                new_cov = sum(weights[i] for i, s in enumerate(sigs) if ss.intersection(s))
                total = adv_demand + new_cov
                su = [f for f in sel if f < 130]; sg = [f for f in sel if f >= 130]
                bdd = [float(hav(new_ll.loc[f].Latitude, new_ll.loc[f].Longitude, alat, alon).min())
                       for f in sg if f in new_ll.index]
                rows.append(dict(threshold_min=T, variant=variant, budget=B,
                                 baseline_pct=adv_demand/TOT*100, coverage_pct=total/TOT*100,
                                 n_upgrades=len(su), n_builds=len(sg),
                                 build_km_to_adv_med=(float(np.median(bdd)) if bdd else np.nan)))
                if variant == "combined" and B in ARTIFACT_BUDGETS:
                    scenarios[str(B)][str(T)] = {
                        "upgrades": recs(su, ex_ll), "greenfield": recs(sg, new_ll),
                        "coverage_share_of_total": round(total / TOT, 4),
                        "total_covered_demand": round(total, 0)}
                    if B in SAVE_DETAIL:
                        new_a = M.assign_to_selection(basic_r, gf_r, su, sg, already, demand)
                        adv_a = M.advanced_assignment(exT, advanced_ids, already, T, demand)
                        pd.concat([adv_a[["pop_id", "distance_km", "demand", "ftype"]],
                                   new_a[["pop_id", "distance_km", "demand", "ftype"]]], ignore_index=True
                                  ).rename(columns={"distance_km": "access_min"}).to_csv(
                            OUT / f"covered_rt_{T}min_b{B}.csv", index=False)
                        pd.concat([ex_ll.loc[[i for i in su if i in ex_ll.index]].assign(kind="upgrade"),
                                   new_ll.loc[[i for i in sg if i in new_ll.index]].assign(kind="build")]
                                  ).to_csv(OUT / f"selected_rt_{T}min_b{B}.csv")
                if B in (1, 10, 25, 50) or variant == "upgrade_only" and B == 50:
                    print(f"  [{variant:12s} B={B:2d}] cov={total/TOT*100:5.2f}% upg={len(su):2d} bld={len(sg):2d} {_time.time()-t1:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "roadtime_scenarios.csv", index=False)

    payload = {"budgets": sorted(ARTIFACT_BUDGETS), "radii": THRESH, "unit": "min", "speed_kmh": None,
               "total_demand": TOT, "existing_advanced": recs(sorted(advanced_ids), ex_ll),
               "scenarios": scenarios}
    with open(OUT / "combined_scenarios_roadtime_corrected.json", "w") as f:
        json.dump(payload, f)
    print(f"\nDONE -> roadtime_scenarios.csv + combined_scenarios_roadtime_corrected.json", flush=True)


if __name__ == "__main__":
    main()
