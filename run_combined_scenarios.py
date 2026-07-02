"""
Pre-compute the combined upgrade-or-greenfield scenarios for the interactive map.

For every (budget, radius) in the grid it solves the combined model and records
the optimal mix of facilities to open — which existing basic hospitals to UPGRADE
and which greenfield sites to BUILD — plus coverage and access-distance metrics.

The result is embedded into a static HTML explorer by build_scenario_map.py, so
the grid here defines exactly the budget slider values and radius options the
user can pick.

By default this uses the fast STAGE-1 (max-coverage) solve, which gives the same
site selection as the full two-stage model (stage 2 only breaks distance ties and
was shown not to move sites). Pass --refine to run the full two-stage solve.

Run:
    python run_combined_scenarios.py            # fast, ~15 min
    python run_combined_scenarios.py --refine   # full two-stage, slower

Output:
    outputs/combined_scenarios.json   (embedded by the map)
    outputs/combined_scenarios.csv    (flat summary table)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import greenfield_existing_advanced_model as gm

# --- scenario grid (defines the map's slider + dropdown) ---
BUDGETS   = list(range(0, 76))        # 0 .. 75 (show the full behaviour of the curve)
KM_RADII  = [50, 100, 300]            # km  — distance version
TIME_MIN  = [30, 60, 120, 240]        # min — time-to-treatment version
SPEED_KMH = 40.0                      # single average road speed for km -> min

OUT_DIR  = Path(__file__).resolve().parent.parent / "outputs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Per-solve limits for the fast coverage MILP. The max-coverage LP relaxation is
# weak at very low budgets (a single facility spread fractionally across many
# near-equal candidates), so proving optimality can blow up. We target a tight
# 0.1% gap but keep a binding time cap so no single solve can run away; any cell
# that hits the cap returns its best incumbent (still <=0.1% in practice — the
# grid matches the certified main-model result at B10/R300).
SOLVE_TIME_LIMIT = 120     # seconds, binding safety net
SOLVE_GAP        = 0.001   # 0.1% relative MIP gap


def solve_coverage_fast(pop_weight, already_covered, basic_pairs, gf_pairs,
                        budget, time_limit=SOLVE_TIME_LIMIT, mip_gap=SOLVE_GAP):
    """
    Fast stage-1 max-coverage solve (identical MILP to the main model's
    solve_combined, but built with vectorised scipy sparse instead of Pyomo's
    per-constraint Python loop — seconds instead of minutes for ~380K points).

    max  sum_i p_i z_i
    s.t. z_i <= sum_{f covers i} (u_f + y_f)        for each uncovered point i
         sum u + sum y <= budget
         u, y binary ;  z in [0,1]   (z naturally integral at the optimum)
    """
    uncovered = set(p for p in pop_weight if p not in already_covered)
    b = basic_pairs[basic_pairs["pop_id"].isin(uncovered)][["pop_id", "hosp_id"]].copy()
    g = gf_pairs[gf_pairs["pop_id"].isin(uncovered)][["pop_id", "hosp_id"]].copy()
    b["kind"] = "u"
    g["kind"] = "y"
    pairs = pd.concat([b, g], ignore_index=True)
    pairs["demand"] = pairs["pop_id"].map(pop_weight).fillna(0.0).astype(float)
    pairs = pairs[pairs["demand"] > 0].reset_index(drop=True)
    if len(pairs) == 0:
        return [], []

    # Trivial budgets handled directly — avoids the max-coverage MILP whose LP
    # relaxation is pathologically weak at budget 1 (one facility spread
    # fractionally across thousands of near-equal candidates).
    if budget <= 0:
        return [], []
    if budget == 1:
        # With a single facility there is no interaction: the optimum is simply
        # the facility covering the most uncovered demand (exact, instant).
        cov = pairs.groupby(["kind", "hosp_id"])["demand"].sum()
        kind, hid = cov.idxmax()
        return ([int(hid)], []) if kind == "u" else ([], [int(hid)])

    pop_ids = np.sort(pairs["pop_id"].unique())
    pop_row = {int(p): i for i, p in enumerate(pop_ids)}
    upg_ids = np.sort(pairs.loc[pairs.kind == "u", "hosp_id"].unique())
    gf_ids  = np.sort(pairs.loc[pairs.kind == "y", "hosp_id"].unique())
    u_col = {int(h): i for i, h in enumerate(upg_ids)}
    g_col = {int(h): i for i, h in enumerate(gf_ids)}

    n_pop, n_u, n_g = len(pop_ids), len(upg_ids), len(gf_ids)
    u_start, g_start, z_start = 0, n_u, n_u + n_g
    n_vars = n_u + n_g + n_pop

    prow = pairs["pop_id"].map(pop_row).to_numpy()
    is_u = (pairs.kind == "u").to_numpy()
    fac_col = np.where(
        is_u,
        u_start + pairs["hosp_id"].map(u_col).to_numpy(dtype=float),
        g_start + pairs["hosp_id"].map(g_col).to_numpy(dtype=float),
    ).astype(int)

    zrows = np.arange(n_pop)
    # coverage rows 0..n_pop-1 : z_i - sum coverers <= 0 ; budget row n_pop
    rows = np.concatenate([prow, zrows, np.full(n_u + n_g, n_pop)])
    cols = np.concatenate([fac_col, z_start + zrows, np.arange(u_start, z_start)])
    vals = np.concatenate([-np.ones(len(pairs)), np.ones(n_pop), np.ones(n_u + n_g)])
    A = coo_matrix((vals, (rows, cols)), shape=(n_pop + 1, n_vars)).tocsr()

    con_lb = np.concatenate([np.full(n_pop, -np.inf), [-np.inf]])
    con_ub = np.concatenate([np.zeros(n_pop), [float(budget)]])

    integrality = np.zeros(n_vars, dtype=int)
    integrality[:z_start] = 1                      # only u, y are binary
    pdem = np.array([pop_weight[int(p)] for p in pop_ids])
    c = np.zeros(n_vars)
    c[z_start:] = -pdem                            # maximise covered demand

    res = milp(
        c=c, integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(A, con_lb, con_ub),
        options={"disp": False, "mip_rel_gap": mip_gap, "time_limit": time_limit},
    )
    if res.x is None:
        raise RuntimeError(f"coverage solve failed: {res.message}")
    if not res.success:
        # time/iteration limit hit — use the best incumbent found
        print(f"    [note] accepting incumbent (not certified optimal): {res.message}")
    xv = res.x
    sel_u = [int(upg_ids[i]) for i in range(n_u) if xv[u_start + i] > 0.5]
    sel_g = [int(gf_ids[i])  for i in range(n_g) if xv[g_start + i] > 0.5]
    return sel_u, sel_g


def sites_to_records(ids, coord_df, id_col):
    """[{id, lat, lon}, ...] for the given facility ids."""
    sub = coord_df[coord_df[id_col].isin(ids)]
    return [
        {"id": int(r[id_col]), "lat": float(r["Latitude"]), "lon": float(r["Longitude"])}
        for _, r in sub.iterrows()
    ]


def _to_minutes(df):
    """Copy a distance-pair frame, replacing road km with travel minutes."""
    t = df.copy()
    t["total_dist"] = t["total_dist"].astype(float) / SPEED_KMH * 60.0
    return t


def main(mode: str, refine: bool):
    (pop, existing_pairs, greenfield_pairs, advanced_ids, basic_ids,
     greenfield_ids, all_hosp, new_hosp, total_demand) = gm.load_data()

    if mode == "time":
        existing_pairs   = _to_minutes(existing_pairs)
        greenfield_pairs = _to_minutes(greenfield_pairs)
        radii, unit = TIME_MIN, "min"
        json_out = OUT_DIR / "combined_scenarios_time.json"
        csv_out  = OUT_DIR / "combined_scenarios_time.csv"
        print(f"TIME mode: {SPEED_KMH:.0f} km/h  ->  thresholds {TIME_MIN} min")
    elif mode == "roadtime":
        # Travel time from differentiated per-road-type speeds (precomputed by
        # build_travel_time_matrices.py); the matrices are already in MINUTES.
        existing_pairs   = pd.read_pickle(DATA_DIR / "existing_hospital_times.pkl")
        greenfield_pairs = pd.read_pickle(DATA_DIR / "greenfield_times.pkl")
        radii, unit = TIME_MIN, "min"
        json_out = OUT_DIR / "combined_scenarios_roadtime.json"
        csv_out  = OUT_DIR / "combined_scenarios_roadtime.csv"
        print(f"ROAD-TIME mode: per-road-type speeds  ->  thresholds {TIME_MIN} min")
    else:
        radii, unit = KM_RADII, "km"
        json_out = OUT_DIR / "combined_scenarios_km.json"
        csv_out  = OUT_DIR / "combined_scenarios_km.csv"

    # Context layer: existing advanced hospitals (always shown on the map).
    adv_coords = all_hosp[all_hosp["ID"].isin(advanced_ids)]
    existing_advanced = [
        {"id": int(r["ID"]), "lat": float(r["Latitude"]), "lon": float(r["Longitude"])}
        for _, r in adv_coords.iterrows()
    ]

    scenarios = {str(b): {} for b in BUDGETS}
    flat_rows = []

    for R in radii:
        print(f"\n{'#'*55}\n# Radius {R} {unit}\n{'#'*55}")
        already_covered, pop_weight, basic_r, gf_r = gm.preprocess(
            pop, existing_pairs, greenfield_pairs,
            advanced_ids, basic_ids, greenfield_ids, R,
        )
        scope_demand = sum(pop_weight.values())
        adv_assign = gm.advanced_assignment(
            existing_pairs, advanced_ids, already_covered, R, pop_weight
        )

        for B in BUDGETS:
            if refine:
                sel_u, sel_g, _adv, new_assign = gm.solve_two_stage(
                    pop_weight, already_covered, basic_r, gf_r,
                    existing_pairs, advanced_ids, B, R,
                )
            else:
                sel_u, sel_g = solve_coverage_fast(
                    pop_weight, already_covered, basic_r, gf_r, budget=B,
                )
                new_assign = gm.assign_to_selection(
                    basic_r, gf_r, sel_u, sel_g, already_covered, pop_weight
                )

            summary = gm.build_summary(
                adv_assign, new_assign, sel_u, sel_g, B, R,
                advanced_ids, basic_ids, total_demand, scope_demand,
            )

            scenarios[str(B)][str(R)] = {
                "upgrades":   sites_to_records(sel_u, all_hosp, "ID"),
                "greenfield": sites_to_records(sel_g, new_hosp, "Cluster_ID"),
                "coverage_share_of_total":   round(summary["coverage_share_of_total"], 4),
                "total_covered_demand":      round(summary["total_covered_demand"], 0),
                "newly_covered_demand":      round(summary["newly_covered_demand"], 0),
                "avg_distance_overall_km":   round(summary["avg_distance_overall_km"], 2),
                "avg_distance_upgrade_km":   round(summary["avg_distance_upgrade_km"], 2),
                "avg_distance_greenfield_km": round(summary["avg_distance_greenfield_km"], 2),
            }
            flat_rows.append(summary)
            print(f"  B={B:2d}  upgrades={len(sel_u):2d}  greenfield={len(sel_g):2d}"
                  f"  coverage={summary['coverage_share_of_total']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "budgets": BUDGETS,
        "radii":   radii,
        "unit":    unit,
        "speed_kmh": SPEED_KMH if mode == "time" else None,
        "total_demand": total_demand,
        "existing_advanced": existing_advanced,
        "scenarios": scenarios,
    }
    with open(json_out, "w") as f:
        json.dump(payload, f)
    pd.DataFrame(flat_rows).to_csv(csv_out, index=False)

    print(f"\nSaved {json_out}")
    print(f"Saved {csv_out}")
    print(f"Scenarios: {len(BUDGETS)} budgets x {len(radii)} {unit} = "
          f"{len(BUDGETS) * len(radii)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["km", "time", "roadtime"], default="km",
                        help="km = distance radius; time = travel-time radius (one avg speed); "
                             "roadtime = travel-time radius from differentiated per-road-type speeds.")
    parser.add_argument("--refine", action="store_true",
                        help="Use the full two-stage (max coverage -> min distance) solve.")
    args = parser.parse_args()
    main(mode=args.mode, refine=args.refine)
