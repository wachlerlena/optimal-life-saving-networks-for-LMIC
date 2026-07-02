"""
Baseline upgrade-only facility location model.

Selects which existing BASIC hospitals to upgrade to advanced (thrombectomy)
capability. Existing ADVANCED hospitals are fixed open in every scenario.
Greenfield builds are NOT available here — this is the upgrade-only baseline
that the combined upgrade-or-greenfield model is compared against.

Coverage share is normalised against TOTAL Vietnam population demand
(97,338,396), the same denominator used to make the baseline and the combined
model directly comparable. Because upgrades can only reach the 11,769 population
points that have a recorded road distance to an existing hospital, the baseline
saturates well below 100% — that gap is exactly what greenfield is meant to fill.

Data files (all in data/)
--------------------------------------
existing_hospital_distances.pkl   pop -> 130 existing hospitals  (15,294 rows, 11,769 pop)
all_hospitals.pkl                 facility coordinates / IDs
stroke-facs-100-en.csv            advanced/basic classification for existing hospitals
population.pkl                    406,784 population points with demand weights

Run a single scenario:
    python baseline_upgrade_model.py
Run the full scenario grid (budgets 0-20, radii 5-300 km):
    python baseline_upgrade_model.py --scenarios
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXISTING_DIST_PATH = DATA_DIR / "existing_hospital_distances.pkl"
ALL_HOSPITALS_PATH = DATA_DIR / "all_hospitals.pkl"
STROKE_FACS_PATH   = DATA_DIR / "stroke-facs-100-en.csv"
POPULATION_PATH    = DATA_DIR / "population.pkl"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "baseline_upgrade_results"

N_EXISTING        = 130          # existing hospital IDs: 0 .. 129
BUDGET            = 5            # at most B upgrades (EXACT_BUDGET=False)
MODE              = "coverage"   # "coverage" or "pmedian"
SERVICE_RADIUS_KM = 300.0
DEMAND_COL        = "household_count"
EXACT_BUDGET      = False        # True = exactly B upgrades, False = at most B

TIME_LIMIT        = 300          # seconds
MIP_REL_GAP       = 0.001


# =============================================================================
# HELPERS
# =============================================================================

def load_pickle(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(path, "rb") as f:
            return pickle.load(f)


def _is_advanced(value) -> bool:
    """True only for Vietnamese 'Có' advanced-capability entries."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if "không" in text or "khong" in text:
        return False
    return "có" in text or "co" in text


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class PreparedData:
    hospitals:         pd.DataFrame
    population:        pd.DataFrame
    pairs:             pd.DataFrame
    pop_ids:           np.ndarray
    basic_hosp_ids:    np.ndarray
    advanced_hosp_ids: np.ndarray
    basic_id_to_urow:  Dict[int, int]
    pop_to_row:        Dict[int, int]
    total_demand:      float       # TOTAL Vietnam demand (fixed denominator)


def load_and_prepare_data(service_radius_km: float = SERVICE_RADIUS_KM,
                          demand_col: str = DEMAND_COL) -> PreparedData:
    hospitals  = load_pickle(ALL_HOSPITALS_PATH).reset_index(drop=True)
    population = load_pickle(POPULATION_PATH).copy()
    pairs      = load_pickle(EXISTING_DIST_PATH).copy()

    hospitals["hosp_id"] = hospitals["ID"].astype(int)
    # Upgrade model operates on existing hospitals only (IDs 0 .. 129).
    hospitals = hospitals[hospitals["hosp_id"] < N_EXISTING].copy()

    population["pop_id"] = population["ID"].astype(int)
    population["demand"] = population[demand_col].fillna(0).astype(float)

    # TOTAL Vietnam demand: the fixed denominator for coverage share. Computed
    # over the full population (not just points inside the distance matrix), so
    # it is identical across every budget/radius scenario and comparable to the
    # combined model when that is aligned to the same denominator.
    total_demand = float(population["demand"].sum())

    population = population[population["demand"] > 0].copy()

    pairs["pop_id"]      = pairs["pop_id"].astype(int)
    pairs["hosp_id"]     = pairs["hosp_id"].astype(int)
    pairs["distance_km"] = pairs["total_dist"].astype(float)
    pairs = (
        pairs[pairs["distance_km"] <= service_radius_km]
        .sort_values("distance_km")
        .drop_duplicates(subset=["pop_id", "hosp_id"], keep="first")
        .reset_index(drop=True)
    )

    demand_map      = population.set_index("pop_id")["demand"]
    pairs["demand"] = pairs["pop_id"].map(demand_map)
    pairs = pairs.dropna(subset=["demand"]).copy()
    pairs["demand"] = pairs["demand"].astype(float)

    # Advanced / basic classification from stroke-facs CSV (TT is 1-based; the
    # pkl ID is 0-based, so hosp_id = TT - 1).
    meta = pd.read_csv(STROKE_FACS_PATH)
    meta["hosp_id"] = meta["TT"].astype(int) - 1
    adv_col = next(
        (c for c in meta.columns if "can" in c.lower() and "thiệp" in c.lower()),
        None,
    )
    if adv_col is None:
        raise ValueError(
            f"Cannot find advanced-capability column in {STROKE_FACS_PATH}.\n"
            f"Columns available: {meta.columns.tolist()}"
        )
    meta["already_advanced"] = meta[adv_col].apply(_is_advanced)
    hospitals = hospitals.merge(
        meta[["hosp_id", "already_advanced"]], on="hosp_id", how="left"
    )
    hospitals["already_advanced"] = hospitals["already_advanced"].fillna(False).astype(bool)
    if "Name_English" not in hospitals.columns and "Name_English" in meta.columns:
        hospitals = hospitals.merge(meta[["hosp_id", "Name_English"]], on="hosp_id", how="left")

    valid_hospitals = set(hospitals["hosp_id"])
    pairs = pairs[pairs["hosp_id"].isin(valid_hospitals)].copy()

    advanced_hosp_ids = np.sort(hospitals.loc[hospitals["already_advanced"],  "hosp_id"].astype(int).unique())
    basic_hosp_ids    = np.sort(hospitals.loc[~hospitals["already_advanced"], "hosp_id"].astype(int).unique())
    basic_id_to_urow  = {int(h): r for r, h in enumerate(basic_hosp_ids)}

    pop_ids    = np.sort(pairs["pop_id"].unique())
    pop_to_row = {int(p): r for r, p in enumerate(pop_ids)}

    pairs["pop_row"]           = pairs["pop_id"].map(pop_to_row).astype(int)
    pairs["is_basic_hospital"] = pairs["hosp_id"].isin(basic_id_to_urow)
    pairs["u_row"]             = pairs["hosp_id"].map(basic_id_to_urow)

    return PreparedData(hospitals, population, pairs, pop_ids,
                        basic_hosp_ids, advanced_hosp_ids, basic_id_to_urow,
                        pop_to_row, total_demand)


# =============================================================================
# OPTIMISATION MODEL
# =============================================================================

def build_matrices(data: PreparedData, include_z: bool):
    pairs  = data.pairs.reset_index(drop=True).copy()
    n_x    = len(pairs)
    n_u    = len(data.basic_hosp_ids)
    n_z    = len(data.pop_ids) if include_z else 0
    n_vars = n_x + n_u + n_z

    u_start = n_x
    z_start = n_x + n_u

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0

    # Assignment / coverage link: sum_j x_ij (- z_i) = 0 [coverage] or = 1 [pmedian]
    for pop_row, idx in pairs.groupby("pop_row").groups.items():
        for k in idx:
            rows.append(r); cols.append(int(k)); vals.append(1.0)
        if include_z:
            rows.append(r); cols.append(z_start + int(pop_row)); vals.append(-1.0)
            lb.append(0.0); ub.append(0.0)
        else:
            lb.append(1.0); ub.append(1.0)
        r += 1

    # Linking: x_ij <= u_j for basic hospitals only (advanced are always open)
    for k, row in pairs.iterrows():
        if bool(row["is_basic_hospital"]):
            rows += [r, r]; cols += [int(k), u_start + int(row["u_row"])]; vals += [1.0, -1.0]
            lb.append(-np.inf); ub.append(0.0)
            r += 1

    A = coo_matrix((vals, (rows, cols)), shape=(r, n_vars)).tocsr()
    return A, np.array(lb), np.array(ub), pairs, n_x, n_u, n_z, u_start, z_start


def solve_model(data: PreparedData, budget: int, mode: str = "coverage",
                exact_budget: bool = False, time_limit: float = TIME_LIMIT,
                mip_rel_gap: float = MIP_REL_GAP):
    include_z = mode == "coverage"
    A, lb, ub, pairs, n_x, n_u, n_z, u_start, z_start = build_matrices(data, include_z)
    n_vars = n_x + n_u + n_z

    # Budget constraint over u variables
    Bmat   = coo_matrix(([1.0] * n_u, ([0] * n_u, list(range(u_start, u_start + n_u)))), shape=(1, n_vars)).tocsr()
    A_all  = vstack([A, Bmat]).tocsr()
    lb_all = np.concatenate([lb, [float(budget) if exact_budget else -np.inf]])
    ub_all = np.concatenate([ub, [float(budget)]])

    bounds      = Bounds(np.zeros(n_vars), np.ones(n_vars))
    integrality = np.ones(n_vars, dtype=int)
    constraints = LinearConstraint(A_all, lb_all, ub_all)
    options     = {"disp": False, "mip_rel_gap": mip_rel_gap, "time_limit": time_limit}
    pair_wd     = (pairs["demand"].to_numpy() * pairs["distance_km"].to_numpy()).astype(float)

    if mode == "pmedian":
        c = np.zeros(n_vars); c[:n_x] = pair_wd
        return milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options=options), None, pairs, n_x, n_u, u_start

    # Stage 1: maximise covered demand
    pop_demand = data.population.set_index("pop_id").loc[data.pop_ids, "demand"].to_numpy().astype(float)
    c1 = np.zeros(n_vars); c1[z_start: z_start + n_z] = -pop_demand
    res1 = milp(c=c1, integrality=integrality, bounds=bounds, constraints=constraints, options=options)
    if not res1.success:
        return res1, None, pairs, n_x, n_u, u_start

    best_coverage = -float(res1.fun)

    # Stage 2: minimise weighted distance while preserving best coverage
    Ccov  = coo_matrix((pop_demand, ([0] * n_z, list(range(z_start, z_start + n_z)))), shape=(1, n_vars)).tocsr()
    A_s2  = vstack([A_all, Ccov]).tocsr()
    lb_s2 = np.concatenate([lb_all, [best_coverage - 0.5]])   # 0.5 tol: safe for integer demands
    ub_s2 = np.concatenate([ub_all, [np.inf]])
    c2    = np.zeros(n_vars); c2[:n_x] = pair_wd
    res2  = milp(c=c2, integrality=integrality, bounds=bounds,
                 constraints=LinearConstraint(A_s2, lb_s2, ub_s2), options=options)
    return res2, best_coverage, pairs, n_x, n_u, u_start


# =============================================================================
# SOLUTION EXTRACTION
# =============================================================================

def extract_solution(data: PreparedData, res, pairs, n_x, n_u, u_start, mode, best_coverage=None):
    x = res.x[:n_x]
    u = res.x[u_start: u_start + n_u]

    selected_upgrade_ids = [int(data.basic_hosp_ids[j]) for j, v in enumerate(u) if v > 0.5]
    selected_upgrades    = data.hospitals[data.hospitals["hosp_id"].isin(selected_upgrade_ids)].copy()
    selected_upgrades["selected_for_upgrade"] = True

    pairs = pairs.copy()
    pairs["assigned"] = x > 0.5
    assignments = pairs[pairs["assigned"]][["pop_id", "hosp_id", "demand", "distance_km"]].copy()

    existing_advanced = data.hospitals[data.hospitals["already_advanced"]].copy()
    all_open_advanced = (
        pd.concat([existing_advanced, selected_upgrades], ignore_index=True)
        .drop_duplicates(subset=["hosp_id"])
    )

    covered_demand  = float(assignments["demand"].sum())
    matrix_demand   = float(data.population.set_index("pop_id").loc[data.pop_ids, "demand"].sum())
    total_demand    = float(data.total_demand)
    weighted_dist   = float((assignments["demand"] * assignments["distance_km"]).sum())
    avg_distance    = weighted_dist / covered_demand if covered_demand > 0 else np.nan

    summary = {
        "mode":                             mode,
        "existing_advanced_fixed_open":     len(existing_advanced),
        "selected_new_upgrades":            len(selected_upgrades),
        "all_open_advanced_after_solution": len(all_open_advanced),
        "covered_demand":                   covered_demand,
        # Headline denominator: TOTAL Vietnam demand (comparable across models).
        "total_demand":                     total_demand,
        "coverage_share_of_total":          covered_demand / total_demand if total_demand else np.nan,
        # Reference only: demand reachable inside the existing-hospital matrix.
        "matrix_reachable_demand":          matrix_demand,
        "coverage_share_of_matrix":         covered_demand / matrix_demand if matrix_demand else np.nan,
        "weighted_total_distance":          weighted_dist,
        "average_distance_km":              avg_distance,
        "solver_message":                   str(res.message),
    }
    if best_coverage is not None:
        summary["stage1_best_coverage"] = float(best_coverage)

    return selected_upgrades, existing_advanced, all_open_advanced, assignments, summary


# =============================================================================
# RUNNERS
# =============================================================================

def run_single():
    data = load_and_prepare_data(service_radius_km=SERVICE_RADIUS_KM)

    print(f"Existing hospitals:                   {len(data.hospitals):,}")
    print(f"Existing advanced (fixed open):       {len(data.advanced_hosp_ids):,}")
    print(f"Basic hospitals (upgrade candidates): {len(data.basic_hosp_ids):,}")
    print(f"Matrix-reachable pop points:          {len(data.pop_ids):,}")
    print(f"TOTAL Vietnam demand (denominator):   {data.total_demand:,.0f}")

    res, best_coverage, pairs, n_x, n_u, u_start = solve_model(
        data, budget=BUDGET, mode=MODE, exact_budget=EXACT_BUDGET,
        time_limit=TIME_LIMIT, mip_rel_gap=MIP_REL_GAP,
    )
    print("\nSolver status:", res.message, "| success:", res.success)

    sel_up, exist_adv, all_adv, assignments, summary = extract_solution(
        data, res, pairs, n_x, n_u, u_start, MODE, best_coverage
    )

    print("\n=== SOLUTION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = OUTPUT_DIR / f"upgrade_budget{BUDGET}_{MODE}_radius{int(SERVICE_RADIUS_KM)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["hosp_id", "Name_English", "Latitude", "Longitude", "already_advanced"] if c in data.hospitals.columns]
    sel_up[cols].to_csv(out_dir / "selected_new_upgrades.csv", index=False)
    exist_adv[cols].to_csv(out_dir / "existing_advanced_hospitals.csv", index=False)
    all_adv[cols].to_csv(out_dir / "all_open_advanced_after_solution.csv", index=False)
    assignments.to_csv(out_dir / "assignments.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved to: {out_dir}")


def run_scenarios():
    BUDGETS = list(range(0, 21))
    RADII   = [5, 10, 20, 30, 40, 50, 100, 150, 200, 300]

    rows = []
    for R in RADII:
        print(f"\nRadius {R} km")
        data_R = load_and_prepare_data(service_radius_km=R)
        for B in BUDGETS:
            res, best_cov, pairs, n_x, n_u, u_start = solve_model(
                data_R, budget=B, mode="coverage", exact_budget=False,
                time_limit=TIME_LIMIT, mip_rel_gap=MIP_REL_GAP,
            )
            if res.success:
                _, _, _, _, summ = extract_solution(data_R, res, pairs, n_x, n_u, u_start, "coverage", best_cov)
                summ["budget"] = B
                summ["service_radius_km"] = R
                rows.append(summ)
                print(f"  B={B:2d}  upgrades={summ['selected_new_upgrades']:2d}"
                      f"  coverage_total={summ['coverage_share_of_total']:.4f}")
            else:
                rows.append({"budget": B, "service_radius_km": R, "solver_message": str(res.message)})
                print(f"  B={B:2d}  FAILED: {res.message}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "upgrade_only_scenario_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved scenario grid to: {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", action="store_true",
                        help="Run the full budget x radius scenario grid.")
    args = parser.parse_args()
    if args.scenarios:
        run_scenarios()
    else:
        run_single()
