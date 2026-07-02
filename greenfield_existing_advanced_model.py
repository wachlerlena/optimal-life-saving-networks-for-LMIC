"""
Combined upgrade-or-greenfield facility location model.

Decision per budget slot: upgrade an existing basic hospital  OR  build a new
greenfield site — the model picks the optimal mix automatically.

Data files (all in data/)
-------------------------------------
existing_hospital_distances.pkl   pop → 130 existing hospitals  (15,294 rows)
greenfield_distances.pkl          pop → 3,555 greenfield sites  (682,864 rows)
all_hospitals.pkl                 all 3,685 facilities
new_hospitals.pkl                 3,555 greenfield candidates
stroke-facs-100-en.csv            advanced/basic classification for existing hospitals
population.pkl                    406,784 population points with demand weights

Run once to create the two distance files from the unified matrix:
    python greenfield_existing_advanced_model.py --split-only
Then run normally:
    python greenfield_existing_advanced_model.py
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXISTING_DIST_PATH  = DATA_DIR / "existing_hospital_distances.pkl"
GREENFIELD_DIST_PATH = DATA_DIR / "greenfield_distances.pkl"
UNIFIED_DIST_PATH   = DATA_DIR / "distances_osm_max_300km.pkl"   # source for splitting
ALL_HOSPITALS_PATH  = DATA_DIR / "all_hospitals.pkl"
NEW_HOSPITALS_PATH  = DATA_DIR / "new_hospitals.pkl"
STROKE_FACS_PATH    = DATA_DIR / "stroke-facs-100-en.csv"
POPULATION_PATH     = DATA_DIR / "population.pkl"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "combined_model_results"

N_EXISTING         = 130       # existing hospital IDs: 0 … 129
BUDGET             = 10        # total slots (upgrades + builds combined)
SERVICE_RADIUS_KM  = 300.0
DEMAND_COL         = "household_count"

SOLVER_NAME        = "highs"
TIME_LIMIT_SECONDS = 600
MIP_GAP            = 0.001


# =============================================================================
# HELPERS
# =============================================================================

def load_pickle(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with open(path, "rb") as f:
            return pickle.load(f)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _is_advanced(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if "không" in text or "khong" in text:
        return False
    return "có" in text or "co" in text


# =============================================================================
# SPLIT UNIFIED DISTANCE FILE (run once)
# =============================================================================

def split_distance_file():
    """
    Splits distances_osm_max_300km.pkl into two separate files:
      - existing_hospital_distances.pkl  (hosp_id 0-129)
      - greenfield_distances.pkl         (hosp_id 130+)
    Safe to re-run; will overwrite existing files.
    """
    print("Loading unified distance matrix …")
    unified = load_pickle(UNIFIED_DIST_PATH)
    unified["pop_id"]    = unified["pop_id"].astype(int)
    unified["hosp_id"]   = unified["hosp_id"].astype(int)
    unified["total_dist"] = unified["total_dist"].astype(float)

    existing   = unified[unified["hosp_id"] < N_EXISTING][["pop_id", "hosp_id", "total_dist"]].reset_index(drop=True)
    greenfield = unified[unified["hosp_id"] >= N_EXISTING][["pop_id", "hosp_id", "total_dist"]].reset_index(drop=True)

    save_pickle(existing,   EXISTING_DIST_PATH)
    save_pickle(greenfield, GREENFIELD_DIST_PATH)

    print(f"Saved existing_hospital_distances.pkl  — {len(existing):,} rows, "
          f"{existing['hosp_id'].nunique()} hospitals, "
          f"{existing['pop_id'].nunique():,} pop points")
    print(f"Saved greenfield_distances.pkl         — {len(greenfield):,} rows, "
          f"{greenfield['hosp_id'].nunique()} candidates, "
          f"{greenfield['pop_id'].nunique():,} pop points")


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    if not EXISTING_DIST_PATH.exists() or not GREENFIELD_DIST_PATH.exists():
        print("Distance files not found — splitting unified matrix first …")
        split_distance_file()

    existing_pairs   = load_pickle(EXISTING_DIST_PATH)
    greenfield_pairs = load_pickle(GREENFIELD_DIST_PATH)
    all_hosp         = load_pickle(ALL_HOSPITALS_PATH).reset_index(drop=True)
    new_hosp         = load_pickle(NEW_HOSPITALS_PATH).reset_index(drop=True)

    # All pop points that appear in either distance file
    pop_ids_in_scope = (
        set(existing_pairs["pop_id"].astype(int))
        | set(greenfield_pairs["pop_id"].astype(int))
    )
    pop_df = load_pickle(POPULATION_PATH).copy()
    pop_df["pop_id"] = pop_df["ID"].astype(int)
    pop_df["demand"] = pop_df[DEMAND_COL].fillna(0).astype(float)
    # TOTAL Vietnam demand: fixed headline denominator, shared with the baseline
    # upgrade model so coverage shares are directly comparable across both models.
    total_vietnam_demand = float(pop_df["demand"].sum())
    pop = pop_df[pop_df["pop_id"].isin(pop_ids_in_scope)][["pop_id", "demand"]].copy()

    # Classify existing hospitals as advanced / basic via stroke-facs CSV
    stroke_facs = pd.read_csv(STROKE_FACS_PATH)
    stroke_facs["hosp_id"] = stroke_facs["TT"].astype(int) - 1
    adv_col = next(
        (c for c in stroke_facs.columns if "can" in c.lower() and "thiệp" in c.lower()),
        None,
    )
    if adv_col is None:
        raise ValueError(
            f"Cannot find advanced-capability column in {STROKE_FACS_PATH}.\n"
            f"Columns available: {stroke_facs.columns.tolist()}"
        )
    stroke_facs["already_advanced"] = stroke_facs[adv_col].apply(_is_advanced)

    existing_hosp = all_hosp[all_hosp["ID"] < N_EXISTING].copy()
    existing_hosp["hosp_id"] = existing_hosp["ID"].astype(int)
    existing_hosp = existing_hosp.merge(
        stroke_facs[["hosp_id", "already_advanced"]], on="hosp_id", how="left"
    )
    existing_hosp["already_advanced"] = existing_hosp["already_advanced"].fillna(False).astype(bool)

    advanced_ids   = set(existing_hosp.loc[existing_hosp["already_advanced"],  "hosp_id"].astype(int))
    basic_ids      = set(existing_hosp.loc[~existing_hosp["already_advanced"], "hosp_id"].astype(int))

    new_hosp["hosp_id"] = new_hosp["Cluster_ID"].astype(int)
    greenfield_ids = set(new_hosp["hosp_id"].astype(int))

    print(f"Existing advanced hospitals (fixed open): {len(advanced_ids)}")
    print(f"Basic hospitals (upgrade candidates):     {len(basic_ids)}")
    print(f"Greenfield candidates:                    {len(greenfield_ids):,}")
    print(f"Population points in scope:               {len(pop):,}")
    print(f"TOTAL Vietnam demand (denominator):       {total_vietnam_demand:,.0f}")

    return pop, existing_pairs, greenfield_pairs, advanced_ids, basic_ids, greenfield_ids, all_hosp, new_hosp, total_vietnam_demand


# =============================================================================
# PRE-PROCESSING
# =============================================================================

def preprocess(
    pop: pd.DataFrame,
    existing_pairs: pd.DataFrame,
    greenfield_pairs: pd.DataFrame,
    advanced_ids: set,
    basic_ids: set,
    greenfield_ids: set,
    service_radius_km: float,
) -> Tuple[set, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    already_covered : pop_ids covered by existing advanced hospitals within radius
    pop_weight      : dict  pop_id → demand  (full scope — hospital + greenfield)
    basic_pairs_r   : basic-hospital pairs within radius
    gf_pairs_r      : greenfield pairs within radius (full 384K pop scope)
    """
    # Fixed coverage: existing advanced hospitals only
    adv_pairs = existing_pairs[
        existing_pairs["hosp_id"].isin(advanced_ids)
        & (existing_pairs["total_dist"] <= service_radius_km)
    ]
    already_covered = set(adv_pairs["pop_id"].astype(int))

    # Upgrade pairs: basic hospitals within radius (hospital pop scope only)
    basic_pairs_r = existing_pairs[
        existing_pairs["hosp_id"].isin(basic_ids)
        & (existing_pairs["total_dist"] <= service_radius_km)
    ].copy()

    # Greenfield pairs: full 384K population scope, within radius
    gf_pairs_r = greenfield_pairs[
        greenfield_pairs["hosp_id"].isin(greenfield_ids)
        & (greenfield_pairs["total_dist"] <= service_radius_km)
    ].copy()

    pop_weight = pop.set_index("pop_id")["demand"].to_dict()

    return already_covered, pop_weight, basic_pairs_r, gf_pairs_r


# =============================================================================
# MIP MODEL
# =============================================================================

def solve_combined(
    pop_weight: Dict[int, float],
    already_covered: set,
    basic_pairs: pd.DataFrame,
    gf_pairs: pd.DataFrame,
    budget: int,
    solver_name: str = "highs",
    time_limit: int = 600,
    mip_gap: float = 0.001,
):
    """
    Variables:  u_j ∈ {0,1} upgrade basic hospital j
                y_g ∈ {0,1} build greenfield site g
                z_i ∈ {0,1} newly covered pop point i

    Budget:     sum(u_j) + sum(y_g) <= budget

    Coverage:   z_i <= sum(u_j : j covers i) + sum(y_g : g covers i)
                       for every i not already covered by existing advanced hospitals

    Objective:  maximise  fixed_already_covered_demand + sum(p_i * z_i)
    """
    fixed_demand  = sum(pop_weight.get(p, 0) for p in already_covered)
    uncovered_set = set(p for p in pop_weight if p not in already_covered)

    basic_unc = basic_pairs[basic_pairs["pop_id"].isin(uncovered_set)]
    gf_unc    = gf_pairs[gf_pairs["pop_id"].isin(uncovered_set)]

    active_basic = sorted(basic_unc["hosp_id"].astype(int).unique())
    active_gf    = sorted(gf_unc["hosp_id"].astype(int).unique())

    reachable = (
        set(basic_unc["pop_id"].astype(int))
        | set(gf_unc["pop_id"].astype(int))
    )
    model_pop = sorted(reachable)

    basic_coverers: Dict[int, List[int]] = (
        basic_unc.groupby("pop_id")["hosp_id"]
        .apply(lambda s: sorted(s.astype(int).unique().tolist()))
        .to_dict()
    )
    gf_coverers: Dict[int, List[int]] = (
        gf_unc.groupby("pop_id")["hosp_id"]
        .apply(lambda s: sorted(s.astype(int).unique().tolist()))
        .to_dict()
    )

    total_model_demand = sum(pop_weight.values())

    print(f"\n{'='*60}")
    print(f"Budget:                          {budget}")
    print(f"Already covered (fixed):         {len(already_covered):,} pop pts  "
          f"({fixed_demand / total_model_demand * 100:.1f}% of scope)")
    print(f"Uncovered reachable pop points:  {len(model_pop):,}")
    print(f"  via upgrade candidates:        {len(set(basic_unc['pop_id'])):,}")
    print(f"  via greenfield builds:         {len(set(gf_unc['pop_id'])):,}")
    print(f"Active upgrade candidates:       {len(active_basic)}")
    print(f"Active greenfield candidates:    {len(active_gf):,}")
    print(f"{'='*60}\n")

    mdl = pyo.ConcreteModel()
    mdl.J = pyo.Set(initialize=active_basic)
    mdl.G = pyo.Set(initialize=active_gf)
    mdl.I = pyo.Set(initialize=model_pop)

    mdl.u = pyo.Var(mdl.J, domain=pyo.Binary)
    mdl.y = pyo.Var(mdl.G, domain=pyo.Binary)
    # z_i relaxed to continuous [0,1]: in a max-coverage objective each z_i is
    # pushed to its upper bound (= facility coverage) automatically, so the
    # optimum is identical to the binary formulation but with ~384K fewer
    # integer variables — only the facility choices (u, y) stay binary.
    mdl.z = pyo.Var(mdl.I, domain=pyo.UnitInterval)

    mdl.budget = pyo.Constraint(
        expr=sum(mdl.u[j] for j in mdl.J) + sum(mdl.y[g] for g in mdl.G) <= int(budget)
    )

    def coverage_rule(m, i):
        upg  = sum(m.u[j] for j in basic_coverers.get(i, []) if j in active_basic)
        build = sum(m.y[g] for g in gf_coverers.get(i, [])   if g in active_gf)
        return m.z[i] <= upg + build

    mdl.coverage = pyo.Constraint(mdl.I, rule=coverage_rule)

    mdl.obj = pyo.Objective(
        expr=fixed_demand + sum(float(pop_weight.get(i, 0)) * mdl.z[i] for i in mdl.I),
        sense=pyo.maximize,
    )

    solver = pyo.SolverFactory(solver_name)
    # Option names/channels differ between the legacy CLI 'highs' interface and
    # the highspy/APPSI interface (which is what's active when the 'highs'
    # binary isn't on PATH). Set them on every channel so the limits actually
    # bind regardless of which interface Pyomo resolves to.
    for opt_obj in (getattr(solver, "options", None),
                    getattr(solver, "highs_options", None)):
        if opt_obj is not None:
            try:
                opt_obj["time_limit"]  = time_limit
                opt_obj["mip_rel_gap"] = mip_gap
            except Exception:
                pass
    cfg = getattr(solver, "config", None)
    if cfg is not None:
        for attr, val in (("time_limit", time_limit), ("mip_gap", mip_gap)):
            try:
                setattr(cfg, attr, val)
            except Exception:
                pass

    result = solver.solve(mdl, tee=True)

    sel_upgrades   = [int(j) for j in mdl.J if pyo.value(mdl.u[j]) is not None and pyo.value(mdl.u[j]) > 0.5]
    sel_greenfield = [int(g) for g in mdl.G if pyo.value(mdl.y[g]) is not None and pyo.value(mdl.y[g]) > 0.5]
    newly_covered  = [int(i) for i in mdl.I if pyo.value(mdl.z[i]) is not None and pyo.value(mdl.z[i]) > 0.5]
    obj_val        = float(pyo.value(mdl.obj))

    return sel_upgrades, sel_greenfield, newly_covered, fixed_demand, obj_val, result


# =============================================================================
# STAGE 2 — MINIMISE WEIGHTED DISTANCE AMONG MAX-COVERAGE SOLUTIONS
# =============================================================================

def refine_min_distance(
    pop_weight: Dict[int, float],
    already_covered: set,
    basic_pairs: pd.DataFrame,
    gf_pairs: pd.DataFrame,
    budget: int,
    coverage_floor: float,
    time_limit: int = 600,
    mip_gap: float = 0.001,
    cov_tol: float = 0.5,
):
    """
    Lexicographic stage 2 (mirrors the baseline upgrade model, extended with
    greenfield): among every selection of <= budget new facilities that keeps
    the stage-1 maximum newly-covered demand, pick the one minimising
    demand-weighted ROAD distance for the newly covered population.

    Only the facility choices (u upgrades, y greenfield) are binary; the
    assignment x is continuous — for fixed u, y the assignment LP is naturally
    integral, so the relaxation is exact while staying fast.

    coverage_floor : stage-1 maximum newly-covered demand (D^new).
    cov_tol        : 0.5 is safe because demands are integer-valued.
    """
    uncovered = set(p for p in pop_weight if p not in already_covered)

    b = basic_pairs[basic_pairs["pop_id"].isin(uncovered)][["pop_id", "hosp_id", "total_dist"]].copy()
    g = gf_pairs[gf_pairs["pop_id"].isin(uncovered)][["pop_id", "hosp_id", "total_dist"]].copy()
    b["ftype"] = "upgrade"
    g["ftype"] = "greenfield"
    pairs = pd.concat([b, g], ignore_index=True)
    pairs["demand"] = pairs["pop_id"].map(pop_weight).fillna(0.0).astype(float)
    pairs = pairs[pairs["demand"] > 0].reset_index(drop=True)

    if len(pairs) == 0:
        empty = pairs.assign(distance_km=pairs.get("total_dist", []))
        return [], [], empty, 0.0

    pop_ids = np.sort(pairs["pop_id"].unique())
    pop_row = {int(p): i for i, p in enumerate(pop_ids)}
    upg_ids = np.sort(pairs.loc[pairs.ftype == "upgrade",    "hosp_id"].unique())
    gf_ids  = np.sort(pairs.loc[pairs.ftype == "greenfield", "hosp_id"].unique())
    u_col = {int(h): i for i, h in enumerate(upg_ids)}
    g_col = {int(h): i for i, h in enumerate(gf_ids)}

    n_pop, n_u, n_g = len(pop_ids), len(upg_ids), len(gf_ids)
    n_x = len(pairs)
    u_start, g_start = n_x, n_x + n_u
    n_vars = n_x + n_u + n_g

    prow    = pairs["pop_id"].map(pop_row).to_numpy()
    is_up   = (pairs.ftype == "upgrade").to_numpy()
    var_col = np.where(
        is_up,
        u_start + pairs["hosp_id"].map(u_col).to_numpy(dtype=float),
        g_start + pairs["hosp_id"].map(g_col).to_numpy(dtype=float),
    ).astype(int)
    xidx   = np.arange(n_x)
    demand = pairs["demand"].to_numpy()
    dist   = pairs["total_dist"].to_numpy()

    # --- constraints (vectorised sparse assembly) ---
    # (1) assignment: sum_f x_{i,f} <= 1               rows 0 .. n_pop-1
    # (2) linking:    x_{i,f} - var_f <= 0             rows n_pop .. n_pop+n_x-1
    # (3) budget:     sum u + sum y <= budget          row  n_pop+n_x
    # (4) floor:      sum_f demand*x >= floor - tol    row  n_pop+n_x+1
    link_rows = n_pop + xidx
    rows = np.concatenate([prow, link_rows, link_rows,
                           np.full(n_u + n_g, n_pop + n_x),
                           np.full(n_x, n_pop + n_x + 1)])
    cols = np.concatenate([xidx, xidx, var_col,
                           np.arange(u_start, n_vars),
                           xidx])
    vals = np.concatenate([np.ones(n_x), np.ones(n_x), -np.ones(n_x),
                           np.ones(n_u + n_g),
                           demand])
    n_rows = n_pop + n_x + 2
    A = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_vars)).tocsr()

    con_lb = np.concatenate([np.zeros(n_pop), np.full(n_x, -np.inf),
                             [-np.inf], [coverage_floor - cov_tol]])
    con_ub = np.concatenate([np.ones(n_pop), np.zeros(n_x),
                             [float(budget)], [np.inf]])

    integrality = np.zeros(n_vars, dtype=int)
    integrality[u_start:] = 1
    c = np.zeros(n_vars)
    c[xidx] = demand * dist

    print(f"  [stage 2] min-distance refine: {n_x:,} assignments, "
          f"{n_u + n_g:,} facility binaries, {n_rows:,} constraints …")

    res = milp(
        c=c, integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(A, con_lb, con_ub),
        options={"disp": False, "mip_rel_gap": mip_gap, "time_limit": time_limit},
    )

    if not res.success or res.x is None:
        print(f"  [stage 2] WARNING: {res.message} — falling back to stage-1 selection")
        return None, None, None, None

    xv = res.x
    sel_up = [int(upg_ids[i]) for i in range(n_u) if xv[u_start + i] > 0.5]
    sel_gf = [int(gf_ids[i])  for i in range(n_g) if xv[g_start + i] > 0.5]

    pairs["xval"] = xv[:n_x]
    assigned = (
        pairs[pairs["xval"] > 1e-6]
        .sort_values("xval", ascending=False)
        .groupby("pop_id", as_index=False).first()
        [["pop_id", "hosp_id", "total_dist", "demand", "ftype"]]
        .rename(columns={"total_dist": "distance_km"})
    )
    return sel_up, sel_gf, assigned, float(res.fun)


def advanced_assignment(existing_pairs, advanced_ids, already_covered,
                        service_radius_km, pop_weight) -> pd.DataFrame:
    """Nearest existing-advanced facility (road distance) for each already-covered
    population point — used for the existing-advanced distance channel."""
    adv = existing_pairs[
        existing_pairs["hosp_id"].isin(advanced_ids)
        & existing_pairs["pop_id"].isin(already_covered)
        & (existing_pairs["total_dist"] <= service_radius_km)
    ].copy()
    adv = adv.sort_values("total_dist").groupby("pop_id", as_index=False).first()
    adv["demand"]      = adv["pop_id"].map(pop_weight).fillna(0.0).astype(float)
    adv["distance_km"] = adv["total_dist"].astype(float)
    adv["ftype"]       = "existing_advanced"
    return adv[["pop_id", "hosp_id", "distance_km", "demand", "ftype"]]


def _wavg_distance(df: pd.DataFrame) -> float:
    d = float(df["demand"].sum())
    return float((df["demand"] * df["distance_km"]).sum() / d) if d > 0 else float("nan")


# =============================================================================
# OUTPUTS
# =============================================================================

def build_summary(
    adv_assign, new_assign, sel_upgrades, sel_greenfield,
    budget, service_radius_km, advanced_ids, basic_ids,
    total_vietnam_demand, scope_demand,
) -> dict:
    upg_assign = new_assign[new_assign["ftype"] == "upgrade"]
    gf_assign  = new_assign[new_assign["ftype"] == "greenfield"]

    fixed_demand     = float(adv_assign["demand"].sum())
    upgrade_demand   = float(upg_assign["demand"].sum())
    greenfield_demand = float(gf_assign["demand"].sum())
    newly_demand     = float(new_assign["demand"].sum())
    total_covered    = fixed_demand + newly_demand

    overall = pd.concat(
        [adv_assign[["demand", "distance_km"]], new_assign[["demand", "distance_km"]]],
        ignore_index=True,
    )

    return {
        "budget":                           budget,
        "service_radius_km":                service_radius_km,
        "existing_advanced_fixed_open":     len(advanced_ids),
        "basic_hospitals_available":        len(basic_ids),
        "selected_upgrades":                len(sel_upgrades),
        "selected_greenfield_builds":       len(sel_greenfield),
        "total_slots_used":                 len(sel_upgrades) + len(sel_greenfield),
        # --- coverage by channel ---
        "existing_advanced_covered_demand": fixed_demand,
        "upgrade_covered_demand":           upgrade_demand,
        "greenfield_covered_demand":        greenfield_demand,
        "newly_covered_demand":             newly_demand,
        "total_covered_demand":             total_covered,
        # --- denominators / coverage shares ---
        "total_demand":                     total_vietnam_demand,
        "coverage_share_of_total":          total_covered / total_vietnam_demand if total_vietnam_demand else 0,
        "total_scope_demand":               scope_demand,
        "coverage_share_of_scope":          total_covered / scope_demand if scope_demand else 0,
        # --- demand-weighted ROAD distance by channel (km) ---
        "avg_distance_existing_advanced_km": _wavg_distance(adv_assign),
        "avg_distance_upgrade_km":           _wavg_distance(upg_assign),
        "avg_distance_greenfield_km":        _wavg_distance(gf_assign),
        "avg_distance_newly_covered_km":     _wavg_distance(new_assign),
        "avg_distance_overall_km":           _wavg_distance(overall),
    }


def save_outputs(
    summary, sel_upgrades, sel_greenfield,
    adv_assign, new_assign, all_hosp, new_hosp,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "combined_summary.csv", index=False)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    upg_df = all_hosp[all_hosp["ID"].isin(sel_upgrades)].copy()
    upg_df.to_csv(OUTPUT_DIR / "selected_upgrades.csv", index=False)
    print(f"\nSelected upgrades ({len(upg_df)}):")
    print(upg_df[["ID", "Latitude", "Longitude"]].to_string(index=False))

    gf_df = new_hosp[new_hosp["Cluster_ID"].isin(sel_greenfield)].copy()
    gf_df.to_csv(OUTPUT_DIR / "selected_greenfield.csv", index=False)
    print(f"\nSelected greenfield builds ({len(gf_df)}):")
    if len(gf_df):
        print(gf_df[["Cluster_ID", "Latitude", "Longitude"]].to_string(index=False))

    # Assignments: one row per covered pop point -> serving facility + road distance
    cols = ["pop_id", "hosp_id", "distance_km", "demand", "ftype"]
    assignments = (
        pd.concat([adv_assign[cols], new_assign[cols]], ignore_index=True)
        .rename(columns={"hosp_id": "facility_id", "ftype": "facility_type"})
    )
    assignments.to_csv(OUTPUT_DIR / "assignments.csv", index=False)
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


# =============================================================================
# TWO-STAGE ORCHESTRATION (max coverage -> min weighted distance)
# =============================================================================

def assign_to_selection(basic_r, gf_r, sel_u, sel_g, already_covered, pop_weight):
    """Nearest selected new facility (road distance) per newly covered pop point."""
    unc = set(p for p in pop_weight if p not in already_covered)
    b = basic_r[basic_r["hosp_id"].isin(sel_u) & basic_r["pop_id"].isin(unc)][["pop_id", "hosp_id", "total_dist"]].copy()
    g = gf_r[gf_r["hosp_id"].isin(sel_g) & gf_r["pop_id"].isin(unc)][["pop_id", "hosp_id", "total_dist"]].copy()
    b["ftype"] = "upgrade"
    g["ftype"] = "greenfield"
    cov = pd.concat([b, g], ignore_index=True)
    if len(cov):
        cov = cov.sort_values("total_dist").groupby("pop_id", as_index=False).first()
    cov["demand"]      = cov["pop_id"].map(pop_weight).fillna(0.0).astype(float)
    cov = cov.rename(columns={"total_dist": "distance_km"})
    return cov[["pop_id", "hosp_id", "distance_km", "demand", "ftype"]]


def solve_two_stage(pop_weight, already_covered, basic_r, gf_r,
                    existing_pairs, advanced_ids, budget, service_radius_km):
    """
    Lexicographic solve:
      Stage 1 — maximise newly covered demand (Pyomo/HiGHS coverage model).
      Stage 2 — minimise demand-weighted road distance among all selections
                that preserve the stage-1 coverage (scipy/HiGHS assignment).
    Returns selected upgrades, selected greenfield, and the per-point
    assignment frames for the existing-advanced and the newly covered channels.
    """
    sel_u1, sel_g1, _newly1, fixed_d, obj1, _ = solve_combined(
        pop_weight, already_covered, basic_r, gf_r,
        budget=budget, solver_name=SOLVER_NAME,
        time_limit=TIME_LIMIT_SECONDS, mip_gap=MIP_GAP,
    )
    coverage_floor = obj1 - fixed_d   # stage-1 maximum newly covered demand

    sel_u, sel_g, new_assign, _wd = refine_min_distance(
        pop_weight, already_covered, basic_r, gf_r,
        budget=budget, coverage_floor=coverage_floor,
        time_limit=TIME_LIMIT_SECONDS, mip_gap=MIP_GAP,
    )
    if sel_u is None:   # stage-2 failed: keep stage-1 selection, assign nearest
        sel_u, sel_g = sel_u1, sel_g1
        new_assign = assign_to_selection(basic_r, gf_r, sel_u, sel_g, already_covered, pop_weight)

    adv_assign = advanced_assignment(
        existing_pairs, advanced_ids, already_covered, service_radius_km, pop_weight
    )
    return sel_u, sel_g, adv_assign, new_assign


# =============================================================================
# SCENARIO SWEEP
# =============================================================================

def run_scenarios(
    pop, existing_pairs, greenfield_pairs,
    advanced_ids, basic_ids, greenfield_ids,
    total_vietnam_demand,
    budgets: List[int],
    radii: List[float],
) -> pd.DataFrame:
    rows = []
    for R in radii:
        print(f"\n{'#'*60}\n# Radius = {R} km\n{'#'*60}")
        already_covered, pop_weight, basic_r, gf_r = preprocess(
            pop, existing_pairs, greenfield_pairs,
            advanced_ids, basic_ids, greenfield_ids, R,
        )
        scope_demand = sum(pop_weight.values())
        for B in budgets:
            sel_u, sel_g, adv_a, new_a = solve_two_stage(
                pop_weight, already_covered, basic_r, gf_r,
                existing_pairs, advanced_ids, B, R,
            )
            row = build_summary(
                adv_a, new_a, sel_u, sel_g, B, R,
                advanced_ids, basic_ids, total_vietnam_demand, scope_demand,
            )
            rows.append(row)
            print(f"  B={B:2d}  upgrades={len(sel_u):2d}  greenfield={len(sel_g):2d}"
                  f"  coverage={row['coverage_share_of_total']:.3f}"
                  f"  avg_dist={row['avg_distance_newly_covered_km']:.1f}km")
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    pop, existing_pairs, greenfield_pairs, advanced_ids, basic_ids, greenfield_ids, all_hosp, new_hosp, total_vietnam_demand = load_data()

    already_covered, pop_weight, basic_pairs_r, gf_pairs_r = preprocess(
        pop, existing_pairs, greenfield_pairs,
        advanced_ids, basic_ids, greenfield_ids,
        SERVICE_RADIUS_KM,
    )

    sel_upg, sel_gf, adv_assign, new_assign = solve_two_stage(
        pop_weight, already_covered, basic_pairs_r, gf_pairs_r,
        existing_pairs, advanced_ids, BUDGET, SERVICE_RADIUS_KM,
    )

    summary = build_summary(
        adv_assign, new_assign, sel_upg, sel_gf,
        BUDGET, SERVICE_RADIUS_KM, advanced_ids, basic_ids,
        total_vietnam_demand, sum(pop_weight.values()),
    )

    save_outputs(
        summary, sel_upg, sel_gf,
        adv_assign, new_assign, all_hosp, new_hosp,
    )

    # Uncomment to run full scenario grid:
    # scenario_df = run_scenarios(
    #     pop, existing_pairs, greenfield_pairs,
    #     advanced_ids, basic_ids, greenfield_ids, total_vietnam_demand,
    #     budgets=[0, 1, 2, 3, 5, 7, 10],
    #     radii=[50, 100, 300],
    # )
    # scenario_df.to_csv(OUTPUT_DIR / "combined_scenarios.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-only", action="store_true",
                        help="Only split the unified distance file, then exit.")
    args = parser.parse_args()
    if args.split_only:
        split_distance_file()
    else:
        main()
