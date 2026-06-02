import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import gurobipy as gp
from gurobipy import GRB


# Reuse graph/routing utilities from the map builder.
from build_population_map import (
    build_graph_from_roads_geojson,
    build_graph_kdtree,
    ensure_numeric,
    load_pickle,
    load_roads_geojson,
    load_stroke_facilities_csv,
    snap_points_to_graph_nodes,
)


from analysis import GADM_TO_CONSOLIDATED, PROVINCE_TO_REGION, assign_admin_regions


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------


DATA_DIR = "Data_vietnam"
OUTPUT_DIR = "optimization_output"
POPULATION_PATH = os.path.join(DATA_DIR, "population.pkl")
HOSPITAL_PATH = os.path.join(DATA_DIR, "existing_hospitals_100", "all_hospitals.pkl")
ROADS_PATH = os.path.join(DATA_DIR, "road_osm_preprocessed.geojson")
STROKE_CSV_PATH = os.path.join(DATA_DIR, "stroke-facs-100-en.csv")
GADM_PATH = "GADM_administrative boundaries.gpkg"


# MCLP parameters
TIME_THRESHOLD_MIN = 60      # "covered" if reachable within this many minutes
NUM_NEW_FACILITIES = 5       # number of PSCs to upgrade to CSCs (k)




# ---------------------------------------------------------------------------
# SECTION 0: DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------


def load_data():
    """
    Load population demand points, existing hospital locations, road-network
    graph, and administrative boundaries. Reuses loading logic from
    build_population_map.py and analysis.py.


    Returns:
        population_df: DataFrame [ID, lat, lon, household_count, province, region]
        hospitals_df: DataFrame with stroke capability flags and province
        G: NetworkX road graph with travel_time_min edge weights
        tree: cKDTree for snapping points to graph nodes
        node_list: list of (lon, lat) graph nodes
    """
    print("Loading population data...")
    population_df = load_pickle(POPULATION_PATH)
    print(f"  {len(population_df):,} demand points")


    print("Loading hospital data...")
    hospitals_df = load_pickle(HOSPITAL_PATH)
    if isinstance(hospitals_df.index, pd.Index) and hospitals_df.index.name == "ID":
        hospitals_df = hospitals_df.reset_index(drop=True)


    # Merge stroke facility info (thrombolysis / thrombectomy capability).
    stroke_df = load_stroke_facilities_csv(STROKE_CSV_PATH)
    hospitals_df["ID"] = ensure_numeric(hospitals_df["ID"]).astype(int)
    hospitals_df = hospitals_df.merge(stroke_df, on="ID", how="left")
    hospitals_df["specialized_equipment"] = hospitals_df["specialized_equipment"].fillna(False)


    n_adv = hospitals_df["specialized_equipment"].sum()
    n_basic = len(hospitals_df) - n_adv
    print(f"  {len(hospitals_df)} hospitals ({n_adv} advanced / {n_basic} basic)")


    # Load GADM for province assignment.
    print("Loading administrative boundaries...")
    admin_gdf = gpd.read_file(GADM_PATH, layer="ADM_ADM_1")
    population_df = assign_admin_regions(population_df, "lat", "lon", admin_gdf)
    hospitals_df = assign_admin_regions(hospitals_df, "Latitude", "Longitude", admin_gdf)


    # Build road graph.
    print("Building road graph (this may take a minute)...")
    roads_geojson = load_roads_geojson(ROADS_PATH)
    G = build_graph_from_roads_geojson(roads_geojson)
    tree, node_list, _ = build_graph_kdtree(G)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")


    return population_df, hospitals_df, G, tree, node_list




def compute_travel_times(hospitals_df, population_df, G, tree, node_list):
    """
    Run single-source Dijkstra from every hospital and compute travel time
    from each population node to each hospital.


    Returns:
        hosp_times: dict {hospital_id: {graph_node: travel_time_min}}
        pop_snapped: DataFrame with 'population_graph_node' column
        hospital_snap: DataFrame with 'hospital_graph_node' column
    """
    # Snap hospitals to graph.
    hospital_snap = snap_points_to_graph_nodes(
        hospitals_df, lat_col="Latitude", lon_col="Longitude",
        tree=tree, node_list=node_list, label="hospital",
    )


    # Run Dijkstra from each hospital.
    print(f"Running Dijkstra from each of {len(hospital_snap)} hospitals...")
    t0 = time.time()
    hosp_times = {}
    for _, hrow in hospital_snap.iterrows():
        hid = int(hrow["ID"])
        hnode = hrow["hospital_graph_node"]
        if hnode not in G:
            continue
        time_dict = nx.single_source_dijkstra_path_length(
            G, hnode, weight="travel_time_min"
        )
        hosp_times[hid] = time_dict
    elapsed = time.time() - t0
    print(f"  Dijkstra completed for {len(hosp_times)} hospitals in {elapsed:.0f}s")


    # Snap population to graph.
    print("Snapping population points to road graph...")
    pop_snapped = snap_points_to_graph_nodes(
        population_df, lat_col="lat", lon_col="lon",
        tree=tree, node_list=node_list, label="population",
    )


    return hosp_times, pop_snapped, hospital_snap




def build_mclp_inputs(hospitals_df, pop_snapped, hosp_times, time_threshold):
    """
    Construct the MCLP input sets from travel time data.


    Args:
        hospitals_df: Hospital DataFrame with 'ID' and 'specialized_equipment'
        pop_snapped: Population DataFrame with 'population_graph_node'
        hosp_times: dict {hospital_id: {graph_node: travel_time_min}}
        time_threshold: coverage threshold in minutes (S)


    Returns:
        I: list of demand node indices (0-based)
        J: list of candidate stroke center IDs (all hospitals)
        p: dict {i: population weight} (household_count)
        B: dict {j: 1 if already advanced, 0 otherwise}
        N: dict {i: [j for j in J if travel_time(i,j) <= S]}
    """
    print(f"\nBuilding MCLP inputs (threshold = {time_threshold} min)...")


    I = list(range(len(pop_snapped)))
    J = list(hosp_times.keys())


    # Population weight per demand node.
    weights = pop_snapped["household_count"].fillna(0).values
    p = {i: float(weights[i]) for i in I}


    # Binary: already has advanced capabilities?
    adv_set = set(
        hospitals_df.loc[hospitals_df["specialized_equipment"] == True, "ID"].astype(int)
    )
    B = {j: (1 if j in adv_set else 0) for j in J}


    # Neighborhood sets: which hospitals cover each demand node within threshold.
    pop_nodes = pop_snapped["population_graph_node"].values
    N = {i: [] for i in I}


    for j in J:
        time_dict = hosp_times[j]
        for i in I:
            t = time_dict.get(pop_nodes[i])
            if t is not None and t <= time_threshold:
                N[i].append(j)


    # Statistics.
    covered_by_existing = sum(1 for i in I if any(B[j] == 1 for j in N[i]))
    print(f"  Demand nodes: {len(I):,}")
    print(f"  Candidate facilities: {len(J)}")
    print(f"  Already advanced: {sum(B.values())}")
    print(f"  Nodes already covered by existing CSCs: {covered_by_existing:,} "
          f"({100*covered_by_existing/len(I):.1f}%)")
    avg_neighbors = np.mean([len(N[i]) for i in I])
    print(f"  Avg facilities within {time_threshold} min per demand node: {avg_neighbors:.1f}")


    return I, J, p, B, N




# ---------------------------------------------------------------------------
# SECTION 1: BASELINE — MAXIMUM COVERING LOCATION PROBLEM (MCLP)
# ---------------------------------------------------------------------------


def solve_mclp(I, J, p, B, N, k, solver_time_limit=1200):
    """
    Standard MCLP (Church & ReVelle, 1974) — solved with Gurobi.


    Maximize total population covered within the time threshold by upgrading
    at most k Primary Stroke Centers to Comprehensive Stroke Centers.


    Decision Variables:
        x_j ∈ {0, 1} — whether to upgrade center j  ∀ j ∈ J
        y_i ∈ {0, 1} — whether demand node i is covered


    Objective:
        max  Σ_{i ∈ I}  p[i] * y[i]


    Constraints:
        (Coverage)          Σ_{j ∈ N[i]} (x_j + B_j) ≥ y_i      ∀ i ∈ I
        (Mutual Exclus.)    x_j + B_j ≤ 1                        ∀ j ∈ J
        (Budget)            Σ_{j ∈ J} x_j ≤ k


    Args:
        I: list of demand node indices
        J: list of all stroke center IDs
        p: dict {i: population weight}
        B: dict {j: 1 if already advanced (CSC), 0 otherwise}
        N: dict {i: list of facility IDs within threshold}
        k: max number of upgrades allowed
        solver_time_limit: max solver runtime in seconds (default 600 = 10 min)


    Returns:
        dict with keys:
            'status': solver status string
            'objective': total population covered (float)
            'coverage_pct': fraction of total population covered
            'upgrades': list of facility IDs selected for upgrade
            'y_covered': number of demand nodes covered
            'mip_gap': optimality gap at termination
    """
    print(f"\n{'='*70}")
    print(f"SOLVING MCLP (budget k={k})")
    print(f"{'='*70}")


    # --- Define the model ---
    model = gp.Model("MCLP_Baseline")
    model.setParam("TimeLimit", solver_time_limit)
    model.setParam("MIPGap", 0.01)


    # Decision variables.
    # x_j: upgrade decision for ALL centers j ∈ J
    x = {j: model.addVar(vtype=GRB.BINARY, name=f"x_{j}") for j in J}
    # y_i: coverage indicator
    y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in I}


    # --- Objective: maximize total covered population ---
    model.setObjective(
        gp.quicksum(p[i] * y[i] for i in I), GRB.MAXIMIZE
    )


    # --- Coverage constraints ---
    # Σ_{j ∈ N[i]} (x_j + B_j) >= y_i
    for i in I:
        neighbors = N[i]
        if not neighbors:
            model.addConstr(y[i] == 0, name=f"NoCoverage_{i}")
        else:
            model.addConstr(
                gp.quicksum(x[j] + B[j] for j in neighbors) >= y[i],
                name=f"Coverage_{i}",
            )


    # --- Mutual exclusivity constraint ---
    # x_j + B_j <= 1  ∀ j ∈ J  (cannot upgrade an existing CSC)
    for j in J:
        model.addConstr(x[j] + B[j] <= 1, name=f"MutExcl_{j}")


    # --- Budget constraint ---
    # Σ_{j ∈ J} x_j <= k
    model.addConstr(
        gp.quicksum(x[j] for j in J) <= k, name="Budget"
    )


    # --- Solve ---
    print(f"Variables: {len(x) + len(y):,} "
          f"({len(x)} upgrade + {len(y):,} coverage)")
    print(f"Solving (Gurobi, time limit {solver_time_limit}s, MIPGap 1%)...")


    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0


    status = model.Status
    status_str = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "SubOptimal",
    }.get(status, f"Status_{status}")


    print(f"\nSolver status: {status_str} (solved in {elapsed:.1f}s)")


    if status == GRB.INFEASIBLE:
        print("ERROR: Problem is infeasible.")
        return {"status": status_str, "objective": 0, "coverage_pct": 0,
                "upgrades": [], "y_covered": 0, "mip_gap": None}


    if model.SolCount == 0:
        print("ERROR: No feasible solution found.")
        return {"status": status_str, "objective": 0, "coverage_pct": 0,
                "upgrades": [], "y_covered": 0, "mip_gap": None}


    # --- Extract results ---
    mip_gap = model.MIPGap
    obj_val = model.ObjVal
    total_pop = sum(p[i] for i in I)
    coverage_pct = obj_val / total_pop if total_pop > 0 else 0


    upgrades = [j for j in J if x[j].X > 0.5]
    n_covered = sum(1 for i in I if y[i].X > 0.5)


    print(f"\n--- MCLP Results ---")
    print(f"Objective (total pop covered): {obj_val:,.0f}")
    print(f"Coverage: {100*coverage_pct:.1f}% of total population")
    print(f"Demand nodes covered: {n_covered:,} / {len(I):,}")
    print(f"Facilities upgraded: {len(upgrades)} / {k} (budget)")
    print(f"Optimality gap: {100*mip_gap:.2f}%")
    print(f"\nSelected upgrades (center IDs): {upgrades}")


    return {
        "status": status_str,
        "objective": obj_val,
        "coverage_pct": coverage_pct,
        "upgrades": upgrades,
        "y_covered": n_covered,
        "mip_gap": mip_gap,
    }




# ---------------------------------------------------------------------------
# SECTION 2: EQUITY-EFFICIENCY MCLP
# ---------------------------------------------------------------------------


def build_equity_inputs(hospitals_df, pop_snapped, hosp_times, time_threshold):
    """
    Build inputs in the format required by solve_equity_mclp.


    Uses the same unified formulation as the baseline MCLP:
        - J: ALL stroke centers
        - B[j]: facility-level (1 if existing CSC, 0 otherwise)
        - N[i]: ALL facilities within threshold (not just candidates)
        - R, I_r: macro-region grouping of demand nodes


    Returns:
        I, J, R, I_r, p, B, N, P_r, P_total
    """
    print(f"\nBuilding equity-MCLP inputs (threshold = {time_threshold} min)...")


    I = list(range(len(pop_snapped)))


    # J = ALL hospitals.
    J = list(hosp_times.keys())


    # B[j] = facility-level: 1 if existing CSC, 0 otherwise.
    adv_set = set(
        hospitals_df.loc[hospitals_df["specialized_equipment"] == True, "ID"].astype(int)
    )
    B = {j: (1 if j in adv_set else 0) for j in J}


    # Population weight per demand node.
    weights = pop_snapped["household_count"].fillna(0).values
    p = {i: float(weights[i]) for i in I}
    P_total = sum(p[i] for i in I)


    # Province grouping (34 consolidated provinces as equity regions).
    regions = pop_snapped["province"].values
    R = sorted(set(r for r in regions if pd.notna(r) and r != "Unknown"))
    I_r = {r: [] for r in R}
    for i in I:
        r = regions[i]
        if r in I_r:
            I_r[r].append(i)


    P_r = {r: sum(p[i] for i in I_r[r]) for r in R}


    # N[i] = ALL facilities within threshold (evaluates distance to all j in J).
    pop_nodes = pop_snapped["population_graph_node"].values
    N = {i: [] for i in I}
    for j in J:
        time_dict = hosp_times[j]
        for i in I:
            t = time_dict.get(pop_nodes[i])
            if t is not None and t <= time_threshold:
                N[i].append(j)


    # Statistics.
    n_adv = sum(B.values())
    already_covered = sum(1 for i in I if any(B[j] == 1 for j in N[i]))
    print(f"  Demand nodes: {len(I):,}")
    print(f"  All facilities (J): {len(J)}")
    print(f"  Existing advanced (CSC): {n_adv}")
    print(f"  Nodes already covered by CSCs: {already_covered:,} "
          f"({100*already_covered/len(I):.1f}%)")
    print(f"  Provinces (equity regions): {len(R)} ({', '.join(R)})")
    for r in R:
        print(f"    {r}: {len(I_r[r]):,} nodes, pop {P_r[r]:,.0f}")


    return I, J, R, I_r, p, B, N, P_r, P_total




def generate_mock_data():
    """
    Generate mock data for testing both solve_mclp and solve_equity_mclp
    without loading real data. Produces at least 3 regions to demonstrate
    the equity tradeoff.


    Returns:
        I, J, R, I_r, p, B, N, P_r, P_total
        where B[j] is facility-level (1 if already advanced CSC).
    """
    np.random.seed(42)


    # 3 regions, 100 demand nodes each.
    R = ["Region_A", "Region_B", "Region_C"]
    I = list(range(300))
    I_r = {
        "Region_A": list(range(0, 100)),
        "Region_B": list(range(100, 200)),
        "Region_C": list(range(200, 300)),
    }


    # Population: Region_A is urban (high pop), C is rural (low pop).
    p = {}
    for i in I_r["Region_A"]:
        p[i] = float(np.random.randint(500, 2000))
    for i in I_r["Region_B"]:
        p[i] = float(np.random.randint(200, 800))
    for i in I_r["Region_C"]:
        p[i] = float(np.random.randint(50, 300))


    P_total = sum(p[i] for i in I)
    P_r = {r: sum(p[i] for i in I_r[r]) for r in R}


    # 20 facilities spread across regions.
    J = list(range(20))


    # B[j]: facility-level — facilities 0, 1, 2 are existing CSCs (in Region_A area).
    B = {j: 0 for j in J}
    B[0] = 1
    B[1] = 1
    B[2] = 1


    # N[i]: which facilities can cover each demand node (ALL facilities, incl. CSCs).
    # Facilities 0-7 cover Region_A, 8-13 cover B, 14-19 cover C (with overlap).
    N = {i: [] for i in I}
    for i in I_r["Region_A"]:
        N[i] = [j for j in range(0, 8) if np.random.random() < 0.4]
    for i in I_r["Region_B"]:
        N[i] = [j for j in range(6, 14) if np.random.random() < 0.35]
    for i in I_r["Region_C"]:
        N[i] = [j for j in range(12, 20) if np.random.random() < 0.25]


    return I, J, R, I_r, p, B, N, P_r, P_total




def solve_equity_mclp(I, J, R, I_r, p, B, N, k, lambda_val, P_r, P_total,
                      solver_time_limit=1200):
    """
    Equity-Efficiency MCLP — solved with Gurobi.


    Balances maximizing national coverage proportion (efficiency) with
    minimizing the mean absolute deviation of regional coverage rates from
    the national average (equity).


    Decision Variables:
        x[j] ∈ {0, 1}  — upgrade center j              ∀ j ∈ J
        y[i] ∈ {0, 1}  — demand node i is covered
        C_r[r] ≥ 0     — proportion of population covered in region r
        C_bar ≥ 0       — national coverage proportion
        d[r] ≥ 0        — |C_r[r] - C_bar| (absolute deviation)


    Objective:
        max Z = (1 - λ) * C_bar  -  (λ / |R|) * Σ_{r∈R} d[r]


    Constraints:
        Coverage:     Σ_{j∈N[i]} (x_j + B_j) ≥ y_i         ∀ i ∈ I
        Mutual Excl.: x_j + B_j ≤ 1                         ∀ j ∈ J
        Exist. Floor: B_i ≤ y_i                              ∀ i ∈ I
        Budget:       Σ_{j∈J} x_j ≤ k
        Regional:     C_r[r] = (1/P_r) * Σ_{i∈I_r} p[i]*y[i]  ∀ r ∈ R
        National:     C_bar = (1/P_total) * Σ_{i∈I} p[i]*y[i]
        MAD+:         C_r[r] - C_bar ≤ d[r]                 ∀ r ∈ R
        MAD-:         C_bar - C_r[r] ≤ d[r]                 ∀ r ∈ R


    Args:
        I, J, R, I_r, p, B, N: model inputs (see build_equity_inputs)
        k: upgrade budget
        lambda_val: equity weight (0 = pure efficiency, 1 = pure equity)
        P_r: dict of regional populations
        P_total: total population
        solver_time_limit: max solver seconds (default 600 = 10 min)


    Returns:
        dict with status, Z, C_bar, sum_d, upgrades, regional_coverage, mip_gap
    """
    print(f"\n--- Solving Equity-MCLP (k={k}, lambda={lambda_val}) ---")


    model = gp.Model(f"EquityMCLP_k{k}_l{lambda_val}")
    model.setParam("TimeLimit", solver_time_limit)
    model.setParam("MIPGap", 0.01)


    # Decision variables.
    # x_j: upgrade decision for ALL centers j ∈ J
    x = {j: model.addVar(vtype=GRB.BINARY, name=f"x_{j}") for j in J}
    # y_i: coverage indicator (binary per LaTeX spec).
    y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in I}
    C_r = {r: model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Cr_{r}") for r in R}
    C_bar = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C_bar")
    d = {r: model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"d_{r}") for r in R}


    # Objective: max Z = (1-lambda)*C_bar - (lambda/|R|)*sum(d[r])
    model.setObjective(
        (1 - lambda_val) * C_bar
        - (lambda_val / len(R)) * gp.quicksum(d[r] for r in R),
        GRB.MAXIMIZE,
    )


    # Coverage constraint: Σ_{j∈N[i]} (x_j + B_j) >= y_i
    for i in I:
        neighbors = N[i]
        if not neighbors:
            model.addConstr(y[i] == 0, name=f"NoCov_{i}")
        else:
            model.addConstr(
                gp.quicksum(x[j] + B[j] for j in neighbors) >= y[i],
                name=f"Cov_{i}",
            )


    # Mutual exclusivity constraint: x_j + B_j <= 1  ∀ j ∈ J
    for j in J:
        model.addConstr(x[j] + B[j] <= 1, name=f"MutExcl_{j}")


    # Enforce baseline constraint: y_i >= x_j + B_j  ∀ i ∈ I, ∀ j ∈ N_i
    # If any facility covering node i is open (upgraded or existing CSC), i must be covered.
    for i in I:
        neighbors = N[i]
        for j in neighbors:
            model.addConstr(y[i] >= x[j] + B[j], name=f"EnfBase_{i}_{j}")


    # Budget constraint.
    model.addConstr(gp.quicksum(x[j] for j in J) <= k, name="Budget")


    # Regional coverage definition.
    for r in R:
        if P_r[r] > 0:
            model.addConstr(
                C_r[r] == (1.0 / P_r[r]) * gp.quicksum(p[i] * y[i] for i in I_r[r]),
                name=f"RegCov_{r}",
            )
        else:
            model.addConstr(C_r[r] == 0, name=f"RegCov_{r}")


    # National coverage definition.
    model.addConstr(
        C_bar == (1.0 / P_total) * gp.quicksum(p[i] * y[i] for i in I),
        name="NatCov",
    )


    # MAD constraints (linearization of |C_r - C_bar|).
    for r in R:
        model.addConstr(C_r[r] - C_bar <= d[r], name=f"MAD_pos_{r}")
        model.addConstr(C_bar - C_r[r] <= d[r], name=f"MAD_neg_{r}")


    # Solve.
    n_vars = model.NumVars
    print(f"  Variables: {n_vars:,}")
    print(f"  Solving (Gurobi, time limit {solver_time_limit}s, MIPGap 1%)...")


    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0


    status = model.Status
    status_str = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SUBOPTIMAL: "SubOptimal",
    }.get(status, f"Status_{status}")


    if status == GRB.INFEASIBLE:
        print(f"  INFEASIBLE (solved in {elapsed:.1f}s)")
        return {"status": "Infeasible", "Z": None, "C_bar": None,
                "sum_d": None, "upgrades": [], "regional_coverage": {},
                "k": k, "lambda": lambda_val, "mip_gap": None}


    if model.SolCount == 0:
        print(f"  No feasible solution found (solved in {elapsed:.1f}s)")
        return {"status": status_str, "Z": None, "C_bar": None,
                "sum_d": None, "upgrades": [], "regional_coverage": {},
                "k": k, "lambda": lambda_val, "mip_gap": None}


    # Extract results.
    mip_gap = model.MIPGap
    Z_val = model.ObjVal
    C_bar_val = C_bar.X
    upgrades = sorted([j for j in J if x[j].X > 0.5])
    reg_cov = {r: C_r[r].X for r in R}

    # Compute sum_d post-hoc from actual C_r values (solver d_r may not be tight when lambda=0).
    sum_d_val = sum(abs(reg_cov[r] - C_bar_val) for r in R)
    mad_val = sum_d_val / len(R)


    print(f"  Status: {status_str} ({elapsed:.1f}s)")
    print(f"  Z = {Z_val:.6f}")
    print(f"  C_bar (national coverage) = {C_bar_val:.4f} ({100*C_bar_val:.1f}%)")
    print(f"  MAD (mean abs deviation) = {mad_val:.6f}")
    print(f"  sum_d = {sum_d_val:.5f}")
    print(f"  Optimality gap: {100*mip_gap:.2f}%")
    print(f"  Upgrades ({len(upgrades)}): {upgrades}")
    for r in R:
        print(f"    {r}: {100*reg_cov[r]:.1f}%")


    return {
        "status": status_str,
        "Z": Z_val,
        "C_bar": C_bar_val,
        "sum_d": sum_d_val,
        "mad": mad_val,
        "upgrades": upgrades,
        "regional_coverage": reg_cov,
        "k": k,
        "lambda": lambda_val,
        "mip_gap": mip_gap,
    }




# ---------------------------------------------------------------------------
# SECTION 3: SCENARIO COMPARISON & RESULTS EXPORT
# ---------------------------------------------------------------------------


def run_scenarios(I, J, R, I_r, p, B, N, P_r, P_total,
                  k_values=(10,), lambda_values=(0.5, 0.8, 1),
                  solver_time_limit=1200):
    """
    Run solve_equity_mclp across all combinations of k and lambda.


    lambda_val = 0.0 is the pure efficiency baseline (standard MCLP).
    lambda_val = 1.0 is the pure equity model (minimize regional deviation).


    Returns:
        results: list of result dicts from solve_equity_mclp
    """
    results = []
    total_runs = len(k_values) * len(lambda_values)
    run_idx = 0


    for k in k_values:
        for lam in lambda_values:
            run_idx += 1
            print(f"\n{'='*70}")
            print(f"SCENARIO {run_idx}/{total_runs}: k={k}, lambda={lam}")
            print(f"{'='*70}")


            result = solve_equity_mclp(
                I, J, R, I_r, p, B, N,
                k=k, lambda_val=lam,
                P_r=P_r, P_total=P_total,
                solver_time_limit=solver_time_limit,
            )
            results.append(result)


    return results


def compute_regional_coverage(upgrades, I, B, N, p, I_r, P_r):
    """
    Compute per-region coverage from a set of upgrades (post-hoc).

    A demand node i is covered if at least one facility j in N[i] is either
    already advanced (B[j]=1) or selected for upgrade (j in upgrades).

    Returns:
        regional_coverage: dict {r: coverage_proportion} for each region r
    """
    upgrade_set = set(upgrades)
    regional_coverage = {}
    for r, nodes in I_r.items():
        pop_r = P_r[r]
        if pop_r == 0:
            regional_coverage[r] = 0.0
            continue
        covered_pop = sum(
            p[i] for i in nodes
            if any((B[j] == 1 or j in upgrade_set) for j in N[i])
        )
        regional_coverage[r] = covered_pop / pop_r
    return regional_coverage


def compute_mad(regional_coverage, C_bar, R):
    """
    Compute sum_d and MAD (mean absolute deviation) from regional coverage.

    Args:
        regional_coverage: dict {r: coverage_proportion}
        C_bar: national coverage proportion
        R: list of region names

    Returns:
        (sum_d, mad) where sum_d = Σ|C_r - C_bar|, mad = sum_d / |R|
    """
    sum_d = sum(abs(regional_coverage.get(r, 0) - C_bar) for r in R)
    mad = sum_d / len(R) if len(R) > 0 else 0.0
    return sum_d, mad


def export_results(results, R, hospitals_df, output_path, baseline_results=None):
    """
    Write scenario comparison results to a text file.


    Compares across k x lambda grid:
        - Objective value Z (combined efficiency-equity score)
        - National coverage C_bar (efficiency metric)
        - Sum of regional deviations sum(d_r) (equity metric, lower = more equal)
        - Per-region coverage rates
        - Selected upgrade centers


    The comparison highlights the tradeoff:
        - As lambda increases (more equity weight), C_bar may decrease but sum(d_r)
          should decrease, meaning more equitable regional distribution.
        - As k increases, both metrics should improve (more budget = better).
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STROKE FACILITY OPTIMIZATION - SCENARIO COMPARISON RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Model: Equity-Efficiency Maximum Covering Location Problem (MCLP)")
    lines.append("Objective: max Z = (1-lambda)*C_bar - (lambda/|R|)*sum(d_r)")
    lines.append("  where C_bar = national coverage proportion")
    lines.append("        MAD    = (1/|R|) * sum|C_r - C_bar| (mean absolute deviation)")
    lines.append("        sum_d  = sum|C_r - C_bar| = MAD * |R|")
    lines.append("        lambda = equity weight (0=pure efficiency, 1=pure equity)")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  lambda=0.0: Pure efficiency baseline (maximize total population covered)")
    lines.append("  lambda=0.5: Balanced efficiency-equity tradeoff")
    lines.append("  lambda=1.0: Pure equity (minimize regional inequality)")
    lines.append("")


    # Baseline MCLP section.
    if baseline_results:
        lines.append("=" * 80)
        lines.append("BASELINE MCLP RESULTS (no equity constraint)")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"{'k':>3} | {'Status':>8} | {'Objective':>12} | "
                     f"{'Coverage':>8} | {'MAD':>8} | {'MIPGap':>7} | {'#Upgr':>6} | Upgraded IDs")
        lines.append("-" * 100)
        for bres in baseline_results:
            ids_str = ", ".join(str(u) for u in bres.get("upgrades", []))
            gap = bres.get("mip_gap")
            gap_str = f"{100*gap:.2f}%" if gap is not None else "N/A"
            obj = bres.get("objective", 0) or 0
            cov_pct = bres.get("coverage_pct", 0) or 0
            mad = bres.get("mad", 0) or 0
            lines.append(
                f"{bres['k']:>3} | {bres.get('status','N/A'):>8} | "
                f"{obj:>12,.0f} | "
                f"{cov_pct:>7.1%} | "
                f"{mad:>8.5f} | "
                f"{gap_str:>7} | "
                f"{len(bres.get('upgrades', [])):>6} | {ids_str}"
            )
        lines.append("-" * 100)
        lines.append("")

        # Per-province coverage for baseline.
        lines.append("Baseline provincial coverage:")
        for bres in baseline_results:
            reg_cov = bres.get("regional_coverage", {})
            nat_cov = bres.get("coverage_pct", 0) or 0
            if not reg_cov:
                continue
            lines.append(f"  --- k={bres['k']} (national coverage={nat_cov:.1%}) ---")
            for r in R:
                cov = reg_cov.get(r, 0)
                diff = cov - nat_cov
                lines.append(f"    {r:<25} {100*cov:>6.1f}%  (delta = {100*diff:>+5.1f}%)")
        lines.append("")


    # Summary table (equity scenarios).
    lines.append("=" * 80)
    lines.append("EQUITY-EFFICIENCY MCLP SCENARIOS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("-" * 90)
    lines.append(f"{'k':>3} | {'lambda':>6} | {'Status':>8} | {'Z':>10} | "
                 f"{'C_bar':>8} | {'MAD':>8} | {'sum_d':>8} | {'MIPGap':>7} | {'#Upgr':>6} | Upgraded IDs")
    lines.append("-" * 100)


    for res in results:
        if res["Z"] is None:
            lines.append(f"{res['k']:>3} | {res['lambda']:>6.1f} | "
                         f"{'INFEAS':>8} | {'N/A':>10} | {'N/A':>8} | "
                         f"{'N/A':>8} | {'N/A':>8} | {'N/A':>7} | {'N/A':>6} | N/A")
        else:
            ids_str = ", ".join(str(u) for u in res["upgrades"])
            gap = res.get("mip_gap")
            gap_str = f"{100*gap:.2f}%" if gap is not None else "N/A"
            mad = res.get("mad", res["sum_d"] / len(R))
            lines.append(
                f"{res['k']:>3} | {res['lambda']:>6.1f} | "
                f"{res['status']:>8} | {res['Z']:>10.6f} | "
                f"{res['C_bar']:>7.1%} | {mad:>8.5f} | {res['sum_d']:>8.5f} | "
                f"{gap_str:>7} | "
                f"{len(res['upgrades']):>6} | {ids_str}"
            )


    lines.append("-" * 100)
    lines.append("")


    # Regional coverage detail per scenario.
    lines.append("=" * 80)
    lines.append("REGIONAL COVERAGE DETAIL")
    lines.append("=" * 80)


    for res in results:
        if res["Z"] is None:
            continue
        lines.append("")
        mad = res.get("mad", res["sum_d"] / len(R))
        lines.append(f"--- k={res['k']}, lambda={res['lambda']:.1f} "
                     f"(C_bar={res['C_bar']:.1%}, MAD={mad:.5f}) ---")
        for r in R:
            cov = res["regional_coverage"].get(r, 0)
            diff = cov - res["C_bar"]
            lines.append(f"  {r:<25} {100*cov:>6.1f}%  (delta = {100*diff:>+5.1f}%)")


    lines.append("")


    # Upgrade details (with hospital names if available).
    lines.append("=" * 80)
    lines.append("UPGRADE DETAILS")
    lines.append("=" * 80)


    for res in results:
        if res["Z"] is None or not res["upgrades"]:
            continue
        lines.append("")
        lines.append(f"--- k={res['k']}, lambda={res['lambda']:.1f} ---")
        for hid in res["upgrades"]:
            if hospitals_df is not None and len(hospitals_df[hospitals_df["ID"] == hid]) > 0:
                row = hospitals_df[hospitals_df["ID"] == hid].iloc[0]
                name = row.get("Name_English", "Unknown")
                prov = row.get("province", "Unknown")
                lines.append(f"  ID {hid}: {name} ({prov})")
            else:
                lines.append(f"  ID {hid}")


    lines.append("")


    # Tradeoff analysis.
    lines.append("=" * 80)
    lines.append("TRADEOFF ANALYSIS")
    lines.append("=" * 80)
    lines.append("")


    # Group by k, compare across lambda.
    k_values = sorted(set(r["k"] for r in results))
    for k in k_values:
        k_results = [r for r in results if r["k"] == k and r["Z"] is not None]
        if len(k_results) < 2:
            continue
        lines.append(f"Budget k={k}:")

        # Compare against explicit baseline MCLP if available.
        if baseline_results:
            bl = next((b for b in baseline_results if b.get("k") == k), None)
            if bl and bl.get("coverage_pct"):
                lines.append(f"  Baseline MCLP coverage: {bl['coverage_pct']:.1%}")

        eff = next((r for r in k_results if r["lambda"] == 0.0), None)
        equity = next((r for r in k_results if r["lambda"] == 1.0), None)
        if eff and equity:
            cov_loss = eff["C_bar"] - equity["C_bar"]
            eff_mad = eff.get("mad", eff["sum_d"] / len(R))
            eq_mad = equity.get("mad", equity["sum_d"] / len(R))
            mad_gain = eff_mad - eq_mad
            lines.append(f"  Efficiency (lambda=0) -> Equity (lambda=1) tradeoff:")
            lines.append(f"    Coverage change:  {100*eff['C_bar']:.1f}% -> "
                         f"{100*equity['C_bar']:.1f}% "
                         f"(delta = {-100*cov_loss:+.1f}%)")
            lines.append(f"    MAD change:       {eff_mad:.5f} -> "
                         f"{eq_mad:.5f} "
                         f"(delta = {-mad_gain:+.5f})")
            if cov_loss > 0 and mad_gain > 0:
                lines.append(f"    -> Sacrificed {100*cov_loss:.1f}pp coverage for "
                             f"{mad_gain:.5f} less MAD")
            elif cov_loss <= 0 and mad_gain > 0:
                lines.append(f"    -> Improved both coverage and equity (no tradeoff)")
        lines.append("")


    # Write to file.
    output_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)


    print(f"\nResults written to: {output_path}")
    print(output_text)


    return output_text




# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    print("=" * 70)
    print("STROKE FACILITY LOCATION OPTIMIZATION - VIETNAM")
    print("=" * 70)


    # --- Set USE_MOCK_DATA=True to test without Dijkstra computation ---
    USE_MOCK_DATA = False

    # Scenario grid parameters.
    K_VALUES = (10,)
    LAMBDA_VALUES = (0.5, 0.8, 1)

    if USE_MOCK_DATA:
        print("\nUsing MOCK DATA for testing...")
        I, J, R, I_r, p, B, N, P_r, P_total = generate_mock_data()
        hospitals_df = None
    else:
        # Step 1: Load data.
        population_df, hospitals_df, G, tree, node_list = load_data()


        # Step 2: Compute travel times (Dijkstra from all hospitals).
        hosp_times, pop_snapped, hospital_snap = compute_travel_times(
            hospitals_df, population_df, G, tree, node_list
        )


        # Step 3: Build equity-MCLP inputs (provinces as equity regions).
        I, J, R, I_r, p, B, N, P_r, P_total = build_equity_inputs(
            hospitals_df, pop_snapped, hosp_times, TIME_THRESHOLD_MIN
        )

    # Step 4: Run baseline MCLP for each k.
    baseline_results = []
    for k in K_VALUES:
        bres = solve_mclp(I, J, p, B, N, k=k)
        bres["k"] = k
        # Compute post-hoc provincial coverage for the baseline upgrades.
        bres["regional_coverage"] = compute_regional_coverage(
            bres["upgrades"], I, B, N, p, I_r, P_r
        )
        # Compute MAD for baseline.
        C_bar_bl = bres.get("coverage_pct", 0) or 0
        bres["sum_d"], bres["mad"] = compute_mad(bres["regional_coverage"], C_bar_bl, R)
        baseline_results.append(bres)


    # Step 5: Run equity-efficiency scenarios across k x lambda grid.
    results = run_scenarios(
        I, J, R, I_r, p, B, N,
        k_values=K_VALUES, lambda_values=LAMBDA_VALUES,
        P_r=P_r, P_total=P_total,
    )


    # Step 6: Export comparison results.
    output_path = os.path.join(OUTPUT_DIR, "optimization_results.txt")
    export_results(results, R, hospitals_df, output_path,
                   baseline_results=baseline_results)


    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")



