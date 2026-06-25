# =============================================================================
# BASE MAXIMAL COVERING LOCATION PROBLEM MODEL
# =============================================================================
# Purpose of this file:
# This script solves the baseline MCLP-style hospital upgrade model.
# A population point is counted as covered if it can reach at least one open
# advanced stroke-care hospital within SERVICE_RADIUS_MINUTES = 270 minutes.
#
# How to run from the command line(example):
#   python optimize_upgrade_base_mclp.py --b 5
#   python optimize_upgrade_base_mclp.py --b 10 --time-limit 600 --msg
#
# Main outputs:
# Results/new_upgrade_models/base_mclp/
# - base_mclp_b<b>_summary.json
# - base_mclp_b<b>_selected_facilities.csv
# - base_mclp_b<b>_population_summary.csv
# - scenario_comparison.csv
# =============================================================================

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import pulp
from upgrade_model_common import SERVICE_RADIUS_MINUTES, build_model_data, build_run_summary, ensure_results_dir, write_run_outputs

# Name used in output files and PuLP problem names.
MODEL_NAME = "base_mclp"

# -----------------------------------------------------------------------------
# SOLVE THE BASE MCLP MODEL
# -----------------------------------------------------------------------------
# Inputs:
# - b: number of hospital upgrades allowed, equal to budget mention in thesis as k.
# - time_limit: CBC solver time limit in seconds.
# - msg: whether CBC should print solver logs.
#
# Decision variables:
# - x_j = 1 if hospital j is selected for upgrade.
# - y_i = 1 if population point i is covered within 270 minutes.
#
# Objective:
# Maximize the number of households covered within the service radius.
def solve_base_mclp(b: int, time_limit: int = 600, msg: bool = False):
    data = build_model_data()
    problem = pulp.LpProblem(f"{MODEL_NAME}_b{b}", pulp.LpMaximize)

    fixed_set = set(data.fixed_facility_ids)
    all_facility_ids = [int(facility_id) for facility_id in data.candidate_facility_ids]
    B_param: dict[int, int] = {facility_id: int(facility_id in fixed_set) for facility_id in all_facility_ids}
    x_vars: dict[int, pulp.LpVariable] = {
        facility_id: pulp.LpVariable(f"x_{facility_id}", cat="Binary")
        for facility_id in all_facility_ids
    }

    # Keep only arcs that are inside the 270-minute service radius.
    # These arcs define whether a population point can be covered.
    coverage_arcs = data.arcs[data.arcs["travel_minutes"] <= SERVICE_RADIUS_MINUTES].copy()
    coverage_arcs = coverage_arcs.dropna(subset=["pop_id", "facility_id", "travel_minutes"]).copy()
    coverage_arcs["pop_id"] = coverage_arcs["pop_id"].astype(int)
    coverage_arcs["facility_id"] = coverage_arcs["facility_id"].astype(int)
    coverage_arcs = coverage_arcs.drop_duplicates(subset=["pop_id", "facility_id"]).copy()
    coverage_groups = coverage_arcs.groupby("pop_id")
    # Create one coverage variable for each population point.
    y_vars: dict[int, pulp.LpVariable] = {
        int(pop_id): pulp.LpVariable(f"y_{int(pop_id)}", cat="Binary")
        for pop_id in data.population["pop_id"].astype(int).tolist()
    }

    # Objective: maximize covered household count.
    problem += pulp.lpSum(
        float(household_count) * y_vars[int(pop_id)]
        for pop_id, household_count in data.population[["pop_id", "household_count"]].itertuples(index=False)
    )

    # Link y_i to open facilities. If no open facility is within 270 minutes,
    # y_i must be zero.
    for pop_id in data.population["pop_id"].astype(int).tolist():
        group = coverage_groups.get_group(pop_id) if pop_id in coverage_groups.groups else None
        if group is None:
            problem += y_vars[pop_id] == 0
            continue

        facility_ids = group["facility_id"].astype(int).tolist()
        if facility_ids:
            # y_i can become 1 if at least one covering hospital is fixed open or selected.
            problem += pulp.lpSum(x_vars[facility_id] + B_param[facility_id] for facility_id in facility_ids) >= y_vars[pop_id]
        else:
            problem += y_vars[pop_id] == 0

    # Existing advanced hospitals cannot be selected again as upgrades.
    for facility_id in all_facility_ids:
        problem += x_vars[facility_id] + B_param[facility_id] <= 1

    # Budget constraint: select at most b new upgrades.
    problem += pulp.lpSum(x_vars[facility_id] for facility_id in all_facility_ids) <= b

    # Solve the binary linear program with CBC through PuLP.
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    problem.solve(solver)
    print(f"Status: {pulp.LpStatus[problem.status]}")

    # Read selected upgrade decisions from the solved x variables.
    selected_ids = [j for j, var in x_vars.items() if pulp.value(var) is not None and pulp.value(var) > 0.5]
    open_ids = sorted(set(selected_ids) | fixed_set)
    return data, problem, open_ids


# -----------------------------------------------------------------------------
# RUN THE BASE MODEL AND WRITE OUTPUTS
# -----------------------------------------------------------------------------
# This wrapper solves the base model, builds reporting tables and writes the
# standard output files.
def run_base_model(b: int, output_root: Path | None = None, time_limit: int = 600, msg: bool = False):
    data, problem, open_ids = solve_base_mclp(b=b, time_limit=time_limit, msg=msg)

    assignment_df = data.arcs[data.arcs["facility_id"].isin(open_ids)].copy()
    assignment_df = assignment_df.sort_values(["pop_id", "travel_minutes", "facility_id"]).drop_duplicates("pop_id")
    assignment_df = assignment_df.rename(
        columns={
            "facility_id": "best_facility_id",
            "name_english": "best_facility_name",
            "travel_minutes": "best_travel_minutes",
        }
    )
    assignment_df = assignment_df[
        [
            "pop_id",
            "lat",
            "lon",
            "household_count",
            "best_travel_minutes",
            "best_facility_id",
            "best_facility_name",
        ]
    ].copy()
    # Keep all population points, including points not served by any open facility.
    assignment_df = data.population.merge(assignment_df, on=["pop_id", "lat", "lon", "household_count"], how="left")
    assignment_df["best_facility_id"] = assignment_df["best_facility_id"].fillna(-1).astype(int)
    assignment_df["best_facility_name"] = assignment_df["best_facility_name"].fillna("Unserved")
    assignment_df["best_travel_minutes"] = assignment_df["best_travel_minutes"].fillna(999.0)
    assignment_df["objective_weight"] = assignment_df["household_count"].where(
        assignment_df["best_travel_minutes"] <= SERVICE_RADIUS_MINUTES,
        0,
    )

    summary, selected_facilities, population_summary = build_run_summary(
        model_name=MODEL_NAME,
        scenario="base",
        b=b,
        facilities=data.facilities,
        open_facility_ids=open_ids,
        assignment_df=assignment_df,
        objective_value=float(pulp.value(problem.objective) or 0.0),
        total_demand_weight=float(data.population["household_count"].sum()),
    )

    output_dir = ensure_results_dir(MODEL_NAME) if output_root is None else Path(output_root)
    file_prefix = f"{MODEL_NAME}_b{b}"
    write_run_outputs(output_dir, file_prefix, summary, selected_facilities, population_summary)
    pd.DataFrame([summary]).to_csv(output_dir / "scenario_comparison.csv", index=False)
    return summary

# -----------------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# -----------------------------------------------------------------------------
# Parse command-line settings so the model can be run from Terminal.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the base MCLP upgrade model.")
    parser.add_argument("--b", type=int, default=1, help="Number of upgrades to select.")
    parser.add_argument("--time-limit", type=int, default=600, help="Solver time limit in seconds.")
    parser.add_argument("--msg", action="store_true", help="Show solver output.")
    return parser.parse_args()

# Entry point used when this file is run directly.
def main() -> None:
    args = parse_args()
    run_base_model(b=args.b, time_limit=args.time_limit, msg=args.msg)


if __name__ == "__main__":
    main()


