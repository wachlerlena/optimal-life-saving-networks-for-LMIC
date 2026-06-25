# =============================================================================
# WEIGHTED HOSPITAL UPGRADE MODEL
# =============================================================================
# Purpose of this file:
# This script solves the outcome-weighted hospital upgrade model.
# Instead of treating every covered population point equally, it multiplies each
# household count by a time-to-treatment benefit function.
#
# Two scenarios can be solved:
# - exponential: benefit decays exponentially with travel time.
# - logistic: benefit follows the calibrated logistic curve.
#
# How to run from the command line:
#   python optimize_upgrade_weighted_assignment.py --b 5 --scenarios exponential, logistic
#   python optimize_upgrade_weighted_assignment.py --b 10 --time-limit 600 --msg
#
# Main outputs:
# Results/new_upgrade_models/weighted_thesis/
# - weighted_thesis_<scenario>_b<b>_summary.json
# - weighted_thesis_<scenario>_b<b>_selected_facilities.csv
# - weighted_thesis_<scenario>_b<b>_population_summary.csv
# - scenario_comparison.csv
# =============================================================================

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import pulp
from upgrade_model_common import WEIGHT_FUNCTIONS, build_model_data, build_run_summary, ensure_results_dir, write_run_outputs



# Name used in output files and PuLP problem names.
MODEL_NAME = "weighted_thesis"


# -----------------------------------------------------------------------------
# SOLVE ONE WEIGHTED OPTIMIZATION MODEL
# -----------------------------------------------------------------------------
# Inputs:
# - b: number of hospital upgrades allowed.
# - scenario: "exponential" or "logistic".
# - time_limit: CBC solver time limit in seconds.
# - msg: whether CBC should print solver logs.
#
# Decision variables:
# - x_j = 1 if hospital j is selected for upgrade.
# - y_ij = 1 if population point i is assigned to hospital j.
#
# Objective:
# Maximize household_count_i * benefit_function(travel_time_ij) * y_ij.
def solve_weighted_assignment(b: int, scenario: str, time_limit: int = 600, msg: bool = False):
   if scenario not in WEIGHT_FUNCTIONS:
       raise ValueError(f"Unknown weight scenario: {scenario}")

   # Load the shared cleaned model data.
   data = build_model_data()
   problem = pulp.LpProblem(f"{MODEL_NAME}_{scenario}_b{b}", pulp.LpMaximize)


   fixed_set = set(data.fixed_facility_ids)
   B_param = {facility_id: 1 if facility_id in fixed_set else 0 for facility_id in data.candidate_facility_ids}
   x_vars: dict[int, pulp.LpVariable] = {
       facility_id: pulp.LpVariable(f"x_{facility_id}", cat="Binary")
       for facility_id in data.candidate_facility_ids
   }


   arc_data = data.arcs.copy()


   # Keep (a) nearest fixed thrombectomy hospital per pop cell
   #      (b) nearest candidate tPA-only hospital per pop cell
   # This gives at most 2 arcs per cell and is the minimal set needed
   # for upgrade decisions to differ between exponential and logistic objectives.
   fixed_ids_set = set(data.fixed_facility_ids)
   candidate_ids_set = set(data.candidate_facility_ids) - fixed_ids_set


   arcs_fixed = (
       arc_data[arc_data["facility_id"].isin(fixed_ids_set)]
       .sort_values(["pop_id", "travel_minutes"])
       .drop_duplicates(subset="pop_id", keep="first")
   )
   arcs_candidate = (
       arc_data[arc_data["facility_id"].isin(candidate_ids_set)]
       .sort_values(["pop_id", "travel_minutes"])
       .drop_duplicates(subset="pop_id", keep="first")
   )
   arc_data = (
       pd.concat([arcs_fixed, arcs_candidate], ignore_index=True)
       .sort_values(["pop_id", "travel_minutes"])
       .reset_index(drop=True)
   )
   # Pick the correct precomputed time-benefit column.
   # The objective coefficient is population weight times time-benefit weight.
   weight_column = "weight_exponential" if scenario == "exponential" else "weight_logistic"
   arc_data["objective_weight"] = arc_data["household_count"] * arc_data[weight_column]

   # Create assignment variables only for the reduced arc set. 
   y_vars: dict[tuple[int, int], pulp.LpVariable] = {}
   for row in arc_data[["pop_id", "facility_id"]].itertuples(index=False):
       pop_id = int(row.pop_id)
       facility_id = int(row.facility_id)
       y_vars[(pop_id, facility_id)] = pulp.LpVariable(f"y_{pop_id}_{facility_id}", cat="Binary")

   # x_vars represent the binary treatment-state C_j; y<=x links each assignment
   # to an open/updated facility so the linear objective stays equivalent to the thesis form.
   # Objective: assign each population point to at most one open facility and
   # maximize the weighted expected outcome. 
   problem += pulp.lpSum(
       float(row.objective_weight) * y_vars[(int(row.pop_id), int(row.facility_id))]
       for row in arc_data[["pop_id", "facility_id", "objective_weight"]].itertuples(index=False)
   )

   # Each population point may be assigned to at most one facility 
   for pop_id, group in arc_data.groupby("pop_id"):
       problem += (
           pulp.lpSum(
               y_vars[(int(pop_id), int(facility_id))]
               for facility_id in group["facility_id"].astype(int).tolist()
           )
           <= 1
       )

   # A population point can only be assigned to a hospital that is either already
   # fixed open or selected for upgrade.    
   for (pop_id, facility_id), var in y_vars.items():
       problem += var <= x_vars[facility_id] + B_param[facility_id]

   # Existing advanced hospitals cannot be selected again as upgrades. 
   for facility_id in data.candidate_facility_ids:
       problem += x_vars[facility_id] + B_param[facility_id] <= 1

   # Budget constraint: select at most b new upgrades.
   problem += pulp.lpSum(x_vars[facility_id] for facility_id in data.candidate_facility_ids) <= b

   # Solve the binary linear program with CBC through PuLP.
   solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
   problem.solve(solver)
   print(f"[b={b}, scenario={scenario}] Status: {pulp.LpStatus[problem.status]}")

   # Read selected upgrade decisions from the solved x variables.
   selected_ids = [facility_id for facility_id, var in x_vars.items() if pulp.value(var) and pulp.value(var) > 0.5]
   open_ids = sorted(set(selected_ids) | fixed_set)

   # After solving, assign every population point to the nearest open hospital.
   # This full assignment is used for reporting, not for changing the optimization.
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
   assignment_df["objective_weight"] = assignment_df["household_count"] * WEIGHT_FUNCTIONS[scenario](assignment_df["best_travel_minutes"])
   assignment_df.loc[assignment_df["best_facility_id"] < 0, "objective_weight"] = 0.0
   return data, problem, open_ids, assignment_df



# -----------------------------------------------------------------------------
# RUN ONE OR MORE WEIGHTED SCENARIOS AND WRITE OUTPUTS
# -----------------------------------------------------------------------------
# This wrapper solves each requested scenario, writes the standard output files
# and creates a scenario_comparison.csv file.
def run_weighted_model(
   b: int,
   scenarios: list[str],
   output_root: Path | None = None,
   time_limit: int = 600,
   msg: bool = False,
):
   output_dir = ensure_results_dir(MODEL_NAME) if output_root is None else Path(output_root)
   summaries: list[dict[str, object]] = []

   # Solve each requested benefit-function scenario one by one.
   for scenario in scenarios:
       data, problem, open_ids, assignment_df = solve_weighted_assignment(b=b, scenario=scenario, time_limit=time_limit, msg=msg)
       objective_value = float(pulp.value(problem.objective) or 0.0)
       summary, selected_facilities, population_summary = build_run_summary(
           model_name=MODEL_NAME,
           scenario=scenario,
           b=b,
           facilities=data.facilities,
           open_facility_ids=open_ids,
           assignment_df=assignment_df,
           objective_value=objective_value,
           total_demand_weight=float(data.population["household_count"].sum()),
           weight_scenario=scenario,
       )
       summary["objective_value"] = objective_value
       summary["total_demand_weight"] = float(data.population["household_count"].sum())
       file_prefix = f"weighted_{scenario}_b{b}"
       write_run_outputs(output_dir, file_prefix, summary, selected_facilities, population_summary)
       summaries.append(summary)


   pd.DataFrame(summaries).to_csv(output_dir / "scenario_comparison.csv", index=False)
   return summaries



# -----------------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# -----------------------------------------------------------------------------
# Parse command-line settings so the model can be run from Terminal.
def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description="Solve the weighted thesis upgrade model.")
   parser.add_argument("--b", type=int, default=1, help="Number of upgrades to select.")
   parser.add_argument(
       "--weight-scenarios",
       type=str,
       default="exponential,logistic",
       help="Comma-separated weight scenarios to solve.",
   )
   parser.add_argument("--time-limit", type=int, default=600, help="Solver time limit in seconds.")
   parser.add_argument("--msg", action="store_true", help="Show solver output.")
   return parser.parse_args()


def main() -> None:
   args = parse_args()
   scenarios = [item.strip() for item in args.weight_scenarios.split(",") if item.strip()]
   run_weighted_model(b=args.b, scenarios=scenarios, time_limit=args.time_limit, msg=args.msg)


if __name__ == "__main__":
   main()
