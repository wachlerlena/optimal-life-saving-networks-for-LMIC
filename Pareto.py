
import pandas as pd
import matplotlib.pyplot as plt
import pulp

from optimize_upgrade_weighted_assignment import solve_weighted_assignment

EXISTING_ADVANCED_CENTERS = 55
MAXIMUM_COVERED_90 = 0.8956701628820758
MAX_B = 75
MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP = 77643797.03586955
MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG = 77991149.62346442

rows = []

for scenario in ["exponential", "logistic"]:
    for b in range(0, 76):
        data, problem, open_ids, assignment_df = solve_weighted_assignment(
            b=b,
            scenario=scenario,
            msg=False,
        )

        total_pop = assignment_df["household_count"].sum()

        selected_facilities = sorted(set(open_ids) - set(data.fixed_facility_ids))

        rows.append({
            "scenario": scenario,
            "b": b,
            "total_advanced_centers": EXISTING_ADVANCED_CENTERS + b,
            "objective_value": float(pulp.value(problem.objective) or 0),
            "covered_90_pct": 100 * assignment_df.loc[
                assignment_df["best_travel_minutes"] <= 90,
                "household_count"
            ].sum() / total_pop,
            "covered_180_pct": 100 * assignment_df.loc[
                assignment_df["best_travel_minutes"] <= 180,
                "household_count"
            ].sum() / total_pop,
            "covered_270_pct": 100 * assignment_df.loc[
                assignment_df["best_travel_minutes"] <= 270,
                "household_count"
            ].sum() / total_pop,
            "share_gt_270_pct": 100 * assignment_df.loc[
                assignment_df["best_travel_minutes"] > 270,
                "household_count"
            ].sum() / total_pop,
            "selected_facility_ids": ",".join(map(str, selected_facilities)),
        })



pareto = pd.DataFrame(rows)
pareto["objective_value_million"] = pareto["objective_value"] / 1_000_000
MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP_M = MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP / 1_000_000
MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG_M = MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG / 1_000_000
pareto.to_csv("pareto_curve_results.csv", index=False)
