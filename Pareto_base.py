import pandas as pd
import pulp

from optimize_upgrade_base_mclp import solve_base_mclp

EXISTING_ADVANCED_CENTERS = 55

rows = []

for b in range(0, 76):
    data, problem, open_ids = solve_base_mclp(
        b=b,
        msg=False,
    )

    assignment_df = data.arcs[data.arcs["facility_id"].isin(open_ids)].copy()
    assignment_df = (
        assignment_df
        .sort_values(["pop_id", "travel_minutes", "facility_id"])
        .drop_duplicates("pop_id")
    )

    assignment_df = data.population.merge(
        assignment_df[["pop_id", "travel_minutes", "facility_id"]],
        on="pop_id",
        how="left",
    )

    assignment_df["travel_minutes"] = assignment_df["travel_minutes"].fillna(999.0)

    total_pop = assignment_df["household_count"].sum()

    selected_facilities = sorted(set(open_ids) - set(data.fixed_facility_ids))

    covered_90 = assignment_df.loc[
        assignment_df["travel_minutes"] <= 90,
        "household_count"
    ].sum()

    covered_180 = assignment_df.loc[
        assignment_df["travel_minutes"] <= 180,
        "household_count"
    ].sum()

    covered_270 = assignment_df.loc[
        assignment_df["travel_minutes"] <= 270,
        "household_count"
    ].sum()

    gt_270 = assignment_df.loc[
        assignment_df["travel_minutes"] > 270,
        "household_count"
    ].sum()

    rows.append({
        "model": "base_mclp",
        "b": b,
        "total_advanced_centers": EXISTING_ADVANCED_CENTERS + b,
        "objective_value": float(pulp.value(problem.objective) or 0),
        "covered_90_pct": 100 * covered_90 / total_pop,
        "covered_180_pct": 100 * covered_180 / total_pop,
        "covered_270_pct": 100 * covered_270 / total_pop,
        "share_gt_270_pct": 100 * gt_270 / total_pop,
        "selected_facility_ids": ",".join(map(str, selected_facilities)),
    })

results = pd.DataFrame(rows)
results.to_csv("base_pareto_results.csv", index=False)

print("Saved: base_pareto_results.csv")