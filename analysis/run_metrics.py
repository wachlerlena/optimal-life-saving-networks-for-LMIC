"""
Step 4 + 5 CLI - build disrupted scenarios and compute accessibility metrics.

Inputs (must exist; produced by run_dwebc.py):
    analysis/outputs/edge_centrality.pkl
    analysis/outputs/pop_assignment.pkl   (only used for sanity; we re-snap)

Outputs (analysis/outputs/):
    travel_times_wide.pkl      -- per-population travel time for each (scenario, tier).
                                   Columns: pop_id, household_count, region,
                                            lat, lon,
                                            tt_baseline_any, tt_A_1pct_any, ...,
                                            tt_baseline_comprehensive, ...
    tt_summary.csv             -- (scenario x tier x region) -> weighted mean / p50 / p95 / unreachable share
    catchment.csv              -- (scenario x tier x region x cutoff_min) -> weight share reachable
    catchment_deltas.csv       -- catchment share difference vs baseline
    tt_deltas.csv              -- weighted mean / p50 / p95 deltas vs baseline
    metrics_summary.txt        -- human-readable headline numbers
"""
from __future__ import annotations

import argparse
import pickle
import time

import pandas as pd

from analysis import config
from analysis.data_loaders import (
    assign_region,
    load_hospitals_snapped,
    load_or_build_graph,
    load_population_snapped,
)
from analysis.disruption import build_all_scenarios
from analysis.metrics import (
    compute_tier_travel_times,
    isochrone_catchment_share,
    travel_time_summary,
)

TIERS = ("any", "comprehensive")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steps 4-5: disruption scenarios + accessibility metrics")
    p.add_argument("--edge-centrality",
                   default=str(config.OUTPUT_DIR / "edge_centrality.pkl"),
                   help="Path to edge centrality pickle from run_dwebc.py")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    print("=" * 70)
    print("Steps 4-5: Disruption scenarios + accessibility metrics")
    print("=" * 70)

    # ----- Load graph + snapped points -----
    G, node_list, tree = load_or_build_graph(force=args.force_rebuild)
    pop_df = load_population_snapped(tree, node_list, force=args.force_rebuild)
    hosp_df = load_hospitals_snapped(tree, node_list, force=args.force_rebuild)

    # Region label (vectorized over lat).
    pop_df = pop_df.copy()
    pop_df["region"] = assign_region(pop_df, lat_col="lat")
    print(f"[run] region split:\n{pop_df['region'].value_counts().to_string()}")

    # ----- Edge centrality -----
    print(f"[run] loading edge centrality: {args.edge_centrality}")
    edge_centrality_df = pd.read_pickle(args.edge_centrality)
    if not edge_centrality_df["centrality"].is_monotonic_decreasing:
        edge_centrality_df = edge_centrality_df.sort_values("centrality", ascending=False).reset_index(drop=True)
    print(f"[run]   {len(edge_centrality_df)} edges, top C(e) = {edge_centrality_df['centrality'].max():.0f}")

    # ----- Step 4: build scenarios -----
    scenarios = build_all_scenarios(G, edge_centrality_df, config.DISRUPTION_FRACTIONS)
    scenario_order = list(scenarios.keys())  # baseline, A_1pct, B_5pct, C_10pct

    # ----- Step 5: per-(scenario,tier) travel-time runs -----
    weights = pop_df.set_index("ID")["household_count"]
    regions = pop_df.set_index("ID")["region"]

    travel_times_by_run: dict[tuple[str, str], pd.Series] = {}
    tt_summary_rows = []
    catchment_rows = []

    for scen_name in scenario_order:
        scen = scenarios[scen_name]
        for tier in TIERS:
            ttr = compute_tier_travel_times(scen.graph, pop_df, hosp_df, tier, scen_name)
            travel_times_by_run[(scen_name, tier)] = ttr.travel_time_min

            tt_sum = travel_time_summary(ttr.travel_time_min, weights, regions)
            tt_sum = tt_sum.assign(scenario=scen_name, tier=tier).reset_index()
            tt_summary_rows.append(tt_sum)

            for region_label in list(regions.unique()) + ["Overall"]:
                if region_label == "Overall":
                    mask = pd.Series(True, index=ttr.travel_time_min.index)
                else:
                    mask = (regions == region_label)
                sub_tt = ttr.travel_time_min[mask]
                sub_w = weights[mask]
                catch = isochrone_catchment_share(sub_tt, sub_w)
                for cutoff, share in catch.items():
                    catchment_rows.append({
                        "scenario": scen_name,
                        "tier": tier,
                        "region": region_label,
                        "cutoff_min": cutoff,
                        "reachable_weight_share": share,
                    })

    tt_summary = pd.concat(tt_summary_rows, ignore_index=True)
    tt_summary = tt_summary[[
        "scenario", "tier", "region",
        "total_weight", "reachable_weight", "unreachable_share",
        "tt_weighted_mean", "tt_weighted_p50", "tt_weighted_p95",
    ]]

    catchment = pd.DataFrame(catchment_rows)

    # ----- Wide per-population table for inspection / plotting -----
    wide = pop_df[["ID", "lat", "lon", "household_count", "region"]].copy()
    wide = wide.rename(columns={"ID": "pop_id"}).set_index("pop_id")
    for (scen_name, tier), series in travel_times_by_run.items():
        wide[f"tt_{scen_name}_{tier}"] = series
    wide = wide.reset_index()

    # ----- Deltas vs baseline -----
    tt_deltas = tt_summary.merge(
        tt_summary[tt_summary["scenario"] == "baseline"]
            [["tier", "region", "tt_weighted_mean", "tt_weighted_p50",
              "tt_weighted_p95", "unreachable_share"]]
            .rename(columns={
                "tt_weighted_mean": "tt_weighted_mean_baseline",
                "tt_weighted_p50": "tt_weighted_p50_baseline",
                "tt_weighted_p95": "tt_weighted_p95_baseline",
                "unreachable_share": "unreachable_share_baseline",
            }),
        on=["tier", "region"],
        how="left",
    )
    tt_deltas["delta_mean_min"] = tt_deltas["tt_weighted_mean"] - tt_deltas["tt_weighted_mean_baseline"]
    tt_deltas["delta_p50_min"] = tt_deltas["tt_weighted_p50"] - tt_deltas["tt_weighted_p50_baseline"]
    tt_deltas["delta_p95_min"] = tt_deltas["tt_weighted_p95"] - tt_deltas["tt_weighted_p95_baseline"]
    tt_deltas["delta_unreachable_share"] = (
        tt_deltas["unreachable_share"] - tt_deltas["unreachable_share_baseline"]
    )

    baseline_catch = catchment[catchment["scenario"] == "baseline"].rename(
        columns={"reachable_weight_share": "baseline_share"}
    )[["tier", "region", "cutoff_min", "baseline_share"]]
    catchment_deltas = catchment.merge(
        baseline_catch, on=["tier", "region", "cutoff_min"], how="left"
    )
    catchment_deltas["share_drop"] = (
        catchment_deltas["baseline_share"] - catchment_deltas["reachable_weight_share"]
    )
    catchment_deltas["share_drop_pct_of_baseline"] = (
        catchment_deltas["share_drop"] / catchment_deltas["baseline_share"].replace(0, pd.NA)
    )

    # ----- Save -----
    out_dir = config.OUTPUT_DIR
    wide.to_pickle(out_dir / "travel_times_wide.pkl")
    tt_summary.to_csv(out_dir / "tt_summary.csv", index=False)
    catchment.to_csv(out_dir / "catchment.csv", index=False)
    catchment_deltas.to_csv(out_dir / "catchment_deltas.csv", index=False)
    tt_deltas.to_csv(out_dir / "tt_deltas.csv", index=False)

    # ----- Headline text summary -----
    lines = []
    lines.append("Steps 4-5 metrics summary")
    lines.append("=" * 60)
    lines.append(f"runtime_seconds   : {time.time()-t_start:.1f}")
    lines.append(f"population_rows   : {len(pop_df)}")
    lines.append(f"scenarios         : {scenario_order}")
    lines.append("")
    lines.append("Scenario edge counts:")
    for name in scenario_order:
        s = scenarios[name]
        lines.append(f"  {name:<10} edges={s.graph.number_of_edges():>7}  "
                     f"removed={s.removed_edge_count:>6}  "
                     f"threshold={s.centrality_threshold:.0f}"
                     if s.fraction_removed > 0 else
                     f"  {name:<10} edges={s.graph.number_of_edges():>7}  (no removal)")
    lines.append("")
    lines.append("OVERALL travel-time summary (weighted by household_count):")
    overall = tt_summary[tt_summary["region"] == "Overall"][[
        "scenario", "tier", "tt_weighted_mean", "tt_weighted_p50",
        "tt_weighted_p95", "unreachable_share",
    ]].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    lines.append(overall)
    lines.append("")
    lines.append("OVERALL catchment share (population weight reachable within cutoff):")
    overall_catch = catchment[catchment["region"] == "Overall"].pivot_table(
        index=["scenario", "tier"], columns="cutoff_min",
        values="reachable_weight_share"
    )
    lines.append(overall_catch.to_string(float_format=lambda x: f"{x:.3f}"))
    lines.append("")
    lines.append("Files written:")
    for f in ["travel_times_wide.pkl", "tt_summary.csv", "catchment.csv",
              "catchment_deltas.csv", "tt_deltas.csv"]:
        lines.append(f"  {out_dir / f}")

    summary = "\n".join(lines)
    (out_dir / "metrics_summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":
    main()
