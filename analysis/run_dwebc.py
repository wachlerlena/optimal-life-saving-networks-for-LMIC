"""
CLI entrypoint for Step 3: Demand-Weighted Edge Betweenness Centrality.

Usage (from project root):
    python -m analysis.run_dwebc                     # full 406k-row run
    python -m analysis.run_dwebc --sample 5000       # smoke test on a sample
    python -m analysis.run_dwebc --min-lat 21        # only northern band
    python -m analysis.run_dwebc --force-rebuild     # ignore caches

Outputs (analysis/outputs/):
    edge_centrality.pkl   -- per-edge C(e) with metadata
    pop_assignment.pkl    -- per-population nearest hospital + path stats
    unreachable.pkl       -- subset of populated rows with no path / >300km
    dwebc_summary.txt         -- human-readable run summary
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config
from analysis.data_loaders import (
    load_hospitals_snapped,
    load_or_build_graph,
    load_population_snapped,
)
from analysis.dwebc import compute_dwebc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: Demand-Weighted Edge Betweenness Centrality")
    p.add_argument("--sample", type=int, default=None,
                   help="If set, sample this many population rows (with seed=0).")
    p.add_argument("--min-lat", type=float, default=None,
                   help="Drop population rows with lat below this value.")
    p.add_argument("--max-lat", type=float, default=None,
                   help="Drop population rows with lat above this value.")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Ignore on-disk caches for graph and snapped points.")
    p.add_argument("--output-suffix", type=str, default="",
                   help="Append this suffix to all output filenames (e.g. _smoke).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    print("=" * 70)
    print("Step 3: Demand-Weighted Edge Betweenness Centrality")
    print("=" * 70)

    # ----- Load graph + snapped points -----
    G, node_list, tree = load_or_build_graph(force=args.force_rebuild)
    pop_df = load_population_snapped(tree, node_list, force=args.force_rebuild)
    hosp_df = load_hospitals_snapped(tree, node_list, force=args.force_rebuild)

    # ----- Optional subsetting for validation / smoke tests -----
    if args.min_lat is not None:
        before = len(pop_df)
        pop_df = pop_df[pop_df["lat"] >= args.min_lat].copy()
        print(f"[run] min_lat filter: {before} -> {len(pop_df)}")
    if args.max_lat is not None:
        before = len(pop_df)
        pop_df = pop_df[pop_df["lat"] <= args.max_lat].copy()
        print(f"[run] max_lat filter: {before} -> {len(pop_df)}")
    if args.sample is not None:
        n = min(args.sample, len(pop_df))
        pop_df = pop_df.sample(n=n, random_state=0).reset_index(drop=True)
        print(f"[run] sampled {n} population rows for validation")

    # ----- DWEBC -----
    result = compute_dwebc(G, pop_df, hosp_df)

    # ----- Save outputs -----
    suffix = args.output_suffix
    edge_out = config.OUTPUT_DIR / f"edge_centrality{suffix}.pkl"
    pop_out = config.OUTPUT_DIR / f"pop_assignment{suffix}.pkl"
    unr_out = config.OUTPUT_DIR / f"unreachable{suffix}.pkl"
    summary_out = config.OUTPUT_DIR / f"dwebc_summary{suffix}.txt"

    result.edge_centrality.to_pickle(edge_out)
    result.pop_assignment.to_pickle(pop_out)
    result.unreachable.to_pickle(unr_out)

    # ----- Summary -----
    ec = result.edge_centrality
    pa = result.pop_assignment
    populated = pa[pa["household_count"] > 0]
    reachable = populated[populated["reachable_300km"]]

    lines = []
    lines.append("DWEBC run summary")
    lines.append("=" * 50)
    lines.append(f"runtime_seconds            : {time.time()-t_start:.1f}")
    lines.append(f"graph_nodes                : {G.number_of_nodes()}")
    lines.append(f"graph_edges                : {G.number_of_edges()}")
    lines.append(f"population_rows_in         : {len(pa)}")
    lines.append(f"populated_rows_w_i_gt_0    : {len(populated)}")
    lines.append(f"reachable_within_300km     : {len(reachable)}")
    lines.append(f"unreachable (no-path/>300) : {len(result.unreachable)}")
    lines.append("")
    lines.append("centrality C(e) distribution:")
    lines.append(str(ec["centrality"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999])))
    lines.append("")

    nonzero_edges = ec[ec["centrality"] > 0]
    lines.append(f"edges with C(e) > 0        : {len(nonzero_edges)} / {len(ec)} "
                 f"({100.0 * len(nonzero_edges) / max(len(ec), 1):.1f}%)")
    if len(nonzero_edges) > 0:
        lines.append(f"top 1% C(e) threshold      : {ec['centrality'].quantile(0.99):.1f}")
        lines.append(f"top 5% C(e) threshold      : {ec['centrality'].quantile(0.95):.1f}")
        lines.append(f"top 10% C(e) threshold     : {ec['centrality'].quantile(0.90):.1f}")
    lines.append("")
    if len(reachable) > 0:
        lines.append("travel_time_min (reachable) distribution:")
        lines.append(str(reachable["travel_time_min"].describe(
            percentiles=[0.5, 0.9, 0.95, 0.99])))
        lines.append("")
        lines.append("total_distance_km (reachable) distribution:")
        lines.append(str(reachable["total_distance_km"].describe(
            percentiles=[0.5, 0.9, 0.95, 0.99])))

    summary = "\n".join(lines)
    summary_out.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"[run] wrote: {edge_out.name}, {pop_out.name}, {unr_out.name}, {summary_out.name}")


if __name__ == "__main__":
    main()
