"""
Step 6 - Edge-reinforcement optimisation (greedy heuristic vs random baseline).

This step builds on the Step 3-5 pipeline. It asks: under the worst disruption
scenario (Scenario C = top 10% of edges by demand-weighted centrality removed),
which edges should we *reinforce* (protect from failure) to recover the most
1-hour stroke-care coverage, given a fixed budget K?

Model (see methods 3.5)
-----------------------
  E        all edges in the road network.
  E*       candidate set = the Scenario C edges (top 10% by C(e)). These are the
           edges that fail under Scenario C and are therefore the only edges
           worth reinforcing.
  x_e      binary decision: 1 if edge e in E* is reinforced (stays intact under
           Scenario C), 0 if it is left to fail.
  Budget   sum_e x_e <= K.
  Objective: maximise P(60) = population weight within 60 min of any stroke
           centre, evaluated on the graph (full network minus the unreinforced
           E* edges).

The full MILP is intractable at 268k-node scale, so we solve it with a greedy
heuristic: reinforce the K edges with the highest centrality C(e). We compare
this against a random baseline (K edges drawn uniformly from E*, repeated
n_random_runs times) to show the centrality ranking is an actionable
prioritisation, not just chance.

Implementation note
-------------------
Rather than copying the whole graph for every budget level (slow), we build the
Scenario C graph once (full network with all E* edges removed) and then *add
back* the protected edges for each evaluation, removing them again afterwards.
Adding then removing the same edges restores Scenario C exactly.

Inputs (must exist; produced by run_dwebc.py / run_metrics.py and their caches):
    analysis/outputs/edge_centrality.pkl
    analysis/outputs/cache/road_graph.pkl  (+ node_index, *_snapped caches)

Outputs (analysis/outputs/):
    optimisation_results.csv      -- coverage at each budget for greedy + random (mean/std)
    optimisation_random_runs.csv  -- every individual random run (reproducibility)
    optimisation_summary.txt      -- human-readable headline numbers
    figures/fig6_recovery_curve.png

Run from the project root:
    python -m analysis.run_optimisation
    python -m analysis.run_optimisation --n-random-runs 5   # fast smoke test
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from analysis import config
from analysis.data_loaders import (
    assign_region,
    load_hospitals_snapped,
    load_or_build_graph,
    load_population_snapped,
)
from analysis.metrics import (
    compute_tier_travel_times,
    isochrone_catchment_share,
)

# Scenario C candidate fraction (top 10% of edges by C(e)).
CANDIDATE_FRACTION = 0.10

# Budget levels expressed as a fraction of the candidate set E* (Logic B).
# 0.0 = Scenario C floor (nothing reinforced); 1.0 = full protection ceiling.
DEFAULT_BUDGET_FRACTIONS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)

# 1-hour coverage is the headline objective.
COVERAGE_CUTOFF_MIN = 60


def _log(msg: str) -> None:
    print(f"[optimisation] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 6: edge-reinforcement optimisation (greedy vs random)"
    )
    p.add_argument("--edge-centrality",
                   default=str(config.OUTPUT_DIR / "edge_centrality.pkl"),
                   help="Path to edge centrality pickle from run_dwebc.py")
    p.add_argument("--tier", default="any", choices=("any", "comprehensive"),
                   help="Hospital tier the coverage objective is measured against")
    p.add_argument("--n-random-runs", type=int, default=30,
                   help="Random-baseline repetitions per budget level")
    p.add_argument("--seed", type=int, default=42,
                   help="Base RNG seed for the random baseline (reproducible)")
    p.add_argument("--fine-sweep", action="store_true",
                   help="Dense greedy-only sweep to locate the optimal budget K "
                        "(knee / target / average-slope cutoffs). Skips the random baseline.")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def collect_candidate_edge_records(G, edge_centrality_df: pd.DataFrame):
    """
    Return the E* edge records (u, v, attr_dict) that actually exist in G, in
    descending C(e) order. These are the Scenario C edges and the only edges
    eligible for reinforcement.
    """
    n_total = len(edge_centrality_df)
    n_candidate = int(round(CANDIDATE_FRACTION * n_total))
    if n_candidate <= 0:
        raise ValueError(f"candidate fraction yields zero edges from {n_total}")

    estar = edge_centrality_df.iloc[:n_candidate]
    threshold = float(estar["centrality"].iloc[-1])

    records = []
    skipped = 0
    for _, row in estar.iterrows():
        u = (row["u_lon"], row["u_lat"])
        v = (row["v_lon"], row["v_lat"])
        if G.has_edge(u, v):
            records.append((u, v, dict(G[u][v])))
        else:
            skipped += 1

    _log(f"E*: {n_candidate} candidate edges (top {CANDIDATE_FRACTION:.0%}), "
         f"{len(records)} present in graph, {skipped} not-in-graph, "
         f"threshold C(e) >= {threshold:.0f}")
    return records, n_candidate, threshold


def shares_from_tt(tt, weights, regions, region_labels) -> dict:
    """
    Derive 1-hour coverage shares (Overall + per region) from a single
    travel-time series. No extra routing: the Dijkstra was already run to
    produce `tt`, so each region is just a mask over the same series.
    """
    out = {}
    for rl in region_labels:
        if rl == "Overall":
            mask = pd.Series(True, index=tt.index)
        else:
            mask = (regions.reindex(tt.index) == rl)
        sub_tt = tt[mask]
        sub_w = weights.reindex(tt.index)[mask]
        share = isochrone_catchment_share(
            sub_tt, sub_w, cutoffs_min=(COVERAGE_CUTOFF_MIN,)
        )
        out[rl] = share[COVERAGE_CUTOFF_MIN]
    return out


def coverage_on_graph(G, pop_df, hosp_df, tier, weights, regions, region_labels, label):
    """Per-region 1-hour coverage of `tier` hospitals on graph G."""
    ttr = compute_tier_travel_times(G, pop_df, hosp_df, tier, label)
    return shares_from_tt(ttr.travel_time_min, weights, regions, region_labels)


def _fine_K_grid(m: int) -> list[int]:
    """
    Budget points for the fine sweep: dense in 0-10% of E* (where returns are
    steepest, so the knee lives there), coarser out to 100%. Returns a sorted
    list of unique integer K values including 0 and m.
    """
    fine_step = max(1, int(round(0.0025 * m)))    # ~0.25% of E*
    coarse_step = max(1, int(round(0.025 * m)))   # ~2.5% of E*
    ks = set([0, m])
    ks.update(range(0, int(round(0.10 * m)) + 1, fine_step))
    ks.update(range(int(round(0.10 * m)), m + 1, coarse_step))
    return sorted(k for k in ks if 0 <= k <= m)


def find_knee_index(K_arr: np.ndarray, cov_arr: np.ndarray) -> int:
    """
    Kneedle for a monotone-increasing, concave curve: normalise both axes to
    [0,1] and return the index of maximum vertical distance above the chord
    joining the endpoints (which, after normalisation, is the line y = x).
    """
    x = K_arr.astype(float)
    y = cov_arr.astype(float)
    x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
    y = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0
    return int(np.argmax(y - x))


def run_fine_sweep(G, records, pop_df, hosp_df, weights, regions, region_labels,
                   tier, out_dir, t_start) -> None:
    """
    Greedy-only dense sweep of P_60(K) to locate the 'optimal' reinforcement
    budget. Reports three cutoffs on the same curve:
      (1) knee      : Kneedle max-curvature point (diminishing-returns elbow);
      (2) target    : smallest K closing >= {50,80,90}% of the recoverable gap;
      (3) avg-slope : largest K whose marginal gain per edge still exceeds the
                      network-wide average slope (ceiling-floor)/m.
    """
    m = len(records)

    # Intact ceiling (full graph) and Scenario C floor (all E* removed).
    intact_shares = coverage_on_graph(
        G, pop_df, hosp_df, tier, weights, regions, region_labels, "intact")
    ceiling = intact_shares["Overall"]

    G_C = G.copy()
    for u, v, _ in records:
        G_C.remove_edge(u, v)

    K_grid = _fine_K_grid(m)
    _log(f"fine sweep: {len(K_grid)} budget points over [0, {m}] edges "
         f"(greedy only, no random baseline)")

    rows = []
    added = 0  # nested greedy: edges are added incrementally, never removed.
    for K in K_grid:
        while added < K:
            u, v, data = records[added]
            G_C.add_edge(u, v, **data)
            added += 1
        shares = coverage_on_graph(
            G_C, pop_df, hosp_df, tier, weights, regions, region_labels, f"greedy_K{K}")
        row = {"K": K, "budget_fraction_of_Estar": K / m,
               "greedy_coverage_1h": shares["Overall"]}
        for rl in region_labels:
            if rl != "Overall":
                row[f"coverage_{rl}"] = shares[rl]
        rows.append(row)
        _log(f"  K={K:>6} ({K/m:>6.2%}): P(60)={shares['Overall']:.4f}")

    curve = pd.DataFrame(rows)
    floor = float(curve.loc[curve["K"] == 0, "greedy_coverage_1h"].iloc[0])
    gap = ceiling - floor

    K_arr = curve["K"].to_numpy(dtype=float)
    cov_arr = curve["greedy_coverage_1h"].to_numpy(dtype=float)
    curve["gap_closed"] = (cov_arr - floor) / gap if gap > 0 else np.nan
    # Marginal gain per edge between consecutive sampled budgets.
    curve["marginal_per_edge"] = np.gradient(cov_arr, K_arr)
    curve["intact_ceiling_1h"] = ceiling
    curve["scenarioC_floor_1h"] = floor

    # ----- Cutoff 1: Kneedle knee -----
    knee_i = find_knee_index(K_arr, cov_arr)
    knee_K = int(K_arr[knee_i])

    # ----- Cutoff 2: target gap closed -----
    target_rows = {}
    gc = curve["gap_closed"].to_numpy()
    for t in (0.50, 0.80, 0.90):
        hit = np.where(gc >= t)[0]
        target_rows[t] = int(K_arr[hit[0]]) if len(hit) else None

    # ----- Cutoff 3: average-slope crossing -----
    avg_slope = gap / m if m > 0 else np.nan  # per-edge gain over the full range
    above = curve["marginal_per_edge"].to_numpy() >= avg_slope
    # Largest K (from the start) whose local marginal still beats the average.
    avgslope_K = int(K_arr[np.where(above)[0][-1]]) if above.any() else 0

    def _at(K):
        r = curve.loc[curve["K"] == K].iloc[0]
        return r["greedy_coverage_1h"], r["gap_closed"]

    # ----- Save curve + cutoffs -----
    curve.to_csv(out_dir / "optimisation_fine_curve.csv", index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        xs = curve["budget_fraction_of_Estar"] * 100.0
        ax1.plot(xs, cov_arr * 100.0, "-", color="#1f6f3a", lw=2, label="Greedy P(60)")
        ax1.axhline(ceiling * 100, color="#333", ls=":", lw=1, label=f"Ceiling {ceiling*100:.1f}%")
        ax1.axhline(floor * 100, color="#888", ls=":", lw=1, label=f"Floor {floor*100:.1f}%")
        ax1.axvline(knee_K / m * 100, color="#b03030", ls="--", lw=1.5,
                    label=f"Knee K={knee_K} ({knee_K/m:.1%})")
        if target_rows[0.80] is not None:
            ax1.axvline(target_rows[0.80] / m * 100, color="#d08000", ls="--", lw=1.5,
                        label=f"80% gap K={target_rows[0.80]}")
        ax1.axvline(avgslope_K / m * 100, color="#3060b0", ls="--", lw=1.5,
                    label=f"Avg-slope K={avgslope_K}")
        ax1.set_ylabel(f"1-hour coverage ('{tier}', % pop. weight)")
        ax1.set_title("Fine recovery curve and candidate optimal budgets K")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(xs, curve["marginal_per_edge"] * 1e3, "-", color="#6a3d9a", lw=2,
                 label="Marginal gain per edge")
        ax2.axhline(avg_slope * 1e3, color="#3060b0", ls=":", lw=1.2,
                    label="Network-wide average slope")
        ax2.set_xlabel("Budget: % of candidate edges E* reinforced")
        ax2.set_ylabel("Marginal coverage gain\nper edge (pp x10^-3)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig6c_marginal_returns.png", dpi=150)
        plt.close(fig)
        _log(f"wrote {fig_dir / 'fig6c_marginal_returns.png'}")
    except Exception as e:  # pragma: no cover
        _log(f"WARNING: could not render marginal-returns plot: {e}")

    # ----- Text report -----
    lines = []
    lines.append("Step 6 - optimal reinforcement budget K (fine greedy sweep)")
    lines.append("=" * 64)
    lines.append(f"runtime_seconds      : {time.time()-t_start:.1f}")
    lines.append(f"tier (objective)     : {tier}")
    lines.append(f"candidate set |E*|   : {m} edges")
    lines.append(f"sweep budget points  : {len(K_grid)}")
    lines.append(f"intact ceiling       : {ceiling*100:.2f}%")
    lines.append(f"Scenario C floor     : {floor*100:.2f}%")
    lines.append(f"recoverable gap      : {gap*100:.2f} pp")
    lines.append("")
    lines.append("Candidate 'optimal' budgets (most coverage per edge):")
    lines.append("-" * 64)
    kc, kg = _at(knee_K)
    lines.append(f"[1] Knee (Kneedle)        : K={knee_K} ({knee_K/m:.1%} of E*)  "
                 f"-> P(60)={kc*100:.2f}%  gap closed={kg*100:.1f}%")
    for t in (0.50, 0.80, 0.90):
        K = target_rows[t]
        if K is None:
            lines.append(f"[2] Target {int(t*100)}% gap closed : not reached")
        else:
            c, g_ = _at(K)
            lines.append(f"[2] Target {int(t*100)}% gap closed : K={K} ({K/m:.1%} of E*)  "
                         f"-> P(60)={c*100:.2f}%  gap closed={g_*100:.1f}%")
    ac, ag = _at(avgslope_K)
    lines.append(f"[3] Avg-slope crossing    : K={avgslope_K} ({avgslope_K/m:.1%} of E*)  "
                 f"-> P(60)={ac*100:.2f}%  gap closed={ag*100:.1f}%")
    lines.append("    (largest K whose marginal gain per edge still beats the "
                 "network-wide average)")
    lines.append("")
    lines.append("Files written:")
    for f in ["optimisation_fine_curve.csv", "figures/fig6c_marginal_returns.png"]:
        lines.append(f"  {out_dir / f}")

    summary = "\n".join(lines)
    (out_dir / "optimisation_optimalK.txt").write_text(summary, encoding="utf-8")
    print()
    print(summary)


def main() -> None:
    args = parse_args()
    t_start = time.time()

    print("=" * 70)
    print("Step 6: Edge-reinforcement optimisation (greedy heuristic vs random)")
    print("=" * 70)

    # ----- Load graph + snapped points (reuse Step 3-5 caches) -----
    G, node_list, tree = load_or_build_graph(force=args.force_rebuild)
    pop_df = load_population_snapped(tree, node_list, force=args.force_rebuild)
    hosp_df = load_hospitals_snapped(tree, node_list, force=args.force_rebuild)
    pop_df = pop_df.copy()
    pop_df["region"] = assign_region(pop_df, lat_col="lat")
    weights = pop_df.set_index("ID")["household_count"]
    regions = pop_df.set_index("ID")["region"]
    # Overall first, then each region in fixed North -> South order.
    present = [r for r in ("North", "Central", "South", "Unknown")
               if r in set(regions.unique())]
    region_labels = ["Overall"] + present

    # ----- Edge centrality (already descending by C(e)) -----
    _log(f"loading edge centrality: {args.edge_centrality}")
    edge_centrality_df = pd.read_pickle(args.edge_centrality)
    if not edge_centrality_df["centrality"].is_monotonic_decreasing:
        edge_centrality_df = (edge_centrality_df
                              .sort_values("centrality", ascending=False)
                              .reset_index(drop=True))

    records, n_candidate, threshold = collect_candidate_edge_records(G, edge_centrality_df)
    n_estar = len(records)  # actual reinforceable edges present in graph

    # ----- Optional: dense greedy sweep to locate the optimal budget K -----
    if args.fine_sweep:
        run_fine_sweep(G, records, pop_df, hosp_df, weights, regions, region_labels,
                       args.tier, config.OUTPUT_DIR, t_start)
        return

    # ----- Reference points: intact ceiling + Scenario C floor -----
    intact_shares = coverage_on_graph(
        G, pop_df, hosp_df, args.tier, weights, regions, region_labels, "intact")
    intact_cov = intact_shares["Overall"]
    _log(f"intact-network ceiling P({COVERAGE_CUTOFF_MIN}) = {intact_cov:.4f}")

    # Build Scenario C graph once: remove every reinforceable E* edge.
    G_C = G.copy()
    for u, v, _ in records:
        G_C.remove_edge(u, v)
    _log(f"Scenario C graph: {G_C.number_of_edges()} edges "
         f"({n_estar} candidate edges removed)")

    def shares_with_protected(protected_records, label) -> dict:
        """Per-region coverage when only `protected_records` of E* are reinforced."""
        for u, v, data in protected_records:
            G_C.add_edge(u, v, **data)
        try:
            return coverage_on_graph(
                G_C, pop_df, hosp_df, args.tier, weights, regions, region_labels, label)
        finally:
            for u, v, _ in protected_records:
                G_C.remove_edge(u, v)

    # ----- Budget sweep -----
    budget_fractions = DEFAULT_BUDGET_FRACTIONS
    results_rows = []         # overall, one row per budget
    regional_rows = []        # (budget x region), greedy + random mean/std
    random_run_rows = []      # every individual random run (overall)

    for frac in budget_fractions:
        K = int(round(frac * n_estar))
        K = min(K, n_estar)

        # --- Greedy: protect the top-K edges by C(e) (records are pre-sorted). ---
        greedy_protected = records[:K]
        greedy_shares = shares_with_protected(greedy_protected, f"greedy_K{K}")
        greedy_cov = greedy_shares["Overall"]

        # --- Random baseline: K edges drawn uniformly from E*. ---
        # Collect per-region shares across runs so 6.3 can compare greedy vs random
        # within each region (the random Dijkstra is reused for every region).
        random_shares = {rl: [] for rl in region_labels}
        if K == 0 or K == n_estar:
            # Degenerate: random == greedy (protect none / protect all).
            for rl in region_labels:
                random_shares[rl].append(greedy_shares[rl])
        else:
            for r in range(args.n_random_runs):
                rng = np.random.default_rng(args.seed + 1000 * len(results_rows) + r)
                idx = rng.choice(n_estar, size=K, replace=False)
                rand_protected = [records[i] for i in idx]
                rs = shares_with_protected(rand_protected, f"random_K{K}_r{r}")
                for rl in region_labels:
                    random_shares[rl].append(rs[rl])
                random_run_rows.append({
                    "budget_fraction_of_Estar": frac,
                    "K_edges_reinforced": K,
                    "run": r,
                    "coverage_1h": rs["Overall"],
                })

        overall_random = np.asarray(random_shares["Overall"], dtype=float)
        results_rows.append({
            "budget_fraction_of_Estar": frac,
            "K_edges_reinforced": K,
            "greedy_coverage_1h": greedy_cov,
            "random_coverage_1h_mean": float(overall_random.mean()),
            "random_coverage_1h_std": float(overall_random.std(ddof=0)),
            "n_random_runs": len(overall_random),
        })

        for rl in region_labels:
            rand_arr = np.asarray(random_shares[rl], dtype=float)
            regional_rows.append({
                "budget_fraction_of_Estar": frac,
                "K_edges_reinforced": K,
                "region": rl,
                "greedy_coverage_1h": greedy_shares[rl],
                "random_coverage_1h_mean": float(rand_arr.mean()),
                "random_coverage_1h_std": float(rand_arr.std(ddof=0)),
                "n_random_runs": len(rand_arr),
            })

        _log(f"K={K:>6} ({frac:>5.0%} of E*): "
             f"greedy={greedy_cov:.4f}  "
             f"random={overall_random.mean():.4f}+/-{overall_random.std(ddof=0):.4f}  | "
             + "  ".join(f"{rl}={greedy_shares[rl]:.3f}"
                         for rl in region_labels if rl != "Overall"))

    results = pd.DataFrame(results_rows)
    regional = pd.DataFrame(regional_rows)
    random_runs = pd.DataFrame(random_run_rows)

    # Scenario C floor is the greedy coverage at K=0.
    floor_cov = float(results.loc[results["K_edges_reinforced"] == 0,
                                  "greedy_coverage_1h"].iloc[0])
    gap = intact_cov - floor_cov

    def gap_closed(cov):
        return (cov - floor_cov) / gap if gap > 0 else np.nan

    results["intact_ceiling_1h"] = intact_cov
    results["scenarioC_floor_1h"] = floor_cov
    results["greedy_gap_closed"] = results["greedy_coverage_1h"].map(gap_closed)
    results["random_gap_closed"] = results["random_coverage_1h_mean"].map(gap_closed)

    # ----- Per-region floor / ceiling / gap-closed -----
    # Each region has its own Scenario C floor (greedy at K=0) and intact ceiling.
    region_floor = (regional[regional["K_edges_reinforced"] == 0]
                    .set_index("region")["greedy_coverage_1h"])
    regional["intact_ceiling_1h"] = regional["region"].map(intact_shares)
    regional["scenarioC_floor_1h"] = regional["region"].map(region_floor)

    def _region_gap_closed(row):
        g = row["intact_ceiling_1h"] - row["scenarioC_floor_1h"]
        return (row["greedy_coverage_1h"] - row["scenarioC_floor_1h"]) / g if g > 0 else np.nan

    regional["greedy_gap_closed"] = regional.apply(_region_gap_closed, axis=1)

    # ----- Save tables -----
    out_dir = config.OUTPUT_DIR
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "optimisation_results.csv", index=False)
    regional.to_csv(out_dir / "optimisation_regional.csv", index=False)
    random_runs.to_csv(out_dir / "optimisation_random_runs.csv", index=False)

    # ----- Recovery-curve plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        xs = results["budget_fraction_of_Estar"] * 100.0
        ax.plot(xs, results["greedy_coverage_1h"] * 100.0,
                "o-", color="#1f6f3a", lw=2, label="Greedy (centrality-ranked)")
        ax.plot(xs, results["random_coverage_1h_mean"] * 100.0,
                "s--", color="#b03030", lw=2, label="Random baseline (mean)")
        ax.fill_between(
            xs,
            (results["random_coverage_1h_mean"] - results["random_coverage_1h_std"]) * 100.0,
            (results["random_coverage_1h_mean"] + results["random_coverage_1h_std"]) * 100.0,
            color="#b03030", alpha=0.15, label="Random +/- 1 std",
        )
        ax.axhline(intact_cov * 100.0, color="#333", ls=":", lw=1.2,
                   label=f"Intact ceiling ({intact_cov*100:.1f}%)")
        ax.axhline(floor_cov * 100.0, color="#888", ls=":", lw=1.2,
                   label=f"Scenario C floor ({floor_cov*100:.1f}%)")
        ax.set_xlabel("Budget: % of candidate edges E* reinforced")
        ax.set_ylabel(f"1-hour coverage of '{args.tier}' stroke centres (% pop. weight)")
        ax.set_title("Recovery curve: greedy reinforcement vs random baseline (Scenario C)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig6_recovery_curve.png", dpi=150)
        plt.close(fig)
        _log(f"wrote {fig_dir / 'fig6_recovery_curve.png'}")

        # Regional recovery curves (greedy only): one line per region.
        fig, ax = plt.subplots(figsize=(8, 5))
        region_colors = {"North": "#1f6f3a", "Central": "#d08000",
                         "South": "#b03030", "Unknown": "#666666"}
        for rl in region_labels:
            if rl == "Overall":
                continue
            sub = regional[regional["region"] == rl].sort_values("budget_fraction_of_Estar")
            ax.plot(sub["budget_fraction_of_Estar"] * 100.0,
                    sub["greedy_coverage_1h"] * 100.0,
                    "o-", lw=2, color=region_colors.get(rl, None), label=rl)
        ax.set_xlabel("Budget: % of candidate edges E* reinforced")
        ax.set_ylabel(f"1-hour coverage of '{args.tier}' stroke centres (% pop. weight)")
        ax.set_title("Regional recovery under greedy reinforcement (Scenario C)")
        ax.legend(loc="upper left", fontsize=9, title="Region")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig6b_recovery_by_region.png", dpi=150)
        plt.close(fig)
        _log(f"wrote {fig_dir / 'fig6b_recovery_by_region.png'}")
    except Exception as e:  # pragma: no cover - plotting must not break the run
        _log(f"WARNING: could not render recovery-curve plot: {e}")

    # ----- Headline text summary -----
    lines = []
    lines.append("Step 6 optimisation summary (edge reinforcement under Scenario C)")
    lines.append("=" * 64)
    lines.append(f"runtime_seconds        : {time.time()-t_start:.1f}")
    lines.append(f"tier (objective)       : {args.tier}")
    lines.append(f"coverage cutoff        : {COVERAGE_CUTOFF_MIN} min")
    lines.append(f"candidate set E*       : {n_candidate} edges "
                 f"({n_estar} present in graph), C(e) >= {threshold:.0f}")
    lines.append(f"random runs / budget   : {args.n_random_runs}")
    lines.append("")
    lines.append(f"intact-network ceiling : {intact_cov*100:.2f}%")
    lines.append(f"Scenario C floor       : {floor_cov*100:.2f}%")
    lines.append(f"recoverable gap        : {gap*100:.2f} percentage points")
    lines.append("")
    lines.append("Recovery curve (1-hour coverage, % of population weight):")
    disp = results.copy()
    disp["budget_%E*"] = (disp["budget_fraction_of_Estar"] * 100).map(lambda x: f"{x:.0f}%")
    disp["greedy_%"] = (disp["greedy_coverage_1h"] * 100).map(lambda x: f"{x:.2f}")
    disp["random_%"] = (disp["random_coverage_1h_mean"] * 100).map(lambda x: f"{x:.2f}")
    disp["random_std"] = (disp["random_coverage_1h_std"] * 100).map(lambda x: f"{x:.2f}")
    disp["greedy_gap_closed_%"] = (disp["greedy_gap_closed"] * 100).map(
        lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
    lines.append(disp[[
        "budget_%E*", "K_edges_reinforced", "greedy_%", "random_%", "random_std",
        "greedy_gap_closed_%",
    ]].to_string(index=False))
    lines.append("")

    # ----- Regional breakdown (greedy 1-hour coverage, % of population weight) -----
    lines.append("Regional breakdown - greedy 1-hour coverage (% of population weight):")
    reg_pivot = regional.pivot(index="budget_fraction_of_Estar",
                               columns="region", values="greedy_coverage_1h") * 100.0
    col_order = [r for r in region_labels if r in reg_pivot.columns]
    reg_pivot = reg_pivot[col_order]
    reg_pivot.index = (reg_pivot.index * 100).map(lambda x: f"{x:.0f}%")
    reg_pivot.index.name = "budget_%E*"
    lines.append(reg_pivot.to_string(float_format=lambda x: f"{x:.2f}"))
    lines.append("")
    lines.append("Per-region recoverable gap closed by greedy (%, floor->ceiling):")
    gap_pivot = regional.pivot(index="budget_fraction_of_Estar",
                               columns="region", values="greedy_gap_closed") * 100.0
    gap_pivot = gap_pivot[[r for r in region_labels if r in gap_pivot.columns]]
    gap_pivot.index = (gap_pivot.index * 100).map(lambda x: f"{x:.0f}%")
    gap_pivot.index.name = "budget_%E*"
    lines.append(gap_pivot.to_string(
        float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "n/a"))
    lines.append("")
    lines.append("Files written:")
    for f in ["optimisation_results.csv", "optimisation_regional.csv",
              "optimisation_random_runs.csv",
              "figures/fig6_recovery_curve.png",
              "figures/fig6b_recovery_by_region.png"]:
        lines.append(f"  {out_dir / f}")

    summary = "\n".join(lines)
    (out_dir / "optimisation_summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":
    main()
