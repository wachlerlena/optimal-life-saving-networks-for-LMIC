"""
Thesis figures from Steps 3-5 outputs.

Reads CSV/pickle artifacts already produced by run_dwebc.py and run_metrics.py
and produces a focused set of PNG figures saved to analysis/outputs/figures/.

Figures produced:
  fig1_travel_time_by_scenario.png
        Weighted mean travel time per scenario, faceted by tier (any /
        comprehensive), grouped by region.
  fig2_catchment_shrinkage.png
        Reachable population share at 60 / 120 / 180 min cutoffs, per
        scenario, faceted by tier. Overall (national) figures.
  fig3_regional_delta_mean.png
        Per-region weighted-mean travel-time DELTA vs baseline, per scenario,
        faceted by tier. Quantifies regional disparity.
  fig4_unreachable_share.png
        Population share that loses all connectivity to a tier under each
        scenario. Faceted by tier; lines by region.
  fig5_centrality_lorenz.png
        Lorenz-style curve of demand-weighted edge centrality. x = cumulative
        fraction of edges (sorted descending by C(e)), y = cumulative share of
        total C(e). Marks 1% / 5% / 10% thresholds.

Usage:
    python -m analysis.make_plots
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import config

SCENARIO_ORDER = ["baseline", "A_1pct", "B_5pct", "C_10pct"]
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "A_1pct":   "A (top 1%)",
    "B_5pct":   "B (top 5%)",
    "C_10pct":  "C (top 10%)",
}
TIER_LABELS = {"any": "Any stroke centre", "comprehensive": "Comprehensive only"}
REGION_ORDER = ["North", "Central", "South", "Overall"]
REGION_COLORS = {
    "North":   "#1f77b4",
    "Central": "#ff7f0e",
    "South":   "#2ca02c",
    "Overall": "#444444",
}
SCENARIO_COLORS = {
    "baseline": "#4a90d9",
    "A_1pct":   "#f6c244",
    "B_5pct":   "#e87832",
    "C_10pct":  "#c5283d",
}


def _log(msg: str) -> None:
    print(f"[plots] {msg}", flush=True)


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi":      110,
        "savefig.dpi":     180,
        "savefig.bbox":    "tight",
        "font.family":     "DejaVu Sans",
        "axes.titlesize":  12,
        "axes.labelsize":  10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Figure 1: weighted mean travel time by scenario, region, tier
# ---------------------------------------------------------------------------
def fig_travel_time_by_scenario(tt_summary: pd.DataFrame, out_path: Path) -> None:
    regions = [r for r in REGION_ORDER if r != "Overall"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    for ax, tier in zip(axes, ("any", "comprehensive")):
        sub = tt_summary[(tt_summary["tier"] == tier) & (tt_summary["region"].isin(regions))]
        pivot = (
            sub.pivot(index="scenario", columns="region", values="tt_weighted_mean")
               .reindex(SCENARIO_ORDER)[regions]
        )

        x = np.arange(len(pivot.index))
        width = 0.26
        for i, region in enumerate(regions):
            ax.bar(
                x + (i - 1) * width,
                pivot[region].to_numpy(),
                width,
                color=REGION_COLORS[region],
                label=region,
                edgecolor="white",
                linewidth=0.6,
            )

        overall = tt_summary[(tt_summary["tier"] == tier) & (tt_summary["region"] == "Overall")]
        overall = overall.set_index("scenario").reindex(SCENARIO_ORDER)["tt_weighted_mean"]
        for xi, v in zip(x, overall.to_numpy()):
            ax.plot([xi - 1.5 * width, xi + 1.5 * width], [v, v],
                    color="black", lw=1.3, ls="--", alpha=0.65)

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in pivot.index])
        ax.set_title(TIER_LABELS[tier])
        ax.set_ylabel("Weighted mean travel time (min)") if tier == "any" else None
        ax.grid(axis="y", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS[r]) for r in regions]
    handles.append(plt.Line2D([], [], color="black", lw=1.3, ls="--", label="Overall"))
    labels = regions + ["Overall"]
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out_path)
    plt.close(fig)
    _log(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2: catchment shrinkage at 60/120/180 min
# ---------------------------------------------------------------------------
def fig_catchment_shrinkage(catchment: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    cutoffs = sorted(catchment["cutoff_min"].unique())
    n_cut = len(cutoffs)

    for ax, tier in zip(axes, ("any", "comprehensive")):
        sub = catchment[(catchment["tier"] == tier) & (catchment["region"] == "Overall")]
        pivot = (
            sub.pivot(index="scenario", columns="cutoff_min", values="reachable_weight_share")
               .reindex(SCENARIO_ORDER)[cutoffs]
        )

        x = np.arange(len(pivot.index))
        width = 0.78 / n_cut
        cmap_colors = ["#74add1", "#fdae61", "#d73027"]
        for i, c in enumerate(cutoffs):
            ax.bar(
                x + (i - (n_cut - 1) / 2) * width,
                pivot[c].to_numpy() * 100,
                width,
                color=cmap_colors[i % len(cmap_colors)],
                label=f"≤ {c} min",
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in pivot.index])
        ax.set_title(TIER_LABELS[tier])
        ax.set_ylabel("Population share reachable (%)") if tier == "any" else None
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path)
    plt.close(fig)
    _log(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3: regional delta vs baseline
# ---------------------------------------------------------------------------
def fig_regional_delta(tt_deltas: pd.DataFrame, out_path: Path) -> None:
    regions = [r for r in REGION_ORDER if r != "Overall"]
    non_baseline = [s for s in SCENARIO_ORDER if s != "baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, tier in zip(axes, ("any", "comprehensive")):
        sub = tt_deltas[
            (tt_deltas["tier"] == tier)
            & (tt_deltas["region"].isin(regions))
            & (tt_deltas["scenario"].isin(non_baseline))
        ]
        pivot = (
            sub.pivot(index="scenario", columns="region", values="delta_mean_min")
               .reindex(non_baseline)[regions]
        )

        x = np.arange(len(pivot.index))
        width = 0.26
        for i, region in enumerate(regions):
            vals = pivot[region].to_numpy()
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                color=REGION_COLORS[region],
                label=region,
                edgecolor="white",
                linewidth=0.6,
            )
            for xi, v in zip(x + (i - 1) * width, vals):
                ax.text(xi, v + 1.2, f"+{v:.0f}", ha="center", va="bottom",
                        fontsize=8, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in pivot.index])
        ax.set_title(TIER_LABELS[tier])
        ax.set_ylabel("Δ mean travel time vs baseline (min)") if tier == "any" else None
        ax.axhline(0, color="#888", lw=0.8)
        ax.grid(axis="y", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=REGION_COLORS[r]) for r in regions]
    fig.legend(handles, regions, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out_path)
    plt.close(fig)
    _log(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 4: unreachable share growth
# ---------------------------------------------------------------------------
def fig_unreachable_share(tt_summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tier in zip(axes, ("any", "comprehensive")):
        sub = tt_summary[tt_summary["tier"] == tier]
        for region in REGION_ORDER:
            row = sub[sub["region"] == region].set_index("scenario").reindex(SCENARIO_ORDER)
            ax.plot(
                [SCENARIO_LABELS[s] for s in SCENARIO_ORDER],
                row["unreachable_share"].to_numpy() * 100,
                marker="o",
                lw=2 if region == "Overall" else 1.4,
                ls="--" if region == "Overall" else "-",
                color=REGION_COLORS[region],
                label=region,
            )
        ax.set_title(TIER_LABELS[tier])
        ax.set_ylabel("Population share unreachable (%)") if tier == "any" else None
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path)
    plt.close(fig)
    _log(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 5: centrality Lorenz curve
# ---------------------------------------------------------------------------
def fig_centrality_lorenz(edge_centrality: pd.DataFrame, out_path: Path) -> None:
    df = edge_centrality.sort_values("centrality", ascending=False).reset_index(drop=True)
    total = float(df["centrality"].sum())
    n = len(df)
    if total <= 0 or n == 0:
        _log("WARNING: edge_centrality has no positive mass; skipping Lorenz figure")
        return

    cum_share = df["centrality"].cumsum().to_numpy() / total
    edge_frac = (np.arange(1, n + 1)) / n

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(edge_frac * 100, cum_share * 100, color="#c5283d", lw=2, label="Cumulative C(e)")
    ax.plot([0, 100], [0, 100], color="#888", lw=0.8, ls=":", label="Uniform distribution")

    for frac, color, label in [
        (0.01, SCENARIO_COLORS["A_1pct"], "Top 1% (A)"),
        (0.05, SCENARIO_COLORS["B_5pct"], "Top 5% (B)"),
        (0.10, SCENARIO_COLORS["C_10pct"], "Top 10% (C)"),
    ]:
        idx = max(0, int(round(frac * n)) - 1)
        share = float(cum_share[idx]) * 100
        ax.axvline(frac * 100, color=color, lw=1, ls="--", alpha=0.7)
        ax.scatter([frac * 100], [share], color=color, s=42, zorder=5)
        ax.annotate(
            f"{label}\n{share:.1f}% of total C(e)",
            xy=(frac * 100, share),
            xytext=(frac * 100 + 4, share - 8),
            fontsize=8.5, color=color,
        )

    ax.set_xlabel("Cumulative share of edges (sorted by C(e) desc) [%]")
    ax.set_ylabel("Cumulative share of total demand-weighted centrality [%]")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    _log(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis figures from Step 3-5 outputs.")
    p.add_argument("--tt-summary",       default=str(config.OUTPUT_DIR / "tt_summary.csv"))
    p.add_argument("--catchment",        default=str(config.OUTPUT_DIR / "catchment.csv"))
    p.add_argument("--tt-deltas",        default=str(config.OUTPUT_DIR / "tt_deltas.csv"))
    p.add_argument("--edge-centrality",  default=str(config.OUTPUT_DIR / "edge_centrality.pkl"))
    p.add_argument("--out-dir",          default=str(config.OUTPUT_DIR / "figures"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    t0 = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading {args.tt_summary}")
    tt_summary = pd.read_csv(args.tt_summary)
    _log(f"loading {args.catchment}")
    catchment = pd.read_csv(args.catchment)
    _log(f"loading {args.tt_deltas}")
    tt_deltas = pd.read_csv(args.tt_deltas)
    _log(f"loading {args.edge_centrality}")
    edge_centrality = pd.read_pickle(args.edge_centrality)

    fig_travel_time_by_scenario(tt_summary, out_dir / "fig1_travel_time_by_scenario.png")
    fig_catchment_shrinkage(catchment,      out_dir / "fig2_catchment_shrinkage.png")
    fig_regional_delta(tt_deltas,           out_dir / "fig3_regional_delta_mean.png")
    fig_unreachable_share(tt_summary,       out_dir / "fig4_unreachable_share.png")
    fig_centrality_lorenz(edge_centrality,  out_dir / "fig5_centrality_lorenz.png")

    _log(f"done in {time.time()-t0:.1f}s; figures in {out_dir}")


if __name__ == "__main__":
    main()
