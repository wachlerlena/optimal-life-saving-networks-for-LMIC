"""
Step 5 - Accessibility metrics and disparity calculation.

For a given (scenario graph, hospital source set) pair, compute the travel time
from every population graph node to its nearest reachable hospital using a
single multi-source Dijkstra. Then derive:

  * Travel-time summary    : mean / median / p95 / unreachable-share, both
                             overall and stratified by region.
  * Isochrone catchments   : share of TOTAL POPULATION WEIGHT reachable within
                             60 / 120 / 180 minutes.
  * Per-row travel times   : returned as a Series for delta calculations
                             vs the baseline scenario.

Tier definitions (per methodology):
  * "any"           -> all 130 stroke centers (primary OR comprehensive)
  * "comprehensive" -> the 55 comprehensive (thrombectomy-capable) centers
                       i.e. specialized_equipment == True

Important: this step uses TRAVEL TIME on the graph, NOT total-distance. The
300 km cap from Step 2 was a clinical-relevance filter for the DWEBC routing
step; Step 5 explicitly says "if a node loses all connectivity to a tier, flag
it as unreachable" -- which is a graph-connectivity criterion, not a distance
cap. Population nodes whose travel time is undefined (Dijkstra didn't reach
them on the disrupted graph) are the unreachable set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from analysis import config


@dataclass
class TierTravelTimes:
    scenario_name: str
    tier: str                  # "any" or "comprehensive"
    travel_time_min: pd.Series # indexed by pop_id; NaN = unreachable
    n_reachable: int
    n_unreachable: int


def _log(msg: str) -> None:
    print(f"[metrics] {msg}", flush=True)


def hospital_nodes_for_tier(hosp_df: pd.DataFrame, tier: str) -> list:
    """Return the unique graph-node coordinate tuples for the requested tier."""
    if tier == "any":
        sub = hosp_df
    elif tier == "comprehensive":
        sub = hosp_df[hosp_df["specialized_equipment"] == True]  # noqa: E712
    else:
        raise ValueError(f"unknown tier: {tier!r}")
    # Deduplicate while preserving order
    seen = set()
    nodes = []
    for n in sub["hospital_node"]:
        if n not in seen:
            seen.add(n)
            nodes.append(n)
    return nodes


def compute_tier_travel_times(
    G: nx.Graph,
    pop_df: pd.DataFrame,
    hosp_df: pd.DataFrame,
    tier: str,
    scenario_name: str,
) -> TierTravelTimes:
    """
    Run a single multi-source Dijkstra rooted at the tier's hospital nodes on
    graph G. Return per-population travel time (NaN if unreachable).
    """
    sources = hospital_nodes_for_tier(hosp_df, tier)
    # Only sources that actually exist in this (possibly disrupted) graph.
    sources_in_g = [s for s in sources if s in G]
    if not sources_in_g:
        raise ValueError(f"no tier='{tier}' hospital nodes present in scenario '{scenario_name}'")

    t0 = time.time()
    # path_length only -> faster than predecessor_and_distance when we don't need paths.
    dist = nx.multi_source_dijkstra_path_length(
        G, sources=sources_in_g, weight="travel_time_min"
    )
    _log(f"  [{scenario_name}/{tier}] Dijkstra: {len(sources_in_g)} sources, "
         f"{len(dist)} reached, {time.time()-t0:.1f}s")

    pop_nodes = pop_df["population_node"].to_numpy()
    pop_ids = pop_df["ID"].to_numpy()
    n = len(pop_df)

    tt = np.full(n, np.nan, dtype=float)
    for i in range(n):
        d = dist.get(pop_nodes[i])
        if d is not None:
            tt[i] = d

    series = pd.Series(tt, index=pd.Index(pop_ids, name="pop_id"),
                       name=f"tt_{scenario_name}_{tier}")
    n_reachable = int(np.isfinite(tt).sum())
    n_unreachable = n - n_reachable
    return TierTravelTimes(
        scenario_name=scenario_name,
        tier=tier,
        travel_time_min=series,
        n_reachable=n_reachable,
        n_unreachable=n_unreachable,
    )


def isochrone_catchment_share(
    travel_times: pd.Series,
    weights: pd.Series,
    cutoffs_min: tuple[int, ...] = config.ISOCHRONE_WINDOWS_MIN,
) -> dict[int, float]:
    """
    Share of TOTAL WEIGHT (e.g. household count) whose travel time <= each
    cutoff. Returns {cutoff_min: share_in_[0..1]}.
    """
    aligned = pd.DataFrame({"tt": travel_times, "w": weights}).dropna(subset=["w"])
    total_w = float(aligned["w"].sum())
    if total_w <= 0:
        return {c: 0.0 for c in cutoffs_min}
    out = {}
    for c in cutoffs_min:
        reachable_w = float(aligned.loc[aligned["tt"].notna() & (aligned["tt"] <= c), "w"].sum())
        out[c] = reachable_w / total_w
    return out


def travel_time_summary(
    travel_times: pd.Series,
    weights: pd.Series,
    regions: pd.Series,
) -> pd.DataFrame:
    """
    Per-region (and overall) summary: weighted mean travel time, median p50,
    p95, unreachable-share. All by weighted population.
    """
    df = pd.DataFrame({
        "tt": travel_times,
        "w": weights,
        "region": regions,
    }).copy()

    def _agg(sub: pd.DataFrame) -> pd.Series:
        w = sub["w"].to_numpy(dtype=float)
        tt = sub["tt"].to_numpy(dtype=float)
        total_w = w.sum()
        reach_mask = np.isfinite(tt)
        reach_w = w[reach_mask].sum()
        unreach_share = (total_w - reach_w) / total_w if total_w > 0 else np.nan

        if reach_w <= 0:
            return pd.Series({
                "total_weight": total_w,
                "reachable_weight": reach_w,
                "unreachable_share": unreach_share,
                "tt_weighted_mean": np.nan,
                "tt_weighted_p50": np.nan,
                "tt_weighted_p95": np.nan,
            })

        tt_r = tt[reach_mask]
        w_r = w[reach_mask]
        mean_tt = float(np.average(tt_r, weights=w_r))

        order = np.argsort(tt_r)
        tt_sorted = tt_r[order]
        w_sorted = w_r[order]
        cum_w = np.cumsum(w_sorted)
        cum_share = cum_w / cum_w[-1]

        def _wq(q: float) -> float:
            idx = int(np.searchsorted(cum_share, q, side="left"))
            idx = min(idx, len(tt_sorted) - 1)
            return float(tt_sorted[idx])

        return pd.Series({
            "total_weight": total_w,
            "reachable_weight": reach_w,
            "unreachable_share": unreach_share,
            "tt_weighted_mean": mean_tt,
            "tt_weighted_p50": _wq(0.50),
            "tt_weighted_p95": _wq(0.95),
        })

    rows = []
    for region, sub in df.groupby("region"):
        s = _agg(sub)
        s["region"] = region
        rows.append(s)
    overall = _agg(df)
    overall["region"] = "Overall"
    rows.append(overall)

    out = pd.DataFrame(rows).set_index("region")
    return out
