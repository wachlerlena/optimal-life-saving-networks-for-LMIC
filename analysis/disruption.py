"""
Step 4 - Deterministic tiered edge disruption.

Given the per-edge centrality table from Step 3, build three scenario graphs by
removing the most critical edges at the 1%, 5%, and 10% level. The ranking is
strictly deterministic (descending C(e)); ties broken by edge ordering in the
edge_centrality DataFrame (which itself is already sorted by centrality).

The disrupted graphs are NOT subgraphs of the giant component - removing
critical edges may fracture the network, and the metric step in Step 5 is
expected to detect that as "unreachable" for affected population nodes.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from analysis import config

DISRUPTION_CACHE = config.OUTPUT_DIR / "cache"
DISRUPTION_CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class DisruptionScenario:
    name: str                      # "baseline", "A_1pct", "B_5pct", "C_10pct"
    fraction_removed: float        # 0.0 for baseline
    graph: nx.Graph
    removed_edge_count: int
    centrality_threshold: float    # min C(e) of removed edges; nan for baseline


def _log(msg: str) -> None:
    print(f"[disruption] {msg}", flush=True)


def _scenario_name(fraction: float) -> str:
    if fraction == 0.0:
        return "baseline"
    pct = int(round(fraction * 100))
    letter = {1: "A", 5: "B", 10: "C"}.get(pct, f"X{pct}")
    return f"{letter}_{pct}pct"


def build_disrupted_graph(
    G: nx.Graph,
    edge_centrality_df: pd.DataFrame,
    fraction: float,
) -> DisruptionScenario:
    """
    Build a disrupted scenario by removing the top `fraction` of edges, sorted
    by C(e) descending. Returns a new graph and metadata.

    edge_centrality_df columns required: u_lon, u_lat, v_lon, v_lat, centrality.
    """
    name = _scenario_name(fraction)

    if fraction <= 0.0:
        return DisruptionScenario(
            name=name,
            fraction_removed=0.0,
            graph=G.copy(),
            removed_edge_count=0,
            centrality_threshold=float("nan"),
        )

    n_total = len(edge_centrality_df)
    n_remove = int(round(fraction * n_total))
    if n_remove <= 0:
        raise ValueError(f"fraction {fraction} yields zero edges to remove from {n_total}")

    # edge_centrality_df is already sorted descending by centrality (see dwebc.py)
    top_edges = edge_centrality_df.iloc[:n_remove]
    threshold = float(top_edges["centrality"].iloc[-1])

    H = G.copy()
    removed = 0
    skipped_missing = 0
    for _, row in top_edges.iterrows():
        u = (row["u_lon"], row["u_lat"])
        v = (row["v_lon"], row["v_lat"])
        if H.has_edge(u, v):
            H.remove_edge(u, v)
            removed += 1
        else:
            skipped_missing += 1

    _log(f"{name}: removed {removed}/{n_remove} edges "
         f"(threshold C(e) >= {threshold:.0f}, skipped {skipped_missing} not-in-graph)")

    return DisruptionScenario(
        name=name,
        fraction_removed=fraction,
        graph=H,
        removed_edge_count=removed,
        centrality_threshold=threshold,
    )


def build_all_scenarios(
    G: nx.Graph,
    edge_centrality_df: pd.DataFrame,
    fractions: tuple[float, ...] = config.DISRUPTION_FRACTIONS,
) -> dict[str, DisruptionScenario]:
    """
    Returns a dict keyed by scenario name. Always includes 'baseline'.
    """
    scenarios: dict[str, DisruptionScenario] = {}
    t0 = time.time()

    baseline = build_disrupted_graph(G, edge_centrality_df, 0.0)
    scenarios[baseline.name] = baseline
    _log(f"baseline: {baseline.graph.number_of_edges()} edges")

    for f in fractions:
        s = build_disrupted_graph(G, edge_centrality_df, f)
        scenarios[s.name] = s
        # Quick connectivity sanity check.
        n_components = nx.number_connected_components(s.graph)
        _log(f"  {s.name}: {s.graph.number_of_edges()} edges, "
             f"{n_components} connected components")

    _log(f"all scenarios built in {time.time()-t0:.1f}s")
    return scenarios
