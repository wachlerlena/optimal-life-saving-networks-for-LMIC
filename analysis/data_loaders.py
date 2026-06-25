"""
Loaders for the accessibility pipeline.

Reuses graph construction / snapping helpers from build_population_map.py so the
analytical pipeline sees the exact same edge weights as the interactive map.

The loaders cache intermediate artifacts to analysis/outputs/cache/ as pickles so
re-running downstream steps doesn't re-pay the cost of GeoJSON parsing or KD-tree
snapping (each of which is the slowest part of the pipeline).
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from analysis import config

# Pull helpers from the existing map-builder script so the graph build and edge
# weights stay in lock-step with the interactive map.
sys.path.insert(0, str(config.ROOT))
from build_population_map import (  # noqa: E402
    build_graph_from_roads_geojson,
    haversine_km,
    load_pickle,
    load_roads_geojson,
    load_stroke_facilities_csv,
)

CACHE_DIR = config.OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_CACHE = CACHE_DIR / "road_graph.pkl"
NODE_INDEX_CACHE = CACHE_DIR / "node_index.pkl"
POP_SNAP_CACHE = CACHE_DIR / "population_snapped.pkl"
HOSP_SNAP_CACHE = CACHE_DIR / "hospitals_snapped.pkl"


def _log(msg: str) -> None:
    print(f"[loaders] {msg}", flush=True)


def load_or_build_graph(force: bool = False) -> tuple[nx.Graph, list[tuple[float, float]], cKDTree]:
    """
    Return (graph, node_list, kdtree). node_list[i] == (lon, lat) for kdtree row i.
    Graph + node_list cached to disk; kdtree rebuilt from node_list on load.
    """
    if not force and GRAPH_CACHE.exists() and NODE_INDEX_CACHE.exists():
        _log(f"loading cached graph from {GRAPH_CACHE.name}")
        t0 = time.time()
        with open(GRAPH_CACHE, "rb") as f:
            G = pickle.load(f)
        with open(NODE_INDEX_CACHE, "rb") as f:
            node_list = pickle.load(f)
        _log(f"  loaded {G.number_of_nodes()} nodes / {G.number_of_edges()} edges in {time.time()-t0:.1f}s")
    else:
        _log(f"parsing road GeoJSON: {config.ROADS_GEOJSON}")
        gj = load_roads_geojson(str(config.ROADS_GEOJSON))
        G = build_graph_from_roads_geojson(gj)
        node_list = list(G.nodes())

        with open(GRAPH_CACHE, "wb") as f:
            pickle.dump(G, f)
        with open(NODE_INDEX_CACHE, "wb") as f:
            pickle.dump(node_list, f)
        _log(f"  cached graph to {GRAPH_CACHE.name}")

    coords = np.asarray(node_list, dtype=float)  # (N, 2) -> (lon, lat)
    tree = cKDTree(coords)
    return G, node_list, tree


def _snap(df: pd.DataFrame, lat_col: str, lon_col: str,
          tree: cKDTree, node_list: list, label: str) -> pd.DataFrame:
    out = df.copy()
    pts = out[[lon_col, lat_col]].to_numpy(dtype=float)
    _, idxs = tree.query(pts, k=1)
    snapped = [node_list[i] for i in idxs]
    snap_lon = np.array([n[0] for n in snapped], dtype=float)
    snap_lat = np.array([n[1] for n in snapped], dtype=float)

    out[f"{label}_node_lon"] = snap_lon
    out[f"{label}_node_lat"] = snap_lat
    # Store as object column of (lon, lat) tuples for direct graph indexing.
    out[f"{label}_node"] = snapped
    out[f"{label}_to_graph_km"] = haversine_km(
        out[lat_col].to_numpy(dtype=float),
        out[lon_col].to_numpy(dtype=float),
        snap_lat, snap_lon,
    )
    _log(
        f"{label} snap: n={len(out)} mean={out[f'{label}_to_graph_km'].mean():.3f}km "
        f"median={out[f'{label}_to_graph_km'].median():.3f}km "
        f"max={out[f'{label}_to_graph_km'].max():.3f}km"
    )
    return out


def load_population_snapped(tree: cKDTree, node_list: list,
                             force: bool = False) -> pd.DataFrame:
    """
    Population dataframe with snapped graph nodes and snap distances.

    Uses the RAW centroid columns `xcoord` (lon) and `ycoord` (lat) for snapping
    because the `lat`/`lon` columns in population.pkl have already been snapped
    by a prior pipeline against a different road graph. Re-snapping the
    already-snapped points underestimates the true offset (the methodology
    baseline is ~3.56 km mean snap; the pre-snapped columns give ~0.5 km).
    """
    if not force and POP_SNAP_CACHE.exists():
        _log(f"loading cached population snap from {POP_SNAP_CACHE.name}")
        with open(POP_SNAP_CACHE, "rb") as f:
            return pickle.load(f)

    _log(f"loading population: {config.POPULATION_PKL}")
    pop = load_pickle(str(config.POPULATION_PKL))
    required = {"ID", "xcoord", "ycoord", "household_count"}
    missing = required - set(pop.columns)
    if missing:
        raise ValueError(f"population pkl missing columns: {missing}")

    pop = pop[["ID", "xcoord", "ycoord", "household_count"]].copy()
    pop = pop.rename(columns={"xcoord": "lon", "ycoord": "lat"})
    pop["household_count"] = pd.to_numeric(pop["household_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    pop["ID"] = pop["ID"].astype(int)
    _log(f"  population rows: {len(pop)} (w_i>0: {(pop['household_count']>0).sum()})")

    snapped = _snap(pop, "lat", "lon", tree, node_list, "population")

    with open(POP_SNAP_CACHE, "wb") as f:
        pickle.dump(snapped, f)
    _log(f"  cached population snap to {POP_SNAP_CACHE.name}")
    return snapped


def load_hospitals_snapped(tree: cKDTree, node_list: list,
                            force: bool = False) -> pd.DataFrame:
    """
    Hospital dataframe with the `specialized_equipment` tier flag merged in
    (True = comprehensive / thrombectomy, False = primary / thrombolysis-only)
    and snapped graph nodes.
    """
    if not force and HOSP_SNAP_CACHE.exists():
        _log(f"loading cached hospital snap from {HOSP_SNAP_CACHE.name}")
        with open(HOSP_SNAP_CACHE, "rb") as f:
            return pickle.load(f)

    _log(f"loading hospitals: {config.HOSPITALS_PKL}")
    h = load_pickle(str(config.HOSPITALS_PKL))
    if isinstance(h.index, pd.Index) and h.index.name == "ID":
        h = h.reset_index(drop=True)
    h = h.loc[:, ~h.columns.duplicated()].copy()
    h["ID"] = pd.to_numeric(h["ID"], errors="coerce").astype("Int64")
    h = h.dropna(subset=["ID", "Latitude", "Longitude"]).copy()
    h["ID"] = h["ID"].astype(int)

    stroke = load_stroke_facilities_csv(str(config.STROKE_FACS_CSV))
    h = h.merge(stroke, on="ID", how="left")
    h["specialized_equipment"] = h["specialized_equipment"].fillna(False).astype(bool)

    n_comp = int(h["specialized_equipment"].sum())
    n_prim = len(h) - n_comp
    _log(f"  tier split: {n_prim} primary / {n_comp} comprehensive (expected 75 / 55)")

    snapped = _snap(h, "Latitude", "Longitude", tree, node_list, "hospital")

    with open(HOSP_SNAP_CACHE, "wb") as f:
        pickle.dump(snapped, f)
    _log(f"  cached hospital snap to {HOSP_SNAP_CACHE.name}")
    return snapped


def assign_region(df: pd.DataFrame, lat_col: str = "lat") -> pd.Series:
    """Vectorized North/Central/South label from a latitude column."""
    lat = df[lat_col].to_numpy(dtype=float)
    out = np.full(len(df), "Unknown", dtype=object)
    out[lat >= 18.0] = "North"
    out[(lat >= 14.0) & (lat < 18.0)] = "Central"
    out[lat < 14.0] = "South"
    return pd.Series(out, index=df.index, name="region")
