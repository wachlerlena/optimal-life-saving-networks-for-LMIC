# =============================================================================
# BUILD DISTANCE MATRIX
# =============================================================================
# Purpose of this file:
# This script builds the precomputed population-to-hospital distance matrix used
# by the optimization models. It converts the Vietnam road GeoJSON into a road
# graph, snaps population points and hospitals to that graph, and stores travel
# distances/times up to MAX_DIST_KM.
# Average run time is about 50 minutes on a laptop with 16GB RAM and 8 CPU cores.
#
# How to run from the command line:
#   python build_distance_matrix.py
#
# Main output:
# Data_vietnam/existing_hospitals_100/distances_osm_max_300km.pkl
#
# Important limitation:
# Travel times depend on the road-speed assumptions in infer_speed_from_properties
# and on snapping population/hospital coordinates to the nearest road node.
# =============================================================================

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

# PATHS AND CONSTANTS
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_vietnam"
ROADS_GEOJSON_PATH = DATA_DIR / "road_osm_preprocessed.geojson"
OUTPUT_PATH = DATA_DIR / "existing_hospitals_100" / "distances_osm_max_300km.pkl"
MAX_DIST_KM = 300.0
CUTOFF_MINUTES = 270.0
SNAP_SPEED_KMH = 25.0


# Great-circle distance in kilometers. Used for snapping and filtering.
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0088 * c

# Stable graph node key from a lon/lat coordinate pair.
def node_key(coord, precision=6):
    lon, lat = coord
    return (round(float(lon), precision), round(float(lat), precision))

# Estimate road speed from OSM-style RTT/RST attributes.
def infer_speed_from_properties(props):
    rtt = props.get("rtt", None)
    rst = props.get("rst", None)

    speed_by_rtt = {
        16: 70.0,
        14: 55.0,
        15: 40.0,
        999: 25.0,
    }

    default_speed_kmh = 40.0
    speed = speed_by_rtt.get(rtt, default_speed_kmh)

    if rst == 2:
        speed = speed * 0.70

    return max(speed, 5.0)

# -----------------------------------------------------------------------------
# BUILD ROAD GRAPH
# -----------------------------------------------------------------------------
# Convert each road LineString into graph edges with length and travel time.
def build_graph_from_roads_geojson(geojson_obj, precision=6):
    G = nx.Graph()

    features = geojson_obj.get("features", [])

    total_lines = 0
    skipped_non_lines = 0
    skipped_short = 0
    zero_length_segments = 0
    added_edges = 0
    updated_edges = 0

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}

        if geom.get("type") != "LineString":
            skipped_non_lines += 1
            continue

        coords = geom.get("coordinates", [])
        if coords is None or len(coords) < 2:
            skipped_short += 1
            continue

        total_lines += 1
        speed_kmh = infer_speed_from_properties(props)

        for i in range(len(coords) - 1):
            c1 = coords[i]
            c2 = coords[i + 1]

            n1 = node_key(c1, precision=precision)
            n2 = node_key(c2, precision=precision)

            lon1, lat1 = n1
            lon2, lat2 = n2

            seg_len_km = float(haversine_km(lat1, lon1, lat2, lon2))
            if seg_len_km == 0.0:
                zero_length_segments += 1
                continue

            travel_time_min = (seg_len_km / speed_kmh) * 60.0

            if n1 not in G:
                G.add_node(n1, x=lon1, y=lat1)
            if n2 not in G:
                G.add_node(n2, x=lon2, y=lat2)

            if G.has_edge(n1, n2):
                if seg_len_km < G[n1][n2]["length_km"]:
                    G[n1][n2].update({
                        "length_km": seg_len_km,
                        "travel_time_min": travel_time_min,
                        "speed_kmh": speed_kmh,
                        "rtt": props.get("rtt"),
                        "rst": props.get("rst"),
                    })
                    updated_edges += 1
            else:
                G.add_edge(
                    n1,
                    n2,
                    length_km=seg_len_km,
                    travel_time_min=travel_time_min,
                    speed_kmh=speed_kmh,
                    rtt=props.get("rtt"),
                    rst=props.get("rst"),
                )
                added_edges += 1

    print("Road graph build summary:")
    print(f"  LineString features processed: {total_lines}")
    print(f"  Skipped non-LineString features: {skipped_non_lines}")
    print(f"  Skipped short/invalid lines: {skipped_short}")
    print(f"  Zero-length segments skipped: {zero_length_segments}")
    print(f"  Graph nodes: {G.number_of_nodes()}")
    print(f"  Graph edges: {G.number_of_edges()}")
    print(f"  New edges added: {added_edges}")
    print(f"  Existing edges updated: {updated_edges}")

    components = list(nx.connected_components(G))
    component_sizes = sorted((len(c) for c in components), reverse=True)
    print(f"  Connected components: {len(component_sizes)}")
    print(f"  Largest components: {component_sizes[:5]}")

    if component_sizes:
        giant = max(components, key=len)
        G = G.subgraph(giant).copy()
        print("Keeping only giant component for routing.")
        print(f"  Giant component nodes: {G.number_of_nodes()}")
        print(f"  Giant component edges: {G.number_of_edges()}")

    return G

# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------
# 1. Load road GeoJSON and build graph.
# 2. Snap population points and hospitals to nearest graph nodes.
# 3. Run shortest-path searches from hospitals.
# 4. Save all population-hospital pairs within MAX_DIST_KM.
def main() -> None:
    with open(ROADS_GEOJSON_PATH, "r", encoding="utf-8") as handle:
        geojson_obj = json.load(handle)

    # Build the routing graph and nearest-node index.
    G = build_graph_from_roads_geojson(geojson_obj)
    node_list = list(G.nodes())
    node_coords = np.array([[n[0], n[1]] for n in node_list], dtype=float)
    tree = cKDTree(node_coords)

    # Load and snap population
    population = pd.read_pickle(DATA_DIR / "population.pkl")
    population = population.rename(columns={"ID": "pop_id"}).copy()
    population = population[["pop_id", "lat", "lon"]].dropna(subset=["lat", "lon"]).reset_index(drop=True)
    population["pop_id"] = population["pop_id"].astype("int64")

    pop_query = population[["lon", "lat"]].to_numpy(dtype=float)
    _, pop_idxs = tree.query(pop_query, k=1)
    pop_nodes = [node_list[i] for i in pop_idxs]
    snapped_lat = np.array([n[1] for n in pop_nodes], dtype=float)
    snapped_lon = np.array([n[0] for n in pop_nodes], dtype=float)

    pop_snap_km = haversine_km(population["lat"].to_numpy(dtype=float), population["lon"].to_numpy(dtype=float), snapped_lat, snapped_lon)
    pop_snap_min = pop_snap_km / SNAP_SPEED_KMH * 60.0

    pop_node_index: dict[tuple, list] = defaultdict(list)
    for pop_id, node, snap_km, snap_min in zip(population["pop_id"].to_numpy(dtype=int), pop_nodes, pop_snap_km, pop_snap_min):
        pop_node_index[node].append((int(pop_id), float(snap_km), float(snap_min)))

    # Load and snap facilities
    facilities = pd.read_pickle(DATA_DIR / "existing_hospitals_100" / "all_hospitals.pkl")
    facilities = facilities.rename(columns={"ID": "facility_id"}).copy()
    facilities = facilities[["facility_id", "Latitude", "Longitude"]].rename(columns={"Latitude": "lat", "Longitude": "lon"}).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    facilities["facility_id"] = facilities["facility_id"].astype("int64")

    fac_query = facilities[["lon", "lat"]].to_numpy(dtype=float)
    _, fac_idxs = tree.query(fac_query, k=1)
    fac_nodes = [node_list[i] for i in fac_idxs]
    fac_snapped_lat = np.array([n[1] for n in fac_nodes], dtype=float)
    fac_snapped_lon = np.array([n[0] for n in fac_nodes], dtype=float)

    fac_snap_km = haversine_km(facilities["lat"].to_numpy(dtype=float), facilities["lon"].to_numpy(dtype=float), fac_snapped_lat, fac_snapped_lon)
    fac_snap_min = fac_snap_km / SNAP_SPEED_KMH * 60.0

    records: list[tuple] = []
    processed = 0
    for facility_row, hosp_node, hosp_snap_km, hosp_snap_min in zip(facilities.itertuples(index=False), fac_nodes, fac_snap_km, fac_snap_min):
        processed += 1
        facility_id = int(facility_row.facility_id)
        try:
            dist_dict, path_dict = nx.single_source_dijkstra(G, source=hosp_node, weight="travel_time_min", cutoff=CUTOFF_MINUTES)
        except Exception:
            dist_dict = {}
            path_dict = {}

        for reached_node, network_minutes in dist_dict.items():
            if reached_node not in pop_node_index:
                continue
            path = path_dict.get(reached_node, None)
            for pop_id, pop_km, pop_min in pop_node_index[reached_node]:
                travel_minutes_network = float(pop_min + float(network_minutes) + float(hosp_snap_min))
                if path is None:
                    continue
                road_distance = sum(G[u][v]["length_km"] for u, v in zip(path[:-1], path[1:]))
                total_dist = float(pop_km) + float(road_distance) + float(hosp_snap_km)
                if total_dist > MAX_DIST_KM:
                    continue
                records.append((int(pop_id), facility_id, float(pop_km), float(road_distance), float(hosp_snap_km), float(total_dist), float(travel_minutes_network)))

        if processed % 10 == 0:
            print(f"Processed {processed}/{len(facilities)} hospitals, {len(records)} pairs so far")

    distances = pd.DataFrame(records, columns=[
        "pop_id",
        "hosp_id",
        "pop_to_road_dist",
        "road_distance",
        "hosp_to_road_dist",
        "total_dist",
        "travel_minutes_network",
    ])

    if len(distances) > 0:
        distances["pop_id"] = distances["pop_id"].astype("int64")
        distances["hosp_id"] = distances["hosp_id"].astype("int64")
        for col in ["pop_to_road_dist", "road_distance", "hosp_to_road_dist", "total_dist", "travel_minutes_network"]:
            distances[col] = distances[col].astype("float64")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    distances.to_pickle(OUTPUT_PATH)

    print(f"Final summary: total pairs={len(distances)}")
    if len(distances) > 0:
        print("  total_dist min/mean/median/max = " +
              f"{distances['total_dist'].min():.4f} / {distances['total_dist'].mean():.4f} / {distances['total_dist'].median():.4f} / {distances['total_dist'].max():.4f}")
        print("  travel_minutes_network min/mean/median/max = " +
              f"{distances['travel_minutes_network'].min():.4f} / {distances['travel_minutes_network'].mean():.4f} / {distances['travel_minutes_network'].median():.4f} / {distances['travel_minutes_network'].max():.4f}")
        print(f"  unique pop_ids: {distances['pop_id'].nunique()}")
        print(f"  unique hosp_ids: {distances['hosp_id'].nunique()}")


if __name__ == "__main__":
    main()
