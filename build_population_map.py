# =============================================================================
# BUILD POPULATION MAP - MAIN PYTHON FILE
# =============================================================================
# Purpose of this file:
# This is the main script that reads the data, builds the map, calculates routes,
# compares travel-time models, and saves the final interactive HTML map.
#
# The code is split into clear parts:
# 1. Load and clean population, hospital, road, and stroke-equipment data.
# 2. Build visual map layers, such as the heatmap and hospital icons.
# 3. Convert the road GeoJSON into a NetworkX graph for routing.
# 4. Snap population points and hospitals to the nearest road node using KDTree.
# 5. Calculate route alternatives and compare three travel-time models.
# 6. Send prepared map data to map_ui.py, which handles the side panel and clicks.
#
# Main modelling idea:
# - Distance model: straight-line distance, no road network.
# - Constant speed model: road-network distance, one speed for all roads.
# - Road-adjusted model: road-network route with speed based on RTT and RST.
#
# Important limitation:
# The clickable population markers use a sample controlled by max_marker_points.
# The heatmap uses the aggregated population data, but the full set of 400k+
# points is not all shown as individual clickable markers because that would make
# the browser very slow.
# =============================================================================

import argparse
import json
import os
import pickle
from collections import Counter

import folium
import networkx as nx
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

# UI helper: contains the injected HTML/CSS/JavaScript side panel.
from map_ui import inject_side_panel_and_routes

# We use KDTree to quickly find the nearest road-network node
# for each population point and each hospital point.
# This makes snapping coordinates to the road graph much faster.
# Max distance 300km
try:
    from scipy.spatial import cKDTree
except ImportError as e:
    raise ImportError(
        "This script requires scipy for fast nearest-node lookup. "
        "Install it with: pip install scipy"
    ) from e

# Load saved data from a pickle file.
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Convert a pandas Series to numeric values. Invalid values become NaN.
def ensure_numeric(series):
    return pd.to_numeric(series, errors="coerce")

# -----------------------------------------------------------------------------
# Convert the Vietnamese stroke-facility label into a Python True/False value.
# In the CSV, "có" means that the hospital has intervention/specialized
# stroke equipment. Everything else is treated as False.
# -----------------------------------------------------------------------------
def normalize_specialized_equipment(value):
    if pd.isna(value):
        return False

    text = str(value).strip().lower().replace("\n", " ")
    return text == "có"

# -----------------------------------------------------------------------------
# Load the stroke-facility CSV and match it to hospital IDs.
# The CSV uses TT, where TT = hospital ID + 1, so we subtract 1 to align it
# with the hospital pickle file. The result is a small table with:
# hospital ID + whether it has specialized stroke equipment.
# -----------------------------------------------------------------------------
def load_stroke_facilities_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stroke facilities CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"TT", "BV có can\nthiệp"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Stroke facilities CSV is missing required columns: {sorted(missing_cols)}"
        )

    df = df.copy()
    df["TT"] = ensure_numeric(df["TT"])
    df = df.dropna(subset=["TT"]).copy()
    df["TT"] = df["TT"].astype(int)

    # CSV TT = hospital ID + 1
    df["ID"] = df["TT"] - 1

    df["specialized_equipment"] = df["BV có can\nthiệp"].apply(
        normalize_specialized_equipment
    )

    # Extract English name and address from the CSV.
    if "Name_English" in df.columns:
        df["Name_English"] = df["Name_English"].fillna("").astype(str).str.strip()
    else:
        df["Name_English"] = ""

    if "Địa chỉ" in df.columns:
        df["Address_CSV"] = df["Địa chỉ"].fillna("").astype(str).str.strip()
    else:
        df["Address_CSV"] = ""

    print("Stroke facilities CSV loaded.")
    print(f"  Rows: {len(df)}")
    print("  Specialized equipment counts:")
    print(df["specialized_equipment"].value_counts(dropna=False))

    return df[["ID", "specialized_equipment", "Name_English", "Address_CSV"]]

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle (haversine) distance in kilometers.
    Used for the distance model and for checking snap distances.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0088 * c

def node_key(coord, precision=6):
    """
    Stable node key from lon/lat coordinate pair.
    Rounding prevents tiny decimal differences from creating duplicate nodes.
    """
    lon, lat = coord
    return (round(float(lon), precision), round(float(lat), precision))

def infer_speed_from_properties(props):
    """
    Estimate driving speed (km/h) for one road segment using RTT and RST.

    RTT = route intended use / road class.
    RST = road surface type.

    Speed assumptions:
    - RTT 16: motorway / expressway  -> 70 km/h
    - RTT 14: primary road           -> 55 km/h
    - RTT 15: secondary road          -> 40 km/h
    - RTT 999: local road             -> 25 km/h
    - RST 2: unpaved, 30% slower
    """
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
# Safely convert a value to an integer.
# Folium and JSON output need clean IDs; this helper avoids crashes when an ID
# is missing or cannot be converted.
# -----------------------------------------------------------------------------
def safe_int(value):
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Add a latitude/longitude point to a route if it is valid.
# This also avoids adding the same coordinate twice in a row, which keeps route
# lines cleaner when actual points and snapped road nodes overlap.
# -----------------------------------------------------------------------------
def add_coord_if_valid(coord_list, lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return
    pt = [float(lat), float(lon)]
    if not coord_list or coord_list[-1] != pt:
        coord_list.append(pt)
        
def build_hospital_icon_html(has_specialized, size_px=24):
    """
    Hospital marker as HTML: green circle + white cross if specialized,
    red circle + white cross otherwise.
    """
    bg_color = "#19a74a" if has_specialized else "#e53935"

    cross_thickness = max(4, int(size_px * 0.22))
    cross_length = max(10, int(size_px * 0.54))
    border_px = max(2, int(size_px * 0.08))

    html = f"""
    <div style="
        position: relative;
        width: {size_px}px;
        height: {size_px}px;
        border-radius: 50%;
        background: {bg_color};
        border: {border_px}px solid black;
        box-sizing: border-box;
    ">
        <div style="
            position: absolute;
            left: 50%;
            top: 50%;
            width: {cross_length}px;
            height: {cross_thickness}px;
            background: white;
            transform: translate(-50%, -50%);
            border-radius: 1px;
        "></div>
        <div style="
            position: absolute;
            left: 50%;
            top: 50%;
            width: {cross_thickness}px;
            height: {cross_length}px;
            background: white;
            transform: translate(-50%, -50%);
            border-radius: 1px;
        "></div>
    </div>
    """
    return html       

def aggregate_population_for_visualization(
    population_df,
    lat_col="lat",
    lon_col="lon",
    weight_col="household_count",
):
    """
    Aggregate population rows by exact coordinate so the heatmap and
    circle layer reflect total household count per visible map position.
    """
    required_cols = {lat_col, lon_col, weight_col}
    missing_cols = required_cols - set(population_df.columns)
    if missing_cols:
        raise ValueError(
            f"Population dataframe is missing required columns for aggregation: {sorted(missing_cols)}"
        )

    df = population_df[[lat_col, lon_col, weight_col]].copy()
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    df[weight_col] = ensure_numeric(df[weight_col]).fillna(0.0)
    df[weight_col] = df[weight_col].clip(lower=0.0)

    agg_df = (
        df.groupby([lat_col, lon_col], as_index=False)[weight_col]
          .sum()
          .rename(columns={weight_col: "household_count_sum"})
    )

    print("Aggregated population visualization summary:")
    print(f"  Raw rows: {len(df)}")
    print(f"  Unique coordinate points: {len(agg_df)}")
    print("  Aggregated household_count_sum distribution:")
    print(agg_df["household_count_sum"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]))

    return agg_df

def build_weighted_heat_data(
    aggregated_df,
    lat_col="lat",
    lon_col="lon",
    weight_col="household_count_sum",
    clip_quantile=0.995,
):
    """
    Build weighted heatmap input for Folium HeatMap: [lat, lon, weight].
    Clips outliers, applies log1p transform, normalizes to 0..1.
    """
    required_cols = {lat_col, lon_col, weight_col}
    missing_cols = required_cols - set(aggregated_df.columns)
    if missing_cols:
        raise ValueError(
            f"Aggregated dataframe is missing required heatmap columns: {sorted(missing_cols)}"
        )

    df = aggregated_df[[lat_col, lon_col, weight_col]].copy()
    df = df.dropna(subset=[lat_col, lon_col, weight_col]).copy()

    if len(df) == 0:
        return []

    clip_value = float(df[weight_col].quantile(clip_quantile))
    if not np.isfinite(clip_value) or clip_value <= 0:
        clip_value = float(df[weight_col].max())

    if not np.isfinite(clip_value) or clip_value <= 0:
        df["heat_weight"] = 0.0
    else:
        clipped = df[weight_col].clip(upper=clip_value)
        transformed = np.log1p(clipped)
        max_transformed = float(transformed.max())

        if max_transformed > 0:
            df["heat_weight"] = transformed / max_transformed
        else:
            df["heat_weight"] = 0.0

    print("Weighted heatmap summary:")
    print(f"  Rows used: {len(df)}")
    print(f"  Weight column: {weight_col}")
    print(f"  Clip quantile: {clip_quantile}")
    print(f"  Clip value: {clip_value:.4f}")
    print("  Final normalized heat weight distribution:")
    print(df["heat_weight"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]))

    return df[[lat_col, lon_col, "heat_weight"]].values.tolist()

def add_population_intensity_circles(
    map_obj,
    aggregated_df,
    lat_col="lat",
    lon_col="lon",
    weight_col="household_count_sum",
):
    """
    Optional circle layer for zoomed-in inspection. Hidden by default.
    Larger household totals get bigger, warmer-colored circles.
    """
    required_cols = {lat_col, lon_col, weight_col}
    missing_cols = required_cols - set(aggregated_df.columns)
    if missing_cols:
        raise ValueError(
            f"Aggregated dataframe is missing required circle-layer columns: {sorted(missing_cols)}"
        )

    layer = folium.FeatureGroup(name="Population intensity circles", show=False)

    for _, row in aggregated_df.iterrows():
        value = float(row[weight_col])

        if value <= 50:
            radius = 2
            color = "#2c7fb8"
            fill_opacity = 0.20
        elif value <= 200:
            radius = 3
            color = "#41b6c4"
            fill_opacity = 0.25
        elif value <= 800:
            radius = 5
            color = "#7fcdbb"
            fill_opacity = 0.30
        elif value <= 2500:
            radius = 7
            color = "#c7e9b4"
            fill_opacity = 0.35
        elif value <= 10000:
            radius = 10
            color = "#fdae61"
            fill_opacity = 0.42
        else:
            radius = 13
            color = "#d73027"
            fill_opacity = 0.50

        folium.CircleMarker(
            location=[float(row[lat_col]), float(row[lon_col])],
            radius=radius,
            color=color,
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            opacity=0,
        ).add_to(layer)

    layer.add_to(map_obj)

def build_nearest_hospital_lookup(population_df, hospital_df, sample_size=5000):
    """
    Sample population points and match each to its nearest hospital by
    coordinate distance. Only sampled rows become clickable markers.
    """
    if len(population_df) == 0 or len(hospital_df) == 0:
        return population_df.copy()

    sample_df = population_df.sample(
        min(sample_size, len(population_df)),
        random_state=0
    ).reset_index(drop=True)

    pop_coords = sample_df[["lat", "lon"]].to_numpy(dtype=float)
    hosp_coords = hospital_df[["Latitude", "Longitude"]].to_numpy(dtype=float)

    dists = np.sum((pop_coords[:, None, :] - hosp_coords[None, :, :]) ** 2, axis=2)
    nearest_idx = np.argmin(dists, axis=1)

    sample_df["nearest_hospital_id"] = hospital_df.iloc[nearest_idx]["ID"].to_numpy()
    sample_df["nearest_hospital_lat"] = hospital_df.iloc[nearest_idx]["Latitude"].to_numpy()
    sample_df["nearest_hospital_lon"] = hospital_df.iloc[nearest_idx]["Longitude"].to_numpy()

    return sample_df

def attach_precomputed_distance_for_selected_hospital(interactive_df, distances_df):
    """
    Attach precomputed road distances for the already-selected nearest hospital.
    Only matches on the existing (pop_id, hosp_id) pair, never replaces the
    selected hospital.
    """
    df = interactive_df.copy()
    d = distances_df.copy()

    required_cols = {"pop_id", "hosp_id", "total_dist"}
    if not required_cols.issubset(d.columns):
        raise ValueError("distances file must contain 'pop_id', 'hosp_id', and 'total_dist' columns")

    df["ID"] = ensure_numeric(df["ID"])
    df["nearest_hospital_id"] = ensure_numeric(df["nearest_hospital_id"])

    d["pop_id"] = ensure_numeric(d["pop_id"])
    d["hosp_id"] = ensure_numeric(d["hosp_id"])
    d["total_dist"] = ensure_numeric(d["total_dist"])

    df = df.dropna(subset=["ID", "nearest_hospital_id"]).copy()
    d = d.dropna(subset=["pop_id", "hosp_id", "total_dist"]).copy()

    df["ID"] = df["ID"].astype(int)
    df["nearest_hospital_id"] = df["nearest_hospital_id"].astype(int)
    d["pop_id"] = d["pop_id"].astype(int)
    d["hosp_id"] = d["hosp_id"].astype(int)

    d_pair = (
        d.sort_values(["pop_id", "hosp_id", "total_dist"])
         .groupby(["pop_id", "hosp_id"], as_index=False)
         .first()
         .rename(columns={"total_dist": "distance_to_hospital_km"})
    )

    df = df.merge(
        d_pair[["pop_id", "hosp_id", "distance_to_hospital_km"]],
        left_on=["ID", "nearest_hospital_id"],
        right_on=["pop_id", "hosp_id"],
        how="left"
    )

    df.drop(columns=["pop_id", "hosp_id"], inplace=True, errors="ignore")

    df["distance_source"] = np.where(
        df["distance_to_hospital_km"].notna(),
        "precomputed_selected_hospital",
        "missing"
    )

    after_non_null = df["distance_to_hospital_km"].notna().sum()
    print("Precomputed distance attach summary:")
    print(f"  Distances attached for selected hospital pairs: {after_non_null}")
    print(f"  Missing after pair-merge: {len(df) - after_non_null}")

    return df

# Road network from GeoJSON
# -----------------------------------------------------------------------------
# Load the road network GeoJSON.
# This file contains road geometries and road attributes such as RTT and RST.
# Later, these lines are converted into a graph so NetworkX can search routes.
# -----------------------------------------------------------------------------
def load_roads_geojson(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Road GeoJSON not found: {path}")

    print(f"Loading road GeoJSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = gj.get("features", [])
    print(f"Road features loaded: {len(features)}")

    geom_types = Counter(feat.get("geometry", {}).get("type") for feat in features)
    print(f"Geometry types: {dict(geom_types)}")

    return gj

# -----------------------------------------------------------------------------
# Convert road LineStrings into a NetworkX graph.
# Each coordinate becomes a node and each road segment becomes an edge.
# Each edge stores:
# - length_km: physical road distance
# - travel_time_min: time using speed inferred from RTT/RST
# - speed_kmh, rtt, rst: useful metadata for explanation and modelling
#
# NetworkX can then run shortest_path on either distance or travel time.
# -----------------------------------------------------------------------------
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

        # A road LineString is split into small segments.
        # Each segment becomes one graph edge that can be used in routing.
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

            # Segment travel time is stored directly on the edge.
            # The road-adjusted model later sums these segment times.
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
# Build a KDTree from road-node coordinates.
# KDTree makes nearest-node lookup fast. Without it, snapping every population
# point and hospital to the road network would require comparing against every
# road node, which is too slow for large data.
# -----------------------------------------------------------------------------
def build_graph_kdtree(G):
    node_list = list(G.nodes())
    node_coords = np.array([[n[0], n[1]] for n in node_list], dtype=float)
    tree = cKDTree(node_coords)
    return tree, node_list, node_coords

# -----------------------------------------------------------------------------
# Snap real-world points to the nearest road-graph node.
# Hospitals and population points usually do not lie exactly on a road node, so
# routing needs the closest graph node as the start/end point. The function also
# stores the snap distance so the final road distance can include this offset.
# -----------------------------------------------------------------------------
def snap_points_to_graph_nodes(df, lat_col, lon_col, tree, node_list, label):
    out = df.copy()

    query_points = out[[lon_col, lat_col]].to_numpy(dtype=float)
    dists, idxs = tree.query(query_points, k=1)

    snapped_nodes = [node_list[i] for i in idxs]
    snapped_lon = np.array([n[0] for n in snapped_nodes], dtype=float)
    snapped_lat = np.array([n[1] for n in snapped_nodes], dtype=float)

    out[f"{label}_graph_node"] = snapped_nodes
    out[f"{label}_graph_lon"] = snapped_lon
    out[f"{label}_graph_lat"] = snapped_lat
    out[f"{label}_to_graph_km"] = haversine_km(
        out[lat_col].to_numpy(dtype=float),
        out[lon_col].to_numpy(dtype=float),
        snapped_lat,
        snapped_lon,
    )

    print(f"{label} snapping summary:")
    print(f"  Rows snapped: {len(out)}")
    print(f"  Mean {label}_to_graph_km: {out[f'{label}_to_graph_km'].mean():.4f}")
    print(f"  Median {label}_to_graph_km: {out[f'{label}_to_graph_km'].median():.4f}")
    print(f"  Max {label}_to_graph_km: {out[f'{label}_to_graph_km'].max():.4f}")

    return out

def build_top_k_hospital_popup_data(
    interactive_df,
    hospital_snap_df,
    graph,
    k=3,
):
    """
    Build hospital data using road-adjusted speed model only.
    Routes are only calculated for hospitals within 300 km straight-line distance.
    Result is stored as JSON in top_k_hospitals_json per population row.
    """
    MAX_ROUTE_DISTANCE_KM = 300.0
    out = interactive_df.copy()

    required_pop_cols = {
        "ID",
        "lat",
        "lon",
        "population_graph_node",
        "population_graph_lon",
        "population_graph_lat",
        "population_to_graph_km",
    }
    missing_pop_cols = [c for c in required_pop_cols if c not in out.columns]
    if missing_pop_cols:
        print(f"Cannot build top-k hospital popup data. Missing population columns: {missing_pop_cols}")
        out["top_k_hospitals_json"] = "[]"
        return out

    required_hosp_cols = {
        "ID",
        "Latitude",
        "Longitude",
        "specialized_equipment",
        "hospital_graph_node",
        "hospital_graph_lon",
        "hospital_graph_lat",
        "hospital_to_graph_km",
    }
    missing_hosp_cols = [c for c in required_hosp_cols if c not in hospital_snap_df.columns]
    if missing_hosp_cols:
        print(f"Cannot build top-k hospital popup data. Missing hospital columns: {missing_hosp_cols}")
        out["top_k_hospitals_json"] = "[]"
        return out

    hosp_df = hospital_snap_df.copy()
    hosp_df["ID"] = ensure_numeric(hosp_df["ID"])
    hosp_df = hosp_df.dropna(subset=["ID", "Latitude", "Longitude"]).copy()
    hosp_df["ID"] = hosp_df["ID"].astype(int)

    hosp_coords = hosp_df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    hosp_ids = hosp_df["ID"].to_numpy(dtype=int)

    hospital_lookup = {}
    for _, row in hosp_df.iterrows():
        hospital_lookup[int(row["ID"])] = row

    route_cache = {}
    computed_rows = 0
    no_path_count = 0
    bad_node_count = 0
    skipped_too_far = 0

    for idx, row in out.iterrows():
        pop_lat = row.get("lat")
        pop_lon = row.get("lon")
        pop_node = row.get("population_graph_node")
        pop_graph_lat = row.get("population_graph_lat")
        pop_graph_lon = row.get("population_graph_lon")
        pop_offset_km = row.get("population_to_graph_km")

        if pd.isna(pop_lat) or pd.isna(pop_lon):
            out.at[idx, "top_k_hospitals_json"] = "[]"
            continue

        pop_coord = np.array([[float(pop_lat), float(pop_lon)]], dtype=float)
        coord_dists = np.sum((hosp_coords - pop_coord) ** 2, axis=1)

        top_k = min(k, len(hosp_df))
        top_idx = np.argsort(coord_dists)[:top_k]

        hospital_results = []

        for hosp_array_idx in top_idx:
            hosp_id = int(hosp_ids[hosp_array_idx])
            hosp_row = hospital_lookup[hosp_id]

            specialized = bool(hosp_row.get("specialized_equipment", False))
            hosp_node = hosp_row.get("hospital_graph_node")
            hosp_graph_lat = hosp_row.get("hospital_graph_lat")
            hosp_graph_lon = hosp_row.get("hospital_graph_lon")
            hosp_lat = hosp_row.get("Latitude")
            hosp_lon = hosp_row.get("Longitude")
            hosp_offset_km = hosp_row.get("hospital_to_graph_km")

            straight_line_km = float(haversine_km(pop_lat, pop_lon, hosp_lat, hosp_lon))

            if straight_line_km > MAX_ROUTE_DISTANCE_KM:
                skipped_too_far += 1
                hospital_results.append({
                    "hospital_id": hosp_id,
                    "specialized_equipment": specialized,
                    "distance_km": None,
                    "estimated_travel_time_min": None,
                    "route_available": False,
                    "route_coords": None,
                })
                continue

            route_distance_km = None
            route_time_min = None
            route_coords = None

            if (
                pop_node is not None
                and hosp_node is not None
                and not pd.isna(pop_offset_km)
                and not pd.isna(hosp_offset_km)
                and pop_node in graph
                and hosp_node in graph
            ):
                cache_key = (pop_node, hosp_node)

                if cache_key in route_cache:
                    cached = route_cache[cache_key]
                else:
                    cached = {"km": None, "time_min": None, "coords": None}

                    try:
                        nodes = nx.shortest_path(
                            graph,
                            source=pop_node,
                            target=hosp_node,
                            weight="travel_time_min"
                        )

                        network_km = 0.0
                        network_time_min = 0.0

                        for u, v in zip(nodes[:-1], nodes[1:]):
                            edge = graph[u][v]
                            network_km += float(edge["length_km"])
                            network_time_min += float(edge["travel_time_min"])

                        coords = []
                        add_coord_if_valid(coords, pop_lat, pop_lon)
                        add_coord_if_valid(coords, pop_graph_lat, pop_graph_lon)

                        for node in nodes:
                            lon, lat = node
                            add_coord_if_valid(coords, lat, lon)

                        add_coord_if_valid(coords, hosp_graph_lat, hosp_graph_lon)
                        add_coord_if_valid(coords, hosp_lat, hosp_lon)

                        cached["km"] = float(pop_offset_km) + network_km + float(hosp_offset_km)
                        cached["time_min"] = network_time_min
                        cached["coords"] = coords

                    except nx.NetworkXNoPath:
                        no_path_count += 1
                    except nx.NodeNotFound:
                        bad_node_count += 1

                    route_cache[cache_key] = cached

                route_distance_km = cached["km"]
                route_time_min = cached["time_min"]
                route_coords = cached["coords"]
            else:
                bad_node_count += 1

            hospital_results.append({
                "hospital_id": hosp_id,
                "specialized_equipment": specialized,
                "distance_km": route_distance_km,
                "estimated_travel_time_min": route_time_min,
                "route_available": route_coords is not None and len(route_coords) >= 2,
                "route_coords": route_coords,
            })

        hospital_results.sort(
            key=lambda h: (
                h["estimated_travel_time_min"] is None
                or (isinstance(h["estimated_travel_time_min"], float) and pd.isna(h["estimated_travel_time_min"])),
                float(h["estimated_travel_time_min"])
                if h["estimated_travel_time_min"] is not None
                and not (isinstance(h["estimated_travel_time_min"], float) and pd.isna(h["estimated_travel_time_min"]))
                else float("inf")
            )
        )

        for rank, hosp in enumerate(hospital_results, start=1):
            hosp["rank"] = rank

        out.at[idx, "top_k_hospitals_json"] = json.dumps(hospital_results)
        computed_rows += 1

        if computed_rows % 100 == 0:
            print(f"  Built hospital data for {computed_rows} population rows...")

    print(f"Hospital popup data complete.")
    print(f"  Population rows processed: {computed_rows}")
    print(f"  No-path cases: {no_path_count}")
    print(f"  Bad/missing node cases: {bad_node_count}")
    print(f"  Skipped (>300 km): {skipped_too_far}")
    print(f"  Cached route pairs: {len(route_cache)}")

    return out

# Main map builder
# -----------------------------------------------------------------------------
# Main orchestration function.
# This function calls all earlier helper functions in the correct order:
# load data -> filter data -> create map -> add heatmap -> load hospitals ->
# build road graph -> snap points -> calculate routes/models -> add markers ->
# inject UI -> save HTML.
# -----------------------------------------------------------------------------
def build_population_map(
    population_path,
    hospital_path=None,
    distances_path=None,
    roads_geojson_path=None,
    stroke_facilities_path=None,
    output_path="population_map.html",
    max_marker_points=1000,
    min_lat=None,
    max_lat=None,
):
    population_df = load_pickle(population_path)

    required_pop_cols = {"ID", "lat", "lon"}
    if not required_pop_cols.issubset(population_df.columns):
        raise ValueError("population.pkl must contain 'ID', 'lat', and 'lon' columns")

    if len(population_df) == 0:
        raise ValueError("population.pkl is empty")

    print("Population dataframe loaded.")
    print(f"  Population rows before geographic filtering: {len(population_df)}")
    print(f"  Population columns: {list(population_df.columns)}")

    if min_lat is not None:
        before = len(population_df)
        population_df = population_df[population_df["lat"] >= min_lat].copy()
        print(f"Applied min_lat filter: kept {len(population_df)} / {before} rows (lat >= {min_lat})")

    if max_lat is not None:
        before = len(population_df)
        population_df = population_df[population_df["lat"] <= max_lat].copy()
        print(f"Applied max_lat filter: kept {len(population_df)} / {before} rows (lat <= {max_lat})")

    if len(population_df) == 0:
        raise ValueError("No population rows remain after applying geographic filters.")

    print(f"  Population rows after geographic filtering: {len(population_df)}")

    center_lat = float(population_df["lat"].median())
    center_lon = float(population_df["lon"].median())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron"
    )

    # The heatmap should represent all population data, not only the clickable sample.
    # Therefore, we aggregate the full population dataframe before making heat data.
    aggregated_population_df = aggregate_population_for_visualization(
        population_df=population_df,
        lat_col="lat",
        lon_col="lon",
        weight_col="household_count",
    )

    heat_data = build_weighted_heat_data(
        aggregated_df=aggregated_population_df,
        lat_col="lat",
        lon_col="lon",
        weight_col="household_count_sum",
        clip_quantile=0.995,
    )

    HeatMap(
        heat_data,
        radius=24,
        blur=20,
        max_zoom=17,
        min_opacity=0.12,
        gradient={
            0.05: "#2c7fb8",
            0.20: "#41b6c4",
            0.40: "#a1dab4",
            0.60: "#fecc5c",
            0.78: "#fd8d3c",
            0.90: "#f03b20",
            1.00: "#bd0026",
        },
        name="Population heatmap",
    ).add_to(m)

    add_population_intensity_circles(
        map_obj=m,
        aggregated_df=aggregated_population_df,
        lat_col="lat",
        lon_col="lon",
        weight_col="household_count_sum",
    )

    hospital_marker_lookup = {}
    hospital_panel_data = {}

    # Hospitals
    hospitals_df = None
    if hospital_path and os.path.exists(hospital_path):
        try:
            hospitals_df = load_pickle(hospital_path)

            # Fix ambiguous pandas situation where ID exists both as index and as column
            if isinstance(hospitals_df.index, pd.Index) and hospitals_df.index.name == "ID":
                hospitals_df = hospitals_df.reset_index(drop=True)

            # Extra safety: if ID somehow exists more than once as columns, keep one
            hospitals_df = hospitals_df.loc[:, ~hospitals_df.columns.duplicated()].copy()

            print("Hospitals dataframe loaded.")
            print(f"  Hospital rows: {len(hospitals_df)}")
            print(f"  Hospital columns: {list(hospitals_df.columns)}")
            print(f"  Hospital index name: {hospitals_df.index.name}")

            # Load specialized-equipment data and merge onto hospitals_df
            if stroke_facilities_path is None:
                raise ValueError("stroke_facilities_path is required")

            stroke_df = load_stroke_facilities_csv(stroke_facilities_path)

            hospitals_df["ID"] = ensure_numeric(hospitals_df["ID"])
            hospitals_df = hospitals_df.dropna(subset=["ID"]).copy()
            hospitals_df["ID"] = hospitals_df["ID"].astype(int)

            hospitals_df = hospitals_df.merge(
                stroke_df,
                on="ID",
                how="left"
            )

            hospitals_df["specialized_equipment"] = hospitals_df["specialized_equipment"].fillna(False)

            print("Merged specialized-equipment information into hospitals dataframe.")
            print(hospitals_df["specialized_equipment"].value_counts(dropna=False))

            if "Latitude" in hospitals_df.columns and "Longitude" in hospitals_df.columns:
                cluster = MarkerCluster(name="Hospitals").add_to(m)

                for _, row in hospitals_df.iterrows():
                    hospital_id = safe_int(row.get("ID"))
                    has_specialized = bool(row.get("specialized_equipment", False))

                    # Use Name_English from stroke-facs CSV.
                    name_text = ""
                    if "Name_English" in row.index and pd.notna(row.get("Name_English")):
                        name_text = str(row["Name_English"]).strip()

                    # Use Address_CSV from stroke-facs CSV.
                    address_text = ""
                    if "Address_CSV" in row.index and pd.notna(row.get("Address_CSV")):
                        address_text = str(row["Address_CSV"]).strip()

                    hospital_icon_html = build_hospital_icon_html(
                        has_specialized=has_specialized,
                        size_px=24,
                    )

                    marker = folium.Marker(
                        location=[float(row["Latitude"]), float(row["Longitude"])],
                        icon=folium.DivIcon(
                            html=hospital_icon_html,
                            icon_size=(24, 24),
                            icon_anchor=(12, 12),
                            class_name="empty"
                        ),
                    ).add_to(cluster)

                    if hospital_id is not None:
                        hospital_marker_lookup[str(hospital_id)] = marker.get_name()

                        hospital_panel_data[str(hospital_id)] = {
                            "hospital_id": hospital_id,
                            "hospital_name": name_text,
                            "specialized_equipment": has_specialized,
                            "address_text": address_text,
                        }

        except Exception as e:
            print(f"Warning: could not load hospitals file: {e}")
            hospitals_df = None

    if hospitals_df is None:
        raise ValueError("Hospitals file is required for correct nearest hospital assignment.")

    # IMPORTANT FIX:
    # ALWAYS choose nearest hospital with the original correct method first
    # Only these sampled population rows become clickable population markers.
    # This keeps the browser map responsive and prevents the HTML file from becoming huge.
    interactive_df = build_nearest_hospital_lookup(
        population_df=population_df,
        hospital_df=hospitals_df,
        sample_size=max_marker_points
    )
    print("Assigned nearest hospital IDs using ORIGINAL coordinate-based nearest-hospital logic.")

    # Attach precomputed distances only for the already chosen hospital pair
    if distances_path and os.path.exists(distances_path):
        try:
            distances_df = load_pickle(distances_path)
            print("Distances dataframe loaded.")
            print(f"  Distance rows: {len(distances_df)}")
            print(f"  Distance columns: {list(distances_df.columns)}")

            interactive_df = attach_precomputed_distance_for_selected_hospital(
                interactive_df=interactive_df,
                distances_df=distances_df,
            )

        except Exception as e:
            print(f"Warning: could not attach precomputed distances: {e}")
            interactive_df["distance_to_hospital_km"] = np.nan
            interactive_df["distance_source"] = "missing"
    else:
        interactive_df["distance_to_hospital_km"] = np.nan
        interactive_df["distance_source"] = "missing"

    print(f"Interactive sample size actually used: {len(interactive_df)}")

    precomputed_coverage = interactive_df["distance_to_hospital_km"].notna().sum()
    print(f"Precomputed distance coverage for selected hospitals: {precomputed_coverage} / {len(interactive_df)}")

    # Build graph from roads GeoJSON
    G = None
    tree = None
    node_list = None

    if roads_geojson_path and os.path.exists(roads_geojson_path):
        try:
            roads_geojson = load_roads_geojson(roads_geojson_path)

            G = build_graph_from_roads_geojson(roads_geojson)
            tree, node_list, _ = build_graph_kdtree(G)
            print("KDTree built for road graph nodes.")

        except Exception as e:
            print(f"Warning: could not build graph from road GeoJSON: {e}")
            G = None
            tree = None
            node_list = None
    else:
        print("No road GeoJSON found; computed network distances will be skipped.")

    # Snap population + hospitals to graph
    if G is not None and tree is not None:
        try:
            interactive_df = snap_points_to_graph_nodes(
                interactive_df,
                lat_col="lat",
                lon_col="lon",
                tree=tree,
                node_list=node_list,
                label="population",
            )

            hospital_snap_df = hospitals_df.copy()
            hospital_snap_df = snap_points_to_graph_nodes(
                hospital_snap_df,
                lat_col="Latitude",
                lon_col="Longitude",
                tree=tree,
                node_list=node_list,
                label="hospital",
            )

            interactive_df = build_top_k_hospital_popup_data(
                interactive_df=interactive_df,
                hospital_snap_df=hospital_snap_df,
                graph=G,
                k=3,
            )

        except Exception as e:
            print(f"Warning: graph snapping or routing failed: {e}")
    else:
        print("Skipping graph snapping/routing because graph is unavailable.")

    final_coverage = interactive_df["distance_to_hospital_km"].notna().sum()
    print(f"Final distance coverage: {final_coverage} / {len(interactive_df)}")

    if "distance_source" in interactive_df.columns:
        print("Distance source breakdown:")
        print(interactive_df["distance_source"].value_counts(dropna=False))

    marker_cluster = MarkerCluster(name="Population points").add_to(m)

    route_data = {}
    population_marker_lookup = {}
    population_panel_data = {}

    for _, row in interactive_df.iterrows():
        pop_id = safe_int(row.get("ID"))

        household_count = (
            int(row["household_count"])
            if "household_count" in row and not pd.isna(row.get("household_count"))
            else "N/A"
        )

        top_hospitals = []
        if row.get("top_k_hospitals_json"):
            try:
                top_hospitals = json.loads(row["top_k_hospitals_json"])
            except Exception:
                top_hospitals = []

        hospital_cards = []

        for hosp in top_hospitals:
            hospital_id = hosp.get("hospital_id", "N/A")
            specialized = bool(hosp.get("specialized_equipment", False))

            hospital_cards.append({
                "rank": hosp.get("rank", None),
                "hospital_id": hospital_id,
                "specialized_equipment": specialized,
                "distance_km": hosp.get("distance_km"),
                "estimated_travel_time_min": hosp.get("estimated_travel_time_min"),
                "route_available": hosp.get("route_available", False),
            })

        if pop_id is not None:
            population_panel_data[str(pop_id)] = {
                "population_id": pop_id,
                "household_count": household_count,
                "hospitals": hospital_cards,
            }

        marker = folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=6,
            color="blue",
            fill=True,
            fill_color="cyan",
            fill_opacity=0.7,
        ).add_to(marker_cluster)

        if pop_id is not None:
            population_marker_lookup[str(pop_id)] = marker.get_name()

        if pop_id is not None:
            route_data[str(pop_id)] = {}

            for hosp in top_hospitals:
                hospital_id = hosp.get("hospital_id", None)
                if hospital_id is None:
                    continue

                coords = hosp.get("route_coords", None)
                route_available = bool(hosp.get("route_available", False))

                if (
                    route_available
                    and isinstance(coords, list)
                    and len(coords) >= 2
                ):
                    route_data[str(pop_id)][str(hospital_id)] = coords


    inject_side_panel_and_routes(
        m,
        route_data,
        population_panel_data,
        population_marker_lookup,
        hospital_panel_data,
        hospital_marker_lookup,
    )

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Map saved to {output_path}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an HTML map of population points with correct nearest hospital ID, road-network distance, and clickable route drawing."
    )
    parser.add_argument(
        "--population",
        default="Data_vietnam/population.pkl",
        help="Path to population pickle file"
    )
    parser.add_argument(
        "--hospitals",
        default="Data_vietnam/existing_hospitals_100/all_hospitals.pkl",
        help="Path to hospitals pickle file"
    )
    parser.add_argument(
        "--distances",
        default="Data_vietnam/existing_hospitals_100/distances_osm_max_300km.pkl",
        help="Path to precomputed road-distance pickle file"
    )
    parser.add_argument(
        "--roads-geojson",
        default="Data_vietnam/road_osm_preprocessed.geojson",
        help="Path to road network GeoJSON"
    )
    parser.add_argument(
        "--stroke-facilities",
        default="Data_vietnam/stroke-facs-100-en.csv",
        help="Path to CSV file with specialized-equipment information"
    )
    parser.add_argument(
        "--output",
        default="population_map.html",
        help="Output HTML file path"
    )
    parser.add_argument(
        "--max-marker-points",
        type=int,
        default=1000,
        help="Maximum population markers for interactive clicks"
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        default=None,
        help="Only process population points with latitude >= this value"
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=None,
        help="Only process population points with latitude <= this value"
    )

    args = parser.parse_args()

    build_population_map(
        population_path=args.population,
        hospital_path=args.hospitals,
        distances_path=args.distances,
        roads_geojson_path=args.roads_geojson,
        stroke_facilities_path=args.stroke_facilities,
        output_path=args.output,
        max_marker_points=args.max_marker_points,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
    )