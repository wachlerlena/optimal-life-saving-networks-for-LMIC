# =============================================================================
# UPGRADE MODEL COMMON - SHARED HELPER FUNCTIONS
# =============================================================================
# Purpose of this file:
# This file contains the shared code used by both optimization models:
# 1. Load and clean hospital, population and distance data.
# 2. Build the common ModelData object used by the solvers.
# 3. Convert travel time into outcome weights for the weighted models.
# 4. Calculate summary statistics after a model has been solved.
# 5. Write model results to CSV and JSON files.
#
# How to use this file:
# Do not run this file directly. It is imported by:
# - optimize_upgrade_base_mclp.py
# - optimize_upgrade_weighted_assignment.py
#
# Main modelling idea:
# - Fixed facilities are hospitals that already have advanced stroke care.
# - Candidate facilities are all hospitals that can be selected by the model.
# - Arcs are population-to-hospital travel-time records.
# - Weighted models use the exponential or logistic function to value earlier
#   treatment more highly than later treatment.
# =============================================================================

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

# -----------------------------------------------------------------------------
# PATHS AND MODEL CONSTANTS
# -----------------------------------------------------------------------------
# BASE_DIR is the folder where this Python file is located.
# All input and output paths are defined relative to this folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_vietnam"
RESULTS_DIR = BASE_DIR / "Results" / "new_upgrade_models"
ROADS_GEOJSON_PATH = DATA_DIR / "road_osm_preprocessed.geojson"

# Default speed used when a stored road-network travel time is not available.
SPEED_KMH = 35.0
# 270 minutes is the treatment window used as the hard cut-off in the base MCLP
# and as the point after which the weighted functions are set to zero.
SERVICE_RADIUS_MINUTES = 270.0
# Parameters for the two time-to-treatment benefit functions.
# EXPO_TAU controls the speed of exponential decay.
EXPO_TAU = 150.0
LOGISTIC_K = 0.01866932674
LOGISTIC_T0 = 121.0

# -----------------------------------------------------------------------------
# DATA CONTAINER
# -----------------------------------------------------------------------------
# ModelData keeps the cleaned input tables together so each solver receives
# exactly the same population, facility, distance, fixed-facility and candidate
# data. frozen=True prevents accidental changes to the container fields.
@dataclass(frozen=True)
class ModelData:
    population: pd.DataFrame
    facilities: pd.DataFrame
    arcs: pd.DataFrame
    fixed_facility_ids: list[int]
    candidate_facility_ids: list[int]

#Small helper functions for cleaning and converting data values
def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()

# Convert Vietnamese yes/no fields to a Boolean.
# In the source CSV, "Có" means yes.
def _as_yes(value: object) -> bool:
    return _clean_text(value) == "Có"


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
# ROAD GRAPH HELPER
# -----------------------------------------------------------------------------
# Convert a road GeoJSON object into a NetworkX graph.
# This helper is kept here so the same graph logic can be reused if needed by
# both optimization and distance-building code.
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

# ---------------------------------------------------------------------------
# TIME-TO-TREATMENT BENEFIT FUNCTIONS
# ---------------------------------------------------------------------------
# These functions transform travel time into a benefit weight between 0 and 1.
# The optimization objective multiplies this weight by household_count.
def exponential_weight(minutes: Iterable[float] | float) -> np.ndarray:
    minutes_array = np.asarray(minutes, dtype=float)
    weights = np.exp(-minutes_array / EXPO_TAU)
    weights[minutes_array > 270.0] = 0.0
    return weights


def logistic_weight(minutes: Iterable[float] | float) -> np.ndarray:
    minutes_array = np.asarray(minutes, dtype=float)
    weights = 1.0 - 1.0 / (1.0 + np.exp(-LOGISTIC_K * (minutes_array - LOGISTIC_T0)))
    weights[minutes_array > 270.0] = 0.0
    return weights


WEIGHT_FUNCTIONS: dict[str, Callable[[Iterable[float] | float], np.ndarray]] = {
    "exponential": exponential_weight,
    "logistic": logistic_weight,
}


# Create the  results directory if it does not already exist.
def ensure_results_dir(*parts: str) -> Path:
    path = RESULTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
# Load the hospital metadata and stroke-capability CSV, then merge them into one
# facility table. The facility_id alignment matters: TT in the CSV equals
# hospital ID + 1, so the code subtracts 1.
def load_facilities() -> pd.DataFrame:
    facility_csv = pd.read_csv(DATA_DIR / "stroke-facs-100-en.csv")
    hospital_meta = pd.read_pickle(DATA_DIR / "existing_hospitals_100" / "all_hospitals.pkl")

    intervention_col = next(col for col in facility_csv.columns if "can" in col and "thiệp" in col)
    thrombolysis_col = next(col for col in facility_csv.columns if "tiêu sợi" in col)

    facility_csv = facility_csv.copy()
    facility_csv["facility_id"] = pd.to_numeric(facility_csv["TT"], errors="coerce").astype("Int64") - 1
    facility_csv["name_vietnamese"] = facility_csv["Tên bệnh viện"].map(_clean_text)
    facility_csv["name_english"] = facility_csv["Name_English"].map(_clean_text)
    facility_csv["facility_type"] = facility_csv["Loại hình"].map(_clean_text)
    facility_csv["address"] = facility_csv["Địa chỉ"].map(_clean_text)
    facility_csv["fixed_open_default"] = facility_csv[intervention_col].map(_as_yes)
    facility_csv["is_intervention_center"] = facility_csv["fixed_open_default"]
    facility_csv["is_thrombolysis_center"] = facility_csv[thrombolysis_col].map(_as_yes)
    facility_csv["csv_latitude"] = pd.to_numeric(facility_csv["latitude"], errors="coerce")
    facility_csv["csv_longitude"] = pd.to_numeric(facility_csv["longitude"], errors="coerce")
    facility_csv = facility_csv[
        [
            "facility_id",
            "name_vietnamese",
            "name_english",
            "facility_type",
            "address",
            "fixed_open_default",
            "is_intervention_center",
            "is_thrombolysis_center",
            "csv_latitude",
            "csv_longitude",
        ]
    ]

    hospital_meta = hospital_meta.rename(columns={"ID": "facility_id"}).copy()
    hospital_meta["facility_id"] = pd.to_numeric(hospital_meta["facility_id"], errors="coerce").astype("Int64")
    facilities = hospital_meta.merge(facility_csv, on="facility_id", how="left")
    facilities["Latitude"] = pd.to_numeric(facilities["Latitude"], errors="coerce")
    facilities["Longitude"] = pd.to_numeric(facilities["Longitude"], errors="coerce")
    facilities["lat"] = facilities["Latitude"].fillna(facilities["csv_latitude"])
    facilities["lon"] = facilities["Longitude"].fillna(facilities["csv_longitude"])
    facilities["name_english"] = facilities["name_english"].fillna("")
    facilities["name_vietnamese"] = facilities["name_vietnamese"].fillna("")
    facilities["facility_type"] = facilities["facility_type"].fillna("")
    facilities["address"] = facilities["address"].fillna("")
    facilities["fixed_open_default"] = facilities["fixed_open_default"].fillna(False).astype(bool)
    facilities["is_intervention_center"] = facilities["is_intervention_center"].fillna(False).astype(bool)
    facilities["is_thrombolysis_center"] = facilities["is_thrombolysis_center"].fillna(False).astype(bool)
    facilities = facilities.sort_values("facility_id").reset_index(drop=True)
    return facilities


def load_population() -> pd.DataFrame:
    population = pd.read_pickle(DATA_DIR / "population.pkl")
    population = population.rename(columns={"ID": "pop_id"}).copy()
    population["pop_id"] = pd.to_numeric(population["pop_id"], errors="coerce").astype("Int64")
    population["household_count"] = pd.to_numeric(population["household_count"], errors="coerce").fillna(0.0)
    population["lat"] = pd.to_numeric(population["lat"], errors="coerce")
    population["lon"] = pd.to_numeric(population["lon"], errors="coerce")
    return population

# Load the precomputed population-to-hospital distance matrix.
# If road-network travel minutes exist, use them. Otherwise convert distance to
# travel time with the constant SPEED_KMH assumption.
def load_distances() -> pd.DataFrame:
    distances = pd.read_pickle(DATA_DIR / "existing_hospitals_100" / "distances_osm_max_300km.pkl")
    distances = distances.copy()
    distances["pop_id"] = pd.to_numeric(distances["pop_id"], errors="coerce").astype("Int64")
    distances["hosp_id"] = pd.to_numeric(distances["hosp_id"], errors="coerce").astype("Int64")
    distances["total_dist"] = pd.to_numeric(distances["total_dist"], errors="coerce")
    if "travel_minutes_network" in distances.columns:
        distances["travel_minutes"] = pd.to_numeric(
            distances["travel_minutes_network"], errors="coerce"
        )
    else:
        distances["travel_minutes"] = distances["total_dist"] / SPEED_KMH * 60.0
    cols = ["pop_id", "hosp_id", "total_dist", "travel_minutes"]
    if "travel_minutes_network" in distances.columns:
        cols.append("travel_minutes_network")
    return distances[cols]


def build_model_data() -> ModelData:
    """
    Build the full input object used by both optimization models.

    This function combines three prepared datasets:
    1. population points,
    2. hospital/facility information,
    3. precomputed population-to-hospital travel times.

    The result is a ModelData object containing clean population data,
    clean facility data, all feasible assignment arcs, fixed existing stroke
    centers and all candidate hospitals that may be selected by the model.
    """
    
    # Load cleaned population points from population.pkl.
    # Each row represents one population grid cell.
    population = load_population()

    # Load hospital metadata and stroke-care capability information.
    # This identifies which hospitals already have advanced stroke care
    # and which hospitals are possible upgrade candidates.
    facilities = load_facilities()

    # Load the precomputed travel-time matrix between population points
    # and hospitals. This avoids recomputing road-network distances inside
    # the optimization model.
    distances = load_distances()

    # Keep only the population columns needed by the optimization model.
    # Sorting by pop_id gives stable and reproducible output files.
    population = population[["pop_id", "lat", "lon", "household_count"]].sort_values("pop_id").reset_index(drop=True)

    facilities = facilities[facilities["facility_id"].notna()].copy()
    facilities["facility_id"] = facilities["facility_id"].astype(int)

    # Existing intervention/thrombectomy centers are fixed open.
    # The model does not choose these; they are always available.
    fixed_facility_ids = facilities.loc[facilities["fixed_open_default"], "facility_id"].astype(int).tolist()

    # All hospitals with a valid facility_id are included as candidate
    # facilities in the model. Existing fixed centers are also included here,
    # but B_param later prevents selecting them again as upgrades.
    candidate_facility_ids = facilities["facility_id"].astype(int).tolist()

# Merge travel-time rows with population demand.
    # Each row now represents one possible population-to-hospital connection
    # and includes the household count of that population point.
    arcs = distances.merge(population, on="pop_id", how="inner")

    # Add hospital attributes to every population-to-hospital arc.
    # left_on="hosp_id" comes from the distance matrix.
    # right_on="facility_id" comes from the cleaned hospital table.
    arcs = arcs.merge(
        facilities[
            [
                "facility_id",
                "name_english",
                "name_vietnamese",
                "facility_type",
                "address",
                "lat",
                "lon",
                "fixed_open_default",
                "is_intervention_center",
                "is_thrombolysis_center",
            ]
        ],
        left_on="hosp_id",
        right_on="facility_id",
        how="inner",
        suffixes=("", "_facility"),
    )
    # Compute the exponential and logistic time-benefit weights for each arc.
    arcs["weight_exponential"] = exponential_weight(arcs["travel_minutes"])
    arcs["weight_logistic"] = logistic_weight(arcs["travel_minutes"])
    # Sort arcs so that nearest hospitals appear first for each population point.
    # This also makes later filtering, assignment, and output reproducible.
    arcs = arcs.sort_values(["pop_id", "travel_minutes", "hosp_id"]).reset_index(drop=True)

    return ModelData(
        population=population,
        facilities=facilities,
        arcs=arcs,
        fixed_facility_ids=fixed_facility_ids,
        candidate_facility_ids=candidate_facility_ids,
    )

# Weighted quantile for travel-time summaries. Household count is used as the
# weight, so larger population cells matter more.
def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    if values.empty:
        return float("nan")
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    order = np.argsort(values_array)
    values_sorted = values_array[order]
    weights_sorted = weights_array[order]
    cumulative = np.cumsum(weights_sorted)
    cutoff = quantile * weights_sorted.sum()
    index = np.searchsorted(cumulative, cutoff, side="left")
    index = min(index, len(values_sorted) - 1)
    return float(values_sorted[index])

# Weighted mean travel time. Returns NaN when there is no population weight.
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        return float("nan")
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))

# Count and share population by relevant travel-time bands.
# These bands are used to compare the models after optimization.
def compute_band_statistics(travel_minutes: pd.Series, weights: pd.Series) -> dict[str, float]:
    total_weight = float(weights.sum()) if len(weights) else 0.0
    if total_weight == 0.0:
        return {
            "population_lt_90": 0.0,
            "population_91_180": 0.0,
            "population_181_270": 0.0,
            "population_gt_270": 0.0,
            "share_lt_90": 0.0,
            "share_91_180": 0.0,
            "share_181_270": 0.0,
            "share_gt_270": 0.0,
            "share_within_90_min": 0.0,
            "share_within_180_min": 0.0,
            "share_within_270_min": 0.0,
        }

    travel = np.asarray(travel_minutes, dtype=float)
    pop_weights = np.asarray(weights, dtype=float)
    lt_90 = float(pop_weights[travel <= 90.0].sum())
    b_91_180 = float(pop_weights[(travel > 90.0) & (travel <= 180.0)].sum())
    b_181_270 = float(pop_weights[(travel > 180.0) & (travel <= 270.0)].sum())
    gt_270 = float(pop_weights[travel > 270.0].sum())

    share_lt_90 = lt_90 / total_weight
    share_91_180 = b_91_180 / total_weight
    share_181_270 = b_181_270 / total_weight
    share_gt_270 = gt_270 / total_weight

    return {
        "population_lt_90": lt_90,
        "population_91_180": b_91_180,
        "population_181_270": b_181_270,
        "population_gt_270": gt_270,
        "share_lt_90": share_lt_90,
        "share_91_180": share_91_180,
        "share_181_270": share_181_270,
        "share_gt_270": share_gt_270,
        "share_within_90_min": share_lt_90,
        "share_within_180_min": share_lt_90 + share_91_180,
        "share_within_270_min": share_lt_90 + share_91_180 + share_181_270,
    }

# Build a clean output table for all open facilities:
# existing fixed facilities plus newly selected upgrades.
def build_selected_facilities(
    facilities: pd.DataFrame,
    open_facility_ids: list[int],
    model_name: str,
    scenario: str,
    b: int,
) -> pd.DataFrame:
    selected = facilities[facilities["facility_id"].isin(open_facility_ids)].copy()
    selected = selected.sort_values("facility_id").reset_index(drop=True)
    selected["selection_order"] = np.arange(1, len(selected) + 1)
    selected["model"] = model_name
    selected["scenario"] = scenario
    selected["b"] = b
    return selected[
        [
            "facility_id",
            "name_vietnamese",
            "name_english",
            "facility_type",
            "address",
            "lat",
            "lon",
            "Latitude",
            "Longitude",
            "hosp_dist_road_estrada",
            "is_thrombolysis_center",
            "is_intervention_center",
            "fixed_open_default",
            "selection_order",
            "model",
            "scenario",
            "b",
        ]
    ]

# Build one output row per population point showing the assigned hospital,
# travel time, coverage bucket and objective contribution.
def build_population_summary(
    assignment_df: pd.DataFrame,
    model_name: str,
    scenario: str,
    b: int,
) -> pd.DataFrame:
    population_summary = assignment_df.copy().sort_values("pop_id").reset_index(drop=True)
    population_summary["model"] = model_name
    population_summary["scenario"] = scenario
    population_summary["b"] = b
    population_summary["covered_within_radius"] = population_summary["best_travel_minutes"] <= SERVICE_RADIUS_MINUTES
    population_summary["bucket"] = np.select(
        [
            population_summary["best_travel_minutes"] <= 90.0,
            population_summary["best_travel_minutes"] <= 180.0,
            population_summary["best_travel_minutes"] <= 270.0,
        ],
        ["<90", "91-180", "181-270"],
        default=">270",
    )
    return population_summary[
        [
            "pop_id",
            "lat",
            "lon",
            "household_count",
            "best_travel_minutes",
            "best_facility_id",
            "best_facility_name",
            "covered_within_radius",
            "objective_weight",
            "bucket",
            "model",
            "scenario",
            "b",
        ]
    ]

# Combine selected-facility output, population-level output, and aggregate
# metrics into the final summary used by both model scripts.
def build_run_summary(
    model_name: str,
    scenario: str,
    b: int,
    facilities: pd.DataFrame,
    open_facility_ids: list[int],
    assignment_df: pd.DataFrame,
    objective_value: float,
    total_demand_weight: float | None = None,
    weight_scenario: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    selected_facilities = build_selected_facilities(facilities, open_facility_ids, model_name, scenario, b)
    fixed_ids = facilities.loc[facilities["fixed_open_default"], "facility_id"].astype(int).tolist()
    selected_ids = [facility_id for facility_id in selected_facilities["facility_id"].astype(int).tolist() if facility_id not in set(fixed_ids)]

    if total_demand_weight is None:
        total_demand_weight = float(assignment_df["household_count"].sum())
    band_stats = compute_band_statistics(assignment_df["best_travel_minutes"], assignment_df["household_count"])
    weighted_mean_minutes = weighted_mean(assignment_df["best_travel_minutes"], assignment_df["household_count"])
    median_minutes = weighted_quantile(assignment_df["best_travel_minutes"], assignment_df["household_count"], 0.5)
    p90_minutes = weighted_quantile(assignment_df["best_travel_minutes"], assignment_df["household_count"], 0.9)
    covered_weight_share = float(objective_value / total_demand_weight) if total_demand_weight else 0.0

    summary: dict[str, object] = {
        "model": model_name,
        "scenario": scenario,
        "selected_count": len(selected_ids),
        "fixed_count": len(fixed_ids),
        "objective_value": float(objective_value),
        "total_demand_weight": float(total_demand_weight),
        "covered_weight_share": covered_weight_share,
        "weighted_mean_travel_minutes": weighted_mean_minutes,
        "median_travel_minutes": median_minutes,
        "p90_travel_minutes": p90_minutes,
        "selected_facilities": selected_ids,
        "fixed_facilities": fixed_ids,
        "selected_names": selected_facilities["name_english"].fillna("").tolist(),
        "b": int(b),
        "coverage_radius_minutes": SERVICE_RADIUS_MINUTES,
        **band_stats,
    }
    if weight_scenario is not None:
        summary["weight_scenario"] = weight_scenario

    population_summary = build_population_summary(assignment_df, model_name, scenario, b)
    return summary, selected_facilities, population_summary

# Write the three standard output files for each run:
# - JSON summary
# - selected facilities CSV
# - population assignment/summary CSV
def write_run_outputs(
    output_dir: Path,
    file_prefix: str,
    summary: dict[str, object],
    selected_facilities: pd.DataFrame,
    population_summary: pd.DataFrame,
) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{file_prefix}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    selected_facilities.to_csv(output_dir / f"{file_prefix}_selected_facilities.csv", index=False)
    population_summary.to_csv(output_dir / f"{file_prefix}_population_summary.csv", index=False)
