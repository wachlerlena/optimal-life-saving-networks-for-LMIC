# =============================================================================
# INDIVIDUAL ARTIFACT MAP - WEIGHTED HOSPITAL UPGRADE MODEL
# =============================================================================
# Purpose of this file:
# This script builds the interactive HTML map for the weighted hospital-upgrade
# model. It uses the same base population/hospital/road-map functions as
# build_population_map.py, but adds one extra layer of functionality:
# a budget slider that shows which hospitals are selected for upgrade by the
# weighted optimization model for each budget level.
#
# Main workflow:
# 1. Load population, hospital, road-network, stroke-facility, and distance data.
# 2. Load selected upgrade hospitals from the weighted Pareto CSV.
# 3. Build a population heatmap and hospital marker layer.
# 4. Add clickable sampled population points with nearest-hospital route data.
# 5. Inject the side panel from map_ui.py.
# 6. Inject a budget slider that recolors hospitals by selected upgrade status.
# 7. Save the finished Folium map as an HTML file.
#
# Color meaning for hospital markers:
# - Green: hospital already has advanced/specialized stroke capability.
# - Red: hospital does not currently have advanced capability.
# - Blue: hospital is selected for upgrade at the active budget level.
#
# Important limitation:
# The map does not solve the optimization model. It only visualizes results that
# were already written to the weighted Pareto CSV by the optimization code.
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import folium
import networkx as nx
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

# Reuse helper functions from the main group map file.
# This avoids duplicating road-graph, heatmap, hospital-icon, and routing logic.
from build_population_map import (
    add_coord_if_valid,
    add_population_intensity_circles,
    aggregate_population_for_visualization,
    attach_precomputed_distance_for_selected_hospital,
    build_graph_from_roads_geojson,
    build_graph_kdtree,
    build_hospital_icon_html,
    build_nearest_hospital_lookup,
    build_top_k_hospital_popup_data,
    build_weighted_heat_data,
    ensure_numeric,
    load_pickle,
    load_roads_geojson,
    load_stroke_facilities_csv,
    safe_int,
    snap_points_to_graph_nodes,
)

# UI helper that adds the right-side click panel and route drawing logic.
from map_ui import inject_side_panel_and_routes

# Project root is assumed to be one folder above this script.
# All default data paths are built relative to this location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default Pareto result location for the weighted thesis model.
# The CSV must contain at least a budget column and a selected_facility_ids column.
PARETO_PATHS = {
    "weighted_thesis": str(PROJECT_ROOT / "Pareto" / "Weighted" / "pareto_curve_results.csv"),
}


# -----------------------------------------------------------------------------
# SMALL PARSING HELPERS
# -----------------------------------------------------------------------------

def parse_budget_dir_name(path_obj: Path):
    """
    Extract the budget number from a folder name like b0, b1, b2, etc.

    This helper is kept for compatibility with older result-folder workflows.
    In the current weighted map script, selected upgrades are read from a Pareto
    CSV instead of from budget folders.
    """
    match = re.match(r"^b(\d+)$", path_obj.name)
    return int(match.group(1)) if match else None


def parse_id_set(value):
    """
    Convert a stored selected-facility value into a Python set of integers.

    The Pareto CSV stores selected hospital IDs as text, for example:
    "[12, 34, 56]" or "12, 34, 56".
    This function converts that text into {12, 34, 56}.
    Invalid tokens are ignored instead of crashing the map generation.
    """
    # Empty or missing values mean no selected upgrade facilities.
    if pd.isna(value):
        return set()

    # Convert to clean text before parsing.
    text = str(value).strip()
    if not text:
        return set()

    # Remove surrounding list brackets if the CSV stores IDs as "[1, 2, 3]".
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    out = set()

    # Split by comma and try to convert each token to an integer hospital ID.
    for token in text.split(","):
        token = token.strip().strip("\"'")
        if token:
            try:
                out.add(int(token))
            except ValueError:
                pass

    return out


def find_first_present_column(df, candidates):
    """
    Find the first matching column name from a list of possible names.

    This makes the script tolerant to small naming differences in the Pareto CSV,
    such as "b" versus "budget". Matching is case-insensitive.
    """
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def parse_bool_series(series):
    """
    Convert a text/numeric pandas Series into True/False values.

    This helper is kept for compatibility with earlier versions of the artifact.
    Values such as "1", "true", "yes", and "y" are treated as True.
    """
    truthy = {"1", "true", "t", "yes", "y"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


# -----------------------------------------------------------------------------
# LOAD WEIGHTED MODEL SELECTED UPGRADES
# -----------------------------------------------------------------------------

def load_selected_upgrades_by_budget(
    results_root=None,
    model=None,
    scenario=None,
    pareto_csv_path=None,
):
    """
    Load selected upgrade hospital IDs for each budget from the Pareto CSV.

    The map needs a dictionary like:
        {0: set(), 1: {12}, 2: {12, 34}, ...}

    The budget slider then uses this dictionary to recolor hospital markers.
    The function expands missing intermediate budgets by carrying forward the
    previous selected set, so the slider has a valid state for every integer
    budget from 0 to max_budget.
    """
    # Resolve and validate the Pareto CSV path.
    pareto_path = Path(pareto_csv_path)

    if not pareto_path.exists():
        raise FileNotFoundError(
            f"Pareto file not found: {pareto_path}"
        )

    # sep=None lets pandas infer comma/semicolon separators.
    pareto = pd.read_csv(pareto_path, sep=None, engine="python")

    selected = {}

    # Find the budget column. Usually this is called "b".
    b_col = find_first_present_column(
        pareto,
        ["b", "budget"]
    )

    # Find the selected hospital ID column written by the optimization model.
    ids_col = find_first_present_column(
        pareto,
        [
            "selected_facility_ids",
        ]
    )

    # Fail early if the required columns are not present.
    if b_col is None:
        raise ValueError("Could not find budget column in Pareto file.")

    if ids_col is None:
        raise ValueError(
            "Could not find selected facility ID column in Pareto file."
        )

    # Clean budget values and drop rows that cannot be assigned to a budget.
    pareto[b_col] = ensure_numeric(pareto[b_col])
    pareto = pareto.dropna(subset=[b_col])

    # Parse selected hospital IDs for every budget row.
    for _, row in pareto.iterrows():
        selected[int(row[b_col])] = parse_id_set(row[ids_col])

    # Budget 0 should always mean no additional upgrades.
    if 0 not in selected:
        selected[0] = set()

    max_budget = max(selected.keys())

    # Expand missing budget levels so the slider can move one step at a time.
    expanded = {}
    current = set()

    for b in range(max_budget + 1):
        if b in selected:
            current = set(selected[b])
        expanded[b] = set(current)

    # Print the loaded selection sets for verification in the terminal.
    print("Selected upgrades loaded from Pareto CSV:")
    for b in range(max_budget + 1):
        print(f"  b={b}: {sorted(expanded[b])}")

    return expanded


# -----------------------------------------------------------------------------
# INJECT BUDGET SLIDER INTO THE FOLIUM HTML MAP
# -----------------------------------------------------------------------------

def inject_budget_slider_control(
    map_obj,
    hospital_marker_lookup,
    hospital_upgrade_state,
    selected_by_budget,
    model_title="Exponential  weighted Model",
):
    """
    Add the weighted-model budget slider to the map HTML.

    The slider does not recompute anything. It only changes hospital marker colors
    based on selected_by_budget:
    - green = already advanced/specialized hospital,
    - red = not advanced and not selected,
    - blue = selected for upgrade under the active budget.
    """
    # Highest budget controls the slider maximum.
    max_budget = max(selected_by_budget)

    # Convert Python dictionaries to JavaScript-compatible JSON strings.
    state_json = json.dumps(hospital_upgrade_state)
    selected_json = json.dumps({str(k): sorted(v) for k, v in selected_by_budget.items()})

    # HTML, CSS, and JavaScript inserted directly into the Folium map.
    # This creates the visible slider panel and updates marker icons in the browser.
    html = f"""
    <style>
        #budget-slider-panel {{
            position: fixed;
            top: 16px;
            left: 16px;
            z-index: 9999;
            background: rgba(255,255,255,0.97);
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
            padding: 10px 12px;
            width: 330px;
            font-family: Arial, sans-serif;
        }}
        #budget-slider-title {{
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        #budget-slider-value {{
            color: #1f6feb;
        }}
        #budget-slider-input {{
            width: 100%;
        }}
        #selected-facility-overview {{
            margin-top: 8px;
            font-size: 12px;
            line-height: 1.35;
            color: #333;
            max-height: 90px;
            overflow-y: auto;
            border-top: 1px solid #ddd;
            padding-top: 6px;
        }}
    </style>

    <div id="budget-slider-panel">
        <div id="model-title">
            {model_title}
        </div>

        <div id="budget-slider-title">
            Upgrade budget (k): <span id="budget-slider-value">0</span>
        </div>
        <input id="budget-slider-input" type="range" min="0" max="{max_budget}" step="1" value="0" />
        <div id="selected-facility-overview">
            Selected upgrade facility IDs: none
        </div>
    </div>

    <script>
        var hospitalUpgradeState = {state_json};
        var selectedUpgradesByBudget = {selected_json};

        // Build the same hospital icon shape as the Python Folium marker,
        // but allow JavaScript to change the background color dynamically.
        function buildHospitalIconHtmlByColor(bgColor, sizePx) {{
            var crossThickness = Math.max(4, Math.floor(sizePx * 0.22));
            var crossLength = Math.max(10, Math.floor(sizePx * 0.54));
            var borderPx = Math.max(2, Math.floor(sizePx * 0.08));

            return '<div style="position:relative;width:' + sizePx + 'px;height:' + sizePx + 'px;border-radius:50%;background:' + bgColor + ';border:' + borderPx + 'px solid black;box-sizing:border-box;">' +
                '<div style="position:absolute;left:50%;top:50%;width:' + crossLength + 'px;height:' + crossThickness + 'px;background:white;transform:translate(-50%,-50%);border-radius:1px;"></div>' +
                '<div style="position:absolute;left:50%;top:50%;width:' + crossThickness + 'px;height:' + crossLength + 'px;background:white;transform:translate(-50%,-50%);border-radius:1px;"></div>' +
            '</div>';
        }}

        // Apply one budget scenario by recoloring all hospital markers.
        function applyBudgetScenario(budget) {{
            var b = String(budget);
            var selectedIds = selectedUpgradesByBudget[b] || [];
            var selectedSet = new Set(selectedIds.map(String));

            document.getElementById("budget-slider-value").textContent = b;

            var overview = document.getElementById("selected-facility-overview");
            overview.textContent = selectedIds.length
                ? "Selected upgrade facility IDs: " + selectedIds.join(", ")
                : "Selected upgrade facility IDs: none";

            Object.keys(hospitalMarkerLookup).forEach(function(hospitalId) {{
                var markerVarName = hospitalMarkerLookup[hospitalId];
                var state = hospitalUpgradeState[hospitalId] || {{existing_advanced: false}};

                var color = "#e53935";
                if (state.existing_advanced) {{
                    color = "#19a74a";
                }} else if (selectedSet.has(String(hospitalId))) {{
                    color = "#1f6feb";
                }}

                try {{
                    var markerObj = eval(markerVarName);
                    markerObj.setIcon(L.divIcon({{
                        html: buildHospitalIconHtmlByColor(color, 24),
                        iconSize: [24, 24],
                        iconAnchor: [12, 12],
                        className: "empty"
                    }}));
                }} catch (err) {{
                    console.log("Budget slider marker update failed:", hospitalId, err);
                }}
            }});
        }}

        // Initialize the slider when the map page has loaded.
        window.addEventListener("load", function() {{
            var slider = document.getElementById("budget-slider-input");
            slider.addEventListener("input", function(ev) {{
                applyBudgetScenario(parseInt(ev.target.value, 10));
            }});
            applyBudgetScenario(0);
        }});
    </script>
    """

    # Replace the JavaScript placeholder with the actual Folium marker lookup.
    # The lookup maps hospital IDs to the generated JavaScript marker variable names.
    html = html.replace("hospitalMarkerLookup", json.dumps(hospital_marker_lookup))

    # Attach the completed HTML/JavaScript block to the map.
    map_obj.get_root().html.add_child(folium.Element(html))


# -----------------------------------------------------------------------------
# MAIN MAP BUILDER
# -----------------------------------------------------------------------------

def build_population_map(
   population_path,
   hospital_path=None,
   distances_path=None,
   roads_geojson_path=None,
   stroke_facilities_path=None,
   output_path="population_map_weighted_casper.html",
   scenario="exponential",
   model="weighted_thesis",
   pareto_csv=str(PROJECT_ROOT / "Pareto" / "Weighted" / "pareto_curve_results.csv"),
   max_marker_points=1000,
   min_lat=None,
   max_lat=None,
):
   """
   Build and save the weighted-model interactive population map.

   This orchestration function calls the imported helper functions in sequence:
   load data -> create base map -> add population layers -> add hospital markers
   -> attach route data -> inject UI -> save HTML.
   """
   # Load the full population grid from a pickle file.
   population_df = load_pickle(population_path)

   # The population file must contain an ID and coordinates.
   required_pop_cols = {"ID", "lat", "lon"}
   if not required_pop_cols.issubset(population_df.columns):
       raise ValueError("population.pkl must contain 'ID', 'lat', and 'lon' columns")

   # Stop immediately if the population file is empty.
   if len(population_df) == 0:
       raise ValueError("population.pkl is empty")

   print("Population dataframe loaded.")
   print(f"  Population rows before geographic filtering: {len(population_df)}")
   print(f"  Population columns: {list(population_df.columns)}")

   # Optional latitude filter for testing or creating regional maps.
   if min_lat is not None:
       before = len(population_df)
       population_df = population_df[population_df["lat"] >= min_lat].copy()
       print(f"Applied min_lat filter: kept {len(population_df)} / {before} rows (lat >= {min_lat})")

   # Optional upper latitude filter for testing or creating regional maps.
   if max_lat is not None:
       before = len(population_df)
       population_df = population_df[population_df["lat"] <= max_lat].copy()
       print(f"Applied max_lat filter: kept {len(population_df)} / {before} rows (lat <= {max_lat})")

   # Filtering should not remove all population rows.
   if len(population_df) == 0:
       raise ValueError("No population rows remain after applying geographic filters.")

   print(f"  Population rows after geographic filtering: {len(population_df)}")

   # Center the map on the median population coordinate.
   center_lat = float(population_df["lat"].median())
   center_lon = float(population_df["lon"].median())

   # Create the base Folium map.
   m = folium.Map(
       location=[center_lat, center_lon],
       zoom_start=6,
       tiles="CartoDB positron"
   )

   # Custom panes control drawing order.
   # Hospitals should appear above population points.
   folium.map.CustomPane("hospitalPane", z_index=700).add_to(m)
   folium.map.CustomPane("populationPane", z_index=450).add_to(m)

   # The heatmap should represent all population data, not only the clickable sample.
   # Therefore, we aggregate the full population dataframe before making heat data.
   aggregated_population_df = aggregate_population_for_visualization(
       population_df=population_df,
       lat_col="lat",
       lon_col="lon",
       weight_col="household_count",
   )

   # Convert aggregated population counts into Folium HeatMap input.
   heat_data = build_weighted_heat_data(
       aggregated_df=aggregated_population_df,
       lat_col="lat",
       lon_col="lon",
       weight_col="household_count_sum",
       clip_quantile=0.995,
   )

   # Add the population heatmap layer.
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

   # Add circle markers that represent population intensity at aggregated locations.
   add_population_intensity_circles(
       map_obj=m,
       aggregated_df=aggregated_population_df,
       lat_col="lat",
       lon_col="lon",
       weight_col="household_count_sum",
   )

   # Load the weighted model's selected upgrade hospitals for each budget.
   selected_upgrades_by_budget = load_selected_upgrades_by_budget(
       pareto_csv_path=pareto_csv,
   )
   max_budget = max(selected_upgrades_by_budget.keys())

   # These dictionaries are passed to the JavaScript UI.
   # They connect hospital IDs to marker objects and metadata.
   hospital_marker_lookup = {}
   hospital_panel_data = {}
   hospital_upgrade_state = {}

   # --------------------------------------------------------------------------
   # LOAD AND ADD HOSPITAL MARKERS
   # --------------------------------------------------------------------------
   hospitals_df = None
   if hospital_path and os.path.exists(hospital_path):
       try:
           # Load hospital coordinates and metadata.
           hospitals_df = load_pickle(hospital_path)

           # Fix ambiguous pandas situation where ID exists both as index and as column.
           if isinstance(hospitals_df.index, pd.Index) and hospitals_df.index.name == "ID":
               hospitals_df = hospitals_df.reset_index(drop=True)

           # Extra safety: if ID somehow exists more than once as columns, keep one.
           hospitals_df = hospitals_df.loc[:, ~hospitals_df.columns.duplicated()].copy()

           print("Hospitals dataframe loaded.")
           print(f"  Hospital rows: {len(hospitals_df)}")
           print(f"  Hospital columns: {list(hospitals_df.columns)}")
           print(f"  Hospital index name: {hospitals_df.index.name}")

           # Stroke-facilities CSV is required because it marks advanced/specialized hospitals.
           if stroke_facilities_path is None:
               raise ValueError("stroke_facilities_path is required")

           # Load specialized-equipment data and merge onto hospitals_df.
           stroke_df = load_stroke_facilities_csv(stroke_facilities_path)

           # Clean hospital IDs before merging.
           hospitals_df["ID"] = ensure_numeric(hospitals_df["ID"])
           hospitals_df = hospitals_df.dropna(subset=["ID"]).copy()
           hospitals_df["ID"] = hospitals_df["ID"].astype(int)

           # Add specialized-equipment information to every hospital row.
           hospitals_df = hospitals_df.merge(
               stroke_df,
               on="ID",
               how="left"
           )

           # Missing specialized-equipment values are treated as False.
           hospitals_df["specialized_equipment"] = hospitals_df["specialized_equipment"].fillna(False)

           print("Merged specialized-equipment information into hospitals dataframe.")
           print(hospitals_df["specialized_equipment"].value_counts(dropna=False))

           # Add one marker per hospital if coordinates are available.
           if "Latitude" in hospitals_df.columns and "Longitude" in hospitals_df.columns:
               hospital_group = folium.FeatureGroup(name="Hospitals (individual)", show=True).add_to(m)

               for _, row in hospitals_df.iterrows():
                   hospital_id = safe_int(row.get("ID"))
                   has_specialized = bool(row.get("specialized_equipment", False))

                   # Optional English hospital name for the side panel.
                   name_text = ""
                   if "Name_English" in row.index and pd.notna(row.get("Name_English")):
                       name_text = str(row["Name_English"]).strip()

                   # Optional address for the side panel.
                   address_text = ""
                   if "Address_CSV" in row.index and pd.notna(row.get("Address_CSV")):
                       address_text = str(row["Address_CSV"]).strip()

                   # Build the initial hospital icon.
                   # It will later be recolored by the budget slider if selected.
                   hospital_icon_html = build_hospital_icon_html(
                       has_specialized=has_specialized,
                       size_px=24,
                    )

                   # Add the hospital marker to the hospital pane.
                   marker = folium.Marker(
                       location=[float(row["Latitude"]), float(row["Longitude"])],
                       icon=folium.DivIcon(
                           html=hospital_icon_html,
                           icon_size=(24, 24),
                           icon_anchor=(12, 12),
                           class_name="empty"
                        ),
                        pane="hospitalPane",
                   ).add_to(hospital_group)

                   # Store marker and metadata for the click panel and slider.
                   if hospital_id is not None:
                       hospital_marker_lookup[str(hospital_id)] = marker.get_name()
                       hospital_upgrade_state[str(hospital_id)] = {
                           "existing_advanced": has_specialized,
                        }

                       hospital_panel_data[str(hospital_id)] = {
                           "hospital_id": hospital_id,
                           "hospital_name": name_text,
                           "specialized_equipment": has_specialized,
                           "address_text": address_text,
                       }

       except Exception as e:
           print(f"Warning: could not load hospitals file: {e}")
           hospitals_df = None

   # The hospital data is required because population points need nearest-hospital assignment.
   if hospitals_df is None:
       raise ValueError("Hospitals file is required for correct nearest hospital assignment.")

   # --------------------------------------------------------------------------
   # BUILD CLICKABLE POPULATION SAMPLE
   # --------------------------------------------------------------------------

   # Always choose nearest hospital with the original coordinate-based method first.
   # Only these sampled population rows become clickable population markers.
   # This keeps the browser map responsive and prevents the HTML file from becoming huge.
   interactive_df = build_nearest_hospital_lookup(
       population_df=population_df,
       hospital_df=hospitals_df,
       sample_size=max_marker_points
   )
   print("Assigned nearest hospital IDs using ORIGINAL coordinate-based nearest-hospital logic.")

   # Attach precomputed distances only for the already chosen hospital pair.
   # This uses the saved road-distance matrix instead of recomputing all routes.
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

   # --------------------------------------------------------------------------
   # BUILD ROAD GRAPH AND ROUTE DATA FOR POPUPS
   # --------------------------------------------------------------------------
   G = None
   tree = None
   node_list = None

   # Build graph from roads GeoJSON if a road file is available.
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

   # Snap sampled population points and hospitals to the road graph.
   # This makes route drawing possible in the side panel.
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

           # Build route and travel-time data for the top-k nearest hospitals shown in the popup.
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

   # --------------------------------------------------------------------------
   # ADD CLICKABLE POPULATION MARKERS AND PREPARE SIDE-PANEL DATA
   # --------------------------------------------------------------------------
   marker_cluster = MarkerCluster(name="Population points", show=False).add_to(m)

   # Data structures passed to map_ui.py.
   route_data = {}
   population_marker_lookup = {}
   population_panel_data = {}

   for _, row in interactive_df.iterrows():
       pop_id = safe_int(row.get("ID"))

       # Household count shown in the side panel.
       household_count = (
           int(row["household_count"])
           if "household_count" in row and not pd.isna(row.get("household_count"))
           else "N/A"
       )

       # top_k_hospitals_json is created by build_top_k_hospital_popup_data.
       top_hospitals = []
       if row.get("top_k_hospitals_json"):
           try:
               top_hospitals = json.loads(row["top_k_hospitals_json"])
           except Exception:
               top_hospitals = []

       # Convert hospital route records into a compact side-panel card format.
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

       # Store population metadata for the JavaScript side panel.
       if pop_id is not None:
           population_panel_data[str(pop_id)] = {
               "population_id": pop_id,
               "household_count": household_count,
               "hospitals": hospital_cards,
           }

       # Add a clickable marker for the sampled population point.
       marker = folium.CircleMarker(
           location=[float(row["lat"]), float(row["lon"])],
           radius=6,
           color="blue",
           fill=True,
           fill_color="cyan",
           fill_opacity=0.7,
       ).add_to(marker_cluster)

       # Store the generated marker variable name for the side-panel UI.
       if pop_id is not None:
           population_marker_lookup[str(pop_id)] = marker.get_name()

       # Store route coordinates by population ID and hospital ID.
       if pop_id is not None:
           route_data[str(pop_id)] = {}

           for hosp in top_hospitals:
               hospital_id = hosp.get("hospital_id", None)
               if hospital_id is None:
                   continue

               coords = hosp.get("route_coords", None)
               route_available = bool(hosp.get("route_available", False))

               # Only save valid route lines with at least two coordinates.
               if (
                   route_available
                   and isinstance(coords, list)
                   and len(coords) >= 2
               ):
                   route_data[str(pop_id)][str(hospital_id)] = coords

   # Add the right-side click panel and route drawing JavaScript.
   inject_side_panel_and_routes(
       m,
       route_data,
       population_panel_data,
       population_marker_lookup,
       hospital_panel_data,
       hospital_marker_lookup,
   )

   # Add the weighted-model budget slider to the left side of the map.
   inject_budget_slider_control(
       map_obj=m,
       hospital_marker_lookup=hospital_marker_lookup,
       hospital_upgrade_state=hospital_upgrade_state,
       selected_by_budget=selected_upgrades_by_budget,
       )

   # Add standard Folium layer controls and save the final HTML map.
   folium.LayerControl().add_to(m)
   m.save(output_path)
   print(f"Map saved to {output_path}")


# -----------------------------------------------------------------------------
# COMMAND LINE INTERFACE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description="Create an HTML map of population points with correct nearest hospital ID, road-network distance, and clickable route drawing."
   )

   # Input population grid.
   parser.add_argument(
       "--population",
       default=str(PROJECT_ROOT / "Data_vietnam" / "population.pkl"),
       help="Path to population pickle file"
   )

   # Input hospital metadata.
   parser.add_argument(
       "--hospitals",
       default=str(PROJECT_ROOT / "Data_vietnam" / "existing_hospitals_100" / "all_hospitals.pkl"),
       help="Path to hospitals pickle file"
   )

   # Precomputed population-to-hospital road distances.
   parser.add_argument(
       "--distances",
       default=str(PROJECT_ROOT / "Data_vietnam" / "existing_hospitals_100" / "distances_osm_max_300km.pkl"),
       help="Path to precomputed road-distance pickle file"
   )

   # Road network used for optional route drawing.
   parser.add_argument(
       "--roads-geojson",
       default=str(PROJECT_ROOT / "Data_vietnam" / "road_osm_preprocessed.geojson"),
       help="Path to road network GeoJSON"
   )

   # Stroke-care capability information.
   parser.add_argument(
       "--stroke-facilities",
       default=str(PROJECT_ROOT / "Data_vietnam" / "stroke-facs-100-en.csv"),
       help="Path to CSV file with specialized-equipment information"
   )

   # Output HTML path.
   parser.add_argument(
       "--output",
       default="population_map_weighted_casper.html",
       help="Output HTML file path"
   )

   # Scenario label. This is mainly metadata for distinguishing exponential/logistic maps.
   parser.add_argument(
       "--scenario",
       default="exponential",
       help="Scenario name used for selected-facility outputs (for example: exponential, logistic)"
   )

   # Model name used to resolve the default Pareto CSV.
   parser.add_argument(
       "--model",
       default="weighted_thesis",
       choices=["weighted_thesis", "base_mclp"],
       help="Which model results to display"
   )

   # Optional manual override for the selected-upgrade CSV.
   parser.add_argument(
       "--pareto-csv",
       default=None,
       help="Override Pareto CSV path. If omitted, resolved automatically from --model."
   )

   # Limit the number of clickable population markers to keep the browser responsive.
   parser.add_argument(
       "--max-marker-points",
       type=int,
       default=1000,
       help="Maximum population markers for interactive clicks"
   )

   # Optional latitude filters for smaller test maps.
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

   # Use the default weighted Pareto CSV unless the user provided a custom path.
   if args.pareto_csv is None:
       args.pareto_csv = PARETO_PATHS["weighted_thesis"]

   # Build and save the map.
   build_population_map(
       population_path=args.population,
       hospital_path=args.hospitals,
       distances_path=args.distances,
       roads_geojson_path=args.roads_geojson,
       stroke_facilities_path=args.stroke_facilities,
       output_path=args.output,
       scenario=args.scenario,
       model=args.model,
       pareto_csv=args.pareto_csv,
       max_marker_points=args.max_marker_points,
       min_lat=args.min_lat,
       max_lat=args.max_lat,
   )
