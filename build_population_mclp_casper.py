# =============================================================================
# INDIVIDUAL ARTIFACT MAP - BASE MCLP UPGRADE VISUALIZATION
# =============================================================================
# Purpose of this file:
# This script creates the individual HTML artifact for the thesis. It starts from
# the shared population map logic, adds hospital markers adds population
# accessibility information and injects a budget slider that shows which
# hospitals are selected for upgrade by the base MCLP Pareto results.
#
# Main workflow:
# 1. Load population, hospital, road-network, distance, and stroke-facility data.
# 2. Build a Folium map with a population heatmap and hospital markers.
# 3. Load selected hospital upgrades from the Pareto CSV for each budget level.
# 4. Add a JavaScript slider that recolors hospital markers by budget.
# 5. Add clickable population markers and side-panel route information.
# 6. Save the final interactive map as an HTML file.
#
# Color meaning in the budget slider:
# - Green hospital marker: already has advanced stroke capability.
# - Blue hospital marker: selected for upgrade at the current budget.
# - Red hospital marker: not currently advanced and not selected for upgrade.
#
# Important limitation:
# The map only creates clickable markers for a sampled subset of population
# points, controlled by --max-marker-points. The heatmap still uses the full
# population dataset. This is necessary because plotting all population points
# individually would make the HTML file too large and the browser too slow.
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
from map_ui import inject_side_panel_and_routes

# Project root is used to build default paths that work when the script is run
# from the thesis repository instead of from a fixed working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default Pareto result files.
# These CSV files contain the selected hospital upgrade IDs for each budget b.
PARETO_PATHS = {
    "base_mclp": "/Users/casperklusener/Documents/Thesis/Pareto/Base MCLP/base_pareto_results_extended.csv",
}

# -----------------------------------------------------------------------------
# Read a folder name such as "b6" and return the numeric budget 6.
# This helper is kept for compatibility with budget-folder based outputs.
# -----------------------------------------------------------------------------
def parse_budget_dir_name(path_obj: Path):
    match = re.match(r"^b(\d+)$", path_obj.name)
    return int(match.group(1)) if match else None

# -----------------------------------------------------------------------------
# Convert a selected-facility list stored in a CSV cell into a Python set.
# The Pareto CSV may store selected IDs as strings like "[1, 4, 9]".
# -----------------------------------------------------------------------------
def parse_id_set(value):
    if pd.isna(value):
        return set()
    
    # Convert the value to text so the same parsing works for strings and numbers.
    text = str(value).strip()
    if not text:
        return set()

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    out = set()
    for token in text.split(","):
        token = token.strip().strip("\"'")
        if token:
            try:
                out.add(int(token))
            except ValueError:
                pass
    return out

# -----------------------------------------------------------------------------
# Find the first matching column name in a dataframe.
# This makes the Pareto CSV loader robust to small naming differences such as
# "b" versus "budget".
# -----------------------------------------------------------------------------
def find_first_present_column(df, candidates):
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def parse_bool_series(series):
    truthy = {"1", "true", "t", "yes", "y"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


# -----------------------------------------------------------------------------
# Load selected upgrade decisions from a Pareto CSV.
# The output is a dictionary: budget b -> set of selected hospital IDs.
# -----------------------------------------------------------------------------
def load_selected_upgrades_by_budget(
    results_root=None,
    model=None,
    scenario=None,
    pareto_csv_path=None,
):
    pareto_path = Path(pareto_csv_path)

    if not pareto_path.exists():
        raise FileNotFoundError(
            f"Pareto file not found: {pareto_path}"
        )
    # sep=None lets pandas infer whether the CSV is comma- or semicolon-separated.
    pareto = pd.read_csv(pareto_path, sep=None, engine="python")

    selected = {}

    # Find the budget column and the selected-facility column.
    b_col = find_first_present_column(
        pareto,
        ["b", "budget"]
    )

    ids_col = find_first_present_column(
        pareto,
        [
            "selected_facility_ids",
        ]
    )

    if b_col is None:
        raise ValueError("Could not find budget column in Pareto file.")

    if ids_col is None:
        raise ValueError(
            "Could not find selected facility ID column in Pareto file."
        )

    pareto[b_col] = ensure_numeric(pareto[b_col])
    pareto = pareto.dropna(subset=[b_col])

    # Read selected upgrade IDs for every budget row in the Pareto CSV.
    for _, row in pareto.iterrows():
        selected[int(row[b_col])] = parse_id_set(row[ids_col])

    if 0 not in selected:
        selected[0] = set()

    max_budget = max(selected.keys())

    # Expand missing budgets by carrying forward the last available selected set.
    # This prevents the slider from breaking if a budget level is absent from the CSV.
    expanded = {}
    current = set()

    for b in range(max_budget + 1):
        if b in selected:
            current = set(selected[b])
        expanded[b] = set(current)

    print("Selected upgrades loaded from Pareto CSV:")
    for b in range(max_budget + 1):
        print(f"  b={b}: {sorted(expanded[b])}")

    return expanded

# -----------------------------------------------------------------------------
# Inject the HTML, CSS, and JavaScript for the budget slider.
# The slider does not rerun the optimization. It only reads precomputed selected
# upgrade IDs and recolors hospital markers for the chosen budget.
# -----------------------------------------------------------------------------
def inject_budget_slider_control(map_obj, hospital_marker_lookup, hospital_upgrade_state, selected_by_budget, model_title="Baseline  MCLP Model",):
    max_budget = max(selected_by_budget)

    state_json = json.dumps(hospital_upgrade_state)
    selected_json = json.dumps({str(k): sorted(v) for k, v in selected_by_budget.items()})

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

        function buildHospitalIconHtmlByColor(bgColor, sizePx) {{
            var crossThickness = Math.max(4, Math.floor(sizePx * 0.22));
            var crossLength = Math.max(10, Math.floor(sizePx * 0.54));
            var borderPx = Math.max(2, Math.floor(sizePx * 0.08));

            return '<div style="position:relative;width:' + sizePx + 'px;height:' + sizePx + 'px;border-radius:50%;background:' + bgColor + ';border:' + borderPx + 'px solid black;box-sizing:border-box;">' +
                '<div style="position:absolute;left:50%;top:50%;width:' + crossLength + 'px;height:' + crossThickness + 'px;background:white;transform:translate(-50%,-50%);border-radius:1px;"></div>' +
                '<div style="position:absolute;left:50%;top:50%;width:' + crossThickness + 'px;height:' + crossLength + 'px;background:white;transform:translate(-50%,-50%);border-radius:1px;"></div>' +
            '</div>';
        }}

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

        window.addEventListener("load", function() {{
            var slider = document.getElementById("budget-slider-input");
            slider.addEventListener("input", function(ev) {{
                applyBudgetScenario(parseInt(ev.target.value, 10));
            }});
            applyBudgetScenario(0);
        }});
    </script>
    """
    # Insert the actual Folium marker variable names into the JavaScript code.
    # These names allow JavaScript to find and recolor existing hospital markers.
    html = html.replace("hospitalMarkerLookup", json.dumps(hospital_marker_lookup))
    map_obj.get_root().html.add_child(folium.Element(html))




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
   output_path="population_map_base_mclp.html",
   scenario="MCLP",
   model="base_mclp",
   pareto_csv="/Users/casperklusener/Documents/Thesis/Pareto/Base MCLP/base_pareto_results_extended.csv",
   max_marker_points=1000,
   min_lat=None,
   max_lat=None,
):
   """
   Build the individual artifact map and save it as an HTML file.

   The function combines the shared population-accessibility map with the
   individual optimization result. The optimization result is not solved here;
   it is read from the Pareto CSV and visualized with the budget slider.
   """

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

   # Center the map on the median population coordinate.
   # This is more robust than using the mean when there are outlying coordinates.
   center_lat = float(population_df["lat"].median())
   center_lon = float(population_df["lon"].median())

   # Create the base Folium map. 
   m = folium.Map(
       location=[center_lat, center_lon],
       zoom_start=6,
       tiles="CartoDB positron"
   )

   # Create custom panes so hospital markers stay visually above population layers.
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

   # Load the optimization output that links each budget to selected upgrade IDs.
   selected_upgrades_by_budget = load_selected_upgrades_by_budget(
       pareto_csv_path=pareto_csv,
   )
   max_budget = max(selected_upgrades_by_budget.keys())


   hospital_marker_lookup = {}
   hospital_panel_data = {}
   hospital_upgrade_state = {}


   # -------------------------------------------------------------------------
   # HOSPITAL LOADING AND HOSPITAL MARKERS
   # -------------------------------------------------------------------------
   # Load hospitals, merge stroke capability information, and draw one marker
   # per hospital. These markers are later recolored by the budget slider.
   # -------------------------------------------------------------------------
   hospitals_df = None
   if hospital_path and os.path.exists(hospital_path):
       try:
           # Load the hospital pickle containing hospital IDs and coordinates.
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

           # Load the CSV that indicates whether each hospital has specialized
           # stroke intervention capability.
           stroke_df = load_stroke_facilities_csv(stroke_facilities_path)

           # Clean hospital IDs before merging with the stroke-facility CSV. 
           hospitals_df["ID"] = ensure_numeric(hospitals_df["ID"])
           hospitals_df = hospitals_df.dropna(subset=["ID"]).copy()
           hospitals_df["ID"] = hospitals_df["ID"].astype(int)

           # Merge stroke capability data onto the hospital table.
           hospitals_df = hospitals_df.merge(
               stroke_df,
               on="ID",
               how="left"
           )


           hospitals_df["specialized_equipment"] = hospitals_df["specialized_equipment"].fillna(False)


           print("Merged specialized-equipment information into hospitals dataframe.")
           print(hospitals_df["specialized_equipment"].value_counts(dropna=False))


           if "Latitude" in hospitals_df.columns and "Longitude" in hospitals_df.columns:
               # Hospital markers are stored in a separate layer so users can
               # show or hide them with the layer control.
               hospital_group = folium.FeatureGroup(name="Hospitals (individual)", show=True).add_to(m)

               # Add one marker per hospital and store its metadata for the side panel. 
               for _, row in hospitals_df.iterrows():
                   hospital_id = safe_int(row.get("ID"))
                   has_specialized = bool(row.get("specialized_equipment", False))
                   
                   name_text = ""
                   if "Name_English" in row.index and pd.notna(row.get("Name_English")):
                       name_text = str(row["Name_English"]).strip()
                       
                   address_text = ""
                   if "Address_CSV" in row.index and pd.notna(row.get("Address_CSV")):
                       address_text = str(row["Address_CSV"]).strip()

                   # Initial icon color shows current capability before slider changes.                       
                   hospital_icon_html = build_hospital_icon_html(
                       has_specialized=has_specialized,
                       size_px=24,
                    )
                   
                   # Add the hospital marker to the map.
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


   if hospitals_df is None:
       raise ValueError("Hospitals file is required for correct nearest hospital assignment.")


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


   # -------------------------------------------------------------------------
   # ROAD GRAPH AND ROUTING DATA
   # -------------------------------------------------------------------------
   # Build the road graph only for clickable route visualization. The heavy
   # optimization distances are already precomputed and loaded above.
   # -------------------------------------------------------------------------
   # Build graph from roads GeoJSON
   G = None
   tree = None
   node_list = None


   if roads_geojson_path and os.path.exists(roads_geojson_path):
       try:
           # Load road network geometry and convert it to a NetworkX graph.
           roads_geojson = load_roads_geojson(roads_geojson_path)

           # Build the graph and a KDTree for fast nearest-road-node lookup.
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

           # For each sampled population point, compute the top-k hospital options
           # used in the clickable side panel.
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

   # Population markers are placed in a marker cluster and hidden by default.
   # The heatmap is the main population layer; markers are only for inspection.
   marker_cluster = MarkerCluster(name="Population points", show=False).add_to(m)

   # These dictionaries are passed to map_ui.py.
   # route_data stores drawable route coordinates.
   # population_panel_data stores text/statistics shown after clicking a population point.
   route_data = {}
   population_marker_lookup = {}
   population_panel_data = {}

   # Build one clickable population marker and one side-panel record per sampled point.
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

       # Draw the sampled population point as a clickable circle marker. 
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

     
   inject_budget_slider_control(
       map_obj=m,
       hospital_marker_lookup=hospital_marker_lookup,
       hospital_upgrade_state=hospital_upgrade_state,
       selected_by_budget=selected_upgrades_by_budget,
       )


   folium.LayerControl().add_to(m)
   m.save(output_path)
   print(f"Map saved to {output_path}")


# -----------------------------------------------------------------------------
# COMMAND LINE INTERFACE(CLI)
# -----------------------------------------------------------------------------
# Running this file directly creates the HTML artifact.
# Example:
# python individual_artifact_map.py --output population_map_base_mclp.html
# -----------------------------------------------------------------------------
if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description="Create an HTML map of population points with correct nearest hospital ID, road-network distance, and clickable route drawing."
   )
   parser.add_argument(
       "--population",
       default=str(PROJECT_ROOT / "Data_vietnam" / "population.pkl"),
       help="Path to population pickle file"
   )
   parser.add_argument(
       "--hospitals",
       default=str(PROJECT_ROOT / "Data_vietnam" / "existing_hospitals_100" / "all_hospitals.pkl"),
       help="Path to hospitals pickle file"
   )
   parser.add_argument(
       "--distances",
       default=str(PROJECT_ROOT / "Data_vietnam" / "existing_hospitals_100" / "distances_osm_max_300km.pkl"),
       help="Path to precomputed road-distance pickle file"
   )
   parser.add_argument(
       "--roads-geojson",
       default=str(PROJECT_ROOT / "Data_vietnam" / "road_osm_preprocessed.geojson"),
       help="Path to road network GeoJSON"
   )
   parser.add_argument(
       "--stroke-facilities",
       default=str(PROJECT_ROOT / "Data_vietnam" / "stroke-facs-100-en.csv"),
       help="Path to CSV file with specialized-equipment information"
   )
   parser.add_argument(
       "--output",
       default="population_map_base_mclp.html",
       help="Output HTML file path"
   )
   parser.add_argument(
       "--scenario",
       default="MCLP",
       help="Scenario name used for selected-facility outputs"
   )
   parser.add_argument(
       "--pareto-csv",
       default=None,
       help="Override Pareto CSV path. If omitted, resolved automatically from --model."
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

   if args.pareto_csv is None:
       args.pareto_csv = PARETO_PATHS["base_mclp"]

   build_population_map(
       population_path=args.population,
       hospital_path=args.hospitals,
       distances_path=args.distances,
       roads_geojson_path=args.roads_geojson,
       stroke_facilities_path=args.stroke_facilities,
       output_path=args.output,
       scenario=args.scenario,
       model="base_mclp",
       pareto_csv=args.pareto_csv,
       max_marker_points=args.max_marker_points,
       min_lat=args.min_lat,
       max_lat=args.max_lat,
   )