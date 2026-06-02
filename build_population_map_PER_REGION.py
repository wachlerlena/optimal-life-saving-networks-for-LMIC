# =============================================================================
# BUILD POPULATION MAP - PER REGION (PROVINCE) VERSION
# =============================================================================
# Purpose:
# Builds an interactive Folium stroke map for a SINGLE consolidated province.
# All population points within the selected province are included (no sampling,
# no lat/lon filters). All hospitals nationwide are displayed.
#
# Usage:
#   1. Set SELECTED_PROVINCE below to the desired province name.
#   2. Run:  python build_population_map_PER_REGION.py
#   3. Output: stroke_map_{Province_Name}.html
#
# Valid province names (34 consolidated, 2025 reform):
#   An Giang, Bac Ninh, Ca Mau, Can Tho, Cao Bang, Da Nang, Dak Lak,
#   Dien Bien, Dong Nai, Dong Thap, Gia Lai, Ha Tinh, Hai Phong, Hanoi,
#   Ho Chi Minh City, Hue, Hung Yen, Khanh Hoa, Lai Chau, Lam Dong,
#   Lang Son, Lao Cai, Nghe An, Ninh Binh, Phu Tho, Quang Ngai,
#   Quang Ninh, Quang Tri, Son La, Tay Ninh, Thai Nguyen, Thanh Hoa,
#   Tuyen Quang, Vinh Long
# =============================================================================

import json
import os
import sys

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point

# Reuse utilities from the main map builder.
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
    haversine_km,
    load_pickle,
    load_roads_geojson,
    load_stroke_facilities_csv,
    normalize_specialized_equipment,
    safe_int,
    snap_points_to_graph_nodes,
)
from map_ui import inject_side_panel_and_routes

# Province/region mapping from analysis.py.
from analysis import (
    GADM_PATH,
    GADM_TO_CONSOLIDATED,
    PROVINCE_TO_REGION,
    assign_admin_regions,
)

# =============================================================================
# >>> SET THE PROVINCE TO BUILD THE MAP FOR HERE <<<
# =============================================================================
SELECTED_PROVINCE = "Ha Tinh"
# =============================================================================

# Valid province names for validation.
VALID_PROVINCES = sorted(PROVINCE_TO_REGION.keys())

# ---------------------------------------------------------------------------
# Data paths (same defaults as build_population_map.py)
# ---------------------------------------------------------------------------
POPULATION_PATH = "Data_vietnam/population.pkl"
HOSPITAL_PATH = "Data_vietnam/existing_hospitals_100/all_hospitals.pkl"
DISTANCES_PATH = "Data_vietnam/existing_hospitals_100/distances_osm_max_300km.pkl"
ROADS_PATH = "Data_vietnam/road_osm_preprocessed.geojson"
STROKE_CSV_PATH = "Data_vietnam/stroke-facs-100-en.csv"


def build_province_map(selected_province):
    """
    Build an interactive Folium stroke map for a single consolidated province.
    All population points in the province are interactive (clickable with routes).
    All hospitals nationwide are displayed.
    """

    # ------------------------------------------------------------------
    # Validate province name
    # ------------------------------------------------------------------
    if selected_province not in PROVINCE_TO_REGION:
        print(f"ERROR: '{selected_province}' is not a valid province name.")
        print(f"Valid provinces ({len(VALID_PROVINCES)}):")
        for p in VALID_PROVINCES:
            print(f"  - {p}")
        sys.exit(1)

    province_safe = selected_province.replace(" ", "_")
    output_path = f"stroke_map_{province_safe}.html"

    print("=" * 70)
    print(f"Building stroke map for province: {selected_province}")
    print(f"Output file: {output_path}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load population data
    # ------------------------------------------------------------------
    print("\n[1/10] Loading population data...")
    population_df = load_pickle(POPULATION_PATH)

    required_pop_cols = {"ID", "lat", "lon"}
    if not required_pop_cols.issubset(population_df.columns):
        raise ValueError("population.pkl must contain 'ID', 'lat', and 'lon' columns")

    print(f"  Total population rows: {len(population_df)}")

    # ------------------------------------------------------------------
    # Step 2: Assign provinces via GADM spatial join
    # ------------------------------------------------------------------
    print("\n[2/10] Assigning provinces via GADM spatial join...")
    admin_gdf = gpd.read_file(GADM_PATH, layer="ADM_ADM_1")
    population_df = assign_admin_regions(population_df, "lat", "lon", admin_gdf)
    print(f"  Province assignment complete. Unique provinces: {population_df['province'].nunique()}")

    # ------------------------------------------------------------------
    # Step 3: Filter to selected province (no lat/lon filters, no sampling)
    # ------------------------------------------------------------------
    print(f"\n[3/10] Filtering population to province: {selected_province}")
    province_df = population_df[population_df["province"] == selected_province].copy()
    province_df = province_df.reset_index(drop=True)

    if len(province_df) == 0:
        raise ValueError(
            f"No population points found for province '{selected_province}'. "
            f"Check that the province name matches one of: {VALID_PROVINCES}"
        )

    print(f"  Population points in {selected_province}: {len(province_df)}")

    # ------------------------------------------------------------------
    # Step 4: Create base Folium map (auto-centered on province)
    # ------------------------------------------------------------------
    print("\n[4/10] Creating base map...")
    center_lat = float(province_df["lat"].median())
    center_lon = float(province_df["lon"].median())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="CartoDB positron",
    )

    # Fit map bounds to the province extent with some padding.
    sw = [float(province_df["lat"].min()), float(province_df["lon"].min())]
    ne = [float(province_df["lat"].max()), float(province_df["lon"].max())]
    m.fit_bounds([sw, ne], padding=[20, 20])

    # ------------------------------------------------------------------
    # Step 5: Build heatmap from all province population points
    # ------------------------------------------------------------------
    print("\n[5/10] Building heatmap...")
    aggregated_population_df = aggregate_population_for_visualization(
        population_df=province_df,
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

    # ------------------------------------------------------------------
    # Step 6: Load ALL hospitals (nationwide, not filtered)
    # ------------------------------------------------------------------
    print("\n[6/10] Loading hospitals (all nationwide)...")
    hospitals_df = load_pickle(HOSPITAL_PATH)

    if isinstance(hospitals_df.index, pd.Index) and hospitals_df.index.name == "ID":
        hospitals_df = hospitals_df.reset_index(drop=True)
    hospitals_df = hospitals_df.loc[:, ~hospitals_df.columns.duplicated()].copy()

    stroke_df = load_stroke_facilities_csv(STROKE_CSV_PATH)
    hospitals_df["ID"] = ensure_numeric(hospitals_df["ID"])
    hospitals_df = hospitals_df.dropna(subset=["ID"]).copy()
    hospitals_df["ID"] = hospitals_df["ID"].astype(int)
    hospitals_df = hospitals_df.merge(stroke_df, on="ID", how="left")
    hospitals_df["specialized_equipment"] = hospitals_df["specialized_equipment"].fillna(False)

    print(f"  Total hospitals: {len(hospitals_df)}")
    print(hospitals_df["specialized_equipment"].value_counts(dropna=False))

    # Add hospital markers to map
    hospital_marker_lookup = {}
    hospital_panel_data = {}
    cluster = MarkerCluster(name="Hospitals").add_to(m)

    for _, row in hospitals_df.iterrows():
        hospital_id = safe_int(row.get("ID"))
        has_specialized = bool(row.get("specialized_equipment", False))

        name_text = ""
        if "Name_English" in row.index and pd.notna(row.get("Name_English")):
            name_text = str(row["Name_English"]).strip()

        address_text = ""
        if "Address_CSV" in row.index and pd.notna(row.get("Address_CSV")):
            address_text = str(row["Address_CSV"]).strip()

        hospital_icon_html = build_hospital_icon_html(
            has_specialized=has_specialized, size_px=24
        )

        marker = folium.Marker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            icon=folium.DivIcon(
                html=hospital_icon_html,
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                class_name="empty",
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

    # ------------------------------------------------------------------
    # Step 7: Build interactive markers for ALL province points (no sampling)
    # ------------------------------------------------------------------
    print(f"\n[7/10] Building nearest-hospital lookup for all {len(province_df)} points (no sampling)...")
    interactive_df = build_nearest_hospital_lookup(
        population_df=province_df,
        hospital_df=hospitals_df,
        sample_size=len(province_df),  # No sampling: use all points
    )
    print(f"  Interactive points: {len(interactive_df)}")

    # Attach precomputed distances
    if os.path.exists(DISTANCES_PATH):
        try:
            distances_df = load_pickle(DISTANCES_PATH)
            interactive_df = attach_precomputed_distance_for_selected_hospital(
                interactive_df=interactive_df,
                distances_df=distances_df,
            )
        except Exception as e:
            print(f"  Warning: could not attach precomputed distances: {e}")
            interactive_df["distance_to_hospital_km"] = np.nan
            interactive_df["distance_source"] = "missing"
    else:
        interactive_df["distance_to_hospital_km"] = np.nan
        interactive_df["distance_source"] = "missing"

    # ------------------------------------------------------------------
    # Step 8: Build road network graph
    # ------------------------------------------------------------------
    print("\n[8/10] Building road network graph...")
    G = None
    tree = None
    node_list = None

    if os.path.exists(ROADS_PATH):
        roads_geojson = load_roads_geojson(ROADS_PATH)
        G = build_graph_from_roads_geojson(roads_geojson)
        tree, node_list, _ = build_graph_kdtree(G)
        print("  KDTree built for road graph nodes.")
    else:
        print("  No road GeoJSON found; routing will be skipped.")

    # ------------------------------------------------------------------
    # Step 9: Snap points to graph and calculate routes
    # ------------------------------------------------------------------
    if G is not None and tree is not None:
        print(f"\n[9/10] Snapping {len(interactive_df)} points and calculating routes...")
        interactive_df = snap_points_to_graph_nodes(
            interactive_df,
            lat_col="lat",
            lon_col="lon",
            tree=tree,
            node_list=node_list,
            label="population",
        )

        hospital_snap_df = snap_points_to_graph_nodes(
            hospitals_df.copy(),
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
    else:
        print("\n[9/10] Skipping graph snapping/routing (no road graph).")

    # ------------------------------------------------------------------
    # Step 10: Add population markers and inject UI
    # ------------------------------------------------------------------
    print(f"\n[10/10] Adding {len(interactive_df)} population markers and injecting UI...")
    marker_cluster = MarkerCluster(name="Population points").add_to(m)

    route_data = {}
    population_marker_lookup = {}
    population_panel_data = {}

    for row_idx, (_, row) in enumerate(interactive_df.iterrows()):
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
            hospital_cards.append({
                "rank": hosp.get("rank", None),
                "hospital_id": hosp.get("hospital_id", "N/A"),
                "specialized_equipment": bool(hosp.get("specialized_equipment", False)),
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
            route_data[str(pop_id)] = {}

            for hosp in top_hospitals:
                hospital_id = hosp.get("hospital_id", None)
                if hospital_id is None:
                    continue
                coords = hosp.get("route_coords", None)
                route_available = bool(hosp.get("route_available", False))
                if route_available and isinstance(coords, list) and len(coords) >= 2:
                    route_data[str(pop_id)][str(hospital_id)] = coords

        if (row_idx + 1) % 500 == 0:
            print(f"  Added {row_idx + 1} / {len(interactive_df)} markers...")

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
    print(f"\nMap saved to {output_path}")
    print(f"Province: {selected_province}")
    print(f"Population points: {len(interactive_df)}")
    print(f"Hospitals (nationwide): {len(hospitals_df)}")


if __name__ == "__main__":
    build_province_map(SELECTED_PROVINCE)
