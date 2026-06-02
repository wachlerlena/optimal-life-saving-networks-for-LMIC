# =============================================================================
# ANALYSIS.PY - Data Understanding and Analytics Metrics
# =============================================================================
# Produces:
#   1. Spatial & distributional analysis (geographic spread, anomalies)
#   2. Travel-time analysis (road-network based, total and by region)
#   3. Golden-hour analysis for advanced stroke treatment (60 min threshold)
#
# Uses the same road graph and speed model as build_population_map.py.
# Outputs are printed to the console and saved as PNG figures.
# =============================================================================

import os
import pickle
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

# Reuse graph/routing utilities from the map builder.
from build_population_map import (
    build_graph_from_roads_geojson,
    build_graph_kdtree,
    ensure_numeric,
    haversine_km,
    load_pickle,
    load_roads_geojson,
    load_stroke_facilities_csv,
    normalize_specialized_equipment,
    snap_points_to_graph_nodes,
)

try:
    import networkx as nx
except ImportError as e:
    raise ImportError("networkx is required: pip install networkx") from e

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Paths (same defaults as build_population_map.py)
# ---------------------------------------------------------------------------
POPULATION_PATH = "Data_vietnam/population.pkl"
HOSPITAL_PATH = "Data_vietnam/existing_hospitals_100/all_hospitals.pkl"
DISTANCES_PATH = "Data_vietnam/existing_hospitals_100/distances_osm_max_300km.pkl"
ROADS_PATH = "Data_vietnam/road_osm_preprocessed.geojson"
STROKE_CSV_PATH = "Data_vietnam/stroke-facs-100-en.csv"
GADM_PATH = "GADM_administrative boundaries.gpkg"
OUTPUT_DIR = "analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Province consolidation (2025 reform: 63 → 34 provinces/cities).
# Maps GADM ADM_ADM_1 NAME_1 values to the new consolidated province names.
# ---------------------------------------------------------------------------
GADM_TO_CONSOLIDATED = {
    # Unchanged
    "Hà Nội": "Hanoi",
    "Thừa Thiên Huế": "Hue",
    "Lai Châu": "Lai Chau",
    "Điện Biên": "Dien Bien",
    "Sơn La": "Son La",
    "Lạng Sơn": "Lang Son",
    "Quảng Ninh": "Quang Ninh",
    "Thanh Hóa": "Thanh Hoa",
    "Nghệ An": "Nghe An",
    "Hà Tĩnh": "Ha Tinh",
    "Cao Bằng": "Cao Bang",
    # Merged
    "Tuyên Quang": "Tuyen Quang",
    "Hà Giang": "Tuyen Quang",
    "Lào Cai": "Lao Cai",
    "Yên Bái": "Lao Cai",
    "Thái Nguyên": "Thai Nguyen",
    "Bắc Kạn": "Thai Nguyen",
    "Phú Thọ": "Phu Tho",
    "Vĩnh Phúc": "Phu Tho",
    "Hoà Bình": "Phu Tho",
    "Bắc Ninh": "Bac Ninh",
    "Bắc Giang": "Bac Ninh",
    "Hưng Yên": "Hung Yen",
    "Thái Bình": "Hung Yen",
    "Hải Phòng": "Hai Phong",
    "Hải Dương": "Hai Phong",
    "Ninh Bình": "Ninh Binh",
    "Nam Định": "Ninh Binh",
    "Hà Nam": "Ninh Binh",
    "Quảng Bình": "Quang Tri",
    "Quảng Trị": "Quang Tri",
    "Đà Nẵng": "Da Nang",
    "Quảng Nam": "Da Nang",
    "Quảng Ngãi": "Quang Ngai",
    "Kon Tum": "Quang Ngai",
    "Gia Lai": "Gia Lai",
    "Bình Định": "Gia Lai",
    "Khánh Hòa": "Khanh Hoa",
    "Ninh Thuận": "Khanh Hoa",
    "Lâm Đồng": "Lam Dong",
    "Đắk Nông": "Lam Dong",
    "Bình Thuận": "Lam Dong",
    "Đắk Lắk": "Dak Lak",
    "Phú Yên": "Dak Lak",
    "Hồ Chí Minh": "Ho Chi Minh City",
    "Bình Dương": "Ho Chi Minh City",
    "Bà Rịa - Vũng Tàu": "Ho Chi Minh City",
    "Đồng Nai": "Dong Nai",
    "Bình Phước": "Dong Nai",
    "Tây Ninh": "Tay Ninh",
    "Long An": "Tay Ninh",
    "Cần Thơ": "Can Tho",
    "Sóc Trăng": "Can Tho",
    "Hậu Giang": "Can Tho",
    "Bến Tre": "Vinh Long",
    "Vĩnh Long": "Vinh Long",
    "Trà Vinh": "Vinh Long",
    "Đồng Tháp": "Dong Thap",
    "Tiền Giang": "Dong Thap",
    "Cà Mau": "Ca Mau",
    "Bạc Liêu": "Ca Mau",
    "An Giang": "An Giang",
    "Kiên Giang": "An Giang",
}

# Region mapping for the 34 consolidated provinces.
PROVINCE_TO_REGION = {
    "Hanoi": "Red River Delta",
    "Hue": "North Central Coast",
    "Lai Chau": "Northwest",
    "Dien Bien": "Northwest",
    "Son La": "Northwest",
    "Lao Cai": "Northwest",
    "Lang Son": "Northeast",
    "Quang Ninh": "Northeast",
    "Cao Bang": "Northeast",
    "Tuyen Quang": "Northeast",
    "Thai Nguyen": "Northeast",
    "Phu Tho": "Northeast",
    "Bac Ninh": "Red River Delta",
    "Hung Yen": "Red River Delta",
    "Hai Phong": "Red River Delta",
    "Ninh Binh": "Red River Delta",
    "Thanh Hoa": "North Central Coast",
    "Nghe An": "North Central Coast",
    "Ha Tinh": "North Central Coast",
    "Quang Tri": "North Central Coast",
    "Da Nang": "South Central Coast",
    "Quang Ngai": "South Central Coast",
    "Gia Lai": "Central Highlands",
    "Khanh Hoa": "South Central Coast",
    "Lam Dong": "Central Highlands",
    "Dak Lak": "Central Highlands",
    "Ho Chi Minh City": "Southeast",
    "Dong Nai": "Southeast",
    "Tay Ninh": "Southeast",
    "Can Tho": "Mekong River Delta",
    "Vinh Long": "Mekong River Delta",
    "Dong Thap": "Mekong River Delta",
    "Ca Mau": "Mekong River Delta",
    "An Giang": "Mekong River Delta",
}


def assign_admin_regions(df, lat_col, lon_col, admin_gdf):
    """
    Spatial-join a DataFrame with lat/lon columns against GADM province polygons.
    Maps the 63 GADM provinces to the 34 consolidated provinces (2025 reform),
    then assigns the 8 socioeconomic regions.
    Returns the DataFrame with 'province' and 'region' columns added.
    """
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Ensure CRS match.
    if admin_gdf.crs != points_gdf.crs:
        admin_gdf = admin_gdf.to_crs(points_gdf.crs)

    joined = gpd.sjoin(points_gdf, admin_gdf[["NAME_1", "geometry"]], how="left", predicate="within")

    # Some points may fall on boundaries/water and not match — use nearest for those.
    # Project to UTM 48N (Vietnam) for accurate nearest-neighbor distance calculation.
    unmatched = joined["NAME_1"].isna()
    if unmatched.any():
        utm_crs = "EPSG:32648"
        unmatched_points = points_gdf.loc[unmatched.values].to_crs(utm_crs)
        admin_utm = admin_gdf[["NAME_1", "geometry"]].to_crs(utm_crs)
        nearest = gpd.sjoin_nearest(unmatched_points, admin_utm, how="left")
        # sjoin_nearest may produce duplicates; keep first match per original index
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched, "NAME_1"] = nearest["NAME_1"].values

    result = df.copy()
    # Two-step mapping: GADM name → consolidated province → region.
    result["province"] = joined["NAME_1"].map(GADM_TO_CONSOLIDATED).values
    result["region"] = result["province"].map(PROVINCE_TO_REGION).fillna("Unknown")
    return result


if __name__ == "__main__":
    # =====================================================================
    # 1. LOAD DATA
    # =====================================================================
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    population_df = load_pickle(POPULATION_PATH)
    hospitals_df = load_pickle(HOSPITAL_PATH)
    if isinstance(hospitals_df.index, pd.Index) and hospitals_df.index.name == "ID":
        hospitals_df = hospitals_df.reset_index(drop=True)

    stroke_df = load_stroke_facilities_csv(STROKE_CSV_PATH)
    hospitals_df["ID"] = ensure_numeric(hospitals_df["ID"]).astype(int)
    hospitals_df = hospitals_df.merge(stroke_df, on="ID", how="left")
    hospitals_df["specialized_equipment"] = hospitals_df["specialized_equipment"].fillna(False)

    print(f"Population points: {len(population_df):,}")
    print(f"Hospitals: {len(hospitals_df)}")
    print(f"  Advanced (thrombectomy): {hospitals_df['specialized_equipment'].sum()}")
    print(f"  Basic (thrombolysis only): {(~hospitals_df['specialized_equipment']).sum()}")

    # Load GADM administrative boundaries (province level).
    print(f"\nLoading administrative boundaries from: {GADM_PATH}")
    admin_gdf = gpd.read_file(GADM_PATH, layer="ADM_ADM_1")
    print(f"  Provinces loaded: {len(admin_gdf)}")

    # Assign province and region via spatial join.
    print("Assigning provinces to population points (spatial join)...")
    population_df = assign_admin_regions(population_df, "lat", "lon", admin_gdf)
    print(f"  Provinces assigned: {population_df['province'].notna().sum():,} / {len(population_df):,}")
    unmatched_pop = population_df["region"] == "Unknown"
    if unmatched_pop.any():
        print(f"  Points with unknown region: {unmatched_pop.sum()}")

    print("Assigning provinces to hospitals (spatial join)...")
    hospitals_df = assign_admin_regions(hospitals_df, "Latitude", "Longitude", admin_gdf)
    print(f"  Provinces assigned: {hospitals_df['province'].notna().sum()} / {len(hospitals_df)}")

    # =====================================================================
    # 2. SPATIAL & DISTRIBUTIONAL ANALYSIS
    # =====================================================================
    print("\n" + "=" * 70)
    print("SPATIAL & DISTRIBUTIONAL ANALYSIS")
    print("=" * 70)

    # --- 2a. Population distribution by region ---
    print("\n--- Population distribution by region ---")
    region_pop = (
        population_df.groupby("region")["household_count"]
        .agg(["count", "sum"])
        .rename(columns={"count": "grid_cells", "sum": "total_households"})
        .sort_values("total_households", ascending=False)
    )
    region_pop["pct_households"] = (
        100.0 * region_pop["total_households"] / region_pop["total_households"].sum()
    )
    print(region_pop.to_string())

    print("\n--- Population distribution by province ---")
    province_pop = (
        population_df.groupby(["region", "province"])["household_count"]
        .agg(["count", "sum"])
        .rename(columns={"count": "grid_cells", "sum": "total_households"})
        .sort_values("total_households", ascending=False)
    )
    province_pop["pct_households"] = (
        100.0 * province_pop["total_households"] / province_pop["total_households"].sum()
    )
    print(province_pop.to_string())

    # --- 2b. Hospital distribution by region ---
    print("\n--- Hospital distribution by region ---")
    hosp_region = hospitals_df.groupby("region").agg(
        total=("ID", "count"),
        advanced=("specialized_equipment", "sum"),
    ).sort_values("total", ascending=False)
    hosp_region["basic"] = hosp_region["total"] - hosp_region["advanced"]
    print(hosp_region.to_string())

    print("\n--- Hospital distribution by province ---")
    hosp_province = hospitals_df.groupby(["region", "province"]).agg(
        total=("ID", "count"),
        advanced=("specialized_equipment", "sum"),
    ).sort_values("total", ascending=False)
    hosp_province["basic"] = hosp_province["total"] - hosp_province["advanced"]
    print(hosp_province.to_string())

    # --- 2c. Geographic spread figure (with basemap via contextily) ---
    import contextily as ctx

    fig, axes = plt.subplots(1, 2, figsize=(12, 14))

    # Convert population sample to GeoDataFrame in EPSG:3857 for contextily.
    sample_pop = population_df.sample(min(50000, len(population_df)), random_state=0)
    gdf_pop = gpd.GeoDataFrame(
        sample_pop,
        geometry=gpd.points_from_xy(sample_pop["lon"], sample_pop["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Panel 1: Population density heatmap on basemap.
    sc = axes[0].scatter(
        gdf_pop.geometry.x, gdf_pop.geometry.y,
        c=np.log1p(sample_pop["household_count"].values),
        s=0.4, alpha=0.5, cmap="YlOrRd", zorder=2,
    )
    axes[0].set_title("Population density (log household count)", fontsize=11)
    axes[0].set_axis_off()
    ctx.add_basemap(axes[0], source=ctx.providers.CartoDB.Positron, zoom=7)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("bottom", size="3%", pad=0.15)
    plt.colorbar(sc, cax=cax, label="log(1 + household_count)", orientation="horizontal")

    # Panel 2: Hospital locations on basemap.
    gdf_hosp = gpd.GeoDataFrame(
        hospitals_df,
        geometry=gpd.points_from_xy(hospitals_df["Longitude"], hospitals_df["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    advanced_mask = hospitals_df["specialized_equipment"] == True
    axes[1].scatter(
        gdf_pop.geometry.x, gdf_pop.geometry.y,
        s=0.1, alpha=0.1, c="gray", zorder=1,
    )
    axes[1].scatter(
        gdf_hosp.loc[~advanced_mask].geometry.x,
        gdf_hosp.loc[~advanced_mask].geometry.y,
        s=70, c="red", marker="+", linewidths=1.8, zorder=3,
        label=f"Basic ({(~advanced_mask).sum()})",
    )
    axes[1].scatter(
        gdf_hosp.loc[advanced_mask].geometry.x,
        gdf_hosp.loc[advanced_mask].geometry.y,
        s=70, c="#00cc44", marker="+", linewidths=1.8, zorder=3,
        label=f"Advanced ({advanced_mask.sum()})",
    )
    axes[1].set_title("Hospital locations by type", fontsize=11)
    axes[1].set_axis_off()
    ctx.add_basemap(axes[1], source=ctx.providers.CartoDB.Positron, zoom=7)
    axes[1].legend(loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "spatial_overview.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR}/spatial_overview.png")

    # --- 2d. Anomaly / outlier detection ---
    print("\n--- Anomaly & outlier detection ---")

    # Coordinate bounds for Vietnam.
    VN_LAT_MIN, VN_LAT_MAX = 8.0, 24.0
    VN_LON_MIN, VN_LON_MAX = 102.0, 110.0

    out_of_bounds_pop = population_df[
        (population_df["lat"] < VN_LAT_MIN)
        | (population_df["lat"] > VN_LAT_MAX)
        | (population_df["lon"] < VN_LON_MIN)
        | (population_df["lon"] > VN_LON_MAX)
    ]
    print(f"Population points outside Vietnam bounds: {len(out_of_bounds_pop)}")

    out_of_bounds_hosp = hospitals_df[
        (hospitals_df["Latitude"] < VN_LAT_MIN)
        | (hospitals_df["Latitude"] > VN_LAT_MAX)
        | (hospitals_df["Longitude"] < VN_LON_MIN)
        | (hospitals_df["Longitude"] > VN_LON_MAX)
    ]
    print(f"Hospitals outside Vietnam bounds: {len(out_of_bounds_hosp)}")
    if len(out_of_bounds_hosp) > 0:
        print(out_of_bounds_hosp[["ID", "Latitude", "Longitude"]].to_string())

    # Extreme household density spikes (>99.9th percentile).
    p999 = population_df["household_count"].quantile(0.999)
    density_outliers = population_df[population_df["household_count"] > p999]
    print(f"\nHousehold count > 99.9th percentile ({p999:.0f}):")
    print(f"  {len(density_outliers)} points ({100*len(density_outliers)/len(population_df):.2f}%)")
    print(f"  Max household count: {population_df['household_count'].max():.0f}")

    # Isolated hospitals (> 100 km from any other hospital).
    hosp_coords = hospitals_df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    isolated = []
    for i, row in hospitals_df.iterrows():
        dists = haversine_km(
            row["Latitude"], row["Longitude"],
            hosp_coords[:, 0], hosp_coords[:, 1],
        )
        dists_sorted = np.sort(dists)
        nearest_other = dists_sorted[1] if len(dists_sorted) > 1 else np.inf
        if nearest_other > 100:
            isolated.append((int(row["ID"]), row.get("Name_English", ""), nearest_other))

    print(f"\nIsolated hospitals (>100 km from nearest other hospital): {len(isolated)}")
    for hid, name, d in isolated:
        print(f"  Hospital {hid}: {name} — {d:.1f} km to nearest peer")

    # =====================================================================
    # 3. ROAD-NETWORK TRAVEL TIME ANALYSIS
    # =====================================================================
    print("\n" + "=" * 70)
    print("ROAD-NETWORK TRAVEL TIME ANALYSIS")
    print("=" * 70)
    print("Building road graph (this may take a minute)...")

    roads_geojson = load_roads_geojson(ROADS_PATH)
    G = build_graph_from_roads_geojson(roads_geojson)
    tree, node_list, _ = build_graph_kdtree(G)

    # Snap all hospitals to graph.
    hospital_snap = snap_points_to_graph_nodes(
        hospitals_df, lat_col="Latitude", lon_col="Longitude",
        tree=tree, node_list=node_list, label="hospital",
    )

    # --- Run single-source Dijkstra from every hospital ---
    print(f"\nRunning Dijkstra from each of {len(hospital_snap)} hospitals...")
    # hosp_times[hosp_id] = {graph_node: travel_time_min}
    hosp_times = {}
    hosp_is_advanced = {}
    for _, hrow in hospital_snap.iterrows():
        hid = int(hrow["ID"])
        hnode = hrow["hospital_graph_node"]
        if hnode not in G:
            continue
        time_dict = nx.single_source_dijkstra_path_length(
            G, hnode, weight="travel_time_min"
        )
        hosp_times[hid] = time_dict
        hosp_is_advanced[hid] = bool(hrow.get("specialized_equipment", False))

    print(f"  Dijkstra completed for {len(hosp_times)} hospitals.")

    # Snap ALL population points to graph (needed for accurate travel times).
    print("Snapping all population points to road graph...")
    pop_snapped = snap_points_to_graph_nodes(
        population_df, lat_col="lat", lon_col="lon",
        tree=tree, node_list=node_list, label="population",
    )

    # For each population point find:
    #   - travel time to nearest ANY hospital
    #   - travel time to nearest ADVANCED hospital
    print("Computing travel times for all population points...")

    nearest_any_time = np.full(len(pop_snapped), np.inf)
    nearest_any_id = np.full(len(pop_snapped), -1, dtype=int)
    nearest_adv_time = np.full(len(pop_snapped), np.inf)
    nearest_adv_id = np.full(len(pop_snapped), -1, dtype=int)

    pop_nodes = pop_snapped["population_graph_node"].values

    for hid, time_dict in hosp_times.items():
        is_adv = hosp_is_advanced[hid]
        for i, pnode in enumerate(pop_nodes):
            t = time_dict.get(pnode)
            if t is None:
                continue
            if t < nearest_any_time[i]:
                nearest_any_time[i] = t
                nearest_any_id[i] = hid
            if is_adv and t < nearest_adv_time[i]:
                nearest_adv_time[i] = t
                nearest_adv_id[i] = hid

    pop_snapped["travel_time_any_min"] = nearest_any_time
    pop_snapped["nearest_hospital_id"] = nearest_any_id
    pop_snapped["travel_time_advanced_min"] = nearest_adv_time
    pop_snapped["nearest_advanced_id"] = nearest_adv_id

    # Replace inf with NaN for display.
    pop_snapped.loc[pop_snapped["travel_time_any_min"] == np.inf, "travel_time_any_min"] = np.nan
    pop_snapped.loc[pop_snapped["travel_time_advanced_min"] == np.inf, "travel_time_advanced_min"] = np.nan

    reachable_any = pop_snapped["travel_time_any_min"].notna()
    reachable_adv = pop_snapped["travel_time_advanced_min"].notna()

    # --- 3a. Overall average travel times ---
    print("\n--- Overall travel-time statistics (road-network, minutes) ---")
    print(f"Reachable by any hospital: {reachable_any.sum():,} / {len(pop_snapped):,} "
          f"({100*reachable_any.mean():.1f}%)")
    print(f"Reachable by advanced hospital: {reachable_adv.sum():,} / {len(pop_snapped):,} "
          f"({100*reachable_adv.mean():.1f}%)")
    print()

    for label, col in [("Nearest ANY hospital", "travel_time_any_min"),
                        ("Nearest ADVANCED hospital", "travel_time_advanced_min")]:
        vals = pop_snapped[col].dropna()
        if len(vals) == 0:
            print(f"{label}: No reachable points")
            continue
        print(f"{label}:")
        print(f"  Mean:   {vals.mean():.1f} min")
        print(f"  Median: {vals.median():.1f} min")
        print(f"  Std:    {vals.std():.1f} min")
        print(f"  Min:    {vals.min():.1f} min")
        print(f"  Max:    {vals.max():.1f} min")
        print(f"  P25:    {vals.quantile(0.25):.1f} min")
        print(f"  P75:    {vals.quantile(0.75):.1f} min")
        print(f"  P90:    {vals.quantile(0.90):.1f} min")
        print(f"  P95:    {vals.quantile(0.95):.1f} min")
        print()

    # --- 3b. Weighted average (by household count) ---
    print("--- Household-weighted travel times ---")
    hh = pop_snapped["household_count"].fillna(0)
    for label, col in [("Nearest ANY hospital", "travel_time_any_min"),
                        ("Nearest ADVANCED hospital", "travel_time_advanced_min")]:
        valid = pop_snapped[col].notna()
        w = hh[valid]
        v = pop_snapped.loc[valid, col]
        if w.sum() > 0:
            weighted_mean = (v * w).sum() / w.sum()
            print(f"{label}: weighted mean = {weighted_mean:.1f} min")
        else:
            print(f"{label}: no data")
    print()

    # --- 3c. Travel time by region ---
    print("--- Average travel time by region ---")
    region_stats = []
    for region in sorted(pop_snapped["region"].unique()):
        mask = pop_snapped["region"] == region
        sub = pop_snapped[mask]
        n_points = len(sub)
        n_households = sub["household_count"].sum()

        any_vals = sub["travel_time_any_min"].dropna()
        adv_vals = sub["travel_time_advanced_min"].dropna()

        hh_sub = sub["household_count"].fillna(0)

        any_mean = any_vals.mean() if len(any_vals) > 0 else np.nan
        adv_mean = adv_vals.mean() if len(adv_vals) > 0 else np.nan

        # Weighted means.
        any_valid = sub["travel_time_any_min"].notna()
        adv_valid = sub["travel_time_advanced_min"].notna()
        w_any = hh_sub[any_valid]
        w_adv = hh_sub[adv_valid]
        any_wmean = (sub.loc[any_valid, "travel_time_any_min"] * w_any).sum() / w_any.sum() if w_any.sum() > 0 else np.nan
        adv_wmean = (sub.loc[adv_valid, "travel_time_advanced_min"] * w_adv).sum() / w_adv.sum() if w_adv.sum() > 0 else np.nan

        region_stats.append({
            "Region": region,
            "Grid cells": n_points,
            "Households": int(n_households),
            "Any hosp (mean min)": round(any_mean, 1) if not np.isnan(any_mean) else None,
            "Any hosp (weighted min)": round(any_wmean, 1) if not np.isnan(any_wmean) else None,
            "Advanced (mean min)": round(adv_mean, 1) if not np.isnan(adv_mean) else None,
            "Advanced (weighted min)": round(adv_wmean, 1) if not np.isnan(adv_wmean) else None,
        })

    region_df = pd.DataFrame(region_stats)
    print(region_df.to_string(index=False))

    # --- 3c2. Travel time by province ---
    print("\n--- Average travel time by province ---")
    province_stats = []
    for province in pop_snapped["province"].dropna().unique():
        mask = pop_snapped["province"] == province
        sub = pop_snapped[mask]
        n_households = sub["household_count"].fillna(0).sum()

        any_vals = sub["travel_time_any_min"].dropna()
        adv_vals = sub["travel_time_advanced_min"].dropna()

        hh_sub = sub["household_count"].fillna(0)
        any_valid = sub["travel_time_any_min"].notna()
        adv_valid = sub["travel_time_advanced_min"].notna()
        w_any = hh_sub[any_valid]
        w_adv = hh_sub[adv_valid]
        any_wmean = (sub.loc[any_valid, "travel_time_any_min"] * w_any).sum() / w_any.sum() if w_any.sum() > 0 else np.nan
        adv_wmean = (sub.loc[adv_valid, "travel_time_advanced_min"] * w_adv).sum() / w_adv.sum() if w_adv.sum() > 0 else np.nan

        region = PROVINCE_TO_REGION.get(province, "Unknown")
        province_stats.append({
            "Province": province,
            "Region": region,
            "Households": int(n_households),
            "Any hosp (weighted min)": round(any_wmean, 1) if not np.isnan(any_wmean) else None,
            "Advanced (weighted min)": round(adv_wmean, 1) if not np.isnan(adv_wmean) else None,
        })

    province_travel_df = pd.DataFrame(province_stats).sort_values(
        "Advanced (weighted min)", ascending=False, na_position="first"
    )
    print(province_travel_df.to_string(index=False))

    # --- 3d. Travel time distribution figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    any_vals = pop_snapped["travel_time_any_min"].dropna()
    adv_vals = pop_snapped["travel_time_advanced_min"].dropna()

    if len(any_vals) > 0:
        axes[0].hist(any_vals.clip(upper=300), bins=60, color="#21918c", edgecolor="white", alpha=0.8)
        axes[0].axvline(60, color="#222222", linestyle="--", linewidth=1.5, label="60 min (golden hour)")
        axes[0].set_title("Travel time to nearest ANY hospital")
        axes[0].set_xlabel("Minutes")
        axes[0].set_ylabel("Population points")
        axes[0].legend()

    if len(adv_vals) > 0:
        axes[1].hist(adv_vals.clip(upper=300), bins=60, color="#440154", edgecolor="white", alpha=0.8)
        axes[1].axvline(60, color="#222222", linestyle="--", linewidth=1.5, label="60 min (golden hour)")
        axes[1].set_title("Travel time to nearest ADVANCED hospital")
        axes[1].set_xlabel("Minutes")
        axes[1].set_ylabel("Population points")
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "travel_time_distributions.png"), dpi=150)
    print(f"\nSaved: {OUTPUT_DIR}/travel_time_distributions.png")

    # --- 3e. Regional bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(region_df))
    width = 0.35
    any_means = region_df["Any hosp (weighted min)"].fillna(0).values
    adv_means = region_df["Advanced (weighted min)"].fillna(0).values

    ax.bar(x - width / 2, any_means, width, label="Any hospital", color="#21918c")
    ax.bar(x + width / 2, adv_means, width, label="Advanced hospital", color="#440154")
    ax.axhline(60, color="#222222", linestyle="--", linewidth=1.2, label="Golden hour (60 min)")
    ax.set_xticks(x)
    ax.set_xticklabels(region_df["Region"].values, rotation=35, ha="right")
    ax.set_ylabel("Household-weighted mean travel time (min)")
    ax.set_title("Mean travel time by region (road-network)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "travel_time_by_region.png"), dpi=150)
    print(f"Saved: {OUTPUT_DIR}/travel_time_by_region.png")

    # =====================================================================
    # 4. GOLDEN-HOUR ANALYSIS — ADVANCED TREATMENT
    # =====================================================================
    print("\n" + "=" * 70)
    print("GOLDEN-HOUR ANALYSIS (60 MINUTES TO ADVANCED STROKE CENTER)")
    print("=" * 70)

    GOLDEN_HOUR_MIN = 60.0

    total_grid_cells = len(pop_snapped)
    total_households = pop_snapped["household_count"].fillna(0).sum()

    # Within golden hour to advanced.
    within_adv = pop_snapped["travel_time_advanced_min"] <= GOLDEN_HOUR_MIN
    outside_adv = pop_snapped["travel_time_advanced_min"].notna() & ~within_adv
    unreachable_adv = pop_snapped["travel_time_advanced_min"].isna()

    cells_within = within_adv.sum()
    cells_outside = outside_adv.sum()
    cells_unreachable = unreachable_adv.sum()

    hh_within = pop_snapped.loc[within_adv, "household_count"].fillna(0).sum()
    hh_outside = pop_snapped.loc[outside_adv, "household_count"].fillna(0).sum()
    hh_unreachable = pop_snapped.loc[unreachable_adv, "household_count"].fillna(0).sum()

    print(f"\nThreshold: {GOLDEN_HOUR_MIN:.0f} minutes to nearest ADVANCED stroke center")
    print(f"\n{'Category':<30} {'Grid cells':>12} {'% cells':>10} {'Households':>14} {'% households':>14}")
    print("-" * 82)
    print(f"{'Within golden hour':<30} {cells_within:>12,} {100*cells_within/total_grid_cells:>9.1f}% {hh_within:>14,.0f} {100*hh_within/total_households:>13.1f}%")
    print(f"{'Outside golden hour':<30} {cells_outside:>12,} {100*cells_outside/total_grid_cells:>9.1f}% {hh_outside:>14,.0f} {100*hh_outside/total_households:>13.1f}%")
    print(f"{'Unreachable (no road path)':<30} {cells_unreachable:>12,} {100*cells_unreachable/total_grid_cells:>9.1f}% {hh_unreachable:>14,.0f} {100*hh_unreachable/total_households:>13.1f}%")
    print(f"{'TOTAL':<30} {total_grid_cells:>12,} {'100.0%':>10} {total_households:>14,.0f} {'100.0%':>14}")

    # Also do golden hour for ANY hospital.
    print(f"\n--- Same breakdown for ANY hospital (not just advanced) ---")
    within_any = pop_snapped["travel_time_any_min"] <= GOLDEN_HOUR_MIN
    outside_any = pop_snapped["travel_time_any_min"].notna() & ~within_any
    unreachable_any = pop_snapped["travel_time_any_min"].isna()

    cells_within_a = within_any.sum()
    cells_outside_a = outside_any.sum()
    cells_unreachable_a = unreachable_any.sum()
    hh_within_a = pop_snapped.loc[within_any, "household_count"].fillna(0).sum()
    hh_outside_a = pop_snapped.loc[outside_any, "household_count"].fillna(0).sum()
    hh_unreachable_a = pop_snapped.loc[unreachable_any, "household_count"].fillna(0).sum()

    print(f"\n{'Category':<30} {'Grid cells':>12} {'% cells':>10} {'Households':>14} {'% households':>14}")
    print("-" * 82)
    print(f"{'Within golden hour':<30} {cells_within_a:>12,} {100*cells_within_a/total_grid_cells:>9.1f}% {hh_within_a:>14,.0f} {100*hh_within_a/total_households:>13.1f}%")
    print(f"{'Outside golden hour':<30} {cells_outside_a:>12,} {100*cells_outside_a/total_grid_cells:>9.1f}% {hh_outside_a:>14,.0f} {100*hh_outside_a/total_households:>13.1f}%")
    print(f"{'Unreachable (no road path)':<30} {cells_unreachable_a:>12,} {100*cells_unreachable_a/total_grid_cells:>9.1f}% {hh_unreachable_a:>14,.0f} {100*hh_unreachable_a/total_households:>13.1f}%")
    print(f"{'TOTAL':<30} {total_grid_cells:>12,} {'100.0%':>10} {total_households:>14,.0f} {'100.0%':>14}")

    # --- 4b. Golden hour by region (advanced) ---
    print(f"\n--- Golden-hour coverage by region (ADVANCED hospitals) ---")
    golden_region = []
    for region in sorted(pop_snapped["region"].unique()):
        sub = pop_snapped[pop_snapped["region"] == region]
        n = len(sub)
        hh = sub["household_count"].fillna(0).sum()

        w = sub["travel_time_advanced_min"] <= GOLDEN_HOUR_MIN
        o = sub["travel_time_advanced_min"].notna() & ~w
        u = sub["travel_time_advanced_min"].isna()

        hh_w = sub.loc[w, "household_count"].fillna(0).sum()
        hh_o = sub.loc[o, "household_count"].fillna(0).sum()
        hh_u = sub.loc[u, "household_count"].fillna(0).sum()

        golden_region.append({
            "Region": region,
            "Households": int(hh),
            "Within 60 min (%)": round(100 * hh_w / hh, 1) if hh > 0 else 0,
            "Outside 60 min (%)": round(100 * hh_o / hh, 1) if hh > 0 else 0,
            "Unreachable (%)": round(100 * hh_u / hh, 1) if hh > 0 else 0,
        })

    golden_df = pd.DataFrame(golden_region)
    print(golden_df.to_string(index=False))

    # --- 4b2. Golden hour by province ---
    print(f"\n--- Golden-hour coverage by province (ADVANCED hospitals) ---")
    golden_province = []
    for province in pop_snapped["province"].dropna().unique():
        sub = pop_snapped[pop_snapped["province"] == province]
        hh = sub["household_count"].fillna(0).sum()
        if hh == 0:
            continue

        w = sub["travel_time_advanced_min"] <= GOLDEN_HOUR_MIN
        o = sub["travel_time_advanced_min"].notna() & ~w
        u = sub["travel_time_advanced_min"].isna()

        hh_w = sub.loc[w, "household_count"].fillna(0).sum()

        golden_province.append({
            "Province": province,
            "Region": PROVINCE_TO_REGION.get(province, "Unknown"),
            "Households": int(hh),
            "Within 60 min (%)": round(100 * hh_w / hh, 1),
        })

    golden_prov_df = pd.DataFrame(golden_province).sort_values("Within 60 min (%)")
    print(golden_prov_df.to_string(index=False))

    # --- 4c. Golden-hour choropleth map ---
    from matplotlib.lines import Line2D

    # Compute per-province golden-hour coverage percentage.
    prov_coverage = {}
    for province in pop_snapped["province"].dropna().unique():
        sub = pop_snapped[pop_snapped["province"] == province]
        hh = sub["household_count"].fillna(0).sum()
        if hh == 0:
            prov_coverage[province] = np.nan
            continue
        hh_within = sub.loc[
            sub["travel_time_advanced_min"] <= GOLDEN_HOUR_MIN, "household_count"
        ].fillna(0).sum()
        prov_coverage[province] = 100.0 * hh_within / hh

    # Dissolve GADM polygons into the 34 consolidated provinces.
    map_gdf = admin_gdf.copy()
    map_gdf["consolidated"] = map_gdf["NAME_1"].map(GADM_TO_CONSOLIDATED)
    map_gdf = map_gdf.dissolve(by="consolidated", as_index=False)
    map_gdf["coverage_pct"] = map_gdf["consolidated"].map(prov_coverage)

    fig, ax = plt.subplots(figsize=(10, 12))

    map_gdf.plot(
        column="coverage_pct",
        cmap="viridis",
        linewidth=0.4,
        edgecolor="#333333",
        legend=True,
        legend_kwds={"label": "% households within 60 min", "shrink": 0.6},
        missing_kwds={"color": "#d9d9d9", "label": "No data"},
        ax=ax,
        vmin=0,
        vmax=100,
    )

    # Overlay advanced hospital markers.
    ax.scatter(
        hospitals_df.loc[hospitals_df["specialized_equipment"], "Longitude"],
        hospitals_df.loc[hospitals_df["specialized_equipment"], "Latitude"],
        s=50, c="red", marker="^", zorder=5, label="Advanced hospital",
        edgecolors="white", linewidths=0.5,
    )
    ax.set_title("Golden-hour coverage by province: Advanced stroke centers (60 min)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markeredgecolor="white", markersize=8, label="Advanced hospital"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "golden_hour_map.png"), dpi=150)
    print(f"\nSaved: {OUTPUT_DIR}/golden_hour_map.png")

    # --- 4d. Golden-hour stacked bar by region ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(golden_df))
    w_vals = golden_df["Within 60 min (%)"].values
    o_vals = golden_df["Outside 60 min (%)"].values
    u_vals = golden_df["Unreachable (%)"].values

    ax.bar(x, w_vals, color="#440154", label="Within 60 min")
    ax.bar(x, o_vals, bottom=w_vals, color="#21918c", label="Outside 60 min")
    ax.bar(x, u_vals, bottom=w_vals + o_vals, color="#fde725", label="Unreachable")
    ax.set_xticks(x)
    ax.set_xticklabels(golden_df["Region"].values, rotation=35, ha="right")
    ax.set_ylabel("% of households")
    ax.set_title("Golden-hour coverage by region (advanced stroke centers)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "golden_hour_by_region.png"), dpi=150)
    print(f"Saved: {OUTPUT_DIR}/golden_hour_by_region.png")

    # =====================================================================
    # 5. PROVINCE-LEVEL COVERAGE CSV
    # =====================================================================
    print("\n" + "=" * 70)
    print("GENERATING PROVINCE-LEVEL COVERAGE CSV")
    print("=" * 70)

    coverage_rows = []
    for province in sorted(pop_snapped["province"].dropna().unique()):
        sub = pop_snapped[pop_snapped["province"] == province]
        hh = sub["household_count"].fillna(0).sum()
        if hh == 0:
            coverage_rows.append({
                "Province": province,
                "Within 60 min (%)": 0.0,
                "Within 180 min (%)": 0.0,
                "Over 180 min (%)": 0.0,
            })
            continue

        hh_60 = sub.loc[
            sub["travel_time_advanced_min"] <= 60, "household_count"
        ].fillna(0).sum()
        hh_180 = sub.loc[
            (sub["travel_time_advanced_min"] > 60) & (sub["travel_time_advanced_min"] <= 180),
            "household_count"
        ].fillna(0).sum()
        hh_over = sub.loc[
            (sub["travel_time_advanced_min"] > 180) | (sub["travel_time_advanced_min"].isna()),
            "household_count"
        ].fillna(0).sum()

        coverage_rows.append({
            "Province": province,
            "Within 60 min (%)": round(100.0 * hh_60 / hh, 1),
            "Within 180 min (%)": round(100.0 * hh_180 / hh, 1),
            "Over 180 min (%)": round(100.0 * hh_over / hh, 1),
        })

    coverage_csv_df = pd.DataFrame(coverage_rows)
    coverage_csv_path = os.path.join(OUTPUT_DIR, "province_coverage_advanced.csv")
    coverage_csv_df.to_csv(coverage_csv_path, index=False)
    print(coverage_csv_df.to_string(index=False))
    print(f"\nSaved: {coverage_csv_path}")

    # --- 5b. Province coverage visualizations ---
    cov = coverage_csv_df.sort_values("Within 60 min (%)", ascending=True).reset_index(drop=True)

    # Figure 1: Horizontal stacked bar — all 34 provinces, coverage tiers
    fig, ax = plt.subplots(figsize=(10, 12))
    y = np.arange(len(cov))
    bar_h = 0.7
    ax.barh(y, cov["Within 60 min (%)"], height=bar_h, color="#440154", label="Within 60 min")
    ax.barh(y, cov["Within 180 min (%)"], height=bar_h, left=cov["Within 60 min (%)"],
            color="#21918c", label="60–180 min")
    ax.barh(y, cov["Over 180 min (%)"], height=bar_h,
            left=cov["Within 60 min (%)"] + cov["Within 180 min (%)"],
            color="#fde725", label="Over 180 min")
    ax.set_yticks(y)
    ax.set_yticklabels(cov["Province"], fontsize=8)
    ax.set_xlabel("% of population")
    ax.set_title("Provincial coverage by advanced stroke centers")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "province_coverage_stacked.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR}/province_coverage_stacked.png")

    # Figure 2: Lollipop chart — golden-hour (60 min) coverage ranked
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = plt.cm.viridis(cov["Within 60 min (%)"] / 100.0)
    ax.hlines(y, xmin=0, xmax=cov["Within 60 min (%)"], color=colors, linewidth=1.5)
    ax.scatter(cov["Within 60 min (%)"], y, color=colors, s=40, zorder=3)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, label="50% threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(cov["Province"], fontsize=8)
    ax.set_xlabel("% population within 60 min of advanced stroke center")
    ax.set_title("Golden-hour accessibility by province (ranked)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "province_golden_hour_lollipop.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR}/province_golden_hour_lollipop.png")

    # Figure 3: Choropleth map — 60-min coverage rate per province
    admin_plot = admin_gdf.copy()
    admin_plot["province"] = admin_plot["NAME_1"].map(GADM_TO_CONSOLIDATED)
    admin_plot = admin_plot.dissolve(by="province", as_index=False)
    admin_plot = admin_plot.merge(cov[["Province", "Within 60 min (%)"]], left_on="province", right_on="Province", how="left")

    fig, ax = plt.subplots(figsize=(8, 12))
    admin_plot.plot(
        column="Within 60 min (%)", ax=ax, cmap="viridis", edgecolor="white",
        linewidth=0.5, legend=True, vmin=0, vmax=100,
        legend_kwds={"label": "% within 60 min", "shrink": 0.4},
    )
    ax.set_title("Golden-hour coverage rate by province")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "province_coverage_choropleth.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR}/province_coverage_choropleth.png")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print(f"  - spatial_overview.png")
    print(f"  - travel_time_distributions.png")
    print(f"  - travel_time_by_region.png")
    print(f"  - golden_hour_map.png")
    print(f"  - golden_hour_by_region.png")
    print(f"  - province_coverage_stacked.png")
    print(f"  - province_golden_hour_lollipop.png")
    print(f"  - province_coverage_choropleth.png")
