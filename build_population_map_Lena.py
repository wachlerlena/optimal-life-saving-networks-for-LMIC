# =============================================================================
# BUILD POPULATION MAP - INDIVIDUAL ARTIFACT (Lena)
# =============================================================================
# Purpose:
# Builds the individual thesis artifact: a nationwide interactive Folium stroke
# map with scenario comparison capabilities.
#
# Features:
# - Population heatmap (nationwide)
# - Province overlay polygons colored by 60-min coverage % (viridis, dynamic)
# - Hospital markers: green (CSC/advanced), red (PSC/basic), blue (upgraded)
# - Sampled clickable population markers with route calculations
# - LEFT panel: scenario selector (k dropdown + lambda slider + Show button)
#               + baseline vs scenario comparison stats table (34 provinces)
# - RIGHT panel: marker click details (same as mockup)
# - LayerControl: toggle heatmap, province overlay, hospitals, population points
#
# Usage:
#   1. Run:  python build_population_map_Lena.py
#   2. Output: stroke_map_Lena.html
# =============================================================================

# =============================================================================
# SCENARIO DATA (hardcoded from optimization results)
# =============================================================================
# Each scenario contains:
#   - upgraded_ids: hospital IDs upgraded from PSC to CSC
#   - national_coverage_60min: % of population within 60 min of any CSC
#   - provincial: dict of 34 provinces → {"pct_60min": float}
#
# Missing data that still needs to be provided:
#   - national_coverage_60min for all scenarios except baseline (currently 0.0)
#     (requires household-weighted average across provinces)
# =============================================================================

# Baseline province coverage (from analysis output, no upgrades).
_BASELINE_PROVINCIAL = {
    "An Giang": 68.1, "Bac Ninh": 88.8, "Ca Mau": 36.5, "Can Tho": 86.9,
    "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 4.0, "Dien Bien": 0.0,
    "Dong Nai": 53.7, "Dong Thap": 58.2, "Gia Lai": 29.1, "Ha Tinh": 46.0,
    "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 89.8, "Hue": 91.6,
    "Hung Yen": 51.8, "Khanh Hoa": 0.7, "Lai Chau": 0.0, "Lam Dong": 18.3,
    "Lang Son": 12.3, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 81.3,
    "Phu Tho": 70.4, "Quang Ngai": 26.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
    "Son La": 0.0, "Tay Ninh": 73.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
    "Tuyen Quang": 5.5, "Vinh Long": 44.0,
}

def _prov(data):
    """Helper: convert {province: pct} to {province: {"pct_60min": pct}}."""
    return {k: {"pct_60min": v} for k, v in data.items()}

SCENARIO_DATA = {
    "baseline": {
        "upgraded_ids": [],
        "national_coverage_60min": 62.4,
        "provincial": _prov(_BASELINE_PROVINCIAL),
    },
    # --- k=5 scenarios ---
    "k5_lambda0": {
        "upgraded_ids": [38, 79, 81, 99, 123],
        "national_coverage_60min": 69.3,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.8, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 56.3, "Dong Thap": 75.1, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 97.2, "Hue": 91.6,
            "Hung Yen": 97.0, "Khanh Hoa": 0.7, "Lai Chau": 0.0, "Lam Dong": 25.4,
            "Lang Son": 12.3, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 85.8,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 77.3, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 68.2,
        }),
    },
    "k5_lambda015": {
        "upgraded_ids": [38, 79, 81, 99, 123],
        "national_coverage_60min": 69.3,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.8, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 56.3, "Dong Thap": 75.1, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 97.2, "Hue": 91.6,
            "Hung Yen": 97.0, "Khanh Hoa": 0.7, "Lai Chau": 0.0, "Lam Dong": 25.4,
            "Lang Son": 12.3, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 85.8,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 77.3, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 68.2,
        }),
    },
    "k5_lambda03": {
        "upgraded_ids": [79, 81, 87, 99, 100],
        "national_coverage_60min": 68.4,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.8, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 56.3, "Dong Thap": 71.2, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 97.2, "Hue": 91.6,
            "Hung Yen": 51.8, "Khanh Hoa": 34.1, "Lai Chau": 0.0, "Lam Dong": 26.6,
            "Lang Son": 12.3, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 81.3,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 74.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 73.4,
        }),
    },
    "k5_lambda05": {
        "upgraded_ids": [61, 79, 81, 87, 100],
        "national_coverage_60min": 67.6,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 53.7, "Dong Thap": 71.2, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 89.8, "Hue": 91.6,
            "Hung Yen": 51.8, "Khanh Hoa": 34.1, "Lai Chau": 0.0, "Lam Dong": 26.5,
            "Lang Son": 65.8, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 81.3,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 74.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 73.4,
        }),
    },
    "k5_lambda08": {
        "upgraded_ids": [61, 80, 81, 87, 89],
        "national_coverage_60min": 66.7,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 53.7, "Dong Thap": 58.2, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 89.8, "Hue": 91.6,
            "Hung Yen": 51.8, "Khanh Hoa": 65.3, "Lai Chau": 0.0, "Lam Dong": 26.5,
            "Lang Son": 65.8, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 81.3,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 73.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 44.0,
        }),
    },
    "k5_lambda1": {
        "upgraded_ids": [61, 80, 81, 87, 89],
        "national_coverage_60min": 66.7,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 53.7, "Dong Thap": 58.2, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 89.8, "Hue": 91.6,
            "Hung Yen": 51.8, "Khanh Hoa": 65.3, "Lai Chau": 0.0, "Lam Dong": 26.5,
            "Lang Son": 65.8, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 81.3,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 73.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 44.0,
        }),
    },
    # --- k=10 scenarios ---
    "k10_lambda0": {
        "upgraded_ids": [38, 79, 81, 87, 88, 89, 99, 115, 120, 123],
        "national_coverage_60min": 72.7,
        "provincial": _prov({
            "An Giang": 74.1, "Bac Ninh": 88.8, "Ca Mau": 36.5, "Can Tho": 87.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 56.3, "Dong Thap": 89.3, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 97.2, "Hue": 91.6,
            "Hung Yen": 97.0, "Khanh Hoa": 65.3, "Lai Chau": 0.0, "Lam Dong": 43.8,
            "Lang Son": 12.3, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 85.8,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 79.1, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 86.9,
        }),
    },
    "k10_lambda015": {
        "upgraded_ids": [38, 61, 79, 81, 86, 87, 89, 99, 100, 115],
        "national_coverage_60min": 72.4,
        "provincial": _prov({
            "An Giang": 74.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 87.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 62.8, "Dien Bien": 0.0,
            "Dong Nai": 56.3, "Dong Thap": 85.5, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 97.2, "Hue": 91.6,
            "Hung Yen": 97.0, "Khanh Hoa": 67.3, "Lai Chau": 0.0, "Lam Dong": 26.6,
            "Lang Son": 65.8, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 85.8,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 0.0, "Tay Ninh": 76.6, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 73.4,
        }),
    },
    "k10_lambda03": {
        "upgraded_ids": [38, 59, 61, 80, 81, 86, 87, 89, 99, 100],
        "national_coverage_60min": 72.0,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 62.8, "Dien Bien": 0.0,
            "Dong Nai": 56.3, "Dong Thap": 71.2, "Gia Lai": 55.7, "Ha Tinh": 46.0,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 97.2, "Hue": 91.6,
            "Hung Yen": 97.0, "Khanh Hoa": 67.3, "Lai Chau": 0.0, "Lam Dong": 26.6,
            "Lang Son": 65.8, "Lao Cai": 24.2, "Nghe An": 44.0, "Ninh Binh": 85.8,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 31.9, "Tay Ninh": 74.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 73.4,
        }),
    },
    "k10_lambda05": {
        "upgraded_ids": [51, 59, 61, 62, 79, 81, 86, 87, 89, 100],
        "national_coverage_60min": 70.1,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 62.8, "Dien Bien": 0.0,
            "Dong Nai": 53.7, "Dong Thap": 71.2, "Gia Lai": 55.7, "Ha Tinh": 77.8,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 89.8, "Hue": 91.6,
            "Hung Yen": 51.8, "Khanh Hoa": 67.3, "Lai Chau": 1.0, "Lam Dong": 26.5,
            "Lang Son": 65.8, "Lao Cai": 51.3, "Nghe An": 44.0, "Ninh Binh": 81.3,
            "Phu Tho": 70.4, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 31.9, "Tay Ninh": 74.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 5.5, "Vinh Long": 73.4,
        }),
    },
    "k10_lambda08": {
        "upgraded_ids": [51, 59, 61, 62, 78, 80, 81, 87, 89, 100],
        "national_coverage_60min": 69.9,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 53.7, "Dong Thap": 71.2, "Gia Lai": 55.7, "Ha Tinh": 78.8,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 89.8, "Hue": 91.6,
            "Hung Yen": 51.8, "Khanh Hoa": 65.3, "Lai Chau": 1.0, "Lam Dong": 26.5,
            "Lang Son": 65.8, "Lao Cai": 52.3, "Nghe An": 44.0, "Ninh Binh": 81.3,
            "Phu Tho": 70.8, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 31.9, "Tay Ninh": 74.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 29.3, "Vinh Long": 73.4,
        }),
    },
    "k10_lambda1": {
        "upgraded_ids": [51, 59, 61, 62, 78, 80, 81, 87, 89, 100],
        "national_coverage_60min": 69.9,
        "provincial": _prov({
            "An Giang": 68.1, "Bac Ninh": 88.9, "Ca Mau": 36.5, "Can Tho": 86.9,
            "Cao Bang": 0.0, "Da Nang": 84.9, "Dak Lak": 44.8, "Dien Bien": 0.0,
            "Dong Nai": 53.7, "Dong Thap": 71.2, "Gia Lai": 55.7, "Ha Tinh": 77.8,
            "Hai Phong": 93.7, "Hanoi": 98.0, "Ho Chi Minh City": 89.8, "Hue": 91.6,
            "Hung Yen": 51.8, "Khanh Hoa": 65.3, "Lai Chau": 1.0, "Lam Dong": 26.5,
            "Lang Son": 65.8, "Lao Cai": 52.3, "Nghe An": 44.0, "Ninh Binh": 81.3,
            "Phu Tho": 70.8, "Quang Ngai": 37.8, "Quang Ninh": 68.3, "Quang Tri": 71.1,
            "Son La": 31.9, "Tay Ninh": 74.9, "Thai Nguyen": 73.0, "Thanh Hoa": 74.3,
            "Tuyen Quang": 29.3, "Vinh Long": 73.4,
        }),
    },
}

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import json
import os

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from build_population_map import (
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

from analysis import (
    GADM_PATH,
    GADM_TO_CONSOLIDATED,
    PROVINCE_TO_REGION,
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
POPULATION_PATH = "Data_vietnam/population.pkl"
HOSPITAL_PATH = "Data_vietnam/existing_hospitals_100/all_hospitals.pkl"
DISTANCES_PATH = "Data_vietnam/existing_hospitals_100/distances_osm_max_300km.pkl"
ROADS_PATH = "Data_vietnam/road_osm_preprocessed.geojson"
STROKE_CSV_PATH = "Data_vietnam/stroke-facs-100-en.csv"
OUTPUT_PATH = "stroke_map_Lena.html"

# Marker sample size for interactive population points (same as mockup).
MAX_MARKER_POINTS = 1000

# The 34 consolidated provinces in display order.
PROVINCES = sorted(PROVINCE_TO_REGION.keys())


# =============================================================================
# LEFT PANEL: Analytics + Scenario Selector (injected HTML/CSS/JS)
# =============================================================================

def _build_left_panel_html(map_name, scenario_data_json):
    """
    Build the HTML/CSS/JS for the left analytics panel.
    Contains: scenario selector (k dropdown, lambda slider, Show button)
    and a baseline vs scenario comparison stats table (34 provinces, 60-min only).
    """
    return f"""
    <style>
        #analytics-panel {{
            position: fixed;
            top: 16px;
            left: 16px;
            width: 380px;
            max-height: calc(100vh - 32px);
            background: rgba(255, 255, 255, 0.97);
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
            z-index: 9999;
            overflow-y: auto;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
        }}

        #analytics-panel-header {{
            display: flex;
            align-items: center;
            padding: 14px 16px;
            border-bottom: 1px solid #e6e6e6;
            background: #fafafa;
            position: sticky;
            top: 0;
            z-index: 2;
        }}

        #analytics-panel-title {{
            font-size: 18px;
            font-weight: 700;
            color: #222;
        }}

        #analytics-panel-content {{
            padding: 14px 16px 18px 16px;
            overflow-y: auto;
        }}

        .scenario-section {{
            margin-bottom: 16px;
        }}

        .scenario-section label {{
            display: block;
            font-size: 13px;
            font-weight: 700;
            color: #444;
            margin-bottom: 4px;
        }}

        .scenario-section select,
        .scenario-section input[type=range] {{
            width: 100%;
            margin-bottom: 8px;
        }}

        .scenario-section select {{
            padding: 6px 8px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 13px;
        }}

        #lambda-value-display {{
            font-size: 12px;
            color: #555;
            text-align: center;
            margin-top: -4px;
            margin-bottom: 8px;
        }}

        #show-scenario-btn {{
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 8px;
            background: #1f6feb;
            color: white;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 4px;
        }}

        #show-scenario-btn:hover {{
            background: #1558be;
        }}

        .stats-divider {{
            border: none;
            border-top: 1px solid #e6e6e6;
            margin: 14px 0;
        }}

        #stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}

        #stats-table th {{
            background: #f5f5f5;
            padding: 5px 4px;
            text-align: center;
            font-weight: 700;
            border-bottom: 2px solid #ddd;
            color: #333;
        }}

        #stats-table th:first-child {{
            text-align: left;
        }}

        #stats-table td {{
            padding: 4px;
            text-align: center;
            border-bottom: 1px solid #eee;
            color: #333;
        }}

        #stats-table td:first-child {{
            text-align: left;
            font-weight: 600;
            font-size: 11px;
        }}

        #stats-table tr.national-row {{
            background: #f0f7ff;
            font-weight: 700;
        }}

        #stats-table tr.national-row td {{
            border-bottom: 2px solid #ccc;
            font-weight: 700;
        }}

        .stats-header {{
            font-size: 14px;
            font-weight: 700;
            color: #222;
            margin-bottom: 8px;
        }}

        .lambda-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #666;
            margin-top: -2px;
            margin-bottom: 6px;
        }}

        .delta-positive {{
            color: #1c6b2a;
        }}

        .delta-negative {{
            color: #9e2020;
        }}
    </style>

    <div id="analytics-panel">
        <div id="analytics-panel-header">
            <div id="analytics-panel-title">Stroke Facility Optimization</div>
        </div>
        <div id="analytics-panel-content">
            <!-- Scenario selector -->
            <div class="scenario-section">
                <label for="k-select">Upgrade budget (k)</label>
                <select id="k-select">
                    <option value="current">Current (no upgrades)</option>
                    <option value="5">k = 5 (upgrade 5 centers)</option>
                    <option value="10">k = 10 (upgrade 10 centers)</option>
                </select>

                <label for="lambda-slider">Equity weight (&lambda;)</label>
                <input type="range" id="lambda-slider" min="0" max="5" step="1" value="0">
                <div class="lambda-labels">
                    <span>0</span>
                    <span>0.15</span>
                    <span>0.3</span>
                    <span>0.5</span>
                    <span>0.8</span>
                    <span>1.0</span>
                </div>
                <div id="lambda-value-display">&lambda; = 0</div>

                <button id="show-scenario-btn" onclick="applyScenario()">Show</button>
            </div>

            <hr class="stats-divider">

            <!-- Stats comparison table -->
            <div class="stats-header">Golden-Hour Coverage (60 min)</div>
            <table id="stats-table">
                <thead>
                    <tr>
                        <th>Province</th>
                        <th>Baseline</th>
                        <th>Scenario</th>
                        <th>&Delta;</th>
                    </tr>
                </thead>
                <tbody id="stats-table-body">
                    <!-- Filled by JS -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Scenario data embedded from Python
        var scenarioData = {scenario_data_json};

        // Lambda slider stops: index -> actual value
        var lambdaStops = [0, 0.15, 0.3, 0.5, 0.8, 1.0];
        var lambdaKeys = ["0", "015", "03", "05", "08", "1"];

        // Update lambda display when slider moves
        document.getElementById("lambda-slider").addEventListener("input", function() {{
            var idx = parseInt(this.value);
            var val = lambdaStops[idx];
            document.getElementById("lambda-value-display").innerHTML = "&lambda; = " + String(val);
        }});

        function getScenarioKey() {{
            var k = document.getElementById("k-select").value;
            if (k === "current") return "baseline";
            var lambdaIdx = parseInt(document.getElementById("lambda-slider").value);
            var lambdaKey = lambdaKeys[lambdaIdx];
            return "k" + k + "_lambda" + lambdaKey;
        }}

        function updateStatsTable(scenarioKey) {{
            var baseline = scenarioData["baseline"];
            var scenario = scenarioData[scenarioKey];
            if (!scenario) scenario = baseline;

            var provinces = {json.dumps(PROVINCES)};
            var tbody = document.getElementById("stats-table-body");
            var html = "";

            // National row
            var bNat = baseline.national_coverage_60min;
            var sNat = scenario.national_coverage_60min;
            var dNat = sNat - bNat;
            var dNatClass = dNat > 0 ? "delta-positive" : (dNat < 0 ? "delta-negative" : "");
            var dNatText = (sNat === 0 && scenarioKey !== "baseline") ? "-" : ((dNat >= 0 ? "+" : "") + dNat.toFixed(1));
            html += '<tr class="national-row">';
            html += '<td>National</td>';
            html += '<td>' + bNat.toFixed(1) + '%</td>';
            html += '<td>' + (sNat === 0 && scenarioKey !== "baseline" ? "-" : sNat.toFixed(1) + '%') + '</td>';
            html += '<td class="' + dNatClass + '">' + (scenarioKey === "baseline" ? "" : dNatText) + '</td>';
            html += '</tr>';

            // Province rows
            provinces.forEach(function(prov) {{
                var bProv = baseline.provincial[prov] || {{}};
                var sProv = scenario.provincial[prov] || {{}};
                var bVal = bProv.pct_60min || 0;
                var sVal = sProv.pct_60min || 0;
                var delta = sVal - bVal;
                var deltaClass = delta > 0.05 ? "delta-positive" : (delta < -0.05 ? "delta-negative" : "");
                var deltaText = scenarioKey === "baseline" ? "" : ((delta >= 0 ? "+" : "") + delta.toFixed(1));
                html += '<tr>';
                html += '<td>' + prov + '</td>';
                html += '<td>' + bVal.toFixed(1) + '%</td>';
                html += '<td>' + sVal.toFixed(1) + '%</td>';
                html += '<td class="' + deltaClass + '">' + deltaText + '</td>';
                html += '</tr>';
            }});

            tbody.innerHTML = html;
        }}

        function buildHospitalIconHtml(bgColor, borderColor) {{
            return '<div style="position:relative;width:24px;height:24px;border-radius:50%;background:' + bgColor + ';border:2px solid ' + borderColor + ';box-sizing:border-box;">' +
                '<div style="position:absolute;left:50%;top:50%;width:13px;height:5px;background:white;transform:translate(-50%,-50%);border-radius:1px;"></div>' +
                '<div style="position:absolute;left:50%;top:50%;width:5px;height:13px;background:white;transform:translate(-50%,-50%);border-radius:1px;"></div></div>';
        }}

        function updateHospitalColors(scenarioKey) {{
            var scenario = scenarioData[scenarioKey];
            var upgradedIds = scenario ? scenario.upgraded_ids : [];

            Object.keys(hospitalMarkerLookup).forEach(function(hospId) {{
                var markerVarName = hospitalMarkerLookup[hospId];
                try {{
                    var markerObj = eval(markerVarName);
                    if (markerObj) {{
                        var isUpgraded = upgradedIds.indexOf(parseInt(hospId)) >= 0;
                        var bgColor, borderColor;
                        if (isUpgraded) {{
                            bgColor = "#1f6feb";
                            borderColor = "#0a3d91";
                        }} else {{
                            var hospData = hospitalPanelData[hospId];
                            if (hospData && hospData.specialized_equipment) {{
                                bgColor = "#19a74a";
                                borderColor = "black";
                            }} else {{
                                bgColor = "#e53935";
                                borderColor = "black";
                            }}
                        }}
                        var newIcon = L.divIcon({{
                            html: buildHospitalIconHtml(bgColor, borderColor),
                            iconSize: [24, 24],
                            iconAnchor: [12, 12],
                            className: 'empty'
                        }});
                        markerObj.setIcon(newIcon);
                    }}
                }} catch(e) {{}}
            }});
        }}

        function updateProvinceOverlay(scenarioKey) {{
            var scenario = scenarioData[scenarioKey];
            if (!scenario) return;

            var provLayer = (typeof _getProvinceLayer === "function") ? _getProvinceLayer() : null;
            if (provLayer) {{
                provLayer.eachLayer(function(layer) {{
                    var provName = layer.feature.properties.province;
                    var provData = scenario.provincial[provName];
                    var pct = provData ? provData.pct_60min : 0;
                    var color = viridisColor(pct / 100.0);
                    layer.setStyle({{ fillColor: color, fillOpacity: 0.45 }});
                }});
            }}
        }}

        // Viridis-like colormap (simplified 5-stop version for 0..1 range)
        function viridisColor(t) {{
            t = Math.max(0, Math.min(1, t));
            var stops = [
                [0.0,  [68, 1, 84]],
                [0.25, [59, 82, 139]],
                [0.5,  [33, 145, 140]],
                [0.75, [94, 201, 98]],
                [1.0,  [253, 231, 37]]
            ];
            for (var i = 0; i < stops.length - 1; i++) {{
                if (t >= stops[i][0] && t <= stops[i+1][0]) {{
                    var frac = (t - stops[i][0]) / (stops[i+1][0] - stops[i][0]);
                    var r = Math.round(stops[i][1][0] + frac * (stops[i+1][1][0] - stops[i][1][0]));
                    var g = Math.round(stops[i][1][1] + frac * (stops[i+1][1][1] - stops[i][1][1]));
                    var b = Math.round(stops[i][1][2] + frac * (stops[i+1][1][2] - stops[i][1][2]));
                    return "rgb(" + r + "," + g + "," + b + ")";
                }}
            }}
            return "rgb(253,231,37)";
        }}

        function applyScenario() {{
            var scenarioKey = getScenarioKey();
            console.log("Applying scenario:", scenarioKey);
            updateStatsTable(scenarioKey);
            updateHospitalColors(scenarioKey);
            updateProvinceOverlay(scenarioKey);
        }}

        // Initialize stats table - retry until map is ready (large HTML may delay parsing)
        function initTable() {{
            var tbody = document.getElementById("stats-table-body");
            if (tbody) {{
                updateStatsTable("baseline");
            }} else {{
                setTimeout(initTable, 500);
            }}
        }}
        window.addEventListener('load', function() {{
            setTimeout(initTable, 300);
        }});
    </script>
    """


# =============================================================================
# RIGHT PANEL: Marker Click Details (same as mockup map_ui.py, adapted)
# =============================================================================

def _build_right_panel_html(
    map_name,
    route_data,
    population_panel_data,
    population_marker_lookup,
    hospital_panel_data,
    hospital_marker_lookup,
):
    """
    Build the right-side panel for marker click details.
    Same functionality as map_ui.py inject_side_panel_and_routes(), but
    integrated here to coexist with the left panel.
    """
    route_json = json.dumps(route_data)
    population_panel_json = json.dumps(population_panel_data)
    population_marker_lookup_json = json.dumps(population_marker_lookup)
    hospital_panel_json = json.dumps(hospital_panel_data)
    hospital_marker_lookup_json = json.dumps(hospital_marker_lookup)

    return f"""
    <style>
        #population-side-panel {{
            position: fixed;
            top: 16px;
            right: 16px;
            width: 430px;
            max-height: calc(100vh - 32px);
            background: rgba(255, 255, 255, 0.97);
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
            z-index: 9999;
            overflow: hidden;
            display: none;
            font-family: Arial, sans-serif;
        }}

        #population-side-panel.open {{
            display: flex;
            flex-direction: column;
        }}

        #population-side-panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 16px;
            border-bottom: 1px solid #e6e6e6;
            background: #fafafa;
            position: sticky;
            top: 0;
            z-index: 2;
        }}

        #population-side-panel-title {{
            font-size: 18px;
            font-weight: 700;
            color: #222;
        }}

        #population-side-panel-close {{
            border: none;
            background: transparent;
            font-size: 24px;
            line-height: 1;
            cursor: pointer;
            color: #666;
        }}

        #population-side-panel-close:hover {{
            color: #111;
        }}

        #population-side-panel-content {{
            padding: 14px 16px 18px 16px;
            overflow-y: auto;
        }}

        .panel-meta {{
            margin-bottom: 12px;
            font-size: 14px;
            color: #333;
            line-height: 1.45;
        }}

        .panel-meta b {{
            color: #111;
        }}

        .hospital-card {{
            border: 1px solid #e8e8e8;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            background: #fcfcfc;
        }}

        .hospital-card-title {{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #1f1f1f;
        }}

        .hospital-card-row {{
            font-size: 14px;
            line-height: 1.45;
            color: #333;
            margin-bottom: 3px;
        }}

        .hospital-chip {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            margin-left: 6px;
        }}

        .chip-yes {{
            background: #dff5e3;
            color: #1c6b2a;
        }}

        .chip-no {{
            background: #fde3e3;
            color: #9e2020;
        }}

        .route-btn {{
            margin-top: 10px;
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 700;
            background: #1f6feb;
            color: white;
        }}

        .route-btn:hover {{
            background: #1558be;
        }}

        .route-btn:disabled {{
            background: #bfc7d3;
            cursor: not-allowed;
        }}

        .panel-empty {{
            color: #555;
            font-size: 14px;
            line-height: 1.5;
        }}
    </style>

    <div id="population-side-panel">
        <div id="population-side-panel-header">
            <div id="population-side-panel-title">Details</div>
            <button id="population-side-panel-close" onclick="hidePopulationPanel()">&#215;</button>
        </div>
        <div id="population-side-panel-content">
            <div class="panel-empty">Click a population point or hospital to view details.</div>
        </div>
    </div>

    <script>
        var routeData = {route_json};
        var populationPanelData = {population_panel_json};
        var populationMarkerLookup = {population_marker_lookup_json};
        var hospitalPanelData = {hospital_panel_json};
        var hospitalMarkerLookup = {hospital_marker_lookup_json};
        var activeRouteLine = null;
        var activePopulationId = null;

        function formatNumber(value, decimals) {{
            if (value === null || value === undefined || isNaN(value)) return "Not available";
            return Number(value).toFixed(decimals);
        }}

        function clearActiveRoute() {{
            if (activeRouteLine) {{
                try {{
                    {map_name}.removeLayer(activeRouteLine);
                }} catch (err) {{}}
                activeRouteLine = null;
            }}
        }}

        function hidePopulationPanel() {{
            clearActiveRoute();
            var panel = document.getElementById("population-side-panel");
            if (panel) panel.classList.remove("open");
        }}

        function showHospitalRoute(popId, hospitalId) {{
            clearActiveRoute();
            var popRoutes = routeData[String(popId)];
            if (!popRoutes) return;
            var coords = popRoutes[String(hospitalId)];
            if (!coords || coords.length < 2) return;

            activeRouteLine = L.polyline(coords, {{
                color: "red", weight: 4, opacity: 0.9
            }}).addTo({map_name});

            try {{
                {map_name}.fitBounds(activeRouteLine.getBounds(), {{padding: [20, 20]}});
            }} catch (err) {{}}
        }}

        function renderPopulationPanel(popId) {{
            clearActiveRoute();
            var panel = document.getElementById("population-side-panel");
            var content = document.getElementById("population-side-panel-content");
            var title = document.getElementById("population-side-panel-title");
            if (!panel || !content || !title) return;

            var data = populationPanelData[String(popId)];
            title.innerHTML = "Population details";

            if (!data) {{
                content.innerHTML = '<div class="panel-empty">No data available.</div>';
                panel.classList.add("open");
                return;
            }}

            var hospitals = data.hospitals || [];
            var html = '<div class="panel-meta">';
            html += '<div><b>Population ID:</b> ' + data.population_id + '</div>';
            html += '<div><b>Household count:</b> ' + data.household_count + '</div>';
            html += '</div>';

            if (hospitals.length === 0) {{
                html += '<div class="panel-empty">Top hospital data not available.</div>';
            }} else {{
                hospitals.forEach(function(h, index) {{
                    var hospitalId = h.hospital_id;
                    var specialized = !!h.specialized_equipment;
                    var chipClass = specialized ? "chip-yes" : "chip-no";
                    var chipText = specialized ? "Advanced (CSC)" : "Basic (PSC)";
                    var distanceText = formatNumber(h.distance_km, 2);
                    if (distanceText !== "Not available") distanceText += " km";
                    var timeText = formatNumber(h.estimated_travel_time_min, 1);
                    if (timeText !== "Not available") timeText += " min";

                    var hospInfo = hospitalPanelData[String(hospitalId)];
                    var hospName = hospInfo ? (hospInfo.hospital_name || "") : "";
                    var hospAddress = hospInfo ? (hospInfo.address_text || "") : "";
                    var cardTitle = hospName
                        ? (index + 1) + '. ' + hospName + ' (ID ' + hospitalId + ')'
                        : (index + 1) + '. Hospital ' + hospitalId;

                    html += '<div class="hospital-card">';
                    html += '<div class="hospital-card-title">' + cardTitle + '</div>';
                    if (hospAddress) html += '<div class="hospital-card-row"><b>Address:</b> ' + hospAddress + '</div>';
                    html += '<div class="hospital-card-row"><b>Stroke equipment:</b> <span class="hospital-chip ' + chipClass + '">' + chipText + '</span></div>';
                    html += '<div class="hospital-card-row"><b>Distance:</b> ' + distanceText + '</div>';
                    html += '<div class="hospital-card-row"><b>Estimated travel time:</b> ' + timeText + '</div>';

                    if (h.route_available) {{
                        html += '<button class="route-btn" onclick="showHospitalRoute(\\'' + data.population_id + '\\', \\'' + hospitalId + '\\')">Show route</button>';
                    }} else {{
                        html += '<button class="route-btn" disabled>Route not available</button>';
                    }}
                    html += '</div>';
                }});
            }}

            content.innerHTML = html;
            panel.classList.add("open");
        }}

        function showPopulationPanel(popId) {{
            activePopulationId = popId;
            renderPopulationPanel(popId);
        }}

        function showHospitalPanel(hospitalId) {{
            clearActiveRoute();
            var panel = document.getElementById("population-side-panel");
            var content = document.getElementById("population-side-panel-content");
            var title = document.getElementById("population-side-panel-title");
            if (!panel || !content || !title) return;

            var data = hospitalPanelData[String(hospitalId)];
            title.innerHTML = "Hospital details";

            if (!data) {{
                content.innerHTML = '<div class="panel-empty">No data available.</div>';
                panel.classList.add("open");
                return;
            }}

            var chipClass = data.specialized_equipment ? "chip-yes" : "chip-no";
            var chipText = data.specialized_equipment ? "Advanced (CSC)" : "Basic (PSC)";
            var hospitalName = data.hospital_name || "";
            var titleText = hospitalName
                ? hospitalName + ' (ID ' + data.hospital_id + ')'
                : 'Hospital ' + data.hospital_id;

            var html = '<div class="hospital-card">';
            html += '<div class="hospital-card-title">' + titleText + '</div>';
            html += '<div class="hospital-card-row"><b>Hospital ID:</b> ' + data.hospital_id + '</div>';
            if (hospitalName) html += '<div class="hospital-card-row"><b>Name:</b> ' + hospitalName + '</div>';
            html += '<div class="hospital-card-row"><b>Stroke equipment:</b> <span class="hospital-chip ' + chipClass + '">' + chipText + '</span></div>';
            if (data.address_text) html += '<div class="hospital-card-row"><b>Address:</b> ' + data.address_text + '</div>';
            html += '</div>';

            content.innerHTML = html;
            panel.classList.add("open");
        }}

        function bindMarkerClicks() {{
            Object.keys(populationMarkerLookup).forEach(function(popId) {{
                var markerVarName = populationMarkerLookup[popId];
                try {{
                    var markerObj = eval(markerVarName);
                    if (markerObj) {{
                        markerObj.on('click', function() {{ showPopulationPanel(popId); }});
                    }}
                }} catch (err) {{}}
            }});
            Object.keys(hospitalMarkerLookup).forEach(function(hospitalId) {{
                var markerVarName = hospitalMarkerLookup[hospitalId];
                try {{
                    var markerObj = eval(markerVarName);
                    if (markerObj) {{
                        markerObj.on('click', function() {{ showHospitalPanel(hospitalId); }});
                    }}
                }} catch (err) {{}}
            }});
        }}

        window.showHospitalRoute = showHospitalRoute;
        window.hidePopulationPanel = hidePopulationPanel;
        window.showPopulationPanel = showPopulationPanel;
        window.showHospitalPanel = showHospitalPanel;

        window.addEventListener('load', function() {{
            setTimeout(bindMarkerClicks, 500);
        }});
    </script>
    """


# =============================================================================
# PROVINCE OVERLAY: GeoJSON polygons colored by coverage
# =============================================================================

def build_province_geojson(admin_gdf):
    """
    Dissolve 63 GADM provinces into 34 consolidated provinces and return
    a GeoDataFrame with province names for overlay coloring.
    """
    gdf = admin_gdf.copy()
    gdf["province"] = gdf["NAME_1"].map(GADM_TO_CONSOLIDATED)
    gdf = gdf.dropna(subset=["province"])

    # Dissolve by consolidated province to get 34 polygons.
    province_gdf = gdf.dissolve(by="province", as_index=False)[["province", "geometry"]]
    return province_gdf


def viridis_hex(t):
    """Python-side viridis approximation for initial polygon coloring."""
    t = max(0.0, min(1.0, t))
    stops = [
        (0.0,  (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5,  (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0,  (253, 231, 37)),
    ]
    for i in range(len(stops) - 1):
        if stops[i][0] <= t <= stops[i + 1][0]:
            frac = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
            r = int(stops[i][1][0] + frac * (stops[i + 1][1][0] - stops[i][1][0]))
            g = int(stops[i][1][1] + frac * (stops[i + 1][1][1] - stops[i][1][1]))
            b = int(stops[i][1][2] + frac * (stops[i + 1][1][2] - stops[i][1][2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fde725"


# =============================================================================
# MAIN MAP BUILDER
# =============================================================================

def build_lena_map():
    """Build the individual thesis artifact map."""

    print("=" * 70)
    print("Building Individual Artifact: stroke_map_Lena.html")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load population data
    # ------------------------------------------------------------------
    print("\n[1/9] Loading population data...")
    population_df = load_pickle(POPULATION_PATH)

    required_pop_cols = {"ID", "lat", "lon"}
    if not required_pop_cols.issubset(population_df.columns):
        raise ValueError("population.pkl must contain 'ID', 'lat', and 'lon' columns")

    print(f"  Total population rows: {len(population_df)}")

    # ------------------------------------------------------------------
    # Step 2: Create base Folium map (Vietnam-wide)
    # ------------------------------------------------------------------
    print("\n[2/9] Creating base map...")
    center_lat = float(population_df["lat"].median())
    center_lon = float(population_df["lon"].median())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    # ------------------------------------------------------------------
    # Step 3: Build population heatmap
    # ------------------------------------------------------------------
    print("\n[3/9] Building population heatmap...")
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

    # ------------------------------------------------------------------
    # Step 4: Build province overlay (viridis by baseline 60-min coverage)
    # ------------------------------------------------------------------
    print("\n[4/9] Building province overlay polygons...")
    admin_gdf = gpd.read_file(GADM_PATH, layer="ADM_ADM_1")
    province_gdf = build_province_geojson(admin_gdf)

    baseline_provincial = SCENARIO_DATA["baseline"]["provincial"]

    def style_function(feature):
        prov_name = feature["properties"]["province"]
        prov_data = baseline_provincial.get(prov_name, {})
        pct = prov_data.get("pct_60min", 0.0)
        color = viridis_hex(pct / 100.0)
        return {
            "fillColor": color,
            "fillOpacity": 0.45,
            "color": "#333",
            "weight": 1.5,
            "opacity": 0.7,
        }

    province_geojson = json.loads(province_gdf.to_json())
    for feature in province_geojson["features"]:
        feature["properties"] = {"province": feature["properties"]["province"]}

    province_layer = folium.GeoJson(
        province_geojson,
        name="Province coverage overlay",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["province"],
            aliases=[""],
            style="font-size:13px; font-weight:bold; background:rgba(255,255,255,0.85); border-radius:4px; padding:4px 8px;",
        ),
        overlay=True,
        show=True,
    )
    province_layer.add_to(m)

    province_layer_name = province_layer.get_name()

    # ------------------------------------------------------------------
    # Step 5: Load hospitals and add markers
    # ------------------------------------------------------------------
    print("\n[5/9] Loading hospitals...")
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

    hospital_marker_lookup = {}
    hospital_panel_data = {}
    # Two layers: individual (default visible) and clustered (hidden).
    hospital_group = folium.FeatureGroup(name="Hospitals (individual)", show=True).add_to(m)
    hospital_cluster = MarkerCluster(name="Hospitals (clustered)", show=False).add_to(m)
    hospital_group_name = hospital_group.get_name()
    hospital_cluster_name = hospital_cluster.get_name()

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

        # Add to the individual (non-clustered) layer.
        marker = folium.Marker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            icon=folium.DivIcon(
                html=hospital_icon_html,
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                class_name="empty",
            ),
        ).add_to(hospital_group)

        # Also add a duplicate to the clustered layer.
        folium.Marker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            icon=folium.DivIcon(
                html=hospital_icon_html,
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                class_name="empty",
            ),
        ).add_to(hospital_cluster)

        if hospital_id is not None:
            hospital_marker_lookup[str(hospital_id)] = marker.get_name()
            hospital_panel_data[str(hospital_id)] = {
                "hospital_id": hospital_id,
                "hospital_name": name_text,
                "specialized_equipment": has_specialized,
                "address_text": address_text,
            }

    # ------------------------------------------------------------------
    # Step 6: Build interactive population markers (sampled)
    # ------------------------------------------------------------------
    print(f"\n[6/9] Building interactive population markers (sample={MAX_MARKER_POINTS})...")
    interactive_df = build_nearest_hospital_lookup(
        population_df=population_df,
        hospital_df=hospitals_df,
        sample_size=MAX_MARKER_POINTS,
    )
    print(f"  Interactive points: {len(interactive_df)}")

    # Attach precomputed distances.
    if os.path.exists(DISTANCES_PATH):
        try:
            distances_df = load_pickle(DISTANCES_PATH)
            interactive_df = attach_precomputed_distance_for_selected_hospital(
                interactive_df=interactive_df,
                distances_df=distances_df,
            )
        except Exception as e:
            print(f"  Warning: could not attach distances: {e}")
            interactive_df["distance_to_hospital_km"] = np.nan
            interactive_df["distance_source"] = "missing"
    else:
        interactive_df["distance_to_hospital_km"] = np.nan
        interactive_df["distance_source"] = "missing"

    # ------------------------------------------------------------------
    # Step 7: Build road graph, snap, calculate routes
    # ------------------------------------------------------------------
    print("\n[7/9] Building road network and calculating routes...")
    G = None
    tree = None
    node_list = None

    if os.path.exists(ROADS_PATH):
        roads_geojson = load_roads_geojson(ROADS_PATH)
        G = build_graph_from_roads_geojson(roads_geojson)
        tree, node_list, _ = build_graph_kdtree(G)
    else:
        print("  No road GeoJSON found; routing will be skipped.")

    if G is not None and tree is not None:
        interactive_df = snap_points_to_graph_nodes(
            interactive_df, lat_col="lat", lon_col="lon",
            tree=tree, node_list=node_list, label="population",
        )
        hospital_snap_df = snap_points_to_graph_nodes(
            hospitals_df.copy(), lat_col="Latitude", lon_col="Longitude",
            tree=tree, node_list=node_list, label="hospital",
        )
        interactive_df = build_top_k_hospital_popup_data(
            interactive_df=interactive_df,
            hospital_snap_df=hospital_snap_df,
            graph=G,
            k=3,
        )

    # ------------------------------------------------------------------
    # Step 8: Add population markers to map
    # ------------------------------------------------------------------
    print(f"\n[8/9] Adding population markers...")
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

    # ------------------------------------------------------------------
    # Step 9: Inject UI panels and save
    # ------------------------------------------------------------------
    print("\n[9/9] Injecting UI panels and saving...")

    map_name = m.get_name()

    # Inject province layer JS reference for scenario overlay updates.
    province_layer_js = f"""
    <script>
        var _provinceLayerVarName = "{province_layer_name}";
        function _getProvinceLayer() {{ return window[_provinceLayerVarName]; }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(province_layer_js))

    # Inject left analytics panel.
    scenario_data_json = json.dumps(SCENARIO_DATA)
    left_panel_html = _build_left_panel_html(map_name, scenario_data_json)
    m.get_root().html.add_child(folium.Element(left_panel_html))

    # Inject right details panel.
    right_panel_html = _build_right_panel_html(
        map_name,
        route_data,
        population_panel_data,
        population_marker_lookup,
        hospital_panel_data,
        hospital_marker_lookup,
    )
    m.get_root().html.add_child(folium.Element(right_panel_html))

    # Layer control.
    folium.LayerControl().add_to(m)

    # Inject mutual-exclusion logic for hospital layers (deferred).
    hospital_toggle_js = f"""
    <script>
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                var hospIndividual = window["{hospital_group_name}"];
                var hospClustered = window["{hospital_cluster_name}"];
                var map = window["{map_name}"];

                if (map) {{
                    map.on('overlayadd', function(e) {{
                        if (e.layer === hospIndividual && map.hasLayer(hospClustered)) {{
                            map.removeLayer(hospClustered);
                        }} else if (e.layer === hospClustered && map.hasLayer(hospIndividual)) {{
                            map.removeLayer(hospIndividual);
                        }}
                    }});
                }}
            }}, 200);
        }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(hospital_toggle_js))

    m.save(OUTPUT_PATH)
    print(f"\nMap saved to {OUTPUT_PATH}")
    print(f"Population points (heatmap): {len(population_df)}")
    print(f"Population points (interactive): {len(interactive_df)}")
    print(f"Hospitals: {len(hospitals_df)}")
    print(f"Provinces: {len(province_gdf)}")


if __name__ == "__main__":
    build_lena_map()
