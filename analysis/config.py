"""
Shared paths and constants for the Vietnam stroke-care accessibility pipeline.
Steps 3-5 of the methodology read these.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data_vietnam"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POPULATION_PKL = DATA_DIR / "population.pkl"
HOSPITALS_PKL = DATA_DIR / "existing_hospitals_100" / "all_hospitals.pkl"
ROADS_GEOJSON = DATA_DIR / "road_osm_preprocessed.geojson"
STROKE_FACS_CSV = DATA_DIR / "stroke-facs-100-en.csv"

# Step 2 hard cap on clinically relevant route distance (km).
MAX_ROUTE_DISTANCE_KM = 300.0

# Step 4 disruption percentiles (top-X fraction of edges by C(e) removed).
DISRUPTION_FRACTIONS = (0.01, 0.05, 0.10)

# Step 5 isochrone windows (minutes).
ISOCHRONE_WINDOWS_MIN = (60, 120, 180)

# Regional latitude bands for North / Central / South Vietnam stratification.
# Standard cut: North >= 18 deg N, Central 14-18 deg N, South < 14 deg N.
REGION_BOUNDS = {
    "North": (18.0, 90.0),
    "Central": (14.0, 18.0),
    "South": (-90.0, 14.0),
}


def region_for_lat(lat: float) -> str:
    for name, (lo, hi) in REGION_BOUNDS.items():
        if lo <= lat < hi:
            return name
    return "Unknown"
