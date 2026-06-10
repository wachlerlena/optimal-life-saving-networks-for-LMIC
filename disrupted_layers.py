"""
Disrupted-scenario impact layers for the combined Folium map.

For each (scenario × tier), plots a per-population-point CircleMarker layer
on top of the existing combined map. The points used are the SAME 1,000-row
sample build_population_map.py uses for its blue interactive markers
(`pop.sample(1000, random_state=0)`), so each blue dot has a perfectly
co-located colored twin per scenario.

Each colored point is:
    * placed at the blue dot's lat/lon (the pre-snapped column in
      population.pkl)
    * colored by that single point's travel-time delta vs baseline
    * sized (radius) by household_count, log-scaled
    * turns BLACK if the point becomes unreachable under the scenario
      (baseline reachable, scenario NaN)

This is the per-point view: visual alignment is exact, but coverage is
limited to the 1k sample. Underlying analysis still operates on all 406k
rows; this is purely a visualization choice for the map.
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd

SAMPLE_SIZE = 1000
SAMPLE_SEED = 0  # matches build_population_map.build_nearest_hospital_lookup

SCENARIOS = ("A_1pct", "B_5pct", "C_10pct")
SCENARIO_LABELS = {
    "A_1pct": "A (top 1%)",
    "B_5pct": "B (top 5%)",
    "C_10pct": "C (top 10%)",
}
TIERS = ("any", "comprehensive")
TIER_LABELS = {"any": "any centre", "comprehensive": "comprehensive only"}

# Delta-min bins → fill color. Bin edges are inclusive on the right.
DELTA_BINS = [
    (5,      "#1a9850", "≤ 5 min"),
    (30,     "#a6d96a", "5–30 min"),
    (60,     "#fee08b", "30–60 min"),
    (120,    "#fdae61", "60–120 min"),
    (np.inf, "#d73027", "> 120 min"),
]
UNREACHABLE_COLOR = "#111111"


def _log(msg: str) -> None:
    print(f"[disrupted-layers] {msg}", flush=True)


def _color_for_delta(delta_min: float) -> str:
    if delta_min is None or (isinstance(delta_min, float) and np.isnan(delta_min)):
        return DELTA_BINS[0][1]
    for upper, color, _ in DELTA_BINS:
        if delta_min <= upper:
            return color
    return DELTA_BINS[-1][1]


def _radius_for_weight(w: float, ref_max: float) -> float:
    if w <= 0 or ref_max <= 0:
        return 3.0
    scaled = np.log1p(w) / np.log1p(ref_max)
    return float(np.clip(3.0 + 6.0 * scaled, 3.0, 9.0))


def build_disruption_geojson(
    travel_times_wide_path: str | Path,
    population_path: str | Path | None = None,
    sample_size: int = SAMPLE_SIZE,
    sample_seed: int = SAMPLE_SEED,
) -> dict:
    """
    Build GeoJSON FeatureCollections for all scenario × tier combinations.

    Returns a dict keyed by (scenario, tier) strings, e.g. ("A_1pct", "any").
    Each value is a GeoJSON FeatureCollection with point features whose
    properties carry the radius, color, and tooltip for Leaflet rendering.
    Used by map_server.py to serve disruption layers lazily.
    """
    tt_path = Path(travel_times_wide_path)
    if not tt_path.exists():
        _log(f"WARNING: {tt_path} not found; returning empty disruption data.")
        return {}

    tt = pd.read_pickle(tt_path)
    needed = {"pop_id", "tt_baseline_any", "tt_baseline_comprehensive"}
    if needed - set(tt.columns):
        _log("WARNING: travel_times_wide missing required columns; returning empty.")
        return {}

    if population_path is None:
        return {}
    pop_path = Path(population_path)
    if not pop_path.exists():
        return {}

    pop = pd.read_pickle(pop_path)
    sample = (
        pop.sample(min(sample_size, len(pop)), random_state=sample_seed)
           .reset_index(drop=True)[["ID", "lat", "lon", "household_count"]]
    )

    tt_cols = [c for c in tt.columns if c.startswith("tt_")]
    merged = sample.merge(
        tt[["pop_id"] + tt_cols].rename(columns={"pop_id": "ID"}),
        on="ID",
        how="left",
    )
    merged = merged.dropna(subset=["tt_baseline_any", "tt_baseline_comprehensive"])

    weights = merged["household_count"].astype(float).to_numpy()
    ref_max = float(np.quantile(weights[weights > 0], 0.99)) if (weights > 0).any() else 1.0

    result = {}
    for scen in SCENARIOS:
        for tier in TIERS:
            col_scen = f"tt_{scen}_{tier}"
            col_base = f"tt_baseline_{tier}"
            if col_scen not in merged.columns:
                continue

            tt_base = merged[col_base].to_numpy()
            tt_scen = merged[col_scen].to_numpy()
            delta = tt_scen - tt_base
            newly_unreach = pd.isna(tt_scen) & pd.notna(tt_base)

            features = []
            for i in range(len(merged)):
                row_lat = float(merged["lat"].iat[i])
                row_lon = float(merged["lon"].iat[i])
                row_w = float(merged["household_count"].iat[i])
                if not (np.isfinite(row_lat) and np.isfinite(row_lon)):
                    continue

                if newly_unreach[i]:
                    color = UNREACHABLE_COLOR
                    delta_text = "unreachable"
                elif np.isnan(delta[i]):
                    continue
                else:
                    color = _color_for_delta(delta[i])
                    delta_text = f"{delta[i]:+.0f} min"

                radius = _radius_for_weight(row_w, ref_max)
                tooltip = (
                    f"Scenario {SCENARIO_LABELS[scen]} / {TIER_LABELS[tier]}<br>"
                    f"Pop ID: {int(merged['ID'].iat[i])}<br>"
                    f"Δ travel time: <b>{delta_text}</b><br>"
                    f"Households here: {int(row_w):,}"
                )

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row_lon, row_lat],
                    },
                    "properties": {
                        "radius": radius,
                        "color": color,
                        "fillOpacity": 0.85,
                        "tooltip": tooltip,
                    },
                })

            result[(scen, tier)] = {"type": "FeatureCollection", "features": features}
            _log(f"  GeoJSON {scen}/{tier}: {len(features)} features")

    return result


def add_disrupted_scenario_layers(
    m: folium.Map,
    travel_times_wide_path: str | Path,
    population_path: str | Path | None = None,
    sample_size: int = SAMPLE_SIZE,
    sample_seed: int = SAMPLE_SEED,
) -> bool:
    """Add 6 toggleable per-point scenario × tier layers + a legend.

    The sample is reproduced from population.pkl with the same seed
    build_population_map uses, so the colored markers visually coincide
    with the blue interactive population dots.

    Returns True if layers were added, False if any required input is missing.
    """
    tt_path = Path(travel_times_wide_path)
    if not tt_path.exists():
        _log(f"WARNING: {tt_path} not found; disrupted-scenario layers will be skipped.")
        return False

    _log(f"loading {tt_path.name}")
    tt = pd.read_pickle(tt_path)
    needed = {"pop_id", "tt_baseline_any", "tt_baseline_comprehensive"}
    missing = needed - set(tt.columns)
    if missing:
        _log(f"WARNING: travel_times_wide missing columns {missing}; skipping")
        return False

    # Reproduce build_population_map's blue-dot sample.
    if population_path is None:
        _log("population_path not provided; cannot reproduce blue-dot sample")
        return False
    pop_path = Path(population_path)
    if not pop_path.exists():
        _log(f"WARNING: {pop_path} not found; disrupted-scenario layers will be skipped.")
        return False

    _log(f"sampling {sample_size} rows from {pop_path.name} (seed={sample_seed})")
    pop = pd.read_pickle(pop_path)
    sample = (
        pop.sample(min(sample_size, len(pop)), random_state=sample_seed)
           .reset_index(drop=True)[["ID", "lat", "lon", "household_count"]]
    )

    # Join the sample with travel_times_wide on ID == pop_id so each blue dot
    # gets all 8 tt_* columns.
    tt_cols = [c for c in tt.columns if c.startswith("tt_")]
    merged = sample.merge(
        tt[["pop_id"] + tt_cols].rename(columns={"pop_id": "ID"}),
        on="ID",
        how="left",
    )
    n_missing = merged["tt_baseline_any"].isna().sum()
    if n_missing:
        _log(f"  WARNING: {n_missing} sampled rows had no match in travel_times_wide")
        merged = merged.dropna(subset=["tt_baseline_any", "tt_baseline_comprehensive"])
    _log(f"  rendering {len(merged)} per-point markers per layer")

    # Reference for radius scaling: 99th-percentile household_count across the
    # sample, so the typical point gets a modest radius and dense pockets pop.
    weights = merged["household_count"].astype(float).to_numpy()
    if (weights > 0).any():
        ref_max = float(np.quantile(weights[weights > 0], 0.99))
    else:
        ref_max = 1.0

    for scen in SCENARIOS:
        for tier in TIERS:
            col_scen = f"tt_{scen}_{tier}"
            col_base = f"tt_baseline_{tier}"
            if col_scen not in merged.columns:
                _log(f"skipping {col_scen}: column not found")
                continue

            tt_base = merged[col_base].to_numpy()
            tt_scen = merged[col_scen].to_numpy()
            delta = tt_scen - tt_base
            newly_unreach = pd.isna(tt_scen) & pd.notna(tt_base)

            n_unreach = int(newly_unreach.sum())
            with np.errstate(invalid="ignore"):
                med = float(np.nanmedian(delta)) if np.isfinite(delta).any() else float("nan")
            _log(f"  {scen}/{tier}: median Δ={med:.0f} min, "
                 f"{n_unreach}/{len(merged)} newly unreachable")

            layer_name = (
                f"Δ travel: scenario {SCENARIO_LABELS[scen]} — {TIER_LABELS[tier]}"
            )
            layer = folium.FeatureGroup(name=layer_name, show=False)

            for i in range(len(merged)):
                row_lat = float(merged["lat"].iat[i])
                row_lon = float(merged["lon"].iat[i])
                row_w   = float(merged["household_count"].iat[i])
                if not (np.isfinite(row_lat) and np.isfinite(row_lon)):
                    continue

                if newly_unreach[i]:
                    color = UNREACHABLE_COLOR
                    delta_text = "unreachable"
                elif np.isnan(delta[i]):
                    continue
                else:
                    color = _color_for_delta(delta[i])
                    delta_text = f"{delta[i]:+.0f} min"

                radius = _radius_for_weight(row_w, ref_max)
                tooltip = (
                    f"Scenario {SCENARIO_LABELS[scen]} / {TIER_LABELS[tier]}<br>"
                    f"Pop ID: {int(merged['ID'].iat[i])}<br>"
                    f"Δ travel time: <b>{delta_text}</b><br>"
                    f"Households here: {int(row_w):,}"
                )

                folium.CircleMarker(
                    location=[row_lat, row_lon],
                    radius=radius,
                    color=color,
                    weight=0.4,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    tooltip=tooltip,
                ).add_to(layer)
            layer.add_to(m)

    add_disruption_chrome(m)
    return True


def add_disruption_chrome(m: folium.Map) -> None:
    """Add the Δ-colour legend and the mutual-exclusion toggle behaviour.

    Kept separate from the layer geometry so server mode (which adds the
    coloured points lazily via placeholders + fetch) can still get the legend
    and the one-scenario-at-a-time toggling.
    """
    _add_legend(m)
    _add_mutual_exclusion_js(m)


def _add_mutual_exclusion_js(m: folium.Map) -> None:
    """Inject JS so only one Δ-travel scenario layer can be on at a time.

    When a Δ-travel overlay is toggled on, any other Δ-travel overlay that is
    currently on is removed — and, because Leaflet's LayerControl keeps its
    checkboxes in sync with the map's layers, the previous checkbox unticks
    itself. This prevents the coloured population points from stacking across
    scenarios. The heatmap, hospital, population, and critical-edge overlays are
    unaffected and remain independently toggleable.
    """
    map_name = m.get_name()
    js = f"""
<script>
(function() {{
    function isDelta(name) {{
        // Layer names start with "\\u0394 travel:" (Δ travel:).
        return typeof name === "string" && name.indexOf("\\u0394 travel:") === 0;
    }}
    window.addEventListener('load', function() {{
        var map = {map_name};
        if (!map) return;
        var activeDelta = null;
        map.on('overlayadd', function(e) {{
            if (!isDelta(e.name)) return;
            var prev = activeDelta;
            activeDelta = e.layer;
            if (prev && prev !== e.layer) {{
                // Defer removal until Leaflet finishes processing this click.
                // L.Control.Layers adds every checked overlay in one loop, so
                // removing the previous layer synchronously here would let that
                // same loop re-add it — and the coloured points would stack.
                setTimeout(function() {{
                    if (map.hasLayer(prev)) map.removeLayer(prev);
                }}, 0);
            }}
        }});
        map.on('overlayremove', function(e) {{
            if (!isDelta(e.name)) return;
            if (e.layer === activeDelta) activeDelta = null;
        }});
    }});
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(js))


def _add_legend(m: folium.Map) -> None:
    rows = []
    for _upper, color, label in DELTA_BINS:
        rows.append(
            f'<div style="display:flex;align-items:center;margin:3px 0;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'border-radius:50%;background:{color};margin-right:8px;'
            f'border:1px solid rgba(0,0,0,0.25);"></span>'
            f'<span style="font-size:12px;">{label}</span>'
            f'</div>'
        )
    rows.append(
        f'<div style="display:flex;align-items:center;margin:3px 0;">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'border-radius:50%;background:{UNREACHABLE_COLOR};margin-right:8px;'
        f'border:1px solid rgba(0,0,0,0.4);"></span>'
        f'<span style="font-size:12px;">unreachable</span>'
        f'</div>'
    )

    html = f"""
    <div id="disrupted-layer-legend" style="
        position: fixed;
        top: 24px;
        left: 24px;
        background: rgba(255,255,255,0.95);
        padding: 10px 14px;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
        z-index: 9999;
        font-family: Arial, sans-serif;
        max-width: 280px;
    ">
        <div onclick="var b=document.getElementById('disrupted-layer-legend-body');var open=b.style.display!=='none';b.style.display=open?'none':'block';this.querySelector('.legend-caret').textContent=open?'▸':'▾';"
             style="font-weight:600; font-size:13px; cursor:pointer; user-select:none; display:flex; align-items:center; justify-content:space-between;">
            <span>Δ travel time vs baseline (per population point)</span>
            <span class="legend-caret" style="margin-left:10px; color:#888;">▾</span>
        </div>
        <div id="disrupted-layer-legend-body" style="margin-top:6px;">
            {''.join(rows)}
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))
