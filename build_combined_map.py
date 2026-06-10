"""
Combined map builder.

Produces a single HTML file containing BOTH the interactive population /
hospital / routing map (from build_population_map.py) AND the critical-edge
overlay tiers (from analysis/build_critical_edge_map.py).

All layers are toggleable through the LayerControl in the upper-right corner:
  - Population heatmap
  - Hospitals (clustered, clickable)
  - Population points (clustered, clickable, route drawing)
  - Top 1% / Top 1-5% / Top 5-10% critical road edges
  - Δ travel scenario layers (mutually exclusive; one at a time)

If edge_centrality.pkl is missing the script logs a warning and still produces
the population-only map.

Two output modes:
  * default (inline): everything embedded in one self-contained HTML file.
    Works from file:// with no server. Larger file, more browser RAM.
  * --server-mode: a lightweight HTML plus a data/ folder of static JSON files
    that are fetched on demand (heatmap on load; routes on click; disruption
    layers on toggle). Much lower RAM; must be served over HTTP (Apache2,
    `python -m http.server`, ...). The delta colour legend and the
    one-scenario-at-a-time toggle behaviour are present in BOTH modes.

Usage:
    python build_combined_map.py                 # inline, self-contained
    python build_combined_map.py --server-mode   # lightweight + data/ folder
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import folium
import pandas as pd

# Re-use the orchestration in build_population_map without modifying it.
# We monkey-patch folium.Map.save + folium.LayerControl while it runs so we
# can grab the map object and add critical-edge layers on top before saving.
import build_population_map as bpm
from analysis import config
from analysis.build_critical_edge_map import (
    add_critical_edge_layers,
    add_legend,
)
from analysis.disrupted_layers import (
    add_disrupted_scenario_layers,
    add_disruption_chrome,
    build_disruption_geojson,
)


def _log(msg: str) -> None:
    print(f"[combined-map] {msg}", flush=True)


class _NoOpLayerControl:
    """Stand-in for folium.LayerControl during the population build so we can
    add our own at the very end (after critical-edge layers join the map)."""

    def __init__(self, *args, **kwargs):
        pass

    def add_to(self, _m):
        return self


def _run_population_build_capture_map(build_kwargs: dict, _data_out: dict | None = None) -> folium.Map:
    """Call build_population_map.build_population_map() but capture the map
    object instead of saving, and suppress its LayerControl so we can add a
    single one after all layers (including critical edges) are present."""
    captured: dict = {}

    real_save = folium.Map.save
    real_layer_control = folium.LayerControl

    def capture_save(self, *args, **kwargs):  # noqa: ARG001
        captured["map"] = self

    folium.Map.save = capture_save
    folium.LayerControl = _NoOpLayerControl
    try:
        bpm.build_population_map(**build_kwargs, _data_out=_data_out)
    finally:
        folium.Map.save = real_save
        folium.LayerControl = real_layer_control

    if "map" not in captured:
        raise RuntimeError("Population build did not produce a map object.")

    return captured["map"]


def _write_json(path: Path, data) -> None:
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, separators=(",", ":"))


def _add_disruption_placeholders(m: folium.Map) -> None:
    """Add empty (off-by-default) FeatureGroups as placeholders for the
    lazily-loaded disruption layers, so their LayerControl checkboxes appear."""
    from analysis.disrupted_layers import SCENARIOS, SCENARIO_LABELS, TIERS, TIER_LABELS
    for scen in SCENARIOS:
        for tier in TIERS:
            layer_name = f"Δ travel: scenario {SCENARIO_LABELS[scen]} — {TIER_LABELS[tier]}"
            folium.FeatureGroup(name=layer_name, show=False).add_to(m)


def _inject_lazy_layer_js(m: folium.Map, data_dir_name: str = "data") -> None:
    """Inject JavaScript that fetches static JSON files when a layer is toggled.

    All paths are relative to the HTML file, so a plain static file server
    serves them with no server-side code. This handles the disruption layers;
    the heatmap loader lives in
    build_population_map._add_lazy_heatmap, and per-point routes are fetched
    by map_ui.py via window._ROUTES_URL_TPL (set below).
    """
    from analysis.disrupted_layers import SCENARIOS, SCENARIO_LABELS, TIERS, TIER_LABELS

    entries = []
    for scen in SCENARIOS:
        for tier in TIERS:
            layer_name = f"Δ travel: scenario {SCENARIO_LABELS[scen]} — {TIER_LABELS[tier]}"
            url = f"{data_dir_name}/disruption_{scen}_{tier}.json"
            escaped_name = layer_name.replace('"', '\\"')
            entries.append(f'  "{escaped_name}": "{url}"')
    lazy_map_js = "{\n" + ",\n".join(entries) + "\n}"

    routes_url_template = f"{data_dir_name}/routes_"  # + popId + ".json"

    map_name = m.get_name()
    js = f"""
<script>
(function() {{
    var LAZY_URLS = {lazy_map_js};
    var loaded = {{}};

    function addGeoJsonPoints(geojson, layer) {{
        L.geoJSON(geojson, {{
            pointToLayer: function(feature, latlng) {{
                var p = feature.properties;
                var marker = L.circleMarker(latlng, {{
                    radius: p.radius,
                    color: p.color,
                    weight: 0.4,
                    fill: true,
                    fillColor: p.color,
                    fillOpacity: p.fillOpacity || 0.85
                }});
                if (p.tooltip) marker.bindTooltip(p.tooltip, {{sticky: true}});
                return marker;
            }}
        }}).addTo(layer);
    }}

    // Route data: fetched per pop-id from data/routes_<id>.json
    window._ROUTES_URL_TPL = "{routes_url_template}";

    window.addEventListener('load', function() {{
        setTimeout(function() {{
            {map_name}.on('overlayadd', function(e) {{
                if (loaded[e.name]) return;

                var url = LAZY_URLS[e.name];
                if (url) {{
                    loaded[e.name] = true;
                    fetch(url)
                        .then(function(r) {{ return r.json(); }})
                        .then(function(geojson) {{ addGeoJsonPoints(geojson, e.layer); }})
                        .catch(function(err) {{ console.warn('Lazy layer load failed:', e.name, err); }});
                }}
            }});
        }}, 800);
    }});
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(js))


def _add_critical_edges_if_available(m: folium.Map, edge_centrality_path: Path) -> None:
    if not edge_centrality_path.exists():
        _log(
            f"WARNING: {edge_centrality_path} not found. "
            "Critical-edge tiers will be skipped."
        )
        return

    _log(f"loading edge centrality: {edge_centrality_path}")
    edge_df = pd.read_pickle(edge_centrality_path)
    _log(
        f"  {len(edge_df)} edges; C(e) range "
        f"[{edge_df['centrality'].min():.0f}, {edge_df['centrality'].max():.0f}]"
    )

    summary = add_critical_edge_layers(m, edge_df)
    add_legend(m, summary)


def build_combined_map(
    population_path: str,
    hospital_path: str,
    distances_path: str,
    roads_geojson_path: str,
    stroke_facilities_path: str,
    edge_centrality_path: str,
    travel_times_wide_path: str,
    output_path: str,
    max_marker_points: int,
    min_lat: float | None,
    max_lat: float | None,
    server_mode: bool = False,
    _data_out: dict | None = None,
) -> None:
    print("=" * 70)
    print("Combined population + critical-edge map builder")
    print(f"  mode: {'server (lazy data/ files)' if server_mode else 'inline (self-contained)'}")
    print("=" * 70)

    build_kwargs = dict(
        population_path=population_path,
        hospital_path=hospital_path,
        distances_path=distances_path,
        roads_geojson_path=roads_geojson_path,
        stroke_facilities_path=stroke_facilities_path,
        output_path=output_path,
        max_marker_points=max_marker_points,
        min_lat=min_lat,
        max_lat=max_lat,
        server_mode=server_mode,
    )

    internal_data: dict = {}
    m = _run_population_build_capture_map(build_kwargs, _data_out=internal_data)

    _add_critical_edges_if_available(m, Path(edge_centrality_path))

    if server_mode:
        # Disruption layers are loaded lazily from static JSON files.
        # Add empty placeholder FeatureGroups so the LayerControl checkboxes
        # appear, plus the JS that fetches each layer's data on toggle.
        _add_disruption_placeholders(m)

        # Determine data/ directory next to the output HTML.
        data_dir = Path(output_path).parent / "data"
        data_dir.mkdir(exist_ok=True)
        _inject_lazy_layer_js(m, data_dir_name="data")

        # The delta colour legend + one-scenario-at-a-time toggle behaviour are
        # normally added by add_disrupted_scenario_layers (skipped here because
        # the points are lazy), so add them explicitly.
        add_disruption_chrome(m)

        # Write heatmap data.
        heat_data = internal_data.get("heat_data")
        if heat_data is not None:
            _write_json(data_dir / "heatmap.json", heat_data)
            _log(f"wrote heatmap.json ({len(heat_data)} points, "
                 f"{(data_dir / 'heatmap.json').stat().st_size / (1024*1024):.1f} MB) → {data_dir}")

        # Write route data as one JSON file per population point.
        route_data = internal_data.get("route_data", {})
        for pop_id, routes in route_data.items():
            _write_json(data_dir / f"routes_{pop_id}.json", routes)
        _log(f"wrote {len(route_data)} route JSON files → {data_dir}")

        # Write disruption GeoJSON files.
        disruption_data = build_disruption_geojson(
            travel_times_wide_path=Path(travel_times_wide_path),
            population_path=Path(population_path),
        )
        for (scen, tier), geojson in disruption_data.items():
            _write_json(data_dir / f"disruption_{scen}_{tier}.json", geojson)
        _log(f"wrote {len(disruption_data)} disruption GeoJSON files → {data_dir}")

        if _data_out is not None:
            _data_out["route_data"] = route_data
            _data_out["disruption_data"] = disruption_data
    else:
        # Inline: embed the coloured points directly (this also adds the legend
        # and the mutual-exclusion toggle behaviour via add_disruption_chrome).
        add_disrupted_scenario_layers(
            m,
            Path(travel_times_wide_path),
            population_path=Path(population_path),
        )

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    _log(f"saved {output_path} ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a single HTML map combining population/hospital/"
                    "routing layers with critical-edge overlay tiers."
    )
    p.add_argument("--population", default=str(config.POPULATION_PKL))
    p.add_argument("--hospitals",  default=str(config.HOSPITALS_PKL))
    p.add_argument(
        "--distances",
        default=str(config.HOSPITALS_PKL.parent / "distances_osm_max_300km.pkl"),
    )
    p.add_argument("--roads-geojson",     default=str(config.ROADS_GEOJSON))
    p.add_argument("--stroke-facilities", default=str(config.STROKE_FACS_CSV))
    p.add_argument(
        "--edge-centrality",
        default=str(config.OUTPUT_DIR / "edge_centrality.pkl"),
    )
    p.add_argument(
        "--travel-times-wide",
        default=str(config.OUTPUT_DIR / "travel_times_wide.pkl"),
    )
    p.add_argument("--output", default=str(config.ROOT / "combined_map.html"))
    p.add_argument("--max-marker-points", type=int, default=1000)
    p.add_argument("--min-lat", type=float, default=None)
    p.add_argument("--max-lat", type=float, default=None)
    p.add_argument(
        "--server-mode",
        action="store_true",
        help=(
            "Build a lightweight HTML + separate data/ JSON files for lazy loading. "
            "Use this when hosting on Apache2 or any static file server. "
            "The data/ directory is created next to the output HTML."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Ensure imports of build_population_map (sibling file) resolve when this
    # script is launched from a different cwd.
    sys.path.insert(0, str(config.ROOT))

    build_combined_map(
        population_path=args.population,
        hospital_path=args.hospitals,
        distances_path=args.distances,
        roads_geojson_path=args.roads_geojson,
        stroke_facilities_path=args.stroke_facilities,
        edge_centrality_path=args.edge_centrality,
        travel_times_wide_path=args.travel_times_wide,
        output_path=args.output,
        max_marker_points=args.max_marker_points,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        server_mode=args.server_mode,
    )


if __name__ == "__main__":
    main()
