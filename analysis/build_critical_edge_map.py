"""
Critical-edge map overlay.

Reads edge_centrality.pkl from Step 3 (DWEBC) and renders the top 1% / 5% / 10%
of edges on a Folium map. The three tiers correspond directly to the disruption
scenarios A / B / C from Step 4.

Layers are rendered EXCLUSIVELY so the colors layer cleanly:
    * Red       = top 1%        (Scenario A)
    * Orange    = top 1-5%      (added in Scenario B)
    * Yellow    = top 5-10%     (added in Scenario C)

Each tier is a single folium GeoJson layer (one batched object, not thousands
of individual PolyLine objects) so the HTML stays manageable with ~27k edges.

Hospital markers from the existing pipeline are reused for spatial reference.

Usage:
    python -m analysis.build_critical_edge_map
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import folium
import pandas as pd

from analysis import config

# Reuse the hospital icon helper + stroke-facility loader from the existing
# build script so the visual style matches population_map.html.
sys.path.insert(0, str(config.ROOT))
from build_population_map import (  # noqa: E402
    build_hospital_icon_html,
    load_pickle,
    load_stroke_facilities_csv,
)


TIER_DEFS = [
    # (label, lower_quantile, upper_quantile, color, weight_px, opacity, z_layer_order)
    ("Top 5-10% (Scenario C only)", 0.90, 0.95, "#fee08b", 1.2, 0.55, 1),
    ("Top 1-5% (Scenario B adds)",  0.95, 0.99, "#fc8d59", 1.8, 0.75, 2),
    ("Top 1% (Scenario A)",          0.99, 1.00, "#d73027", 2.8, 0.90, 3),
]


def _log(msg: str) -> None:
    print(f"[crit-map] {msg}", flush=True)


def edges_to_geojson(edges: pd.DataFrame) -> dict:
    """Convert an edge DataFrame to a GeoJSON FeatureCollection of LineStrings."""
    features = []
    for row in edges.itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(row.u_lon), float(row.u_lat)],
                    [float(row.v_lon), float(row.v_lat)],
                ],
            },
            "properties": {
                "centrality": float(row.centrality),
                "length_km": float(row.length_km) if pd.notna(row.length_km) else None,
                "travel_time_min": float(row.travel_time_min) if pd.notna(row.travel_time_min) else None,
                "rtt": int(row.rtt) if pd.notna(row.rtt) else None,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def add_critical_edge_layers(m: folium.Map, edge_df: pd.DataFrame) -> dict:
    """
    Add one FeatureGroup per tier to the map. Returns per-tier counts and
    centrality thresholds for the summary text.
    """
    summary = {}
    total = len(edge_df)
    edge_df = edge_df.sort_values("centrality", ascending=False).reset_index(drop=True)

    # Pre-compute rank-based slices so the tiers are deterministic and disjoint.
    rank = edge_df.index  # 0 = most critical
    for label, lo_q, hi_q, color, weight, opacity, _z in TIER_DEFS:
        # rank cut: top (1 - lo_q) fraction is the union of higher tiers.
        # We want the slice between rank quantiles [1-hi_q, 1-lo_q).
        lo_rank = int(round((1.0 - hi_q) * total))
        hi_rank = int(round((1.0 - lo_q) * total))
        slice_df = edge_df.iloc[lo_rank:hi_rank]
        thr_top = float(slice_df["centrality"].max()) if len(slice_df) else float("nan")
        thr_bot = float(slice_df["centrality"].min()) if len(slice_df) else float("nan")
        _log(f"{label}: {len(slice_df)} edges  "
             f"C(e) in [{thr_bot:.0f}, {thr_top:.0f}]")

        if len(slice_df) == 0:
            continue

        gj = edges_to_geojson(slice_df)

        layer = folium.FeatureGroup(name=label, show=False)
        folium.GeoJson(
            data=gj,
            name=label,
            style_function=lambda feat, _c=color, _w=weight, _o=opacity: {
                "color": _c,
                "weight": _w,
                "opacity": _o,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["centrality", "length_km", "travel_time_min", "rtt"],
                aliases=["C(e):", "Length (km):", "Travel time (min):", "RTT class:"],
                localize=True,
                sticky=False,
            ),
        ).add_to(layer)
        layer.add_to(m)

        summary[label] = {
            "n_edges": len(slice_df),
            "thr_min": thr_bot,
            "thr_max": thr_top,
            "color": color,
        }

    return summary


def add_hospital_reference_layer(m: folium.Map) -> None:
    """Plot all 130 hospitals as small, non-clustered reference markers."""
    hospitals = load_pickle(str(config.HOSPITALS_PKL))
    if hospitals.index.name == "ID":
        hospitals = hospitals.reset_index(drop=True)
    hospitals = hospitals.loc[:, ~hospitals.columns.duplicated()].copy()
    hospitals["ID"] = pd.to_numeric(hospitals["ID"], errors="coerce").astype("Int64")
    hospitals = hospitals.dropna(subset=["ID", "Latitude", "Longitude"]).copy()

    stroke = load_stroke_facilities_csv(str(config.STROKE_FACS_CSV))
    hospitals = hospitals.merge(stroke, on="ID", how="left")
    hospitals["specialized_equipment"] = hospitals["specialized_equipment"].fillna(False).astype(bool)

    layer = folium.FeatureGroup(name="Hospitals (reference)", show=True)
    for _, r in hospitals.iterrows():
        icon_html = build_hospital_icon_html(
            has_specialized=bool(r["specialized_equipment"]),
            size_px=18,
        )
        folium.Marker(
            location=[float(r["Latitude"]), float(r["Longitude"])],
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(18, 18),
                icon_anchor=(9, 9),
                class_name="empty",
            ),
            tooltip=f"Hospital {int(r['ID'])} "
                    f"{'(comprehensive)' if r['specialized_equipment'] else '(primary)'}",
        ).add_to(layer)
    layer.add_to(m)
    _log(f"hospital reference layer: {len(hospitals)} markers")


def add_legend(m: folium.Map, summary: dict) -> None:
    """A small fixed-position HTML legend in the bottom-right."""
    rows = []
    for label, info in summary.items():
        # Show only the tier ("Top 5-10%"), dropping the "(Scenario …)" suffix
        # and the edge-count / centrality-threshold detail.
        display_label = label.split(" (")[0]
        rows.append(
            f'<div style="display:flex; align-items:center; margin:4px 0;">'
            f'<span style="display:inline-block; width:22px; height:4px; '
            f'background:{info["color"]}; margin-right:8px;"></span>'
            f'<span style="font-size:12px;">{display_label}</span>'
            f'</div>'
        )

    legend_html = f"""
    <div id="critical-edge-legend" style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        background: rgba(255,255,255,0.95);
        padding: 10px 14px;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
        z-index: 9999;
        font-family: Arial, sans-serif;
        max-width: 360px;
    ">
        <div onclick="var b=document.getElementById('critical-edge-legend-body');var open=b.style.display!=='none';b.style.display=open?'none':'block';this.querySelector('.legend-caret').textContent=open?'▸':'▾';"
             style="font-weight:600; font-size:13px; cursor:pointer; user-select:none; display:flex; align-items:center; justify-content:space-between;">
            <span>Edges Ranked by Demand-Weighted Edge Betweenness Centrality</span>
            <span class="legend-caret" style="margin-left:10px; color:#888;">▾</span>
        </div>
        <div id="critical-edge-legend-body" style="margin-top:6px;">
            {''.join(rows)}
            <div style="font-size:11px; color:#555; margin-top:6px;">
                Tiers correspond to disruption Scenarios A (1%), B (5%), C (10%).
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render top 1/5/10% critical edges on a Folium map")
    p.add_argument("--edge-centrality",
                   default=str(config.OUTPUT_DIR / "edge_centrality.pkl"))
    p.add_argument("--output",
                   default=str(config.ROOT / "critical_edges_map.html"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("=" * 70)
    print("Critical-edge map overlay")
    print("=" * 70)

    _log(f"loading edge centrality: {args.edge_centrality}")
    edge_df = pd.read_pickle(args.edge_centrality)
    _log(f"  {len(edge_df)} edges; C(e) range [{edge_df['centrality'].min():.0f}, "
         f"{edge_df['centrality'].max():.0f}]")

    # Approximate Vietnam center.
    m = folium.Map(location=[16.0, 106.5], zoom_start=6, tiles="CartoDB positron")

    summary = add_critical_edge_layers(m, edge_df)
    add_hospital_reference_layer(m)
    add_legend(m, summary)

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = Path(args.output)
    m.save(str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    _log(f"saved {out_path} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
