"""
Interactive edge-reinforcement explorer for the combined Folium map.

Two parts:

1. compute_explorer_data()
   Precomputes, for a sequence of reinforcement budgets K along the greedy
   (centrality-ranked) order, how 1-hour stroke-care accessibility recovers.
   For every K snapshot it aggregates the per-population reachability onto a
   0.1-degree grid and records the share of household weight within 60 minutes
   per cell, plus national and regional coverage. It also exports the ordered
   spine edges (E* in descending C(e)) so the front-end can reveal the top-K
   reinforced segments as the slider moves.

2. add_reinforcement_explorer()
   Injects a Leaflet layer (grid choropleth + spine) and a K-slider control
   panel into an existing Folium map, driven entirely client-side from the
   precomputed JSON. Moving the slider recolours the grid and reveals more of
   the reinforced spine in real time.

CLI (precompute + cache the JSON only):
    python -m analysis.reinforcement_explorer
        -> analysis/outputs/reinforcement_explorer.json

The combined-map builder calls add_reinforcement_explorer() with this JSON.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config
from analysis.data_loaders import (
    assign_region,
    load_hospitals_snapped,
    load_or_build_graph,
    load_population_snapped,
)
from analysis.metrics import compute_tier_travel_times
from analysis.run_optimisation import collect_candidate_edge_records
from analysis.disrupted_layers import (
    SAMPLE_SEED,
    SAMPLE_SIZE,
    _radius_for_weight,
)

EXPLORER_JSON = config.OUTPUT_DIR / "reinforcement_explorer.json"

GRID_DEG = 0.1                 # choropleth cell size (degrees)
COVERAGE_CUTOFF_MIN = 60       # 1-hour accessibility threshold
TIER = "any"                   # coverage measured against any stroke centre

# Budget snapshots as fractions of the candidate set E* (greedy order).
SNAPSHOT_FRACTIONS = (
    0.0, 0.0025, 0.01, 0.025, 0.05, 0.10,
    0.20, 0.30, 0.425, 0.50, 0.70, 0.85, 1.0,
)


def _log(msg: str) -> None:
    print(f"[reinf-explorer] {msg}", flush=True)


def compute_explorer_data(edge_centrality_path: Path | None = None,
                          force_rebuild: bool = False) -> dict:
    """Run the greedy snapshot sweep and return the explorer data dict."""
    t0 = time.time()
    edge_centrality_path = edge_centrality_path or (config.OUTPUT_DIR / "edge_centrality.pkl")

    G, node_list, tree = load_or_build_graph(force=force_rebuild)
    pop_df = load_population_snapped(tree, node_list, force=force_rebuild).copy()
    hosp_df = load_hospitals_snapped(tree, node_list, force=force_rebuild)
    pop_df["region"] = assign_region(pop_df, lat_col="lat")

    pop_ids = pop_df["ID"].to_numpy()
    lat = pop_df["lat"].to_numpy(dtype=float)
    lon = pop_df["lon"].to_numpy(dtype=float)
    w = pop_df["household_count"].to_numpy(dtype=float)
    region = pop_df["region"].to_numpy()

    # ----- Assign each population point to a 0.1-degree grid cell -----
    ci = np.floor(lat / GRID_DEG).astype(int)
    cj = np.floor(lon / GRID_DEG).astype(int)
    keys = ci.astype(np.int64) * 100000 + cj  # composite key
    uniq, inv = np.unique(keys, return_inverse=True)
    n_cells = len(uniq)
    cell_clat = (np.floor(uniq / 100000) + 0.5) * GRID_DEG  # not used directly
    cell_ci = (uniq // 100000)
    cell_cj = (uniq % 100000)
    # int division on negatives is fine here (all VN lats/lons positive).
    cell_lat = (cell_ci + 0.5) * GRID_DEG
    cell_lon = (cell_cj + 0.5) * GRID_DEG

    cell_total_w = np.zeros(n_cells, dtype=float)
    np.add.at(cell_total_w, inv, w)
    total_w = float(w.sum())
    _log(f"grid: {n_cells} populated {GRID_DEG}-deg cells over {len(pop_df)} pop points")

    # ----- Candidate spine edges (E*, descending C(e)) -----
    edge_centrality_df = pd.read_pickle(edge_centrality_path)
    if not edge_centrality_df["centrality"].is_monotonic_decreasing:
        edge_centrality_df = (edge_centrality_df
                              .sort_values("centrality", ascending=False)
                              .reset_index(drop=True))
    records, n_candidate, threshold = collect_candidate_edge_records(G, edge_centrality_df)
    n_estar = len(records)

    # Scenario C graph (all candidate edges removed); greedy adds them back.
    G_C = G.copy()
    for u, v, _ in records:
        G_C.remove_edge(u, v)

    # Region masks for the readout.
    masks = {r: (region == r) for r in ("North", "Central", "South")}

    # ----- Per-point sample (mirrors disrupted_layers so dots co-locate) -----
    # Same 1,000-row sample + seed as the Δ-travel layers, taken from
    # population.pkl's pre-snapped lat/lon so the reinforcement dots sit exactly
    # on top of the disruption dots. Coloured by travel time at each K.
    pop_raw = pd.read_pickle(config.POPULATION_PKL)
    sample = (pop_raw.sample(min(SAMPLE_SIZE, len(pop_raw)), random_state=SAMPLE_SEED)
              .reset_index(drop=True)[["ID", "lat", "lon", "household_count"]])
    sample = sample[np.isfinite(sample["lat"]) & np.isfinite(sample["lon"])].reset_index(drop=True)
    sample_ids = sample["ID"].astype(int).to_numpy()
    s_w = sample["household_count"].astype(float).to_numpy()
    s_ref_max = float(np.quantile(s_w[s_w > 0], 0.99)) if (s_w > 0).any() else 1.0
    sample_radius = [round(_radius_for_weight(w, s_ref_max), 2) for w in s_w]
    _log(f"per-point sample: {len(sample)} dots (seed={SAMPLE_SEED}, mirrors Δ layers)")

    # ----- Snapshot sweep (nested greedy: add edges incrementally) -----
    snap_Ks = sorted({min(n_estar, int(round(f * n_estar))) for f in SNAPSHOT_FRACTIONS})
    snapshots = []
    pt_abs = []  # absolute per-point travel time per snapshot (for delta colouring)
    added = 0
    for K in snap_Ks:
        while added < K:
            u, v, data = records[added]
            G_C.add_edge(u, v, **data)
            added += 1
        ttr = compute_tier_travel_times(G_C, pop_df, hosp_df, TIER, f"K{K}")
        tt = ttr.travel_time_min.reindex(pop_ids).to_numpy(dtype=float)
        covered = np.isfinite(tt) & (tt <= COVERAGE_CUTOFF_MIN)

        # Per-point absolute travel time for the sample (delta derived post-loop).
        pt = ttr.travel_time_min.reindex(sample_ids).to_numpy(dtype=float)
        pt_abs.append(pt)

        cov_w = np.zeros(n_cells, dtype=float)
        np.add.at(cov_w, inv, np.where(covered, w, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            cell_frac = np.where(cell_total_w > 0, cov_w / cell_total_w, 0.0)

        nat = float(w[covered].sum() / total_w) if total_w > 0 else 0.0
        reg_cov = {}
        for r, mk in masks.items():
            tw = float(w[mk].sum())
            reg_cov[r] = float(w[mk & covered].sum() / tw) if tw > 0 else 0.0

        snapshots.append({
            "K": int(K),
            "pct": round(K / n_estar, 5),
            "coverage": round(nat, 5),
            "region": {r: round(v, 5) for r, v in reg_cov.items()},
            # per-mille integers keep the JSON small
            "cell_cov": [int(round(x * 1000)) for x in cell_frac],
        })
        _log(f"  K={K:>6} ({K/n_estar:5.1%}): national coverage {nat*100:5.2f}%")

    floor = snapshots[0]["coverage"]
    ceiling = snapshots[-1]["coverage"]

    # ----- Per-point Δ vs baseline (mirrors the disruption Δ layers) -----
    # Baseline = intact network = the last snapshot (all candidate edges restored).
    # delta = tt_at_K - tt_baseline; 0 means fully recovered to baseline access.
    # -1 = unreachable at this K (matches the Δ layer's black "unreachable" dots).
    base_pt = pt_abs[-1]
    for idx, cur in enumerate(pt_abs):
        delta = cur - base_pt
        snapshots[idx]["pt_delta"] = [
            int(max(0, round(delta[j])))
            if (np.isfinite(cur[j]) and np.isfinite(base_pt[j])) else -1
            for j in range(len(cur))
        ]
        # Absolute travel time per point at this K (minutes; -1 = unreachable).
        # Drives the "absolute travel time" and "1-hour coverage" point colourings.
        snapshots[idx]["pt_abs"] = [
            int(round(cur[j])) if np.isfinite(cur[j]) else -1
            for j in range(len(cur))
        ]

    # Static per-point "K to recover": the smallest K at which the point first
    # falls within the 1-hour cutoff. Adding edges only improves access, so
    # coverage is monotonic in K and this first crossing is well defined.
    # -1 means the point is never covered, even with the full budget.
    n_pts = len(base_pt)
    pt_k_recover = [-1] * n_pts
    for idx, cur in enumerate(pt_abs):
        K = snapshots[idx]["K"]
        for j in range(n_pts):
            if pt_k_recover[j] == -1 and np.isfinite(cur[j]) and cur[j] <= COVERAGE_CUTOFF_MIN:
                pt_k_recover[j] = int(K)

    # ----- Ordered spine edge coordinates (flat, rounded) -----
    spine = []
    for u, v, _ in records:
        spine.extend([round(u[1], 4), round(u[0], 4), round(v[1], 4), round(v[0], 4)])

    data = {
        "grid_deg": GRID_DEG,
        "cutoff_min": COVERAGE_CUTOFF_MIN,
        "tier": TIER,
        "n_estar": int(n_estar),
        "floor": floor,
        "ceiling": ceiling,
        "cells": {
            "lat": [round(float(x), 4) for x in cell_lat],
            "lon": [round(float(x), 4) for x in cell_lon],
        },
        "points": {
            "lat": [round(float(x), 4) for x in sample["lat"].to_numpy()],
            "lon": [round(float(x), 4) for x in sample["lon"].to_numpy()],
            "radius": sample_radius,
            "k_recover": pt_k_recover,
        },
        "snapshots": snapshots,
        "spine": spine,
    }
    _log(f"explorer data built in {time.time()-t0:.1f}s "
         f"({len(snapshots)} snapshots, {n_estar} spine edges)")
    return data


def _explorer_js(data: dict, map_name: str) -> str:
    """Build the injected <script> (+ panel) string for a Folium map."""
    data_json = json.dumps(data, separators=(",", ":"))
    js = r"""
<script>
(function() {
  var D = __DATA__;
  function ready() {
    var map = __MAPNAME__;
    if (typeof map === 'undefined' || !map) { setTimeout(ready, 300); return; }

    var renderer = L.canvas({ padding: 0.5 });
    var d = D.grid_deg / 2.0;
    var cells = D.cells;

    // --- Coverage choropleth layer (one rectangle per populated cell) ---
    var covLayer = L.layerGroup();
    var rects = [];
    for (var i = 0; i < cells.lat.length; i++) {
      var la = cells.lat[i], lo = cells.lon[i];
      var rect = L.rectangle([[la - d, lo - d], [la + d, lo + d]], {
        renderer: renderer, stroke: false, fillOpacity: 0.6, fillColor: '#cccccc'
      });
      rects.push(rect); rect.addTo(covLayer);
    }

    // --- Reinforced-spine layer (revealed up to current K) ---
    var spineLayer = L.layerGroup();
    var spineRenderer = L.canvas({ padding: 0.5 });

    // --- Per-point layer (1,000-sample, coloured by travel time at K) ---
    // Dedicated pane above the default overlay pane so the population dots
    // always render ON TOP of the coverage grid (which lives in the overlay
    // pane); otherwise the grid's translucent fill, drawn last depending on
    // toggle order, would tint the dot colours. z=440 keeps them below the
    // scenario Δ dots (scenarioPane=450) so that layering stays consistent.
    if (!map.getPane('reinfPointsPane')) {
      map.createPane('reinfPointsPane');
      map.getPane('reinfPointsPane').style.zIndex = 440;
    }
    var ptLayer = L.layerGroup();
    var ptRenderer = L.canvas({ pane: 'reinfPointsPane', padding: 0.5 });
    var ptMarkers = [];
    if (D.points) {
      for (var pi = 0; pi < D.points.lat.length; pi++) {
        var pm = L.circleMarker([D.points.lat[pi], D.points.lon[pi]], {
          renderer: ptRenderer, radius: D.points.radius[pi], weight: 0.3,
          color: '#333', fillOpacity: 0.85, fillColor: '#cccccc'
        });
        ptMarkers.push(pm); pm.addTo(ptLayer);
      }
    }
    function colorForDelta(d) {
      // Mirrors disrupted_layers.DELTA_BINS exactly (Δ travel time vs baseline).
      if (d < 0) return '#111111';         // unreachable
      if (d <= 5) return '#1a9850';
      if (d <= 30) return '#a6d96a';
      if (d <= 60) return '#fee08b';
      if (d <= 120) return '#fdae61';
      return '#d73027';
    }
    function colorForAbs(t) {
      // Absolute minutes to the nearest centre. Distinct BLUE sequential ramp
      // (darker = farther) so it never reads like the red->green grid/Δ palettes.
      if (t < 0) return '#111111';          // unreachable
      if (t <= 30) return '#eff3ff';
      if (t <= 60) return '#bdd7e7';
      if (t <= 120) return '#6baed6';
      if (t <= 240) return '#3182bd';
      return '#08519c';
    }
    function colorForCov(t) {
      // Binary 1-hour coverage from absolute time + the study cutoff.
      if (t < 0) return '#111111';          // unreachable
      return (t <= D.cutoff_min) ? '#1a9850' : '#cccccc';
    }
    function colorForKRecover(k) {
      // Static: how much budget before this point is first covered.
      // green = cheap/early, red = needs most of the budget, black = never.
      if (k < 0) return '#111111';
      var frac = D.n_estar > 0 ? k / D.n_estar : 0;
      if (frac <= 0)    return '#1a9850';
      if (frac <= 0.05) return '#66bd63';
      if (frac <= 0.15) return '#fee08b';
      if (frac <= 0.40) return '#fdae61';
      return '#d73027';
    }
    var ptMode = 'delta';
    function paintPoints(snap) {
      if (!ptMarkers.length) return;
      var pj;
      if (ptMode === 'krecover') {
        var kr = (D.points && D.points.k_recover) || [];
        for (pj = 0; pj < ptMarkers.length; pj++)
          ptMarkers[pj].setStyle({ fillColor: colorForKRecover(kr[pj]) });
        return;
      }
      var arr, fn;
      if (ptMode === 'abs')      { arr = snap.pt_abs;   fn = colorForAbs; }
      else if (ptMode === 'cov') { arr = snap.pt_abs;   fn = colorForCov; }
      else                       { arr = snap.pt_delta; fn = colorForDelta; }
      if (!arr) return;
      for (pj = 0; pj < ptMarkers.length; pj++)
        ptMarkers[pj].setStyle({ fillColor: fn(arr[pj]) });
    }
    function setPtLegend(mode) {
      var el = document.getElementById('reinf-pt-leg');
      if (!el) return;
      var dot = function(c){ return '<span style="color:' + c + ';">&#9679;</span>'; };
      var html;
      if (mode === 'abs') {
        html = 'points = travel time to nearest centre: '
          + dot('#bdd7e7') + '&le;60 ' + dot('#6baed6') + '&le;120 '
          + dot('#3182bd') + '&le;240 ' + dot('#08519c') + '&gt;240 '
          + dot('#111') + 'unreach.';
      } else if (mode === 'cov') {
        html = 'points = 1-hour coverage: ' + dot('#1a9850') + 'within ' + D.cutoff_min
          + ' min ' + dot('#cccccc') + 'beyond ' + dot('#111') + 'unreachable.';
      } else if (mode === 'krecover') {
        html = 'points = budget to first coverage (static): ' + dot('#1a9850') + 'at K=0 '
          + dot('#66bd63') + '&le;5% ' + dot('#fee08b') + '&le;15% '
          + dot('#fdae61') + '&le;40% E* ' + dot('#d73027') + 'more ' + dot('#111') + 'never.';
      } else {
        html = 'points = &#916; travel time vs baseline (matches scenario layers): '
          + dot('#1a9850') + '&le;5 ' + dot('#a6d96a') + '&le;30 ' + dot('#fee08b') + '&le;60 '
          + dot('#fdae61') + '&le;120 ' + dot('#d73027') + '&gt;120 ' + dot('#111') + 'unreach.';
      }
      el.innerHTML = html;
    }

    function colorFor(frac) {
      // red (low) -> amber -> green (high) accessibility
      var stops = [[0,215,48,39],[0.25,252,141,89],[0.5,254,224,139],
                   [0.75,145,207,96],[1,26,152,80]];
      if (frac <= 0) return 'rgb(215,48,39)';
      if (frac >= 1) return 'rgb(26,152,80)';
      for (var s = 1; s < stops.length; s++) {
        if (frac <= stops[s][0]) {
          var a = stops[s-1], b = stops[s];
          var t = (frac - a[0]) / (b[0] - a[0]);
          var r = Math.round(a[1] + t*(b[1]-a[1]));
          var g = Math.round(a[2] + t*(b[2]-a[2]));
          var bl = Math.round(a[3] + t*(b[3]-a[3]));
          return 'rgb(' + r + ',' + g + ',' + bl + ')';
        }
      }
      return 'rgb(26,152,80)';
    }

    function paint(idx) {
      var snap = D.snapshots[idx];
      var cov = snap.cell_cov;
      for (var i = 0; i < rects.length; i++) {
        rects[i].setStyle({ fillColor: colorFor(cov[i] / 1000) });
      }
      // spine reveal: first K segments from the flat coord array
      spineLayer.clearLayers();
      var K = snap.K, sp = D.spine, segs = [];
      for (var e = 0; e < K; e++) {
        var o = e * 4;
        segs.push([[sp[o], sp[o+1]], [sp[o+2], sp[o+3]]]);
      }
      if (segs.length) {
        L.polyline(segs, { renderer: spineRenderer, color: '#3b0a70',
          weight: 1.1, opacity: 0.75 }).addTo(spineLayer);
      }
      // per-point recolour using the currently selected point-colour mode
      paintPoints(snap);
      // readout
      var r = snap.region;
      document.getElementById('reinf-k').textContent =
        snap.K.toLocaleString() + ' edges (' + (snap.pct*100).toFixed(1) + '% of E*)';
      document.getElementById('reinf-cov').textContent = (snap.coverage*100).toFixed(1) + '%';
      var gap = D.ceiling - D.floor;
      var gc = gap > 0 ? (snap.coverage - D.floor) / gap * 100 : 0;
      document.getElementById('reinf-gap').textContent = gc.toFixed(0) + '%';
      document.getElementById('reinf-n').textContent = (r.North*100).toFixed(0) + '%';
      document.getElementById('reinf-c').textContent = (r.Central*100).toFixed(0) + '%';
      document.getElementById('reinf-s').textContent = (r.South*100).toFixed(0) + '%';
    }

    // --- Control panel ---
    var panel = L.control({ position: 'bottomright' });
    panel.onAdd = function() {
      var div = L.DomUtil.create('div', 'reinf-panel');
      div.innerHTML =
        '<div class="reinf-hd" style="font-weight:600;font-size:13px;cursor:pointer;'
        + 'display:flex;justify-content:space-between;align-items:center;">'
        + '<span>Reinforcement explorer</span><span class="reinf-caret" style="color:#888;margin-left:8px;">&#9662;</span></div>'
        + '<div class="reinf-body" style="margin-top:6px;">'
        + '<div style="font-size:11px;color:#555;margin-bottom:6px;">Drag to reinforce the top-K critical edges and watch 1-hour coverage recover.</div>'
        + '<input id="reinf-slider" type="range" min="0" max="' + (D.snapshots.length-1) + '" value="0" step="1" style="width:100%;">'
        + '<div style="font-size:12px;margin-top:6px;"><b>K:</b> <span id="reinf-k"></span></div>'
        + '<div style="display:flex;gap:10px;margin-top:6px;">'
        + '<div style="flex:1;"><div style="font-size:10px;color:#777;">1-hour coverage</div><div style="font-size:20px;font-weight:600;" id="reinf-cov"></div></div>'
        + '<div style="flex:1;"><div style="font-size:10px;color:#777;">gap closed</div><div style="font-size:20px;font-weight:600;" id="reinf-gap"></div></div></div>'
        + '<div style="font-size:11px;color:#555;margin-top:6px;display:flex;gap:8px;">'
        + '<span>North <b id="reinf-n"></b></span><span>Central <b id="reinf-c"></b></span><span>South <b id="reinf-s"></b></span></div>'
        + '<div style="font-size:11px;margin-top:8px;line-height:1.7;">'
        + '<label style="cursor:pointer;display:inline-flex;align-items:center;gap:4px;vertical-align:middle;"><input type="checkbox" id="reinf-show-cov"> coverage grid</label>&nbsp;&nbsp;'
        + '<label style="cursor:pointer;display:inline-flex;align-items:center;gap:4px;vertical-align:middle;"><input type="checkbox" id="reinf-show-spine"> reinforced spine</label><br>'
        + '<label style="cursor:pointer;display:inline-flex;align-items:center;gap:4px;vertical-align:middle;"><input type="checkbox" id="reinf-show-pts"> population points</label></div>'
        + '<div id="reinf-pt-mode-wrap" style="display:none;margin-top:6px;font-size:11px;color:#555;">point colour: '
        + '<select id="reinf-pt-mode" style="font-size:11px;">'
        + '<option value="delta">&#916; vs baseline</option>'
        + '<option value="abs">absolute travel time</option>'
        + '<option value="cov">1-hour coverage</option>'
        + '<option value="krecover">K to recover (static)</option>'
        + '</select></div>'
        + '<div id="reinf-grid-leg" style="display:none;align-items:center;gap:8px;margin-top:8px;font-size:11px;color:#555;">'
        + '<span>grid: low</span><span style="flex:1;height:8px;border-radius:4px;background:linear-gradient(90deg,#d73027,#fc8d59,#fee08b,#91cf60,#1a9850);"></span><span>high</span></div>'
        + '<div id="reinf-pt-leg" style="display:none;margin-top:6px;font-size:10px;color:#555;">'
        + 'points = &#916; travel time vs baseline (matches scenario layers): '
        + '<span style="color:#1a9850;">&#9679;</span>&le;5 '
        + '<span style="color:#a6d96a;">&#9679;</span>&le;30 '
        + '<span style="color:#fee08b;">&#9679;</span>&le;60 '
        + '<span style="color:#fdae61;">&#9679;</span>&le;120 '
        + '<span style="color:#d73027;">&#9679;</span>&gt;120 '
        + '<span style="color:#111;">&#9679;</span>unreach.<br>'
        + 'at 0% this equals the &#34;&#916; scenario C&#34; layer; greens toward baseline as K rises.</div>'
        + '</div>';
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      return div;
    };
    panel.addTo(map);

    // Coverage grid and spine start OFF; the user enables them via the
    // checkboxes. paint(0) still styles them so they look right when shown.
    paint(0);

    document.getElementById('reinf-slider').addEventListener('input', function() {
      paint(parseInt(this.value, 10));
    });
    document.getElementById('reinf-show-cov').addEventListener('change', function() {
      if (this.checked) covLayer.addTo(map); else map.removeLayer(covLayer);
      document.getElementById('reinf-grid-leg').style.display = this.checked ? 'flex' : 'none';
    });
    document.getElementById('reinf-show-spine').addEventListener('change', function() {
      if (this.checked) spineLayer.addTo(map); else map.removeLayer(spineLayer);
    });
    var ptsBox = document.getElementById('reinf-show-pts');
    if (ptsBox) ptsBox.addEventListener('change', function() {
      if (this.checked) ptLayer.addTo(map); else map.removeLayer(ptLayer);
      document.getElementById('reinf-pt-leg').style.display = this.checked ? 'block' : 'none';
      var mw = document.getElementById('reinf-pt-mode-wrap');
      if (mw) mw.style.display = this.checked ? 'block' : 'none';
      if (this.checked) {
        setPtLegend(ptMode);
        paintPoints(D.snapshots[parseInt(document.getElementById('reinf-slider').value, 10)]);
      }
    });
    var modeSel = document.getElementById('reinf-pt-mode');
    if (modeSel) modeSel.addEventListener('change', function() {
      ptMode = this.value;
      setPtLegend(ptMode);
      paintPoints(D.snapshots[parseInt(document.getElementById('reinf-slider').value, 10)]);
    });
    var hd = document.querySelector('.reinf-hd');
    if (hd) hd.addEventListener('click', function() {
      var b = document.querySelector('.reinf-body');
      var open = b.style.display !== 'none';
      b.style.display = open ? 'none' : 'block';
      document.querySelector('.reinf-caret').innerHTML = open ? '&#9656;' : '&#9662;';
    });
  }
  if (document.readyState === 'complete') ready();
  else window.addEventListener('load', function() { setTimeout(ready, 600); });
})();
</script>
<style>
.reinf-panel { background: rgba(255,255,255,0.96); padding: 10px 14px;
  border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,0.18);
  font-family: Arial, sans-serif; max-width: 260px; }
</style>
"""
    return js.replace("__DATA__", data_json).replace("__MAPNAME__", map_name)


def add_reinforcement_explorer(m, data: dict | None = None,
                               data_path: Path | None = None) -> bool:
    """
    Inject the reinforcement explorer (grid + spine + K slider) into a Folium
    map. Provide either a precomputed `data` dict or a `data_path` JSON file.
    Returns True if injected, False if data was unavailable.
    """
    import folium

    if data is None:
        data_path = data_path or EXPLORER_JSON
        if not Path(data_path).exists():
            _log(f"WARNING: {data_path} not found; reinforcement explorer skipped. "
                 f"Run `python -m analysis.reinforcement_explorer` first.")
            return False
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    js = _explorer_js(data, m.get_name())
    m.get_root().html.add_child(folium.Element(js))
    _log(f"injected reinforcement explorer ({len(data['snapshots'])} snapshots)")
    return True


def main() -> None:
    import argparse
    import sys
    # Force UTF-8 stdout/stderr so log lines containing non-ASCII (e.g. the "Δ"
    # in "mirrors Δ layers") don't crash on Windows consoles that default to a
    # legacy codepage when output is piped/redirected.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description="Precompute the reinforcement-explorer JSON")
    p.add_argument("--edge-centrality",
                   default=str(config.OUTPUT_DIR / "edge_centrality.pkl"))
    p.add_argument("--output", default=str(EXPLORER_JSON))
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()

    data = compute_explorer_data(Path(args.edge_centrality), force_rebuild=args.force_rebuild)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    _log(f"wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
