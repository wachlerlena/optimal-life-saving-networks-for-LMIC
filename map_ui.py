# =============================================================================
# MAP UI - SIDE PANEL, BUTTONS, AND BROWSER INTERACTIONS
# =============================================================================
# Purpose of this file:
# This file contains the HTML, CSS, and JavaScript that is injected into the
# Folium map. The main Python file prepares the data. This file decides how that
# data is shown in the browser after the user clicks a marker.
#
# Why keep this separate?
# - build_population_map.py remains focused on data and route modelling.
# - map_ui.py remains focused on visual design and click behavior.
# - There are still only two Python files, so the project stays simple.
#
# The JavaScript in this file handles:
# - opening the side panel
# - drawing route lines on the Leaflet/Folium map
# - showing hospital details
# =============================================================================

import json

import folium

# -----------------------------------------------------------------------------
# Inject the side panel and browser logic into the Folium map.
# -----------------------------------------------------------------------------
def inject_side_panel_and_routes(
    map_obj,
    population_panel_data,
    population_marker_lookup,
    hospital_panel_data,
    hospital_marker_lookup,
    facility_meta=None,
    assign_dist=None,
    assign_path=None,
    greenfield_scenarios=None,
    time_label="at 40 km/h",
):
    """
    Inject the right-side panel and the JavaScript needed for:
    - clicking a population marker to see the advanced facility it is assigned to
      *under the currently selected scenario* (existing advanced hospitals plus
      any upgrades / greenfield sites for the chosen budget and radius),
    - the road distance AND estimated travel time to that facility,
    - drawing the assignment line, which updates live when budget/radius change,
    - clicking hospital markers to see hospital details.

    facility_meta : {str(fac_id): {"lat","lon","kind","name"}}
    assign_dist   : {str(pop_id): {str(fac_id): [dist_km, time_min]}}

    If greenfield_scenarios is provided, also inject a top-left scenario
    explorer (budget slider + radius dropdown) that overlays the optimal
    upgrade / greenfield sites for the selected (budget, radius).
    """
    map_name = map_obj.get_name()

    facility_meta = facility_meta or {}
    assign_dist = assign_dist or {}
    assign_path = assign_path or {}

    population_panel_json = json.dumps(population_panel_data)
    population_marker_lookup_json = json.dumps(population_marker_lookup)
    hospital_panel_json = json.dumps(hospital_panel_data)
    hospital_marker_lookup_json = json.dumps(hospital_marker_lookup)
    facility_meta_json = json.dumps(facility_meta)
    assign_dist_json = json.dumps(assign_dist)
    assign_path_json = json.dumps(assign_path)
    time_label_json = json.dumps(time_label)

    html = f"""
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

        .chip-adv {{ background: #dff5e3; color: #1c6b2a; }}
        .chip-upg {{ background: #e2eefb; color: #14508c; }}
        .chip-gf  {{ background: #efe4fb; color: #5b2a91; }}

        .scenario-note {{
            font-size: 12px;
            color: #555;
            background: #f3f5f8;
            border-radius: 8px;
            padding: 7px 10px;
            margin-bottom: 12px;
        }}
        .scenario-note b {{ color: #222; }}

        .assigned-card {{
            border: 1px solid #cfe3d4;
            border-left: 5px solid #1a9e5f;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 12px;
            background: #f7fcf9;
        }}

        .cov-badge {{
            display: inline-block;
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }}
        .cov-in  {{ background: #dff5e3; color: #1c6b2a; }}
        .cov-out {{ background: #fdeBd6; color: #9a5413; }}

        .alt-title {{
            font-size: 13px;
            font-weight: 700;
            color: #444;
            margin: 6px 0 6px;
        }}
        .alt-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: #333;
            padding: 6px 0;
            border-top: 1px solid #f0f0f0;
        }}
        .alt-row .alt-name {{ flex: 1; }}
        .alt-row .alt-metric {{ font-variant-numeric: tabular-nums; color: #555; white-space: nowrap; }}
        .alt-btn {{
            border: none; border-radius: 7px; padding: 5px 9px; cursor: pointer;
            font-size: 12px; font-weight: 700; background: #eef1f6; color: #1f3a5f;
        }}
        .alt-btn:hover {{ background: #dde4ee; }}

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
        var populationPanelData = {population_panel_json};
        var populationMarkerLookup = {population_marker_lookup_json};
        var hospitalPanelData = {hospital_panel_json};
        var hospitalMarkerLookup = {hospital_marker_lookup_json};
        var facilityMeta = {facility_meta_json};   // {{facId: {{lat, lon, kind, name}}}}
        var timeModelLabel = {time_label_json};
        var assignDist = {assign_dist_json};        // {{popId: {{facId: [dist_km, time_min]}}}}
        var assignPath = {assign_path_json};        // {{popId: {{facId: [[lat,lon],...] actual road route}}}}
        var activeAssignedLine = null;
        var activePopulationId = null;
        var activePanelType = null;                 // "pop" | "hosp" | null

        function clearAssignedLine() {{
            if (activeAssignedLine) {{
                try {{ {map_name}.removeLayer(activeAssignedLine); }} catch (err) {{}}
                activeAssignedLine = null;
            }}
        }}

        // --- scenario state (driven by the top-left budget slider + radius) -----
        function getScenarios() {{
            return (typeof greenfieldScenarios !== "undefined" && greenfieldScenarios)
                ? greenfieldScenarios : null;
        }}

        function getScenarioState() {{
            var bEl = document.getElementById("scenario-budget");
            var rEl = document.getElementById("scenario-radius");
            var sc = getScenarios();
            return {{
                budget: bEl ? String(bEl.value) : "0",
                radius: rEl ? String(rEl.value) : null,
                unit:   sc ? (sc.unit || "km") : "km"
            }};
        }}

        // Existing advanced hospitals are always open; fall back to the hospital
        // layer's "specialized_equipment" flag if no scenario file is present.
        function getExistingAdvancedIds() {{
            var ids = new Set();
            var sc = getScenarios();
            if (sc && sc.existing_advanced) {{
                sc.existing_advanced.forEach(function(r) {{ ids.add(String(r.id)); }});
            }} else {{
                Object.keys(hospitalPanelData).forEach(function(hid) {{
                    if (hospitalPanelData[hid] && hospitalPanelData[hid].specialized_equipment)
                        ids.add(String(hid));
                }});
            }}
            return ids;
        }}

        // Advanced facilities active for a (budget, radius): existing advanced
        // hospitals + the upgrades and greenfield sites chosen in that scenario.
        function getActiveAdvancedIds(budget, radius) {{
            var ids = getExistingAdvancedIds();
            var sc = getScenarios();
            if (sc && sc.scenarios && sc.scenarios[budget] && sc.scenarios[budget][radius]) {{
                var s = sc.scenarios[budget][radius];
                (s.upgrades   || []).forEach(function(r) {{ ids.add(String(r.id)); }});
                (s.greenfield || []).forEach(function(r) {{ ids.add(String(r.id)); }});
            }}
            return ids;
        }}

        // Active advanced facilities reachable from this point, nearest first.
        function rankedFacilitiesFor(popId) {{
            var st = getScenarioState();
            var dists = assignDist[String(popId)];
            if (!dists) return {{ state: st, ranked: [] }};
            var active = getActiveAdvancedIds(st.budget, st.radius);
            var ranked = [];
            Object.keys(dists).forEach(function(fid) {{
                if (active.has(String(fid))) {{
                    ranked.push({{ id: String(fid), km: dists[fid][0], min: dists[fid][1] }});
                }}
            }});
            ranked.sort(function(a, b) {{ return a.km - b.km; }});
            return {{ state: st, ranked: ranked }};
        }}

        function facilityKindChip(kind) {{
            if (kind === "greenfield") return {{ cls: "chip-gf",  text: "New greenfield site" }};
            if (kind === "upgrade")    return {{ cls: "chip-upg", text: "Upgraded hospital" }};
            return {{ cls: "chip-adv", text: "Existing advanced" }};
        }}

        function isCovered(rec, st) {{
            if (!rec || st.radius == null || isNaN(Number(st.radius))) return null;
            var metric = (st.unit === "min") ? rec[1] : rec[0];
            return metric <= Number(st.radius);
        }}

        // --- map lines ----------------------------------------------------------
        function drawFacilityLine(popId, facId) {{
            clearAssignedLine();
            var mv = populationMarkerLookup[String(popId)];
            var fac = facilityMeta[String(facId)];
            if (!mv || !fac) return;
            var p;
            try {{ p = eval(mv).getLatLng(); }} catch (e) {{ return; }}

            var dists = assignDist[String(popId)] || {{}};
            var rec = dists[String(facId)];
            var st = getScenarioState();
            var covered = isCovered(rec, st);
            var color = (covered === false) ? "#e8590c" : "#1a9e5f";

            var roadGeom = (assignPath[String(popId)] || {{}})[String(facId)];
            var latlngs = (roadGeom && roadGeom.length)
                ? [[p.lat, p.lng]].concat(roadGeom).concat([[fac.lat, fac.lon]])
                : [[p.lat, p.lng], [fac.lat, fac.lon]];
            activeAssignedLine = L.polyline(latlngs, {{
                color: color, weight: 4, opacity: 0.9
            }}).addTo({map_name});
        }}

        function drawAssignedLine(popId) {{
            var res = rankedFacilitiesFor(popId);
            if (res.ranked.length === 0) {{ clearAssignedLine(); return; }}
            drawFacilityLine(popId, res.ranked[0].id);
        }}

        function hidePopulationPanel() {{
            clearAssignedLine();
            activePanelType = null;
            var panel = document.getElementById("population-side-panel");
            if (panel) panel.classList.remove("open");
        }}

        function fmtDist(rec) {{ return Number(rec[0]).toFixed(1) + " km"; }}
        function fmtTime(rec) {{
            var m = Number(rec[1]);
            if (m < 1) return "< 1 min";
            if (m < 90) return Math.round(m) + " min";
            var h = Math.floor(m / 60), r = Math.round(m % 60);
            return h + " h " + (r < 10 ? "0" + r : r) + " min";
        }}

        function renderPopulationPanel(popId) {{
            var panel = document.getElementById("population-side-panel");
            var content = document.getElementById("population-side-panel-content");
            var title = document.getElementById("population-side-panel-title");
            if (!panel || !content || !title) return;

            var data = populationPanelData[String(popId)];
            title.innerHTML = "Population point";

            if (!data) {{
                content.innerHTML = '<div class="panel-empty">No data available for this population point.</div>';
                panel.classList.add("open");
                return;
            }}

            var res = rankedFacilitiesFor(popId);
            var st = res.ranked.length ? res.state : getScenarioState();
            var unit = st.unit;

            var html = "";
            html += '<div class="panel-meta">';
            html += '<div><b>Population ID:</b> ' + data.population_id + '</div>';
            html += '<div><b>Household count:</b> ' + data.household_count + '</div>';
            html += '</div>';

            // Which scenario is currently selected.
            if (st.radius != null) {{
                var radLabel = (unit === "min")
                    ? ("Time radius: " + st.radius + " min")
                    : ("Service radius: " + st.radius + " km");
                html += '<div class="scenario-note">Assignment for current scenario — '
                     +  '<b>Budget:</b> ' + st.budget + ' new facilities · <b>' + radLabel + '</b></div>';
            }} else {{
                html += '<div class="scenario-note">Assignment to the nearest existing advanced facility '
                     +  '(no scenario explorer loaded).</div>';
            }}

            if (res.ranked.length === 0) {{
                html += '<div class="panel-empty">No advanced facility reachable within 300 km '
                     +  'under this scenario.</div>';
                content.innerHTML = html;
                panel.classList.add("open");
                return;
            }}

            // Assigned facility = nearest active advanced facility.
            var a = res.ranked[0];
            var fac = facilityMeta[a.id] || {{}};
            var rec = [a.km, a.min];
            var chip = facilityKindChip(fac.kind);
            var covered = isCovered(rec, st);

            html += '<div class="assigned-card">';
            html += '<div class="hospital-card-title">Assigned advanced facility</div>';
            html += '<div class="hospital-card-row">' + (fac.name || ("Facility " + a.id))
                 +  ' <span class="hospital-chip ' + chip.cls + '">' + chip.text + '</span></div>';
            html += '<div class="hospital-card-row"><b>Road distance:</b> ' + fmtDist(rec) + '</div>';
            html += '<div class="hospital-card-row"><b>Estimated travel time:</b> ' + fmtTime(rec)
                 +  ' <span style="color:#888">(' + timeModelLabel + ')</span></div>';
            if (covered !== null) {{
                html += covered
                    ? '<div class="hospital-card-row"><span class="cov-badge cov-in">Within ' + st.radius + ' ' + unit + '</span></div>'
                    : '<div class="hospital-card-row"><span class="cov-badge cov-out">Beyond ' + st.radius + ' ' + unit + '</span></div>';
            }}
            html += '</div>';

            // A few alternative active facilities (next nearest), each clickable.
            if (res.ranked.length > 1) {{
                html += '<div class="alt-title">Other nearby advanced facilities</div>';
                res.ranked.slice(1, 4).forEach(function(f) {{
                    var fm = facilityMeta[f.id] || {{}};
                    var metric = (unit === "min") ? fmtTime([f.km, f.min]) : fmtDist([f.km, f.min]);
                    html += '<div class="alt-row">';
                    html += '<span class="alt-name">' + (fm.name || ("Facility " + f.id)) + '</span>';
                    html += '<span class="alt-metric">' + metric + '</span>';
                    html += '<button class="alt-btn" onclick="drawFacilityLine(\\'' + data.population_id
                         +  '\\', \\'' + f.id + '\\')">Show</button>';
                    html += '</div>';
                }});
            }}

            content.innerHTML = html;
            panel.classList.add("open");
        }}

        function showPopulationPanel(popId) {{
            activePopulationId = popId;
            activePanelType = "pop";
            renderPopulationPanel(popId);
            drawAssignedLine(popId);
        }}

        // Called by the scenario explorer whenever budget/radius changes, so an
        // open population point re-assigns to the new set of advanced facilities.
        function onPopulationScenarioChange() {{
            if (activePanelType !== "pop" || activePopulationId == null) return;
            var panel = document.getElementById("population-side-panel");
            if (!panel || !panel.classList.contains("open")) return;
            renderPopulationPanel(activePopulationId);
            drawAssignedLine(activePopulationId);
        }}

        function showHospitalPanel(hospitalId) {{
            clearAssignedLine();
            activePanelType = "hosp";

            var panel = document.getElementById("population-side-panel");
            var content = document.getElementById("population-side-panel-content");
            var title = document.getElementById("population-side-panel-title");
            if (!panel || !content || !title) return;

            var data = hospitalPanelData[String(hospitalId)];
            title.innerHTML = "Hospital details";

            if (!data) {{
                content.innerHTML = '<div class="panel-empty">No data available for this hospital.</div>';
                panel.classList.add("open");
                return;
            }}

            var chipClass = data.specialized_equipment ? "chip-yes" : "chip-no";
            var chipText = data.specialized_equipment ? "Advanced (thrombolysis and mechanical thrombectomy)" : "Basic (thrombolysis)";

            var hospitalName = data.hospital_name || "";
            var titleText = hospitalName
                ? hospitalName + ' (ID ' + data.hospital_id + ')'
                : 'Hospital ' + data.hospital_id;

            var html = "";
            html += '<div class="hospital-card">';
            html += '<div class="hospital-card-title">' + titleText + '</div>';
            html += '<div class="hospital-card-row"><b>Hospital ID:</b> ' + data.hospital_id + '</div>';
            if (hospitalName) {{
                html += '<div class="hospital-card-row"><b>Name:</b> ' + hospitalName + '</div>';
            }}
            html += '<div class="hospital-card-row"><b>Stroke equipment:</b> <span class="hospital-chip ' + chipClass + '">' + chipText + '</span></div>';
            if (data.address_text) {{
                html += '<div class="hospital-card-row"><b>Address:</b> ' + data.address_text + '</div>';
            }}
            html += '</div>';

            content.innerHTML = html;
            panel.classList.add("open");
        }}

        function bindMarkerClicks() {{
            var populationBoundCount = 0;
            var hospitalBoundCount = 0;

            Object.keys(populationMarkerLookup).forEach(function(popId) {{
                var markerVarName = populationMarkerLookup[popId];
                try {{
                    var markerObj = eval(markerVarName);
                    if (markerObj) {{
                        markerObj.on('click', function() {{ showPopulationPanel(popId); }});
                        populationBoundCount += 1;
                    }}
                }} catch (err) {{
                    console.log("Failed to bind population marker click for popId:", popId, err);
                }}
            }});

            Object.keys(hospitalMarkerLookup).forEach(function(hospitalId) {{
                var markerVarName = hospitalMarkerLookup[hospitalId];
                try {{
                    var markerObj = eval(markerVarName);
                    if (markerObj) {{
                        markerObj.on('click', function() {{ showHospitalPanel(hospitalId); }});
                        hospitalBoundCount += 1;
                    }}
                }} catch (err) {{
                    console.log("Failed to bind hospital marker click for hospitalId:", hospitalId, err);
                }}
            }});

            console.log("Bound population side-panel clicks:", populationBoundCount);
            console.log("Bound hospital side-panel clicks:", hospitalBoundCount);
        }}

        window.drawFacilityLine = drawFacilityLine;
        window.hidePopulationPanel = hidePopulationPanel;
        window.showPopulationPanel = showPopulationPanel;
        window.showHospitalPanel = showHospitalPanel;
        window.onPopulationScenarioChange = onPopulationScenarioChange;

        window.addEventListener('load', function() {{
            setTimeout(bindMarkerClicks, 500);
        }});
    </script>
    """

    map_obj.get_root().html.add_child(folium.Element(html))

    if greenfield_scenarios:
        _inject_scenario_explorer(map_obj, map_name, greenfield_scenarios)


def _inject_scenario_explorer(map_obj, map_name, scenarios):
    """
    Top-left scenario explorer overlay. Reads the pre-computed combined
    upgrade-or-greenfield scenarios and, for the selected budget + radius,
    draws on the Folium/Leaflet map:
      * green circle  = existing hospital to UPGRADE
      * purple square = new greenfield site to BUILD
    plus optional service-radius rings and live coverage / distance stats.
    """
    scenarios_json = json.dumps(scenarios)

    html = f"""
    <style>
        #scenario-panel {{
            position: fixed; top: 16px; left: 16px; width: 270px;
            background: rgba(255,255,255,0.97); border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.18); z-index: 9999;
            padding: 14px 16px; font-family: Arial, sans-serif;
        }}
        #scenario-panel h3 {{ margin: 0 0 10px; font-size: 16px; color: #222; }}
        #scenario-panel .row {{ margin: 9px 0; }}
        #scenario-panel label {{ display:block; font-size:12px; color:#555; margin-bottom:3px; }}
        #scenario-panel input[type=range] {{ width: 100%; }}
        #scenario-panel select {{ width: 100%; padding: 4px; }}
        #scenario-panel .val {{ font-weight: 700; color: #111; }}
        #scenario-cov {{ font-size: 25px; font-weight: 700; color: #1a7f37; margin: 6px 0 0; }}
        #scenario-panel .muted {{ color:#777; font-size:11px; margin-bottom:6px; }}
        #scenario-panel .stat {{ display:flex; justify-content:space-between; font-size:13px; margin:3px 0; }}
        #scenario-panel .stat b {{ font-variant-numeric: tabular-nums; }}
        #scenario-panel hr {{ border:none; border-top:1px solid #eee; margin:10px 0; }}
        #scenario-panel .sw {{ display:inline-block; width:13px; height:13px; margin-right:6px; vertical-align:-2px; }}
        #scenario-panel .sw-up {{ background:#1a9e5f; border-radius:50%; border:2px solid #fff; }}
        #scenario-panel .sw-gf {{ background:#7B2FBE; border:2px solid #fff; transform:rotate(45deg); }}
    </style>

    <div id="scenario-panel">
        <h3>Scenario explorer</h3>
        <div class="row">
            <label>Budget (new facilities): <span class="val" id="scenario-budget-val"></span></label>
            <input type="range" id="scenario-budget" min="0" max="0" step="1" value="0"/>
        </div>
        <div class="row">
            <label id="scenario-radius-label">Service radius</label>
            <select id="scenario-radius"></select>
        </div>
        <div class="row">
            <label><input type="checkbox" id="scenario-rings"/> Show service-radius rings</label>
        </div>
        <hr/>
        <div id="scenario-cov"></div>
        <div class="muted">share of total Vietnam demand</div>
        <div class="stat"><span><span class="sw sw-up"></span>Upgrades</span><b id="scenario-nup"></b></div>
        <div class="stat"><span><span class="sw sw-gf"></span>Greenfield builds</span><b id="scenario-ngf"></b></div>
        <div class="stat"><span>Avg access · upgrade</span><b id="scenario-dup"></b></div>
        <div class="stat"><span>Avg access · greenfield</span><b id="scenario-dgf"></b></div>
    </div>

    <script>
        var greenfieldScenarios = {scenarios_json};
        var scenarioUnit  = greenfieldScenarios.unit || "km";
        var scenarioSpeed = greenfieldScenarios.speed_kmh || 40;
        var scenarioLayer = null;
        var scenarioRings = null;
        var gfDivIcon = null;

        function updateScenarioLayer() {{
            if (!scenarioLayer) return;
            var b = document.getElementById("scenario-budget").value;
            var r = document.getElementById("scenario-radius").value;
            document.getElementById("scenario-budget-val").textContent = b;

            var byB = greenfieldScenarios.scenarios[b];
            if (!byB || !byB[r]) return;
            var s = byB[r];
            scenarioLayer.clearLayers();
            scenarioRings.clearLayers();
            var showRings = document.getElementById("scenario-rings").checked;
            // ring radius in metres; a time radius is converted via the average speed
            var Rm = (scenarioUnit === "min")
                ? scenarioSpeed * Number(r) / 60 * 1000
                : Number(r) * 1000;

            (s.upgrades || []).forEach(function(h) {{
                L.circleMarker([h.lat, h.lon], {{
                    radius: 7, color: "#fff", weight: 2,
                    fillColor: "#1a9e5f", fillOpacity: 1
                }}).bindTooltip("Upgrade · hospital " + h.id).addTo(scenarioLayer);
                if (showRings) L.circle([h.lat, h.lon], {{radius: Rm, color: "#1a9e5f", weight: 1, fillOpacity: 0.05}}).addTo(scenarioRings);
            }});
            (s.greenfield || []).forEach(function(h) {{
                L.marker([h.lat, h.lon], {{icon: gfDivIcon}})
                    .bindTooltip("Build · site " + h.id).addTo(scenarioLayer);
                if (showRings) L.circle([h.lat, h.lon], {{radius: Rm, color: "#7B2FBE", weight: 1, fillOpacity: 0.05}}).addTo(scenarioRings);
            }});

            document.getElementById("scenario-cov").textContent = (s.coverage_share_of_total * 100).toFixed(1) + "%";
            document.getElementById("scenario-nup").textContent = (s.upgrades || []).length;
            document.getElementById("scenario-ngf").textContent = (s.greenfield || []).length;
            document.getElementById("scenario-dup").textContent = isNaN(s.avg_distance_upgrade_km) ? "–" : s.avg_distance_upgrade_km.toFixed(1) + " " + scenarioUnit;
            document.getElementById("scenario-dgf").textContent = isNaN(s.avg_distance_greenfield_km) ? "–" : s.avg_distance_greenfield_km.toFixed(1) + " " + scenarioUnit;

            // Re-assign an open population point to the new active facility set.
            if (typeof window.onPopulationScenarioChange === "function") {{
                window.onPopulationScenarioChange();
            }}
        }}

        function initScenarioExplorer() {{
            gfDivIcon = L.divIcon({{
                className: "", iconSize: [18,18], iconAnchor: [9,9],
                html: '<div style="width:14px;height:14px;background:#7B2FBE;'
                    + 'border:2px solid #fff;transform:rotate(45deg);'
                    + 'box-shadow:0 1px 3px rgba(0,0,0,.4);"></div>'
            }});

            var bSlider = document.getElementById("scenario-budget");
            var budgets = greenfieldScenarios.budgets;
            bSlider.min = Math.min.apply(null, budgets);
            bSlider.max = Math.max.apply(null, budgets);
            bSlider.value = Math.min(10, bSlider.max);

            var lbl = document.getElementById("scenario-radius-label");
            if (lbl) lbl.textContent = (scenarioUnit === "min") ? "Time radius (min)" : "Service radius (km)";

            var rSel = document.getElementById("scenario-radius");
            greenfieldScenarios.radii.forEach(function(R) {{
                var o = document.createElement("option");
                o.value = R; o.textContent = R + " " + scenarioUnit;
                rSel.appendChild(o);
            }});
            rSel.value = greenfieldScenarios.radii[greenfieldScenarios.radii.length - 1];

            ["scenario-budget", "scenario-radius", "scenario-rings"].forEach(function(id) {{
                document.getElementById(id).addEventListener("input", updateScenarioLayer);
            }});
            updateScenarioLayer();
        }}

        // Folium creates the map variable in a later script, so defer all
        // map-dependent setup until {map_name} actually exists (poll up to ~6 s).
        function _bootScenarioExplorer() {{
            var tries = 0;
            (function attempt() {{
                tries += 1;
                if (typeof {map_name} === "undefined") {{
                    if (tries < 40) setTimeout(attempt, 150);
                    return;
                }}
                try {{
                    scenarioLayer = L.layerGroup().addTo({map_name});
                    scenarioRings = L.layerGroup().addTo({map_name});
                    initScenarioExplorer();
                }} catch (err) {{
                    console.log("scenario explorer init error:", err);
                    if (tries < 40) setTimeout(attempt, 150);
                }}
            }})();
        }}

        if (document.readyState === "complete") _bootScenarioExplorer();
        else window.addEventListener("load", _bootScenarioExplorer);
    </script>
    """

    map_obj.get_root().html.add_child(folium.Element(html))

