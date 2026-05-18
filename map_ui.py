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
    route_data,
    population_panel_data,
    population_marker_lookup,
    hospital_panel_data,
    hospital_marker_lookup,
):
    """
    Inject the right-side panel and the JavaScript needed for:
    - clicking population markers to see top 3 hospitals
    - drawing the route to a selected hospital
    - clicking hospital markers to see hospital details
    """
    map_name = map_obj.get_name()

    route_json = json.dumps(route_data)
    population_panel_json = json.dumps(population_panel_data)
    population_marker_lookup_json = json.dumps(population_marker_lookup)
    hospital_panel_json = json.dumps(hospital_panel_data)
    hospital_marker_lookup_json = json.dumps(hospital_marker_lookup)

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
                }} catch (err) {{
                    console.log("Could not remove previous route:", err);
                }}
                activeRouteLine = null;
            }}
        }}

        function hidePopulationPanel() {{
            clearActiveRoute();
            var panel = document.getElementById("population-side-panel");
            if (panel) {{
                panel.classList.remove("open");
            }}
        }}

        function showHospitalRoute(popId, hospitalId) {{
            clearActiveRoute();

            var popRoutes = routeData[String(popId)];
            if (!popRoutes) {{
                console.log("No route data found for population ID:", popId);
                return;
            }}

            var coords = popRoutes[String(hospitalId)];
            if (!coords || coords.length < 2) {{
                console.log("No route geometry available:", popId, hospitalId);
                return;
            }}

            activeRouteLine = L.polyline(coords, {{
                color: "red",
                weight: 4,
                opacity: 0.9
            }}).addTo({map_name});

            try {{
                {map_name}.fitBounds(activeRouteLine.getBounds(), {{padding: [20, 20]}});
            }} catch (err) {{
                console.log("Could not fit bounds for route:", err);
            }}
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
                content.innerHTML = '<div class="panel-empty">No data available for this population point.</div>';
                panel.classList.add("open");
                return;
            }}

            var hospitals = data.hospitals || [];

            var html = "";
            html += '<div class="panel-meta">';
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
                    var chipText = specialized ? "Advanced (thrombolysis and mechanical thrombectomy)" : "Basic (thrombolysis)";

                    var distanceText = formatNumber(h.distance_km, 2);
                    if (distanceText !== "Not available") distanceText += " km";

                    var timeText = formatNumber(h.estimated_travel_time_min, 1);
                    if (timeText !== "Not available") timeText += " min";

                    // Look up hospital name and address from hospitalPanelData
                    var hospInfo = hospitalPanelData[String(hospitalId)];
                    var hospName = hospInfo ? (hospInfo.hospital_name || "") : "";
                    var hospAddress = hospInfo ? (hospInfo.address_text || "") : "";

                    var cardTitle = hospName
                        ? (index + 1) + '. ' + hospName + ' (ID ' + hospitalId + ')'
                        : (index + 1) + '. Hospital ' + hospitalId;

                    html += '<div class="hospital-card">';
                    html += '<div class="hospital-card-title">' + cardTitle + '</div>';

                    if (hospAddress) {{
                        html += '<div class="hospital-card-row"><b>Address:</b> ' + hospAddress + '</div>';
                    }}

                    html += '<div class="hospital-card-row"><b>Stroke equipment:</b> <span class="hospital-chip ' + chipClass + '">' + chipText + '</span></div>';
                    html += '<div class="hospital-card-row"><b>Distance:</b> ' + distanceText + '</div>';
                    html += '<div class="hospital-card-row"><b>Estimated travel time:</b> ' + timeText + '</div>';

                    if (h.route_available) {{
                        html += '<button class="route-btn" onclick="showHospitalRoute(\\'' + data.population_id + '\\', \\''
                             + hospitalId + '\\')">Show route</button>';
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
                        markerObj.on('click', function() {{
                            showPopulationPanel(popId);
                        }});
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
                        markerObj.on('click', function() {{
                            showHospitalPanel(hospitalId);
                        }});
                        hospitalBoundCount += 1;
                    }}
                }} catch (err) {{
                    console.log("Failed to bind hospital marker click for hospitalId:", hospitalId, err);
                }}
            }});

            console.log("Bound population side-panel clicks:", populationBoundCount);
            console.log("Bound hospital side-panel clicks:", hospitalBoundCount);
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

    map_obj.get_root().html.add_child(folium.Element(html))

