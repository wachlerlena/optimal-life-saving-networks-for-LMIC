<!-- ============================================================================
 Population-point colouring overlay for the scenario explorer.
 Self-contained: reads only globals the explorer already defines
 (populationMarkerLookup, facilityMeta, rankedFacilitiesFor, isCovered,
 getScenarioState, showPopulationPanel). No rebuild-time data needed, so the
 same file is (a) embedded by build_population_map.py at build time and
 (b) injectable into an already-built HTML by add_point_coloring.py.

 Two views the corrected per-point road-time data makes possible:
   - "Access time": colour every display point by its road travel time to the
     nearest OPEN advanced facility under the current scenario (fixed clinical
     bands), so the whole access surface shows at once and visibly improves as
     the budget rises.
   - "Serving facility": colour each point by the kind of facility that serves
     it (existing / upgrade / new build), making each new facility's catchment
     visible.
============================================================================ -->
<style>
  #pt-color-panel { position: fixed; bottom: 16px; left: 16px; width: 214px;
    background: rgba(255,255,255,0.97); border-radius: 12px; z-index: 10000;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18); padding: 12px 14px;
    font-family: Arial, sans-serif; }
  #pt-color-panel h4 { margin: 0 0 4px; font-size: 14px; color: #222; }
  #pt-color-panel .sub { font-size: 11px; color: #777; margin-bottom: 8px; }
  #pt-color-panel .seg { display:flex; gap:4px; margin-bottom:9px; }
  #pt-color-panel .seg button { flex:1; font-size:11px; padding:6px 0; cursor:pointer;
    border:1px solid #ccc; background:#f6f6f6; border-radius:6px; line-height:1.15; }
  #pt-color-panel .seg button.on { background:#1f3a5f; color:#fff; border-color:#1f3a5f; }
  #pt-legend { font-size:11px; color:#333; }
  #pt-legend .lr { display:flex; align-items:center; margin:3px 0; }
  #pt-legend .sw { width:12px; height:12px; border-radius:50%; margin-right:7px;
    border:1px solid #fff; box-shadow:0 0 0 1px #bbb; }
  #pt-readout { font-size:11px; color:#555; margin-top:8px; border-top:1px solid #eee; padding-top:6px; }
  #pt-readout b { color:#222; font-variant-numeric: tabular-nums; }
  /* In a colour mode, also hide any orphaned cluster bubbles that markercluster
     can leave in the DOM after the group is removed from the map. */
  body.pt-colormode .marker-cluster { display: none !important; }
</style>
<div id="pt-color-panel">
  <h4>Colour population points</h4>
  <div class="sub">the sampled clickable points</div>
  <div class="seg">
    <button data-m="off">Off</button>
    <button data-m="access">Access<br>time</button>
    <button data-m="kind">Serving<br>facility</button>
  </div>
  <div id="pt-legend"></div>
  <div id="pt-readout"></div>
</div>
<script>
(function(){
  var MODE = "off", overlay = null, refs = [], theMap = null, clusterGroup = null;

  function clinColor(min){
    if (min == null) return "#bdbdbd";
    if (min <= 15)  return "#1a9850";
    if (min <= 30)  return "#a6d96a";
    if (min <= 60)  return "#fdae61";
    if (min <= 120) return "#f46d43";
    return "#d73027";
  }
  function kindColor(kind){
    if (kind === "greenfield") return "#6a51a3";
    if (kind === "upgrade")    return "#1a9850";
    return "#b2182b"; // existing advanced ("advanced")
  }
  function findMap(){
    if (theMap) return theMap;
    for (var k in window){
      try { if (window[k] instanceof L.Map){ theMap = window[k]; break; } } catch(e){}
    }
    return theMap;
  }
  function assignedFor(popId){
    var res = (typeof rankedFacilitiesFor === "function") ? rankedFacilitiesFor(popId) : null;
    if (!res || !res.ranked || !res.ranked.length) return {min:null, kind:null, reachable:false, covered:false};
    var a = res.ranked[0], fac = (facilityMeta[a.id] || {});
    var cov = isCovered([a.km, a.min], res.state);
    return {min:a.min, kind:fac.kind, reachable:true, covered:(cov !== false)};
  }
  // Build the non-clustered overlay once, reading coords from the existing
  // markers. Returns true only when at least one marker resolved, so a too-early
  // call does not latch an empty overlay.
  function buildOverlay(){
    if (overlay && refs.length) return true;
    if (typeof populationMarkerLookup === "undefined") return false;
    findMap();
    var tmp = [];
    Object.keys(populationMarkerLookup).forEach(function(id){
      var mk; try { mk = eval(populationMarkerLookup[id]); } catch(e){ return; }
      if (!mk || !mk.getLatLng) return;
      var ll = mk.getLatLng();
      tmp.push({id:id, lat:ll.lat, lon:ll.lng});
    });
    if (!tmp.length) return false;
    refs = tmp;
    overlay = L.layerGroup();
    refs.forEach(function(p){
      p.cm = L.circleMarker([p.lat, p.lon], {radius:6, weight:1, color:"#333",
        fillColor:"#bdbdbd", fillOpacity:0.9});
      p.cm.on('click', (function(pid){ return function(){
        if (window.showPopulationPanel) window.showPopulationPanel(pid);
      }; })(p.id));
      overlay.addLayer(p.cm);
    });
    return true;
  }
  // The sampled clickable markers live in a MarkerCluster; reach its group via
  // any marker's __parent._group (robust to Leaflet minification, which mangles
  // class names). Hiding it in colour mode replaces the cluster bubbles with the
  // clean colour overlay; the density heatmap stays toggleable in the layers
  // control. Restored when the user switches colouring off.
  function findClusterGroup(){
    if (clusterGroup) return clusterGroup;
    var ids = Object.keys(populationMarkerLookup);
    for (var i = 0; i < ids.length; i++){
      var mk; try { mk = eval(populationMarkerLookup[ids[i]]); } catch(e){ continue; }
      if (mk && mk.__parent && mk.__parent._group){ clusterGroup = mk.__parent._group; break; }
    }
    return clusterGroup;
  }
  function hideBaseLayers(){
    findMap(); var g = findClusterGroup();
    if (theMap && g && theMap.hasLayer(g)) theMap.removeLayer(g);
    document.body.classList.add("pt-colormode");
  }
  function restoreBaseLayers(){
    findMap();
    document.body.classList.remove("pt-colormode");
    if (theMap && clusterGroup && !theMap.hasLayer(clusterGroup)){
      try { theMap.addLayer(clusterGroup); } catch(e){}
    }
  }
  function legend(){
    var el = document.getElementById("pt-legend");
    if (!el) return;
    if (MODE === "off"){ el.innerHTML = '<div style="color:#777">Clustered clickable points.</div>'; return; }
    if (MODE === "access"){
      el.innerHTML =
        '<div class="lr"><span class="sw" style="background:#1a9850"></span>&le; 15 min</div>'+
        '<div class="lr"><span class="sw" style="background:#a6d96a"></span>15&ndash;30 min</div>'+
        '<div class="lr"><span class="sw" style="background:#fdae61"></span>30&ndash;60 min</div>'+
        '<div class="lr"><span class="sw" style="background:#f46d43"></span>60&ndash;120 min</div>'+
        '<div class="lr"><span class="sw" style="background:#d73027"></span>&gt; 120 min</div>';
    } else {
      el.innerHTML =
        '<div class="lr"><span class="sw" style="background:#b2182b"></span>existing advanced</div>'+
        '<div class="lr"><span class="sw" style="background:#1a9850"></span>upgraded hospital</div>'+
        '<div class="lr"><span class="sw" style="background:#6a51a3"></span>new build</div>'+
        '<div class="lr"><span class="sw" style="background:#bdbdbd"></span>none within 300 km</div>';
    }
  }
  function recolor(){
    Array.prototype.forEach.call(document.querySelectorAll('#pt-color-panel .seg button'),
      function(b){ b.classList.toggle('on', b.getAttribute('data-m') === MODE); });
    legend();
    var ro = document.getElementById("pt-readout");
    findMap();
    if (MODE === "off"){
      if (overlay && theMap && theMap.hasLayer(overlay)) theMap.removeLayer(overlay);
      restoreBaseLayers();
      if (ro) ro.innerHTML = "";
      return;
    }
    if (!buildOverlay()){ if (ro) ro.innerHTML = "loading points…"; return; }
    hideBaseLayers();
    if (theMap && !theMap.hasLayer(overlay)) theMap.addLayer(overlay);
    // Population-weighted median access time of the covered points. Weighting by
    // household_count corrects the uniform (cell-weighted) sample so the figure
    // is representative of people, not cells, and matches the thesis (~24 min at
    // 60 min / B10). The panel above already reports the authoritative coverage %.
    var pairs = [], nCov = 0;
    refs.forEach(function(p){
      var a = assignedFor(p.id), col;
      if (MODE === "access") col = clinColor(a.reachable ? a.min : null);
      else col = a.reachable ? kindColor(a.kind) : "#bdbdbd";
      if (a.covered && a.min != null){
        var pd = (typeof populationPanelData !== "undefined") ? populationPanelData[p.id] : null;
        var w = (pd && typeof pd.household_count === "number" && pd.household_count > 0) ? pd.household_count : 1;
        pairs.push([a.min, w]); nCov++;
      }
      if (p.cm) p.cm.setStyle({fillColor: col, fillOpacity: a.reachable ? 0.92 : 0.5});
    });
    if (ro){
      pairs.sort(function(x, y){ return x[0] - y[0]; });
      var half = 0; pairs.forEach(function(d){ half += d[1]; }); half /= 2;
      var acc = 0, med = "–";
      for (var i = 0; i < pairs.length; i++){ acc += pairs[i][1]; if (acc >= half){ med = Math.round(pairs[i][0]); break; } }
      ro.innerHTML = "Median access of covered: <b>" + med + " min</b><br>" +
                     "<span style='color:#888'>population-weighted; " + nCov + " of " + refs.length + " shown pts covered</span>";
    }
  }
  window.recolorPopulationPoints = recolor;
  window.setPopColorMode = function(m){ MODE = m; recolor(); };

  function wire(){
    Array.prototype.forEach.call(document.querySelectorAll('#pt-color-panel .seg button'),
      function(b){ b.addEventListener('click', function(){ window.setPopColorMode(b.getAttribute('data-m')); }); });
    ["scenario-budget","scenario-radius"].forEach(function(id){
      var el = document.getElementById(id);
      if (el) el.addEventListener('input', function(){ if (MODE !== "off") setTimeout(recolor, 40); });
    });
  }
  // Boot: wait until Leaflet + the markers actually resolve, then enable.
  var tries = 0;
  (function boot(){
    tries++;
    var ready = (typeof L !== "undefined") && (typeof populationMarkerLookup !== "undefined")
                && Object.keys(populationMarkerLookup).length > 0 && buildOverlay();
    if (!ready){ if (tries < 100) return setTimeout(boot, 150); }
    wire(); legend();
  })();
})();
</script>
