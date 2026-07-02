# Advanced Stroke-Care Coverage in Vietnam

Optimisation code and interactive artifact for a BSc thesis on **where to upgrade
or build advanced stroke-care centers in Vietnam** to maximise the population
reached within a clinical time standard, for a given budget. A two-stage
maximum-covering model (MCLP) solved over budgets and time thresholds on real
per-road-type travel times.

## Layout

**`01_optimization_models/`** — the location models
- `greenfield_existing_advanced_model.py` — core two-stage MCLP: pick which existing hospitals to upgrade and which greenfield sites to build to maximise covered demand within a budget. Provides the data-loading/assignment helpers reused elsewhere.
- `baseline_upgrade_model.py` — simpler upgrade-only baseline (no new builds).

**`02_travel_time/`** — road-graph travel times (minutes)
- `corrected_roadtime_grid.py` — the main pipeline: builds within-threshold road-time matrices (Dijkstra on per-road-type speeds) and solves every budget×threshold scenario. Fixes the coverage bug and produces the canonical thesis outputs + map data.
- `build_travel_time_matrices.py` — precomputes the pop→facility travel-time matrices used by the roadtime scenarios.

**`03_scenarios/`** — scenario generation and coverage analysis
- `run_combined_scenarios.py` — solves the model across the budget×radius grid (`--mode km/time/roadtime`).
- `true_advanced_baseline.py` — diagnostic: true road coverage of the existing centers (proved the data bug).
- `time_to_treatment_extension.py` — travel-time (minutes) version of the coverage analysis.

**`04_robustness/`**
- `robustness_corrected.py` — radius / speed / cost robustness checks.

**`06_interactive_map/`** — the self-contained HTML explorer
- `build_population_map.py` — builds the interactive map (heatmap, hospitals, scenario slider, click-to-route).
- `map_ui.py` — the injected side-panel HTML/CSS/JS imported by the builder.
- `point_coloring_overlay.js` — overlay colouring population points by access time / serving facility.
- `add_point_coloring.py` — injects the overlay into an already-built map HTML.

## Setup

```bash
pip install -r requirements.txt
```

Scripts read a `data/` folder and write to `outputs/`/`maps/` (not included here —
large and reproducible).
