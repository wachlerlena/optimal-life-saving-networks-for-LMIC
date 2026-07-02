"""
Time-to-treatment extension (simple, isolated add-on).

Coverage stays the central objective. This script adds the lens the supervisor
asked for ("translate to time rather than distance", "choose an average speed and
simple translation", "proportion of people within a time radius", "Pareto curve
with budgets and time to service") and uses it to expose the limitation of pure
distance/binary coverage.

It does NOT modify the baseline or combined models. It reuses their functions
read-only: it converts the road distances to travel time via a single average
speed, then runs the *same* max-coverage solve with TIME thresholds instead of
km radii.

Three deliverables (in outputs/time_to_treatment/):
  1. tt_coverage_grid.csv      coverage(B, time-threshold) — the Pareto data
  2. tt_access_distribution.csv for the loosest-threshold plan, how the covered
                                population actually splits across access times
  3. tt_pareto.png, tt_access_cdf.png  the two figures

Run:  python time_to_treatment_extension.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import greenfield_existing_advanced_model as gm
from run_combined_scenarios import solve_coverage_fast

# --- settings (the supervisor's "choose an average speed and simple translation") ---
SPEED_KMH      = 40.0                      # single average road speed; CONFIRM with supervisor
THRESHOLDS_MIN = [30, 60, 120, 240]        # time-to-treatment radii (minutes)
BUDGETS        = [0, 1, 2, 3, 5, 7, 10, 15]

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "time_to_treatment"


def to_time(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a distance-pair frame, replacing road km with travel minutes."""
    t = df.copy()
    t["total_dist"] = t["total_dist"].astype(float) / SPEED_KMH * 60.0   # km -> minutes
    return t


def wmean(values, weights):
    w = float(weights.sum())
    return float((values * weights).sum() / w) if w > 0 else float("nan")


def wmedian(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    if c[-1] == 0:
        return float("nan")
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (pop, existing_pairs, gf_pairs, advanced_ids, basic_ids, greenfield_ids,
     all_hosp, new_hosp, total_demand) = gm.load_data()

    # One-line translation: distances -> travel minutes.
    existing_t = to_time(existing_pairs)
    gf_t       = to_time(gf_pairs)
    print(f"\nAverage speed: {SPEED_KMH:.0f} km/h  ->  thresholds (min) "
          f"{THRESHOLDS_MIN} = {[round(SPEED_KMH*t/60) for t in THRESHOLDS_MIN]} km equivalent\n")

    # ---- 1. coverage(B, time threshold): the Pareto grid ----
    grid = []
    loosest = max(THRESHOLDS_MIN)
    plans = {}   # budget -> (already, pw, basic_r, gf_r, sel_u, sel_g) at loosest threshold
    for T in THRESHOLDS_MIN:
        already, pw, basic_r, gf_r = gm.preprocess(
            pop, existing_t, gf_t, advanced_ids, basic_ids, greenfield_ids, T)
        fixed = sum(pw[p] for p in already)
        for B in BUDGETS:
            sel_u, sel_g = solve_coverage_fast(pw, already, basic_r, gf_r, B)
            new_assign = gm.assign_to_selection(basic_r, gf_r, sel_u, sel_g, already, pw)
            covered = fixed + float(new_assign["demand"].sum())
            grid.append({
                "budget": B, "threshold_min": T,
                "equiv_km": round(SPEED_KMH * T / 60, 1),
                "coverage_share": covered / total_demand,
                "n_upgrades": len(sel_u), "n_greenfield": len(sel_g),
            })
            if T == loosest:
                plans[B] = (already, pw, basic_r, gf_r, sel_u, sel_g)
        print(f"  T={T:3d} min done")
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(OUT_DIR / "tt_coverage_grid.csv", index=False)

    # ---- 2. access-time distribution of the COVERED population (the limitation) ----
    # Take the plan optimised at the loosest standard (as a pure coverage study
    # would), then ask: of everyone it counts as "covered", how fast can they
    # actually be treated?
    dist_rows = []
    cdf_cache = {}
    for B in BUDGETS:
        already, pw, basic_r, gf_r, sel_u, sel_g = plans[B]
        adv = gm.advanced_assignment(existing_t, advanced_ids, already, loosest, pw)
        new = gm.assign_to_selection(basic_r, gf_r, sel_u, sel_g, already, pw)
        allc = pd.concat([adv[["demand", "distance_km"]], new[["demand", "distance_km"]]],
                         ignore_index=True)            # distance_km is MINUTES here
        tot = float(allc["demand"].sum())
        row = {"budget": B, "covered_demand": tot, "coverage_share": tot / total_demand}
        for thr in THRESHOLDS_MIN:
            row[f"pct_within_{thr}min"] = float(
                allc.loc[allc["distance_km"] <= thr, "demand"].sum() / tot) if tot else float("nan")
        row["mean_access_min"]   = wmean(allc["distance_km"].to_numpy(), allc["demand"].to_numpy())
        row["median_access_min"] = wmedian(allc["distance_km"].to_numpy(), allc["demand"].to_numpy())
        dist_rows.append(row)
        cdf_cache[B] = allc
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIR / "tt_access_distribution.csv", index=False)

    # ---- 3a. Pareto figure: coverage vs budget, one line per time threshold ----
    plt.figure(figsize=(7, 4.5))
    for T in THRESHOLDS_MIN:
        sub = grid_df[grid_df.threshold_min == T].sort_values("budget")
        plt.plot(sub.budget, sub.coverage_share * 100, marker="o", label=f"{T} min")
    plt.xlabel("Budget (new advanced facilities)")
    plt.ylabel("Population coverage (% of total)")
    plt.title(f"Coverage vs budget by time-to-treatment threshold ({SPEED_KMH:.0f} km/h)")
    plt.legend(title="Time radius"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "tt_pareto.png", dpi=150); plt.close()

    # ---- 3b. The limitation figure: access-time CDF of the "covered" population ----
    plt.figure(figsize=(7, 4.5))
    for B in [0, 5, 10, 15]:
        allc = cdf_cache[B].sort_values("distance_km")
        x = allc["distance_km"].to_numpy()
        y = np.cumsum(allc["demand"].to_numpy()) / total_demand * 100
        plt.step(x, y, where="post", label=f"B={B}")
    for thr in THRESHOLDS_MIN:
        plt.axvline(thr, color="grey", ls=":", lw=0.8)
    plt.xlabel("Travel time to nearest advanced facility (min)")
    plt.ylabel("Cumulative population (% of total)")
    plt.title("Where the 'covered' population really sits on access time")
    plt.xlim(0, loosest); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_DIR / "tt_access_cdf.png", dpi=150); plt.close()

    # ---- console summary: the headline limitation ----
    print("\n" + "=" * 64)
    print("LIMITATION OF PURE COVERAGE (budget = 10)")
    g10 = grid_df[grid_df.budget == 10].set_index("threshold_min")["coverage_share"] * 100
    for T in THRESHOLDS_MIN:
        print(f"  coverage within {T:3d} min: {g10[T]:5.1f}%")
    d10 = dist_df[dist_df.budget == 10].iloc[0]
    print(f"  -> a plan optimised at {loosest} min covers {d10.coverage_share*100:.1f}% of the "
          f"country, but median access time is {d10.median_access_min:.0f} min "
          f"and only {d10[f'pct_within_60min']*100:.0f}% are within 60 min.")
    print("=" * 64)
    print(f"\nSaved CSVs + figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
