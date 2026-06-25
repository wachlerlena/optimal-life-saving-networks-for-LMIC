import pandas as pd
import matplotlib.pyplot as plt


EXISTING_ADVANCED_CENTERS = 55
MAXIMUM_COVERED_90 = 0.8956701628820758
MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP = 77643797.03586955
MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG = 77991149.62346442


#------------------------------
# Plotting the Pareto curve for the weighted models 
#------------------------------
pareto = pd.read_csv("/Users/casperklusener/Documents/Thesis/Pareto/pareto_curve_results.csv")

# Only needed if not already in CSV
if "objective_value_million" not in pareto.columns:
    pareto["objective_value_million"] = pareto["objective_value"] / 1_000_000

MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP_M = MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP / 1_000_000
MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG_M = MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG / 1_000_000

# -----------------------------
# Needed to add b=0 baseline points for the plot, because made a mistake and didn't include them in the original CSV output
# -----------------------------
b0_rows = pd.DataFrame([
    {
        "scenario": "exponential",
        "b": 0,
        "total_advanced_centers": 55,
        "objective_value": 	67497607.63013044,   
        "covered_90_pct": 0.7598982317317002 * 100,
    },
    {
        "scenario": "logistic",
        "b": 0,
        "total_advanced_centers": 55,
        "objective_value": 68772870.42293842,   
        "covered_90_pct": 0.7598982317317002 * 100,
    },
])

pareto = pd.concat([b0_rows, pareto], ignore_index=True)
pareto["objective_value_million"] = pareto["objective_value"] / 1_000_000
pareto = pareto.sort_values(["scenario", "b"])

# -----------------------------
# Plot 1: Total expected outcomes
# -----------------------------
plt.figure(figsize=(11, 6))

for scenario in pareto["scenario"].unique():
    s = pareto[pareto["scenario"] == scenario]
    plt.plot(
        s["total_advanced_centers"],
        s["objective_value_million"],
        linewidth=2.0,
        label=scenario,
    )

plt.axvline(
    EXISTING_ADVANCED_CENTERS,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label="Existing advanced stroke centers",
)

plt.axhline(
    MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP_M,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label=f"Maximum exponential ({MAXIMUM_TOTAL_EXPECTED_OUTCOMES_EXP_M:.1f} million)",
)

plt.axhline(
    MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG_M,
    color="tab:red",
    linestyle="--",
    linewidth=1.8,
    label=f"Maximum logistic ({MAXIMUM_TOTAL_EXPECTED_OUTCOMES_LOG_M:.1f} million)",
)

plt.xlabel("Number of Advanced Stroke Centers (Existing + New)")
plt.ylabel("Total Expected Outcomes (millions)")
plt.title("Pareto Curve of Total Expected Outcomes")
plt.legend()
plt.grid(alpha=1)
plt.tight_layout()
plt.savefig("pareto_total_expected_outcomes.png", dpi=300)



# -----------------------------
# Plot 2: Coverage within 90 min
# -----------------------------
plt.figure(figsize=(11, 6))

for scenario in pareto["scenario"].unique():
    s = pareto[pareto["scenario"] == scenario]
    plt.plot(
        s["total_advanced_centers"],
        s["covered_90_pct"],
        linewidth=2.0,
        label=scenario,
    )

plt.axhline(
    MAXIMUM_COVERED_90 * 100,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label=f"Maximum ≤90 min coverage ({MAXIMUM_COVERED_90 * 100:.1f}%)",
)

plt.axvline(
    EXISTING_ADVANCED_CENTERS,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label="Existing advanced stroke centers",
)

plt.xlabel("Number of Advanced Stroke Centers (Existing + New)")
plt.ylabel("Population Covered within 90 Minutes (%)")
plt.title("Pareto Curve of Population Access within 90 Minutes")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("pareto_coverage_90min.png", dpi=300)
plt.show()

#-----------------------------
# Plotting the Pareto curve for the base MCLP model
#-----------------------------

base_pareto = pd.read_csv("base_pareto_results.csv")

"""
template = base_pareto.loc[base_pareto["b"] == 23].iloc[0]

new_rows = []

for b in range(28, 76):  
    row = template.copy()
    row["b"] = b
    row["total_advanced_centers"] = 55 + b
    new_rows.append(row)

base_pareto = pd.concat(
    [base_pareto, pd.DataFrame(new_rows)],
    ignore_index=True
)

base_pareto = base_pareto.sort_values("b")

base_pareto.to_csv("base_pareto_results_extended.csv", index=False)
"""

MAXIMUM_BASE_COVERED_90 = 0.8639723321514359
MAXIMUM_BASE_OBJECTIVE = 96887247.0
MAXIMUM_BASE_OBJECTIVE_M = MAXIMUM_BASE_OBJECTIVE / 1_000_000

if "objective_value_million" not in base_pareto.columns:
    base_pareto["objective_value_million"] = base_pareto["objective_value"] / 1_000_000

# -----------------------------
# Base Plot 1: Objective value
# -----------------------------
plt.figure(figsize=(11, 6))

plt.plot(
    base_pareto["total_advanced_centers"],
    base_pareto["objective_value_million"],
    linewidth=2.0,
    label="base MCLP",
)

plt.axvline(
    EXISTING_ADVANCED_CENTERS,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label="Existing advanced stroke centers",
)

plt.axhline(
    MAXIMUM_BASE_OBJECTIVE_M,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label=f"Maximum base objective ({MAXIMUM_BASE_OBJECTIVE_M:.1f} million)",
)

plt.xlabel("Number of Advanced Stroke Centers (Existing + New)")
plt.ylabel("Population Covered within 270 Minutes (millions)")
plt.title("Pareto Curve of Base MCLP Objective")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("base_pareto_objective.png", dpi=300)


# -----------------------------
# Base Plot 2: Coverage within 90 min
# -----------------------------
plt.figure(figsize=(11, 6))

plt.plot(
    base_pareto["total_advanced_centers"],
    base_pareto["covered_90_pct"],
    linewidth=2.0,
    label="base MCLP",
)

plt.axhline(
    MAXIMUM_BASE_COVERED_90 * 100,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label=f"Maximum ≤90 min coverage ({MAXIMUM_BASE_COVERED_90 * 100:.1f}%)",
)

plt.axvline(
    EXISTING_ADVANCED_CENTERS,
    color="tab:green",
    linestyle="--",
    linewidth=1.8,
    label="Existing advanced stroke centers",
)

plt.xlabel("Number of Advanced Stroke Centers (Existing + New)")
plt.ylabel("Population Covered within 90 Minutes (%)")
plt.title("Pareto Curve of Base MCLP Population Access within 90 Minutes")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("base_pareto_coverage_90min.png", dpi=300)

plt.show()