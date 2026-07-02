import pandas as pd

REGION = "Pakistan"

carbon_data = pd.read_csv("data/carbon_intensity.csv")

# Remove accidental spaces
carbon_data.columns = carbon_data.columns.str.strip()

experiment_data = pd.read_csv("experiment_results.csv")

intensity = carbon_data.loc[
    carbon_data["Region"] == REGION,
    "gCO2_per_kWh"
].iloc[0]

experiment_data["co2_g"] = (
    experiment_data["co2_kg"] * 1000
)

experiment_data["region"] = REGION

experiment_data["carbon_intensity_gco2_per_kwh"] = intensity

experiment_data.to_csv(
    "carbon_emissions_report.csv",
    index=False
)

summary = experiment_data.groupby(
    "model"
)[["runtime_seconds", "co2_g"]].mean()

print("\nAverage Results")

print(summary)

print("\nGenerated:")

print("carbon_emissions_report.csv")