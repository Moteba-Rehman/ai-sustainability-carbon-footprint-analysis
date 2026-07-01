from transformers import pipeline
from codecarbon import EmissionsTracker
import pandas as pd
import time

# ==========================
# Load Qwen Model
# ==========================

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading {model_name}...")

generator = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto"
)

# ==========================
# Standardized Prompts
# ==========================

prompts = [
    "Explain the importance of renewable energy.",
    "What is the carbon footprint of AI?",
    "How does machine learning help climate change?",
    "Explain large language models in simple words.",
    "What are sustainable AI practices?"
]

results = []

# ==========================
# Run Qwen Experiment
# ==========================

for run in range(1, 6):

    for prompt in prompts:

        tracker = EmissionsTracker(
            project_name="qwen_inference",
            output_file="qwen_codecarbon_logs.csv"
        )

        tracker.start()

        start_time = time.time()

        output = generator(
            prompt,
            max_new_tokens=50,
            do_sample=False
        )

        runtime = time.time() - start_time

        emissions = tracker.stop()

        results.append({
            "Model": "Qwen2.5-1.5B-Instruct",
            "Run": run,
            "Prompt": prompt,
            "Runtime (s)": round(runtime, 4),
            "Generated Text": output[0]["generated_text"],
            "CO2 Emissions (kg)": emissions
        })

print("✅ Qwen experiment completed!")

# ==========================
# Save Detailed Results
# ==========================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "qwen_experiment_results.csv",
    index=False
)

print("Detailed experiment results saved.")

display(results_df.head())

# ==========================
# Create Carbon Emissions Summary
# ==========================

summary = (
    results_df
    .groupby("Model")
    .agg(
        Total_Runs=("Run", "count"),
        Total_Runtime_s=("Runtime (s)", "sum"),
        CO2_Emissions_kg=("CO2 Emissions (kg)", "sum")
    )
    .reset_index()
)

summary.columns = [
    "Model",
    "Total Runs",
    "Total Runtime (s)",
    "CO2 Emissions (kg)"
]

summary.to_csv(
    "qwen_carbon_emissions_report.csv",
    index=False
)

print("Carbon emissions summary saved.")

display(summary)

print("="*60)
print("✅ Qwen experiment completed successfully!")
print("="*60)

print("\nGenerated files:")

display(FileLink("qwen_experiment_results.csv"))
display(FileLink("qwen_carbon_emissions_report.csv"))
display(FileLink("qwen_codecarbon_logs.csv"))