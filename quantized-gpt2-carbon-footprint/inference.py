import os
import time
import warnings
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_NAME = "gpt2"

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

print("Original GPT-2 loaded successfully!")

print("Applying dynamic INT8 quantization...")

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

quantized_model.eval()

print("Quantized GPT-2 loaded successfully!")

prompts = [
    "Artificial intelligence is transforming healthcare by",
    "Climate change can be addressed through",
    "The future of education will depend on",
    "Renewable energy technologies are important because",
    "Machine learning algorithms improve by"
]

print(f"Loaded {len(prompts)} prompts.")

results = []

tracker = EmissionsTracker(
    project_name="quantized-gpt2",
    output_file="quantized_codecarbon_logs.csv"
)

tracker.start()

for run in range(5):
    for prompt in prompts:

        start = time.time()

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = quantized_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        runtime = time.time() - start

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "Run": run + 1,
            "Prompt": prompt,
            "Runtime (s)": runtime,
            "Generated Text": generated_text
        })

tracker.stop()

results_df = pd.DataFrame(results)

results_df.to_csv(
    "quantized_experiment_results.csv",
    index=False
)

print(results_df.head())
print(f"\nTotal Inference Runs: {len(results_df)}")

logs = pd.read_csv("quantized_codecarbon_logs.csv")

summary = pd.DataFrame({
    "Model": ["Quantized GPT-2"],
    "Total Runs": [len(results_df)],
    "Total Runtime (s)": [results_df["Runtime (s)"].sum()],
    "CO2 Emissions (kg)": [logs["emissions"].sum()]
})

summary["CO2 per Inference (kg)"] = (
    summary["CO2 Emissions (kg)"] / summary["Total Runs"]
)

summary["Runtime per Inference (s)"] = (
    summary["Total Runtime (s)"] / summary["Total Runs"]
)

summary["GPU Hours"] = (
    summary["Total Runtime (s)"] / 3600
)

summary["Estimated kWh per Inference"] = (
    logs["energy_consumed"].sum() / summary["Total Runs"]
)

summary.to_csv(
    "quantized_carbon_emissions_report.csv",
    index=False
)

summary

gpt2 = pd.read_csv("carbon_emissions_report.csv")
quantized = pd.read_csv("quantized_carbon_emissions_report.csv")

comparison = pd.concat([gpt2.iloc[[0]], quantized], ignore_index=True)

comparison.to_csv(
    "gpt2_vs_quantized_gpt2.csv",
    index=False
)

comparison

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

plt.bar(
    comparison["Model"],
    comparison["Runtime per Inference (s)"]
)

plt.title("Runtime per Inference")
plt.ylabel("Seconds")

plt.tight_layout()

plt.savefig("runtime_per_inference.png", dpi=300)

plt.show()

plt.figure(figsize=(6,4))

plt.bar(
    comparison["Model"],
    comparison["CO2 per Inference (kg)"]
)

plt.title("CO₂ Emissions per Inference")
plt.ylabel("kg CO₂")

plt.tight_layout()

plt.savefig("co2_per_inference.png", dpi=300)

plt.show()