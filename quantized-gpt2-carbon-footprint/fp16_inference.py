import os
import time
import warnings
import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(device)

model.eval()

print(f"{MODEL_NAME} loaded successfully in FP16.")

prompts = [
    "Artificial intelligence is transforming",
    "Climate change can be reduced by",
    "The future of renewable energy depends on",
    "Machine learning models require",
    "Carbon footprint measurement helps"
]

print(f"Loaded {len(prompts)} prompts.")

results = []

tracker = EmissionsTracker(
    project_name="gpt2-fp16",
    output_file="fp16_codecarbon_logs.csv"
)

tracker.start()

for run in range(5):
    for prompt in prompts:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(device)

        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        runtime = time.time() - start_time

        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        results.append({
            "Run": run + 1,
            "Prompt": prompt,
            "Runtime (s)": runtime,
            "Generated Text": generated_text
        })

tracker.stop()

results_df = pd.DataFrame(results)

results_df.to_csv(
    "fp16_experiment_results.csv",
    index=False
)

print(results_df.head())
print(f"\nTotal inference runs: {len(results_df)}")

import pandas as pd

logs = pd.read_csv("fp16_codecarbon_logs.csv")
results = pd.read_csv("fp16_experiment_results.csv")

summary = pd.DataFrame({
    "Model": ["GPT-2 FP16"],
    "Total Runs": [len(results)],
    "Total Runtime (s)": [results["Runtime (s)"].sum()],
    "Runtime per Inference (s)": [results["Runtime (s)"].mean()],
    "CO2 Emissions (kg)": [logs["emissions"].sum()],
    "CO2 per Inference (kg)": [
        logs["emissions"].sum() / len(results)
    ],
    "Estimated kWh per Inference": [
        logs["energy_consumed"].sum() / len(results)
    ],
    "GPU Hours": [
        results["Runtime (s)"].sum() / 3600
    ]
})

summary.to_csv(
    "fp16_carbon_emissions_report.csv",
    index=False
)

summary