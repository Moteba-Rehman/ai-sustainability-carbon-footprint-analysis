from codecarbon import EmissionsTracker
from transformers import pipeline
import pandas as pd
import time
import torch

# Same prompts as GPT-2
prompts = [
    "Explain the importance of renewable energy.",
    "What is the carbon footprint of AI?",
    "How does machine learning help climate change?",
    "Explain large language models in simple words.",
    "What are sustainable AI practices?"
]

# Official Meta LLaMA model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = 0 if torch.cuda.is_available() else -1

print(f"Loading {model_name}...")

generator = pipeline(
    "text-generation",
    model=model_name,
    device=device
)

results = []

tracker = EmissionsTracker(
    project_name="Llama-3.2-1B",
    output_file="llama_codecarbon_logs.csv",
    log_level="error"
)

tracker.start()

total_runtime = 0

for run in range(1, 6):

    for prompt in prompts:

        start = time.time()

        output = generator(
            prompt,
            max_new_tokens=50,
            do_sample=False
        )

        runtime = time.time() - start
        total_runtime += runtime

        results.append({
            "Model": "TinyLlama-1.1B-Chat-v1.0",
            "Run": run,
            "Prompt": prompt,
            "Runtime (s)": round(runtime, 4),
            "Generated Text": output[0]["generated_text"]
        })

emissions = tracker.stop()

experiment_df = pd.DataFrame(results)

carbon_df = pd.DataFrame([{
    "Model": "TinyLlama-1.1B-Chat-v1.0",
    "Total Runs": 25,
    "Total Runtime (s)": round(total_runtime, 4),
    "CO2 Emissions (kg)": emissions
}])

experiment_df.to_csv("llama_experiment_results.csv", index=False)
carbon_df.to_csv("llama_carbon_emissions_report.csv", index=False)

print("✅ LLaMA experiment completed!")

display(experiment_df.head())
display(carbon_df)