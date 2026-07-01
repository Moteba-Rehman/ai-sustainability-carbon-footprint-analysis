import time
import os
import pandas as pd
import torch

from transformers import pipeline
from codecarbon import EmissionsTracker

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

    from codecarbon import EmissionsTracker
from transformers import pipeline
import pandas as pd
import time
import torch

# Prompts
prompts = [
    "Explain the importance of renewable energy.",
    "What is the carbon footprint of AI?",
    "How does machine learning help climate change?",
    "Explain large language models in simple words.",
    "What are sustainable AI practices?"
]

# Models
models = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large"
]

device = 0 if torch.cuda.is_available() else -1

results = []
carbon_results = []

for model_name in models:

    print(f"\n{'='*60}")
    print(f"Running {model_name}")
    print(f"{'='*60}")

    generator = pipeline(
        "text-generation",
        model=model_name,
        device=device
    )

    tracker = EmissionsTracker(
        project_name=model_name,
        output_file="codecarbon_logs.csv",
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
                "Model": model_name,
                "Run": run,
                "Prompt": prompt,
                "Runtime (s)": round(runtime, 4),
                "Generated Text": output[0]["generated_text"]
            })

    emissions = tracker.stop()

    carbon_results.append({
        "Model": model_name,
        "Total Runs": 25,
        "Total Runtime (s)": round(total_runtime, 4),
        "CO2 Emissions (kg)": emissions
    })

# Save files
experiment_df = pd.DataFrame(results)
experiment_df.to_csv("experiment_results.csv", index=False)

carbon_df = pd.DataFrame(carbon_results)
carbon_df.to_csv("carbon_emissions_report.csv", index=False)

print("\n✅ Experiment Completed Successfully!")

display(experiment_df.head())
display(carbon_df)

import platform
import psutil
import torch

hardware = {
    "Operating System": platform.platform(),
    "Python Version": platform.python_version(),
    "CPU": platform.processor(),
    "CPU Cores": psutil.cpu_count(logical=True),
    "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
    "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    "GPU Available": torch.cuda.is_available(),
    "CUDA Version": torch.version.cuda
}

hardware_df = pd.DataFrame(
    hardware.items(),
    columns=["Specification", "Value"]
)

hardware_df

flops = {
    "Model": ["GPT-2 Small", "GPT-2 Medium", "GPT-2 Large"],
    "Parameters (Millions)": [124, 355, 774],
    "Approx. FLOPs / Token": [
        "≈0.25 GFLOPs",
        "≈0.70 GFLOPs",
        "≈1.50 GFLOPs"
    ]
}

flops_df = pd.DataFrame(flops)

flops_df

from IPython.display import FileLink, display

display(FileLink("experiment_results.csv"))
display(FileLink("codecarbon_logs.csv"))
display(FileLink("carbon_emissions_report.csv"))