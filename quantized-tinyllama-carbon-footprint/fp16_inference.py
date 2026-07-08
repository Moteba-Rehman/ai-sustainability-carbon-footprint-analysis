import time
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from codecarbon import EmissionsTracker

# TinyLlama FP16 Model Loading

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("TinyLlama loaded successfully using FP16.")

prompts = [
    "Artificial Intelligence is",
    "Climate change can be reduced by",
    "The future of education will",
    "Large Language Models are",
    "Renewable energy sources include"
]

results = []

tracker = EmissionsTracker(
    output_file="llama_fp16_codecarbon_logs.csv",
    save_to_file=True
)

tracker.start()

for i in range(5):

    for j, prompt in enumerate(prompts):

        start = time.time()

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

        runtime = time.time() - start

        generated = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        results.append({
            "Run": i + 1,
            "Prompt": j + 1,
            "Runtime (s)": runtime,
            "Output": generated
        })

tracker.stop()

results_df = pd.DataFrame(results)

results_df.to_csv(
    "llama_fp16_experiment_results.csv",
    index=False
)

results_df.head()

logs = pd.read_csv("llama_fp16_codecarbon_logs.csv")

summary = pd.DataFrame({
    "Model": ["TinyLlama FP16"],
    "Total Runs": [len(results_df)],
    "Total Runtime (s)": [results_df["Runtime (s)"].sum()],
    "Runtime per Inference (s)": [results_df["Runtime (s)"].mean()],
    "CO2 Emissions (kg)": [logs["emissions"].sum()],
    "CO2 per Inference (kg)": [logs["emissions"].sum() / len(results_df)],
    "Estimated kWh per Inference": [logs["energy_consumed"].sum() / len(results_df)],
    "GPU Hours": [results_df["Runtime (s)"].sum() / 3600]
})

summary.to_csv(
    "llama_fp16_carbon_emissions_report.csv",
    index=False
)

summary