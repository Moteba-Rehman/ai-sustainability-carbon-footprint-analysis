import bitsandbytes as bnb
import transformers

print("Transformers:", transformers.__version__)
print("BitsAndBytes:", bnb.__version__)

from transformers.utils import is_bitsandbytes_available
print("Available:", is_bitsandbytes_available())

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,
    device_map="auto"
)

print("GPT-2 loaded successfully using BitsAndBytes INT8.")

from codecarbon import EmissionsTracker
import time
import pandas as pd

prompts = [
    "Artificial Intelligence is",
    "Climate change can be reduced by",
    "The future of education will",
    "Large Language Models are",
    "Renewable energy sources include"
]

results = []

tracker = EmissionsTracker(
    output_file="bnb_int8_codecarbon_logs.csv",
    save_to_file=True
)

tracker.start()

for i in range(25):

    prompt = prompts[i % len(prompts)]

    start = time.time()

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

    runtime = time.time() - start

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    results.append({
        "Run": i + 1,
        "Prompt": prompt,
        "Runtime (s)": runtime,
        "Output": generated
    })

tracker.stop()

results_df = pd.DataFrame(results)

results_df.to_csv(
    "bnb_int8_experiment_results.csv",
    index=False
)

results_df.head()

logs = pd.read_csv("bnb_int8_codecarbon_logs.csv")

summary = pd.DataFrame({
    "Model": ["GPT-2 BitsAndBytes INT8"],
    "Total Runs": [len(results_df)],
    "Total Runtime (s)": [results_df["Runtime (s)"].sum()],
    "Runtime per Inference (s)": [results_df["Runtime (s)"].mean()],
    "CO2 Emissions (kg)": [logs["emissions"].sum()],
    "CO2 per Inference (kg)": [
        logs["emissions"].sum() / len(results_df)
    ],
    "Estimated kWh per Inference": [
        logs["energy_consumed"].sum() / len(results_df)
    ],
    "GPU Hours": [
        results_df["Runtime (s)"].sum() / 3600
    ]
})

summary.to_csv(
    "bnb_int8_carbon_emissions_report.csv",
    index=False
)

summary

