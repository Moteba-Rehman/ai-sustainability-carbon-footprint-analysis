import torch
import bitsandbytes as bnb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from codecarbon import EmissionsTracker

print("Transformers:", __import__("transformers").__version__)
print("BitsAndBytes:", bnb.__version__)
print("CUDA Available:", torch.cuda.is_available())

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.eval()

print("TinyLlama loaded successfully using BitsAndBytes INT8.")

results = []

tracker = EmissionsTracker(
    output_file="llama_bnb_int8_codecarbon_logs.csv",
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
    "llama_bnb_int8_experiment_results.csv",
    index=False
)

results_df.head()

logs = pd.read_csv("llama_bnb_int8_codecarbon_logs.csv")

summary = pd.DataFrame({
    "Model": ["TinyLlama BitsAndBytes INT8"],
    "Total Runs": [len(results_df)],
    "Total Runtime (s)": [results_df["Runtime (s)"].sum()],
    "Runtime per Inference (s)": [results_df["Runtime (s)"].mean()],
    "CO2 Emissions (kg)": [logs["emissions"].sum()],
    "CO2 per Inference (kg)": [logs["emissions"].sum() / len(results_df)],
    "Estimated kWh per Inference": [logs["energy_consumed"].sum() / len(results_df)],
    "GPU Hours": [results_df["Runtime (s)"].sum() / 3600]
})

summary.to_csv(
    "llama_bnb_int8_carbon_emissions_report.csv",
    index=False
)

summary