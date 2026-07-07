from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,
    device_map="auto"
)

model.eval()

print("GPT-2 loaded successfully using BitsAndBytes 4-bit (NF4).")

results = []

tracker = EmissionsTracker(
    project_name="gpt2-bitsandbytes-4bit",
    output_file="bnb_4bit_codecarbon_logs.csv"
)

tracker.start()

for run in range(5):
    for prompt in prompts:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(model.device)

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
    "bnb_4bit_experiment_results.csv",
    index=False
)

print(results_df.head())
print(f"\nTotal inference runs: {len(results_df)}")

logs = pd.read_csv("bnb_4bit_codecarbon_logs.csv")

summary = pd.DataFrame({
    "Model": ["BitsAndBytes 4-bit (NF4)"],
    "Total Runs": [len(results_df)],
    "Total Runtime (s)": [results_df["Runtime (s)"].sum()],
    "CO2 Emissions (kg)": [logs["emissions"].sum()]
})

summary["CO2 per Inference (kg)"] = (
    summary["CO2 Emissions (kg)"] /
    summary["Total Runs"]
)

summary["Runtime per Inference (s)"] = (
    summary["Total Runtime (s)"] /
    summary["Total Runs"]
)

summary["GPU Hours"] = (
    summary["Total Runtime (s)"] /
    3600
)

summary["Estimated kWh per Inference"] = (
    logs["energy_consumed"].sum() /
    summary["Total Runs"]
)

summary.to_csv(
    "bnb_4bit_carbon_emissions_report.csv",
    index=False
)

summary