import time
import pandas as pd

from transformers import pipeline
from codecarbon import EmissionsTracker


MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large"
]

RUNS = 6


with open("data/prompts.txt", "r", encoding="utf-8") as file:
    prompts = [line.strip() for line in file if line.strip()]


results = []


for model_name in MODELS:

    print(f"\nLoading {model_name}...")

    generator = pipeline(
        "text-generation",
        model=model_name
    )

    for run in range(1, RUNS + 1):

        tracker = EmissionsTracker(
            project_name="GPT2_Inference",
            output_file="codecarbon_logs.csv",
            save_to_file=True
        )

        tracker.start()

        start = time.time()

        for prompt in prompts:

            generator(
                prompt,
                max_new_tokens=50,
                do_sample=False
            )

        end = time.time()

        emissions = tracker.stop()

        runtime = round(
            end - start,
            4
        )

        results.append(
            {
                "model": model_name,
                "run": run,
                "runtime_seconds": runtime,
                "co2_kg": emissions
            }
        )

        print(
            f"{model_name} | Run {run} | {runtime} sec"
        )


df = pd.DataFrame(results)

df.to_csv(
    "experiment_results.csv",
    index=False
)

print("\nGenerated:")

print("experiment_results.csv")

print("codecarbon_logs.csv")