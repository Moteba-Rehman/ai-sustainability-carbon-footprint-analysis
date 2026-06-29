import time
import pandas as pd

from transformers import pipeline

MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large"
]

with open("data/prompts.txt", "r", encoding="utf-8") as file:
    prompts = [line.strip() for line in file if line.strip()]

results = []

for model_name in MODELS:

    print(f"\nLoading {model_name}...")

    generator = pipeline(
        "text-generation",
        model=model_name
    )

    for prompt in prompts:

        start = time.time()

        output = generator(
            prompt,
            max_new_tokens=50,
            do_sample=False
        )

        end = time.time()

        runtime = round(end - start, 4)

        generated_text = output[0]["generated_text"]

        results.append(
            {
                "model": model_name,
                "prompt": prompt,
                "runtime_seconds": runtime,
                "generated_text": generated_text
            }
        )

        print(f"{model_name}: {runtime} seconds")

df = pd.DataFrame(results)

df.to_csv(
    "inference_results.csv",
    index=False
)

print("\nSaved: inference_results.csv")