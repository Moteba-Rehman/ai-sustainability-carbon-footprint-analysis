# AI Sustainability: Carbon Footprint Analysis

A research project focused on understanding, measuring, and analyzing the carbon footprint of Generative AI systems through literature review, carbon accounting tools, and practical experimentation on open-source Large Language Models (LLMs).

---

## Project Overview

This repository contains the work completed as part of an 8-week research internship on **Generative AI Carbon Footprint Analysis**. The project investigates how Generative AI models consume computational resources, energy, and generate carbon emissions during inference. Carbon accounting tools and standardized experiments were used to evaluate the environmental impact of multiple open-source language models and optimization techniques.

The repository includes literature review reports, carbon tracking tool evaluations, experiment source code, datasets, comparison reports, presentations, visualizations, and supporting documentation.

---

## Objectives

- Understand the environmental impact of Generative AI systems through literature review.
- Study methodologies used to estimate AI-related carbon emissions.
- Evaluate carbon accounting tools for AI workloads.
- Measure runtime, energy consumption, and carbon emissions of language models.
- Compare multiple open-source LLMs under identical experimental conditions.
- Evaluate optimization techniques such as model quantization.
- Promote environmentally sustainable AI through carbon-aware experimentation.

---

## Repository Structure

```text
.
├── literature-review/
├── co2-tracking-tools/
├── gpt2-carbon-footprint/
├── llama-carbon-footprint/
├── qwen-carbon-footprint/
├── mistral-carbon-footprint/
├── quantized-gpt2-carbon-footprint/
├── comparison/
├── presentations/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Work Completed

### Literature Review

- Reviewed foundational and recent research on the environmental impact of Generative AI.
- Studied carbon footprint estimation methodologies.
- Compared major research findings and identified current research challenges.

---

### CO₂ Tracking Tools

Evaluated:

- CodeCarbon
- CarbonTracker

Activities included:

- Tool installation and configuration
- Documentation review
- Matrix multiplication experiment
- Large array sorting experiment
- Python loop experiment
- Runtime measurement
- Energy consumption measurement
- Carbon emission measurement
- Comparative analysis of both tools

---

### GPT-2 Carbon Footprint Analysis

Conducted inference experiments on:

- GPT-2
- GPT-2 Medium
- GPT-2 Large

Generated:

- Inference scripts
- CodeCarbon logs
- Carbon emission reports
- Hardware specification report
- FLOPs estimation report

---

### TinyLlama Carbon Footprint Analysis

Conducted inference experiments using:

- TinyLlama-1.1B-Chat-v1.0

Generated:

- Inference scripts
- CodeCarbon logs
- Carbon emission reports
- Hardware specification report
- FLOPs estimation report

---

### Qwen Carbon Footprint Analysis

Conducted inference experiments using:

- Qwen2.5-1.5B-Instruct

Generated:

- Inference scripts
- CodeCarbon logs
- Carbon emission reports
- Hardware specification report
- FLOPs estimation report

---

### Mistral Carbon Footprint Analysis

Conducted inference experiments using:

- Mistral-7B-Instruct-v0.3

Generated:

- Inference scripts
- CodeCarbon logs
- Carbon emission reports
- Hardware specification report
- FLOPs estimation report

---

### Quantized GPT-2 Energy Profiling

Evaluated the impact of **Dynamic INT8 Quantization** on GPT-2 inference.

Generated:

- Quantized GPT-2 inference scripts
- CodeCarbon logs
- Carbon emission reports
- Hardware specification report
- FLOPs estimation report
- Quantized GPT-2 comparison dataset

The experiments demonstrate how quantization improves computational efficiency while reducing runtime, energy consumption, and estimated carbon emissions without noticeably affecting inference quality.

---

### Comparative Analysis

Performed comparative analysis across:

- GPT-2 family
- TinyLlama-1.1B-Chat-v1.0
- Qwen2.5-1.5B-Instruct
- Mistral-7B-Instruct-v0.3
- Quantized GPT-2

Comparison metrics include:

- Total runtime
- Runtime per inference
- Total CO₂ emissions
- CO₂ emissions per inference
- GPU hours
- Estimated energy consumption
- Performance visualizations

---

## Tools & Technologies

- Python
- PyTorch
- Hugging Face Transformers
- CodeCarbon
- CarbonTracker
- Pandas
- NumPy
- Matplotlib
- Kaggle Notebooks
- Git
- GitHub
- Hugging Face

---

## Repository Contents

- Literature review reports
- CO₂ tracking tool reports
- Python source code
- Experiment datasets
- CodeCarbon logs
- Carbon emission reports
- Hardware specification reports
- FLOPs estimation reports
- Comparison reports
- CSV datasets
- Presentations
- Visualizations
- Screenshots

---

## Project Outputs

### GitHub Repository

https://github.com/Moteba-Rehman/ai-sustainability-carbon-footprint-analysis

### Hugging Face Space

https://huggingface.co/spaces/MotebaRehman/ai-carbon-footprint-dashboard

### Hugging Face Dataset

https://huggingface.co/datasets/MotebaRehman/ai-carbon-footprint-experiment-logs

### Kaggle Notebook

https://www.kaggle.com/code/moteba/ai-carbon-footprint-analysis-tool-comparison

---

## Key Findings

- Carbon emissions generally increase with computational workload and inference time.
- Larger language models require greater computational resources and produce higher estimated carbon emissions.
- Efficient transformer architectures such as TinyLlama achieve lower environmental impact despite competitive performance.
- Dynamic INT8 quantization significantly improves inference efficiency while maintaining comparable output quality.
- CodeCarbon provides detailed runtime, energy consumption, and carbon emission measurements suitable for AI workloads.
- Model architecture, optimization techniques, and hardware acceleration all play important roles in improving the sustainability of Generative AI systems.

---

## License

This repository is intended for educational and research purposes.