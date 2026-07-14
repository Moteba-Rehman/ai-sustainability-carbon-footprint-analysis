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
- Compare multiple quantization strategies for improving the energy efficiency of transformer-based language models.

---

## Repository Structure

```text
.
├── literature-review/
├── co2-tracking-tools/
├── gpt2-carbon-footprint/
├── tinyllama-carbon-footprint/
├── qwen-carbon-footprint/
├── mistral-carbon-footprint/
├── quantized-gpt2-carbon-footprint/
├── quantized-tinyllama-carbon-footprint/
├── instagram-data-collection/
├── tiktok-data-collection/
├── youtube-data-collection/
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

Evaluated multiple quantization methods for GPT-2 inference:

- FP16
- Dynamic INT8 Quantization (PyTorch)
- BitsAndBytes INT8
- BitsAndBytes 4-bit (NF4)

Generated:

- Quantized GPT-2 inference scripts
- CodeCarbon logs
- Carbon emission reports
- Hardware specification report
- FLOPs estimation report
- Quantization methods comparison dataset
- Performance visualizations

The experiments compare multiple quantization strategies under identical experimental conditions to evaluate their impact on runtime, energy consumption, GPU utilization, and estimated carbon emissions. The study highlights that lower numerical precision does not always result in faster inference, emphasizing the importance of hardware-aware evaluation for sustainable AI deployment.

### TinyLlama Quantization Energy Profiling

Evaluated the impact of multiple quantization techniques on TinyLlama inference.

Generated:

- FP16 inference implementation
- Dynamic INT8 inference
- BitsAndBytes INT8 inference
- BitsAndBytes 4-bit (NF4) inference
- CodeCarbon logs
- Carbon emission reports
- Comparison dataset
- Performance visualizations

---

### Social Media Data Collection

Collected publicly available datasets to support trend analysis of AI-generated Ghibli-style content across major social media platforms.

The same keywords were used consistently across all platforms:

- ghibliart
- ghiblistyle
- ghiblistudio
- ghiblitrend

Platforms completed:

#### Instagram

- Public posts collected using Apify
- Cleaned dataset exported as CSV

#### TikTok

- Public videos collected using Apify
- Duplicate removal and dataset cleaning
- Cleaned dataset exported as CSV

#### YouTube

- Public videos collected using yt-dlp
- Dataset generated using keyword-based search
- Cleaned dataset exported as CSV

Each platform contains:

- Data collection notebook
- Requirements file
- Cleaned dataset
- Supporting reports

---

### Comparative Analysis

Performed comparative analysis across:

- GPT-2 model variants (GPT-2, GPT-2 Medium, GPT-2 Large)
- GPT-2 quantization methods (FP32, FP16, Dynamic INT8, BitsAndBytes INT8, BitsAndBytes 4-bit)
- TinyLlama-1.1B-Chat-v1.0
- Qwen2.5-1.5B-Instruct
- Mistral-7B-Instruct-v0.3

Comparison metrics include:

- Total runtime
- Runtime per inference
- Total CO₂ emissions
- CO₂ emissions per inference
- Estimated energy consumption per inference
- GPU hours
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
- Social media datasets
- Data collection notebooks
- CodeCarbon logs
- Carbon emission reports
- Hardware specification reports
- FLOPs estimation reports
- Comparison reports
- Quantization methods comparison datasets
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
- Instagram Data Collection: https://www.kaggle.com/code/moteba/instagram-data-collection
- TikTok Data Collection: https://www.kaggle.com/code/moteba/tiktok-data-collection
- YouTube Data Collection: https://www.kaggle.com/code/moteba/youtube-data-collection

---

## Key Findings

- Carbon emissions generally increase with computational workload and inference time.
- Larger language models require greater computational resources and produce higher estimated carbon emissions.
- Efficient transformer architectures such as TinyLlama achieve lower environmental impact despite competitive performance.
- FP32 and FP16 achieved nearly identical inference performance and environmental impact for GPT-2 on the NVIDIA Tesla T4 GPU.
- Among the evaluated quantization methods, BitsAndBytes 4-bit (NF4) provided the best balance between runtime, energy consumption, and carbon emissions.
- Dynamic INT8 and BitsAndBytes INT8 introduced additional execution overhead despite reducing theoretical computational complexity.
- CodeCarbon provides detailed runtime, energy consumption, and carbon emission measurements suitable for AI workloads.
- Model architecture, quantization strategy, and hardware capabilities all influence the environmental sustainability of Generative AI systems.
- The impact of quantization varied across models and hardware configurations.
- FP16 provided the best performance for GPT-2.
- TinyLlama achieved its best performance using the FP32 baseline, while quantized variants introduced additional computational overhead.
- BitsAndBytes 4-bit consistently outperformed the other evaluated quantization techniques.
- Quantization does not universally reduce runtime or carbon emissions and should be evaluated experimentally.

---

## License

This repository is intended for educational and research purposes.