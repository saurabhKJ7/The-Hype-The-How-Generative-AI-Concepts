# LLM Inference Calculator

This tool estimates the cost, latency, and memory usage of running large language models under different configurations.

## Project Structure

```
├── app.py
├── models/
│   └── inference_profiles.py
├── calculator/
│   └── engine.py
├── scenarios/
│   └── use_case_analysis.py
└── requirements.txt
```

## How to Use

### 1. Calculate Inference Metrics

You can estimate inference metrics for a specific configuration using the `calculate` command.

**Usage:**

```bash
python app.py calculate --model_size <size> --tokens <num> --batch_size <num> --hardware_type <type> --deployment_mode <mode>
```

**Example:**

```bash
python app.py calculate --model_size 7B --tokens 1000 --batch_size 1 --hardware_type T4 --deployment_mode local
```

### 2. Analyze Real-World Scenarios

You can analyze pre-defined, real-world use cases using the `scenarios` command.

**Usage:**

```bash
python app.py scenarios
```

This will output analysis for:
- A chatbot with 1k tokens on a T4 GPU.
- Batch summarization for 10 documents on an A100 GPU.
- Real-time code generation with 4 users using the GPT-4 API.

## Estimation Logic and Assumptions

The calculations are based on a combination of empirical benchmarks and scaling laws.

- **Latency:** Calculated as `base_latency + (token_latency * tokens)`. Batch processing introduces some overhead but provides parallel processing gains.
- **Memory:** For local deployments, memory is estimated as `base_model_memory + (token_memory * batch_size) + overhead`. API-based models do not have memory usage calculated on the client-side.
- **Cost:** For API-hosted models, the cost is based on the provider's pricing per token. For local deployments, the cost is estimated based on the hardware's cost per hour and the calculated latency.

## Model Profiles

The calculator uses pre-defined profiles for the following model sizes:

- **7B:** Based on models like LLaMA-2 7B and Mistral.
- **13B:** Based on models like LLaMA-2 13B.
- **GPT-4:** Based on OpenAI's GPT-4, treated as an API-only model.

Each profile in `models/inference_profiles.py` contains data on parameters, context window, latency, cost, and hardware requirements.