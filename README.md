# DataGovBench

**NOTE: The complete benchmark will be publicly released with the camera-ready version.**

## Getting Started

### Preparation of Sample Data
Please extract the zip file (`datagovbench_sample.zip`).

### Environment Setup

```bash
uv sync
. .venv/bin/activate

export AZURE_OPENAI_API_KEY=<Your azure openai api key>
export AZURE_OPENAI_ENDPOINT=<Endpoint for GPT-4o>
export AZURE_OPENAI_MINI_ENDPOINT=<Endpoint for GPT-4o-mini>
export GEMINI_API_KEY=<Your api key for Gemini>
export GEMINI_ENDPOINT=<Endpoint for Gemini Flash>
export GEMINI_PRO_ENDPOINT=<Endpoint for Gemini Pro>
```

## Run Benchmark

### Table QA

```bash
python -m benchmark.benchmark --dataset DATASET_PATH --type qa_evaluate --output qa --model MODEL_NAME

DATASET_PATH: Path to extracted dataset path
MODEL_NAME: LLM name (gemini, gpt4, huggingface model file names (e.g. mistralai/Devstral-Small-2507 DocTron/Chart-R1))
```

### Table Insight

```bash
python -m benchmark.benchmark --dataset DATASET_PATH --type report_evaluate --output report --model MODEL_NAME

DATASET_PATH: Path to extracted dataset path
MODEL_NAME: LLM name (gemini, gpt4, huggingface model file name (e.g. mistralai/Devstral-Small-2507 DocTron/Chart-R1))
```

