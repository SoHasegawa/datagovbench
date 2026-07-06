# DataGovBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21225447.svg)](https://doi.org/10.5281/zenodo.21225447)

**NOTE: The complete benchmark will be publicly released with the camera-ready version.**

## Getting Started

### Data

**Quick-test sample (in-repo):** extract `datagovbench_sample.zip`.

**Full benchmark (Zenodo, ~2 GB):** archived at
[10.5281/zenodo.21225447](https://doi.org/10.5281/zenodo.21225447).

```bash
wget https://zenodo.org/records/21225447/files/datagovbench.tar.xz
wget https://zenodo.org/records/21225447/files/datagovbench.tar.xz.sha256
sha256sum -c datagovbench.tar.xz.sha256
tar -xJf datagovbench.tar.xz
```

The extracted directory is `opendatabench/`; pass its path as `--dataset` below.
See `LICENSES_THIRD_PARTY.md` inside the archive for per-dataset licenses.
The benchmark includes one CC-BY-NC-SA-4.0 dataset and is therefore intended
for non-commercial use only.

### Environment Setup

```bash
uv sync
. .venv/bin/activate

export OPENAI_API_KEY=<Your OpenAI API key>
export GEMINI_API_KEY=<Your Google AI Studio API key for Gemini>
export ANTHROPIC_API_KEY=<Your Anthropic API key>
```

## Run Benchmark

### Table QA

```bash
python -m benchmark.benchmark --dataset DATASET_PATH --type qa_evaluate --output qa --model MODEL_NAME

DATASET_PATH: Path to extracted dataset path
MODEL_NAME: LLM name (gpt4, gpt4-mini, gpt5, gpt5-mini, gemini, gemini-pro, claude, claude-sonnet, or huggingface model file names (e.g. mistralai/Devstral-Small-2507 DocTron/Chart-R1))
```

### Table Insight

```bash
python -m benchmark.benchmark --dataset DATASET_PATH --type report_evaluate --output report --model MODEL_NAME

DATASET_PATH: Path to extracted dataset path
MODEL_NAME: LLM name (gpt4, gpt4-mini, gpt5, gpt5-mini, gemini, gemini-pro, claude, claude-sonnet, or huggingface model file name (e.g. mistralai/Devstral-Small-2507 DocTron/Chart-R1))
```

