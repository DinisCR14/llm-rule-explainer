# rulescribe – LLM-powered Rule Annotation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**rulescribe** automatically generates human-readable **names** and **descriptions** for fraud-detection rules using prompt-engineered large language models. It implements and benchmarks four prompting strategies—zero-shot, few-shot, one-shot ranked by ROUGE, and one-shot ranked by structural similarity—and evaluates output quality with ROUGE metrics.

The framework was developed for the [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) dataset rule set and is designed to be straightforward to adapt to other rule sets.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Prompting Strategies](#prompting-strategies)
- [Evaluation](#evaluation)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

| Component | Description |
|-----------|-------------|
| `rulescribe/` | Core Python package |
| `scripts/` | CLI scripts for running experiments and analysing rule distributions |
| `data/` | BAF rule set text file |
| `notebooks/` | Jupyter notebooks for result visualisation |

---

## Project Structure

```
llm-rule-explainer/
├── rulescribe/                     # Core Python package
│   ├── __init__.py
│   ├── data_utils.py            # Rule loading & train/test splitting
│   ├── generator.py             # LLM pipeline factory
│   ├── prompting.py             # Zero-shot, few-shot, experiment runner
│   ├── evaluation.py            # ROUGE scoring & one-shot ranking
│   ├── similarity.py            # Structural & BERT similarity
│   ├── visualization.py         # Seaborn/Matplotlib plots
│   └── analysis.py              # Variable-level description analysis
├── data/
│   ├── Allrules.xlsx            # Annotated BAF rules (Rule / Description / Name)
│   └── baf_rulesets.txt         # Raw BAF fraud-detection rule conditions
├── docs/
│   └── README.md                # Links to report & slides (GitHub Releases)
├── notebooks/
│   ├── results.ipynb            # ROUGE score visualisation
│   └── plot_distributions.ipynb # Rule-set distribution plots
├── scripts/
│   ├── run_experiment.py        # Full pipeline CLI entry point
│   ├── distribution_check.py    # Rule distribution analysis
│   └── create_sample.py         # Stratified rule sampling
├── .env.example                 # Environment variable template
├── pyproject.toml               # Package metadata & build config
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

---

## Data

The annotated rule file (`data/Allrules.xlsx`) is included in this repository.
It contains rules derived from the
[Bank Account Fraud (BAF)](https://arxiv.org/abs/2211.13358) dataset,
annotated with human-written descriptions and short names.

The file must contain at least four columns in the following order (the first column
is ignored):

| Column index | Expected header | Contents |
|---|---|---|
| 1 | `Rule` | Raw rule condition string |
| 2 | `Description` | Human-written description |
| 3 | `Name` | Short rule name |

To explore rule distributions or create a stratified sample, use the scripts
in the `scripts/` folder (see [CLI Scripts](#cli-scripts)).

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/DinisCR14/llm-rule-explainer.git
cd llm-rule-explainer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your Hugging Face token
cp .env.example .env
# Edit .env and set HUGGINGFACE_API_TOKEN=<your_token>

# 5. Run a quick zero-shot experiment (GPT-2, no GPU required)
python -c "
from rulescribe import split_rules_dataset, run_experiment
train, test = split_rules_dataset('data/Allrules.xlsx')
run_experiment(train, test, strategy='zs', model='gpt2', n_test=5)
"
```

---

## Installation

### Requirements

- Python 3.10 or higher
- (Optional) CUDA-capable GPU for quantised large models

### Steps

```bash
pip install -r requirements.txt
```

For GPU support with large models (LLaMA, Falcon), ensure a compatible version of `bitsandbytes` and `torch` is installed for your CUDA version. Refer to the [bitsandbytes installation guide](https://github.com/TimDettmers/bitsandbytes).

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_API_TOKEN` | Yes | Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Required for gated models (LLaMA-2, Falcon). |

---

## Usage

### Programmatic API

```python
from rulescribe import split_rules_dataset, run_experiment, get_rouge_recall_f1, plot_rouge_scores

# Load and split the rule set
train_df, test_df = split_rules_dataset("data/Allrules.xlsx")

# Run a few-shot experiment with GPT-2
run_experiment(
    train_df, test_df,
    strategy="fs",        # "zs", "fs", "os_rank", or "os_sim"
    n_examples=2,
    model="gpt2",
    temperature=1.0,
    n_test=30,
    output_dir="Results",
    output_file="fs_gpt2_temp1.0.json",
)

# Visualise results
recall, f1 = get_rouge_recall_f1(["Results/fs_gpt2_temp1.0.json"])
plot_rouge_scores(recall, f1, labels=["FS"])
```

### CLI Scripts

```bash
# Full pipeline (all models, temperatures, strategies)
python scripts/run_experiment.py

# Check rule distribution
python scripts/distribution_check.py data/baf_rulesets.txt

# Create a stratified sample of 50 rules
python scripts/create_sample.py data/baf_rulesets.txt
```

---

## Prompting Strategies

| Strategy | Key | Description |
|----------|-----|-------------|
| Zero-shot | `"zs"` | Prompts the model with the rule only, no examples. |
| Few-shot | `"fs"` | Provides *n* labeled examples before the target rule. |
| One-shot ROUGE-ranked | `"os_rank"` | Selects the single training example that previously maximised ROUGE recall. |
| One-shot similarity-ranked | `"os_sim"` | Selects the training example structurally most similar to the target rule (custom score or BERT). |

All strategies support an optional `step_by_step=True` flag that appends a chain-of-thought cue (*"Let's think step by step."*).

---

## Evaluation

Generated names and descriptions are scored with **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** against ground-truth annotations. Results are saved as JSON files and can be visualised with the built-in plotting utilities.

```python
from rulescribe import get_rouge_recall_f1, plot_rouge_scores

recall, f1 = get_rouge_recall_f1(["Results/zs_gpt2.json", "Results/fs_gpt2.json"])
plot_rouge_scores(recall, f1, labels=["ZS", "FS"], model="gpt2")
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/results.ipynb` | ROUGE score comparison across all models and strategies |
| `notebooks/plot_distributions.ipynb` | Rule-set condition-count and variable-frequency distributions |

To launch:

```bash
jupyter notebook notebooks/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to report bugs, suggest features, and submit pull requests.

---

## Reports & Slides

The full technical report and presentation slides are available in the
[Releases](https://github.com/DinisCR14/llm-rule-explainer/releases) section of this repository.
See [docs/README.md](docs/README.md) for details.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
