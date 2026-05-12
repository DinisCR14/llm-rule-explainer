# Contributing to AECD

Thank you for taking the time to contribute! This document explains how to
report bugs, propose features, and submit pull requests.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Suggest a Feature](#how-to-suggest-a-feature)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Submitting a Pull Request](#submitting-a-pull-request)

---

## Code of Conduct

Be respectful and constructive in all interactions. Harassment of any kind
will not be tolerated.

---

## How to Report a Bug

1. Search [existing issues](../../issues) to avoid duplicates.
2. Open a new issue and include:
   - A clear, descriptive title.
   - Steps to reproduce the problem.
   - The full error message / traceback.
   - Your Python version and OS.
   - Relevant package versions (`pip freeze | grep -E "torch|transformers|rouge"`).

---

## How to Suggest a Feature

Open an issue with the label **enhancement** and describe:

- The problem it solves.
- A proposed API or workflow change.
- Any alternatives you considered.

---

## Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/DinisCR14/llm-rule-explainer.git
cd llm-rule-explainer

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy the environment template
cp .env.example .env
# Edit .env and add your HUGGINGFACE_API_TOKEN
```

---

## Coding Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) and use type annotations.
- Keep functions focused; prefer small, composable units over large monoliths.
- Use `from __future__ import annotations` in all new modules.
- Write docstrings in [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).
- Do not commit `.env`, model weights, or large result files.
- Run your code against a lightweight model (e.g., `gpt2`) to confirm it works before opening a PR.

---

## Submitting a Pull Request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes, keeping commits small and focused.
3. Ensure the code is importable and the notebooks can still find `from aecd import *`.
4. Push to your fork and open a pull request against `main`.
5. Fill in the pull request template, linking the related issue.

Pull requests are reviewed on a best-effort basis. Please be patient and
responsive to feedback.
