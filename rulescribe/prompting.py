"""Prompting strategies: zero-shot, few-shot, one-shot-ranked, and experiment runner."""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Valid prompting strategy identifiers
_VALID_STRATEGIES: frozenset[str] = frozenset({"zs", "fs", "os_rank", "os_sim"})


def _parse_name_description(
    text: str, skip_count: int
) -> Tuple[Optional[str], Optional[str]]:
    """Extract the (skip_count + 1)-th Name/Description pair from generated text.

    Parameters
    ----------
    text:
        Raw model output string.
    skip_count:
        Number of example pairs to skip before extracting the target pair.

    Returns
    -------
    tuple[str | None, str | None]
        ``(name, description)``; either element may be *None* if not found.
    """
    name_matches = list(re.finditer(r"(?<=Name:).*?(?=\n|$)", text, re.DOTALL))
    desc_matches = list(re.finditer(r"(?<=Description:).*?(?=\n|$)", text, re.DOTALL))

    name = (
        name_matches[skip_count].group(0).strip()
        if len(name_matches) > skip_count
        else None
    )
    description = (
        desc_matches[skip_count].group(0).strip()
        if len(desc_matches) > skip_count
        else None
    )
    return name, description


def zero_shot(
    rule: str,
    generator,
    step_by_step: bool = False,
) -> Tuple[str, Optional[str], Optional[str], str]:
    """Generate a name and description for *rule* with no in-context examples.

    Parameters
    ----------
    rule:
        Fraud-detection rule condition string.
    generator:
        Hugging Face text-generation pipeline (see :mod:`rulescribe.generator`).
    step_by_step:
        Append a chain-of-thought cue when *True*.

    Returns
    -------
    tuple[str, str | None, str | None, str]
        ``(prompt, name, description, raw_output)``

    Raises
    ------
    ValueError
        If *rule* is empty.
    RuntimeError
        If the model fails to generate text.
    """
    if not rule:
        raise ValueError("rule must not be empty.")

    prompt = (
        "Create a name and a description for this rule.\n"
        f"Rule:{rule}\n"
        "Name:\n"
        "Description:\n"
    )
    if step_by_step:
        prompt += "\nLet's think step by step."

    try:
        raw = generator(
            prompt, max_length=300, num_return_sequences=1, truncation=True
        )[0]["generated_text"]
    except Exception as exc:
        raise RuntimeError(f"Text generation failed: {exc}") from exc

    name, description = _parse_name_description(raw, 0)
    return prompt, name, description, raw


def few_shot(
    examples_df: pd.DataFrame,
    rule: str,
    generator,
    step_by_step: bool = False,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Generate a name and description using in-context examples.

    Parameters
    ----------
    examples_df:
        DataFrame with columns ``Rule``, ``Description``, ``Name`` used as
        in-context examples.
    rule:
        Target rule to annotate.
    generator:
        Hugging Face text-generation pipeline.
    step_by_step:
        Append a chain-of-thought cue when *True*.

    Returns
    -------
    tuple[str | None, ...]
        ``(prompt, name, description, raw_output)``; all *None* when the
        prompt exceeds the model's context window.

    Raises
    ------
    ValueError
        If *rule* is empty.
    RuntimeError
        If the model fails to generate text.
    """
    if not rule:
        raise ValueError("rule must not be empty.")

    prompt = ""
    for _, row in examples_df.iterrows():
        prompt += (
            f"Rule: {row['Rule']}\n"
            f"Description: {row['Description']}\n"
            f"Name: {row['Name']}\n\n"
        )
    prompt += f"Rule: {rule}\n"
    if step_by_step:
        prompt += "Let's think step by step."

    tokenizer = AutoTokenizer.from_pretrained(generator.model.config._name_or_path)
    n_prompt_tokens = len(tokenizer(prompt)["input_ids"])
    max_ctx = generator.model.config.max_position_embeddings

    if n_prompt_tokens >= max_ctx:
        print(f"Prompt too long ({n_prompt_tokens} tokens). Skipping.")
        return None, None, None, None

    max_new_tokens = max(max_ctx - n_prompt_tokens, 1)
    try:
        raw = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            truncation=True,
        )[0]["generated_text"]
    except Exception as exc:
        raise RuntimeError(f"Text generation failed: {exc}") from exc

    name, description = _parse_name_description(raw, len(examples_df))
    return prompt, name, description, raw


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str = "zs",
    step_by_step: bool = False,
    n_examples: int = 2,
    n_test: int = 30,
    model: str = "gpt2",
    temperature: float = 1.0,
    use_bert: bool = False,
    output_dir: str = "Results",
    output_file: str = "rules_with_rouge.json",
) -> None:
    """Run a prompting experiment and save ROUGE-scored results to a JSON file.

    Parameters
    ----------
    train_df, test_df:
        DataFrames with columns ``Rule``, ``Description``, ``Name``.
    strategy:
        Prompting strategy: ``"zs"`` (zero-shot), ``"fs"`` (few-shot),
        ``"os_rank"`` (one-shot ranked by ROUGE), or ``"os_sim"`` (one-shot
        ranked by structural similarity).
    step_by_step:
        Append a chain-of-thought cue when *True*.
    n_examples:
        Number of in-context examples (ignored for ``"zs"``).
    n_test:
        Number of test rules to evaluate.
    model:
        Hugging Face model identifier.
    temperature:
        Sampling temperature.
    use_bert:
        Use BERT-based similarity scoring for the ``"os_sim"`` strategy.
    output_dir:
        Directory where the JSON result file will be written.
    output_file:
        Filename for the JSON result file.

    Raises
    ------
    ValueError
        If *strategy* is not one of the supported options.
    """
    from rouge_score import rouge_scorer as _rouge_scorer

    from .evaluation import rank_one_shot_examples
    from .generator import load_generator
    from .similarity import rank_similar_rules

    # Validate cheaply before the expensive model load
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of {_VALID_STRATEGIES}."
        )

    generator = load_generator(model=model, temperature=temperature)
    test_sample = test_df.sample(n=n_test, random_state=42)
    scorer = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results: dict = {}
    for _, row in tqdm(
        test_sample.iterrows(), total=len(test_sample), desc="Processing rules"
    ):
        rule = row["Rule"]
        ref_name = row["Name"]
        ref_desc = row["Description"]

        prompt: Optional[str] = None
        generated_name: Optional[str] = None
        generated_description: Optional[str] = None

        if strategy == "zs":
            try:
                prompt, generated_name, generated_description, _ = zero_shot(
                    rule, generator, step_by_step
                )
            except ValueError as exc:
                print(f"Warning: {exc} – skipping rule: {rule!r}")
                generated_name = "No Name Generated"
                generated_description = "No Description Generated"

        elif strategy == "fs":
            examples = train_df.sample(n=n_examples)
            result = few_shot(examples, rule, generator, step_by_step)
            if result[0] is None:
                prompt = "Prompt was too long"
                generated_name = "No Name Generated"
                generated_description = "No Description Generated"
            else:
                prompt, generated_name, generated_description, _ = result

        elif strategy == "os_rank":
            desc_rankings, name_rankings = rank_one_shot_examples(
                train_df, rule, ref_desc, ref_name, generator, scorer, step_by_step
            )
            top_desc = desc_rankings.iloc[0] if not desc_rankings.empty else None
            top_name = name_rankings.iloc[0] if not name_rankings.empty else None
            try:
                prompt = (top_name["Prompt Name"] or "") + (
                    top_desc["Prompt Description"] or ""
                )
                generated_name = (
                    top_name["Generated Name"] if top_name is not None else "No Name Generated"
                )
                generated_description = (
                    top_desc["Generated Description"]
                    if top_desc is not None
                    else "No Description Generated"
                )
            except (KeyError, TypeError) as exc:
                print(f"Warning: {exc} – skipping rule: {rule!r}")
                generated_name = "No Name Generated"
                generated_description = "No Description Generated"

        elif strategy == "os_sim":
            rankings = rank_similar_rules(rule, train_df, use_bert)
            top_example = rankings.iloc[[0]]
            result = few_shot(top_example, rule, generator, step_by_step)
            if result[0] is None:
                prompt = "Prompt was too long"
                generated_name = "No Name Generated"
                generated_description = "No Description Generated"
            else:
                prompt, generated_name, generated_description, _ = result

        # Score descriptions
        if ref_desc and generated_description:
            desc_scores = scorer.score(ref_desc, generated_description)
        else:
            desc_scores = {"rouge1": None, "rouge2": None, "rougeL": None}
            print(f"Skipping description scoring for rule: {rule!r}")

        # Score names
        if ref_name and generated_name:
            name_scores = scorer.score(ref_name, generated_name)
        else:
            name_scores = {"rouge1": None, "rouge2": None, "rougeL": None}
            print(f"Skipping name scoring for rule: {rule!r}")

        results[rule] = {
            "Type of Prompting": strategy,
            "Prompt": prompt,
            "Generated Name": generated_name,
            "Reference Name": ref_name,
            "Name ROUGE Scores": name_scores,
            "Generated Description": generated_description,
            "Reference Description": ref_desc,
            "Description ROUGE Scores": desc_scores,
        }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=4)
    print(f"Results saved to '{output_path}'.")
