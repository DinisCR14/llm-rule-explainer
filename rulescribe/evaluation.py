"""ROUGE-based evaluation and one-shot example ranking."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from .prompting import _parse_name_description


_VALID_ROUGE_TYPES: frozenset[str] = frozenset({"rouge1", "rouge2", "rougeL"})


def rank_one_shot_examples(
    train_df: pd.DataFrame,
    test_rule: str,
    reference_description: str,
    reference_name: str,
    generator,
    scorer: rouge_scorer.RougeScorer,
    step_by_step: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank training examples by ROUGE-1 recall for a single test rule.

    For each training example a one-shot response is generated and scored
    against the reference outputs, separately for descriptions and names.

    Parameters
    ----------
    train_df:
        Training examples with columns ``Rule``, ``Description``, ``Name``.
    test_rule:
        The rule to annotate.
    reference_description, reference_name:
        Ground-truth outputs used for scoring.
    generator:
        Hugging Face text-generation pipeline.
    scorer:
        Pre-initialised ``rouge_score.rouge_scorer.RougeScorer``.
    step_by_step:
        Append a chain-of-thought cue when *True*.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(description_rankings_df, name_rankings_df)`` sorted by
        ROUGE-1 recall descending.
    """
    description_rankings = []
    name_rankings = []

    for _, train_row in train_df.iterrows():
        example_rule = train_row["Rule"]
        description = train_row["Description"]
        name = train_row["Name"]

        # One-shot prompt targeting description
        prompt_description = (
            f"Rule: {example_rule}\nDescription: {description}\n\n"
            f"Rule: {test_rule}\nDescription:"
        )
        if step_by_step:
            prompt_description += " Let's think step by step."

        out_desc = generator(
            prompt_description, max_length=1024, num_return_sequences=1, truncation=True
        )[0]["generated_text"]
        _, generated_description = _parse_name_description(out_desc, skip_count=1)

        # One-shot prompt targeting name
        prompt_name = (
            f"Rule: {example_rule}\nName: {name}\n\n"
            f"Rule: {test_rule}\nName:"
        )
        if step_by_step:
            prompt_name += " Let's think step by step."

        out_name = generator(
            prompt_name, max_length=1024, num_return_sequences=1, truncation=True
        )[0]["generated_text"]
        generated_name, _ = _parse_name_description(out_name, skip_count=1)

        desc_recall = (
            scorer.score(reference_description, generated_description)["rouge1"].recall
            if generated_description is not None
            else -np.inf
        )
        name_recall = (
            scorer.score(reference_name, generated_name)["rouge1"].recall
            if generated_name is not None
            else -np.inf
        )

        description_rankings.append(
            (example_rule, description, generated_description, desc_recall, prompt_description)
        )
        name_rankings.append(
            (example_rule, name, generated_name, name_recall, prompt_name)
        )

    description_rankings.sort(key=lambda x: x[3], reverse=True)
    name_rankings.sort(key=lambda x: x[3], reverse=True)

    description_df = pd.DataFrame(
        description_rankings,
        columns=["Rule", "Description", "Generated Description", "ROUGE Recall", "Prompt Description"],
    )
    name_df = pd.DataFrame(
        name_rankings,
        columns=["Rule", "Name", "Generated Name", "ROUGE Recall", "Prompt Name"],
    )
    return description_df, name_df


def get_rank_matrix(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    generator,
    step_by_step: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build ROUGE-1 recall rank matrices for all test/train rule pairs.

    Parameters
    ----------
    test_df, train_df:
        DataFrames with columns ``Rule``, ``Description``, ``Name``.
    generator:
        Hugging Face text-generation pipeline.
    step_by_step:
        Chain-of-thought cue flag.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(desc_rank_matrix, name_rank_matrix)`` where rows are test rules
        and columns are training rules.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    desc_rank_matrix = pd.DataFrame(0.0, index=test_df.index, columns=train_df.index)
    name_rank_matrix = pd.DataFrame(0.0, index=test_df.index, columns=train_df.index)

    for test_idx, test_row in test_df.iterrows():
        description_df, name_df = rank_one_shot_examples(
            train_df,
            test_row["Rule"],
            test_row["Description"],
            test_row["Name"],
            generator,
            scorer,
            step_by_step,
        )

        for train_idx, row in description_df.iterrows():
            desc_rank_matrix.loc[test_idx, train_idx] = row["ROUGE Recall"]
        for train_idx, row in name_df.iterrows():
            name_rank_matrix.loc[test_idx, train_idx] = row["ROUGE Recall"]

    return desc_rank_matrix, name_rank_matrix


def get_rouge_recall_f1(
    filenames: List[str],
    rouge_type: str = "rouge1",
    name_desc: str = "desc",
) -> Tuple[List[Optional[List[float]]], List[Optional[List[float]]]]:
    """Read JSON result files and extract ROUGE recall and F1 score lists.

    Each JSON file is expected to follow the structure written by
    :func:`~rulescribe.prompting.run_experiment`.

    Parameters
    ----------
    filenames:
        Paths to JSON result files.
    rouge_type:
        Which ROUGE variant to extract (``"rouge1"``, ``"rouge2"``,
        or ``"rougeL"``).
    name_desc:
        ``"desc"`` for Description ROUGE Scores,
        ``"name"`` for Name ROUGE Scores.

    Returns
    -------
    tuple[list, list]
        ``(recall_list, f1_list)`` – each entry corresponds to one file and is
        either a list of floats or *None* when the file contains no valid scores.

    Raises
    ------
    ValueError
        If *rouge_type* or *name_desc* are invalid.
    """
    if rouge_type not in _VALID_ROUGE_TYPES:
        raise ValueError(
            f"Invalid rouge_type '{rouge_type}'. Must be one of {_VALID_ROUGE_TYPES}."
        )

    _answer_part_map = {
        "name": "Name ROUGE Scores",
        "desc": "Description ROUGE Scores",
    }
    answer_part = _answer_part_map.get(name_desc)
    if answer_part is None:
        raise ValueError("name_desc must be 'name' or 'desc'.")

    recall_list: List[Optional[List[float]]] = []
    f1_list: List[Optional[List[float]]] = []

    for filename in filenames:
        try:
            with open(filename, "r") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            print(f"Error reading '{filename}': {exc}")
            recall_list.append(None)
            f1_list.append(None)
            continue

        recall_scores: List[float] = []
        f1_scores: List[float] = []

        for rule_info in data.values():
            if not isinstance(rule_info, dict) or answer_part not in rule_info:
                continue
            rouge_values = rule_info[answer_part].get(rouge_type, [])
            # ROUGE Score namedtuples are serialised as [precision, recall, fmeasure]
            if isinstance(rouge_values, list) and len(rouge_values) >= 3:
                recall_scores.append(rouge_values[1])
                f1_scores.append(rouge_values[2])
            else:
                print(
                    f"Unexpected format for '{rouge_type}' in '{filename}': "
                    f"{rouge_values}"
                )

        recall_list.append(recall_scores if recall_scores else None)
        f1_list.append(f1_scores if f1_scores else None)

    return recall_list, f1_list
