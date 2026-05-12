"""Rule-similarity scoring and ranking utilities."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


def extract_variable_symbol_pairs(rule: str) -> list[tuple[str, str]]:
    """Extract ``(variable, operator)`` pairs from a rule condition string."""
    return re.findall(r"(\w+_?\w*)\s*([<>=]+)", rule)


def custom_similarity_score(
    rule: str,
    train_rule: str,
    order_factor: float = 1.0,
) -> float:
    """Compute a structure-aware similarity score between two rules.

    Variables that appear at the same position in both rules receive a higher
    order bonus than those at different positions.

    Parameters
    ----------
    rule, train_rule:
        Rule condition strings to compare.
    order_factor:
        Scale of the positional-order bonus (default ``1.0``).

    Returns
    -------
    float
        Non-negative similarity score; higher means more similar.
    """
    rule_pairs = extract_variable_symbol_pairs(rule)
    train_pairs = extract_variable_symbol_pairs(train_rule)

    # Build a map from variable name → list of (operator, position) in rule
    rule_index_map: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx, (var, sym) in enumerate(rule_pairs):
        rule_index_map[var].append((sym, idx))

    score = 0.0
    for idx, (train_var, train_sym) in enumerate(train_pairs):
        for rule_sym, rule_idx in rule_index_map.get(train_var, []):
            base_score = 1.0 if rule_sym == train_sym else 0.5
            order_bonus = order_factor / (1 + abs(rule_idx - idx))
            score += base_score + order_bonus

    return score


def encode_rule(rule: str, tokenizer, model) -> np.ndarray:
    """Encode a rule string into a dense CLS-token embedding via BERT.

    Parameters
    ----------
    rule:
        Rule condition string.
    tokenizer, model:
        Pre-loaded BERT tokenizer and model.

    Returns
    -------
    np.ndarray
        1-D float32 embedding vector.
    """
    inputs = tokenizer(rule, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def bert_similarity_score(tokenizer, model, rule: str, train_rule: str) -> float:
    """Cosine similarity between two BERT CLS-token embeddings.

    Parameters
    ----------
    tokenizer, model:
        Pre-loaded ``bert-base-uncased`` tokenizer and model.
    rule, train_rule:
        Rule condition strings to compare.

    Returns
    -------
    float
        Cosine similarity in ``[-1, 1]``.
    """
    rule_emb = encode_rule(rule, tokenizer, model)
    train_emb = encode_rule(train_rule, tokenizer, model)
    return float(
        cosine_similarity(rule_emb.reshape(1, -1), train_emb.reshape(1, -1))[0][0]
    )


def rank_similar_rules(
    rule: str,
    train_df: pd.DataFrame,
    use_bert: bool = False,
) -> pd.DataFrame:
    """Rank training rules by similarity to *rule*, highest first.

    When *use_bert* is ``True`` a ``bert-base-uncased`` model is loaded
    once before iterating over training rules.

    Parameters
    ----------
    rule:
        Query rule condition string.
    train_df:
        DataFrame with columns ``Rule``, ``Description``, ``Name``.
    use_bert:
        Use BERT embeddings instead of the custom structural score.

    Returns
    -------
    pd.DataFrame
        Training rules sorted by descending similarity, without the score column.
    """
    tokenizer: Optional[AutoTokenizer] = None
    bert_model: Optional[AutoModel] = None

    if use_bert:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")

    rankings = []
    for _, row in train_df.iterrows():
        train_rule = row["Rule"]
        if use_bert:
            score = bert_similarity_score(tokenizer, bert_model, rule, train_rule)
        else:
            score = custom_similarity_score(rule, train_rule)
        rankings.append((train_rule, row["Description"], row["Name"], score))

    return (
        pd.DataFrame(rankings, columns=["Rule", "Description", "Name", "Score"])
        .sort_values("Score", ascending=False)
        .drop(columns=["Score"])
        .reset_index(drop=True)
    )


def get_similarity_matrix(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise structural similarity between all test and train rules.

    Parameters
    ----------
    test_df, train_df:
        DataFrames with a ``Rule`` column.

    Returns
    -------
    pd.DataFrame
        Matrix with test-rule indices as rows and train-rule indices as
        columns, sorted by column index.
    """
    similarity_matrix = pd.DataFrame(
        index=test_df.index, columns=train_df.index, dtype=float
    )
    for test_idx, test_row in test_df.iterrows():
        for train_idx, train_row in train_df.iterrows():
            similarity_matrix.at[test_idx, train_idx] = custom_similarity_score(
                test_row["Rule"], train_row["Rule"]
            )
    return similarity_matrix.sort_index(axis=1)


def calculate_correlation(
    similarity_matrix: pd.DataFrame,
    rank_matrix: pd.DataFrame,
    method: str = "spearman",
) -> float:
    """Correlation between a similarity matrix and a ROUGE rank matrix.

    Parameters
    ----------
    similarity_matrix, rank_matrix:
        Must have the same shape.
    method:
        ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    float
        Correlation coefficient.
    """
    sim_flat = similarity_matrix.values.flatten()
    rank_flat = rank_matrix.values.flatten()
    mask = ~pd.isna(sim_flat) & ~pd.isna(rank_flat)
    return pd.Series(sim_flat[mask]).corr(pd.Series(rank_flat[mask]), method=method)
