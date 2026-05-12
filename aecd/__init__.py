"""AECD – Automated Explanation of Credit Decisions.

This package provides utilities for generating human-readable names and
descriptions for fraud-detection rules using large language models, and
for evaluating those generations with ROUGE metrics.

Typical usage
-------------
>>> from aecd import load_generator, split_rules_dataset, run_experiment
>>> train_df, test_df = split_rules_dataset("data/Allrules.xlsx")
>>> run_experiment(train_df, test_df, strategy="fs", model="gpt2")
"""

from .analysis import check_conditions_in_des, compare_rule_and_description, extract_variables
from .data_utils import load_rules, split_rules_dataset
from .evaluation import get_rank_matrix, get_rouge_recall_f1, rank_one_shot_examples
from .generator import load_generator
from .prompting import few_shot, run_experiment, zero_shot
from .similarity import (
    bert_similarity_score,
    calculate_correlation,
    custom_similarity_score,
    encode_rule,
    extract_variable_symbol_pairs,
    get_similarity_matrix,
    rank_similar_rules,
)
from .visualization import (
    plot_rouge_means_with_seaborn,          # backward-compatible alias
    plot_rouge_means_with_seaborn_combined, # backward-compatible alias
    plot_rouge_scores,
    plot_rouge_scores_combined,
    plot_shot_results,
)

# Backward-compatible aliases for renamed functions
get_rules_descriptions_names = load_rules
set_generator = load_generator
zero_shot_prompting = zero_shot
few_shot_prompting = few_shot
prompting = run_experiment  # original function name

__all__ = [
    # data_utils
    "load_rules",
    "split_rules_dataset",
    # generator
    "load_generator",
    # prompting
    "zero_shot",
    "few_shot",
    "run_experiment",
    # evaluation
    "rank_one_shot_examples",
    "get_rank_matrix",
    "get_rouge_recall_f1",
    # similarity
    "extract_variable_symbol_pairs",
    "custom_similarity_score",
    "encode_rule",
    "bert_similarity_score",
    "rank_similar_rules",
    "get_similarity_matrix",
    "calculate_correlation",
    # visualization
    "plot_rouge_scores",
    "plot_rouge_scores_combined",
    "plot_shot_results",
    "plot_rouge_means_with_seaborn",
    "plot_rouge_means_with_seaborn_combined",
    # analysis
    "check_conditions_in_des",
    "extract_variables",
    "compare_rule_and_description",
    # backward-compatible aliases
    "get_rules_descriptions_names",
    "set_generator",
    "zero_shot_prompting",
    "few_shot_prompting",
    "prompting",
]
