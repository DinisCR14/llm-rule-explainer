"""Variable-level analysis of generated rule descriptions."""

from __future__ import annotations

import re
from typing import Dict, List, Union


def check_conditions_in_des(
    description: str,
    variable_keywords: Dict[str, Union[str, list, tuple]],
) -> List[str]:
    """Return which variables from *variable_keywords* appear in *description*.

    Each entry in *variable_keywords* maps a variable name to one of:

    * ``str``   – a single keyword that must be present (whole-word match).
    * ``list``  – any keyword in the list must be present (OR logic).
    * ``tuple`` – ``str`` elements are AND conditions; inner ``list`` elements
      within the tuple are OR sub-conditions.

    Parameters
    ----------
    description:
        Generated rule description (lowercased internally).
    variable_keywords:
        Mapping of variable identifiers to their keyword patterns.

    Returns
    -------
    list[str]
        Variable names whose keyword conditions are satisfied.
    """
    matches: List[str] = []
    desc_lower = description.lower()

    def word_present(word: str) -> bool:
        return bool(re.search(r"\b" + re.escape(word.lower()) + r"\b", desc_lower))

    for key, value in variable_keywords.items():
        if isinstance(value, tuple):
            and_keywords = [v for v in value if isinstance(v, str)]
            or_groups = [v for v in value if isinstance(v, list)]
            if all(word_present(w) for w in and_keywords):
                if not or_groups or any(word_present(opt) for opt in or_groups[0]):
                    matches.append(key)
        elif isinstance(value, list):
            if any(word_present(w) for w in value):
                matches.append(key)
        else:
            if word_present(value):
                matches.append(key)

    return matches


def extract_variables(rule: str) -> List[str]:
    """Extract and normalise variable names from a rule condition string.

    ``housing_status_XX`` and ``device_os_YY`` variants are collapsed to
    their base prefixes.

    Parameters
    ----------
    rule:
        Rule condition string (uses ``∧`` as the conjunction operator).

    Returns
    -------
    list[str]
        Ordered list of normalised variable names appearing in the rule.
    """
    raw_vars = re.findall(r"(\w+_?\w*)\s*[<>=]+", rule)
    normalised: List[str] = []
    for var in raw_vars:
        if var.startswith("housing_status_") and len(var) == len("housing_status_XX"):
            normalised.append("housing_status")
        elif var.startswith("device_os_") and len(var) > len("device_os_"):
            normalised.append("device_os")
        else:
            normalised.append(var)
    return normalised


def compare_rule_and_description(
    rule: str,
    description: str,
    variable_keywords: Dict[str, Union[str, list, tuple]],
) -> dict:
    """Compare variables referenced in a rule vs. those mentioned in a description.

    Parameters
    ----------
    rule:
        Rule condition string.
    description:
        Generated description text.
    variable_keywords:
        Passed to :func:`check_conditions_in_des`.

    Returns
    -------
    dict
        Keys:

        * ``variables_in_rule`` – variables extracted from the rule.
        * ``variables_in_description`` – variables detected in the description.
        * ``variables_in_rule_found`` – intersection of the two sets.
        * ``recall`` – fraction of rule variables mentioned in the description.
        * ``precision`` – fraction of description variables that are in the rule.
        * ``f1_score`` – harmonic mean of precision and recall.
        * ``jaccard_similarity`` – |intersection| / |union|.
        * ``extraneous_proportion`` – fraction of description variables absent
          from the rule.
    """
    vars_in_rule = set(extract_variables(rule))
    vars_in_desc = set(check_conditions_in_des(description, variable_keywords))

    intersection = vars_in_rule & vars_in_desc
    union = vars_in_rule | vars_in_desc

    recall = len(intersection) / len(vars_in_rule) if vars_in_rule else 0.0
    precision = len(intersection) / len(vars_in_desc) if vars_in_desc else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = len(intersection) / len(union) if union else 0.0
    extraneous = (
        len(vars_in_desc - vars_in_rule) / len(vars_in_desc) if vars_in_desc else 0.0
    )

    return {
        "variables_in_rule": list(vars_in_rule),
        "variables_in_description": list(vars_in_desc),
        "variables_in_rule_found": list(intersection),
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "jaccard_similarity": jaccard,
        "extraneous_proportion": extraneous,
    }
