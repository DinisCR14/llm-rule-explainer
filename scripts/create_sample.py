"""Stratified sampling of BAF rules by number of conditions."""

from __future__ import annotations

import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import List


def extract_rules(file_path: str) -> tuple[list[dict], Counter]:
    """Parse rule conditions from *file_path* and return per-rule metadata.

    Parameters
    ----------
    file_path:
        Path to the BAF ruleset text file.

    Returns
    -------
    tuple[list[dict], Counter]
        ``(rule_data, variable_count)`` where each entry in *rule_data* has
        keys ``rule``, ``dimensions``, ``variables``, and ``index``.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    with open(file_path, "r") as fh:
        lines = fh.readlines()

    rule_data: list[dict] = []
    variable_count: Counter = Counter()

    for rule_index, line in enumerate(lines, start=1):
        match = re.match(r"conds:\s*(.+)", line)
        if not match:
            continue
        rule = match.group(1)
        conditions = rule.split("∧")
        variables = [
            re.split(r"[><=]", cond.strip())[0].strip() for cond in conditions
        ]
        variable_count.update(variables)
        rule_data.append(
            {
                "rule": rule,
                "dimensions": len(conditions),
                "variables": variables,
                "index": rule_index,
            }
        )

    return rule_data, variable_count


def stratified_sample(
    rule_data: List[dict],
    sample_size: int,
    balance_variables: bool = True,
) -> List[dict]:
    """Draw a stratified sample of rules proportional to condition-count groups.

    Optionally applies a variable-balance heuristic that avoids
    over-representing any single feature variable relative to its frequency
    in the full dataset.

    Parameters
    ----------
    rule_data:
        Output of :func:`extract_rules`.
    sample_size:
        Desired number of sampled rules.
    balance_variables:
        When *True*, rules that would over-represent a variable are skipped.
        The returned sample may be smaller than *sample_size* in this case.

    Returns
    -------
    list[dict]
        Sampled rule dictionaries.
    """
    # Group rules by number of conditions
    stratified: defaultdict[int, list[dict]] = defaultdict(list)
    for rule in rule_data:
        stratified[rule["dimensions"]].append(rule)

    total = len(rule_data)
    sample_counts = {
        dim: int(round(sample_size * len(rules) / total))
        for dim, rules in stratified.items()
    }

    # Correct rounding so the total matches sample_size exactly
    while sum(sample_counts.values()) < sample_size:
        max_dim = max(sample_counts, key=sample_counts.__getitem__)
        sample_counts[max_dim] += 1

    full_var_count = Counter(
        var for rule in rule_data for var in rule["variables"]
    )

    selected: List[dict] = []
    selected_var_count: Counter = Counter()

    for dim, count in sample_counts.items():
        candidates = random.sample(
            stratified[dim], min(count, len(stratified[dim]))
        )
        for rule in candidates:
            if balance_variables:
                target_fraction = (len(selected) + 1) / sample_size
                if any(
                    selected_var_count[v] >= full_var_count[v] * target_fraction
                    for v in rule["variables"]
                ):
                    continue
            selected.append(rule)
            selected_var_count.update(rule["variables"])

    return selected


if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    _default_path = os.path.join(_project_root, "data", "baf_rulesets.txt")
    _file_path = sys.argv[1] if len(sys.argv) > 1 else _default_path

    _rule_data, _variable_count = extract_rules(_file_path)
    _sample = stratified_sample(_rule_data, sample_size=50)
    print(f"Extracted {len(_rule_data)} rules, sampled {len(_sample)}.")
    for entry in _sample:
        print(f"  [{entry['dimensions']} conditions] {entry['rule'][:80]}...")
