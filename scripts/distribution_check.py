"""Analyse the distribution of conditions and variables in the BAF rule set."""

from __future__ import annotations

import re
import sys
from collections import Counter


def analyze_rules(file_path: str) -> tuple[list[dict], Counter, Counter]:
    """Parse rule conditions from *file_path* and print summary statistics.

    Parameters
    ----------
    file_path:
        Path to a text file containing rule conditions prefixed with
        ``conds:``.

    Returns
    -------
    tuple[list[dict], Counter, Counter]
        ``(rule_data, dimensions_count, variable_count)``

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    with open(file_path, "r") as fh:
        content = fh.read()

    rules = re.findall(r"conds:\s*(.+)", content)
    rule_data: list[dict] = []
    variable_count: Counter = Counter()

    for rule in rules:
        conditions = rule.split("∧")
        variables = [
            re.split(r"[><=]", cond.strip())[0].strip() for cond in conditions
        ]
        variable_count.update(variables)
        rule_data.append(
            {"rule": rule, "dimensions": len(conditions), "variables": variables}
        )

    dimensions_count: Counter = Counter(r["dimensions"] for r in rule_data)

    print(f"Total rules: {len(rules)}\n")
    print("Distribution by number of conditions:")
    for dim, count in sorted(dimensions_count.items()):
        print(f"  {dim:2d} conditions: {count} rules")
    print("\nVariables by frequency:")
    for var, count in variable_count.most_common():
        print(f"  {var}: {count}")

    return rule_data, dimensions_count, variable_count


if __name__ == "__main__":
    import os

    # Default path: data/baf_rulesets.txt relative to the project root
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    _default_path = os.path.join(_project_root, "data", "baf_rulesets.txt")
    _file_path = sys.argv[1] if len(sys.argv) > 1 else _default_path

    analyze_rules(_file_path)
