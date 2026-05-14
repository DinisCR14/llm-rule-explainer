"""Data loading and dataset-splitting utilities."""

from __future__ import annotations

import pandas as pd


def load_rules(file: str = "data/Allrules.xlsx", n: int = 25) -> pd.DataFrame:
    """Return the first *n* rules, descriptions, and names from an Excel file.

    Reads columns at positions 1, 2, 3 (zero-indexed), which correspond to
    columns B, C, and D in the spreadsheet.

    Parameters
    ----------
    file:
        Path to the Excel file.
    n:
        Number of rows to return (first *n* after dropping NaN rows).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Rule``, ``Description``, ``Name``.

    Raises
    ------
    FileNotFoundError
        If *file* does not exist.
    ValueError
        If the file contains no valid data or *n* exceeds the row count.
    RuntimeError
        If the file cannot be parsed.
    """
    try:
        df = pd.read_excel(file, usecols=[1, 2, 3]).dropna()
        df.columns = ["Rule", "Description", "Name"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found: {file}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read Excel file '{file}': {exc}") from exc

    if df.empty:
        raise ValueError(f"No valid data found in '{file}'.")

    if n > len(df):
        raise ValueError(
            f"Requested {n} rules but only {len(df)} are available in '{file}'."
        )

    return df.head(n).reset_index(drop=True)


def split_rules_dataset(
    file: str = "data/Allrules.xlsx",
    train_frac: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the rule set into training and test DataFrames.

    Parameters
    ----------
    file:
        Path to the Excel file passed to :func:`load_rules`.
    train_frac:
        Fraction of rows assigned to the training set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)``

    Raises
    ------
    ValueError
        If *train_frac* is not strictly between 0 and 1.
    """
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be strictly between 0 and 1.")

    rules_df = load_rules(file, n=50)

    # Sample before resetting the index so we can drop by original index
    train_df = rules_df.sample(frac=train_frac, random_state=42)
    test_df = rules_df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    return train_df, test_df
