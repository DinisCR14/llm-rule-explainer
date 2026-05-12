"""Plotting utilities for ROUGE score analysis."""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rouge_scores(
    recall_scores: List[Optional[List[float]]],
    f1_scores: List[Optional[List[float]]],
    labels: List[str],
    model: str = "gpt2",
    desc_name: str = "desc",
    bars_points: str = "bars",
) -> None:
    """Bar or point plot of mean ROUGE Recall and F1 scores with 95% CI.

    Parameters
    ----------
    recall_scores, f1_scores:
        Nested lists of scores, one inner list per label.  *None* entries
        are skipped with a warning.
    labels:
        Display labels, one entry per score group.
    model:
        Model name used in the plot title.
    desc_name:
        ``"desc"`` for descriptions, ``"name"`` for names.
    bars_points:
        ``"bars"`` for a bar plot, ``"points"`` for a point plot.

    Raises
    ------
    ValueError
        If *desc_name* or *bars_points* are invalid.
    """
    if desc_name not in {"desc", "name"}:
        raise ValueError("desc_name must be 'desc' or 'name'.")
    if bars_points not in {"bars", "points"}:
        raise ValueError("bars_points must be 'bars' or 'points'.")

    data: dict = {"Label": [], "Score Type": [], "Score": []}
    for label, recall, f1 in zip(labels, recall_scores, f1_scores):
        if recall is None or f1 is None:
            print(f"Skipping label '{label}' due to None values.")
            continue
        data["Label"].extend([label] * len(recall))
        data["Score Type"].extend(["Recall"] * len(recall))
        data["Score"].extend(recall)
        data["Label"].extend([label] * len(f1))
        data["Score Type"].extend(["F1"] * len(f1))
        data["Score"].extend(f1)

    df = pd.DataFrame(data)
    plt.figure(figsize=(13, 5))

    if bars_points == "bars":
        sns.barplot(
            data=df,
            x="Label",
            y="Score",
            hue="Score Type",
            errorbar=("ci", 95),
            palette=["skyblue", "salmon"],
        )
    else:
        sns.pointplot(
            data=df,
            x="Label",
            y="Score",
            hue="Score Type",
            dodge=True,
            join=True,
            markers=["o", "s"],
            capsize=0.1,
            palette=["skyblue", "salmon"],
        )

    title_suffix = "Descriptions" if desc_name == "desc" else "Names"
    plt.title(f"{model} – {title_suffix}", fontsize=24)
    plt.ylabel("Mean ROUGE1 Scores", fontsize=20)
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)
    plt.legend(title="Score Type", fontsize=20, title_fontsize=20)
    plt.show()


def plot_rouge_scores_combined(
    data: pd.DataFrame,
    desc_name: str = "desc",
    model: str = "gpt2",
    save_path: Optional[str] = None,
) -> None:
    """Point plot of ROUGE-1 recall across temperatures and shot counts.

    Parameters
    ----------
    data:
        Long-format DataFrame with columns ``Label``, ``Score``,
        ``Temperature``.
    desc_name:
        ``"desc"`` or ``"name"``.
    model:
        Model name used in the plot title.
    save_path:
        When provided the figure is saved to this path (PNG, 300 dpi)
        instead of being displayed interactively.

    Raises
    ------
    ValueError
        If *desc_name* is invalid.
    """
    if desc_name not in {"desc", "name"}:
        raise ValueError("desc_name must be 'desc' or 'name'.")

    plt.figure(figsize=(12, 7))

    data["Label"] = pd.Categorical(
        data["Label"],
        categories=sorted(data["Label"].unique(), key=lambda x: int(x)),
    )
    data = data.sort_values("Label")

    sns.pointplot(
        data=data,
        x="Label",
        y="Score",
        hue="Temperature",
        dodge=True,
        capsize=0.1,
        palette="tab10",
    )

    title_suffix = "Descriptions" if desc_name == "desc" else "Names"
    plt.title(f"{model} – {title_suffix} Mean ROUGE1 Recall")
    plt.ylabel("Mean ROUGE1 Recall Scores")
    plt.xlabel("Number of Shots")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.legend(title="Temperature", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="png", dpi=300)
        print(f"Figure saved to '{save_path}'.")
    else:
        plt.show()

    plt.close()


def plot_shot_results(
    all_temp_files: List[List[str]],
    labels: List[str],
    temperatures: list,
    model: str = "GPT-4 Turbo",
    image_dir: str = "Images",
) -> None:
    """Visualise ROUGE-1 recall across shot counts for multiple temperatures.

    Unlike :func:`plot_rouge_scores_combined`, this function accepts
    pre-generated result file lists rather than running new experiments.
    It is intended for use in analysis notebooks.

    Parameters
    ----------
    all_temp_files:
        One inner list of JSON filenames per temperature, in the same order
        as *temperatures*.  Each inner list contains one filename per shot
        count (matching *labels*).
    labels:
        Shot-count labels, e.g. ``['0', '1', '2', '3', '4', '5']``.
    temperatures:
        Temperature values corresponding to each entry in *all_temp_files*.
    model:
        Model display name used in the plot title and output filenames.
    image_dir:
        Directory where the plots will be saved.
    """
    from .evaluation import get_rouge_recall_f1

    desc_data: dict = {"Label": [], "Score": [], "Temperature": []}
    name_data: dict = {"Label": [], "Score": [], "Temperature": []}

    for filenames, temperature in zip(all_temp_files, temperatures):
        descs_recall, _ = get_rouge_recall_f1(filenames, name_desc="desc")
        names_recall, _ = get_rouge_recall_f1(filenames, name_desc="name")

        for label, recall in zip(labels, descs_recall):
            if recall is not None:
                desc_data["Label"].extend([label] * len(recall))
                desc_data["Score"].extend(recall)
                desc_data["Temperature"].extend([temperature] * len(recall))

        for label, recall in zip(labels, names_recall):
            if recall is not None:
                name_data["Label"].extend([label] * len(recall))
                name_data["Score"].extend(recall)
                name_data["Temperature"].extend([temperature] * len(recall))

    plot_rouge_scores_combined(
        pd.DataFrame(desc_data),
        desc_name="desc",
        model=model,
        save_path=os.path.join(image_dir, f"{model}_descriptions_rouge1_recall.png"),
    )
    plot_rouge_scores_combined(
        pd.DataFrame(name_data),
        desc_name="name",
        model=model,
        save_path=os.path.join(image_dir, f"{model}_names_rouge1_recall.png"),
    )


# Backward-compatible aliases
plot_rouge_means_with_seaborn = plot_rouge_scores
plot_rouge_means_with_seaborn_combined = plot_rouge_scores_combined
