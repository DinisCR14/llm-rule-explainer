"""Experiment orchestration: shot-count search and full combination runs.

Run this script directly to reproduce the full AECD experimental pipeline:

    python scripts/run_experiment.py

Or import individual helpers for use in notebooks:

    from scripts.run_experiment import get_best_number_of_shots
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# Allow `from aecd import ...` when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aecd import (
    get_rouge_recall_f1,
    plot_rouge_scores_combined,
    run_experiment,
    split_rules_dataset,
)


def get_best_number_of_shots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    temperatures: list,
    model: str = "gpt2",
    shot_i: int = 0,
    shot_f: int = 5,
    output_dir: str = "Results",
) -> None:
    """Run zero- and few-shot experiments across shot counts and temperatures.

    Generates one ROUGE-1 recall plot per model showing performance vs.
    number of in-context examples, coloured by temperature.

    Parameters
    ----------
    train_df, test_df:
        Rule DataFrames produced by :func:`~aecd.split_rules_dataset`.
    temperatures:
        Sampling temperatures to evaluate.
    model:
        Hugging Face model identifier.
    shot_i, shot_f:
        Inclusive range of shot counts to sweep.
    output_dir:
        Root directory for JSON output files.
    """
    desc_data: dict = {"Label": [], "Score": [], "Temperature": []}
    name_data: dict = {"Label": [], "Score": [], "Temperature": []}

    for temperature in temperatures:
        filenames = []
        labels = []

        for num_shots in range(shot_i, shot_f + 1):
            labels.append(str(num_shots))
            output_file = f"{model}_temp{temperature}_{num_shots}_shots.json"
            strategy = "zs" if num_shots == 0 else "fs"

            run_experiment(
                train_df,
                test_df,
                strategy=strategy,
                step_by_step=False,
                n_examples=num_shots,
                model=model,
                temperature=temperature,
                output_dir=output_dir,
                output_file=output_file,
            )
            filenames.append(os.path.join(output_dir, output_file))

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

    image_dir = "Images"
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


def run_all_combinations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_shots: list,
    models: list | None = None,
    temperatures: list | None = None,
) -> None:
    """Run every combination of model, temperature, and prompting strategy.

    Parameters
    ----------
    train_df, test_df:
        Rule DataFrames.
    n_shots:
        Per-model shot count list (one entry per model in *models*).
    models:
        Hugging Face model identifiers.  Defaults to ``["gpt2"]``.
    temperatures:
        Sampling temperatures.  Defaults to ``[1.0]``.
    """
    if models is None:
        models = ["gpt2"]
    if temperatures is None:
        temperatures = [1.0]

    # (strategy, step_by_step, file_prefix, use_bert)
    experiment_configs = [
        ("zs",     False, "zs",           False),
        ("zs",     True,  "zs_step",      False),
        ("fs",     False, "fs",           False),
        ("fs",     True,  "fs_step",      False),
        ("os_sim", False, "os_sim",       False),
        ("os_sim", True,  "os_step_sim",  False),
        ("os_sim", False, "os_sim_bert",  True),
        ("os_sim", True,  "os_step_bert", True),
    ]

    for i, model in enumerate(models):
        for temp in temperatures:
            for strategy, step_by_step, file_prefix, use_bert in experiment_configs:
                output_file = (
                    f"{file_prefix}_{model}_temp{temp}_usebert{use_bert}.json"
                )
                print(
                    f"Running: model={model}, temp={temp}, strategy={strategy}, "
                    f"step_by_step={step_by_step}, use_bert={use_bert}"
                )
                run_experiment(
                    train_df=train_df,
                    test_df=test_df,
                    strategy=strategy,
                    step_by_step=step_by_step,
                    n_examples=n_shots[i],
                    model=model,
                    temperature=temp,
                    use_bert=use_bert,
                    output_file=output_file,
                )


def main() -> None:
    """Entry point: run the full AECD experimental pipeline."""
    train_df, test_df = split_rules_dataset()

    models = [
        "gpt2",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-70b-hf",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-40b",
    ]
    temperatures = [0.25, 0.5, 0.75, 1.0]

    # Step 1 – find the best number of few-shot examples per model
    for model in models:
        get_best_number_of_shots(
            train_df, test_df, temperatures, model=model, output_dir="Shots_Results"
        )

    # Step 2 – run all strategy combinations with the optimal shot counts
    n_shots = [2, 2, 2, 2, 2]
    temperatures = [1, 1, 1, 1, 1]
    run_all_combinations(train_df, test_df, n_shots, models, temperatures)


if __name__ == "__main__":
    main()
