"""Evaluate the checked-in embeddings with exact cross-workbook overlap removed."""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if not __package__:
    # Support running this file directly from a checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    MARK_LABELS,
    QUESTION_IDS,
    embedding_matrix,
    exact_overlap_mask,
    load_embeddings,
)

WORKBOOKS = {
    "latex": {
        "key": "training_mathbert1keylatex.xlsx",
        "main": "training_mathbert3latexsolutions.xlsx",
        "additional": "training_mathbert5traininglatexsolutions.xlsx",
    },
    "algebraic": {
        "key": "training_mathbert2keyalgebraicsolutions.xlsx",
        "main": "training_mathbert4algebraicsolutions.xlsx",
        "additional": "training_mathbert6trainingalgebraicsolutions.xlsx",
    },
}


def score(method, actual, predicted, labels):
    matrix = confusion_matrix(actual, predicted, labels=labels)
    return {
        "method": method,
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "labels": list(labels),
        "support": matrix.sum(axis=1).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(key: pd.DataFrame, main: pd.DataFrame, additional: pd.DataFrame) -> dict:
    key = key.sort_values("question_id")
    if tuple(key["question_id"]) != QUESTION_IDS:
        raise ValueError("Expected one reference key vector for each question ID 1–6")

    key_x = embedding_matrix(key)
    main_x = embedding_matrix(main)
    additional_x = embedding_matrix(additional)
    matches_main = exact_overlap_mask(main_x, additional_x)
    matches_key = exact_overlap_mask(key_x, additional_x)
    question_keep = ~(matches_main | matches_key)
    marks_keep = ~matches_main

    questions = {
        "evaluation_rows": int(question_keep.sum()),
        "exact_duplicates_dropped": int((~question_keep).sum()),
        "methods": [],
    }
    if question_keep.any():
        question_x = additional_x[question_keep]
        if (key_x == 0).all(axis=1).any() or (question_x == 0).all(axis=1).any():
            raise ValueError("Cosine similarity requires nonzero key and evaluation vectors")
        actual = additional.loc[question_keep, "question_id"]
        best_key = cosine_similarity(question_x, key_x).argmax(axis=1)
        baseline = DummyClassifier(strategy="most_frequent").fit(main_x, main["question_id"])
        questions["methods"] = [
            score("nearest-reference cosine", actual, key["question_id"].to_numpy()[best_key], QUESTION_IDS),
            score("majority class", actual, baseline.predict(question_x), QUESTION_IDS),
        ]
    else:
        questions["warning"] = "No additional rows remain for question identification after overlap removal."

    marks = {
        "evaluation_rows": int(marks_keep.sum()),
        "exact_duplicates_dropped": int(matches_main.sum()),
        "methods": [],
    }
    train_marks = main["marks"]
    eval_marks = additional.loc[marks_keep, "marks"]
    missing_train = sorted(set(MARK_LABELS) - set(train_marks))
    missing_eval = sorted(set(MARK_LABELS) - set(eval_marks))
    if missing_train or missing_eval:
        marks["warning"] = (
            "Marks scores skipped because every grade label must appear in main and reduced additional rows. "
            f"Missing in main: {missing_train}. Missing in reduced additional: {missing_eval}."
        )
    else:
        marks_x = additional_x[marks_keep]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(main_x, train_marks)
        except ConvergenceWarning:
            marks["warning"] = "Marks scores skipped because logistic regression did not converge."
        else:
            baseline = DummyClassifier(strategy="most_frequent").fit(main_x, train_marks)
            marks["methods"] = [
                score("multinomial logistic regression", eval_marks, model.predict(marks_x), MARK_LABELS),
                score("majority class", eval_marks, baseline.predict(marks_x), MARK_LABELS),
            ]

    return {
        "data_audit": {
            "key_rows": len(key),
            "main_rows": len(main),
            "additional_rows": len(additional),
            "key_matching_main_rows": int(exact_overlap_mask(main_x, key_x).sum()),
            "additional_matching_main_rows": int(matches_main.sum()),
            "additional_matching_key_rows": int(matches_key.sum()),
        },
        "tasks": {"question_identification": questions, "marks_prediction": marks},
    }


def write_results(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    representation = report["representation"]
    json_path = output_dir / f"{representation}.json"
    csv_path = output_dir / f"{representation}.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as output:
        fields = [
            "task", "key_rows", "main_rows", "additional_rows", "evaluation_rows",
            "exact_duplicates_dropped", "method", "accuracy", "macro_f1", "labels",
            "support", "confusion_matrix",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for task, result in report["tasks"].items():
            for method in result["methods"]:
                writer.writerow({
                    "task": task,
                    "key_rows": report["data_audit"]["key_rows"],
                    "main_rows": report["data_audit"]["main_rows"],
                    "additional_rows": report["data_audit"]["additional_rows"],
                    "evaluation_rows": result["evaluation_rows"],
                    "exact_duplicates_dropped": result["exact_duplicates_dropped"],
                    **method,
                })
    print(f"Wrote {json_path} and {csv_path}")


def print_report(report: dict) -> None:
    audit = report["data_audit"]
    print(f"{report['representation']}: {audit['key_rows']} key, {audit['main_rows']} main, "
          f"{audit['additional_rows']} additional rows")
    print(f"Exact overlaps: {audit['key_matching_main_rows']} key rows match main, "
          f"{audit['additional_matching_main_rows']} additional rows match main, "
          f"{audit['additional_matching_key_rows']} additional rows match key")
    for task, result in report["tasks"].items():
        print(f"\n{task.replace('_', ' ').capitalize()}: {result['evaluation_rows']} evaluation rows, "
              f"{result['exact_duplicates_dropped']} exact duplicates dropped")
        if "warning" in result:
            print(f"Warning: {result['warning']}")
        for method in result["methods"]:
            print(f"  {method['method']}: accuracy {method['accuracy']:.4f}, macro F1 {method['macro_f1']:.4f}")
            support = dict(zip(method["labels"], method["support"]))
            print(f"    Support (label: count): {support}")
            print(f"    Confusion matrix (true rows, predicted columns; order {method['labels']}):")
            for row in method["confusion_matrix"]:
                print(f"    {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=WORKBOOKS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    workbooks = WORKBOOKS[args.representation]
    try:
        datasets = {role: load_embeddings(args.data_dir / name) for role, name in workbooks.items()}
        report = evaluate(**datasets)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    report = {"representation": args.representation, "workbooks": workbooks, **report}
    print_report(report)
    write_results(report, args.output_dir)


if __name__ == "__main__":
    main()
