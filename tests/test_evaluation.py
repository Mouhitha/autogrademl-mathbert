import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_embeddings as evaluation
from src.data import FEATURE_COLUMNS, QUESTION_IDS, embedding_matrix


@pytest.fixture
def datasets():
    def frame(embeddings, questions, marks):
        data = pd.DataFrame(embeddings, columns=FEATURE_COLUMNS)
        data["question_id"] = questions
        data["marks"] = marks
        return data

    basis = np.eye(6, 384)
    key = frame(basis, list(QUESTION_IDS), [2] * 6)
    main = frame(2 * basis, list(QUESTION_IDS), [0, 1, 2, 2, 2, 2])
    # The first row has a main vector with different labels. The second matches only a key.
    additional = frame(
        np.vstack([2 * basis[0], basis[5], 100 * basis[:3]]),
        [4, 6, 1, 2, 3],
        [2, 2, 0, 1, 2],
    )
    return key, main, additional


def test_fit_on_main_only_and_score_without_overlap(datasets, monkeypatch):
    key, main, additional = datasets
    pipelines = []
    make_pipeline = evaluation.make_pipeline

    def capture_pipeline(*steps):
        model = make_pipeline(*steps)
        pipelines.append(model)
        return model

    monkeypatch.setattr(evaluation, "make_pipeline", capture_pipeline)
    report = evaluation.evaluate(key, main, additional)

    assert report["data_audit"]["additional_matching_main_rows"] == 1
    assert report["data_audit"]["additional_matching_key_rows"] == 1
    questions = report["tasks"]["question_identification"]
    assert questions["exact_duplicates_dropped"] == 2
    assert questions["evaluation_rows"] == 3
    assert questions["methods"][0]["accuracy"] == 1.0
    assert questions["methods"][0]["support"] == [1, 1, 1, 0, 0, 0]
    assert questions["methods"][1]["accuracy"] == pytest.approx(1 / 3)

    marks = report["tasks"]["marks_prediction"]
    assert marks["exact_duplicates_dropped"] == 1
    assert marks["evaluation_rows"] == 4
    assert marks["methods"][0]["support"] == [1, 1, 2]
    assert marks["methods"][1]["accuracy"] == 0.5
    (model,) = pipelines
    scaler = model.named_steps["standardscaler"]
    assert scaler.n_samples_seen_ == len(main)
    assert scaler.n_features_in_ == 384
    np.testing.assert_allclose(scaler.mean_, embedding_matrix(main).mean(axis=0))


def test_embedding_inputs_ignore_question_and_marks_labels(datasets):
    _, main, additional = datasets
    for data in (main, additional):
        relabeled = data.assign(question_id=6, marks=0)
        assert embedding_matrix(data).shape == (len(data), 384)
        np.testing.assert_array_equal(embedding_matrix(data), embedding_matrix(relabeled))


def test_grade_scores_skipped_when_overlap_removes_a_class(datasets):
    key, main, additional = datasets
    # Label 1 now appears only on the row that duplicates a main vector.
    additional["marks"] = [1, 2, 0, 2, 2]

    report = evaluation.evaluate(key, main, additional)

    assert report["tasks"]["question_identification"]["methods"]
    marks = report["tasks"]["marks_prediction"]
    assert marks["methods"] == []
    assert "Missing in reduced additional: [1]" in marks["warning"]
