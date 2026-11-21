from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from scripts.evaluate_embeddings import WORKBOOKS
from src.data import (
    FEATURE_COLUMNS,
    MARK_LABELS,
    QUESTION_IDS,
    embedding_matrix,
    exact_overlap_mask,
    load_embeddings,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("representation", WORKBOOKS)
def test_original_workbooks(representation):
    datasets = {}
    expected = {
        "key": ("Que no", 7, 6, [0, 0, 6]),
        "main": ("Que no", 127, 126, [22, 21, 83]),
        "additional": ("Question", 41, 26, [8, 8, 10]),
    }
    for role, (question_column, declared_rows, records, marks) in expected.items():
        path = ROOT / WORKBOOKS[representation][role]
        with path.open("rb") as source:
            workbook = load_workbook(source, read_only=True)
            sheet = workbook["Sheet1"]
            assert sheet.max_row == declared_rows
            assert sheet.max_column == 386
            assert next(sheet.values) == (*FEATURE_COLUMNS, question_column, "Marks")
            workbook.close()

        data = load_embeddings(path)
        datasets[role] = data
        assert list(data.columns) == [*FEATURE_COLUMNS, "question_id", "marks"]
        assert len(data) == records
        assert embedding_matrix(data).shape == (records, 384)
        assert data["question_id"].isin(QUESTION_IDS).all()
        assert data["marks"].value_counts().reindex(MARK_LABELS, fill_value=0).tolist() == marks
        assert data["question_id"].dtype == data["marks"].dtype == np.int64

    assert datasets["main"]["question_id"].value_counts().sort_index().tolist() == [21] * 6
    main = embedding_matrix(datasets["main"])
    assert exact_overlap_mask(main, embedding_matrix(datasets["key"])).sum() == 6
    assert exact_overlap_mask(main, embedding_matrix(datasets["additional"])).sum() == 2


def test_loader_orders_features_and_normalizes_labels(tmp_path):
    row = {f"embed_{index}": index for index in reversed(range(384))}
    path = tmp_path / "reordered.xlsx"
    pd.DataFrame([{"Marks": 2, "Question": 6, **row}]).to_excel(path, engine="openpyxl", index=False)

    data = load_embeddings(path)
    np.testing.assert_array_equal(embedding_matrix(data), np.arange(384)[None, :])
    assert data.loc[0, ["question_id", "marks"]].tolist() == [6, 2]


@pytest.mark.parametrize(("column", "value", "error"), [
    ("Question", 7, "Question must be one of"),
    ("Question", 1.5, "Question must be one of"),
    ("Marks", -1, "Marks must be one of"),
    ("Marks", 3, "Marks must be one of"),
    ("embed_383", None, "embeddings in Excel rows"),
    ("embed_383", "NaN", "embeddings in Excel rows"),
    ("embed_383", "inf", "embeddings in Excel rows"),
])
def test_loader_rejects_invalid_rows(tmp_path, column, value, error):
    row = dict.fromkeys(FEATURE_COLUMNS, 1.0)
    row.update({"Question": 1, "Marks": 2, column: value})
    path = tmp_path / "invalid.xlsx"
    pd.DataFrame([row]).to_excel(path, engine="openpyxl", index=False)

    with pytest.raises(ValueError, match=rf"{error}.*\[2\]"):
        load_embeddings(path)


def test_loader_rejects_missing_feature_column(tmp_path):
    row = dict.fromkeys(FEATURE_COLUMNS[:-1], 1.0)
    path = tmp_path / "missing.xlsx"
    pd.DataFrame([{"Que no": 1, "Marks": 2, **row}]).to_excel(path, engine="openpyxl", index=False)

    with pytest.raises(ValueError, match="missing required columns: embed_383"):
        load_embeddings(path)


def test_loader_rejects_interior_blank_row(tmp_path):
    row = {**dict.fromkeys(FEATURE_COLUMNS, 1.0), "Question": 1, "Marks": 2}
    path = tmp_path / "interior.xlsx"
    pd.DataFrame([row, {}, row]).to_excel(path, engine="openpyxl", index=False)

    with pytest.raises(ValueError, match=r"embeddings in Excel rows \[3\]"):
        load_embeddings(path)


def test_exact_fingerprints_use_every_feature():
    reference = np.zeros((1, 384))
    candidates = np.zeros((3, 384))
    candidates[1, 10] = -0.0
    candidates[2, -1] = 1e-15

    assert exact_overlap_mask(reference, candidates).tolist() == [True, True, False]
