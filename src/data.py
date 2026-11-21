"""Load the workbook schema and compare exact embedding vectors."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FEATURE_COLUMNS = tuple(f"embed_{index}" for index in range(384))
QUESTION_IDS = (1, 2, 3, 4, 5, 6)
MARK_LABELS = (0, 1, 2)


def load_embeddings(path: str | Path) -> pd.DataFrame:
    """Read Sheet1 as 384 float64 features followed by two int64 labels."""
    path = Path(path)
    # The Excel reader trims trailing empty rows and retains interior blank rows.
    sheet = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl", keep_default_na=False)
    question_columns = [name for name in ("Que no", "Question") if name in sheet]
    if len(question_columns) != 1:
        raise ValueError(f"{path.name}: expected exactly one of 'Que no' and 'Question'")

    question_column = question_columns[0]
    required = [*FEATURE_COLUMNS, question_column, "Marks"]
    missing = [column for column in required if column not in sheet]
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {', '.join(missing)}")
    if sheet.empty:
        raise ValueError(f"{path.name}: Sheet1 contains no embedding rows")

    features = sheet.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    features = features.astype(np.float64)
    invalid = ~np.isfinite(features.to_numpy()).all(axis=1)
    if invalid.any():
        rows = (np.flatnonzero(invalid) + 2).tolist()
        raise ValueError(f"{path.name}: missing or non-finite numeric embeddings in Excel rows {rows}")

    for source, target, allowed in (
        (question_column, "question_id", QUESTION_IDS),
        ("Marks", "marks", MARK_LABELS),
    ):
        values = pd.to_numeric(sheet[source], errors="coerce")
        invalid = ~values.isin(allowed)
        if invalid.any():
            rows = (np.flatnonzero(invalid) + 2).tolist()
            raise ValueError(f"{path.name}: {source} must be one of {allowed} in Excel rows {rows}")
        features[target] = values.astype(np.int64)

    return features


def embedding_matrix(data: pd.DataFrame) -> NDArray[np.float64]:
    """Select the ordered feature matrix independently of the label columns."""
    return data.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)


def exact_overlap_mask(
    reference: NDArray[np.float64], candidates: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Flag each candidate whose entire vector occurs in reference."""
    def fingerprints(matrix: NDArray[np.float64]) -> list[bytes]:
        normalized = matrix.astype("<f8", copy=True)
        normalized[normalized == 0] = 0.0  # Treat -0.0 and +0.0 as the same number.
        return [row.tobytes() for row in normalized]

    known = set(fingerprints(reference))
    return np.array([row in known for row in fingerprints(candidates)], dtype=bool)
