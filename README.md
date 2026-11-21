# AutoGradeML: Evaluating Algebra Answers with MathBERT Embeddings

How do algebraic syntax and LaTeX compare when identifying which of six questions a response addresses? Separately, how well does each representation predict the recorded partial-credit mark?

The repository contains already-computed, 384-dimensional MathBERT embeddings for six questions. Each row has a question ID and a recorded `Marks` label of 0, 1, or 2. Each representation has three workbooks with different roles in this evaluation:

- **Reference key** (6 rows). One example per question, each marked 2. All six vectors also appear in the corresponding main workbook.
- **Main solutions** (126 rows). The source of training data for marks prediction and of the most frequent labels for both baselines.
- **Additional solutions** (26 nonempty rows). These supply the evaluation data. Two rows per representation exactly match main vectors.

## Results

Each task scored 24 additional examples per representation after excluding two known exact overlaps. How those examples were collected and whether they came from distinct students or answers are unknown. Exact vector matching catches identical vectors and can miss differently encoded copies of the same answer. [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)

### Question identification

Nearest-reference cosine matched the recorded question ID on **21 of 24 LaTeX examples (87.5%)** and **22 of 24 algebraic examples (91.7%)**. Each prediction picks one of six reference question IDs. [LaTeX JSON](results/latex.json) · [Algebraic JSON](results/algebraic.json)

### Marks prediction

Logistic regression matched the recorded mark on **11 of 24 examples (45.8%)** for each representation. The majority-class baseline matched **8 of 24 (33.3%)** for each. The model uses the embeddings alone. Agreement on fewer than half of these 24 examples is insufficient evidence for reliable automatic grading or readiness for real student assessment. [LaTeX JSON](results/latex.json) · [Algebraic JSON](results/algebraic.json)

## Run the evaluation

From the repository root, with Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/evaluate_embeddings.py --representation latex
python scripts/evaluate_embeddings.py --representation algebraic
```

The script prints counts, scores, and confusion matrices and writes JSON reports and CSV summaries to `results/`. Use `--data-dir` or `--output-dir` to change those locations. To run the tests in the same environment, install `requirements-dev.txt` and run `python -m pytest -q`. [LaTeX results](results/latex.json) · [Algebraic results](results/algebraic.json)

## Limits of the archive

The script reproduces the evaluation on the supplied embeddings for six questions. The original answers, partial-credit rubric, and exact MathBERT settings are unavailable in this archive. Recreating the embeddings or verifying the marks would require those inputs and settings. The evaluation includes no fairness audit. [Data dictionary](docs/DATA_DICTIONARY.md)

The original `projectfinal.ipynb` notebook archives clustering, t-SNE, similarity experiments, and manually assembled summaries. The historical headline percentages lack an independently verified task, split, and denominator. The results above come from the evaluation script. [Original notebook](projectfinal.ipynb) · [Notebook notes](docs/ORIGINAL_NOTEBOOK_NOTES.md)
