# Evaluation protocol

Run each representation separately. The [data dictionary](DATA_DICTIONARY.md) lists the exact key, main, and additional workbooks and the labels read from them.

## Inputs and overlap checks

The feature matrix contains `embed_0` through `embed_383` in numerical order. The loader reads `Sheet1`, ignores trailing empty rows, and rejects missing or non-finite features even when the row has a valid label. The ground truth is the recorded `question_id` or `marks` for the relevant task.

To check overlap, convert each vector to 384 float64 values, normalize signed zero, and use its complete byte sequence as the fingerprint. A match requires equality across all 384 features regardless of the labels. There is no rounding or similarity cutoff. The report counts how many key rows match main rows and how many additional rows match the main or key workbook. The audit counts additional rows, including each occurrence of a repeated vector.

The question task excludes additional rows matching main or key vectors. The marks task excludes additional rows matching main vectors. In these workbooks both rules exclude two of the 26 additional rows per representation, leaving 24 examples per task and representation. The collection process and the source of individual examples are unknown. Exact matching can miss differently encoded copies of the same answer and cannot establish whether rows came from distinct students or answers. The scores describe the remaining rows for these six questions.

## Question identification

Compare each eligible additional vector to the six reference vectors using cosine similarity, the dot product divided by the two vector lengths. The highest similarity picks the question ID. Exact ties select the smallest question ID. The baseline always predicts the most frequent question ID in the main workbook. Ties also select the smallest ID. A prediction is correct for this task when it matches the recorded question ID.

## Partial-credit marks prediction

Fit a feature standardizer and an L2-regularized multinomial logistic regression on main embeddings with `marks` as the target. The model uses scikit-learn's `lbfgs` solver, `C=1`, and a maximum of 2,000 iterations. Apply the fitted standardizer to the reduced additional embeddings before predicting their recorded mark labels. The known question ID is excluded from the inputs. The baseline always predicts the most frequent mark in the main workbook. Ties select the smallest mark.

Both the main and reduced additional workbooks must contain all three grade labels to report marks scores. Missing classes or failure to converge produce a warning and suppress marks scores, including the baseline. The JSON records the warning.

The 45.8% marks prediction accuracy in each report is agreement with 11 of 24 recorded labels. This small set has unknown provenance, and the original rubric and label checks are unavailable. The score is insufficient evidence for reliable automatic grading or for use in real student assessment. [LaTeX JSON](../results/latex.json) · [Algebraic JSON](../results/algebraic.json)

## Model selection and metrics

The method settings are fixed in the script for both representations. Each baseline uses the most frequent label for its task in the main workbook. The marks model and standardizer fit on main embeddings alone. Additional labels are used for coverage checks and scoring. The evaluation uses no threshold tuning, resampling, PCA, or t-SNE. Any future tuning should use splits within the main workbook, keep related rows together where their provenance is known, and fit preprocessing within each training fold. A new independent dataset would be needed to assess a model picked after seeing the additional scores.

- **Accuracy** is the fraction of evaluated rows given the exact recorded label. For question identification it is top-1 accuracy. For marks it counts each wrong label as one error regardless of the size of the mark difference.
- **Macro F1** is the unweighted mean of the label-level F1 scores. Each label's F1 is the harmonic mean of precision and recall. An undefined F1 contributes zero. The label list is fixed at 1–6 for questions and 0–2 for marks.
- **Confusion matrix** rows are recorded labels and columns are predicted labels in the given label order. **Support** is the number of evaluated examples per recorded label and equals the row sum.

Use the generated JSON for dataset names, counts, and metrics, or the CSV for one row per method and task. With 24 eligible additional examples per task and representation, a single prediction changes accuracy by about 4.2 percentage points.
