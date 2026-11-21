# Data dictionary

All six files use `Sheet1` with 386 columns. The 384 numeric features are `embed_0` through `embed_383` in numerical order, followed by a question ID and `Marks`. Question IDs are integers from 1 through 6. The loader calls the labels `question_id` and `marks`.

`Marks` records three ordered grade labels, 0, 1, and 2. The key examples all carry label 2. The workbook contents leave the rules for assigning marks and their consistency across questions unverified. Each distribution below lists counts for marks 0, 1, and 2 in that order.

## `training_mathbert1keylatex.xlsx`

- LaTeX reference key, 6 rows, one per question.
- Features `embed_0` through `embed_383`. `Que no` maps to `question_id` and `Marks` maps to `marks`.
- Mark counts are 0, 0, 6. Each vector also occurs in the LaTeX main workbook.

## `training_mathbert2keyalgebraicsolutions.xlsx`

- Algebraic reference key, 6 rows, one per question.
- Features `embed_0` through `embed_383`. `Que no` maps to `question_id` and `Marks` maps to `marks`.
- Mark counts are 0, 0, 6. Each vector also occurs in the algebraic main workbook.

## `training_mathbert3latexsolutions.xlsx`

- LaTeX main solutions, 126 rows, 21 per question.
- Features `embed_0` through `embed_383`. `Que no` maps to `question_id` and `Marks` maps to `marks`.
- Mark counts are 22, 21, 83. This workbook supplies marks training data and baseline label counts.

## `training_mathbert4algebraicsolutions.xlsx`

- Algebraic main solutions, 126 rows, 21 per question.
- Features `embed_0` through `embed_383`. `Que no` maps to `question_id` and `Marks` maps to `marks`.
- Mark counts are 22, 21, 83. This workbook supplies marks training data and baseline label counts.

## `training_mathbert5traininglatexsolutions.xlsx`

- LaTeX additional solutions, 26 nonempty rows. The worksheet declares 41 rows including the header and 14 trailing blank rows.
- Features `embed_0` through `embed_383`. `Question` maps to `question_id` and `Marks` maps to `marks`.
- Mark counts are 8, 8, 10 before overlap removal. Two vectors match main rows exactly. The reduced additional rows supply the evaluation data.

## `training_mathbert6trainingalgebraicsolutions.xlsx`

- Algebraic additional solutions, 26 nonempty rows. The worksheet declares 41 rows including the header and 14 trailing blank rows.
- Features `embed_0` through `embed_383`. `Question` maps to `question_id` and `Marks` maps to `marks`.
- Mark counts are 8, 8, 10 before overlap removal. Two vectors match main rows exactly. The reduced additional rows supply the evaluation data.

The scripts assign the two `training`-named files the additional evaluation role for this protocol. Their names leave their original use and collection process unconfirmed. The workbooks contain vectors and labels without the original student answers or source records.
