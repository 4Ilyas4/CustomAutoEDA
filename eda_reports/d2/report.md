# EDA Report

Target column: **Species**
Rows: 150 | Columns: 6
Inferred target type: **classification**

## Features most associated with the target

| Feature | Association strength |
|---|---|
| PetalLengthCm | 0.9887 |
| PetalWidthCm | 0.9825 |
| SepalLengthCm | 0.4813 |
| SepalWidthCm | 0.2397 |

## Columns excluded from deep analysis

| Column | Reason |
|---|---|
| Id | identifier |

## Baseline RandomForest model

CV accuracy: 0.9667 (+/- 0.0211), folds=5

Top features by importance:

- PetalLengthCm: 0.4604
- PetalWidthCm: 0.4241
- SepalLengthCm: 0.0909
- SepalWidthCm: 0.0245
