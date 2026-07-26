# EDA Report

Target column: **Survived**
Rows: 891 | Columns: 17
Inferred target type: **classification**

## Features most associated with the target

| Feature | Association strength |
|---|---|
| Title | 0.5732 |
| Sex | 0.5409 |
| Deck | 0.3336 |
| Embarked | 0.1726 |
| Fare | 0.1251 |
| Pclass | 0.0368 |
| Age_imputed | 0.0142 |
| Age | 0.0134 |
| FamilySize | 0.0133 |
| Parch | 0.0002 |
| SibSp | 0.0000 |
| Age_missing_flag | 0.0000 |

## Columns excluded from deep analysis

| Column | Reason |
|---|---|
| PassengerId | identifier |
| Name | identifier |
| Ticket | text |
| Cabin | text |

## Baseline RandomForest model

CV accuracy: 0.8103 (+/- 0.0425), folds=5

Top features by importance:

- Fare: 0.1544
- Title_freq: 0.0899
- Age_imputed: 0.1002
- Age: 0.0975
- Sex_female: 0.0802
- Sex_male: 0.0576
- Pclass: 0.0484
- FamilySize: 0.0369
- Ticket_freq: 0.0342
- Cabin_freq: 0.0204
