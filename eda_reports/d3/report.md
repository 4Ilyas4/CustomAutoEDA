# EDA Report

Target column: **AQI Category**
Rows: 16695 | Columns: 14
Inferred target type: **classification**

## Features most associated with the target

| Feature | Association strength |
|---|---|
| AQI Value | 1.0877 |
| PM2.5 AQI Value | 0.9906 |
| PM2.5 AQI Category | 0.9324 |
| Ozone AQI Category | 0.3447 |
| CO AQI Value | 0.2439 |
| NO2 AQI Category | 0.2313 |
| lng | 0.1904 |
| lat | 0.1758 |
| CO AQI Category | 0.1670 |
| Ozone AQI Value | 0.1360 |
| NO2 AQI Value | 0.0862 |

## Columns excluded from deep analysis

| Column | Reason |
|---|---|
| Country | text |
| City | text |

## Baseline RandomForest model

CV accuracy: 0.9998 (+/- 0.0002), folds=5

Top features by importance:

- AQI Value: 0.3036
- PM2.5 AQI Value: 0.1917
- PM2.5 AQI Category_Good: 0.1871
- PM2.5 AQI Category_Moderate: 0.1433
- PM2.5 AQI Category_Unhealthy: 0.0424
- PM2.5 AQI Category_Unhealthy for Sensitive Groups: 0.0383
- CO AQI Value: 0.0269
- Ozone AQI Value: 0.0170
- Ozone AQI Category_Good: 0.0097
- Ozone AQI Category_Moderate: 0.0076
