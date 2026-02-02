# Air Quality Index (AQI) Forecasting and Early Warning System for Indian Cities

## Databricks 14-Days AI Challenge - Capstone Project

**Author:** Keerthi Amulya
**Date:** January 2026
**Email:** keerthi.amulya.1999@gmail.com

[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Overview](#solution-overview)
4. [Dataset](#dataset)
5. [Architecture](#architecture)
6. [Data Pipeline](#data-pipeline)
7. [Machine Learning Models](#machine-learning-models)
8. [Results](#results)
9. [Project Structure](#project-structure)
10. [Technologies Used](#technologies-used)
11. [How to Run](#how-to-run)
12. [Future Improvements](#future-improvements)

---

## Executive Summary

This project implements an **AI-powered AQI forecasting and early warning system** using Databricks, PySpark, and MLflow. The system predicts next-day AQI values and identifies high-pollution risk days across 275 Indian cities using 9 years of historical data (2015-2023).

### Key Achievements

| Metric | Value |
|--------|-------|
| Total Records Processed | 299,976 |
| Clean Records (Silver) | 297,209 |
| ML-Ready Records (Gold) | 289,034 |
| Cities Covered | 275 |
| Time Period | 2015-2023 |
| ML Features Engineered | 52 |
| Best Classification Accuracy | 89.41% (XGBoost) |
| Best Classification F1 Score | 0.81 |

---

## Problem Statement

### The Challenge: Air Pollution Crisis in India

Air pollution is one of India's most pressing public health emergencies. According to the World Health Organization, **13 of the world's 20 most polluted cities are in India**, contributing to over 1.67 million premature deaths annually.

### Current Limitations

- **Reactive Monitoring:** Existing AQI systems only report historical data, not predictions
- **No Advance Warning:** Citizens cannot plan outdoor activities safely
- **Healthcare Unpreparedness:** Hospitals face sudden surges in respiratory cases
- **Policy Delays:** Government interventions happen after pollution events occur

---

## Solution Overview

An **intelligent AQI forecasting and early warning system** that:

1. **Predicts AQI values 1-7 days in advance** using historical patterns and machine learning
2. **Identifies pollution trends** across different cities and seasons
3. **Generates early warnings** when AQI is predicted to exceed safe thresholds (>150)
4. **Provides actionable insights** for multiple stakeholders

### Target Stakeholders

| Stakeholder | Use Case | Expected Benefit |
|-------------|----------|------------------|
| **Citizens** | Plan outdoor activities based on forecasts | Reduced health risks |
| **Healthcare** | Prepare for respiratory case surges | Optimal resource allocation |
| **Policymakers** | Implement preventive restrictions | Proactive pollution control |
| **Researchers** | Analyze long-term pollution patterns | Better understanding of drivers |

---

## Dataset

**Source:** Central Pollution Control Board (CPCB) Daily AQI Bulletins
**Repository:** https://github.com/urbanemissionsinfo/AQI_bulletins

| Attribute | Value |
|-----------|-------|
| Time Period | May 2015 - December 2023 |
| Coverage | 275 cities across India |
| Raw Records | 299,976 |
| AQI Range | 3 to 500 |
| Average AQI | 124.69 |

### Raw Features

| Feature | Type | Description |
|---------|------|-------------|
| `date` | String | Date of measurement (YYYY-MM-DD) |
| `city` | String | Name of city |
| `station_count` | Integer | Number of active monitoring stations (1-39) |
| `air_quality` | String | Quality category |
| `aqi` | Double | Average Air Quality Index value |
| `prominent_pollutant` | String | Primary pollutant (PM2.5, PM10, NO2, CO, SO2, O3) |

### AQI Categories (CPCB Guidelines)

| AQI Range | Category | Records | Percentage |
|-----------|----------|---------|------------|
| 0-50 | Good | 47,287 | 16% |
| 51-100 | Satisfactory | 104,064 | 35% |
| 101-200 | Moderate | 95,004 | 32% |
| 201-300 | Poor | 34,378 | 12% |
| 301-400 | Very Poor | 13,962 | 4.7% |
| 401-500 | Severe | 2,514 | 0.8% |

---

## Architecture

### Medallion Architecture (Bronze -> Silver -> Gold)

```
aqi_india (Unity Catalog)
|
+-- bronze (Raw Data Layer)
|   +-- aqi_bulletins (299,976 records)
|
+-- silver (Cleaned Data Layer)
|   +-- aqi_cleaned (297,209 records, 20 columns)
|       - Partitioned by year, month
|       - ZORDER optimized by city, date
|
+-- gold (Analytics Layer)
    +-- aqi_ml_features (289,034 records, 52 features)
    +-- city_summary (275 cities aggregated)
    +-- monthly_trends (Time-based aggregates)
    +-- aqi_alert_predictions (Classification results)
```

---

## Data Pipeline

### Bronze Layer (Raw Ingestion)
- Load raw CSV with schema validation
- Add ingestion metadata (timestamp, source file)
- Store in Delta format

### Silver Layer (Data Cleaning)

| Step | Operation | Records Affected |
|------|-----------|------------------|
| 1 | Remove corrupted rows (NULL city/aqi, malformed dates) | 4 rows removed |
| 2 | Parse and validate dates (2015-2023) | Date range: 2015-05-01 to 2023-12-31 |
| 3 | Filter invalid AQI values (0-999) | 0 rows removed |
| 4 | Handle station counts (fill NULLs with 1) | Range: 1 to 39, Avg: 1.80 |
| 5 | Standardize city names | 277 -> 275 cities (2 merged) |
| 6 | Remove duplicate date-city combinations | 2,763 duplicates removed (0.92%) |
| 7 | Add temporal features | year, month, day_of_week, quarter, week_of_year, season, is_weekend |
| 8 | Create AQI risk categories | Good, Satisfactory, Moderate, Poor, Very Poor, Severe |
| 9 | Handle missing values | pollutant NULLs -> 'Unknown' |
| 10 | Create final schema | 20 columns, partitioned by year/month |

**Final Silver Schema:** 297,209 records, 20 columns

### Gold Layer (Feature Engineering)

**52 ML-ready features** organized into groups:

| Feature Group | Count | Features |
|---------------|-------|----------|
| **Identifiers** | 7 | date, city, year, month, day_of_week, quarter, season |
| **Current Values** | 6 | aqi, station_count, aqi_category, aqi_risk_level, prominent_pollutant, is_weekend |
| **Risk Flags** | 2 | is_high_pollution, is_severe_pollution |
| **Lag Features** | 5 | aqi_lag_1, aqi_lag_3, aqi_lag_7, aqi_lag_14, aqi_lag_30 |
| **Rolling Statistics** | 7 | aqi_rolling_avg_7/14/30, aqi_rolling_std_7/14, aqi_rolling_max_7, aqi_rolling_min_7 |
| **Rate of Change** | 4 | aqi_change_1d, aqi_change_7d, aqi_pct_change_1d, aqi_pct_change_7d |
| **City Baselines** | 5 | city_avg_aqi, city_std_aqi, aqi_deviation_from_city_avg, aqi_z_score, aqi_percentile_in_city |
| **Cyclical Encoding** | 6 | month_sin/cos, day_of_week_sin/cos, day_of_month_sin/cos |
| **Interaction Features** | 3 | is_winter_high_pollution_city, weekend_pollution_delta, is_reliable_measurement |
| **Target Variables** | 5 | aqi_next_1d, aqi_next_3d, aqi_next_7d, target_high_aqi_tomorrow, target_severe_aqi_tomorrow |
| **Metadata** | 2 | feature_timestamp, feature_version |

**Final Gold Records:** 289,034 (requiring 30+ days history for lag features)

---

## Machine Learning Models

### Feature Set for Training

```python
features = [
    "aqi", "aqi_lag_1", "aqi_lag_7",
    "aqi_rolling_avg_7", "aqi_rolling_std_7",
    "is_weekend", "is_high_pollution",
    "city_freq", "season_idx"
]
```

### Regression Models (Predict Next-Day AQI Value)

| Model | Metrics Logged to MLflow |
|-------|-------------------------|
| Linear Regression | RMSE, MAE, R2, MAPE |
| Random Forest Regressor | RMSE, MAE, R2, MAPE |

*Parameters:* Random Forest (n_estimators=200, max_depth=10)

### Classification Models (High AQI Alert: AQI > 150)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~89% | Logged | Logged | - |
| **Random Forest Classifier** | **89.28%** | **0.84** | **0.77** | **0.81** |
| **XGBoost Classifier** | **89.41%** | **0.85** | **0.77** | **0.81** |

### Confusion Matrix (Logistic Regression)

```
                    Pred_No_High_AQI  Pred_High_AQI
Actual_No_High_AQI             38,834          2,417
Actual_High_AQI                 3,763         12,739
```

- **True Negatives:** 38,834 (correctly predicted normal AQI)
- **True Positives:** 12,739 (correctly predicted high AQI)
- **False Positives:** 2,417 (false alarms)
- **False Negatives:** 3,763 (missed high AQI events)

### MLflow Experiment Tracking

- **Experiment Path:** `/Users/keerthi.amulya.1999@gmail.com/AQI_ML_Models_Final_2`
- **Tracking URI:** Databricks
- **Logged Artifacts:** Models, confusion matrices, metrics

---

## Results

### Data Quality Summary

| Layer | Records | Data Retention |
|-------|---------|----------------|
| Bronze (Raw) | 299,976 | 100% |
| Silver (Cleaned) | 297,209 | 99.08% |
| Gold (ML-Ready) | 289,034 | 96.35% |

### Top 10 Most Polluted Cities (by Average AQI)

| Rank | City | Avg AQI | High Pollution Days % |
|------|------|---------|----------------------|
| 1 | Jharsuguda | 282.0 | 100% |
| 2 | Byrnihat | 248.67 | 68% |
| 3 | Begusarai | 248.14 | 60% |
| 4 | Angul | 238.13 | 69% |
| 5 | Ghaziabad | 221.65 | 53% |
| 6 | Delhi | 217.14 | 52% |
| 7 | Siwan | 216.19 | 48% |
| 8 | Bhiwadi | 215.92 | 54% |
| 9 | Barrackpore | 215.19 | 67% |
| 10 | Chhapra | 215.05 | 53% |

### Model Performance Summary

- **Best Regression Model:** Random Forest Regressor (logged to MLflow)
- **Best Classification Model:** XGBoost Classifier (89.41% accuracy, 0.81 F1)
- **Alert System:** Successfully identifies ~77% of high pollution events

---

## Technologies Used

| Category | Technologies |
|----------|-------------|
| **Platform** | Databricks (Unity Catalog enabled) |
| **Runtime** | PySpark 4.0.0, Python 3.11 |
| **Data Storage** | Delta Lake |
| **ML Frameworks** | scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow 2.11.4 |
| **Visualization** | Matplotlib, Seaborn |

### Key Libraries

```python
# Data Processing
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Experiment Tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
```

---

## How to Run

### Prerequisites

1. Databricks workspace with Unity Catalog enabled
2. Cluster with PySpark 4.0+ and Python 3.11+
3. Required libraries: scikit-learn, xgboost, mlflow

### Steps

1. **Create Unity Catalog Objects:**
   ```sql
   CREATE CATALOG IF NOT EXISTS aqi_india;
   CREATE SCHEMA IF NOT EXISTS aqi_india.bronze;
   CREATE SCHEMA IF NOT EXISTS aqi_india.silver;
   CREATE SCHEMA IF NOT EXISTS aqi_india.gold;
   CREATE VOLUME IF NOT EXISTS aqi_india.bronze.raw_data;
   ```

2. **Upload Data:**
   ```
   Upload AllIndiaBulletins_Master.csv to /Volumes/aqi_india/bronze/raw_data/
   ```

3. **Run Data Cleaning Notebook:**
   ```
   Execute: AQI Data_cleaning.ipynb.ipynb
   Creates: bronze, silver, gold tables
   ```

4. **Train ML Models:**
   ```
   Execute: AQI_ML_Models_Final_2 (1).ipynb
   Creates: MLflow experiments with logged models
   ```

5. **View Results:**
   ```
   MLflow UI: /Users/<your-email>/AQI_ML_Models_Final_2
   Tables: aqi_india.gold.aqi_alert_predictions
   ```

---

## Future Improvements

### Short-term Enhancements

1. **Hyperparameter Tuning:** Use GridSearchCV or Databricks AutoML
2. **Feature Selection:** Apply SHAP values for feature importance
3. **Model Signatures:** Add input/output signatures to MLflow models
4. **Cross-Validation:** Implement time-series cross-validation

### Long-term Goals

1. **Real-time Weather Integration:** Incorporate weather forecasts
2. **Time-Series Models:** Explore LSTM, Prophet for sequential patterns
3. **Explainability:** Add SHAP/LIME for model interpretability
4. **Dashboard:** Build interactive Databricks SQL dashboard
5. **Alert System:** Automated email/SMS notifications for high AQI predictions
6. **API Deployment:** Create REST endpoint using MLflow Model Serving

---

## Conclusion

This project successfully demonstrates:

- **Medallion Architecture:** Clean data pipeline from raw to ML-ready
- **Feature Engineering:** 52 meaningful features for AQI prediction
- **ML Models:** Both regression and classification for comprehensive forecasting
- **MLflow Integration:** Full experiment tracking and model versioning
- **Actionable Insights:** High-AQI alerts with 89% accuracy

The system is suitable for real-world applications including:
- Public health alerts
- Environmental monitoring
- Decision support systems for policymakers

---

## Acknowledgments

- **Data Source:** Central Pollution Control Board (CPCB), India
- **Repository:** Urban Emissions Info (https://github.com/urbanemissionsinfo)
- **Challenge Sponsors:** Databricks, Codebasics, Indian Data Club

---

**Contact:** Keerthi Amulya (keerthi.amulya.1999@gmail.com)

**Last Updated:** January 2026
