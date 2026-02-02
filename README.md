# Air Quality Index (AQI) Forecasting System

## Databricks 14-Days AI Challenge - Capstone Project

**Author:** Keerthi Amulya
**Date:** January 2026
**Email:** keerthi.amulya.1999@gmail.com

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://air-quality-index-aqi-forecasting-system-by-keerthiamulya.streamlit.app/)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)

---

## Live Application

**Try the live demo:** [https://air-quality-index-aqi-forecasting-system-by-keerthiamulya.streamlit.app/](https://air-quality-index-aqi-forecasting-system-by-keerthiamulya.streamlit.app/)

The application features:
- **AQI Predictions** - Get 1-day, 3-day, and 7-day forecasts for any Indian city
- **AI Chatbot** - Ask questions about air quality powered by Groq LLM
- **Pollution Alerts** - View cities with predicted high pollution levels
- **Health Recommendations** - Personalized advice based on AQI levels

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

This project implements an **AI-powered AQI forecasting and early warning system** using Databricks, PySpark, and MLflow. The system predicts AQI values 1-7 days in advance and identifies high-pollution risk days across 275 Indian cities using 9 years of historical data (2015-2023).

### Key Achievements

| Metric | Value |
|--------|-------|
| Total Records Processed | 299,976 |
| Clean Records (Silver) | 297,209 |
| ML-Ready Records (Gold) | 289,034 |
| Cities Covered | 275 |
| Time Period | 2015-2023 |
| ML Features Engineered | 52 |
| Models Trained | 15 (9 regression + 6 classification) |
| Best Regression R² | 0.812 (XGBoost) |
| Best Classification Accuracy | 91.6% (Random Forest) |

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

1. **Predicts AQI values 1, 3, and 7 days in advance** using historical patterns and machine learning
2. **Identifies pollution trends** across different cities and seasons
3. **Generates early warnings** when AQI is predicted to exceed safe thresholds (>150)
4. **Provides an AI chatbot** for personalized air quality guidance
5. **Delivers actionable insights** through an interactive web interface

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

### Medallion Architecture (Bronze → Silver → Gold)

```
aqi_india (Unity Catalog)
│
├── bronze (Raw Data Layer)
│   └── aqi_bulletins (299,976 records)
│
├── silver (Cleaned Data Layer)
│   └── aqi_cleaned (297,209 records, 20 columns)
│       - Partitioned by year, month
│       - ZORDER optimized by city, date
│
└── gold (Analytics Layer)
    ├── aqi_ml_features (289,034 records, 52 features)
    ├── city_summary (275 cities aggregated)
    └── monthly_trends (Time-based aggregates)
```

### Application Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│  ┌─────────┐  ┌─────────────┐  ┌─────────┐  ┌────────┐ │
│  │  Home   │  │ Predictions │  │ Chatbot │  │ Alerts │ │
│  └─────────┘  └─────────────┘  └─────────┘  └────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ /predict │  │  /chat   │  │ /alerts  │  │ /cities │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  MLflow Models  │       │    Groq LLM     │
│    (XGBoost)    │       │ (llama-3.1-8b)  │
└─────────────────┘       └─────────────────┘
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
| 5 | Standardize city names | 277 → 275 cities (2 merged) |
| 6 | Remove duplicate date-city combinations | 2,763 duplicates removed (0.92%) |
| 7 | Add temporal features | year, month, day_of_week, quarter, week_of_year, season, is_weekend |
| 8 | Create AQI risk categories | Good, Satisfactory, Moderate, Poor, Very Poor, Severe |
| 9 | Handle missing values | pollutant NULLs → 'Unknown' |
| 10 | Create final schema | 20 columns, partitioned by year/month |

**Final Silver Schema:** 297,209 records, 20 columns

### Gold Layer (Feature Engineering)

**52 ML-ready features** organized into groups:

| Feature Group | Count | Features |
|---------------|-------|----------|
| **Identifiers** | 7 | date, city, year, month, day_of_week, quarter, season |
| **Current Values** | 6 | aqi, station_count, aqi_category, aqi_risk_level, prominent_pollutant, is_weekend |
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

### Multi-Horizon Regression Models (Predict AQI Values)

Three models trained for 1-day, 3-day, and 7-day forecasting:

| Model | 1-Day RMSE | 3-Day RMSE | 7-Day RMSE | Best R² |
|-------|-----------|-----------|-----------|---------|
| Linear Regression | 37.18 | 46.71 | 50.13 | 0.806 |
| Random Forest | 36.82 | 44.40 | 45.85 | 0.810 |
| **XGBoost** | **36.60** | **43.87** | **45.50** | **0.812** |

### Classification Models (High AQI Alert: AQI > 150)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 89.7% | 0.642 | 0.879 | 0.742 |
| **Random Forest** | **91.6%** | **0.711** | **0.840** | **0.770** |
| XGBoost | 91.3% | 0.700 | 0.849 | 0.767 |

### Confusion Matrix (Logistic Regression)

```
                      Pred_No_Alert  Pred_Alert
Actual_No_Alert           38,834       2,417
Actual_Alert               3,763      12,739
```

- **True Negatives:** 38,834 (correctly predicted normal AQI)
- **True Positives:** 12,739 (correctly predicted high AQI)
- **False Positives:** 2,417 (false alarms)
- **False Negatives:** 3,763 (missed high AQI events)

### MLflow Experiment Tracking

- **Experiment Path:** `/Users/keerthi.amulya.1999@gmail.com/AQI_ML_Pipeline_Complete_2`
- **Tracking URI:** Databricks
- **Total Models Logged:** 15 (9 regression + 6 classification)
- **Artifacts:** Models, confusion matrices, metrics, feature importance

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

| Task | Best Model | Key Metric |
|------|------------|------------|
| Regression (1-day) | XGBoost | RMSE: 36.60, R²: 0.812 |
| Regression (3-day) | XGBoost | RMSE: 43.87 |
| Regression (7-day) | XGBoost | RMSE: 45.50 |
| Classification | Random Forest | Accuracy: 91.6%, F1: 0.770 |

---

## Project Structure

```
Air-Quality-Index-AQI-Forecasting-System/
│
├── README.md
│
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   ├── streamlit_deploy.py       # Standalone deployment version
│   ├── aqi_api_fastapi.py        # FastAPI backend
│   ├── requirements.txt          # Python dependencies
│   └── .env                      # Environment variables
│
├── data/
│   └── AllIndiaBulletins_Master.csv    # Raw dataset (299,976 records)
│
├── Data Cleaning/
│   └── AQI Data_cleaning.ipynb.ipynb   # Bronze → Silver → Gold pipeline
│
├── Data Visualization/
│   ├── AQI_SQL_commands.ipynb          # SQL queries for analysis
│   ├── AQI_Visualizations_Dashboard.ipynb  # Databricks dashboard
│   └── Screenshot/                     # Dashboard screenshots
│
└── ML_Implementation/
    └── AQI_ML_Pipeline_Complete_3.ipynb    # Model training & MLflow
```

---

## Technologies Used

### Platform & Infrastructure

| Component | Technology |
|-----------|-----------|
| Cloud Platform | Databricks (Unity Catalog) |
| Distributed Processing | Apache Spark 4.0.0 |
| Data Lake | Delta Lake |
| Runtime | Python 3.11 |

### Machine Learning

| Component | Technology |
|-----------|-----------|
| ML Frameworks | scikit-learn, XGBoost |
| Experiment Tracking | MLflow 2.11.4 |
| Model Registry | MLflow |

### Application Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.39.0 |
| Backend | FastAPI |
| LLM Integration | Groq API (llama-3.1-8b-instant) |
| Visualization | Plotly 5.24.0 |

### Key Dependencies

```
streamlit==1.39.0
plotly==5.24.0
python-dotenv==1.0.1
groq==0.9.0
httpx==0.27.2
fastapi==0.115.0
uvicorn==0.30.0
scikit-learn
xgboost
mlflow
```

---

## How to Run

### Option 1: Use the Live Application

Visit: [https://air-quality-index-aqi-forecasting-system-by-keerthiamulya.streamlit.app/](https://air-quality-index-aqi-forecasting-system-by-keerthiamulya.streamlit.app/)

### Option 2: Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KeerthiAmulyaMadireddy/Air-Quality-Index-AQI-Forecasting-System.git
   cd Air-Quality-Index-AQI-Forecasting-System
   ```

2. **Install dependencies:**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file with:
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

### Option 3: Run on Databricks

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

3. **Run Notebooks:**
   - `Data Cleaning/AQI Data_cleaning.ipynb.ipynb` - Creates bronze, silver, gold tables
   - `ML_Implementation/AQI_ML_Pipeline_Complete_3.ipynb` - Trains and logs models to MLflow

---

## Future Improvements

### Short-term Enhancements

- Hyperparameter tuning with GridSearchCV or Databricks AutoML
- Feature selection using SHAP values
- Time-series cross-validation implementation
- Model signatures in MLflow

### Long-term Goals

- Real-time weather data integration
- LSTM/Prophet models for sequential patterns
- Model explainability with SHAP/LIME
- Automated email/SMS alert notifications
- Mobile application development

---

## Acknowledgments

- **Data Source:** Central Pollution Control Board (CPCB), India
- **Repository:** Urban Emissions Info (https://github.com/urbanemissionsinfo)
- **Challenge Sponsors:** Databricks, Codebasics, Indian Data Club
- **LLM Provider:** Groq

---

**Author:** Keerthi Amulya
**Contact:** keerthi.amulya.1999@gmail.com
**Live Demo:** [Streamlit App](https://air-quality-index-aqi-forecasting-system-by-keerthiamulya.streamlit.app/)

**Last Updated:** January 2026
