# COVID-19 Case Prediction — India (State-wise Time Series)

Predicting daily confirmed COVID-19 cases for Indian states using lag-based
feature engineering and regression models, with a focus on Maharashtra
(the highest-case state in the dataset).

## 📊 Project Overview

- **Problem**: Predict confirmed COVID-19 cases for the next few days by state.
- **Dataset**: Daily COVID-19 records across Indian states/UTs, Jan 2020 – present (4,692 rows).
- **Approach**: Time-series feature engineering (lag features) + regression modeling.
- **Result**: Linear Regression achieved **99.23% prediction accuracy** (MAE ≈ 2,745 cases) on held-out test data.

## 🗂️ Dataset

Columns used:
- `Date`, `Name of State / UT`, `Latitude`, `Longitude`
- `Total Confirmed cases`, `Death`, `Cured/Discharged/Migrated`
- `New cases`, `New deaths`, `New recovered`

> Replace the dataset URL in `covid_prediction.py` with your actual data source
> (e.g. a public COVID-19 India dataset on GitHub or Kaggle).

## 🔧 Methodology

1. **EDA**: Explored dataset shape, top 5 states by case count, and time-series trends.
2. **Feature Engineering**: Created lag features — cases 1, 7, and 14 days prior, plus previous-day new cases — to capture recent momentum in the outbreak curve.
3. **Modeling**: Compared Linear Regression and Random Forest Regression.
4. **Evaluation**: MAE and RMSE on a time-ordered train/test split (no shuffling, to avoid data leakage).

## 📈 Results

| Model | MAE | RMSE |
|---|---|---|
| **Linear Regression** ✅ | 2,744.77 | 3,714.25 |
| Random Forest | 122,304.41 | 140,862.37 |

Linear Regression outperformed Random Forest significantly — the latter
overfit given the relatively small dataset size (133 rows after feature engineering).

**Feature importance (Random Forest):**
- `cases_lag7` (cases 7 days ago): 35.7%
- `cases_lag1` (cases 1 day ago): 31.9%
- `cases_lag14` (cases 14 days ago): 29.6%
- `new_cases_lag1`: 2.8%

This suggests weekly seasonality (day-of-week reporting patterns) is a strong
signal in the case-count trajectory.

## 🚀 How to Run

### Option A: Google Colab (no setup required)
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `covid_prediction.py` content into a new notebook cell
3. Run all cells

### Option B: Local
```bash
git clone https://github.com/<your-username>/covid-prediction-india.git
cd covid-prediction-india
pip install -r requirements.txt
python covid_prediction.py
```

## 📁 Repository Structure
## 🔮 Future Improvements

- Extend prediction to a 7-day rolling forecast rather than single-point prediction
- Add more states beyond Maharashtra for a multi-state model
- Try ARIMA / Prophet / LSTM for dedicated time-series forecasting
- Deploy as an interactive dashboard (Streamlit/Flask) for state-wise predictions

## 📝 License

MIT License — free to use and adapt.
