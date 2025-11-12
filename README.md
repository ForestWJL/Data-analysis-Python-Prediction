# Data-analysis-Python-Prediction

End-to-end Python ML pipeline to **predict Singapore HDB resale prices** from tabular features.  
Includes ETL/EDA, feature engineering, model benchmarking, cross-validation & tuning, and explainability.

## 🔍 Problem & Outcome
- **Goal:** learn a mapping from housing attributes (flat type, floor area, lease age, town, month) to **resale price**.
- **Approach:** baseline Linear Regression → Tree → **Random Forest** with K-fold CV and hyperparameter search.
- **Outcome:** strong fit on hold-out data (high R², low RMSE) with interpretable drivers (area, time, lease age, town).
  *(Exact metrics are inside the notebook outputs.)*

## 🧱 Tech Stack
`python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `plotly.express` · `scikit-learn`

## 📁 Repository Structure
# Data-analysis-Python-Prediction
This repo showcases a production-style Python ML pipeline for tabular prediction. leverage matplotlib, seaborn, pandas, numpy, plotly.express, scikit-learn to perform feature engineering (encoders, scaling, date/ratio features), model benchmarking (Linear/Tree/Ensemble), cross-validation &amp; hyper-tuning, error analysis, SHAP explainability,
