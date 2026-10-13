# Telco Customer Churn Prediction

A machine learning web application that predicts customer churn risk using a trained CatBoost model. The app is built with Streamlit and provides churn probability, risk insights, and customer retention suggestions.

## Features

- Predict customer churn probability
- Display customer risk level and insights
- Suggest customer retention actions
- Use a trained CatBoost model with a preprocessing pipeline

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- CatBoost
- Joblib

## Run Locally

1. Clone or download this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Required Artifacts

Ensure these files are present in the project:

- `artifacts/preprocessor.pkl`
- `artifacts/selected_features.pkl`
- `artifacts/models/CatBoost.pkl`

## Model

The app uses a trained CatBoost classifier. Reported baseline test ROC AUC: **0.8369**. Model performance may vary if the model, data, or preprocessing pipeline changes.

## Author

Archit Tomar
