# Insurance Charges Prediction

## Problem
Predict medical insurance charges based on patient characteristics to aid insurers in pricing premiums.

## Dataset
- Source: Kaggle (1,338 records).
- Features: Age, sex, BMI, children, smoker, region.
- Target: Charges (USD).

## Methodology
- Preprocessed data: Encoded categoricals, scaled numericals, log-transformed charges.
- Engineered features: BMI-smoker interaction, age groups.
- Trained models: Linear Regression, Random Forest, XGBoost.
- Evaluated with RMSE, MAE, R².

## Results
- Random Forest best model: R² = X, RMSE = Y USD.
- Key predictors: Smoker status, BMI, age.
- Deployed Streamlit app: [Link].

## Implications
Helps insurers set fair premiums and informs patients about expected costs.