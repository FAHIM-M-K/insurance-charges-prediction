# Insurance Charges Prediction

## Overview
Built a Random Forest regression model to predict medical insurance charges (R²: X). Deployed an interactive Streamlit app.

## Live Demo
Try the app: https://insurance-charges-prediction-g7a1.onrender.com

## Features
- EDA with visualizations (e.g., smoker vs. charges).
- Preprocessing: Encoded categorical variables, scaled numerical features.
- Feature engineering: BMI-smoker interaction, age bins.
- Models: Linear Regression, Random Forest, XGBoost.
- Deployment: Streamlit app on Render.

## How to Run
1. Clone the repo: `git clone https://github.com/FAHIM-M-K/insurance-charges-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook insurance_charges_prediction.ipynb`
4. Run the app: `streamlit run app.py`

## Results
- Random Forest: R² = X, RMSE = Y USD.
- Key predictors: Smoker status, BMI, age.