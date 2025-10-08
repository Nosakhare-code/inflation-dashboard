# Inflation Inference Dashboard
## Overview

This Streamlit dashboard analyzes and predicts inflation trends in Nigeria using data from the Central Bank of Nigeria (CBN).
It explores how money supply, monetary policy rate (MPR), and crude oil prices influence inflation providing insights for macroeconomic planning.

## Project Objective

The project merges inflation and financial data to:

1. Explore the relationship between money supply and inflation.
2. Understand demand-pull vs cost-push inflation trends.
3. Predict future inflation rates using machine learning (Random Forest Regressor).

Data Sources

All datasets are obtained from official CBN and NBS portals:

- Inflation Rate (CBN) https://www.cbn.gov.ng/rates/inflrates.html
- Money and Credit Statistics https://www.cbn.gov.ng/rates/mnycredit.html
- Crude Oil Prices https://www.cbn.gov.ng/rates/crudeoil.html
- Money Market Indicators https://www.cbn.gov.ng/rates/mnymktind.html

## Machine Learning Approach

- Algorithm: RandomForestRegressor
- Goal: Predict inflation rates using economic indicators
- Evaluation Metrics: R² Score, MAE, RMSLE
- Tools: Python (pandas, scikit-learn, seaborn, matplotlib, Streamlit)

## Installation
1. Clone the Repository
git clone https://github.com/Nosakhare-code/inflation-dashboard.git cd inflation-dashboard

2. Install Dependencies
pip install -r requirements.txt

3. Run the App Locally
streamlit run inflation_inference_app.py

## Deploy on Streamlit Cloud

1. Push your files (inflation_inference_app.py, datasets, and requirements.txt) to GitHub.
2. Go to Streamlit Cloud and deploy the app using your repo URL.
3. The app automatically detects dependencies from requirements.txt.

## Upload Your Own Data
You can upload a CSV with the same structure as the training data to generate your own inflation predictions interactively on the dashboard.

## APP Features

1. Inflation trend visualization (YoY, Food, Core Inflation)
2. Correlation and distribution analysis
3. Machine learning predictions
4. Top 15 feature importances
5. User-upload CSV prediction tool
6. Downloadable results and visualizations

You can view the full data analysis and preprocessing steps here: https://github.com/Nosakhare-code/inflation-dashboard/blob/main/CBN%20Money%20supply%20and%20Inflation.ipynb
## Note
### it appears that there is multicollinearity on the correlation analysis. no owrries, Tree models(e.g Randomforest ) don’t estimate coefficients like regression.
### They split the data based on thresholds (e.g., “moneySupply_M3 > 2000”), not on linear relationships.### 

## Contact
Author: Emmanuel Nosakhare Asowata
Email: nosakhareasowata94@gmail.com

## Acknowledgement
Central Bank of Nigeria (CBN) for macroeconomic data
