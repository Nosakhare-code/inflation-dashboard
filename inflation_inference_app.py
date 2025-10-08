import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Inflation Inference", layout="wide")

# --- Typing animation ---
def typing_effect(text, delay=0.03, size="###"):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(f"{size} {typed_text}")
        time.sleep(delay)
    time.sleep(0.5)

# --- Title ---
st.title("📈 Inflation Inference")

# --- Intro with typing effect ---
intro = (
    "This project involves merging Inflation, crude oil, MPR, money supply, etc., for descriptive "
    "and inferential analysis on price of goods and services in Nigeria, enabling "
    "data-driven decision-making to achieve macroeconomic goals.\n\n"
    "It explores relationships among variables to guide contractionary and expansionary "
    "monetary policies.\n\n"
    "Note that stochastic factors like disease outbreaks or wars can affect model accuracy. "
    "Exchange rate data were excluded due to limited availability from CBN.\n\n"
    "Machine learning algorithms perform better with more data."
)
typing_effect(intro, delay=0.01)

# --- Data Sources ---
typing_effect("Data Sources", size="###")
st.markdown("""
- **Inflation data:** [Inflation Rate (CBN)](https://www.cbn.gov.ng/rates/inflrates.html)
- **Money supply:** [Money and Credit Statistics](https://www.cbn.gov.ng/rates/mnycredit.html)
- **Crude Oil:** [Crude Oil Price](https://www.cbn.gov.ng/rates/crudeoil.html)
- **Money Market:** [Money Market Indicators](https://www.cbn.gov.ng/rates/mnymktind.html)
""")

st.divider()

# --- Data Dictionary Section ---
with st.expander("📘 Data Dictionary (Click to Expand)"):
    typing_effect("Inflation Variables (from NBS CPI data)", size="###")
    st.markdown("""
    | Variable | Description | Unit / Meaning |
    |-----------|--------------|----------------|
    | **allItemsYearOn** | YoY % change in CPI for all items. | % |
    | **foodYearOn** | YoY % change in CPI for food items. | % |
    | **allItemsLessFrmProdAndEnergyYearOn** | Core inflation (excluding farm produce and energy). | % |
    """)

    typing_effect("Money Supply Variables (from CBN data)", size="###")
    st.markdown("""
    | Variable | Description | Unit |
    |-----------|--------------|------|
    | **moneySupply_M3** | Broadest money supply (M2 + other liquid assets). | ₦ billions |
    | **moneySupply_M2** | Broad money. | ₦ billions |
    | **narrowMoney** | Currency + demand deposits. | ₦ billions |
    | **creditToPrivateSector** | Credit to private sector. | ₦ billions |
    | **cbnBills** | CBN-issued securities. | ₦ billions |
    """)

st.divider()
typing_effect("Exploratory Data Analysis", size="##")

# --- Load Data ---
merge_df = pd.read_csv("merge_data.csv")

# Check if 'period' exists before converting
if "period" in merge_df.columns:
    merge_df["period"] = pd.to_datetime(merge_df["period"])
    merge_df = merge_df.sort_values("period")
else:
    st.warning("⚠️ The 'period' column is missing in merge_data.csv — plots depending on time may not render properly.")

# Add download buttons for data
st.download_button(
    label="⬇️ Download Merged Data (CSV)",
    data=merge_df.to_csv(index=False).encode("utf-8"),
    file_name="merge_data.csv",
    mime="text/csv"
)

st.divider()

# --- Inflation Trend Plot ---
if "period" in merge_df.columns:
    inflation_vars = ["allItemsYearOn", "foodYearOn", "allItemsLessFrmProdAndEnergyYearOn"]
    fig, ax = plt.subplots(figsize=(12,6))
    for col in inflation_vars:
        if col in merge_df.columns:
            ax.plot(merge_df["period"], merge_df[col], label=col)
    ax.set_title("Inflation Trends Over Time (YoY %)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Inflation Rate (%)")
    ax.legend(title="Inflation Type")
    st.pyplot(fig)

    typing_effect(
        "🟦 Overall inflation shows steady increases from mid-2021 with over 40% growth since 2008. "
        "\n🟨 Food inflation is more volatile and higher — often driving overall inflation (demand-pull). "
        "\n🟩 Core inflation is smoother since it excludes farm produce and energy (cost-push).",
        delay=0.01
    )

# --- Correlation Heatmap ---
numeric_cols = ["allItemsYearOn", "moneySupply_M3", "moneySupply_M2", "narrowMoney"]
available = [col for col in numeric_cols if col in merge_df.columns]
if len(available) > 1:
    corr = merge_df[available].corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    st.pyplot(fig)
    typing_effect("💡 High positive correlation → strong link between money supply and inflation.", delay=0.02)

# --- Distributions ---
if "allItemsYearOn" in merge_df.columns:
    typing_effect("📈 Distribution of Inflation", size="###")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(merge_df["allItemsYearOn"], bins=30, kde=True, color="skyblue", ax=ax)
    st.pyplot(fig)
    typing_effect("Non-symmetrical distribution → presence of outliers.", delay=0.02)

if "moneySupply_M3" in merge_df.columns:
    typing_effect("💵 Distribution of Broad Money Supply (M3)", size="###")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(merge_df["moneySupply_M3"], bins=30, kde=True, color="lightgreen", ax=ax)
    st.pyplot(fig)
    typing_effect("Right-skewed distribution → concentration of outliers at the end.", delay=0.02)

# --- Model Section ---
st.divider()
typing_effect("🤖 Inflation Prediction Model", size="##")

# Load model safely
try:
    model_file = joblib.load("inflation_model.pkl")
    # Detect if it's a GridSearchCV or direct estimator
    if hasattr(model_file, "best_estimator_"):
        model = model_file.best_estimator_
        best_params = model_file.best_params_
    else:
        model = model_file
        best_params = "Not available (base model only)"
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load test data
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

# Add download buttons for test datasets
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="⬇️ Download x_test (CSV)",
        data=x_test.to_csv(index=False).encode("utf-8"),
        file_name="x_test.csv",
        mime="text/csv"
    )
with col2:
    st.download_button(
        label="⬇️ Download y_test (CSV)",
        data=y_test.to_csv(index=False).encode("utf-8"),
        file_name="y_test.csv",
        mime="text/csv"
    )

# Predict and display results
model_prediction = model.predict(x_test)
model_prediction_df = pd.DataFrame(model_prediction, columns=["Model Prediction"])
model_prediction_df["True Values"] = y_test.values
st.write(model_prediction_df.head())

# Download predictions
csv = model_prediction_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Predictions (CSV)",
    data=csv,
    file_name="inflation_predictions.csv",
    mime="text/csv"
)

# --- Model Performance Info ---
typing_effect(
    f"🏆 Best model: **{type(model).__name__}**\n\n"
    f"Best parameters: `{best_params}`",
    delay=0.015
)

# --- Feature Importance ---
if hasattr(model, "feature_importances_"):
    typing_effect("💡 Feature Importance in Predicting Inflation", size="###")
    importances = model.feature_importances_
    features = x_test.columns
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=importances, y=features, hue=features, palette="viridis", legend=False, ax=ax)
    st.pyplot(fig)
else:
    st.info("ℹ️ This model type does not support feature importance.")
