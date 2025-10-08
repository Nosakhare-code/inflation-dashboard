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
st.title("📈 Inflation Inference Dashboard")

# --- Intro with typing effect ---
intro = (
    "This project merges Inflation, Crude Oil, MPR, Money Supply, etc., for descriptive "
    "and inferential analysis on prices of goods and services in Nigeria enabling "
    "data-driven decision-making to achieve macroeconomic goals.\n\n"
    "It explores relationships among variables to guide contractionary and expansionary "
    "monetary policies.\n\n"
    "Note: stochastic factors like disease outbreaks or wars can affect model accuracy. "
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
with st.expander(" Data Dictionary (Click to Expand)"):
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
merge_df["period"] = pd.to_datetime(merge_df["period"])
merge_df = merge_df.sort_values("period")

# Add download buttons for data
st.download_button(
    label="⬇️ Download Merged Data (CSV)",
    data=merge_df.to_csv(index=False).encode("utf-8"),
    file_name="merge_data.csv",
    mime="text/csv"
)

st.divider()

# --- Inflation Trend Plot ---
inflation_vars = ["allItemsYearOn", "foodYearOn", "allItemsLessFrmProdAndEnergyYearOn"]
fig, ax = plt.subplots(figsize=(12,6))
for col in inflation_vars:
    ax.plot(merge_df["period"], merge_df[col], label=col)
ax.set_title("Inflation Trends Over Time (YoY %)")
ax.set_xlabel("Period")
ax.set_ylabel("Inflation Rate (%)")
ax.legend(title="Inflation Type")
st.pyplot(fig)

typing_effect(
    "🟦 Overall inflation shows steady increases from mid-2021 with over 40% growth since 2008.\n"
    "🟨 Food inflation is more volatile and higher often driving overall inflation.\n"
    "🟩 Core inflation is smoother since it excludes farm produce and energy.\n",
    delay=0.01
)

# --- Correlation Heatmap ---
corr = merge_df[["allItemsYearOn", "moneySupply_M3", "moneySupply_M2", "narrowMoney"]].corr()
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
st.pyplot(fig)
typing_effect("💡 High positive correlation (close to +1) → strong relationship between money supply and inflation.", delay=0.02)

# --- Distribution Plots ---
typing_effect("📊 Distribution of Inflation", size="###")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(merge_df["allItemsYearOn"], bins=30, kde=True, color="skyblue", ax=ax)
st.pyplot(fig)

typing_effect("💵 Distribution of Broad Money Supply (M3)", size="###")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(merge_df["moneySupply_M3"], bins=30, kde=True, color="lightgreen", ax=ax)
st.pyplot(fig)

st.divider()
typing_effect("🤖 Inflation Prediction Model", size="##")

# --- Load Trained Model ---
model = joblib.load("inflation_model.pkl")  # Directly use trained model (no GridSearchCV object)

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

# --- Feature Importance ---
typing_effect("Top 15 Most Important Features in Predicting Inflation", size="###")

# Extract feature importances
importances = model.feature_importances_
features = x_test.columns

# Combine and sort
feature_importance_df = (
    pd.DataFrame({"Feature": features, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .head(15)
)

# Plot top 15
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(
    data=feature_importance_df,
    x="Importance",
    y="Feature",
    palette="viridis"
)
ax.set_title("Top 15 Feature Importances")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature Name")
st.pyplot(fig)

# Optional: show table
st.dataframe(feature_importance_df.style.format({"Importance": "{:.4f}"}))

# --- User Upload Section ---
st.divider()
typing_effect("📤 Upload Your Own CSV to Make Predictions", size="##")

uploaded_file = st.file_uploader("Upload your CSV file (must have same columns as training data)", type=["csv"])
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("✅ File Uploaded Successfully! Preview:")
    st.dataframe(user_df.head())

    try:
        user_predictions = model.predict(user_df)
        result_df = pd.DataFrame({
            "Predicted Inflation": user_predictions
        })
        st.write("### 🔮 Predictions Preview")
        st.dataframe(result_df.head())

        # Download button
        st.download_button(
            label="📥 Download Your Predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="user_inflation_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Error making predictions: {e}")

# --- Footer / Contact Info ---
st.divider()
st.markdown("""
 **Contact:** [nosakhareasowata94@gmail.com](mailto:nosakhareasowata94@gmail.com)  
 **View the full notebook:** [GitHub Notebook Viewer](https://github.com/Nosakhare-code/inflation-dashboard/blob/main/CBN%20Money%20supply%20and%20Inflation.ipynb)
""")
