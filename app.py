import streamlit as st
import pandas as pd
import joblib
import os

# ------------------ Page Config ------------------
st.set_page_config(page_title="Insurance Expenses Predictor", page_icon="💰")
st.title("🏥 Insurance Expenses Prediction App")
st.write("Predict future medical expenses based on individual profiles.")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

best_model = load_model()

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Enter Input Features")
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex": sex,
    "smoker": smoker,
    "region": region
}])

st.write("### Input Data")
st.dataframe(input_df)

# ------------------ Prediction ------------------
if st.button("Predict Future Expenses"):
    prediction = best_model.predict(input_df)[0]
    st.success(f"💡 Estimated Future Expenses: **${prediction:,.2f}**")

# ------------------ Model Metrics ------------------
if os.path.exists("model_metrics_comparison.csv"):
    st.write("### 📊 Model Performance Metrics")
    metrics_df = pd.read_csv("model_metrics_comparison.csv")
    st.dataframe(metrics_df)

# ------------------ EDA Summary ------------------
if os.path.exists("eda_plots_insurance/eda_summary.txt"):
    with open("eda_plots_insurance/eda_summary.txt", "r", encoding="utf-8") as f:
        eda_text = f.read()
    st.write("### 🧾 EDA Summary")
    st.text(eda_text)