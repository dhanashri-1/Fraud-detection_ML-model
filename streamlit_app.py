import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load model
model = joblib.load("fraud_xgb_model.pkl")

# Get model feature names (important for CSV upload)
feature_names = model.get_booster().feature_names

# App title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection App")
st.markdown("Enter transaction details manually **or upload a CSV file** to predict whether they are **fraudulent** or **legitimate**.")

# Sidebar
st.sidebar.header("📥 Input Transaction Features")

# Function to get manual input
def user_input_features():
    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = st.sidebar.slider(f'V{i}', -30.0, 30.0, 0.0)
    amount = st.sidebar.slider('💰 Amount', 0.0, 3000.0, 1.0)
    time = st.sidebar.slider('⏱️ Time (seconds)', 0, 172800, 0)
    data = {'Time': time, **v_features, 'Amount': amount}
    return pd.DataFrame([data])

# === Section 1: Manual Prediction === #
st.subheader("🖐️ Manual Transaction Entry")

input_df = user_input_features()

st.subheader("📊 Input Feature Summary")
st.dataframe(input_df.T.style.format(precision=2))

if st.button("🔎 Predict Fraud"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🎯 Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.subheader("🔢 Fraud Probability Score")
    st.progress(int(prob * 100))
    st.write(f"🧠 Confidence of fraud: **{prob:.2%}**")

    st.subheader("📈 Visual Breakdown")
    fig1, ax1 = plt.subplots()
    labels = ['Legitimate', 'Fraud']
    sizes = [1 - prob, prob]
    colors = ['green', 'red']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# === Section 2: Batch Prediction via CSV Upload === #
st.markdown("---")
st.subheader("📁 Upload Dataset (CSV) for Batch Fraud Detection")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:")
        st.dataframe(uploaded_df.head())

        if set(feature_names).issubset(set(uploaded_df.columns)):
            # Predict
            preds = model.predict(uploaded_df[feature_names])
            probs = model.predict_proba(uploaded_df[feature_names])[:, 1]

            uploaded_df["Prediction"] = preds
            uploaded_df["Fraud Probability (%)"] = (probs * 100).round(2)

            st.success("🎯 Predictions generated successfully!")
            st.dataframe(uploaded_df[["Prediction", "Fraud Probability (%)"] + feature_names].head())

            # Downloadable CSV
            csv_data = uploaded_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", data=csv_data, file_name="fraud_predictions.csv", mime="text/csv")
        else:
            st.warning("⚠️ Uploaded CSV does not contain all required columns.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
