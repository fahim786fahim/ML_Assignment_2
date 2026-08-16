from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
st.set_page_config(
page_title="Bank Marketing Classification",
page_icon="",
layout="wide"
)
st.title("Bank Marketing Classification")
st.write(
"This application compares multiple machine learning classification "
"models for predicting whether a customer will subscribe to a term deposit."
)

#Add the model-selection dropdown
st.subheader("Select a Classification Model")
model_files = {
"Logistic Regression": "logistic_regression.pkl",
"Decision Tree": "decision_tree.pkl",
"K-Nearest Neighbors (KNN)": "knn.pkl",
"Naive Bayes": "naive_bayes.pkl",
"Random Forest": "random_forest.pkl"
}
selected_model = st.selectbox(
"Choose a model:",
options=list(model_files.keys())
)
st.write(f"Selected model: {selected_model}")

@st.cache_resource
def load_model(model_filename):
    model_path = MODEL_DIR / model_filename
    return joblib.load(model_path)

selected_model_file = model_files[selected_model]
model = load_model(selected_model_file)

st.success(f"{selected_model} model loaded successfully.")
