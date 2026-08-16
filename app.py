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