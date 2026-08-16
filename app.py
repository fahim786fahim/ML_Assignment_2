from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
accuracy_score,
roc_auc_score,
precision_score,
recall_score,
f1_score,
matthews_corrcoef,
confusion_matrix
)

from sklearn.metrics import (
accuracy_score,
roc_auc_score,
precision_score,
recall_score,
f1_score,
matthews_corrcoef)

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

# Upload test file
st.subheader("Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload the test_data.csv file",
    type=["csv"]
)
if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.success("Test data uploaded successfully.")
    st.write("Test data shape:", test_df.shape)
    st.write("Preview of uploaded test data:")
    st.dataframe(test_df.head())

    # Make predictions on the uploaded CSV
    # Separate features and actual target
    x_uploaded = test_df.drop(columns=["y"])
    y_actual = test_df["y"].map({
        "no": 0,
        "yes": 1
    })
    # Make predictions using the selected model
    y_pred = model.predict(x_uploaded)
    # Get probability of positive class
    y_prob = model.predict_proba(x_uploaded)[:, 1]
    st.subheader("Prediction Results")
    prediction_results = pd.DataFrame({
        "Actual": y_actual,
        "Predicted": y_pred,
        "Probability of Subscription": y_prob
    })
    st.dataframe(prediction_results.head(10))

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_actual, y_pred)
    auc = roc_auc_score(y_actual, y_prob)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    mcc = matthews_corrcoef(y_actual, y_pred)
    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("Precision", f"{precision:.4f}")
    with col2:
        st.metric("AUC Score", f"{auc:.4f}")
    st.metric("Recall", f"{recall:.4f}")
    with col3:
        st.metric("F1 Score", f"{f1:.4f}")
    st.metric("MCC Score", f"{mcc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_actual, y_pred)
    fig, ax = plt.subplots()
    matrix_plot = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    # Display values inside the matrix
    for i in range(2):
        for j in range(2):
            ax.text(j,i, cm[i, j], ha="center", va="center")
    fig.colorbar(matrix_plot, ax=ax)
    st.pyplot(fig)
else:
    st.info("Please upload test_data.csv to evaluate the selected model.")

