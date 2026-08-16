from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
roc_auc_score,
precision_score,
recall_score,
f1_score,
matthews_corrcoef
)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
from tracer.paths import PROJECT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"

df = pd.read_csv(DATA_PATH, sep=";")

print("Print dataset shape : ", df.shape)
print("\nFirst five rows")
print(df.head())

#Get datatypes, column anmes, duplicates, missing values etc
print("\n Column names : ")
print(df.columns.tolist())

print("\n Column types : ")
print(df.dtypes)

print("\n  Missing values per column : ")
print(df.isnull().sum())

print("\n Duplicate rows : ")
print(df.duplicated().sum())

print("\n Target distribution : ")
print(df["y"].value_counts())

print("\n Target distribution (%): ")
print(df["y"].value_counts(normalize=True)*100)

# Identify columns are numeric, categorical, categorical columns contain "unknown"
print("\nData types:")
print(df.dtypes)

print("\nUnique values in categorical columns:")
categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("Check : ")
print(df["marital"].dtype)
print(df.select_dtypes(include="object").columns.tolist())

#unknown check
print("\n Get the list of categorical columns which are unknows")
categorical_columns = df.select_dtypes(include="object").columns;

for col in categorical_columns:
    unknown_count = (df[col] == "unknown").sum()

    if unknown_count > 0:
        print(f"{col}: {unknown_count}")

#check on the missing values
print("\n Missing values")
print(df.isnull().sum())

print("\n Total missing values")
print(df.isnull().sum().sum())


#Separate feature X and target y, preprocessing
x = df.drop(columns=["y"])
y = df["y"]

print("\n Feature matrix shape : ", x.shape)
print("Target shape", y.shape)

print("\n Target values")
print(y.value_counts())

# Spit into trainong and test data
# Rule 80% training data, 20% test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
print("\nTraining features shape:", x_train.shape)
print("Testing features shape:", x_test.shape)
print("\nTraining target distribution:")
print(y_train.value_counts())
print("\nTesting target distribution:")
print(y_test.value_counts())

# Encode target labels: no = 0, yes = 1
y_train = y_train.map({"no": 0, "yes": 1})
y_test = y_test.map({"no": 0, "yes": 1})
print("\nEncoded training target:")
print(y_train.value_counts())
print("\nEncoded testing target:")
print(y_test.value_counts())


# Identify numerical and categorical features
numerical_features = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = x_train.select_dtypes(include=["object"]).columns.tolist()

print("\n Numerical features : ")
print(numerical_features)

print("\n Categorical features : ")
print(categorical_features)

print("\n Number of numerical features : ", len(numerical_features))
print("\n Number of categorical features : ", len(categorical_features))


#Build the preprocessing pipeline
# Preprocessing for numerical features
numerical_transformer = StandardScaler()

# Preprocessing for categorical features
categorical_transformer = OneHotEncoder(
handle_unknown="ignore"
)
# Combine both preprocessing steps
preprocessor = ColumnTransformer(
transformers=[
("num", numerical_transformer, numerical_features),
("cat", categorical_transformer, categorical_features)
]
)
print("\nPreprocessor created successfully.")

# Train the first model: Logistic Regression
# Create Logistic Regression pipeline
logistic_model = Pipeline(
steps=[
("preprocessor", preprocessor),
("classifier", LogisticRegression(
max_iter=1000,
random_state=42
))])

# Train the model
logistic_model.fit(x_train, y_train)
print("\nLogistic Regression training completed.")

#Generate predictions
# Make predictions on the test data
y_pred_logistic = logistic_model.predict(x_test)
# Probability of the positive class (1 = yes)
y_prob_logistic = logistic_model.predict_proba(x_test)[:, 1]
print("\nFirst 10 actual values:")
print(y_test.iloc[:10].tolist())
print("\nFirst 10 predicted values:")
print(y_pred_logistic[:10])
print("\nFirst 10 predicted probabilities:")
print(y_prob_logistic[:10])

#Calculate all 6 required metrics for Logistic Regression
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_auc = roc_auc_score(y_test, y_prob_logistic)
logistic_precision = precision_score(y_test, y_pred_logistic)
logistic_recall = recall_score(y_test, y_pred_logistic)
logistic_f1 = f1_score(y_test, y_pred_logistic)
logistic_mcc = matthews_corrcoef(y_test, y_pred_logistic)

print("\nLogistic Regression Metrics")
print("-----------------------------")
print(f"Accuracy : {logistic_accuracy:.4f}")
print(f"AUC Score : {logistic_auc:.4f}")
print(f"Precision : {logistic_precision:.4f}")
print(f"Recall : {logistic_recall:.4f}")
print(f"F1 Score : {logistic_f1:.4f}")
print(f"MCC Score : {logistic_mcc:.4f}")

# Confusion Matrix
logistic_cm = confusion_matrix(y_test, y_pred_logistic)
print("\nLogistic Regression Confusion Matrix")
print("------------------------------------")
print(logistic_cm)
# Classification Report
print("\nLogistic Regression Classification Report")
print("-----------------------------------------")
print(classification_report(y_test, y_pred_logistic))

# Create Decision Tree pipeline
decision_tree_model = Pipeline(
steps=[
("preprocessor", preprocessor),
("classifier", DecisionTreeClassifier(
random_state=42
))
]
)
# Train Decision Tree
decision_tree_model.fit(x_train, y_train)
print("\nDecision Tree training completed.")

# Make predictions
y_pred_tree = decision_tree_model.predict(x_test)
# Probability of positive class (1 = yes)
y_prob_tree = decision_tree_model.predict_proba(x_test)[:, 1]
# Calculate evaluation metrics
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_auc = roc_auc_score(y_test, y_prob_tree)
tree_precision = precision_score(y_test, y_pred_tree)
tree_recall = recall_score(y_test, y_pred_tree)
tree_f1 = f1_score(y_test, y_pred_tree)
tree_mcc = matthews_corrcoef(y_test, y_pred_tree)
print("\nDecision Tree Metrics")
print("---------------------")
print(f"Accuracy : {tree_accuracy:.4f}")
print(f"AUC Score : {tree_auc:.4f}")
print(f"Precision : {tree_precision:.4f}")
print(f"Recall : {tree_recall:.4f}")
print(f"F1 Score : {tree_f1:.4f}")
print(f"MCC Score : {tree_mcc:.4f}")

#Decision tree confusion matrix
tree_cm = confusion_matrix(y_test, y_pred_tree)
print("\nDecision Tree Confusion Matrix")
print("--------------------------------")
print(tree_cm)
print("\nDecision Tree Classification Report")
print("------------------------------------")
print(classification_report(y_test, y_pred_tree))

# Create KNN pipeline
knn_model = Pipeline(
steps=[
("preprocessor", preprocessor),
("classifier", KNeighborsClassifier(
n_neighbors=5
))
]
)
# Train KNN
knn_model.fit(x_train, y_train)
print("\nKNN training completed.")

# Evaluate KNN
# Make predictions
y_pred_knn = knn_model.predict(x_test)
# Probability of positive class (1 = yes)
y_prob_knn = knn_model.predict_proba(x_test)[:, 1]
# Calculate evaluation metrics
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_auc = roc_auc_score(y_test, y_prob_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)
knn_mcc = matthews_corrcoef(y_test, y_pred_knn)
print("\nKNN Metrics")
print("-----------")
print(f"Accuracy : {knn_accuracy:.4f}")
print(f"AUC Score : {knn_auc:.4f}")
print(f"Precision : {knn_precision:.4f}")
print(f"Recall : {knn_recall:.4f}")
print(f"F1 Score : {knn_f1:.4f}")
print(f"MCC Score : {knn_mcc:.4f}")

# KNN Confusion Matrix
knn_cm = confusion_matrix(y_test, y_pred_knn)
print("\nKNN Confusion Matrix")
print("--------------------")
print(knn_cm)
print("\nKNN Classification Report")
print("-------------------------")
print(classification_report(y_test, y_pred_knn))

#Model 4: Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Separate preprocessor for Gaussian Naive Bayes
nb_preprocessor = ColumnTransformer(
transformers=[
("num", StandardScaler(), numerical_features),
("cat", OneHotEncoder(
handle_unknown="ignore",
sparse_output=False
), categorical_features)
]
)
# Create Gaussian Naive Bayes pipeline
naive_bayes_model = Pipeline(
steps=[
("preprocessor", nb_preprocessor),
("classifier", GaussianNB())
]
)
# Train Naive Bayes
naive_bayes_model.fit(x_train, y_train)
print("\nNaive Bayes training completed.")

# Evaluate Naive Bayes
# Make predictions
y_pred_nb = naive_bayes_model.predict(x_test)
# Probability of positive class (1 = yes)
y_prob_nb = naive_bayes_model.predict_proba(x_test)[:, 1]
# Calculate evaluation metrics
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_auc = roc_auc_score(y_test, y_prob_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
nb_mcc = matthews_corrcoef(y_test, y_pred_nb)
print("\nNaive Bayes Metrics")
print("-------------------")
print(f"Accuracy : {nb_accuracy:.4f}")
print(f"AUC Score : {nb_auc:.4f}")
print(f"Precision : {nb_precision:.4f}")
print(f"Recall : {nb_recall:.4f}")
print(f"F1 Score : {nb_f1:.4f}")
print(f"MCC Score : {nb_mcc:.4f}")

# Naive Bayes Confusion Matrix
nb_cm = confusion_matrix(y_test, y_pred_nb)
print("\nNaive Bayes Confusion Matrix")
print("----------------------------")
print(nb_cm)
print("\nNaive Bayes Classification Report")
print("---------------------------------")
print(classification_report(y_test, y_pred_nb))

# Model 5: Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest pipeline
random_forest_model = Pipeline(
steps=[
("preprocessor", preprocessor),
("classifier", RandomForestClassifier(
n_estimators=100,
random_state=42,
n_jobs=-1))])
# Train Random Forest
random_forest_model.fit(x_train, y_train)
print("\nRandom Forest training completed.")

# Evaluate Random Forest
# Make predictions
y_pred_rf = random_forest_model.predict(x_test)
# Probability of positive class (1 = yes)
y_prob_rf = random_forest_model.predict_proba(x_test)[:, 1]
# Calculate evaluation metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_mcc = matthews_corrcoef(y_test, y_pred_rf)
print("\nRandom Forest Metrics")
print("---------------------")
print(f"Accuracy : {rf_accuracy:.4f}")
print(f"AUC Score : {rf_auc:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall : {rf_recall:.4f}")
print(f"F1 Score : {rf_f1:.4f}")
print(f"MCC Score : {rf_mcc:.4f}")

# Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix")
print("------------------------------")
print(rf_cm)
print("\nRandom Forest Classification Report")
print("-----------------------------------")
print(classification_report(y_test, y_pred_rf))

# ==========================================
# MODEL COMPARISON
# ==========================================
model_results = pd.DataFrame({
"ML Model": [
"Logistic Regression",
"Decision Tree",
"KNN",
"Naive Bayes",
"Random Forest"
],
"Accuracy": [
logistic_accuracy,
tree_accuracy,
knn_accuracy,
nb_accuracy,
rf_accuracy
],
"AUC": [
logistic_auc,
tree_auc,
knn_auc,
nb_auc,
rf_auc
],
"Precision": [
logistic_precision,
tree_precision,
knn_precision,
nb_precision,
rf_precision
],
"Recall": [
logistic_recall,
tree_recall,
knn_recall,
nb_recall,
rf_recall
],
"F1": [
logistic_f1,
tree_f1,
knn_f1,
nb_f1,
rf_f1
],
"MCC": [
logistic_mcc,
tree_mcc,
knn_mcc,
nb_mcc,
rf_mcc
]
})
print("\nFinal Model Comparison")
print("----------------------")
print(model_results.round(4).to_string(index=False))

# Save model comparison results
RESULTS_PATH = PROJECT_ROOT / "model" / "model_results.csv"
model_results.round(4).to_csv(
RESULTS_PATH,
index=False
)
print(f"\nModel comparison results saved to: {RESULTS_PATH}")

# ==========================================
# SAVE TEST DATA
# ==========================================
test_data = x_test.copy()
# Add the original target values back
test_data["y"] = y_test.map({
0: "no",
1: "yes"
})
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
test_data.to_csv(
TEST_DATA_PATH,
index=False
)
print(f"\nTest data saved to: {TEST_DATA_PATH}")
print("Test data shape:", test_data.shape)

import joblib
# ==========================================
# SAVE TRAINED MODELS
# ==========================================
joblib.dump(
logistic_model,
PROJECT_ROOT / "model" / "logistic_regression.pkl"
)
joblib.dump(
decision_tree_model,
PROJECT_ROOT / "model" / "decision_tree.pkl"
)
joblib.dump(
knn_model,
PROJECT_ROOT / "model" / "knn.pkl"
)
joblib.dump(
naive_bayes_model,
PROJECT_ROOT / "model" / "naive_bayes.pkl"
)

joblib.dump(
random_forest_model,
PROJECT_ROOT / "model" / "random_forest.pkl",
compress=3
)
print("\nAll trained models saved successfully.")