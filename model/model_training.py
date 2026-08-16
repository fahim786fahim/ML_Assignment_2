from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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






