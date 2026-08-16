from pathlib import Path

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