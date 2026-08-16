from pathlib import Path

import pandas as pd
import numpy as np
from tracer.paths import PROJECT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"

df = pd.read_csv(DATA_PATH)

print("Print dataset shape : ", df.shape)
print("\nFirst five rows")
print(df.head())

