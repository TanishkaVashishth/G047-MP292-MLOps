
from pathlib import Path
import pandas as pd

# Locate the dataset relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

df = pd.read_csv(DATA_PATH)

print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== FIRST FIVE ROWS ===")
print(df.head().to_string())

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== EXACT DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== NUMERICAL SUMMARY ===")
print(df.describe().to_string())

print("\n=== CATEGORICAL VALUES ===")
for column in df.select_dtypes(include=["object", "string"]).columns:
    if column not in ["record_id", "date"]:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).head(20))
