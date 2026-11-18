
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")

# ISSUE 1: Potential target leakage
print("\n=== TARGET LEAKAGE INVESTIGATION ===")

target = "litres_per_day"
candidate = "litres_paid_in_passbook"

correlation = df[[candidate, target]].corr().iloc[0, 1]
difference = (df[candidate] - df[target]).abs()

print(f"Correlation: {correlation:.6f}")
print(f"Mean absolute difference: {difference.mean():.4f}")
print(f"Maximum absolute difference: {difference.max():.4f}")

print("\nSample values:")
print(df[[candidate, target]].head(10).to_string(index=False))

# ISSUE 2: Suspicious temperature readings
print("\n=== TEMPERATURE INVESTIGATION ===")

print("Most frequent temperatures:")
print(df["temp_c"].value_counts().head(10))

print("\nTemperatures above 60°C:")
invalid = df[df["temp_c"] > 60]

print(f"Count: {len(invalid)}")
print(invalid[["record_id", "date", "temp_c"]].head(20).to_string(index=False))

print("\nTemperature statistics excluding values above 60°C:")
print(df.loc[df["temp_c"] <= 60, "temp_c"].describe())
