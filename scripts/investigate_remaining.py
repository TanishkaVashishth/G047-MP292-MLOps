
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/raw/train.csv")

print("\n=== UNIT MIX-UPS ===")
for col in ["fodder_price_rs", "feed_kg", "temp_c"]:
    print(f"\n{col}")
    print(df[col].describe())
    print("Lowest values:")
    print(df.nsmallest(5, col)[["record_id", col]].to_string(index=False))
    print("Highest values:")
    print(df.nlargest(5, col)[["record_id", col]].to_string(index=False))

print("\n=== DUPLICATE RECORD IDs ===")
duplicates = df[df.duplicated(subset=["record_id"], keep=False)]
print("Affected rows:", len(duplicates))
print(duplicates.sort_values("record_id").to_string(index=False))

print("\n=== REPEATED SENSOR VALUES ===")
for col in ["temp_c", "feed_kg", "fodder_price_rs"]:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

print("\n=== TARGET ANOMALIES ===")
target = "litres_per_day"
print(df[target].describe())
print("\nLowest targets:")
print(df.nsmallest(10, target)[["record_id", target]])
print("\nHighest targets:")
print(df.nlargest(10, target)[["record_id", target]])

print("\n=== CATEGORY NORMALIZATION CHECK ===")
for col in ["breed_mix", "village_cluster"]:
    print(f"\n{col}:")
    print(df[col].map(repr).value_counts(dropna=False))
