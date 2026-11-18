
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

print("\n=== 1. TARGET OUTLIERS ===")
print(df.loc[
    (df["litres_per_day"] < 200) |
    (df["litres_per_day"] > 900),
    ["record_id", "date", "cows", "buffaloes",
     "litres_per_day", "litres_paid_in_passbook"]
].to_string(index=False))

print("\n=== 2. POSSIBLE UNIT MIX-UPS ===")
for col in ["fodder_price_rs", "feed_kg"]:
    print(f"\n{col}")
    print(df.groupby("village_cluster")[col].describe().to_string())

print("\n=== 3. POSSIBLE STUCK SENSOR ===")
for cluster, group in df.sort_values("date").groupby("village_cluster"):
    values = group["temp_c"]
    runs = values.ne(values.shift()).cumsum()
    longest = group.groupby(runs).agg(
        start=("date", "min"),
        end=("date", "max"),
        value=("temp_c", "first"),
        count=("temp_c", "size")
    )
    print(f"\n{cluster}")
    print(longest.sort_values("count", ascending=False).head(5).to_string())

print("\n=== 4. SPELLING VARIANTS ===")
for col in ["breed_mix", "village_cluster"]:
    print(f"\n{col}")
    print(df[col].map(repr).value_counts().to_string())

print("\n=== 5. DUPLICATE RECORD IDS ===")
print(df["record_id"].duplicated().sum())
