
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")

print("\n=== 1. UNIT MIX-UP CHECK ===")
numeric = [
    "cows", "buffaloes", "fodder_price_rs",
    "temp_c", "vet_visits", "feed_kg"
]

for col in numeric:
    print(f"\n{col}")
    print(
        df.groupby("village_cluster")[col]
        .agg(["median", "min", "max"])
    )

print("\n=== 2. STUCK SENSOR CHECK ===")
ordered = df.sort_values("date")

for col in numeric:
    valid = ordered[col].where(
        ~ordered[col].isin([-999, -1, 999, 9999])
    )
    runs = valid.ne(valid.shift()).cumsum()

    results = (
        ordered.assign(value=valid, run=runs)
        .dropna(subset=["value"])
        .groupby("run")
        .agg(
            value=("value", "first"),
            start=("date", "min"),
            end=("date", "max"),
            count=("date", "size"),
        )
        .sort_values("count", ascending=False)
        .head(3)
    )

    print(f"\n{col}")
    print(results)

print("\n=== 3. TARGET TYPO CHECK ===")
print(
    df.nlargest(20, "litres_per_day")[
        ["record_id", "litres_per_day",
         "litres_paid_in_passbook"]
    ].to_string(index=False)
)

print("\n=== 4. SPELLING VARIANTS ===")
for col in ["breed_mix", "village_cluster"]:
    print(f"\n{col}: TRAIN")
    print(df[col].value_counts(dropna=False))

    print(f"\n{col}: LATER")
    print(later[col].value_counts(dropna=False))

print("\n=== STUCK SENSOR VERIFICATION ===")

suspect = df[
    (df["date"] >= "2026-03-12") &
    (df["date"] <= "2026-03-28")
].sort_values("date")

print("Rows in period:", len(suspect))
print("Buffaloes value counts:")
print(suspect["buffaloes"].value_counts().head(10))

print("\n=== TARGET TYPO VERIFICATION ===")

high = df.nlargest(20, "litres_per_day").copy()

high["ratio"] = (
    high["litres_per_day"] /
    high["litres_paid_in_passbook"]
)

print(
    high[
        ["record_id", "litres_per_day",
         "litres_paid_in_passbook", "ratio"]
    ].to_string(index=False)
)

print("\n=== TARGET DISTRIBUTION ===")
print(df["litres_per_day"].describe())
