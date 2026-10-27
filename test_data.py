import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

df = pd.read_csv("ocni_nalez.csv")

# Rozdělení na trénovací a testovací data
train, test = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["target"]
)

test.to_csv("test.csv", index=False)
train.to_csv("train.csv", index=False)

# Rozdělení trénovacích dat do 5 foldů
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(
    skf.split(train, train["target"]), 1
):
    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]

    train_fold.to_csv(f"fold_{i}_train.csv", index=False)
    val_fold.to_csv(f"fold_{i}_val.csv", index=False)

    print(f"\nFOLD {i}")

    for nazev, data in [
        ("Train", train_fold),
        ("Validation", val_fold)
    ]:
        print(f"\n{nazev} ({len(data)} řádků):")
        print(
            data["target"].value_counts(normalize=True)
            .mul(100).round(2).to_string()
        )

# Výpis zastoupení tříd v testovacích datech
print(f"\nTEST ({len(test)} řádků):")
print(
    test["target"].value_counts(normalize=True)
    .mul(100).round(2).to_string()
)