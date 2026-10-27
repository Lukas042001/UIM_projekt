import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Nastavení modelu
# --------------------------------------------------

def vytvor_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("random_forest", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])


# --------------------------------------------------
# 2. Vyhodnocení modelu
# --------------------------------------------------

def vyhodnot_model(model, X, y):
    predikce = model.predict(X)
    pravdepodobnosti = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y, predikce, labels=[0, 1]
    ).ravel()

    specificita = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    return {
        "F1": f1_score(y, predikce, zero_division=0),
        "MCC": matthews_corrcoef(y, predikce),
        "ROC-AUC": roc_auc_score(y, pravdepodobnosti),
        "Recall": recall_score(y, predikce, zero_division=0),
        "Specificita": specificita,
        "Accuracy": accuracy_score(y, predikce)
    }


# --------------------------------------------------
# 3. Pětinásobná cross-validace
# --------------------------------------------------

vysledky = []

sloupce_k_odstraneni = ["target", "_zdroj"]

for i in range(1, 6):
    fold_train = pd.read_csv(f"fold_{i}_train.csv")
    fold_val = pd.read_csv(f"fold_{i}_val.csv")

    X_train = fold_train.drop(
        columns=sloupce_k_odstraneni,
        errors="ignore"
    )
    y_train = fold_train["target"]

    X_val = fold_val.drop(
        columns=sloupce_k_odstraneni,
        errors="ignore"
    )
    y_val = fold_val["target"]

    # Zachování shodných vstupních sloupců a pořadí
    X_val = X_val.reindex(columns=X_train.columns)

    # Model se učí pouze na trénovací části daného foldu
    model = vytvor_model()
    model.fit(X_train, y_train)

    metriky = vyhodnot_model(model, X_val, y_val)
    vysledky.append(metriky)

    print(f"\nFOLD {i}")
    for nazev, hodnota in metriky.items():
        print(f"{nazev}: {hodnota:.4f}")


# --------------------------------------------------
# 4. Souhrn výsledků všech foldů
# --------------------------------------------------

vysledky_df = pd.DataFrame(
    vysledky,
    index=[f"Fold {i}" for i in range(1, 6)]
)

print("\nVÝSLEDKY JEDNOTLIVÝCH FOLDŮ")
print(vysledky_df.round(4))

print("\nPRŮMĚR A SMĚRODATNÁ ODCHYLKA")
souhrn = pd.DataFrame({
    "Průměr": vysledky_df.mean(),
    "Směrodatná odchylka": vysledky_df.std(ddof=1)
})

print(souhrn.round(4))

# Uložení výsledků validace
vysledky_df.to_csv("random_forest_vysledky_folds.csv")
souhrn.to_csv("random_forest_souhrn.csv")