# Kombinace předzpracování dat a použití random forest na jednotlivé foldy

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)

# ==================================================
# 1. PŘEDZPRACOVÁNÍ JEDNOHO FOLDU
# ==================================================

skupiny = {
    "ma_detekce": [f"ma_detekce_{i:02d}" for i in range(1, 7)],
    "exsudaty": [f"exsudaty_{i:02d}" for i in range(1, 9)]
}

def predzpracuj_fold(train_df, val_df):
    train_df = train_df.copy()
    val_df = val_df.copy()

    # Odstranění odlehlých hodnot podle hranic
    # vypočítaných pouze z trénovací části foldu
    for sloupce in skupiny.values():
        for sloupec in sloupce:
            prumer = train_df[sloupec].mean()
            std = train_df[sloupec].std()

            if pd.isna(prumer) or pd.isna(std):
                continue

            dolni = prumer - 3 * std
            horni = prumer + 3 * std

            train_df.loc[
                (train_df[sloupec] < dolni) |
                (train_df[sloupec] > horni),
                sloupec
            ] = np.nan

            val_df.loc[
                (val_df[sloupec] < dolni) |
                (val_df[sloupec] > horni),
                sloupec
            ] = np.nan

    # Výběr nejspolehlivější dostupné hodnoty
    puvodni_sloupce = []

    for nazev, sloupce in skupiny.items():
        serazene = sorted(
            sloupce,
            key=lambda s: int(s.rsplit("_", 1)[1]),
            reverse=True
        )

        puvodni_sloupce.extend(sloupce)

        train_df[nazev + "_vybrano"] = (
            train_df[serazene].bfill(axis=1).iloc[:, 0]
        )
        val_df[nazev + "_vybrano"] = (
            val_df[serazene].bfill(axis=1).iloc[:, 0]
        )

    # Vyřazení původních variant měření
    train_df.drop(columns=puvodni_sloupce, inplace=True)
    val_df.drop(columns=puvodni_sloupce, inplace=True)

    # Oddělení příznaků a cíle
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    # Pomocné sloupce se zdrojem nejsou vstupní příznaky
    pomocne = [
        c for c in X_train.columns
        if c.endswith("_zdroj")
    ]

    X_train.drop(columns=pomocne, inplace=True, errors="ignore")
    X_val.drop(columns=pomocne, inplace=True, errors="ignore")

    # Pouze číselné příznaky
    X_train = X_train.select_dtypes(include="number")
    X_val = X_val.reindex(columns=X_train.columns)

    # Doplnění chybějících hodnot mediánem z trénovacího foldu
    mediany = X_train.median()
    platne = mediany[mediany.notna()].index

    X_train = X_train[platne].fillna(mediany[platne])
    X_val = X_val[platne].fillna(mediany[platne])

    return X_train, y_train, X_val, y_val


# ==================================================
# 2. RANDOM FOREST – EXISTUJÍCÍCH 5 FOLDŮ
# ==================================================

vysledky = []

for i in range(1, 6):
    print(f"\n{'=' * 15} FOLD {i} {'=' * 15}")

    # Načteme pouze příslušnou dvojici foldů
    train_fold = pd.read_csv(f"fold_{i}_train.csv")
    val_fold = pd.read_csv(f"fold_{i}_val.csv")

    X_train, y_train, X_val, y_val = predzpracuj_fold(
        train_fold, val_fold
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predikce = model.predict(X_val)
    pravdepodobnosti = model.predict_proba(X_val)

    index_1 = list(model.classes_).index(1)
    pravdepodobnost_1 = pravdepodobnosti[:, index_1]

    tn, fp, fn, tp = confusion_matrix(
        y_val, predikce, labels=[0, 1]
    ).ravel()

    specificita = tn / (tn + fp) if (tn + fp) else 0.0

    metriky = {
        "Fold": i,
        "F1": f1_score(y_val, predikce, zero_division=0),
        "MCC": matthews_corrcoef(y_val, predikce),
        "ROC-AUC": roc_auc_score(y_val, pravdepodobnost_1),
        "Recall": recall_score(y_val, predikce, zero_division=0),
        "Specificita": specificita,
        "Accuracy": accuracy_score(y_val, predikce)
    }

    vysledky.append(metriky)

    for nazev, hodnota in metriky.items():
        if nazev != "Fold":
            print(f"{nazev}: {hodnota:.4f}")


# ==================================================
# 3. SOUHRN
# ==================================================

vysledky_df = pd.DataFrame(vysledky).set_index("Fold")

print("\nVÝSLEDKY JEDNOTLIVÝCH FOLDŮ")
print(vysledky_df.round(4))

souhrn = pd.DataFrame({
    "Průměr": vysledky_df.mean(),
    "Směrodatná odchylka": vysledky_df.std(ddof=1)
})

print("\nSOUHRN VÝSLEDKŮ")
print(souhrn.round(4))

vysledky_df.to_csv("random_forest_vysledky_folds.csv")
souhrn.to_csv("random_forest_souhrn.csv")

print("\nHotovo. test.csv nebyl načten ani použit.")