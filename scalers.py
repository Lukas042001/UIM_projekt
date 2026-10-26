from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.svm import SVC
# knihovna pro práci s tabulky
import pandas as pd

# Načtení dat
df = pd.read_csv("data/ocni_nalez.csv")
# X obsahuje příznaky bez target, y obsahuje pouze sloupec target
X = df.drop(columns="target")
y = df["target"]

# Stejné rozdělení pro všechny varianty, na 5 částí, před rozdělením promíchání, opakovatelné rozdělení
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Varianty předzpracování
scalers = {
    "Bez škálování": "passthrough",
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

results = []

# Porovnání scalers
for name, scaler in scalers.items():
    # doplnění chybějících hodnot mediánem
    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    # podmínka škálování, přidání do steps
    if scaler != "passthrough":
        steps.append(("scaler", scaler))

    # přídá do steps klasifikátor - SVM
    steps.append(("model", SVC(kernel="rbf", class_weight="balanced")))

    #spuštění steps - spojení všech kroků
    pipeline = Pipeline(steps)

    # hodnocení pomocí MCC
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="matthews_corrcoef"
    )

    # uložení výsledků do results
    results.append({
        "Metoda": name,
        "Průměrné MCC": scores.mean(),
        "Směrodatná odchylka MCC": scores.std()
    })

# Výsledná tabulka - seřazená podle MCC
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Průměrné MCC",
    ascending=False
)
# Vypsání tabulky bez číslování řádku
print(results_df.to_string(index=False))