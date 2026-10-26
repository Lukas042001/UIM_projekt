import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Načtení dat
df = pd.read_csv("ocni_nalez.csv")

data = df.iloc[:, :-1]

# 3. Výběr pouze číselných sloupců
data = data.select_dtypes(include="number")

# 4. Standardizace dat (průměr = 0, směrodatná odchylka = 1)
scaler = StandardScaler()
df = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns,
    index=data.index
)

# Skupiny sloupců
skupiny = {
    "ma_detekce": [
        f"ma_detekce_{i:02d}" for i in range(1, 7)
    ],
    "exsudaty": [
        f"exsudaty_{i:02d}" for i in range(1, 9)
    ]
}

# 1. Odstranění odlehlých hodnot pomocí 3 * std
for nazev, sloupce in skupiny.items():
    for sloupec in sloupce:
        prumer = df[sloupec].mean()
        std = df[sloupec].std()

        dolni_hranice = prumer - 3 * std
        horni_hranice = prumer + 3 * std

        maska = (
            (df[sloupec] < dolni_hranice) |
            (df[sloupec] > horni_hranice)
        )

        df.loc[maska, sloupec] = np.nan
        
df.hist(bins=10, figsize=(14, 10), edgecolor='black')
plt.suptitle('Histogramy', fontsize=16)
plt.tight_layout()
plt.show()

# BOX-PLOTY
plt.figure(figsize=(14, 6))
df.boxplot(rot=90)

plt.title('Box-ploty', fontsize=14)
plt.ylabel('Hodnota')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Výběr nejspolehlivější dostupné hodnoty
for nazev, sloupce in skupiny.items():

    # Slovník: číslo spolehlivosti -> název sloupce
    sloupce_dict = {
        int(sloupec[-1]): sloupec
        for sloupec in sloupce
    }

    # Seřazení od nejvyšší spolehlivosti
    sloupce_serazene = [
        sloupce_dict[klic]
        for klic in sorted(sloupce_dict, reverse=True)
    ]

    # Výběr první dostupné hodnoty
    df[nazev + "_vybrano"] = (
        df[sloupce_serazene].bfill(axis=1).iloc[:, 0]
    )

    # Uložení názvu zdrojového sloupce
    df[nazev + "_zdroj"] = (
        df[sloupce_serazene].notna().idxmax(axis=1)
    )

    # Pokud jsou všechny hodnoty NaN, zdroj bude NaN
    vse_nan = df[sloupce_serazene].isna().all(axis=1)
    df.loc[vse_nan, nazev + "_zdroj"] = np.nan

# BOX-PLOTY
plt.figure(figsize=(14, 6))
df_plot = df.iloc[:, [-2, -4]]
df_plot.boxplot(rot=90)

plt.title('Box-ploty', fontsize=14)
plt.ylabel('Hodnota')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

df["vzdalenost_makula_papila"] = df["vzdalenost_makula_papila"].fillna(df["vzdalenost_makula_papila"].median())
df["prumer_papily"] = df["prumer_papily"].fillna(df["prumer_papily"].median())

vysledky = df.iloc[:, [1,5, 9, 10, 13, 19, 20, 22]]

print(vysledky)

# print(df.iloc[:2, :])
# # 3. Uložení výsledku
# df.to_csv("ocni_nalez_vybrane_hodnoty.csv", index=False)

# print("Hotovo! Výsledky byly uloženy.")