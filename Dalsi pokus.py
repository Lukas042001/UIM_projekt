import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. NASTAVENÍ
# ============================================================

VSTUPNI_SOUBOR = "UIM_projekt/ocni_nalez.csv"

VYSTUP_SOHRN = "data_souhrn.csv"
VYSTUP_BINARNI = "data_binarni.csv"

VYSTUP_BOXPLOTY = "boxploty_predzpracovani.png"
VYSTUP_HEATMAPA = "korelace_binarni_heatmap.png"


# ------------------------------------------------------------
# STANDARDIZACE
# ------------------------------------------------------------
# True  = standardizace se provede
# False = standardizace se neprovede
#
# Pokud ji budeš chtít vypnout, stačí změnit True na False.
# Zbytek programu bude fungovat stejně.
# ------------------------------------------------------------

PROVEST_STANDARDIZACI = True


# ------------------------------------------------------------
# HRANICE PRO OUTLIERY
# ------------------------------------------------------------
# Za outlier považujeme hodnotu mimo:
#
# průměr +/- 3 * směrodatná odchylka
# ------------------------------------------------------------

STD_LIMIT = 3


# ============================================================
# 1. NAČTENÍ DAT
# ============================================================

print("=" * 80)
print("1. NAČTENÍ DAT")
print("=" * 80)

df = pd.read_csv(VSTUPNI_SOUBOR)

print(f"\nPočet řádků: {df.shape[0]}")
print(f"Počet sloupců: {df.shape[1]}")

print("\nNázvy sloupců:")
print(df.columns.tolist())

print("\nPočet chybějících hodnot v jednotlivých sloupcích:")
print(df.isna().sum())

print("\nCelkový počet chybějících hodnot:")
print(df.isna().sum().sum())





# ============================================================
# ODSTRANĚNÍ ZÁZNAMŮ S NEDOSTATEČNOU KVALITOU
# ============================================================

pocet_pred = len(df)

df = df[df["kvalita"] == 1].copy()

pocet_po = len(df)

print("\nOdstranění záznamů s nedostatečnou kvalitou:")
print(f"Počet záznamů před: {pocet_pred}")
print(f"Počet odstraněných: {pocet_pred - pocet_po}")
print(f"Počet záznamů po: {pocet_po}")




# ============================================================
# 2. DEFINICE SLOUPCŮ
# ============================================================

ma_cols = [
    "ma_detekce_01",
    "ma_detekce_02",
    "ma_detekce_03",
    "ma_detekce_04",
    "ma_detekce_05",
    "ma_detekce_06"
]


exsudaty_cols = [
    "exsudaty_01",
    "exsudaty_02",
    "exsudaty_03",
    "exsudaty_04",
    "exsudaty_05",
    "exsudaty_06",
    "exsudaty_07",
    "exsudaty_08"
]


ostatni_spojite = [
    "prumer_papily",
    "vzdalenost_makula_papila"
]


# Všechny spojité příznaky
spojite_sloupce = (
    ma_cols
    + exsudaty_cols
    + ostatni_spojite
)


# Binární příznaky
binarni_sloupce = [
    "am_fm_klasifikace",
    "prescreening",
    "kvalita",
    "target"
]


# Kontrola, jestli všechny požadované sloupce skutečně existují
pozadovane_sloupce = (
    spojite_sloupce
    + binarni_sloupce
)

chybejici_sloupce = [
    col for col in pozadovane_sloupce
    if col not in df.columns
]

if chybejici_sloupce:
    raise ValueError(
        f"V datasetu chybí tyto očekávané sloupce: "
        f"{chybejici_sloupce}"
    )


# ============================================================
# 3. ULOŽENÍ PŮVODNÍCH DAT PRO BOXPLOT
# ============================================================

df_puvodni = df.copy()


# ============================================================
# 4. DETEKCE OUTLIERŮ POMOCÍ 3 * STD
# ============================================================

print("\n" + "=" * 80)
print("2. DETEKCE ODLEHLÝCH HODNOT POMOCÍ 3 × STD")
print("=" * 80)

df_clean = df.copy()

celkem_outlieru = 0


for col in spojite_sloupce:

    # --------------------------------------------------------
    # Výpočet průměru a směrodatné odchylky
    #
    # NaN hodnoty pandas automaticky ignoruje.
    # --------------------------------------------------------

    mean = df_clean[col].mean()
    std = df_clean[col].std()


    # Pokud by byl celý sloupec NaN nebo měl nulové STD
    if pd.isna(std) or std == 0:

        print(
            f"{col:30s}: nelze určit outliery "
            f"(STD = {std})"
        )

        continue


    # --------------------------------------------------------
    # Hranice:
    #
    # mean - 3*std
    # mean + 3*std
    # --------------------------------------------------------

    dolni_hranice = mean - STD_LIMIT * std
    horni_hranice = mean + STD_LIMIT * std


    # --------------------------------------------------------
    # Najdeme konkrétní hodnoty, které jsou mimo hranice
    # --------------------------------------------------------

    mask = (
        (df_clean[col] < dolni_hranice)
        |
        (df_clean[col] > horni_hranice)
    )


    pocet_outlieru = int(mask.sum())

    celkem_outlieru += pocet_outlieru


    print(
        f"{col:30s} | "
        f"outlierů: {pocet_outlieru:3d} | "
        f"hranice: "
        f"[{dolni_hranice:.4f}, "
        f"{horni_hranice:.4f}]"
    )


    # ========================================================
    # DŮLEŽITÉ:
    #
    # NEMAŽEME CELÝ ŘÁDEK.
    #
    # Pouze konkrétní odlehlou hodnotu změníme na NaN.
    # ========================================================

    df_clean.loc[mask, col] = np.nan


print("\nCelkový počet nalezených outlier hodnot:")
print(celkem_outlieru)

print("\nPočet řádků zůstává:")
print(len(df_clean))

print(
    "\nCelkový počet NaN po nahrazení outlierů:"
)

print(
    df_clean[spojite_sloupce]
    .isna()
    .sum()
    .sum()
)


# ============================================================
# 5. ULOŽENÍ STAVU PO ODSTRANĚNÍ OUTLIERŮ
# ============================================================

# Outliery jsou nyní NaN.
# Tuto kopii použijeme později pro boxplot.

df_po_outlierech = df_clean.copy()


# ============================================================
# 6. VYTVOŘENÍ SOUHRNNÝCH PŘÍZNAKŮ
# ============================================================

print("\n" + "=" * 80)
print("3. VYTVOŘENÍ SOUHRNNÝCH PŘÍZNAKŮ")
print("=" * 80)


# ------------------------------------------------------------
# MA_DETEKCE_SOUHRN
#
# Priorita:
#
# ma_detekce_06
#       ↓
# ma_detekce_05
#       ↓
# ma_detekce_04
#       ↓
# ma_detekce_03
#       ↓
# ma_detekce_02
#       ↓
# ma_detekce_01
#
# Pokud například:
#
# 06 = NaN
# 05 = NaN
# 04 = 25
#
# → ma_detekce_souhrn = 25
# ------------------------------------------------------------

ma_priorita = [
    "ma_detekce_01",
    "ma_detekce_02",
    "ma_detekce_03",
    "ma_detekce_04",
    "ma_detekce_05",
    "ma_detekce_06"
]


df_clean["ma_detekce_souhrn"] = (
    df_clean[ma_priorita]
    .bfill(axis=1)
    .iloc[:, 0]
)


# ------------------------------------------------------------
# EXSUDATY_SOUHRN
#
# Priorita:
#
# exsudaty_08
#      ↓
# exsudaty_07
#      ↓
# exsudaty_06
#      ↓
# ...
#      ↓
# exsudaty_01
# ------------------------------------------------------------

exsudaty_priorita = [
    "exsudaty_08",
    "exsudaty_07",
    "exsudaty_06",
    "exsudaty_05",
    "exsudaty_04",
    "exsudaty_03",
    "exsudaty_02",
    "exsudaty_01"
]


df_clean["exsudaty_souhrn"] = (
    df_clean[exsudaty_priorita]
    .bfill(axis=1)
    .iloc[:, 0]
)


print("\nChybějící hodnoty před imputací:")

print(
    "ma_detekce_souhrn:",
    df_clean["ma_detekce_souhrn"].isna().sum()
)

print(
    "exsudaty_souhrn:",
    df_clean["exsudaty_souhrn"].isna().sum()
)


# ============================================================
# 7. ULOŽENÍ SOUHRNNÝCH PŘÍZNAKŮ PRO BOXPLOT
# ============================================================

# Tento stav je ještě PŘED imputací a standardizací.

df_souhrn_pred_imputaci = df_clean[
    [
        "ma_detekce_souhrn",
        "exsudaty_souhrn"
    ]
].copy()


# ============================================================
# 7.1 ANALÝZA EXTRÉMNÍCH HODNOT EXSUDATY_SOUHRN
# ============================================================

print("\n" + "=" * 80)
print("ANALÝZA EXTRÉMNÍCH HODNOT EXSUDATY_SOUHRN")
print("=" * 80)


# ------------------------------------------------------------
# 1. 20 nejvyšších hodnot + jejich TARGET
# ------------------------------------------------------------

print("\n20 nejvyšších hodnot exsudaty_souhrn:")

print(
    df_clean[
        ["exsudaty_souhrn", "target"]
    ]
    .sort_values(
        "exsudaty_souhrn",
        ascending=False
    )
    .head(20)
)


# ------------------------------------------------------------
# 2. Statistiky zvlášť pro target = 0 a target = 1
# ------------------------------------------------------------

print("\nStatistika exsudaty_souhrn podle target:")

print(
    df_clean
    .groupby("target")["exsudaty_souhrn"]
    .describe()
)


# ------------------------------------------------------------
# 3. Kvantily
# ------------------------------------------------------------

print("\nKvantily exsudaty_souhrn:")

print(
    df_clean["exsudaty_souhrn"].quantile(
        [
            0.00,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00
        ]
    )
)


# ------------------------------------------------------------
# 4. Kolik extrémních hodnot patří do jednotlivých tříd
#
# Zde použijeme IQR pouze pro ANALÝZU.
# Nic nemažeme ani neměníme.
# ------------------------------------------------------------

Q1 = df_clean["exsudaty_souhrn"].quantile(0.25)
Q3 = df_clean["exsudaty_souhrn"].quantile(0.75)

IQR = Q3 - Q1

horni_hranice_iqr = Q3 + 1.5 * IQR


extremni_exsudaty = df_clean[
    df_clean["exsudaty_souhrn"] > horni_hranice_iqr
]


print(
    f"\nHorní hranice podle 1.5 × IQR: "
    f"{horni_hranice_iqr:.5f}"
)

print(
    f"Počet hodnot nad touto hranicí: "
    f"{len(extremni_exsudaty)}"
)

print("\nRozdělení těchto hodnot podle target:")

print(
    extremni_exsudaty["target"].value_counts()
)

print("\nProcentuální rozdělení podle target:")

print(
    extremni_exsudaty["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)



# ============================================================
# 7.2 LOGARITMICKÁ TRANSFORMACE EXSUDATY_SOUHRN
# ============================================================

print("\n" + "=" * 80)
print("LOG1P TRANSFORMACE EXSUDATY_SOUHRN")
print("=" * 80)


# ------------------------------------------------------------
# Uložíme si původní hodnoty pro porovnání
# ------------------------------------------------------------

exsudaty_souhrn_pred_log = df_clean["exsudaty_souhrn"].copy()


# ------------------------------------------------------------
# Šikmost před transformací
# ------------------------------------------------------------

skew_pred = df_clean["exsudaty_souhrn"].skew()

print(
    f"\nŠikmost před log1p transformací: "
    f"{skew_pred:.4f}"
)


# ------------------------------------------------------------
# LOG1P TRANSFORMACE
#
# log1p(x) = ln(1 + x)
#
# Je vhodná i pro hodnotu 0:
# log1p(0) = 0
# ------------------------------------------------------------

df_clean["exsudaty_souhrn"] = np.log1p(
    df_clean["exsudaty_souhrn"]
)


# ------------------------------------------------------------
# Šikmost po transformaci
# ------------------------------------------------------------

skew_po = df_clean["exsudaty_souhrn"].skew()

print(
    f"Šikmost po log1p transformaci: "
    f"{skew_po:.4f}"
)


# ------------------------------------------------------------
# Kvantily po transformaci
# ------------------------------------------------------------

print("\nKvantily po log1p transformaci:")

print(
    df_clean["exsudaty_souhrn"].quantile(
        [
            0.00,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00
        ]
    )
)


# ============================================================
# POROVNÁVACÍ BOXPLOT
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6)
)


# Před log1p
axes[0].boxplot(
    exsudaty_souhrn_pred_log.dropna()
)

axes[0].set_title(
    "exsudaty_souhrn před log1p"
)

axes[0].set_ylabel("Hodnota")

axes[0].grid(
    axis="y",
    alpha=0.3
)


# Po log1p
axes[1].boxplot(
    df_clean["exsudaty_souhrn"].dropna()
)

axes[1].set_title(
    "exsudaty_souhrn po log1p"
)

axes[1].set_ylabel("log1p(hodnota)")

axes[1].grid(
    axis="y",
    alpha=0.3
)


fig.suptitle(
    "Vliv log1p transformace na exsudaty_souhrn",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "porovnani_exsudaty_log1p.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ============================================================
# 8. DOPLNĚNÍ CHYBĚJÍCÍCH HODNOT MEDIÁNEM
# ============================================================

print("\n" + "=" * 80)
print("4. DOPLNĚNÍ CHYBĚJÍCÍCH HODNOT MEDIÁNEM")
print("=" * 80)


# Sloupce, které budeme imputovat
sloupce_pro_imputaci = (
    spojite_sloupce
    + [
        "ma_detekce_souhrn",
        "exsudaty_souhrn"
    ]
)


for col in sloupce_pro_imputaci:

    pocet_pred = df_clean[col].isna().sum()

    median = df_clean[col].median()

    if pd.isna(median):

        print(
            f"POZOR: {col} má medián NaN. "
            f"Sloupec pravděpodobně neobsahuje "
            f"žádnou platnou hodnotu."
        )

        continue


    # Doplnění NaN mediánem
    df_clean[col] = df_clean[col].fillna(median)


    pocet_po = df_clean[col].isna().sum()


    print(
        f"{col:30s} | "
        f"NaN před: {pocet_pred:3d} | "
        f"medián: {median:.5f} | "
        f"NaN po: {pocet_po:3d}"
    )


print(
    "\nCelkový počet NaN ve spojitých "
    "a souhrnných příznacích po imputaci:"
)

print(
    df_clean[sloupce_pro_imputaci]
    .isna()
    .sum()
    .sum()
)




# ============================================================
# KORELAČNÍ ANALÝZA NEBINÁRNÍCH DAT
# ============================================================

print("\n" + "=" * 80)
print("KORELAČNÍ ANALÝZA NEBINÁRNÍCH DAT")
print("=" * 80)


# ============================================================
# 1. KORELACE PŘED SLOUČENÍM MA_DETEKCE A EXSUDATY
# ============================================================

print("\n" + "-" * 80)
print("1. KORELACE S TARGET PŘED SLOUČENÍM")
print("-" * 80)


# Původní spojité příznaky
sloupce_pred_sloucenim = [
    "ma_detekce_01",
    "ma_detekce_02",
    "ma_detekce_03",
    "ma_detekce_04",
    "ma_detekce_05",
    "ma_detekce_06",
    "exsudaty_01",
    "exsudaty_02",
    "exsudaty_03",
    "exsudaty_04",
    "exsudaty_05",
    "exsudaty_06",
    "exsudaty_07",
    "exsudaty_08",
    "prumer_papily",
    "vzdalenost_makula_papila"
]


# Přidáme target
data_pred_sloucenim = df_clean[
    sloupce_pred_sloucenim + ["target"]
].copy()


# ------------------------------------------------------------
# Spearmanova korelační matice
# ------------------------------------------------------------

korelace_pred = data_pred_sloucenim.corr(
    method="spearman"
)


# ------------------------------------------------------------
# Pouze korelace jednotlivých příznaků s TARGET
# ------------------------------------------------------------

korelace_target_pred = (
    korelace_pred["target"]
    .drop("target")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print("\nSpearmanova korelace s TARGET před sloučením:")

print(
    korelace_target_pred.round(4)
)


# ============================================================
# HEATMAPA PŘED SLOUČENÍM
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 10)
)


# Pro přehlednost zobrazíme pouze:
#
# příznak × target
#
# místo celé 17×17 matice.

hodnoty_pred = korelace_target_pred.values.reshape(-1, 1)


im = ax.imshow(
    hodnoty_pred,
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    aspect="auto"
)


ax.set_xticks([0])

ax.set_xticklabels(
    ["target"]
)


ax.set_yticks(
    range(len(korelace_target_pred))
)

ax.set_yticklabels(
    korelace_target_pred.index
)


# Hodnoty korelace přímo do grafu
for i, hodnota in enumerate(korelace_target_pred.values):

    ax.text(
        0,
        i,
        f"{hodnota:.3f}",
        ha="center",
        va="center"
    )


ax.set_title(
    "Spearmanova korelace s target před sloučením",
    fontsize=15,
    fontweight="bold"
)


fig.colorbar(
    im,
    ax=ax,
    label="Spearmanova korelace"
)


plt.tight_layout()


plt.savefig(
    "korelace_pred_sloucenim.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 2. KORELACE PO SLOUČENÍ MA_DETEKCE A EXSUDATY
# ============================================================

print("\n" + "-" * 80)
print("2. KORELACE S TARGET PO SLOUČENÍ")
print("-" * 80)


sloupce_po_slouceni = [
    "ma_detekce_souhrn",
    "exsudaty_souhrn",
    "prumer_papily",
    "vzdalenost_makula_papila"
]


data_po_slouceni = df_clean[
    sloupce_po_slouceni + ["target"]
].copy()


# ------------------------------------------------------------
# Spearmanova korelace
# ------------------------------------------------------------

korelace_po = data_po_slouceni.corr(
    method="spearman"
)


# ------------------------------------------------------------
# Korelace jednotlivých příznaků s TARGET
# ------------------------------------------------------------

korelace_target_po = (
    korelace_po["target"]
    .drop("target")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print("\nSpearmanova korelace s TARGET po sloučení:")

print(
    korelace_target_po.round(4)
)


# ============================================================
# CELÁ HEATMAPA PO SLOUČENÍ
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


im = ax.imshow(
    korelace_po.values,
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)


ax.set_xticks(
    range(len(korelace_po.columns))
)

ax.set_yticks(
    range(len(korelace_po.index))
)


ax.set_xticklabels(
    korelace_po.columns,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    korelace_po.index
)


# ------------------------------------------------------------
# Čísla přímo do heatmapy
# ------------------------------------------------------------

for i in range(len(korelace_po.index)):

    for j in range(len(korelace_po.columns)):

        hodnota = korelace_po.iloc[i, j]

        ax.text(
            j,
            i,
            f"{hodnota:.2f}",
            ha="center",
            va="center"
        )


ax.set_title(
    "Spearmanova korelační matice po sloučení příznaků",
    fontsize=15,
    fontweight="bold"
)


fig.colorbar(
    im,
    ax=ax,
    label="Spearmanova korelace"
)


plt.tight_layout()


plt.savefig(
    "korelace_po_slouceni.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 3. PŘÍMÉ POROVNÁNÍ PŘED A PO SLOUČENÍ
# ============================================================

print("\n" + "=" * 80)
print("POROVNÁNÍ KORELACÍ")
print("=" * 80)


print("\nNejsilnější korelace s target PŘED sloučením:")

print(
    korelace_target_pred.round(4)
)


print("\nKorelace s target PO sloučení:")

print(
    korelace_target_po.round(4)
)




# ============================================================
# 9. STANDARDIZACE
# ============================================================

print("\n" + "=" * 80)
print("5. STANDARDIZACE")
print("=" * 80)


# ============================================================
#
#           !!! STANDARDIZACE JE ZDE !!!
#
# Pokud ji chceš vypnout, úplně nahoře nastav:
#
# PROVEST_STANDARDIZACI = False
#
# Není potřeba nic dalšího zakomentovávat.
#
# ============================================================


# Standardizujeme příznaky, které budou použity
# v našem výsledném souhrnném datasetu.
#
# Původní ma_detekce a exsudaty už pro výsledný dataset
# nepotřebujeme, protože byly nahrazeny souhrnnými příznaky.

sloupce_pro_standardizaci = [
    "ma_detekce_souhrn",
    "exsudaty_souhrn",
    "prumer_papily",
    "vzdalenost_makula_papila"
]


if PROVEST_STANDARDIZACI:

    scaler = StandardScaler()

    df_clean[sloupce_pro_standardizaci] = (
        scaler.fit_transform(
            df_clean[sloupce_pro_standardizaci]
        )
    )

    print(
        "\nStandardizace byla PROVEDENA."
    )

    print(
        "Standardizované sloupce:"
    )

    for col in sloupce_pro_standardizaci:
        print(" -", col)


else:

    print(
        "\nStandardizace byla VYPNUTA."
    )


# ============================================================
# 10. ULOŽENÍ STAVU PO STANDARDIZACI
# ============================================================

df_po_standardizaci = df_clean.copy()


# ============================================================
# 11. BOXPLOTY
# ============================================================

print("\n" + "=" * 80)
print("6. VYTVOŘENÍ BOXPLOTŮ")
print("=" * 80)


# ------------------------------------------------------------
# Všechny grafy budou v JEDNOM obrázku.
#
# 1. původní data
# 2. po odstranění outlierů
# 3. souhrnné příznaky
# 4. výsledné příznaky po standardizaci
# ------------------------------------------------------------


fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(20, 26)
)


# ============================================================
# BOXPLOT 1
# PŮVODNÍ DATA
# ============================================================

df_puvodni[spojite_sloupce].boxplot(
    ax=axes[0],
    rot=45
)

axes[0].set_title(
    "1. Původní data",
    fontsize=16,
    fontweight="bold"
)

axes[0].set_ylabel("Hodnota")

axes[0].grid(
    axis="y",
    alpha=0.3
)


# ============================================================
# BOXPLOT 2
# PO NAHRAZENÍ OUTLIERŮ ZA NaN
# ============================================================

df_po_outlierech[spojite_sloupce].boxplot(
    ax=axes[1],
    rot=45
)

axes[1].set_title(
    "2. Data po odstranění odlehlých hodnot (3 × STD)",
    fontsize=16,
    fontweight="bold"
)

axes[1].set_ylabel("Hodnota")

axes[1].grid(
    axis="y",
    alpha=0.3
)


# ============================================================
# BOXPLOT 3
# SOUHRNNÉ PŘÍZNAKY PŘED IMPUTACÍ/STANDARDIZACÍ
# ============================================================

df_souhrn_pred_imputaci[
    [
        "ma_detekce_souhrn",
        "exsudaty_souhrn"
    ]
].boxplot(
    ax=axes[2]
)

axes[2].set_title(
    "3. Souhrnné příznaky před imputací a standardizací",
    fontsize=16,
    fontweight="bold"
)

axes[2].set_ylabel("Hodnota")

axes[2].grid(
    axis="y",
    alpha=0.3
)


# ============================================================
# BOXPLOT 4
# PO IMPUTACI A STANDARDIZACI
# ============================================================

df_po_standardizaci[
    sloupce_pro_standardizaci
].boxplot(
    ax=axes[3],
    rot=25
)


if PROVEST_STANDARDIZACI:

    title = (
        "4. Výsledné příznaky po mediánové imputaci "
        "a standardizaci"
    )

    ylabel = "Standardizovaná hodnota"

else:

    title = (
        "4. Výsledné příznaky po mediánové imputaci "
        "(standardizace vypnuta)"
    )

    ylabel = "Hodnota"


axes[3].set_title(
    title,
    fontsize=16,
    fontweight="bold"
)

axes[3].set_ylabel(ylabel)

axes[3].grid(
    axis="y",
    alpha=0.3
)


# ------------------------------------------------------------
# Nadpis celého obrázku
# ------------------------------------------------------------

fig.suptitle(
    "Porovnání jednotlivých kroků předzpracování dat",
    fontsize=21,
    fontweight="bold",
    y=0.995
)


plt.tight_layout(
    rect=[0, 0, 1, 0.985]
)


# Uložení obrázku
plt.savefig(
    VYSTUP_BOXPLOTY,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    f"\nBoxploty byly uloženy jako: "
    f"{VYSTUP_BOXPLOTY}"
)


# ============================================================
# 12. VYTVOŘENÍ SOUHRNNÉHO CSV
# ============================================================

print("\n" + "=" * 80)
print("7. VYTVOŘENÍ SOUHRNNÉHO CSV")
print("=" * 80)


sloupce_souhrn_csv = [
    "ma_detekce_souhrn",
    "exsudaty_souhrn",
    "prumer_papily",
    "vzdalenost_makula_papila",
    "target"
]


df_souhrn = df_clean[
    sloupce_souhrn_csv
].copy()


df_souhrn.to_csv(
    VYSTUP_SOHRN,
    index=False
)


print(
    f"\nSoubor '{VYSTUP_SOHRN}' byl vytvořen."
)

print(
    "Rozměry souhrnného datasetu:",
    df_souhrn.shape
)


print("\nPrvních 5 řádků:")

print(
    df_souhrn.head()
)


print("\nChybějící hodnoty:")

print(
    df_souhrn.isna().sum()
)


# ============================================================
# 13. VYTVOŘENÍ CSV POUZE S BINÁRNÍMI DATY
# ============================================================

print("\n" + "=" * 80)
print("8. VYTVOŘENÍ BINÁRNÍHO CSV")
print("=" * 80)


df_binarni = df_clean[
    binarni_sloupce
].copy()


# ------------------------------------------------------------
# Kontrola, zda jsou skutečně binární
# ------------------------------------------------------------

print("\nUnikátní hodnoty binárních sloupců:")

for col in binarni_sloupce:

    print(
        f"\n{col}: "
        f"{sorted(df_binarni[col].dropna().unique())}"
    )


df_binarni.to_csv(
    VYSTUP_BINARNI,
    index=False
)


print(
    f"\nSoubor '{VYSTUP_BINARNI}' byl vytvořen."
)

print(
    "Rozměry binárního datasetu:",
    df_binarni.shape
)


# ============================================================
# 14. KORELAČNÍ ANALÝZA BINÁRNÍCH DAT
# ============================================================

print("\n" + "=" * 80)
print("9. KORELAČNÍ ANALÝZA BINÁRNÍCH DAT")
print("=" * 80)


# ------------------------------------------------------------
# Pearsonova korelace
#
# Pro dvě binární proměnné 0/1 odpovídá
# Pearsonova korelace phi koeficientu.
# ------------------------------------------------------------

korelace = df_binarni.corr(
    method="pearson"
)


print("\nCelá korelační matice:")

print(
    korelace.round(3)
)


# ============================================================
# KORELACE S TARGET
# ============================================================

print("\n" + "-" * 50)
print("KORELACE S TARGET")
print("-" * 50)


korelace_target = (
    korelace["target"]
    .drop("target")
    .sort_values(
        key=abs,
        ascending=False
    )
)


print(
    korelace_target.round(4)
)


# ============================================================
# 15. HEATMAPA BINÁRNÍCH PŘÍZNAKŮ
# ============================================================

print("\n" + "=" * 80)
print("10. VYTVOŘENÍ KORELAČNÍ HEATMAPY")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(9, 7)
)


im = ax.imshow(
    korelace.values,
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)


# ------------------------------------------------------------
# Popisky os
# ------------------------------------------------------------

ax.set_xticks(
    range(len(korelace.columns))
)

ax.set_yticks(
    range(len(korelace.index))
)


ax.set_xticklabels(
    korelace.columns,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    korelace.index
)


# ------------------------------------------------------------
# Zápis korelačních koeficientů přímo do polí
# ------------------------------------------------------------

for i in range(len(korelace.index)):

    for j in range(len(korelace.columns)):

        hodnota = korelace.iloc[i, j]

        ax.text(
            j,
            i,
            f"{hodnota:.2f}",
            ha="center",
            va="center"
        )


ax.set_title(
    "Korelační matice binárních příznaků",
    fontsize=16,
    fontweight="bold"
)


fig.colorbar(
    im,
    ax=ax,
    label="Pearsonova korelace"
)


plt.tight_layout()


plt.savefig(
    VYSTUP_HEATMAPA,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    f"\nHeatmapa byla uložena jako: "
    f"{VYSTUP_HEATMAPA}"
)


# ============================================================
# 16. ZÁVĚREČNÁ KONTROLA
# ============================================================

print("\n" + "=" * 80)
print("11. ZÁVĚREČNÁ KONTROLA")
print("=" * 80)


print(
    f"\nPůvodní počet vzorků: "
    f"{len(df)}"
)

print(
    f"Výsledný počet vzorků: "
    f"{len(df_clean)}"
)


print(
    "\nProtože jsme outliery nahrazovali za NaN "
    "a nemažili celé řádky, měl by být počet "
    "vzorků stejný."
)


print(
    "\nPočet NaN v souhrnném datasetu:"
)

print(
    df_souhrn.isna().sum()
)


print(
    "\nPočet případů target:"
)

print(
    df_clean["target"].value_counts()
)


print("\n" + "=" * 80)
print("HOTOVO")
print("=" * 80)


print("\nVytvořené soubory:")

print(
    f"1. {VYSTUP_SOHRN}"
)

print(
    f"2. {VYSTUP_BINARNI}"
)

print(
    f"3. {VYSTUP_BOXPLOTY}"
)

print(
    f"4. {VYSTUP_HEATMAPA}"
)