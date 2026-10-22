
# # Prvotní načtení dat
# import pandas as pd
#
# df = pd.read_csv("UIM_projekt/ocni_nalez.csv")
#
# print(df.head())       # Prvních 5 řádků
# print(df.shape)        # Počet řádků a sloupců
# print(df.columns)      # Názvy sloupců
# print(df.dtypes)       # Datové typy sloupců



import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. NASTAVENÍ
# ============================================================

CSV_PATH = Path("UIM_projekt/ocni_nalez.csv")

# Hodnoty, které budeme považovat za chybějící.
MISSING_VALUES = [
    "",
    " ",
    "Nan",
    "nan",
    "NaN",
    "NAN",
    "None",
    "none",
    "NULL",
    "null",
    "NA",
    "N/A",
]

BINARY_COLUMNS = [
    "kvalita",
    "prescreening",
    "am_fm_klasifikace",
    "target",
]

INTEGER_COLUMNS = [
    f"ma_detekce_{i:02d}"
    for i in range(1, 7)
]

# ============================================================
# 2. NAČTENÍ DAT
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    na_values=MISSING_VALUES,
    keep_default_na=True,
    skipinitialspace=True,
)

print("=" * 65)
print("KONTROLA KVALITY DAT")
print("=" * 65)

print(f"Počet řádků: {len(df)}")
print(f"Počet sloupců: {len(df.columns)}")

print("\nNázvy sloupců:")
print(df.columns.tolist())

# ============================================================
# 3. KONTROLA STRUKTURY
# ============================================================

issues = []

if len(df) != 802:
    issues.append({
        "radek": None,
        "sloupec": None,
        "hodnota": None,
        "problem": f"Očekáváno 802 řádků, nalezeno {len(df)}",
    })

if len(df.columns) != 20:
    issues.append({
        "radek": None,
        "sloupec": None,
        "hodnota": None,
        "problem": (
            f"Očekáváno 19 sloupců, nalezeno {len(df.columns)}"
        ),
    })

if df.columns.duplicated().any():
    duplicates = df.columns[df.columns.duplicated()].tolist()
    print("\nDUPLICITNÍ NÁZVY SLOUPCŮ:", duplicates)

expected_columns = [
    "kvalita",
    "prescreening",
    *[f"ma_detekce_{i:02d}" for i in range(1, 7)],
    *[f"exsudaty_{i:02d}" for i in range(1, 9)],
    "vzdalenost_makula_papila",
    "prumer_papily",
    "am_fm_klasifikace",
    "target",
]

missing_columns = sorted(set(expected_columns) - set(df.columns))
extra_columns = sorted(set(df.columns) - set(expected_columns))

if missing_columns:
    print("\nCHYBĚJÍCÍ SLOUPCE:", missing_columns)

if extra_columns:
    print("\nNEOČEKÁVANÉ SLOUPCE:", extra_columns)

# ============================================================
# 4. PŘEVOD SLOUPCŮ NA ČÍSLA A KONTROLA CHYB
# ============================================================

print("\n" + "=" * 65)
print("KONTROLA JEDNOTLIVÝCH HODNOT")
print("=" * 65)

# Kopie převedených dat pro kontrolu.
numeric_df = pd.DataFrame(index=df.index)

for col in df.columns:
    original = df[col]

    # Ošetří i hodnoty obsahující pouze mezery.
    cleaned = original.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.eq(""))

    # Pokus o převod na číslo.
    converted = pd.to_numeric(cleaned, errors="coerce")

    # Chybějící hodnoty.
    missing_mask = cleaned.isna()

    for idx in df.index[missing_mask]:
        issues.append({
            "radek": idx + 2,  # +2 kvůli hlavičce CSV
            "sloupec": col,
            "hodnota": repr(original.loc[idx]),
            "problem": "Chybějící hodnota",
        })

    # Nečíselné hodnoty, které nejsou chybějící.
    invalid_mask = cleaned.notna() & converted.isna()

    for idx in df.index[invalid_mask]:
        issues.append({
            "radek": idx + 2,
            "sloupec": col,
            "hodnota": repr(original.loc[idx]),
            "problem": "Hodnota není číslo",
        })

    numeric_df[col] = converted

# ============================================================
# 5. NEKONEČNÉ HODNOTY
# ============================================================

for col in numeric_df.columns:
    values = numeric_df[col]

    infinite_mask = values.notna() & ~np.isfinite(values)

    for idx in df.index[infinite_mask]:
        issues.append({
            "radek": idx + 2,
            "sloupec": col,
            "hodnota": repr(df.loc[idx, col]),
            "problem": "Nekonečná hodnota (inf nebo -inf)",
        })

# ============================================================
# 6. BINÁRNÍ SLOUPCE: POUZE 0 NEBO 1
# ============================================================

for col in BINARY_COLUMNS:
    if col not in numeric_df.columns:
        continue

    invalid_mask = (
        numeric_df[col].notna()
        & ~numeric_df[col].isin([0, 1])
    )

    for idx in df.index[invalid_mask]:
        issues.append({
            "radek": idx + 2,
            "sloupec": col,
            "hodnota": repr(df.loc[idx, col]),
            "problem": "Binární sloupec musí obsahovat 0 nebo 1",
        })

# ============================================================
# 7. CELÁ ČÍSLA: POČTY MIKROANEURYSMAT
# ============================================================

for col in INTEGER_COLUMNS:
    if col not in numeric_df.columns:
        continue

    values = numeric_df[col]

    # Hodnota musí být celé číslo.
    fractional_mask = (
        values.notna()
        & np.isfinite(values)
        & (values % 1 != 0)
    )

    for idx in df.index[fractional_mask]:
        issues.append({
            "radek": idx + 2,
            "sloupec": col,
            "hodnota": repr(df.loc[idx, col]),
            "problem": "Počet mikroaneuryzmat musí být celé číslo",
        })

# ============================================================
# 8. ZÁPORNÉ HODNOTY
# ============================================================

# Všechny tyto příznaky by podle popisu měly být nezáporné.
nonnegative_columns = [
    *INTEGER_COLUMNS,
    *[f"exsudaty_{i:02d}" for i in range(1, 9)],
    "vzdalenost_makula_papila",
    "prumer_papily",
]

for col in nonnegative_columns:
    if col not in numeric_df.columns:
        continue

    invalid_mask = (
        numeric_df[col].notna()
        & np.isfinite(numeric_df[col])
        & (numeric_df[col] < 0)
    )

    for idx in df.index[invalid_mask]:
        issues.append({
            "radek": idx + 2,
            "sloupec": col,
            "hodnota": repr(df.loc[idx, col]),
            "problem": "Záporná hodnota, očekávána nezáporná",
        })

# ============================================================
# 9. DUPLICITY
# ============================================================

duplicate_mask = df.duplicated(keep=False)

print("\nPočet řádků patřících mezi duplicity:",
      int(duplicate_mask.sum()))

# Uloží všechny řádky, které jsou přesnými duplicitami.
df.loc[duplicate_mask].to_csv(
    "duplicity.csv",
    index=False,
)

# ============================================================
# 10. SOUHRN A EXPORT REPORTU
# ============================================================

issues_df = pd.DataFrame(
    issues,
    columns=["radek", "sloupec", "hodnota", "problem"],
)

issues_df.to_csv(
    "report_chyb.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 65)
print("VÝSLEDEK KONTROLY")
print("=" * 65)

print("Celkový počet nalezených problémů:", len(issues_df))

if issues_df.empty:
    print("Nebyly nalezeny žádné problémy podle nastavených pravidel.")
else:
    print("\nPočet problémů podle typu:")
    print(issues_df["problem"].value_counts().to_string())

    print("\nPočet problémů podle sloupce:")
    print(
        issues_df.groupby("sloupec", dropna=False)
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nPrvních 30 problémů:")
    print(issues_df.head(30).to_string(index=False))

print("\nReport uložen do: report_chyb.csv")
print("Duplicitní řádky uloženy do: duplicity.csv")



# Kontrola


# Přehled chybějících hodnot
missing = df.isna().sum()

report = pd.DataFrame({
    "chybejici_hodnoty": missing,
    "procento_chybejicich": (missing / len(df) * 100).round(2)
})

print(report.sort_values("chybejici_hodnoty", ascending=False))

# Kontrola, kolik řádků má alespoň jednu chybějící hodnotu
print("\nŘádky s alespoň jednou chybějící hodnotou:",
      df.isna().any(axis=1).sum())

# Kontrola chybějících hodnot podle cílové třídy
print("\nChybějící hodnoty podle target:")
print(df.groupby("target").apply(
    lambda group: group.isna().sum().sum()
))




# Počet vzorků v jednotlivých třídách
print("\n\n\nPočet vzorků podle target:")
print(df["target"].value_counts().sort_index())

# Počet řádků s alespoň jednou chybějící hodnotou
print("\nŘádky s chybějícími hodnotami podle target:")
print(
    df.assign(ma_missing=df.isna().any(axis=1))
      .groupby("target")["ma_missing"]
      .agg(["sum", "count", "mean"])
)

# Převod podílu na procenta
missing_by_target = (
    df.assign(ma_missing=df.isna().any(axis=1))
      .groupby("target")["ma_missing"]
      .mean()
      .mul(100)
)

print("\nProcento řádků s chybějícími hodnotami:")
print(missing_by_target.round(2))








# Označení řádků, které mají alespoň jednu chybějící hodnotu
df_analysis = df.copy()
df_analysis["ma_missing"] = df_analysis.isna().any(axis=1)

# 1. Chybějící hodnoty podle kvality snímku
print("\n\n\n\nChybějící hodnoty podle kvality snímku:")
print(
    df_analysis.groupby("kvalita")["ma_missing"]
    .agg(["sum", "count", "mean"])
    .assign(procenta=lambda x: x["mean"] * 100)
)

# 2. Chybějící hodnoty podle kvality a cílové třídy
print("\nChybějící hodnoty podle kvality a target:")
print(
    df_analysis.groupby(["kvalita", "target"])["ma_missing"]
    .agg(["sum", "count", "mean"])
    .assign(procenta=lambda x: x["mean"] * 100)
    .round(2)
)

# 3. Podíl chybějících hodnot v každém příznaku podle kvality
feature_columns = [
    col for col in df.columns
    if col not in ["kvalita", "prescreening",
                   "am_fm_klasifikace", "target"]
]

print("\nChybějící hodnoty podle kvality a příznaku (%):")
print(
    df.groupby("kvalita")[feature_columns]
    .apply(lambda group: group.isna().mean() * 100)
    .round(2)
)






# kolik hodnot chybí v jednotlivých řádcích
import pandas as pd

# Čtyři sloupce, které jsou vždy vyplněné,
# nejsou součástí této analýzy.
feature_columns = [
    col for col in df.columns
    if col not in [
        "kvalita",
        "prescreening",
        "am_fm_klasifikace",
        "target",
    ]
]

# Počet chybějících příznaků v každém řádku
missing_per_row = df[feature_columns].isna().sum(axis=1)

print("\n\n\nPočet chybějících příznaků v jednom řádku:")
print(missing_per_row.value_counts().sort_index())

print("\nPrůměrný počet chybějících příznaků:")
print(missing_per_row.mean())

print("\nŘádky s alespoň 8 chybějícími příznaky:")
print(
    df.loc[missing_per_row >= 8]
      .assign(pocet_chybejicich=missing_per_row[missing_per_row >= 8])
      [["target", "kvalita", "pocet_chybejicich"]]
      .head(30)
)

# Uložení souhrnu
missing_per_row.value_counts().sort_index().rename_axis(
    "pocet_chybejicich_priznaku"
).to_csv("souhrn_chybejicich_priznaku.csv")






feature_columns = [
    col for col in df.columns
    if col not in [
        "target",
        "kvalita",
        "prescreening",
        "am_fm_klasifikace"
    ]
]

print("Podíl chybějících hodnot podle targetu (%):")

print(
    df.groupby("target")[feature_columns]
      .apply(lambda group: group.isna().mean() * 100)
      .round(2)
      .T
)


















