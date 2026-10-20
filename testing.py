"""Nezávislé testování odevzdaného řešení.

Doplňte funkce nacti_model(), predzpracuj() a predikuj(). Zbytek skriptu neměňte.

Spuštění:
    python testing.py                    # použije cestu z DATA_PATH
    python testing.py cesta/k/test.csv   # nebo cestu z příkazové řádky
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import matice_zamen, mcc

# Adresář odevzdaného řešení; cesty skládejte relativně k němu.
SLOZKA = Path(__file__).resolve().parent

# Cesta k testovacímu souboru. Při testování ji vyučující přepíše.
DATA_PATH = SLOZKA / "data" / "train.csv"


def nacti_model():
    """Načte uložený natrénovaný model (a naučené předzpracování) ze složky models/."""
    raise NotImplementedError("Doplňte načtení modelu.")


def predzpracuj(x: pd.DataFrame):
    """Připraví příznaky pro model.

    Používá jen parametry naučené na trénovacích datech; počet a pořadí řádků se nemění.
    """
    raise NotImplementedError("Doplňte předzpracování.")


def predikuj(model, x) -> np.ndarray:
    """Vrátí predikce 0/1 pro každý řádek x."""
    raise NotImplementedError("Doplňte predikci.")


def main() -> None:
    """Načte testovací data, provede predikci modelem a vypíše matici záměn a MCC."""
    cesta = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    data = pd.read_csv(cesta)
    y = data["target"].to_numpy(dtype=int)
    x = data.drop(columns=["target"])  # target nesmí vstupovat do predikce

    model = nacti_model()
    x = predzpracuj(x)
    p = np.asarray(predikuj(model, x)).astype(int).ravel()

    if len(p) != len(y):
        raise ValueError(f"Počet predikcí ({len(p)}) neodpovídá počtu řádků ({len(y)}).")
    if not np.isin(p, [0, 1]).all():
        raise ValueError("Predikce musí obsahovat pouze hodnoty 0 a 1.")

    pd.DataFrame({"prediction": p}).to_csv(SLOZKA / "predikce.csv", index=False)

    tp, tn, fp, fn = matice_zamen(y, p)
    print(f"Testovací soubor: {cesta} ({len(y)} vzorků)")
    print("Matice záměn (řádky = skutečnost, sloupce = predikce):")
    print("            pred 0  pred 1")
    print(f"  true 0  {tn:>7} {fp:>7}")
    print(f"  true 1  {fn:>7} {tp:>7}")
    print(f"MCC: {mcc(tp, tn, fp, fn):.4f}")


if __name__ == "__main__":
    main()
