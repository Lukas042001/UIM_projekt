"""Metriky pro hodnocení binární klasifikace."""
import numpy as np


def matice_zamen(y: np.ndarray, p: np.ndarray) -> tuple[int, int, int, int]:
    """Vrátí prvky matice záměn (TP, TN, FP, FN) pro skutečné třídy y a predikce p."""
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    return tp, tn, fp, fn


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Vypočítá Matthewsův korelační koeficient; při nulovém jmenovateli vrací 0."""
    jmenovatel = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / jmenovatel if jmenovatel > 0 else 0.0
