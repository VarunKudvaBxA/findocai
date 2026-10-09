"""Calibrate and evaluate the tamper detector on folders of authentic vs tampered images.
Works with the synthetic set or CASIA v2 (put authentic images in <dir>/authentic, tampered in <dir>/tampered).
Usage: python scripts/evaluate_tamper.py --authentic data/synthetic/photo --tampered data/synthetic/tampered
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from findoc.tamper import analyze_tampering


def scores(folder, limit):
    out = []
    for p in sorted(Path(folder).glob("*"))[:limit]:
        img = cv2.imread(str(p))
        if img is not None:
            out.append(analyze_tampering(img)["tamper_score"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authentic", required=True)
    ap.add_argument("--tampered", required=True)
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    s0, s1 = scores(a.authentic, a.limit), scores(a.tampered, a.limit)
    y = np.r_[np.zeros(len(s0)), np.ones(len(s1))]
    s = np.r_[s0, s1]
    fpr, tpr, thr = roc_curve(y, s)
    best = thr[np.argmax(tpr - fpr)]  # Youden's J
    pred = (s >= best).astype(int)
    print(f"AUC={roc_auc_score(y, s):.3f}  threshold={best:.3f}")
    print(f"precision={precision_score(y, pred):.3f} recall={recall_score(y, pred):.3f} F1={f1_score(y, pred):.3f}")


if __name__ == "__main__":
    main()
