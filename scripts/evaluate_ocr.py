"""Measure field-level accuracy: raw OCR vs OCR after OpenCV preprocessing.
This produces the 'preprocessing improved accuracy from X% to Y%' table for your README.
Usage: python scripts/evaluate_ocr.py --data data/synthetic --limit 50
"""
import argparse
import json
from pathlib import Path

import cv2

from findoc.fields import extract_fields
from findoc.ocr import character_error_rate, run_ocr
from findoc.preprocessing import preprocess, resize_max_side, to_gray
from findoc.validation import parse_amount_digits, words_to_number

FIELDS = ["date", "ifsc", "account_number", "amount_digits", "amount_words", "payee"]


def norm(field, value):
    if value is None:
        return None
    if field == "amount_digits":
        return parse_amount_digits(value)
    if field == "amount_words":
        return words_to_number(value)
    return value.strip().lower()


def expected(field, gt):
    if field == "amount_digits":
        return float(gt["amount_value"])
    if field == "amount_words":
        return gt["amount_value"]
    return gt[field].strip().lower()


def evaluate(images, truth, mode):
    hits = {f: 0 for f in FIELDS}
    cers = []
    for path in images:
        gt = truth[path.stem]
        img = cv2.imread(str(path))
        ocr_input = to_gray(resize_max_side(img)) if mode == "raw" else preprocess(img)["clean"]
        res = run_ocr(ocr_input)
        fields = extract_fields(res.text).to_dict()
        for f in FIELDS:
            hits[f] += int(norm(f, fields[f]) == expected(f, gt))
        cers.append(character_error_rate(gt["account_number"] + gt["ifsc"],
                                         (fields["account_number"] or "") + (fields["ifsc"] or "")))
    n = len(images)
    return {f: hits[f] / n for f in FIELDS} | {"avg_CER(acct+ifsc)": sum(cers) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--variant", default="photo", choices=["clean", "photo"])
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    truth = json.loads((Path(args.data) / "ground_truth.json").read_text())
    images = sorted((Path(args.data) / args.variant).glob("*"))[: args.limit]
    raw, pre = evaluate(images, truth, "raw"), evaluate(images, truth, "pre")
    print(f"\nField accuracy on {len(images)} '{args.variant}' images\n")
    print("| Field | Raw OCR | With OpenCV preprocessing |\n|---|---|---|")
    for k in raw:
        print(f"| {k} | {raw[k]:.2%} | {pre[k]:.2%} |" if "CER" not in k else f"| {k} | {raw[k]:.3f} | {pre[k]:.3f} |")


if __name__ == "__main__":
    main()
