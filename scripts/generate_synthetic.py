"""Generate synthetic cheques (clean, phone-photo-style, and tampered) + ground truth JSON.
Usage: python scripts/generate_synthetic.py --n 100 --out data/synthetic
"""
import argparse
import json
from pathlib import Path

import cv2

from findoc.synthetic import augment_like_phone_photo, generate_cheque, simulate_tampering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("clean", "photo", "tampered"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    truth = {}
    for i in range(args.n):
        img, gt = generate_cheque(seed=i)
        name = f"cheque_{i:04d}"
        truth[name] = gt
        cv2.imwrite(str(out / "clean" / f"{name}.png"), img)
        photo = augment_like_phone_photo(img, seed=i)
        cv2.imwrite(str(out / "photo" / f"{name}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 85])
        photo_jpeg = cv2.imread(str(out / "photo" / f"{name}.jpg"))  # what a real upload looks like
        cv2.imwrite(str(out / "tampered" / f"{name}.jpg"), simulate_tampering(photo_jpeg, seed=i),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {args.n} cheques x 3 variants to {out}")


if __name__ == "__main__":
    main()
