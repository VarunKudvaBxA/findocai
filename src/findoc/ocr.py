"""OCR wrapper. Tesseract by default; EasyOCR optional so you can benchmark two engines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OCRResult:
    text: str
    words: list = field(default_factory=list)  # [{"text", "conf", "box": (x, y, w, h)}]
    engine: str = "tesseract"

    @property
    def mean_confidence(self) -> float:
        confs = [w["conf"] for w in self.words if w["conf"] >= 0]
        return float(np.mean(confs)) if confs else 0.0


def _tesseract(img: np.ndarray) -> OCRResult:
    import pytesseract

    cfg = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(data["text"]):
        if t.strip():
            words.append({
                "text": t, "conf": float(data["conf"][i]),
                "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            })
    text = pytesseract.image_to_string(img, config=cfg)
    return OCRResult(text=text, words=words, engine="tesseract")


_easy_reader = None


def _easyocr(img: np.ndarray) -> OCRResult:
    global _easy_reader
    import easyocr  # optional dependency

    if _easy_reader is None:
        _easy_reader = easyocr.Reader(["en"], gpu=False)
    results = _easy_reader.readtext(img)  # [(box, text, conf), ...]
    results.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))  # rough reading order
    words = [{"text": t, "conf": c * 100, "box": tuple(map(int, b[0]))} for b, t, c in results]
    return OCRResult(text="\n".join(r[1] for r in results), words=words, engine="easyocr")


def run_ocr(img: np.ndarray, engine: str = "tesseract") -> OCRResult:
    if engine == "tesseract":
        return _tesseract(img)
    if engine == "easyocr":
        return _easyocr(img)
    raise ValueError(f"Unknown OCR engine: {engine}")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (used for character error rate)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    return edit_distance(reference, hypothesis) / max(len(reference), 1)
