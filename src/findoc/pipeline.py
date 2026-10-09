"""End-to-end orchestration: image in -> structured JSON + fraud risk score out."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .fields import extract_fields
from .ocr import run_ocr
from .preprocessing import preprocess
from .synthetic import SIGNATURE_ROI
from .tamper import analyze_tampering
from .validation import validate_fields


def noisy_or(risks: list[float]) -> float:
    """Combine independent risk signals: P(fraud) = 1 - prod(1 - r_i). Any strong signal dominates."""
    p = 1.0
    for r in risks:
        p *= 1 - min(max(r, 0.0), 1.0)
    return 1 - p


def crop_roi(img: np.ndarray, roi=SIGNATURE_ROI) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


class DocumentAnalyzer:
    def __init__(self, ocr_engine: str = "tesseract", signature_weights: Optional[str] = None):
        self.ocr_engine = ocr_engine
        self.verifier = None
        if signature_weights and Path(signature_weights).exists():
            from .signature import SignatureVerifier  # torch is only needed if weights exist
            self.verifier = SignatureVerifier(signature_weights)

    def analyze(self, image_bgr: np.ndarray, reference_signature: Optional[np.ndarray] = None) -> dict:
        pre = preprocess(image_bgr)
        ocr = run_ocr(pre["clean"], self.ocr_engine)
        fields = extract_fields(ocr.text)
        issues = validate_fields(fields.to_dict())
        # IMPORTANT: forensic analysis runs on the ORIGINAL pixels. Warping/resizing (used to help OCR)
        # resamples the image and destroys the traces that tamper detection relies on.
        tamper = analyze_tampering(image_bgr)

        signature = {"status": "skipped"}
        if self.verifier is not None and reference_signature is not None:
            signature = {"status": "checked",
                         **self.verifier.verify(reference_signature, crop_roi(pre["color"]))}

        # Business risk: validation issues combined with CV evidence
        validation_risk = noisy_or([i["severity"] for i in issues])
        risks = [validation_risk, tamper["tamper_score"] * 0.8]
        if signature["status"] == "checked":
            risks.append(signature["risk"])
        risk = noisy_or(risks)
        decision = "REJECT/REVIEW" if risk >= 0.7 else "MANUAL_CHECK" if risk >= 0.4 else "PASS"

        return {
            "fields": fields.to_dict(),
            "ocr": {"engine": ocr.engine, "mean_confidence": round(ocr.mean_confidence, 1)},
            "validation_issues": issues,
            "tamper": {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in tamper.items() if k != "ela_image"},
            "signature": signature,
            "fraud_risk_score": round(float(risk), 3),
            "decision": decision,
            "_debug": {"preprocessed": pre, "ela_image": tamper["ela_image"]},
        }


def to_json_safe(result: dict) -> dict:
    """Strip image arrays so the result can be returned by the API."""
    return {k: v for k, v in result.items() if not k.startswith("_")}
