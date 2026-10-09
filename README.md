# FinDoc-AI — Cheque Processing & Fraud Detection

An end-to-end computer-vision system that reads a photographed/scanned cheque, extracts and **validates** its fields,
and flags **tampering** and **signature forgery**, returning structured JSON and a fraud-risk score.

```
 image ──► OpenCV preprocessing ──► OCR ──► field extraction ──► business-rule validation ─┐
   │        (perspective, deskew,    (Tesseract/   (regex +         (words==digits, IFSC,   │
   │         denoise, CLAHE)         EasyOCR)      OCR-error fixes)  dates, stale/post-dated)├─► risk score
   ├──────────────► copy-move tamper detection (ORB + displacement clustering) ─────────────┤   + decision
   └──────────────► signature verification (Siamese CNN, PyTorch) ──────────────────────────┘
```

## Why this project
Banks lose money to altered cheques and forged signatures, and still process many documents manually.
This project automates extraction and adds layered fraud signals that a human reviewer can inspect.

## Features
- **OpenCV preprocessing:** perspective correction, Hough-based deskew, non-local-means denoising, CLAHE, adaptive threshold.
- **OCR benchmarking:** Tesseract (default) or EasyOCR; scripts measure field accuracy and CER *with vs. without* preprocessing.
- **Validation engine:** amount-in-words (Indian numbering: lakh/crore) must equal amount-in-digits, IFSC format, post-dated / stale (>3 months) cheques, fuzzy correction of OCR typos.
- **Tamper detection:** copy-move forgery via ORB matches + displacement-vector clustering; ELA heat-map for human review.
- **Signature verification:** Siamese CNN with contrastive loss; evaluated with FAR / FRR / EER on **unseen writers**.
- **Serving:** FastAPI (`/analyze`), Streamlit demo, Docker, pytest, GitHub Actions CI.
- **Synthetic data generator:** cheques with ground truth + phone-photo augmentation + tamper simulation (no public cheque dataset exists).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
sudo apt-get install tesseract-ocr          # macOS: brew install tesseract
pip install -r requirements.txt && pip install -e .

python scripts/generate_synthetic.py --n 100 --out data/synthetic
python scripts/evaluate_ocr.py --data data/synthetic --variant photo
python scripts/evaluate_tamper.py --authentic data/synthetic/photo --tampered data/synthetic/tampered

streamlit run app/streamlit_app.py            # demo UI
uvicorn app.api:app --reload                  # API docs at /docs
pytest -q
```

### Train the signature model (optional, needs a dataset)
Arrange CEDAR / GPDS / a Kaggle signature set as `data/signatures/<writer>/genuine|forged/*.png`, then:
```bash
python scripts/train_signature.py --data data/signatures --epochs 15
```
The API/UI automatically enable signature checks when `models/siamese.pt` exists and a reference signature is uploaded.

### Docker
```bash
docker build -t findoc-ai . && docker run -p 8000:8000 findoc-ai
curl -F "document=@data/synthetic/photo/cheque_0000.jpg" localhost:8000/analyze
```

## Results
> **Replace with your own numbers after running the scripts. Label synthetic results as synthetic.**

Field-level OCR accuracy on 30 simulated phone photos (synthetic cheques, Tesseract) — measured with `scripts/evaluate_ocr.py`:

| Field | Raw OCR | + OpenCV preprocessing |
|---|---|---|
| IFSC | 80.0% | 96.7% |
| Account number | 43.3% | 80.0% |
| Amount (digits) | 46.7% | 76.7% |
| Amount (words) | 100% | 100% |
| Avg. CER (account + IFSC) | 0.354 | 0.109 |

Copy-move tamper detector (synthetic clone-stamp edits): AUC 1.0, F1 1.0 — **synthetic edits are easy**; expect lower on real forgeries.
Signature model: _fill in EER / FAR / FRR on held-out writers after training._

## Design decisions worth knowing
1. **Tamper analysis runs on the original pixels**, not the preprocessed image — resampling destroys forensic traces.
2. **Copy-move scoring clusters displacement vectors** instead of counting matches: repeated letters match too, but with scattered vectors; a cloned region produces one dominant vector.
3. **ELA is visual-only.** We measured AUC≈0.5 on text documents, so it does not drive the score.
4. **Risk = noisy-OR** of independent signals: any strong signal can raise the risk, and each is explainable.
5. **Writer-independent evaluation** for signatures (test writers are never seen in training).

## Limitations & future work
- Field extraction is regex-based; replace/augment with a YOLO/LayoutLM field detector for arbitrary layouts.
- Tested mainly on synthetic cheques; real-world validation needed (handwriting, stamps, MICR line).
- Signature region is a fixed ROI; use a detector for real cheques. Printed-text OCR only; handwritten amounts need a handwriting model.
- Tamper heuristics don't catch every forgery type (e.g. ink erasure/overwriting); a CNN trained on CASIA v2 is the next step.
- Not a production fraud system — a decision-support prototype.

## Repo layout
```
src/findoc/   preprocessing, ocr, fields, validation, tamper, signature, synthetic, pipeline
scripts/      data generation, OCR/tamper evaluation, signature training
app/          FastAPI service + Streamlit demo
tests/        pytest suite       .github/workflows/ci.yml
```
