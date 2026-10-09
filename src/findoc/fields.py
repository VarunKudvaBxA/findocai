"""Turn raw OCR text into structured cheque fields using regular expressions.
(For a production system you would add a layout detector such as YOLO; regex is a transparent baseline.)"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

DATE_RE = re.compile(r"\b(\d{2})\s*[/\-.]\s*(\d{2})\s*[/\-.]\s*(\d{4})\b")
IFSC_RE = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
IFSC_LABEL_RE = re.compile(r"IFSC\s*(?:CODE)?\s*[:\-]?\s*([A-Z0-9]{11})", re.I)
ACCOUNT_RE = re.compile(r"(?:A\s*/\s*C|ACCOUNT)\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*(\d{9,18})", re.I)
AMOUNT_DIGITS_RE = re.compile(r"(?:₹|RS\.?/?-?|INR)\s*([\d,]+(?:\.\d{1,2})?)", re.I)
AMOUNT_WORDS_RE = re.compile(r"RUPEES\s+(.+?)\s+ONLY", re.I | re.S)
PAYEE_RE = re.compile(r"PAY\s+(.+?)\s+(?:OR\s+BEARER|OR\s+ORDER)", re.I | re.S)


@dataclass
class ChequeFields:
    date: Optional[str] = None
    payee: Optional[str] = None
    amount_words: Optional[str] = None
    amount_digits: Optional[str] = None
    account_number: Optional[str] = None
    ifsc: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _first(regex: re.Pattern, text: str, group: int = 1) -> Optional[str]:
    m = regex.search(text)
    return m.group(group).strip() if m else None


_TO_LETTER = str.maketrans("0158", "OISB")


def _extract_ifsc(flat: str) -> Optional[str]:
    m = IFSC_LABEL_RE.search(flat)
    if m:
        return normalize_ifsc(m.group(1))
    return _first(IFSC_RE, flat.upper())


def normalize_ifsc(raw: str) -> str:
    """Fix classic OCR confusions using the known IFSC structure:
    4 letters + '0' + 6 chars (almost always digits)  ->  e.g. HDFC0001234."""
    raw = raw.upper()
    bank = raw[:4].translate(_TO_LETTER)          # 0->O, 1->I ...  (must be letters)
    fifth = "0"                                    # always the digit zero
    tail = raw[5:11].replace("O", "0")             # O->0 in the branch code
    return bank + fifth + tail


def extract_fields(text: str) -> ChequeFields:
    flat = re.sub(r"[ \t]+", " ", text)
    date = None
    m = DATE_RE.search(flat)
    if m:
        date = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    amount_words = _first(AMOUNT_WORDS_RE, flat)
    if amount_words:
        amount_words = re.sub(r"\s+", " ", amount_words)
    payee = _first(PAYEE_RE, flat)
    if payee:
        payee = re.sub(r"\s+", " ", payee)
    return ChequeFields(
        date=date,
        payee=payee,
        amount_words=amount_words,
        amount_digits=_first(AMOUNT_DIGITS_RE, flat),
        account_number=_first(ACCOUNT_RE, flat),
        ifsc=_extract_ifsc(flat),
    )
