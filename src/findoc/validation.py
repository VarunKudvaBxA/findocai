"""Business-rule validation of extracted cheque fields.
These checks catch both OCR mistakes and genuine tampering (e.g. amount in words != digits)."""
from __future__ import annotations

import difflib
import re
from datetime import date, datetime, timedelta
from typing import Optional

UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"thousand": 10**3, "lakh": 10**5, "lakhs": 10**5, "crore": 10**7, "crores": 10**7}
FILLER = {"and", "rupees", "rupee", "only"}
VOCAB = list(UNITS) + list(TENS) + list(SCALES) + ["hundred"] + list(FILLER)


def _fix_token(tok: str) -> Optional[str]:
    """Fuzzy-correct OCR typos ('thausand' -> 'thousand'). Returns None if hopeless."""
    if tok in VOCAB:
        return tok
    match = difflib.get_close_matches(tok, VOCAB, n=1, cutoff=0.8)
    return match[0] if match else None


def words_to_number(text: str) -> Optional[int]:
    """'One Lakh Twenty Five Thousand' -> 125000 (Indian numbering system)."""
    total, current = 0, 0
    for raw in re.findall(r"[a-z]+", text.lower()):
        tok = _fix_token(raw)
        if tok is None:
            return None
        if tok in UNITS:
            current += UNITS[tok]
        elif tok in TENS:
            current += TENS[tok]
        elif tok == "hundred":
            current = max(current, 1) * 100
        elif tok in SCALES:
            total += max(current, 1) * SCALES[tok]
            current = 0
    return total + current


def parse_amount_digits(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def validate_ifsc(ifsc: Optional[str]) -> bool:
    return bool(ifsc and re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", ifsc))


def parse_date(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%d/%m/%Y").date()
    except ValueError:
        return None


def validate_fields(fields: dict, today: Optional[date] = None) -> list[dict]:
    """Return a list of issues: {"code", "severity" (0-1 risk), "message"}."""
    today = today or date.today()
    issues: list[dict] = []

    def add(code: str, severity: float, message: str) -> None:
        issues.append({"code": code, "severity": severity, "message": message})

    # 1. Required fields present
    for key in ("date", "amount_words", "amount_digits", "ifsc", "account_number"):
        if not fields.get(key):
            add(f"MISSING_{key.upper()}", 0.15, f"Could not read field '{key}'")

    # 2. Amount in words must equal amount in digits  (strongest tamper signal)
    words_val = words_to_number(fields["amount_words"]) if fields.get("amount_words") else None
    digits_val = parse_amount_digits(fields.get("amount_digits"))
    if fields.get("amount_words") and words_val is None:
        add("AMOUNT_WORDS_UNREADABLE", 0.2, "Amount in words could not be parsed")
    if words_val is not None and digits_val is not None and abs(words_val - digits_val) > 0.99:
        add("AMOUNT_MISMATCH", 0.7, f"Words={words_val} but digits={digits_val}")

    # 3. IFSC format
    if fields.get("ifsc") and not validate_ifsc(fields["ifsc"]):
        add("BAD_IFSC", 0.3, "IFSC format invalid")

    # 4. Date sanity: not post-dated, not stale (cheques are valid for 3 months in India)
    d = parse_date(fields.get("date"))
    if fields.get("date") and d is None:
        add("BAD_DATE", 0.3, "Date could not be parsed")
    elif d:
        if d > today:
            add("POST_DATED", 0.25, "Cheque is post-dated")
        elif d < today - timedelta(days=92):
            add("STALE_CHEQUE", 0.25, "Cheque older than 3 months")
    return issues
