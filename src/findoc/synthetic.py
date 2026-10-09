"""Synthetic Indian-style cheques with ground truth (solves 'no public cheque dataset')
plus augmentations that mimic phone photos, and a tamper simulator for testing."""
from __future__ import annotations

import random
from datetime import date, timedelta

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
NAMES = ["Rahul Sharma", "Priya Nair", "Amit Verma", "Sneha Iyer", "Karan Mehta", "Anita Desai",
         "Vikram Singh", "Pooja Reddy", "Arjun Patel", "Meera Joshi"]
BANKS = ["HDFC", "ICIC", "SBIN", "UTIB", "PUNB"]


def _below_thousand(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(f"{ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10] + (f" {ONES[n % 10]}" if n % 10 else ""))
    elif n > 0:
        parts.append(ONES[n])
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """125000 -> 'One Lakh Twenty Five Thousand' (Indian system)."""
    if n == 0:
        return "Zero"
    parts = []
    for value, name in ((10**7, "crore"), (10**5, "lakh"), (10**3, "thousand")):
        if n >= value:
            parts.append(f"{_below_thousand(n // value)} {name}")
            n %= value
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts).title()


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def draw_signature(draw: ImageDraw.ImageDraw, box, rng: random.Random):
    x0, y0, x1, y1 = box
    pts = [(x0 + (x1 - x0) * i / 9, (y0 + y1) / 2 + rng.uniform(-0.4, 0.4) * (y1 - y0)) for i in range(10)]
    draw.line(pts, fill=(20, 20, 90), width=3, joint="curve")
    draw.line([(x0, y1 - 5), (x1, y0 + 8)], fill=(20, 20, 90), width=2)


def generate_cheque(seed: int = 0, today: date | None = None) -> tuple[np.ndarray, dict]:
    rng = random.Random(seed)
    today = today or date.today()
    W, H = 1200, 520
    img = Image.new("RGB", (W, H), (235, 245, 252))
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, W - 6, H - 6], outline=(40, 60, 120), width=3)

    amount = rng.choice([1500, 12000, 25300, 45000, 99999, 125000, 250000, 1250000])
    cheque_date = today - timedelta(days=rng.randint(1, 60))
    gt = {
        "date": cheque_date.strftime("%d/%m/%Y"),
        "payee": rng.choice(NAMES),
        "amount_value": amount,
        "amount_words": number_to_words(amount),
        "amount_digits": f"{amount:,}",
        "account_number": "".join(str(rng.randint(0, 9)) for _ in range(12)),
        "ifsc": f"{rng.choice(BANKS)}0{rng.randint(0, 999999):06d}",
    }
    d.text((40, 25), f"{gt['ifsc'][:4]} BANK LTD.", fill=(30, 50, 110), font=_font(34))
    d.text((820, 30), f"Date: {gt['date']}", fill=(0, 0, 0), font=_font(30))
    d.text((40, 120), f"Pay {gt['payee']} or bearer", fill=(0, 0, 0), font=_font(32))
    d.text((40, 200), f"Rupees {gt['amount_words']} only", fill=(0, 0, 0), font=_font(32))
    d.rectangle([800, 255, 1150, 325], outline=(0, 0, 0), width=2)
    d.text((815, 270), f"Rs. {gt['amount_digits']}/-", fill=(0, 0, 0), font=_font(34))
    d.text((40, 300), f"A/C No. {gt['account_number']}", fill=(0, 0, 0), font=_font(28))
    d.text((40, 350), f"IFSC: {gt['ifsc']}", fill=(0, 0, 0), font=_font(28))
    draw_signature(d, (780, 400, 1130, 480), rng)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), gt


# Signature region as fractions of the cheque (x0, y0, x1, y1) -> used by the pipeline
SIGNATURE_ROI = (0.62, 0.74, 0.98, 0.95)


def augment_like_phone_photo(img: np.ndarray, seed: int = 0, strength: float = 1.0) -> np.ndarray:
    """Rotation, perspective warp, blur, uneven lighting and sensor noise."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    canvas = np.full((h + 160, w + 160, 3), rng.integers(90, 140), np.uint8)  # desk background
    canvas[80:80 + h, 80:80 + w] = img
    H, W = canvas.shape[:2]
    j = 25 * strength
    src = np.float32([[80, 80], [80 + w, 80], [80 + w, 80 + h], [80, 80 + h]])
    dst = src + rng.uniform(-j, j, src.shape).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(canvas, m, (W, H), borderValue=(110, 110, 110))
    ang = rng.uniform(-5, 5) * strength
    out = cv2.warpAffine(out, cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0), (W, H),
                         borderValue=(110, 110, 110))
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.6) * strength)
    gradient = np.tile(np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.9, 1.1), W), (H, 1))[..., None]
    out = np.clip(out.astype(np.float32) * gradient, 0, 255)
    out += rng.normal(0, 8 * strength, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def simulate_tampering(img: np.ndarray, seed: int = 0, jpeg_quality: int = 85) -> np.ndarray:
    """Mimic a 'clone-stamp' edit on an uploaded image: copy a text-dense patch over another
    text-dense area (like duplicating digits to change an amount), then re-save as JPEG."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = out.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 150)
    ph, pw = int(h * 0.12), int(w * 0.18)
    cands = []
    for _ in range(200):
        y, x = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
        cands.append((edges[y:y + ph, x:x + pw].mean(), y, x))
    cands.sort(reverse=True)
    _, y1, x1 = cands[0]
    y2, x2 = cands[1][1:]
    for _, y2, x2 in cands[1:]:
        if abs(y2 - y1) > ph or abs(x2 - x1) > pw:  # destination must not overlap the source
            break
    out[y2:y2 + ph, x2:x2 + pw] = out[y1:y1 + ph, x1:x1 + pw]
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)
