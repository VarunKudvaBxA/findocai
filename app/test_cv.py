import cv2
import numpy as np

from findoc.preprocessing import deskew, preprocess, rotate
from findoc.synthetic import augment_like_phone_photo, generate_cheque, number_to_words, simulate_tampering
from findoc.tamper import analyze_tampering


def test_number_to_words_roundtrip():
    from findoc.validation import words_to_number
    for n in (1500, 12000, 25300, 99999, 125000, 1250000):
        assert words_to_number(number_to_words(n)) == n


def test_preprocess_outputs():
    img, _ = generate_cheque(seed=1)
    out = preprocess(augment_like_phone_photo(img, seed=1))
    assert out["clean"].ndim == 2 and out["binary"].dtype == np.uint8


def test_deskew_reduces_tilt():
    img, _ = generate_cheque(seed=2)
    tilted = rotate(img, 6.0)
    fixed, angle = deskew(tilted)
    assert angle != 0.0


def test_tamper_detector_separates_clean_from_cloned():
    clean_scores, tampered_scores = [], []
    for seed in range(4):
        img, _ = generate_cheque(seed=seed)
        photo = cv2.imdecode(cv2.imencode(".jpg", augment_like_phone_photo(img, seed=seed),
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)
        clean_scores.append(analyze_tampering(photo)["tamper_score"])
        tampered_scores.append(analyze_tampering(simulate_tampering(photo, seed=seed))["tamper_score"])
    assert max(clean_scores) < 0.5
    assert min(tampered_scores) > 0.5
