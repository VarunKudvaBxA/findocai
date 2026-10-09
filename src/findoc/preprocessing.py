"""OpenCV preprocessing: turns a messy phone photo/scan into a clean image for OCR.

Order matters:  resize -> perspective fix -> grayscale -> deskew -> denoise -> contrast -> binarize
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def resize_max_side(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Down/up-scale so the longest side is `max_side` (OCR works best at a consistent scale)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if abs(scale - 1.0) < 0.05:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def correct_perspective(img: np.ndarray, min_area_ratio: float = 0.25) -> np.ndarray:
    """Find the document outline (largest 4-corner contour) and flatten it.
    If no clear document is found, return the image unchanged (safe fallback)."""
    gray = to_gray(img)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area_ratio * img_area:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return four_point_transform(img, approx.reshape(4, 2))
    return img


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Median angle (degrees) of long near-horizontal lines (cheque borders, text baselines)."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=max(gray.shape[1] // 4, 50), maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 20:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)


def deskew(img: np.ndarray, max_angle: float = 15.0) -> tuple[np.ndarray, float]:
    angle = estimate_skew_angle(to_gray(img))
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return img, 0.0
    return rotate(img, angle), angle


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE = local contrast boost; better than global equalisation for uneven lighting."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold: each pixel compared with its local neighbourhood (handles shadows)."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )


def preprocess(img_bgr: np.ndarray, fix_perspective: bool = True) -> dict:
    """Full pipeline. Returns every intermediate so the UI/notebooks can display them."""
    img = resize_max_side(img_bgr)
    if fix_perspective:
        img = correct_perspective(img)
        img = resize_max_side(img)
    img, angle = deskew(img)
    gray = to_gray(img)
    clean = enhance_contrast(denoise(gray))
    return {
        "color": img,            # for tamper analysis (keeps colour + original noise pattern)
        "gray": gray,
        "clean": clean,          # best input for OCR
        "binary": binarize(clean),
        "skew_angle": angle,
    }
