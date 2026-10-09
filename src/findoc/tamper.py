"""Tamper detection with classical computer vision (no training data needed).

1. COPY-MOVE detection (drives the score). ORB keypoints are matched *within the same image*.
   A cloned region produces many matches that all share the SAME displacement vector (dx, dy).
   Repeated letters ("a", "e", "0") also match each other, but with many *different* vectors,
   so we count the biggest cluster of identical vectors instead of raw matches.
   This is the key trick: it separated authentic vs. cloned images with AUC=1.0 on our synthetic set.

2. ERROR LEVEL ANALYSIS (visual aid only). Re-save as JPEG and look at the difference image.
   On text-heavy documents it is NOT reliable (we measured AUC ~0.5), so it is shown to a human
   reviewer but does not drive the score. It is far more useful for photographic tampering (CASIA).

Heuristics are a baseline. For production, train a CNN (e.g. on CASIA v2) and use this as a fallback.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


def error_level_analysis(bgr: np.ndarray, quality: int = 90, scale: float = 15.0) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, recompressed).astype(np.float32) * scale
    return np.clip(diff, 0, 255).astype(np.uint8)


def copy_move_detection(bgr: np.ndarray, ratio: float = 0.6, bin_size: int = 4, min_dist: int = 60,
                        score_start: int = 30, score_span: int = 40) -> dict:
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=25)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < 20:
        return {"score": 0.0, "cluster_size": 0, "pairs": []}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3)
    vectors, pairs = [], []
    for m in matches:
        if len(m) < 3:
            continue
        _self, a, b = m  # m[0] is the keypoint matching itself -> ignore
        if a.distance < ratio * b.distance:
            p1, p2 = np.array(kps[a.queryIdx].pt), np.array(kps[a.trainIdx].pt)
            v = p2 - p1
            if np.hypot(*v) > min_dist:
                if v[0] < 0 or (v[0] == 0 and v[1] < 0):
                    v = -v  # A->B and B->A are the same cloning vector
                vectors.append(tuple(np.round(v / bin_size).astype(int)))
                pairs.append((tuple(map(int, p1)), tuple(map(int, p2))))
    if not vectors:
        return {"score": 0.0, "cluster_size": 0, "pairs": []}
    counts = Counter(vectors)

    def neighbourhood(k):  # tolerate +-1 bin of jitter
        return sum(counts.get((k[0] + dx, k[1] + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    best = max(counts, key=neighbourhood)
    size = neighbourhood(best)
    score = float(np.clip((size - score_start) / score_span, 0.0, 1.0))
    return {"score": score, "cluster_size": int(size), "pairs": pairs[:200]}


def ela_visual(bgr: np.ndarray) -> np.ndarray:
    return error_level_analysis(bgr)


def analyze_tampering(bgr: np.ndarray) -> dict:
    cm = copy_move_detection(bgr)
    return {
        "tamper_score": float(cm["score"]),
        "copy_move_cluster_size": cm["cluster_size"],
        "ela_image": ela_visual(bgr),  # for human review only
    }
