"""Signature verification with a Siamese network (PyTorch).

Idea: one CNN maps a signature image to a 128-d vector (embedding). The SAME network is applied
to two signatures. Trained with contrastive loss so that genuine pairs are close and forged/other
pairs are far apart. At inference: distance < threshold  =>  same person.

Expected dataset layout (CEDAR / GPDS / Kaggle can be re-arranged into this):
    data/signatures/<writer_id>/genuine/*.png
    data/signatures/<writer_id>/forged/*.png
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset

IMG_H, IMG_W = 96, 192


def preprocess_signature(img: np.ndarray) -> np.ndarray:
    """Gray -> Otsu -> crop to ink -> pad to fixed aspect -> resize -> ink=1.0, paper=0.0."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(ink > 0)
    if len(xs) > 0:
        ink = ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = ink.shape
    target_ratio = IMG_W / IMG_H
    if w / h < target_ratio:
        pad = int(h * target_ratio - w)
        ink = cv2.copyMakeBorder(ink, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = int(w / target_ratio - h)
        ink = cv2.copyMakeBorder(ink, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    ink = cv2.resize(ink, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return ink.astype(np.float32) / 255.0


class SiameseNet(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128))
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                nn.Linear(128 * (IMG_H // 16) * (IMG_W // 16), emb_dim))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(self.features(x)), dim=1)  # unit length -> distance in [0, 2]

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.embed(a), self.embed(b)


def contrastive_loss(ea, eb, label, margin: float = 1.0):
    """label=1 -> same person (pull together); label=0 -> different/forged (push apart up to margin)."""
    d = F.pairwise_distance(ea, eb)
    return (label * d.pow(2) + (1 - label) * F.relu(margin - d).pow(2)).mean()


class SignaturePairs(Dataset):
    """Randomly sampled pairs. Positive: genuine-genuine. Negative: genuine-forged (hard) or
    genuine-of-another-writer (easy). Balanced 50/50."""

    def __init__(self, root: str | Path, writers: list[str], n_pairs: int = 4000, seed: int = 0):
        self.rng = random.Random(seed)
        self.items = {}
        for w in writers:
            gen = sorted((Path(root) / w / "genuine").glob("*"))
            forg = sorted((Path(root) / w / "forged").glob("*"))
            if len(gen) >= 2:
                self.items[w] = (gen, forg)
        self.writers = list(self.items)
        self.pairs = [self._make_pair() for _ in range(n_pairs)]

    def _make_pair(self):
        w = self.rng.choice(self.writers)
        gen, forg = self.items[w]
        if self.rng.random() < 0.5:
            a, b = self.rng.sample(gen, 2)
            return a, b, 1.0
        if forg and self.rng.random() < 0.7:
            return self.rng.choice(gen), self.rng.choice(forg), 0.0
        other = self.rng.choice([x for x in self.writers if x != w])
        return self.rng.choice(gen), self.rng.choice(self.items[other][0]), 0.0

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _load(path):
        img = cv2.imread(str(path))
        return torch.from_numpy(preprocess_signature(img)).unsqueeze(0)

    def __getitem__(self, i):
        a, b, y = self.pairs[i]
        return self._load(a), self._load(b), torch.tensor(y, dtype=torch.float32)


def compute_far_frr_eer(distances: np.ndarray, labels: np.ndarray) -> dict:
    """labels: 1 = genuine pair. FAR = forgeries accepted, FRR = genuine rejected.
    EER = error where FAR == FRR (lower is better)."""
    fpr, tpr, thr = roc_curve(labels, -distances)  # higher score = more likely genuine
    frr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - frr)))
    return {"eer": float((fpr[idx] + frr[idx]) / 2), "threshold": float(-thr[idx]),
            "far": float(fpr[idx]), "frr": float(frr[idx])}


class SignatureVerifier:
    def __init__(self, weights_path: str | Path, device: str = "cpu"):
        ckpt = torch.load(weights_path, map_location=device)
        self.model = SiameseNet().to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.threshold = float(ckpt["threshold"])
        self.device = device

    @torch.no_grad()
    def distance(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        ta = torch.from_numpy(preprocess_signature(img_a))[None, None].to(self.device)
        tb = torch.from_numpy(preprocess_signature(img_b))[None, None].to(self.device)
        ea, eb = self.model(ta, tb)
        return float(F.pairwise_distance(ea, eb).item())

    def verify(self, reference: np.ndarray, query: np.ndarray) -> dict:
        d = self.distance(reference, query)
        # risk rises smoothly once distance passes the learned threshold
        risk = float(1 / (1 + np.exp(-8 * (d - self.threshold))))
        return {"distance": d, "threshold": self.threshold, "match": d <= self.threshold, "risk": risk}
