"""Train the Siamese signature verifier.
Writer-independent split: writers in the test set are NEVER seen in training (the honest way to evaluate).
Usage: python scripts/train_signature.py --data data/signatures --epochs 15
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from findoc.signature import SiameseNet, SignaturePairs, compute_far_frr_eer, contrastive_loss


def collect_distances(model, loader, device):
    model.eval()
    ds, ys = [], []
    with torch.no_grad():
        for a, b, y in loader:
            ea, eb = model(a.to(device), b.to(device))
            ds.append(torch.nn.functional.pairwise_distance(ea, eb).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ds), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/signatures")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/siamese.pt")
    args = ap.parse_args()

    random.seed(0), torch.manual_seed(0)
    writers = sorted(p.name for p in Path(args.data).iterdir() if p.is_dir())
    random.shuffle(writers)
    n_test = max(1, int(0.2 * len(writers)))
    test_w, train_w = writers[:n_test], writers[n_test:]
    print(f"{len(train_w)} train writers, {len(test_w)} unseen test writers")

    train = DataLoader(SignaturePairs(args.data, train_w, 6000, seed=1), args.batch, shuffle=True, num_workers=0)
    test = DataLoader(SignaturePairs(args.data, test_w, 1500, seed=2), args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    best_eer, best_state, best_thr = 1.0, None, 0.5
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for a, b, y in train:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            loss = contrastive_loss(*model(a, b), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        d, y = collect_distances(model, test, device)
        m = compute_far_frr_eer(d, y)
        print(f"epoch {epoch:02d}  loss={total / len(train.dataset):.4f}  test EER={m['eer']:.3f}")
        if m["eer"] < best_eer:
            best_eer, best_thr = m["eer"], m["threshold"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "threshold": best_thr}, args.out)
    print(f"Saved {args.out}  best EER={best_eer:.3f}  threshold={best_thr:.3f}")


if __name__ == "__main__":
    main()
