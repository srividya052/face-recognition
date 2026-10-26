import argparse
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset

from src.detect import DEVICE, detect_face
from src.lfw import LFWPairs

PAIRS_PER_FOLD = 600
TRAIN_FOLDS = range(0, 7)   # 4200 pairs: the model learns from these
VAL_FOLD = [7]              # 600 pairs: pick best epoch + threshold
TEST_FOLDS = range(8, 10)   # 1200 pairs: used once, for the final report
CACHE = "results/cache_lfw_faces.pt"
TRAINABLE = ["block8", "avgpool_1a", "last_linear", "last_bn"]


def fold_idx(folds):
    return np.concatenate(
        [np.arange(f * PAIRS_PER_FOLD, (f + 1) * PAIRS_PER_FOLD) for f in folds]
    )


def load_faces():
    """Detect + crop all 6000 pairs once (float16 to save RAM), then cache."""
    if os.path.exists(CACHE):
        print("Loading cached faces:", CACHE)
        d = torch.load(CACHE, map_location="cpu")
        return d["f1"], d["f2"], d["y"].numpy().astype(bool)

    ds = LFWPairs(root="data")
    n = 10 * PAIRS_PER_FOLD
    f1 = torch.zeros(n, 3, 160, 160, dtype=torch.float16)
    f2 = torch.zeros(n, 3, 160, 160, dtype=torch.float16)
    for i in range(n):
        a, b, _ = ds[i]
        f1[i] = detect_face(a)[0].half()
        f2[i] = detect_face(b)[0].half()
        if i % 200 == 0:
            print(f"Detecting faces: {i}/{n} pairs")
    y = torch.tensor(ds.labels[:n])
    os.makedirs("results", exist_ok=True)
    torch.save({"f1": f1, "f2": f2, "y": y}, CACHE)
    return f1, f2, y.numpy().astype(bool)


class PairDataset(Dataset):
    def __init__(self, f1, f2, y, idx, augment=False):
        self.f1, self.f2, self.y, self.idx, self.augment = f1, f2, y, idx, augment

    def __len__(self):
        return len(self.idx)

    def _prep(self, t):
        t = t.float()
        if self.augment and torch.rand(1).item() > 0.5:
            t = torch.flip(t, dims=[2])  # horizontal flip
        return t

    def __getitem__(self, i):
        j = int(self.idx[i])
        return self._prep(self.f1[j]), self._prep(self.f2[j]), float(self.y[j])


class ContrastiveLoss(nn.Module):
    """L = y*D^2 + (1-y)*max(0, margin-D)^2   (y=1 same person)"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, target):
        dist = torch.norm(e1 - e2, p=2, dim=1)
        same = target * dist.pow(2)
        diff = (1 - target) * torch.clamp(self.margin - dist, min=0).pow(2)
        return (same + diff).mean()


@torch.no_grad()
def get_distances(model, f1, f2, idx, bs=64):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = torch.from_numpy(idx[i:i + bs])
        e1 = F.normalize(model(f1[j].float().to(DEVICE)), p=2, dim=1)
        e2 = F.normalize(model(f2[j].float().to(DEVICE)), p=2, dim=1)
        out.append(torch.norm(e1 - e2, p=2, dim=1).cpu().numpy())
    return np.concatenate(out)


def best_threshold(d, y):
    thrs = np.linspace(0, 2, 401)
    accs = [((d < t) == y).mean() for t in thrs]
    return thrs[int(np.argmax(accs))], float(max(accs))


def report(d, y, thr):
    pred = d < thr
    fpr, tpr, _ = roc_curve(y, -d)
    return {
        "accuracy": float((pred == y).mean()),
        "FAR": float(pred[~y].mean()),
        "FRR": float((~pred[y]).mean()),
        "threshold": float(thr),
        "AUC": float(auc(fpr, tpr)),
        "TAR@FAR=1%": float(np.interp(0.01, fpr, tpr)),
    }


def set_train_mode(model, unfreeze_all):
    """Only the layers being fine-tuned run in train mode (BatchNorm fix)."""
    if unfreeze_all:
        model.train()
        return
    model.eval()
    for name, m in model.named_modules():
        if any(name.startswith(t) for t in TRAINABLE):
            m.train()


def train(args):
    print(f"Device: {DEVICE} | epochs {args.epochs} | batch {args.batch_size} | lr {args.lr}")
    f1, f2, y = load_faces()
    tr_idx, va_idx, te_idx = fold_idx(TRAIN_FOLDS), fold_idx(VAL_FOLD), fold_idx(TEST_FOLDS)
    print(f"train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)} pairs")

    loader = DataLoader(PairDataset(f1, f2, y, tr_idx, augment=True),
                        batch_size=args.batch_size, shuffle=True)

    model = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    if args.unfreeze_all:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for n, p in model.named_parameters():
            p.requires_grad = any(n.startswith(t) for t in TRAINABLE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train:,}")

    # ---- Pretrained baseline (same protocol as the fine-tuned model) ----
    d_val0 = get_distances(model, f1, f2, va_idx)
    d_te0 = get_distances(model, f1, f2, te_idx)
    thr0, val0 = best_threshold(d_val0, y[va_idx])
    print(f"Pretrained | val acc {val0*100:.2f}%")

    criterion = ContrastiveLoss(args.margin)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val, best_state, best_epoch = val0, copy.deepcopy(model.state_dict()), 0
    losses, val_accs = [], []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        set_train_mode(model, args.unfreeze_all)
        total = 0.0
        for a, b, lab in loader:
            a, b, lab = a.to(DEVICE), b.to(DEVICE), lab.float().to(DEVICE)
            opt.zero_grad()
            loss = criterion(F.normalize(model(a), p=2, dim=1),
                             F.normalize(model(b), p=2, dim=1), lab)
            loss.backward()
            opt.step()
            total += loss.item() * len(lab)
        sched.step()
        losses.append(total / len(tr_idx))

        d_val = get_distances(model, f1, f2, va_idx)
        _, val_acc = best_threshold(d_val, y[va_idx])
        val_accs.append(val_acc)
        print(f"Epoch {ep:02d}/{args.epochs} | loss {losses[-1]:.4f} | val acc {val_acc*100:.2f}%")
        if val_acc > best_val:  # strictly better than anything so far
            best_val, best_epoch = val_acc, ep
            best_state = copy.deepcopy(model.state_dict())
    print(f"Done in {time.time()-t0:.0f}s | best epoch: {best_epoch} (0 = pretrained was best)")

    # ---- Final test: folds 9-10, threshold chosen on the validation fold ----
    model.load_state_dict(best_state)
    torch.save(best_state, args.save_path)
    d_val1 = get_distances(model, f1, f2, va_idx)
    d_te1 = get_distances(model, f1, f2, te_idx)
    thr1, _ = best_threshold(d_val1, y[va_idx])

    yt = y[te_idx]
    res = {"pretrained": report(d_te0, yt, thr0),
           "finetuned": report(d_te1, yt, thr1),
           "best_epoch": best_epoch, "test_pairs": int(len(te_idx))}
    print(json.dumps(res, indent=2))
    with open("results/metrics_finetune_comparison.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---- Plots ----
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), losses, "b-o")
    plt.xlabel("Epoch"); plt.ylabel("Contrastive loss"); plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), [a * 100 for a in val_accs], "g-s", label="Fine-tuned")
    plt.axhline(val0 * 100, color="r", ls="--", label="Pretrained")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation fold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=200)

    plt.figure(figsize=(5, 5))
    for d, lab in [(d_te0, "Pretrained"), (d_te1, "Fine-tuned")]:
        fpr, tpr, _ = roc_curve(yt, -d)
        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Acceptance Rate (FAR)"); plt.ylabel("True Acceptance Rate (TAR)")
    plt.title("ROC on held-out LFW folds 9-10"); plt.legend(); plt.tight_layout()
    plt.savefig("results/roc_pretrained_vs_finetuned.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--unfreeze_all", action="store_true")
    ap.add_argument("--save_path", default="results/model_finetuned.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()