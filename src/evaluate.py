import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from src.lfw import LFWPairs

from src.detect import detect_face
from src.embed import embed_batch

PAIRS_PER_FOLD = 600  # 300 same + 300 different


def compute_embeddings(ds, n, cache):
    if os.path.exists(cache):
        print("Loading cached embeddings:", cache)
        return np.load(cache)
    emb = np.zeros((n, 2, 512), dtype=np.float32)
    no_face = 0
    for i in range(n):
        img1, img2, _ = ds[i]
        faces = []
        for img in (img1, img2):
            face, found = detect_face(img)
            no_face += not found
            faces.append(face)
        emb[i] = embed_batch(torch.stack(faces))
        if i % 200 == 0:
            print(f"{i}/{n} pairs done")
    print("Images where MTCNN found no face:", no_face)
    np.save(cache, emb)
    return emb


def best_threshold(d, y):
    thrs = np.linspace(0, 2, 401)  # distance range for unit vectors
    accs = [((d < t) == y).mean() for t in thrs]
    return thrs[int(np.argmax(accs))]


def cross_validate(d, y, k):
    fold = np.arange(len(d)) // (len(d) // k)
    accs, fars, frrs, thrs = [], [], [], []
    for f in range(k):
        train, test = fold != f, fold == f
        t = best_threshold(d[train], y[train])  # threshold from training folds ONLY
        pred, yt = d[test] < t, y[test]
        accs.append((pred == yt).mean())
        fars.append(pred[~yt].mean())    # FAR: different people wrongly accepted
        frrs.append((~pred[yt]).mean())  # FRR: same person wrongly rejected
        thrs.append(t)
    return accs, fars, frrs, thrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10, help="use 2 for a quick test")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    ds = LFWPairs(root="data")
    n = args.folds * PAIRS_PER_FOLD
    y = np.array(ds.labels[:n]).astype(bool)  # True = same person

    emb = compute_embeddings(ds, n, f"results/lfw_emb_{n}.npy")
    d = np.linalg.norm(emb[:, 0] - emb[:, 1], axis=1)

    accs, fars, frrs, thrs = cross_validate(d, y, args.folds)
    fpr, tpr, _ = roc_curve(y, -d)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "FAR_mean": float(np.mean(fars)), "FRR_mean": float(np.mean(frrs)),
        "threshold_mean": float(np.mean(thrs)), "AUC": float(roc_auc),
        "TAR@FAR=1%": float(np.interp(0.01, fpr, tpr)),
        "TAR@FAR=0.1%": float(np.interp(0.001, fpr, tpr)),
    }
    print(json.dumps(metrics, indent=2))
    with open("results/metrics_pretrained.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"FaceNet (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Acceptance Rate (TAR)")
    plt.title("ROC curve on LFW")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/roc_pretrained.png", dpi=200)


if __name__ == "__main__":
    main