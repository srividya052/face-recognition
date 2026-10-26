import numpy as np
from PIL import Image
from src.lfw import LFWPairs

ds = LFWPairs(root="data")
emb = np.load("results/lfw_emb_6000.npy")
n = len(emb)
y = np.array(ds.labels[:n]).astype(bool)
d = np.linalg.norm(emb[:, 0] - emb[:, 1], axis=1)

print("mean distance, same person:      ", round(float(d[y].mean()), 3))
print("mean distance, different people: ", round(float(d[~y].mean()), 3))

genuine = np.where(y)[0]
worst = genuine[np.argsort(-d[genuine])[:8]]
print("8 worst same-person distances:", [round(float(d[i]), 2) for i in worst])

canvas = Image.new("RGB", (500, 8 * 250))
for r, i in enumerate(worst):
    a, b, _ = ds[i]
    canvas.paste(a.resize((250, 250)), (0, r * 250))
    canvas.paste(b.resize((250, 250)), (250, r * 250))
canvas.save("results/worst_genuine_pairs.png")