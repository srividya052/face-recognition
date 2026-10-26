import os
from PIL import Image
from sklearn.datasets._lfw import _check_fetch_lfw


class LFWPairs:
    """Official LFW 10-fold pairs, downloaded through scikit-learn's mirror."""

    def __init__(self, root="data"):
        _, folder = _check_fetch_lfw(data_home=root, funneled=True,
                                     download_if_missing=True)
        lfw_home = os.path.dirname(folder)
        self.folder = folder
        self.items, self.labels = [], []
        with open(os.path.join(lfw_home, "pairs.txt")) as f:
            next(f)  # header line: "10 300"
            for line in f:
                p = line.split()
                if len(p) == 3:    # same person
                    a, b, lab = (p[0], int(p[1])), (p[0], int(p[2])), 1
                elif len(p) == 4:  # different people
                    a, b, lab = (p[0], int(p[1])), (p[2], int(p[3])), 0
                else:
                    continue
                self.items.append((a, b))
                self.labels.append(lab)

    def _load(self, name, idx):
        path = os.path.join(self.folder, name, f"{name}_{idx:04d}.jpg")
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, b = self.items[i]
        return self._load(*a), self._load(*b), self.labels[i]