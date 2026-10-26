import numpy as np
import torch
from facenet_pytorch import MTCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Returns a 160x160 face, already normalized the way FaceNet expects
_mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    selection_method="center_weighted_size",
    device=DEVICE,
)


def _center_crop_fallback(img):
    """If MTCNN finds no face, crop the center of the image instead."""
    img = img.convert("RGB")
    w, h = img.size
    s = int(min(w, h) * 0.6)    
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((160, 160))
    arr = (np.asarray(img, dtype=np.float32) - 127.5) / 128.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def detect_face(img):
    """PIL image -> (face tensor 3x160x160, found: bool)."""
    img = img.convert("RGB")
    face = _mtcnn(img)
    if face is None:
        return _center_crop_fallback(img), False
    return face, True