import os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from src.detect import DEVICE, detect_face

_model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def load_model(weights_path=None):
    """Loads either pretrained VGGFace2 or custom fine-tuned weights."""
    global _model
    if weights_path and os.path.exists(weights_path):
        model = InceptionResnetV1(pretrained=None).to(DEVICE)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded custom weights from {weights_path}")
    else:
        model = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    model.eval()
    _model = model
    return _model


def get_model():
    return _model


@torch.no_grad()
def embed_batch(faces):
    """Tensor (N,3,160,160) -> L2-normalized embeddings (N,512)."""
    emb = _model(faces.to(DEVICE)).cpu().numpy()
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def embed_image(img):
    """PIL image -> (512-d embedding, face_found)."""
    face, found = detect_face(img)
    return embed_batch(face.unsqueeze(0))[0], found