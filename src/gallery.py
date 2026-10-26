import base64
import io
import os
import pickle
import numpy as np
from PIL import Image
from src.detect import detect_face
from src.embed import embed_batch

GALLERY_CACHE = "results/gallery_cache.pkl"
DEFAULT_IDENTITIES = [
    "George_W_Bush",
    "Colin_Powell",
    "Tony_Blair",
    "Donald_Rumsfeld",
    "Gerhard_Schroeder",
    "Hugo_Chavez",
    "Ariel_Sharon",
    "Serena_Williams",
]


def image_to_base64(img, size=(120, 120)):
    """Resizes and converts PIL image to base64 data URI."""
    thumb = img.copy()
    thumb.thumbnail(size)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class FaceGallery:
    """Manages enrolled identities for 1:N face identification."""

    def __init__(self, cache_file=GALLERY_CACHE, auto_populate=True):
        self.cache_file = cache_file
        self.identities = []  # list of dicts: {'name': str, 'emb': np.ndarray, 'thumb': str}
        self.load()

        if len(self.identities) == 0 and auto_populate:
            self.populate_defaults()

    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.identities = pickle.load(f)
                print(f"Loaded {len(self.identities)} identities from gallery cache.")
            except Exception as e:
                print(f"Error loading gallery cache: {e}")
                self.identities = []

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.identities, f)

    def enroll(self, name, img, save=True):
        """Detects face, computes embedding, and adds to gallery."""
        face, found = detect_face(img)
        emb = embed_batch(face.unsqueeze(0))[0]
        thumb = image_to_base64(img)

        # Replace if name already exists
        for item in self.identities:
            if item["name"].lower() == name.lower():
                item["emb"] = emb
                item["thumb"] = thumb
                if save:
                    self.save()
                return True, "Identity updated in gallery."

        self.identities.append({
            "name": name,
            "emb": emb,
            "thumb": thumb,
        })
        if save:
            self.save()
        return True, "Enrolled successfully into gallery."

    def identify(self, img, top_k=3, threshold=1.08):
        """Matches input image against enrolled identities."""
        if not self.identities:
            return {"found_face": False, "matches": [], "message": "Gallery is empty."}

        face, found = detect_face(img)
        query_emb = embed_batch(face.unsqueeze(0))[0]

        matches = []
        for item in self.identities:
            dist = float(np.linalg.norm(query_emb - item["emb"]))
            # Cosine similarity for normalized embeddings: cos = 1 - (dist^2) / 2
            sim = float(np.dot(query_emb, item["emb"]))
            sim_pct = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100.0))

            matches.append({
                "name": item["name"],
                "distance": round(dist, 3),
                "similarity": round(sim, 3),
                "confidence_pct": round(sim_pct, 1),
                "is_match": dist < threshold,
                "thumb": item.get("thumb", ""),
            })

        matches.sort(key=lambda x: x["distance"])
        top_matches = matches[:top_k]

        return {
            "found_face": found,
            "top_matches": top_matches,
            "best_match": top_matches[0] if top_matches else None,
            "threshold": threshold,
        }

    def populate_defaults(self):
        """Populates gallery with sample known individuals from LFW."""
        img_dir = "data/lfw_home/lfw_funneled"
        if not os.path.exists(img_dir):
            return

        print("Populating default gallery from LFW...")
        for name in DEFAULT_IDENTITIES:
            person_dir = os.path.join(img_dir, name)
            if not os.path.exists(person_dir):
                continue
            images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
            if not images:
                continue
            img_path = os.path.join(person_dir, sorted(images)[0])
            try:
                img = Image.open(img_path).convert("RGB")
                formatted_name = name.replace("_", " ")
                self.enroll(formatted_name, img, save=False)
            except Exception as e:
                print(f"Failed to enroll {name}: {e}")

        self.save()
        print(f"Enrolled {len(self.identities)} default identities.")


# Global instance
gallery = FaceGallery()
