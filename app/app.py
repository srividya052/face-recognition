import base64
import io
import json
import os
import sys
import time
from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np
from PIL import Image
import torch

# Ensure repository root is in sys.path whether executed as `python app/app.py` or `python -m app.app`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.detect import detect_face
from src.embed import embed_batch
from src.gallery import gallery, image_to_base64

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def parse_image_upload(file_storage):
    """Safely opens an uploaded image file into PIL Image."""
    if not file_storage or file_storage.filename == "":
        return None
    try:
        return Image.open(file_storage.stream).convert("RGB")
    except Exception as e:
        print(f"Error parsing image: {e}")
        return None


def face_tensor_to_base64(face_tensor):
    """Converts 3x160x160 normalized tensor back to base64 image preview."""
    arr = face_tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/verify", methods=["POST"])
def verify_faces():
    """1:1 Verification: Compares two uploaded images."""
    t0 = time.time()
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Both image1 and image2 are required."}), 400

    img1 = parse_image_upload(request.files["image1"])
    img2 = parse_image_upload(request.files["image2"])

    if img1 is None or img2 is None:
        return jsonify({"error": "Failed to decode one or both images."}), 400

    face1, found1 = detect_face(img1)
    face2, found2 = detect_face(img2)

    emb1 = embed_batch(face1.unsqueeze(0))[0]
    emb2 = embed_batch(face2.unsqueeze(0))[0]

    # Euclidean distance
    dist = float(np.linalg.norm(emb1 - emb2))
    # Cosine similarity
    cos_sim = float(np.dot(emb1, emb2))
    threshold = 1.08

    is_match = dist < threshold
    # Confidence percentage (0% to 100%)
    confidence = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100.0))

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return jsonify({
        "is_match": bool(is_match),
        "distance": round(dist, 4),
        "cosine_similarity": round(cos_sim, 4),
        "confidence_pct": round(confidence, 1),
        "threshold": threshold,
        "face1_detected": bool(found1),
        "face2_detected": bool(found2),
        "face1_crop": face_tensor_to_base64(face1),
        "face2_crop": face_tensor_to_base64(face2),
        "latency_ms": elapsed_ms,
    })


@app.route("/api/identify", methods=["POST"])
def identify_face():
    """1:N Identification: Matches uploaded query image against gallery."""
    t0 = time.time()
    if "image" not in request.files:
        return jsonify({"error": "Image file is required."}), 400

    img = parse_image_upload(request.files["image"])
    if img is None:
        return jsonify({"error": "Failed to decode image."}), 400

    face, found = detect_face(img)
    if not found:
        return jsonify({
            "found_face": False,
            "message": "No face detected in the image.",
            "top_matches": [],
            "best_match": None,
            "threshold": 1.08,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })

    res = gallery.identify(img, top_k=5, threshold=1.08)
    res["query_crop"] = face_tensor_to_base64(face)
    res["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(res)


@app.route("/api/enroll", methods=["POST"])
def enroll_face():
    """Enrolls a new person into the gallery."""
    name = request.form.get("name", "").strip()
    if not name or "image" not in request.files:
        return jsonify({"error": "Name and image are required."}), 400

    img = parse_image_upload(request.files["image"])
    if img is None:
        return jsonify({"error": "Invalid image file."}), 400

    success, msg = gallery.enroll(name, img)
    return jsonify({"success": success, "message": msg, "total_enrolled": len(gallery.identities)})


@app.route("/api/gallery", methods=["GET"])
def get_gallery():
    """Lists currently enrolled gallery members."""
    items = []
    for item in gallery.identities:
        items.append({
            "name": item["name"],
            "thumb": item.get("thumb", ""),
        })
    return jsonify({"identities": items, "count": len(items)})


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns evaluation metrics and ablation study data."""
    cur_metrics_path = os.path.join(RESULTS_DIR, "metrics_pretrained.json")
    v1_metrics_path = os.path.join(RESULTS_DIR, "v1_most_confident_face", "metrics_pretrained.json")

    cur_metrics = None
    v1_metrics = None

    if os.path.exists(cur_metrics_path):
        try:
            with open(cur_metrics_path, "r") as f:
                cur_metrics = json.load(f)
        except Exception:
            pass

    if os.path.exists(v1_metrics_path):
        try:
            with open(v1_metrics_path, "r") as f:
                v1_metrics = json.load(f)
        except Exception:
            pass

    has_roc = os.path.exists(os.path.join(RESULTS_DIR, "roc_pretrained.png"))
    has_train_curve = os.path.exists(os.path.join(RESULTS_DIR, "training_curves.png"))

    return jsonify({
        "current_metrics": cur_metrics,
        "v1_metrics": v1_metrics,
        "has_roc": has_roc,
        "has_train_curve": has_train_curve,
    })


TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "test_images")


@app.route("/test_images/<path:filename>")
def serve_test_images(filename):
    """Serves sample test images."""
    return send_from_directory(TEST_IMAGES_DIR, filename)


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serves generated plots from the results directory."""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting Face Recognition Web Demo on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
