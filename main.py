# ============================================================
# AgroEye Phase 2 — FastAPI Inference Server
# Assumes setup_model.py has already been run.
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================

import os, glob, base64
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import pipeline

# ── Config (env vars only) ────────────────────────────────────
FRAME_DIR     = "./frames"
MODEL_NAME    = "wambugu71/crop_leaf_diseases_vit"
MODEL_CACHE   = "./model_cache"
FRAME_SIZE    = (224, 224)

app = FastAPI(title="AgroEye Inference Server")

# ── Load from local cache at startup — no download here ───────
print(f"Loading model from cache: {MODEL_CACHE}")
classifier = pipeline(
    "image-classification",
    model=MODEL_NAME,
    device=-1       # CPU always
)
print("Model loaded. Server ready.")

# ── State ─────────────────────────────────────────────────────
state = {"last_frame": None}

# ── Helpers ───────────────────────────────────────────────────
def get_latest_frame(folder: str) -> Path | None:
    frames = sorted(
        glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.png"),
        key=os.path.getmtime
    )
    return Path(frames[-1]) if frames else None


def parse_label(label: str, score: float) -> dict:
    label_clean = label.replace("_", " ").strip()
    is_healthy  = "healthy" in label_clean.lower()

    if is_healthy:
        return {
            "crop_health_score": round(score, 3),
            "disease_detected":  False,
            "disease_name":      None,
            "anomaly_flags":     []
        }

    health = round(1.0 - score, 3)
    flags  = []
    l      = label_clean.lower()
    if any(k in l for k in ["blight", "rot", "wilt"]):   flags.append("wilting")
    if any(k in l for k in ["spot", "rust", "mold", "mildew"]): flags.append("lesions")
    if any(k in l for k in ["mosaic", "virus", "yellow"]):flags.append("yellowing")
    if any(k in l for k in ["pest", "mite", "aphid"]):   flags.append("pest_damage")

    return {
        "crop_health_score": health,
        "disease_detected":  True,
        "disease_name":      label_clean,
        "anomaly_flags":     flags if flags else ["anomaly_detected"]
    }


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "model_connected": True,
        "model":           MODEL_NAME,
        "device":          "cpu",
        "status":          "ready"
    }


@app.get("/infer")
def infer():
    frame_path = get_latest_frame(FRAME_DIR)

    if frame_path is None:
        raise HTTPException(404, detail="No frames found in FRAME_DIR")

    if str(frame_path) == state["last_frame"]:
        raise HTTPException(304, detail=f"No new frame since {frame_path.name}")

    img     = Image.open(frame_path).convert("RGB").resize(FRAME_SIZE, Image.LANCZOS)
    top     = classifier(img, top_k=1)[0]
    parsed  = parse_label(top["label"], top["score"])

    with open(frame_path, "rb") as f:
        frame_bytes = f.read()
        frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")

    state["last_frame"] = str(frame_path)

    return {
        **parsed,
        "confidence":      round(top["score"], 3),
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "source_frame":    frame_path.name,
        "model_connected": True,
        "frame_base64":    frame_base64
    }