# ============================================================
# AgroEye — Model Setup
# Run ONCE before starting the server.
# Downloads and caches the model locally.
# python setup_model.py
# ============================================================

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import os

MODEL_NAME  = "wambugu71/crop_leaf_diseases_vit"
CACHE_DIR   = "./model_cache"

if __name__ == "__main__":
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Cache dir: {CACHE_DIR}\n")

    AutoFeatureExtractor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    AutoModelForImageClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    print("\nDone. Model cached locally.")
    print(f"Set env var: MODEL_CACHE_DIR={CACHE_DIR}")
    print("Now run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload")