import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import keras
import tensorflow as tf

app = FastAPI(title="Potato Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.keras")

model = None


@app.on_event("startup")
async def load_model():
    global model
    print(f"Memuat model dari: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat.")


@app.get("/")
def health_check():
    return {
        "status": "✅ Server berjalan!",
        "message": "Potato Disease Detection API",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Validate file type
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Hanya file JPG/PNG yang diizinkan.")

    contents = await image.read()

    # Validate file size (5MB)
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file melebihi batas 5MB.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap, coba lagi sebentar.")

    try:
        # Preprocess image — sama seperti training (efficientnet.preprocess_input)
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((300, 300))
        arr = np.array(img, dtype=np.float32)          # range [0, 255]
        batched = np.expand_dims(arr, axis=0)          # tambah batch dim dulu
        batched = tf.keras.applications.efficientnet.preprocess_input(batched)  # → [-1, 1]

        # Predict
        predictions = model.predict(batched, verbose=0)
        scores = predictions[0]

        all_predictions = {
            name: f"{float(scores[i]) * 100:.2f}%"
            for i, name in enumerate(CLASS_NAMES)
        }

        max_idx = int(np.argmax(scores))
        return {
            "class": CLASS_NAMES[max_idx],
            "confidence": f"{float(scores[max_idx]) * 100:.2f}%",
            "all_predictions": all_predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {str(e)}")
