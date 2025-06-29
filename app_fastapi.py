import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

clf_model = tf.keras.models.load_model('modelo_clasificador_papa.keras')
seg_model = tf.keras.models.load_model('unet_segmentador_roboflow.keras')

def preprocess_image(file, size):
    image = Image.open(file.file).convert("RGB").resize(size)
    return np.expand_dims(np.array(image) / 255.0, axis=0)

@app.post("/predict/classify")
async def classify_leaf(image: UploadFile = File(...)):
    image_data = preprocess_image(image, (224, 224))
    pred = clf_model.predict(image_data)
    label = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return {"label": label, "confidence": confidence}

@app.post("/predict/segment")
async def segment_leaf(image: UploadFile = File(...)):
    image_data = preprocess_image(image, (128, 128))
    mask = seg_model.predict(image_data)[0, :, :, 0]
    return JSONResponse(content={"mask": mask.tolist()})
