import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse,StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

clf_model = tf.keras.models.load_model('modelo_clasificador_papa.keras')
seg_model = tf.keras.models.load_model('unet_segmentador_roboflow.keras')

def preprocess_image(file, size):
    image = Image.open(file.file).convert("RGB").resize(size)
    return np.expand_dims(np.array(image) / 255.0, axis=0)
    
# Ruta raíz
@app.get("/")
def root():
    return {"message": "API de hojas de papa en funcionamiento"}

# Clasificación de hoja (sana / tizón leve / tizón severo)
@app.post("/predict/classify")
async def classify_leaf(image: UploadFile = File(...)):
    image_data = preprocess_image(image, (224, 224))
    pred = clf_model.predict(image_data)
    label = int(np.argmax(pred))
    confidence = float(np.max(pred))
    # Diccionario de etiquetas
    labels_dict = {
        0: "Sana",
        1: "Tizón leve",
        2: "Tizón severo"
    }

    return {
        "label": label,
        "clase": labels_dict.get(label, "Desconocida"),
        "confidence": confidence,
        "alerta": confidence < 0.5
    }
    
    #return {"label": label, "confidence": confidence}

# Segmentación como lista de valores
@app.post("/predict/segment")
async def segment_leaf(image: UploadFile = File(...)):
    image_data = preprocess_image(image, (128, 128))
    mask = seg_model.predict(image_data)[0, :, :, 0]
    return JSONResponse(content={"mask": mask.tolist()})
    
# NUEVO: Segmentación como imagen PNG
@app.post("/predict/segment-img")
async def segment_leaf_img(image: UploadFile = File(...)):
    image_data = preprocess_image(image, (128, 128))
    mask = seg_model.predict(image_data)[0, :, :, 0]

    # Convertir a imagen en escala de grises
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    # Guardar en memoria
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
    
    
