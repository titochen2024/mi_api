
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Cargar modelos
clf_model = tf.keras.models.load_model('modelo_clasificador_papa.keras')
seg_model = tf.keras.models.load_model('unet_segmentador_roboflow.keras')

def preprocess_image(image, size):
    img = Image.open(image).convert("RGB").resize(size)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

@app.route('/predict/classify', methods=['POST'])
def classify_leaf():
    file = request.files['image']
    image = preprocess_image(file, (224, 224))
    pred = clf_model.predict(image)
    label = np.argmax(pred)
    confidence = float(np.max(pred))
    return jsonify({'label': int(label), 'confidence': confidence})

@app.route('/predict/segment', methods=['POST'])
def segment_leaf():
    file = request.files['image']
    image = preprocess_image(file, (128, 128))
    mask = seg_model.predict(image)[0, :, :, 0]
    return jsonify({'mask': mask.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
