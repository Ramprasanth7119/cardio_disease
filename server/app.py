from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import tensorflow as tf
import cv2
import base64
import os
from PIL import Image

app = Flask(__name__)

# ✅ Allow only your Vercel frontend origin (recommended)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://cardio-disease-sooty.vercel.app",
            "http://localhost:3000"
        ]
    }
})

# ----------------------------
# LOAD TEXT MODEL
# ----------------------------
TEXT_MODEL_PATH = "cardio_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(TEXT_MODEL_PATH):
    raise FileNotFoundError(f"Missing file: {TEXT_MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Missing file: {SCALER_PATH}")

text_model = joblib.load(TEXT_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

TEXT_FEATURES = ["age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active"]

# ----------------------------
# LOAD IMAGE MODEL
# ----------------------------
IMG_SIZE = 224
cnn_model_path = "models/cnn_model.h5"

if not os.path.exists(cnn_model_path):
    raise FileNotFoundError(f"Missing model file: {cnn_model_path}")

image_model = tf.keras.models.load_model(cnn_model_path)

# ----------------------------
# TEXT RISK PREDICTION
# ----------------------------
@app.route("/predict/text", methods=["POST"])
def predict_text():
    data = request.json or {}

    X = np.array([[data.get(f, 0) for f in TEXT_FEATURES]])
    X_scaled = scaler.transform(X)

    pred = int(text_model.predict(X_scaled)[0])
    prob = float(text_model.predict_proba(X_scaled)[0][1])

    increasing = [f for i, f in enumerate(TEXT_FEATURES) if X[0][i] > 0.5][:3]
    decreasing = [f for i, f in enumerate(TEXT_FEATURES) if X[0][i] <= 0.5][:3]

    return jsonify({
        "prediction": "High Risk" if pred == 1 else "Low Risk",
        "probability": round(prob, 3),
        "risk_increasing_factors": increasing,
        "risk_decreasing_factors": decreasing
    })

# ----------------------------
# IMAGE PREDICTION + Grad-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    conv_output = None

    for layer in model.layers:
        x = layer(x)
        if layer == last_conv_layer:
            conv_output = x

    preds_output = x
    grad_model = tf.keras.Model(inputs, [conv_output, preds_output])

    with tf.GradientTape() as tape:
        img_array = tf.cast(img_array, tf.float32)
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / max_val if max_val > 0 else tf.zeros_like(heatmap)

    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = np.uint8(original_img * 255)
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

@app.route("/predict/image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("L").resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0
    input_array = np.expand_dims(img_array, axis=(0, -1))

    _ = image_model.predict(input_array, verbose=0)

    heatmap = make_gradcam_heatmap(input_array, image_model)
    overlay_img = overlay_gradcam(img_array, heatmap)

    _, buffer = cv2.imencode(".png", overlay_img)
    gradcam_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"gradcam_image": gradcam_b64})

# ----------------------------
# FEATURE MAPS
# ----------------------------
@app.route("/feature_maps", methods=["POST"])
def feature_maps():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("L").resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0
    input_array = np.expand_dims(img_array, axis=(0, -1))

    layer = next(l for l in image_model.layers if isinstance(l, tf.keras.layers.Conv2D))
    feature_model = tf.keras.Model(inputs=image_model.inputs, outputs=layer.output)
    feature_maps_out = feature_model.predict(input_array, verbose=0)

    fm_list = []
    for i in range(min(6, feature_maps_out.shape[-1])):
        fmap = feature_maps_out[0, :, :, i]
        fmap = cv2.resize(fmap, (IMG_SIZE, IMG_SIZE))
        fmap = np.uint8(255 * fmap)

        _, buf = cv2.imencode(".png", fmap)
        b64 = base64.b64encode(buf).decode("utf-8")
        fm_list.append(b64)

    return jsonify({"feature_maps": fm_list})

# ✅ Health check endpoint (Render uses this sometimes)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend running"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
