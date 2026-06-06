import io
import base64
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageFilter

app = Flask(__name__, static_folder="static")

model = joblib.load("model.joblib")


def preprocess_image(data_url: str) -> np.ndarray:
    """Convert a canvas data-URL to a 784-dim float array matching MNIST format.

    Replicates the MNIST preprocessing pipeline:
    - Fit digit in 20x20 preserving aspect ratio
    - Center by center of mass in 28x28
    """
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Smooth canvas aliasing so edges look like MNIST's anti-aliased strokes
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    pixels = np.array(img, dtype=np.float32)

    # Find tight bounding box of the drawn digit
    mask = pixels > 10
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return np.zeros((1, 784), dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    digit = pixels[rmin:rmax + 1, cmin:cmax + 1]
    h, w = digit.shape

    # Scale to fit inside 20x20 while preserving aspect ratio (MNIST standard)
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    digit_img = Image.fromarray(digit).resize((new_w, new_h), Image.LANCZOS)
    digit_arr = np.array(digit_img, dtype=np.float32)

    # Center by center of mass within 28x28 (MNIST centering algorithm)
    total = digit_arr.sum()
    if total > 0:
        y_idx, x_idx = np.mgrid[0:new_h, 0:new_w]
        cy = (y_idx * digit_arr).sum() / total
        cx = (x_idx * digit_arr).sum() / total
    else:
        cy, cx = new_h / 2.0, new_w / 2.0

    off_y = int(round(13.5 - cy))
    off_x = int(round(13.5 - cx))

    result = np.zeros((28, 28), dtype=np.float32)
    src_y0 = max(0, -off_y)
    src_x0 = max(0, -off_x)
    dst_y0 = max(0, off_y)
    dst_x0 = max(0, off_x)
    src_y1 = min(new_h, 28 - off_y)
    src_x1 = min(new_w, 28 - off_x)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = digit_arr[src_y0:src_y1, src_x0:src_x1]

    return result.flatten().reshape(1, -1)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/predict")
def predict():
    body = request.get_json(force=True)
    data_url = body.get("image")
    if not data_url:
        return jsonify({"error": "no image provided"}), 400

    try:
        features = preprocess_image(data_url)
        prediction = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        confidence = float(proba[prediction])
        top = sorted(
            enumerate(proba.tolist()), key=lambda x: x[1], reverse=True
        )[:3]
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 1),
            "top3": [{"digit": d, "prob": round(p * 100, 1)} for d, p in top],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
