import io
import base64
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageFilter

app = Flask(__name__, static_folder="static")

model = joblib.load("model.joblib")


def _to_mnist_features(digit_arr: np.ndarray) -> np.ndarray:
    h, w = digit_arr.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    digit_img = Image.fromarray(digit_arr).resize((new_w, new_h), Image.LANCZOS)
    scaled = np.array(digit_img, dtype=np.float32)

    total = scaled.sum()
    if total > 0:
        y_idx, x_idx = np.mgrid[0:new_h, 0:new_w]
        cy = (y_idx * scaled).sum() / total
        cx = (x_idx * scaled).sum() / total
    else:
        cy, cx = new_h / 2.0, new_w / 2.0

    off_y = int(round(13.5 - cy))
    off_x = int(round(13.5 - cx))

    result = np.zeros((28, 28), dtype=np.float32)
    src_y0 = max(0, -off_y);  src_x0 = max(0, -off_x)
    dst_y0 = max(0,  off_y);  dst_x0 = max(0,  off_x)
    src_y1 = min(new_h, 28 - off_y)
    src_x1 = min(new_w, 28 - off_x)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]

    return result.flatten().reshape(1, -1)


def _segment_digits(pixels: np.ndarray) -> list:
    """Split a grayscale image into per-digit sub-images ordered left to right."""
    col_has_ink = np.any(pixels > 10, axis=0)

    in_run = False
    runs = []
    for c, v in enumerate(col_has_ink):
        if v and not in_run:
            in_run = True
            start = c
        elif not v and in_run:
            in_run = False
            runs.append([start, c])
    if in_run:
        runs.append([start, len(col_has_ink)])

    # Merge column runs separated by fewer than 8 px — avoids splitting a single
    # digit on interior gaps (e.g. the gap inside a "4" or "7").
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 8:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])

    segments = []
    for cmin, cmax in merged:
        if cmax - cmin < 5:
            continue
        col_slice = pixels[:, cmin:cmax]
        row_mask = np.any(col_slice > 10, axis=1)
        if not row_mask.any():
            continue
        rmin, rmax = np.where(row_mask)[0][[0, -1]]
        segments.append(pixels[rmin:rmax + 1, cmin:cmax])

    return segments


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
        _, encoded = data_url.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        pixels = np.array(img, dtype=np.float32)

        segments = _segment_digits(pixels)
        if not segments:
            return jsonify({"error": "nothing drawn"}), 400

        digits = []
        for seg in segments:
            features = _to_mnist_features(seg)
            pred = int(model.predict(features)[0])
            proba = model.predict_proba(features)[0]
            digits.append({
                "digit": pred,
                "confidence": round(float(proba[pred]) * 100, 1),
            })

        return jsonify({
            "number": "".join(str(d["digit"]) for d in digits),
            "digits": digits,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
