# Digit Classifier

A web app where you draw a digit (0–9) on a canvas and a machine learning model classifies it in real time.

## Stack

- **Model** — `MLPClassifier` (256 → 128 → 10) trained on MNIST via scikit-learn, exported with joblib
- **Backend** — Flask, serves the API and the static frontend
- **Frontend** — Vanilla HTML/CSS/JS with a `<canvas>` drawing surface

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (downloads MNIST ~12 MB on first run, takes ~2 min)
python train_model.py

# 4. Start the server
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## How it works

1. You draw a digit on the black canvas
2. Clicking **Classify** sends the canvas PNG to `POST /predict`
3. The backend:
   - Crops to the bounding box of the drawing and adds padding
   - Resizes to 28×28 (MNIST resolution) with Lanczos resampling
   - Flattens to a 784-dim float array
   - Runs it through the trained MLP pipeline (StandardScaler → MLP)
4. The UI shows the predicted digit, confidence %, and the top-3 candidates

## Project structure

```
digit-classifier/
├── train_model.py   # fetch MNIST, train MLP, save model.joblib
├── app.py           # Flask server + image preprocessing
├── static/
│   └── index.html   # drawing canvas + result display
└── requirements.txt
```

## Tips for best results

- Draw digits **large** and **centered** on the canvas — same way MNIST digits look
- Use thick, confident strokes
- Clear the canvas between predictions

## Next steps / possible improvements

- Switch to a CNN (TensorFlow/PyTorch) for better tolerance to rotation and stroke variation
- Add multi-digit support (segment the drawing before classifying)
- Show a live preview of the 28×28 input the model actually sees
