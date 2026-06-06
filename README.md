# Number Classifier

A web app where you draw a number on a canvas — single digit or multi-digit (e.g. 42, 137) — and a machine learning model classifies it in real time.

## Stack

- **Model** — `MLPClassifier` (256 → 128 → 10) trained on MNIST via scikit-learn, exported with joblib
- **Backend** — Flask, serves the API and the static frontend
- **Frontend** — Vanilla HTML/CSS/JS with a wide `<canvas>` drawing surface

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

1. You draw a number on the wide black canvas — one digit or several
2. Clicking **Classify** sends the canvas PNG to `POST /predict`
3. The backend:
   - Converts the image to greyscale and applies a Gaussian blur
   - **Segments** the drawing into individual digits using column projection — finds vertical gaps between inked regions and splits there; runs separated by fewer than 8 px are merged so a single digit is never broken apart
   - For each segment: crops to its bounding box, scales to 20×20 preserving aspect ratio, centers by centre of mass in a 28×28 frame (replicating the MNIST preprocessing pipeline), and flattens to a 784-dim float array
   - Runs each array through the trained MLP pipeline (StandardScaler → MLP)
4. The UI shows the full recognised number and a confidence chip for each digit

## Project structure

```
digit-classifier/
├── train_model.py   # fetch MNIST, train MLP, save model.joblib
├── app.py           # Flask server + segmentation + preprocessing
├── static/
│   └── index.html   # drawing canvas + result display
└── requirements.txt
```

## Tips for best results

- Write digits **large** and leave a **clear gap** between them — the segmenter relies on blank vertical space to tell digits apart
- Use thick, confident strokes
- Clear the canvas between predictions

## Possible improvements

- Switch to a CNN (TensorFlow/PyTorch) for better tolerance to rotation and stroke variation
- Use connected-component analysis instead of column projection to handle touching or overlapping digits
- Show a live preview of each 28×28 input the model actually sees
