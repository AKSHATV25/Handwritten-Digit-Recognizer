# Handwritten Digit Recognizer

A Streamlit app that recognizes hand-drawn digits (0–9) using an MLP
classifier trained on scikit-learn's built-in digits dataset. Draw a
digit on the canvas and get an instant prediction with confidence scores.

## Files
- `app.py` — the Streamlit app (drawable canvas + prediction UI)
- `train_model.py` — trains the model and saves `model.pkl`
- `model.pkl` — pre-trained model (already generated, ~97% test accuracy)
- `requirements.txt` — dependencies for deployment

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a new GitHub repo (include `model.pkl` — it's small).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick the repo/branch, and set the main file to `app.py`.
4. Click **Deploy**. You'll get a public URL like
   `https://your-app-name.streamlit.app` — use that as your demo link.

## Retraining the model
If you want to retrain (e.g., tweak the architecture), run:
```bash
python train_model.py
```
This overwrites `model.pkl`.
