"""
Sentiment Analysis Model
Uses TF-IDF + Logistic Regression trained on synthetic data.
In production: swap load_model() for a GPU-backed transformer via the CUDA module.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Training data — positive and negative sentiment examples
TRAIN_DATA = [
    ("I love this product, it works great", "positive"),
    ("Excellent performance and very fast", "positive"),
    ("Amazing results, highly recommend", "positive"),
    ("Outstanding quality and great value", "positive"),
    ("Best experience I've ever had", "positive"),
    ("Really impressed with the accuracy", "positive"),
    ("Works perfectly, no issues at all", "positive"),
    ("Fantastic tool for developers", "positive"),
    ("Incredibly useful and well designed", "positive"),
    ("Smooth and reliable performance", "positive"),
    ("Terrible experience, completely broken", "negative"),
    ("Very slow and keeps crashing", "negative"),
    ("Worst product I have ever used", "negative"),
    ("Doesn't work at all, waste of time", "negative"),
    ("Extremely disappointing results", "negative"),
    ("Buggy and unreliable, avoid this", "negative"),
    ("Poor quality and bad support", "negative"),
    ("Confusing interface and many errors", "negative"),
    ("Failed to deliver what was promised", "negative"),
    ("Constant issues and terrible performance", "negative"),
]


def load_model():
    """Load model from disk or train a fresh one."""
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        return pipeline, None  # pipeline includes vectorizer

    texts, labels = zip(*TRAIN_DATA)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0))
    ])
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline, None


def predict(model, vectorizer, text: str) -> dict:
    """
    Run inference on a single text.
    Returns label, confidence, and raw probabilities.
    Uses CPU on Mac; swap predict() internals for CUDA module on GPU server.
    """
    pipeline = model
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    label = classes[np.argmax(proba)]
    confidence = round(float(np.max(proba)), 4)

    return {
        "text":       text[:100],
        "label":      label,
        "confidence": confidence,
        "scores": {
            cls: round(float(p), 4)
            for cls, p in zip(classes, proba)
        }
    }
