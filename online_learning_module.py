# online_learning.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier

MODEL_PATH = "monkeymania_online_model.pkl"
DATA_PATH = "training_data.csv"

FEATURES = ['gap_pct', 'premarket_volume', 'float', 'sentiment_score']
LABEL = 'trade_outcome'  # 1 for win, 0 for loss

def init_model():
    clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42)
    return clf

def load_training_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        # Empty dataframe with columns ready
        cols = FEATURES + [LABEL]
        return pd.DataFrame(columns=cols)

def save_training_data(df):
    df.to_csv(DATA_PATH, index=False)

def train_or_update_model(model, data):
    if len(data) == 0:
        return model

    X = data[FEATURES].values
    y = data[LABEL].values.astype(int)

    if not hasattr(model, 'classes_'):
        model.partial_fit(X, y, classes=np.array([0, 1]))
    else:
        model.partial_fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return init_model()

def predict_success_prob(model, feature_row):
    x = np.array([[feature_row.get(f, 0) for f in FEATURES]])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(x)[0][1]
        return proba
    else:
        # If model not trained yet, return neutral probability
        return 0.5

def append_trade_data(new_rows):
    """
    new_rows: list of dicts with keys FEATURES + LABEL
    """
    df = load_training_data()
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    save_training_data(df)