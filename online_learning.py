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
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
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


def train_or_update_model(model, new_data):
    """Train or update the ML model"""
    # Clean the data first
    new_data = new_data.fillna({
        'gap_pct': 0.0,
        'premarket_volume': 100000,
        'float': 10000000,
        'sentiment_score': 0.5,
        'aftermarket_move': 0.0,
        'early_premarket_move': 0.0,
        'momentum_aligned': 0.0,
        'volume_surge': 0.0,
        'gap_magnitude': 0.0,
        'volume_change_pct': 0.0,
        'trade_outcome': 0
    })
    if len(new_data) == 0:
        return model

    X = new_data[FEATURES].values
    y = new_data[LABEL].values.astype(int)

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


def predict_success_prob(model, features):
    """Predict probability of successful trade"""
    import numpy as np
    import pandas as pd
    
    # Convert features to DataFrame if needed
    if isinstance(features, dict):
        features_df = pd.DataFrame([features])
    else:
        features_df = features.copy()
    
    # Replace NaN values with defaults
    features_df = features_df.fillna({
        'gap_pct': 0.0,
        'premarket_volume': 100000,
        'float': 10000000,
        'sentiment_score': 0.5,
        'aftermarket_move': 0.0,
        'early_premarket_move': 0.0,
        'momentum_aligned': 0.0,
        'volume_surge': 0.0,
        'gap_magnitude': 0.0,
        'volume_change_pct': 0.0
    })
    
    # Ensure no NaN values remain
    features_df = features_df.fillna(0)
    
    # Make prediction
    if model is None:
        return 0.5
        
    try:
        # Get probability of positive class
        prob = model.predict_proba(features_df)[0][1]
        return prob
    except:
        return 0.5

def append_trade_data(new_rows):
    """
    new_rows: list of dicts with keys FEATURES + LABEL
    """
    if not new_rows:  # Don't process empty data
        return
        
    df = load_training_data()
    new_df = pd.DataFrame(new_rows)
    
    # Only concat if new_df has data and isn't all NaN
    if not new_df.empty and not new_df.isna().all().all():
        df = pd.concat([df, new_df], ignore_index=True)
        save_training_data(df)