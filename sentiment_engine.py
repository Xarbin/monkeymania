# sentiment_engine.py

import random

def fetch_sentiment(ticker, mode="dummy"):
    if mode == "dummy":
        sentiments = [-1, 0, 1]
        chosen = random.choices(sentiments, weights=[0.2, 0.5, 0.3])[0]
        label = {
            -1: "👎 Bearish",
             0: "😐 Neutral",
             1: "👍 Bullish"
        }[chosen]
        return chosen, label
    else:
        raise NotImplementedError("Only dummy mode is supported for now.")