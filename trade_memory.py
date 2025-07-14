# trade_memory.py

import csv
import os

MEMORY_FILE = "trade_memory.csv"
HEADERS = [
    "ticker",
    "entry_price",
    "exit_price",
    "shares",
    "pnl",
    "result",
    "predicted_prob",
    "raw_prob",
    "gap_pct",
    "premarket_volume",
    "float",
    "sentiment_score"
]

def init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def log_trade_result(trade_data: dict):
    init_memory()
    row = [
        trade_data.get("ticker"),
        trade_data.get("entry_price"),
        trade_data.get("exit_price"),
        trade_data.get("shares"),
        trade_data.get("pnl"),
        trade_data.get("result"),
        trade_data.get("predicted_prob"),
        trade_data.get("raw_prob", trade_data.get("predicted_prob")),  # Fallback to predicted_prob
        trade_data.get("gap_pct"),
        trade_data.get("premarket_volume"),
        trade_data.get("float"),
        trade_data.get("sentiment_score", 0),
    ]
    with open(MEMORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)