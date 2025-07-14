# confidence_calibrator.py

# == CALIBRATOR STORAGE ==
import json
import os
from datetime import datetime

CALIBRATION_PATH = "data/confidence_bins.json"
DRIFT_PATH = "data/confidence_drift.json"

try:
    with open(CALIBRATION_PATH, "r") as f:
        confidence_bins = json.load(f)
except:
    confidence_bins = {}  # Format: {'0.5-0.6': [("WIN",), ("LOSS",)]}

try:
    with open(DRIFT_PATH, "r") as f:
        confidence_drift = json.load(f)
except:
    confidence_drift = []  # Format: [{"date": "2024-01-01", "avg_raw": 0.65, "avg_calibrated": 0.58}]

# == BIN KEY GENERATOR ==
def get_bin_key(confidence):
    floor = int(confidence * 10) / 10
    ceil = floor + 0.1
    return f"{floor:.1f}-{ceil:.1f}"

# == TRACK OUTCOME PER BIN ==
def log_confidence_result(confidence, result):
    key = get_bin_key(confidence)
    if key not in confidence_bins:
        confidence_bins[key] = []
    confidence_bins[key].append(result)
    confidence_bins[key] = confidence_bins[key][-500:]  # Trim

# == BIN ACCURACY ESTIMATOR ==
def get_bin_accuracy(confidence):
    key = get_bin_key(confidence)
    outcomes = confidence_bins.get(key, [])
    if not outcomes:
        return 1.0  # default to no adjustment
    wins = sum(1 for r in outcomes if r == "WIN")
    return round(wins / len(outcomes), 3)

# == CALIBRATE CONFIDENCE ==
def calibrate_confidence(raw_conf):
    adjustment = get_bin_accuracy(raw_conf)
    return raw_conf * adjustment

# == TRACK DRIFT ==
def log_confidence_drift(raw_confidences, calibrated_confidences):
    if not raw_confidences or not calibrated_confidences:
        return
    
    avg_raw = sum(raw_confidences) / len(raw_confidences)
    avg_calibrated = sum(calibrated_confidences) / len(calibrated_confidences)
    
    drift_entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "avg_raw": round(avg_raw, 3),
        "avg_calibrated": round(avg_calibrated, 3),
        "drift": round(avg_raw - avg_calibrated, 3)
    }
    
    confidence_drift.append(drift_entry)
    # Keep last 250 entries (1 year of trading)
    confidence_drift[:] = confidence_drift[-250:]
    
    save_confidence_drift()

# == GET BIN STATISTICS ==
def get_bin_statistics():
    stats = {}
    for bin_key, outcomes in confidence_bins.items():
        if outcomes:
            wins = sum(1 for r in outcomes if r == "WIN")
            total = len(outcomes)
            win_rate = wins / total
            stats[bin_key] = {
                "win_rate": round(win_rate, 3),
                "total_trades": total,
                "wins": wins,
                "losses": total - wins
            }
    return stats

# == SAVE ==
def save_confidence_bins():
    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(confidence_bins, f, indent=2)

def save_confidence_drift():
    os.makedirs(os.path.dirname(DRIFT_PATH), exist_ok=True)
    with open(DRIFT_PATH, "w") as f:
        json.dump(confidence_drift, f, indent=2)