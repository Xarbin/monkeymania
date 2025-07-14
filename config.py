# core/config.py
"""Centralized configuration for MonkeyMania"""

import os
from pathlib import Path

class Config:
    """System-wide configuration"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Trading parameters
    MAX_DAILY_TRADES = 5
    POSITION_SIZE_PCT = 0.20  # 20% of capital
    MAX_POSITIONS = 5
    STOP_LOSS_PCT = -0.05
    TAKE_PROFIT_PCT = 0.10
    
    # ML parameters
    MIN_TRAINING_SAMPLES = 30
    CONFIDENCE_THRESHOLD = 0.6
    SHADOW_PICKS_COUNT = 5
    
    # Volume thresholds
    MIN_PREMARKET_VOLUME = 50000
    VOLUME_SURGE_MULTIPLIER = 5.0
    INSTITUTIONAL_VOLUME_PCT = 0.15
    
    # Gap thresholds
    SIGNIFICANT_GAP_PCT = 0.02
    EXTREME_GAP_PCT = 0.05
    
    # Time windows
    OPENING_WINDOW_MINUTES = 30
    POWER_HOUR_START = "15:00"
    
    # Feature lists
    ML_FEATURES = [
        'gap_pct', 'premarket_volume', 'float', 'sentiment_score',
        'momentum_score', 'volume_float_ratio', 'volatility_score'
    ]
    
    # File names
    BROKER_STATE_FILE = 'broker_state.json'
    PATTERN_MEMORY_FILE = 'pattern_memory.json'
    MODEL_FILE = 'monkeymania_online_model.pkl'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True)