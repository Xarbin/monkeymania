# analysis/candle_analyzer.py
"""Synthetic candle analysis from limited data points"""

import numpy as np
from typing import Dict, Tuple

class CandleAnalyzer:
    """Analyzes price action from pre/post market data"""
    
    def __init__(self):
        self.candle_patterns = self._load_pattern_definitions()
    
    def create_synthetic_candles(self, 
                                pre_close: float, 
                                pre_volume: float, 
                                post_close: float, 
                                post_volume: float,
                                regular_open: float, 
                                regular_close: float, 
                                high: float, 
                                low: float) -> Dict:
        """Build synthetic candle data from available points"""
        
        # Calculate key metrics
        volatility_score = (high - low) / regular_open if regular_open > 0 else 0
        
        # Wick analysis
        body_size = abs(regular_close - regular_open)
        upper_wick = high - max(regular_open, regular_close)
        lower_wick = min(regular_open, regular_close) - low
        
        # Gap analysis
        gap_size = (regular_open - pre_close) / pre_close if pre_close > 0 else 0
        gap_filled = self._check_gap_fill(pre_close, regular_open, high, low)
        
        # Price acceptance
        price_acceptance = 1 - (body_size / (high - low)) if (high - low) > 0 else 0
        
        return {
            'volatility_expansion': volatility_score,
            'upper_wick_ratio': upper_wick / body_size if body_size > 0 else 0,
            'lower_wick_ratio': lower_wick / body_size if body_size > 0 else 0,
            'gap_size': gap_size,
            'gap_filled': gap_filled,
            'price_acceptance': price_acceptance,
            'candle_type': self._classify_candle(regular_open, regular_close, high, low)
        }
    
    def _check_gap_fill(self, pre_close: float, open_price: float, 
                       high: float, low: float) -> bool:
        """Check if gap was filled during session"""
        if open_price > pre_close:  # Gap up
            return low <= pre_close
        elif open_price < pre_close:  # Gap down
            return high >= pre_close
        return False
    
    def _classify_candle(self, open_price: float, close: float, 
                        high: float, low: float) -> str:
        """Classify candle pattern"""
        body = abs(close - open_price)
        range_size = high - low
        
        if range_size == 0:
            return 'doji'
        
        body_ratio = body / range_size
        
        if body_ratio < 0.1:
            return 'doji'
        elif close > open_price:
            if (high - close) < body * 0.1:
                return 'bullish_marubozu'
            elif (open_price - low) > body * 2:
                return 'hammer'
            else:
                return 'bullish'
        else:
            if (close - low) < body * 0.1:
                return 'bearish_marubozu'
            elif (high - open_price) > body * 2:
                return 'shooting_star'
            else:
                return 'bearish'
    
    def _load_pattern_definitions(self) -> Dict:
        """Load candle pattern definitions"""
        return {
            'hammer': {'bullish': True, 'reliability': 0.65},
            'shooting_star': {'bullish': False, 'reliability': 0.63},
            'bullish_marubozu': {'bullish': True, 'reliability': 0.71},
            'bearish_marubozu': {'bullish': False, 'reliability': 0.69},
            'doji': {'bullish': None, 'reliability': 0.5}
        }