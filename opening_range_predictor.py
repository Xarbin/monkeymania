# analysis/opening_range_predictor.py
"""Predict opening 30-minute behavior from pre-market data"""

from typing import Dict, Optional
import numpy as np

class OpeningRangePredictor:
    """Predicts opening range behavior from limited data"""
    
    def __init__(self):
        self.historical_patterns = {}
        self.pattern_definitions = self._define_patterns()
    
    def _define_patterns(self) -> Dict:
        """Define opening range patterns"""
        return {
            'gap_and_go': {
                'setup': lambda g, pv: g > 0.03 and pv > 2.0,
                'typical_range': 0.015,
                'continuation_prob': 0.72,
                'peak_time': '09:45'
            },
            'gap_fill': {
                'setup': lambda g, pv: abs(g) > 0.02 and pv < 1.0,
                'typical_range': 0.02,
                'continuation_prob': 0.31,
                'peak_time': '10:00'
            },
            'open_drive': {
                'setup': lambda g, pv: abs(g) < 0.01 and pv > 1.5,
                'typical_range': 0.025,
                'continuation_prob': 0.68,
                'peak_time': '09:35'
            },
            'fake_out': {
                'setup': lambda g, pv: g > 0.04 and pv < 0.5,
                'typical_range': 0.03,
                'continuation_prob': 0.25,
                'peak_time': '09:40'
            }
        }
    
    def predict_opening_behavior(self, 
                               gap_percent: float, 
                               pre_volume_ratio: float,
                               yesterday_close: float,
                               pre_market_trend: str) -> Dict:
        """Predict how the opening 30 minutes will behave"""
        
        # Identify pattern
        pattern = self._identify_pattern(gap_percent, pre_volume_ratio)
        
        if not pattern:
            return self._default_prediction()
        
        # Calculate specific predictions
        pattern_data = self.pattern_definitions[pattern]
        
        # Adjust for market conditions
        volatility_multiplier = self._get_volatility_adjustment()
        
        prediction = {
            'pattern': pattern,
            'expected_range': pattern_data['typical_range'] * volatility_multiplier,
            'direction_probability': pattern_data['continuation_prob'],
            'peak_volatility_time': pattern_data['peak_time'],
            'suggested_entry': self._calculate_entry_strategy(pattern, gap_percent),
            'risk_level': self._assess_risk(pattern, pre_volume_ratio),
            'confidence': self._calculate_confidence(pattern, pre_volume_ratio)
        }
        
        return prediction
    
    def _identify_pattern(self, gap: float, volume_ratio: float) -> Optional[str]:
        """Identify which opening pattern is likely"""
        for pattern_name, pattern_data in self.pattern_definitions.items():
            if pattern_data['setup'](gap, volume_ratio):
                return pattern_name
        return None
    
    def _calculate_entry_strategy(self, pattern: str, gap: float) -> Dict:
        """Calculate specific entry strategy"""
        strategies = {
            'gap_and_go': {
                'entry': 'Buy break of first 5-min high',
                'stop': 'Below first 5-min low',
                'target': f'{gap * 2:.1f}% extension'
            },
            'gap_fill': {
                'entry': 'Short after initial pop',
                'stop': 'Above opening range high',
                'target': 'Yesterday close'
            },
            'open_drive': {
                'entry': 'Buy pullback to VWAP',
                'stop': 'Below VWAP minus ATR',
                'target': 'R-multiple of 2:1'
            },
            'fake_out': {
                'entry': 'Wait for reversal confirmation',
                'stop': 'Beyond opening range',
                'target': 'Gap fill level'
            }
        }
        return strategies.get(pattern, self._default_strategy())
    
    def _default_strategy(self) -> Dict:
        """Default conservative strategy"""
        return {
            'entry': 'Wait for range to establish',
            'stop': 'Outside opening range',
            'target': 'Range projection'
        }
    
    def _assess_risk(self, pattern: str, volume_ratio: float) -> str:
        """Assess risk level of the setup"""
        if pattern == 'gap_and_go' and volume_ratio > 3:
            return 'LOW'
        elif pattern == 'fake_out':
            return 'HIGH'
        elif volume_ratio < 0.5:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _calculate_confidence(self, pattern: str, volume_ratio: float) -> float:
        """Calculate confidence in prediction"""
        base_confidence = self.pattern_definitions[pattern]['continuation_prob']
        
        # Adjust for volume
        if volume_ratio > 2:
            base_confidence *= 1.1
        elif volume_ratio < 0.5:
            base_confidence *= 0.8
            
        return min(base_confidence, 0.95)
    
    def _get_volatility_adjustment(self) -> float:
        """Get market volatility adjustment"""
        # In production, this would check VIX or other volatility measures
        return 1.0
    
    def _default_prediction(self) -> Dict:
        """Default prediction when no pattern matches"""
        return {
            'pattern': 'undefined',
            'expected_range': 0.01,
            'direction_probability': 0.5,
            'suggested_entry': self._default_strategy(),
            'risk_level': 'HIGH',
            'confidence': 0.3
        }