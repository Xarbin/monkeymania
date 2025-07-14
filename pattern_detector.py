# patterns/pattern_detector.py
"""Extract pattern detection from trade_controller"""

from typing import Dict, List, Optional
import numpy as np

class PatternDetector:
    """Centralized pattern detection system"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict:
        """Define all trading patterns"""
        return {
            'gap_and_go': {
                'conditions': lambda f: f['gap_pct'] > 5 and f['early_premarket_move'] > 10,
                'confidence_boost': 0.15,
                'typical_move': 0.08
            },
            'reversal': {
                'conditions': lambda f: f['gap_pct'] < -2 and f['premarket_volume'] > 1000000,
                'confidence_boost': 0.10,
                'typical_move': 0.05
            },
            'volume_spike': {
                'conditions': lambda f: f.get('volume_change_pct', 0) > 500,
                'confidence_boost': 0.20,
                'typical_move': 0.10
            },
            'momentum_surge': {
                'conditions': lambda f: f['gap_pct'] > 20,
                'confidence_boost': 0.25,
                'typical_move': 0.15
            },
            'breakout': {
                'conditions': lambda f: abs(f['gap_pct']) < 2 and f['premarket_volume'] > 2000000,
                'confidence_boost': 0.12,
                'typical_move': 0.06
            },
            'consolidation': {
                'conditions': lambda f: -2 < f['gap_pct'] < 2 and f['premarket_volume'] < 500000,
                'confidence_boost': -0.10,
                'typical_move': 0.02
            }
        }
    
    def identify_patterns(self, features: Dict) -> List[str]:
        """Identify all patterns present in the data"""
        detected_patterns = []
        
        for pattern_name, pattern_data in self.patterns.items():
            if pattern_data['conditions'](features):
                detected_patterns.append(pattern_name)
        
        return detected_patterns if detected_patterns else ['unknown']
    
    def get_primary_pattern(self, features: Dict) -> str:
        """Get the most significant pattern"""
        patterns = self.identify_patterns(features)
        
        if len(patterns) == 1:
            return patterns[0]
        
        # If multiple patterns, prioritize by confidence boost
        pattern_scores = {
            p: self.patterns[p]['confidence_boost'] 
            for p in patterns if p in self.patterns
        }
        
        return max(pattern_scores, key=pattern_scores.get) if pattern_scores else 'unknown'
    
    def get_pattern_confidence(self, pattern: str) -> float:
        """Get confidence adjustment for pattern"""
        return self.patterns.get(pattern, {}).get('confidence_boost', 0)
    
    def get_expected_move(self, pattern: str) -> float:
        """Get typical move size for pattern"""
        return self.patterns.get(pattern, {}).get('typical_move', 0.03)