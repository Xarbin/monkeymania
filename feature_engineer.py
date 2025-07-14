# learning/feature_engineer.py
"""Feature engineering for ML model"""

import numpy as np
from typing import Dict, List
import pandas as pd

class FeatureEngineer:
    """Creates and manages features for ML model"""
    
    def __init__(self):
        self.feature_definitions = self._define_features()
        self.feature_importance = {}
    
    def _define_features(self) -> Dict:
        """Define all feature calculations"""
        return {
            # Basic features
            'gap_pct': lambda d: d.get('gap_pct', 0),
            'premarket_volume': lambda d: d.get('premarket_volume', 100000),
            'float': lambda d: d.get('float', 10000000),
            'sentiment_score': lambda d: d.get('sentiment_score', 0.5),
            
            # Derived features
            'volume_float_ratio': lambda d: d['premarket_volume'] / d['float'] if d.get('float', 0) > 0 else 0,
            'gap_magnitude': lambda d: abs(d.get('gap_pct', 0)),
            'momentum_score': lambda d: d.get('early_premarket_move', 0) * np.sign(d.get('gap_pct', 0)),
            
            # Composite features
            'volatility_score': lambda d: (abs(d.get('gap_pct', 0)) + 
                                          abs(d.get('early_premarket_move', 0))) / 2,
            'institutional_interest': lambda d: 1 if d.get('premarket_volume', 0) > 
                                               d.get('avg_volume', 100000) * 0.15 else 0,
            
            # Time-based features
            'day_of_week': lambda d: d.get('date', pd.Timestamp.now()).dayofweek if hasattr(d.get('date'), 'dayofweek') else 0,
            'is_monday': lambda d: 1 if d.get('day_of_week', 0) == 0 else 0,
            'is_friday': lambda d: 1 if d.get('day_of_week', 0) == 4 else 0,
        }
    
    def engineer_features(self, raw_data: Dict, 
                         include_advanced: bool = True) -> Dict:
        """Create all features from raw data"""
        features = {}
        
        # Calculate basic features
        for feature_name, calculator in self.feature_definitions.items():
            try:
                features[feature_name] = calculator(raw_data)
            except:
                features[feature_name] = 0  # Default value
        
        if include_advanced:
            # Add interaction features
            features.update(self._create_interaction_features(features))
            
            # Add technical features
            features.update(self._create_technical_features(raw_data))
        
        return features
    
    def _create_interaction_features(self, features: Dict) -> Dict:
        """Create interaction features"""
        interactions = {}
        
        # Gap * Volume interaction
        interactions['gap_volume_interaction'] = (
            features.get('gap_magnitude', 0) * 
            np.log1p(features.get('premarket_volume', 1))
        )
        
        # Momentum alignment
        interactions['momentum_aligned'] = (
            1 if features.get('gap_pct', 0) * 
            features.get('momentum_score', 0) > 0 else 0
        )
        
        return interactions
    
    def _create_technical_features(self, raw_data: Dict) -> Dict:
        """Create technical indicator features"""
        technical = {}
        
        # Price position
        if all(k in raw_data for k in ['close', 'high', 'low']):
            price_range = raw_data['high'] - raw_data['low']
            if price_range > 0:
                technical['close_position'] = (
                    (raw_data['close'] - raw_data['low']) / price_range
                )
        
        # ATR-based features
        if 'atr' in raw_data and raw_data.get('close', 0) > 0:
            technical['atr_percentage'] = raw_data['atr'] / raw_data['close']
        
        return technical
    
    def get_feature_vector(self, features: Dict, 
                          feature_list: List[str]) -> np.ndarray:
        """Get feature vector in consistent order"""
        return np.array([features.get(f, 0) for f in feature_list])
    
    def update_feature_importance(self, feature_name: str, 
                                importance_score: float):
        """Track feature importance over time"""
        if feature_name not in self.feature_importance:
            self.feature_importance[feature_name] = []
        
        self.feature_importance[feature_name].append(importance_score)
        
        # Keep rolling average of last 100
        if len(self.feature_importance[feature_name]) > 100:
            self.feature_importance[feature_name] = (
                self.feature_importance[feature_name][-100:]
            )