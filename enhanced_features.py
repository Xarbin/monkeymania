# analysis/enhanced_features.py
"""New enhanced analysis features that integrate with XP/AI system"""

import numpy as np
from typing import Dict, List, Tuple
from skills_engine import award_xp_new, update_skill_accuracy

class EnhancedAnalyzer:
    """Adds new analysis capabilities with XP tracking"""
    
    def __init__(self, xp_tracker=None):
        self.xp_tracker = xp_tracker
        self.candle_analyzer = CandleAnalyzer()
        self.opening_predictor = OpeningRangePredictor()
        self.volume_profiler = VolumeProfiler()
        self.relative_strength = RelativeStrengthAnalyzer()
        
        # Track which analyzer made the best predictions
        self.analyzer_performance = {
            'candle_patterns': {'correct': 0, 'total': 0},
            'opening_range': {'correct': 0, 'total': 0},
            'volume_profile': {'correct': 0, 'total': 0},
            'relative_strength': {'correct': 0, 'total': 0}
        }
    
    def enhance_ticker_analysis(self, existing_features, ticker_data):
        """Add new analysis to existing features"""
        enhanced = existing_features.copy()
        
        # Candle analysis
        candle_features = self.candle_analyzer.analyze(ticker_data)
        enhanced['candle_pattern'] = candle_features['pattern']
        enhanced['candle_strength'] = candle_features['strength']
        
        # Volume profiling  
        volume_analysis = self.volume_profiler.analyze(ticker_data)
        enhanced['volume_signal'] = volume_analysis['signal']
        enhanced['smart_money'] = volume_analysis['institutional_presence']
        
        # Opening prediction
        opening_pred = self.opening_predictor.predict(ticker_data)
        enhanced['opening_pattern'] = opening_pred['pattern']
        enhanced['opening_confidence'] = opening_pred['confidence']
        
        # Relative strength
        rs_analysis = self.relative_strength.analyze(ticker_data)
        enhanced['relative_strength'] = rs_analysis['score']
        enhanced['sector_leader'] = rs_analysis['is_leader']
        
        # Combine into meta-score
        enhanced['enhanced_confidence'] = self._calculate_meta_confidence(enhanced)
        
        return enhanced
    
    def track_prediction_accuracy(self, ticker, predictions, actual_result):
        """Track which analyzer was most accurate and award XP"""
        
        # Check each analyzer's prediction
        analyzers_correct = {}
        
        # Candle pattern accuracy
        if predictions.get('candle_pattern') == 'bullish' and actual_result > 0:
            analyzers_correct['candle_patterns'] = True
            self.analyzer_performance['candle_patterns']['correct'] += 1
        self.analyzer_performance['candle_patterns']['total'] += 1
        
        # Volume profile accuracy
        if predictions.get('smart_money') and actual_result > 0:
            analyzers_correct['volume_profile'] = True
            self.analyzer_performance['volume_profile']['correct'] += 1
        self.analyzer_performance['volume_profile']['total'] += 1
        
        # Award XP to NEW skills based on accuracy
        if self.xp_tracker:
            # New skills to add to skills.json
            if analyzers_correct.get('candle_patterns'):
                self.xp_tracker.award_skill_xp('candle_reading', 5, 'WIN')
            
            if analyzers_correct.get('volume_profile'):
                self.xp_tracker.award_skill_xp('tape_reading', 5, 'WIN')
                
            if predictions.get('opening_pattern') == 'gap_and_go' and actual_result > 5:
                self.xp_tracker.award_skill_xp('opening_drive_mastery', 8, 'WIN')
    
    def _calculate_meta_confidence(self, features):
        """Combine all signals into meta-confidence score"""
        confidence = 0.5  # Base
        
        # Candle pattern boost
        if features.get('candle_strength', 0) > 0.7:
            confidence += 0.1
            
        # Volume confirmation
        if features.get('smart_money'):
            confidence += 0.15
            
        # Opening pattern alignment
        if features.get('opening_confidence', 0) > 0.7:
            confidence += 0.1
            
        # Relative strength
        if features.get('sector_leader'):
            confidence += 0.05
            
        return min(0.95, confidence)
    
    def get_analyzer_stats(self):
        """Get performance stats for each analyzer"""
        stats = {}
        for analyzer, perf in self.analyzer_performance.items():
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                stats[analyzer] = {
                    'accuracy': accuracy,
                    'total_predictions': perf['total'],
                    'confidence_level': self._get_confidence_level(accuracy, perf['total'])
                }
        return stats
    
    def _get_confidence_level(self, accuracy, sample_size):
        """Determine confidence in analyzer based on performance"""
        if sample_size < 10:
            return 'learning'
        elif accuracy > 0.7 and sample_size > 50:
            return 'highly_confident'
        elif accuracy > 0.6:
            return 'confident'
        elif accuracy > 0.5:
            return 'moderate'
        else:
            return 'needs_improvement'