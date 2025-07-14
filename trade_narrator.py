# trade_narrator.py
# MM-B13: Generate human-readable trade justifications and track biases

import json
import os
from datetime import datetime
import numpy as np


class TradeNarrator:
    """Generate human-readable trade justifications and track biases"""
    
    def __init__(self):
        self.bias_memory = self.load_bias_memory()
        self.error_patterns = {}
        
    def load_bias_memory(self):
        """Load bias scores from file"""
        if os.path.exists('bias_memory.json'):
            with open('bias_memory.json', 'r') as f:
                return json.load(f)
        
        # Initialize default trust levels
        return {
            'volume_pressure': 1.0,
            'float_sense': 1.0,
            'sentiment_spike': 1.0,
            'gap_momentum': 1.0,
            'ml_confidence': 1.0,
            'combo_volume_sentiment': 1.0,
            'combo_gap_float': 1.0
        }
        
    def save_bias_memory(self):
        """Save bias scores to file"""
        with open('bias_memory.json', 'w') as f:
            json.dump(self.bias_memory, f, indent=2)
            
    def generate_trade_justification(self, ticker, action, price, predictions):
        """Generate human-readable trade justification"""
        
        # Extract key metrics
        gap = predictions.get('gap_pct', 0)
        volume_ratio = predictions.get('volume_ratio', predictions.get('premarket_volume', 0) / 100000)
        sentiment = predictions.get('sentiment_score', 0.5)
        confidence = predictions.get('final_confidence', predictions.get('predicted_prob', 0.5))
        freshness_adj = predictions.get('freshness_adjustment', 1.0)
        
        # Build justification components
        components = []
        
        # Gap analysis
        if abs(gap) > 5:
            gap_emoji = "🚀" if gap > 0 else "📉"
            components.append(f"Gap: {gap:+.1f}% {gap_emoji}")
            
        # Volume analysis  
        if volume_ratio > 2:
            vol_emoji = "📊" if volume_ratio > 3 else "📈"
            components.append(f"Volume: {volume_ratio:.1f}x float {vol_emoji}")
            
        # Sentiment
        if sentiment > 0.7:
            components.append("Reddit sentiment spike 📱")
        elif sentiment < 0.3:
            components.append("Negative sentiment⚠️")
            
        # Confidence with freshness
        adj_confidence = confidence * freshness_adj
        components.append(f"Confidence: {confidence:.2f}")
        if freshness_adj < 1.0:
            components.append(f"(adjusted: {adj_confidence:.2f})")
            
        # Build final message
        action_emoji = "🟢" if action == "BUY" else "🔴"
        justification = f"{action_emoji} {action} ${ticker} @ ${price:.2f} - "
        justification += " - ".join(components)
        
        return justification
        
    def log_trade_error(self, ticker, expected_move, actual_move, predictions):
        """Log prediction errors for bias adjustment"""
        
        error_delta = abs(expected_move - actual_move)
        error_direction = "overestimate" if expected_move > actual_move else "underestimate"
        
        # Determine error reason
        error_reason = self.analyze_error_reason(
            expected_move, actual_move, predictions
        )
        
        # Create error log
        error_entry = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'expected_move': expected_move,
            'actual_move': actual_move,
            'error_delta': error_delta,
            'error_direction': error_direction,
            'error_reason': error_reason,
            'best_predictor': predictions.get('best_predictor', 'unknown'),
            'confidence': predictions.get('final_confidence', 0)
        }
        
        # Update error patterns
        if error_reason not in self.error_patterns:
            self.error_patterns[error_reason] = []
        self.error_patterns[error_reason].append(error_entry)
        
        # Adjust bias if high-confidence failure
        if predictions.get('final_confidence', 0) > 0.7 and error_delta > 5:
            self.adjust_predictor_bias(
                predictions.get('best_predictor', 'unknown'),
                error_delta
            )
            
        return error_entry
        
    def analyze_error_reason(self, expected, actual, predictions):
        """Analyze why prediction was wrong"""
        
        # Volume flush - high volume but price reversed
        volume_ratio = predictions.get('volume_ratio', predictions.get('premarket_volume', 0) / 100000)
        if volume_ratio > 3 and actual < 0:
            return "volume_flush"
            
        # Sentiment trap - positive sentiment but price dropped
        if predictions.get('sentiment_score', 0) > 0.7 and actual < expected - 5:
            return "sentiment_trap"
            
        # Gap fade - gap up but closed down
        if predictions.get('gap_pct', 0) > 5 and actual < 0:
            return "gap_fade"
            
        # Momentum failure - everything looked good but failed
        if expected > 5 and actual < -2:
            return "momentum_failure"
            
        # Minor variance - close enough
        if abs(expected - actual) < 2:
            return "minor_variance"
            
        return "unknown_pattern"
        
    def adjust_predictor_bias(self, predictor, error_magnitude):
        """Reduce trust in predictor after failure"""
        
        if predictor in self.bias_memory:
            # Reduce trust based on error magnitude
            penalty = min(0.2, error_magnitude / 50)  # Max 20% penalty
            self.bias_memory[predictor] *= (1 - penalty)
            
            # Don't go below 0.3 trust
            self.bias_memory[predictor] = max(0.3, self.bias_memory[predictor])
            
            print(f"📉 Reduced trust in {predictor} to {self.bias_memory[predictor]:.2f}")
            
            self.save_bias_memory()
            
    def get_adjusted_confidence(self, confidence, predictor):
        """Apply bias adjustment to confidence"""
        bias_factor = self.bias_memory.get(predictor, 1.0)
        return confidence * bias_factor
        
    def get_error_summary(self):
        """Get summary of error patterns"""
        summary = {}
        
        for reason, errors in self.error_patterns.items():
            summary[reason] = {
                'count': len(errors),
                'avg_error': np.mean([e['error_delta'] for e in errors]),
                'recent_examples': [e['ticker'] for e in errors[-3:]]
            }
            
        return summary