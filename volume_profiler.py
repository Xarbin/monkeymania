# analysis/volume_profiler.py
"""Advanced volume analysis for Mon Kee"""

import numpy as np
from typing import Dict, List, Tuple

class VolumeProfiler:
    """Analyzes volume patterns to identify smart money"""
    
    def __init__(self):
        self.volume_patterns = {}
    
    def analyze_volume_signature(self,
                                pre_volume: float,
                                regular_volume: float,
                                post_volume: float,
                                avg_volume: float,
                                price_data: Dict) -> Dict:
        """Decode the volume story"""
        
        # Calculate volume ratios
        pre_ratio = pre_volume / avg_volume if avg_volume > 0 else 0
        regular_ratio = regular_volume / avg_volume if avg_volume > 0 else 0
        post_ratio = post_volume / avg_volume if avg_volume > 0 else 0
        
        # Volume momentum
        volume_acceleration = regular_ratio / pre_ratio if pre_ratio > 0 else 0
        volume_exhaustion = post_ratio / regular_ratio if regular_ratio > 0 else 0
        
        # Analyze patterns
        analysis = {
            'pre_market_interest': self._classify_interest(pre_ratio),
            'volume_surge': regular_ratio > 3,
            'volume_climax': regular_ratio > 5 and volume_exhaustion < 0.3,
            'smart_money_accumulation': self._detect_accumulation(
                pre_ratio, regular_ratio, price_data
            ),
            'distribution_signal': self._detect_distribution(
                regular_ratio, post_ratio, price_data
            ),
            'institutional_presence': pre_ratio > 0.15,  # >15% in pre-market
            'retail_frenzy': regular_ratio > 5 and pre_ratio < 0.05,
            'volume_trend': self._calculate_volume_trend(
                pre_ratio, regular_ratio, post_ratio
            )
        }
        
        # Add specific insights
        analysis['interpretation'] = self._interpret_volume(analysis)
        analysis['confidence'] = self._calculate_volume_confidence(analysis)
        
        return analysis
    
    def _classify_interest(self, ratio: float) -> str:
        """Classify pre-market interest level"""
        if ratio > 0.3:
            return 'extreme'
        elif ratio > 0.15:
            return 'high'
        elif ratio > 0.05:
            return 'moderate'
        else:
            return 'low'
    
    def _detect_accumulation(self, pre_ratio: float, 
                           regular_ratio: float, 
                           price_data: Dict) -> bool:
        """Detect smart money accumulation patterns"""
        # Quiet accumulation: moderate volume, price stability
        quiet_accumulation = (
            0.05 < pre_ratio < 0.15 and
            regular_ratio < 2 and
            price_data.get('close', 0) > price_data.get('open', 0)
        )
        
        # Aggressive accumulation: high volume, strong close
        aggressive_accumulation = (
            pre_ratio > 0.15 and
            regular_ratio > 2 and
            price_data.get('close', 0) > price_data.get('high', 1) * 0.95
        )
        
        return quiet_accumulation or aggressive_accumulation
    
    def _detect_distribution(self, regular_ratio: float,
                           post_ratio: float,
                           price_data: Dict) -> bool:
        """Detect distribution patterns"""
        return (
            regular_ratio > 3 and
            post_ratio < 0.1 and
            price_data.get('close', 0) < price_data.get('open', 1)
        )
    
    def _calculate_volume_trend(self, pre: float, 
                              regular: float, 
                              post: float) -> str:
        """Determine volume trend throughout day"""
        if pre < regular and regular < post:
            return 'accelerating'
        elif pre > regular and regular > post:
            return 'decelerating'
        elif regular > pre and regular > post:
            return 'climax'
        else:
            return 'mixed'
    
    def _interpret_volume(self, analysis: Dict) -> str:
        """Generate human-readable volume interpretation"""
        if analysis['smart_money_accumulation']:
            return "Institutional accumulation detected"
        elif analysis['distribution_signal']:
            return "Distribution after run-up - caution"
        elif analysis['volume_climax']:
            return "Volume climax - potential reversal"
        elif analysis['retail_frenzy']:
            return "Retail-driven volume spike"
        else:
            return "Normal volume pattern"
    
    def _calculate_volume_confidence(self, analysis: Dict) -> float:
        """Calculate confidence in volume analysis"""
        confidence = 0.5
        
        if analysis['institutional_presence']:
            confidence += 0.2
        if analysis['volume_surge'] and not analysis['retail_frenzy']:
            confidence += 0.15
        if analysis['smart_money_accumulation']:
            confidence += 0.15
        if analysis['distribution_signal']:
            confidence -= 0.2
            
        return max(0.1, min(0.95, confidence))