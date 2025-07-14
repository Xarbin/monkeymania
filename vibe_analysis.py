# vibe_analysis.py  
# MM-B8: Self-Awareness and Vibe Analysis Module

import pandas as pd
import numpy as np
from datetime import datetime
from skills_engine import get_level_from_xp, get_skill_win_rate
from confidence_calibrator import get_bin_accuracy


class VibeAnalyzer:
    """Analyzes market vibes and Mon Kee's self-awareness"""
    
    def __init__(self):
        self.vibe_thresholds = {
            'strong_aftermarket': 5.0,    # % change threshold
            'strong_premarket': 3.0,      # % change threshold  
            'vibe_divergence': 10.0,      # Difference threshold
            'extreme_volume': 500.0       # Volume change % threshold
        }
    
    def analyze_vibe_effects(self, trade_row):
        """Analyze how vibes affected a specific trade"""
        aftermarket = trade_row.get('aftermarket_move', 0)
        premarket = trade_row.get('early_premarket_move', 0)
        
        analysis = {
            'aftermarket_signal': self._classify_signal(aftermarket, 'aftermarket'),
            'premarket_signal': self._classify_signal(premarket, 'premarket'),
            'vibe_alignment': self._analyze_alignment(aftermarket, premarket),
            'vibe_strength': self._calculate_strength(aftermarket, premarket),
            'vibe_confidence': self._calculate_vibe_confidence(aftermarket, premarket)
        }
        
        return analysis
    
    def _classify_signal(self, value, signal_type):
        """Classify signal strength"""
        threshold = self.vibe_thresholds.get(f'strong_{signal_type}', 5.0)
        abs_value = abs(value)
        
        if abs_value >= threshold:
            direction = "positive" if value > 0 else "negative"
            return f"strong_{direction}"
        elif abs_value >= threshold / 2:
            direction = "positive" if value > 0 else "negative"
            return f"moderate_{direction}"
        else:
            return "weak"
    
    def _analyze_alignment(self, aftermarket, premarket):
        """Analyze alignment between aftermarket and premarket vibes"""
        if aftermarket == 0 and premarket == 0:
            return "no_signal"
        
        if np.sign(aftermarket) == np.sign(premarket):
            magnitude = abs(aftermarket - premarket)
            if magnitude > self.vibe_thresholds['vibe_divergence']:
                return "aligned_divergent"  # Same direction but very different magnitude
            else:
                return "aligned_consistent"  # Same direction, similar magnitude
        else:
            return "conflicted"  # Opposite directions
    
    def _calculate_strength(self, aftermarket, premarket):
        """Calculate overall vibe strength"""
        combined_magnitude = abs(aftermarket) + abs(premarket)
        
        if combined_magnitude >= 15:
            return "extreme"
        elif combined_magnitude >= 8:
            return "strong"
        elif combined_magnitude >= 4:
            return "moderate"
        else:
            return "weak"
    
    def _calculate_vibe_confidence(self, aftermarket, premarket):
        """Calculate confidence based on vibe patterns"""
        base_confidence = 0.5
        
        # Boost for aligned strong signals
        if np.sign(aftermarket) == np.sign(premarket) and abs(aftermarket) > 3 and abs(premarket) > 2:
            base_confidence += 0.2
        
        # Reduce for conflicted signals
        elif np.sign(aftermarket) != np.sign(premarket) and abs(aftermarket) > 2 and abs(premarket) > 2:
            base_confidence -= 0.15
        
        # Boost for extreme single signals
        elif abs(aftermarket) > 8 or abs(premarket) > 5:
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.1), 0.9)
    
    def generate_vibe_insights(self, trade_data):
        """Generate insights about vibe performance"""
        insights = []
        
        if not trade_data:
            return ["No trade data available for vibe analysis"]
        
        # Analyze vibe accuracy
        vibe_results = []
        for trade in trade_data:
            vibe_analysis = self.analyze_vibe_effects(trade)
            actual_result = trade.get('result', 'UNKNOWN')
            
            vibe_results.append({
                'vibe_confidence': vibe_analysis['vibe_confidence'],
                'actual_win': actual_result == 'WIN',
                'vibe_strength': vibe_analysis['vibe_strength'],
                'alignment': vibe_analysis['vibe_alignment']
            })
        
        if vibe_results:
            # Calculate vibe prediction accuracy
            high_vibe_confidence = [r for r in vibe_results if r['vibe_confidence'] > 0.65]
            if high_vibe_confidence:
                high_confidence_accuracy = sum(r['actual_win'] for r in high_vibe_confidence) / len(high_vibe_confidence)
                insights.append(f"🎯 High vibe confidence accuracy: {high_confidence_accuracy*100:.1f}%")
            
            # Analyze alignment patterns
            aligned_trades = [r for r in vibe_results if r['alignment'] == 'aligned_consistent']
            if aligned_trades:
                aligned_accuracy = sum(r['actual_win'] for r in aligned_trades) / len(aligned_trades)
                insights.append(f"✅ Aligned vibe accuracy: {aligned_accuracy*100:.1f}%")
            
            # Analyze conflicted signals
            conflicted_trades = [r for r in vibe_results if r['alignment'] == 'conflicted']
            if conflicted_trades:
                conflicted_accuracy = sum(r['actual_win'] for r in conflicted_trades) / len(conflicted_trades)
                insights.append(f"⚠️ Conflicted vibe accuracy: {conflicted_accuracy*100:.1f}%")
                
                if conflicted_accuracy < 0.4:
                    insights.append("💡 Suggestion: Avoid trades with conflicted vibes")
        
        return insights
    
    def get_vibe_skill_summary(self, skills_xp):
        """Get summary of vibe-related skills"""
        aftermarket_level = get_level_from_xp(skills_xp.get('aftermarket_vibes', 0))
        premarket_level = get_level_from_xp(skills_xp.get('premarket_vibes', 0))
        
        aftermarket_win_rate = get_skill_win_rate('aftermarket_vibes')
        premarket_win_rate = get_skill_win_rate('premarket_vibes')
        
        summary = {
            'aftermarket_vibes': {
                'level': aftermarket_level,
                'xp': skills_xp.get('aftermarket_vibes', 0),
                'win_rate': aftermarket_win_rate,
                'status': self._get_skill_status(aftermarket_level, aftermarket_win_rate)
            },
            'premarket_vibes': {
                'level': premarket_level,
                'xp': skills_xp.get('premarket_vibes', 0), 
                'win_rate': premarket_win_rate,
                'status': self._get_skill_status(premarket_level, premarket_win_rate)
            }
        }
        
        return summary
    
    def _get_skill_status(self, level, win_rate):
        """Determine skill status based on level and win rate"""
        if level >= 50 and win_rate > 0.6:
            return "master"
        elif level >= 25 and win_rate > 0.55:
            return "expert"
        elif level >= 10 and win_rate > 0.5:
            return "proficient"
        elif level >= 5:
            return "developing"
        else:
            return "novice"
    
    def generate_self_assessment(self, skills_xp, recent_trades=None, shadow_results=None):
        """Generate Mon Kee's self-assessment with vibe awareness"""
        assessment_lines = []
        
        # Get vibe skill summary
        vibe_summary = self.get_vibe_skill_summary(skills_xp)
        
        assessment_lines.append("🔮 MON KEE'S VIBE SELF-ASSESSMENT")
        assessment_lines.append("=" * 45)
        
        # Aftermarket vibes assessment
        aftermarket = vibe_summary['aftermarket_vibes']
        assessment_lines.append(f"🌙 Aftermarket Vibes: Level {aftermarket['level']} ({aftermarket['status'].title()})")
        assessment_lines.append(f"   Win Rate: {aftermarket['win_rate']*100:.1f}%")
        
        if aftermarket['level'] >= 25:
            assessment_lines.append("   💭 \"I'm getting really good at reading the evening market mood!\"")
        elif aftermarket['level'] >= 10:
            assessment_lines.append("   💭 \"Starting to understand what happens after hours...\"")
        else:
            assessment_lines.append("   💭 \"Still learning to read aftermarket signals.\"")
        
        # Premarket vibes assessment  
        premarket = vibe_summary['premarket_vibes']
        assessment_lines.append(f"🌅 Premarket Vibes: Level {premarket['level']} ({premarket['status'].title()})")
        assessment_lines.append(f"   Win Rate: {premarket['win_rate']*100:.1f}%")
        
        if premarket['level'] >= 25:
            assessment_lines.append("   💭 \"I can feel the morning market energy before coffee!\"")
        elif premarket['level'] >= 10:
            assessment_lines.append("   💭 \"Getting better at reading premarket momentum...\"")
        else:
            assessment_lines.append("   💭 \"Morning vibes are still a mystery to me.\"")
        
        # Overall vibe assessment
        avg_vibe_level = (aftermarket['level'] + premarket['level']) / 2
        avg_vibe_win_rate = (aftermarket['win_rate'] + premarket['win_rate']) / 2
        
        assessment_lines.append("")
        assessment_lines.append(f"🎭 Overall Vibe Reading: Level {avg_vibe_level:.1f}")
        assessment_lines.append(f"📊 Combined Vibe Accuracy: {avg_vibe_win_rate*100:.1f}%")
        
        # Vibe wisdom based on experience
        if avg_vibe_level >= 50:
            assessment_lines.append("💫 \"I've become one with the market's emotional rhythms...\"")
        elif avg_vibe_level >= 25:
            assessment_lines.append("🧠 \"I'm developing a sixth sense for market sentiment!\"")
        elif avg_vibe_level >= 10:
            assessment_lines.append("📈 \"Learning to trust my gut feelings about market vibes.\"")
        else:
            assessment_lines.append("🤔 \"Still figuring out how to read market emotions.\"")
        
        # Recent performance insights
        if recent_trades:
            vibe_insights = self.generate_vibe_insights(recent_trades)
            if vibe_insights:
                assessment_lines.append("")
                assessment_lines.append("🔍 Recent Vibe Performance:")
                assessment_lines.extend([f"   {insight}" for insight in vibe_insights])
        
        # Shadow portfolio insights
        if shadow_results:
            assessment_lines.append("")
            assessment_lines.append("👁️ Shadow Portfolio Reflection:")
            
            shadow_win_rate = shadow_results.get('win_rate', 0)
            real_win_rate = shadow_results.get('real_win_rate', 0)
            
            if shadow_win_rate > real_win_rate + 0.1:
                assessment_lines.append("   💭 \"My shadow picks are teaching me to trust my instincts more...\"")
            elif shadow_win_rate < real_win_rate - 0.1:
                assessment_lines.append("   💭 \"Good thing I'm selective with my real trades!\"")
            else:
                assessment_lines.append("   💭 \"My real and shadow picks are performing similarly.\"")
        
        # Vibe-based recommendations
        assessment_lines.append("")
        assessment_lines.append("🎯 Vibe Strategy Recommendations:")
        
        if aftermarket['win_rate'] > premarket['win_rate'] + 0.1:
            assessment_lines.append("   • Focus more on aftermarket signals - they're my strength!")
        elif premarket['win_rate'] > aftermarket['win_rate'] + 0.1:
            assessment_lines.append("   • Premarket vibes are working better for me right now.")
        
        if avg_vibe_win_rate < 0.5:
            assessment_lines.append("   • Need to recalibrate my vibe reading - something's off.")
        elif avg_vibe_win_rate > 0.6:
            assessment_lines.append("   • My vibe reading is on fire! Trust the feelings more.")
        
        return "\n".join(assessment_lines)
    
    def detect_vibe_patterns(self, historical_data):
        """Detect patterns in vibe effectiveness"""
        if not historical_data:
            return {}
        
        patterns = {
            'best_vibe_conditions': [],
            'worst_vibe_conditions': [],
            'vibe_correlations': {}
        }
        
        # Analyze when vibes work best
        wins = [trade for trade in historical_data if trade.get('result') == 'WIN']
        losses = [trade for trade in historical_data if trade.get('result') == 'LOSS']
        
        if wins and losses:
            # Calculate average vibe conditions for wins vs losses
            win_after_avg = np.mean([trade.get('aftermarket_move', 0) for trade in wins])
            win_pre_avg = np.mean([trade.get('early_premarket_move', 0) for trade in wins])
            
            loss_after_avg = np.mean([trade.get('aftermarket_move', 0) for trade in losses])
            loss_pre_avg = np.mean([trade.get('early_premarket_move', 0) for trade in losses])
            
            patterns['vibe_correlations'] = {
                'aftermarket_win_bias': win_after_avg - loss_after_avg,
                'premarket_win_bias': win_pre_avg - loss_pre_avg
            }
            
            # Identify best conditions
            if abs(win_after_avg) > abs(loss_after_avg):
                patterns['best_vibe_conditions'].append(f"Aftermarket signals favor wins (avg: {win_after_avg:+.2f}%)")
            
            if abs(win_pre_avg) > abs(loss_pre_avg):
                patterns['best_vibe_conditions'].append(f"Premarket signals favor wins (avg: {win_pre_avg:+.2f}%)")
        
        return patterns


# Global vibe analyzer instance
vibe_analyzer = VibeAnalyzer()


def analyze_vibe_effects(trade_row):
    """Analyze vibe effects for a trade"""
    return vibe_analyzer.analyze_vibe_effects(trade_row)


def get_vibe_skill_summary(skills_xp):
    """Get vibe skills summary"""
    return vibe_analyzer.get_vibe_skill_summary(skills_xp)


def generate_self_assessment(skills_xp, recent_trades=None, shadow_results=None):
    """Generate Mon Kee's self-assessment"""
    return vibe_analyzer.generate_self_assessment(skills_xp, recent_trades, shadow_results)


def detect_vibe_patterns(historical_data):
    """Detect patterns in vibe effectiveness"""
    return vibe_analyzer.detect_vibe_patterns(historical_data)


def generate_vibe_insights(trade_data):
    """Generate insights about vibe performance"""
    return vibe_analyzer.generate_vibe_insights(trade_data)


def calculate_vibe_confidence_boost(aftermarket_move, premarket_move):
    """Calculate confidence boost based on vibe alignment"""
    analyzer = VibeAnalyzer()
    return analyzer._calculate_vibe_confidence(aftermarket_move, premarket_move)


def get_vibe_thresholds():
    """Get vibe classification thresholds"""
    return vibe_analyzer.vibe_thresholds


def update_vibe_thresholds(**kwargs):
    """Update vibe thresholds"""
    vibe_analyzer.vibe_thresholds.update(kwargs)