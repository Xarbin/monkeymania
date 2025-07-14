# xp_tracker.py
# MM-B8: XP and Skill Management Module
# UPDATED: MM-B11 get_top_skills implementation

import json
import os
from skills_engine import (
    load_skills, save_skills, get_level_from_xp, 
    get_xp_progress, award_xp_new, update_skill_accuracy,
    xp_for_level, LEVEL_XP
)
from gui_handlers import log_system_message, format_xp


class XPTracker:
    """Centralized XP and skill management"""
    
    def __init__(self):
        self.skills_xp = load_skills()
        self.skill_display_names = {
            "volume_mastery": "📊 Volume Mastery",
            "gap_instinct": "⚡ Gap Instinct", 
            "float_accuracy": "🪙 Float Accuracy",
            "sentiment_iq": "💬 Sentiment IQ",
            "fundamental_blend": "🧮 Fundamental Blend",
            "machine_iq": "🧠 Machine IQ",
            "momentum_mastery": "📈 Momentum Mastery",
            "gap_fill_intuition": "🎯 Gap Fill Intuition",
            "volume_surge_detector": "🚀 Volume Surge Detector",
            "after_hours_oracle": "🌙 After Hours Oracle",
            "aftermarket_vibes": "🌙 Aftermarket Vibes",
            "premarket_vibes": "🌅 Premarket Vibes",
            "inspiration": "✨ Inspiration"
            "candle_reading": "🕯️ Candle Reading",
            "tape_reading": "📊 Tape Reading",
            "opening_drive_mastery": "🚀 Opening Drive",
            "relative_strength": "💪 Relative Strength",
            "pattern_synthesis": "🧩 Pattern Synthesis",
            "market_intuition": "🔮 Market Intuition",
        }
    
    def get_top_skills(self, limit=5):
        """
        MM-B11: Get top skills by XP for GUI display
        
        Args:
            limit: Number of top skills to return
            
        Returns:
            List of skill info dicts with progress data
        """
        top_skills = []
        
        # Sort skills by current XP
        sorted_skills = sorted(
            self.skills_xp.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for skill_name, skill_xp in sorted_skills[:limit]:
            current_level = get_level_from_xp(skill_xp)
            
            # Calculate XP for current and next level
            if current_level >= 99:
                # Max level reached
                current_level_xp = LEVEL_XP[99]
                next_level_xp = LEVEL_XP[99]
                level_progress = skill_xp - current_level_xp
                level_requirement = 1  # Avoid division by zero
                progress_percent = 100.0
            else:
                current_level_xp = LEVEL_XP[current_level]
                next_level_xp = LEVEL_XP[current_level + 1]
                level_progress = skill_xp - current_level_xp
                level_requirement = next_level_xp - current_level_xp
                progress_percent = (level_progress / level_requirement) * 100
            
            top_skills.append({
                'name': self.skill_display_names.get(skill_name, skill_name),
                'skill_key': skill_name,
                'current_xp': level_progress,
                'next_level_xp': level_requirement,
                'total_xp': skill_xp,
                'level': current_level,
                'progress_percent': progress_percent
            })
        
        return top_skills
    
    def get_total_xp(self):
        """Get total XP across all skills"""
        return sum(self.skills_xp.values())
    
    def get_average_level(self):
        """Get average skill level"""
        if not self.skills_xp:
            return 1
        total_levels = sum(get_level_from_xp(xp) for xp in self.skills_xp.values())
        return total_levels // len(self.skills_xp)
    
    def get_skill_level(self, skill_name):
        """Get level for specific skill"""
        xp = self.skills_xp.get(skill_name, 0)
        return get_level_from_xp(xp)
    
    def get_skill_progress(self, skill_name):
        """Get progress percentage to next level"""
        xp = self.skills_xp.get(skill_name, 0)
        return get_xp_progress(xp)
    
    def award_skill_xp(self, skill_name, base_amount, result=None, confidence=None, pnl=None):
        """Award XP to specific skill with logging"""
        if skill_name not in self.skills_xp:
            # If it's a new enhanced skill, add it
            if skill_name in ['candle_reading', 'tape_reading', 'opening_drive_mastery', 
                             'relative_strength', 'pattern_synthesis', 'market_intuition']:
                self.skills_xp[skill_name] = 0
                save_skills(self.skills_xp)
            else:
                return f"❌ Unknown skill: {skill_name}"
        
        old_level = self.get_skill_level(skill_name)
        xp_result = award_xp_new(skill_name, base_amount, result, confidence, pnl)
        
        # Reload skills after XP award
        self.skills_xp = load_skills()
        new_level = self.get_skill_level(skill_name)
        
        # Handle the extended return value with unlock message
        if isinstance(xp_result, tuple) and len(xp_result) == 3:
            xp_delta, level, unlock_message = xp_result
            return f"🎉 LEVEL UP! {self.skill_display_names.get(skill_name, skill_name)} reached Level {level}!\n✨ {unlock_message}"
        elif isinstance(xp_result, tuple):
            xp_delta, level = xp_result
            return f"🎉 LEVEL UP! {self.skill_display_names.get(skill_name, skill_name)} reached Level {level}!"
        else:
            # Original code continues here (from line ~142 in your file):
            xp_delta = xp_result
            if xp_delta > 0:
                return f"✅ {self.skill_display_names.get(skill_name, skill_name)}: +{xp_delta} XP"
            elif xp_delta < 0:
                return f"⚠️ {self.skill_display_names.get(skill_name, skill_name)}: {xp_delta} XP (overconfident penalty)"
            else:
                return f"➖ {self.skill_display_names.get(skill_name, skill_name)}: No XP change"
        
    def update_skill_accuracy(self, skill_name, feature_value, result):
        """Update skill accuracy tracking"""
        update_skill_accuracy(skill_name, feature_value, result)
    
    def get_skills_summary(self):
        """Get formatted skills summary"""
        summary = []
        
        # Sort skills by XP (highest first)
        sorted_skills = sorted(
            self.skills_xp.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for skill, xp in sorted_skills:
            # Skip inspiration if not unlocked
            if skill == "inspiration" and xp == 0:
                continue
                
            level = get_level_from_xp(xp)
            display_name = self.skill_display_names.get(skill, skill.replace('_', ' ').title())
            
            summary.append({
                'skill': skill,
                'display_name': display_name,
                'level': level,
                'xp': xp,
                'progress': get_xp_progress(xp)
            })
        
        return summary
    
    def update_progress_ui(self, progress_bar, tooltip_callback=None):
        """Update UI progress bar and tooltip"""
        total_xp = self.get_total_xp()
        xp_percent = total_xp % 100
        avg_level = self.get_average_level()
        
        if progress_bar:
            progress_bar.setValue(xp_percent)
            progress_bar.setFormat(f"Skill Growth - {xp_percent}/100 XP")
            
            if tooltip_callback:
                tooltip_text = f"Total XP: {total_xp:,}\nAverage Level: {avg_level}"
                tooltip_callback(tooltip_text)
    
    def generate_skill_report(self, log_widget=None):
        """Generate comprehensive skill report"""
        report_lines = []
        report_lines.append("📊 SKILL PROGRESSION REPORT")
        report_lines.append("=" * 40)
        
        total_xp = self.get_total_xp()
        avg_level = self.get_average_level()
        
        report_lines.append(f"Total XP: {format_xp(total_xp)}")
        report_lines.append(f"Average Level: {avg_level}")
        report_lines.append("")
        
        # Get skill summary
        skills = self.get_skills_summary()
        
        report_lines.append("🏆 TOP SKILLS:")
        for i, skill in enumerate(skills[:5], 1):
            progress = skill['progress']
            report_lines.append(
                f"{i}. {skill['display_name']}: Level {skill['level']} "
                f"({format_xp(skill['xp'])}) - {progress:.1f}% to next"
            )
        
        if len(skills) > 5:
            report_lines.append(f"... and {len(skills) - 5} more skills")
        
        # Find milestone achievements
        milestones = []
        for skill in skills:
            level = skill['level']
            if level >= 99:
                milestones.append(f"🏆 {skill['display_name']} - GRANDMASTER!")
            elif level >= 75:
                milestones.append(f"👑 {skill['display_name']} - Master Level!")
            elif level >= 50:
                milestones.append(f"⭐ {skill['display_name']} - Expert Level!")
            elif level >= 25:
                milestones.append(f"📈 {skill['display_name']} - Advanced Level!")
        
        if milestones:
            report_lines.append("")
            report_lines.append("🎖️ ACHIEVEMENTS:")
            report_lines.extend(milestones)
        
        report_text = "\n".join(report_lines)
        
        if log_widget:
            log_system_message(log_widget, "Generated skill progression report", "INFO")
        
        return report_text
    
    def calculate_feature_attribution(self, features):
        """Calculate which feature was most influential"""
        feature_scores = {}
        
        # Traditional features
        if 'gap_pct' in features:
            feature_scores['gap_instinct'] = abs(features['gap_pct']) * 0.1
        
        if 'premarket_volume' in features:
            feature_scores['volume_mastery'] = features['premarket_volume'] / 1e6
        
        if 'float' in features:
            feature_scores['float_accuracy'] = 10 / max(features['float'] / 1e6, 1)
        
        # Vibe features (MM-B7)
        if 'aftermarket_move' in features:
            feature_scores['aftermarket_vibes'] = abs(features['aftermarket_move']) * 0.2
        
        if 'early_premarket_move' in features:
            feature_scores['premarket_vibes'] = abs(features['early_premarket_move']) * 0.2
        
        # Advanced features
        if 'momentum_aligned' in features and features['momentum_aligned'] > 0:
            feature_scores['momentum_mastery'] = 2.0
        
        if 'volume_surge' in features and features['volume_surge'] > 0:
            feature_scores['volume_surge_detector'] = 3.0
        
        if 'gap_magnitude' in features and features['gap_magnitude'] > 15:
            feature_scores['gap_fill_intuition'] = features['gap_magnitude'] * 0.1
        
        # Find best feature
        if feature_scores:
            best_feature = max(feature_scores, key=feature_scores.get)
            return best_feature, feature_scores[best_feature]
        
        return None, 0
    
    def process_trade_result(self, features, result_str, confidence=None, pnl=None, log_widget=None):
        """Process trade result and award appropriate XP"""
        xp_logs = []
        
        # Determine key feature
        best_feature, feature_strength = self.calculate_feature_attribution(features)
        
        # Base XP amounts for each skill
        skill_base_xp = {
            'gap_instinct': 3,
            'volume_mastery': 5, 
            'float_accuracy': 4,
            'momentum_mastery': 4,
            'volume_surge_detector': 6,
            'gap_fill_intuition': 3,
            'after_hours_oracle': 5,
            'aftermarket_vibes': 4,
            'premarket_vibes': 4
        }
        
        # Award XP to applicable skills
        skills_used = []
        
        # Traditional skill XP
        if features.get('gap_pct', 0) != 0:
            self.update_skill_accuracy('gap_instinct', features['gap_pct'], result_str)
            skills_used.append('gap_instinct')
        
        if features.get('premarket_volume', 0) > 0:
            self.update_skill_accuracy('volume_mastery', features['premarket_volume'], result_str)
            skills_used.append('volume_mastery')
        
        if features.get('float', 0) > 0:
            self.update_skill_accuracy('float_accuracy', features['float'], result_str)
            skills_used.append('float_accuracy')
        
        # Vibe skill XP (MM-B7)
        if features.get('aftermarket_move', 0) != 0:
            self.update_skill_accuracy('aftermarket_vibes', features['aftermarket_move'], result_str)
            skills_used.append('aftermarket_vibes')
        
        if features.get('early_premarket_move', 0) != 0:
            self.update_skill_accuracy('premarket_vibes', features['early_premarket_move'], result_str)
            skills_used.append('premarket_vibes')
        
        # Advanced skill XP
        if features.get('momentum_aligned', 0) > 0:
            self.update_skill_accuracy('momentum_mastery', 1.0, result_str)
            skills_used.append('momentum_mastery')
        
        if features.get('volume_surge', 0) > 0:
            self.update_skill_accuracy('volume_surge_detector', 1.0, result_str)
            skills_used.append('volume_surge_detector')
        
        # Award XP to all used skills
        for skill in skills_used:
            base_xp = skill_base_xp.get(skill, 3)
            xp_log = self.award_skill_xp(skill, base_xp, result_str, confidence, pnl)
            xp_logs.append(xp_log)
            
            if log_widget:
                log_system_message(log_widget, xp_log)
        
        # Award bonus XP to best feature if win
        if result_str == "WIN" and best_feature in skills_used:
            bonus_log = self.award_skill_xp(best_feature, 2, "WIN", confidence, pnl)
            xp_logs.append(f"🎯 {self.skill_display_names.get(best_feature, best_feature)} was key → {bonus_log}")
            
            if log_widget:
                log_system_message(
                    log_widget, 
                    f"{self.skill_display_names.get(best_feature, best_feature)} was key to this trade → +bonus XP"
                )
        
        return xp_logs
    
    def get_skill_biases(self):
        """Get XP-based biases for features"""
        return {
            'gap_bias': 1 + (self.skills_xp.get('gap_instinct', 0) / 1000),
            'volume_bias': 1 + (self.skills_xp.get('volume_mastery', 0) / 1000),
            'float_bias': 1 - (self.skills_xp.get('float_accuracy', 0) / 1000),
            'aftermarket_bias': 1 + (self.skills_xp.get('aftermarket_vibes', 0) / 1000),
            'premarket_bias': 1 + (self.skills_xp.get('premarket_vibes', 0) / 1000)
        }
    
    def check_inspiration_unlock(self, shadow_results=None):
        """Check if inspiration should be unlocked based on shadow performance"""
        current_inspiration = self.skills_xp.get('inspiration', 0)
        
        if current_inspiration > 0:
            return False, ""  # Already unlocked
        
        # Inspiration triggers (from original implementation)
        if shadow_results:
            shadow_win_rate = shadow_results.get('win_rate', 0)
            real_win_rate = shadow_results.get('real_win_rate', 0)
            big_winner_missed = shadow_results.get('big_winner_missed', False)
            
            inspiration_triggered = False
            inspiration_reason = ""
            
            if shadow_win_rate > 0.6 and shadow_win_rate > real_win_rate + 0.2:
                inspiration_triggered = True
                inspiration_reason = "shadow portfolio significantly outperformed!"
            
            if big_winner_missed:
                inspiration_triggered = True
                inspiration_reason = shadow_results.get('big_winner_reason', "missed a big winner!")
            
            if inspiration_triggered:
                from skills_engine import XP_MULTIPLIER
                inspiration_xp = 10 * XP_MULTIPLIER
                self.skills_xp['inspiration'] = inspiration_xp
                save_skills(self.skills_xp)
                
                return True, inspiration_reason
        
        return False, ""
    
    def save_state(self):
        """Save current XP state"""
        save_skills(self.skills_xp)
    
    def reload_state(self):
        """Reload XP state from disk"""
        self.skills_xp = load_skills()


# Global XP tracker instance
xp_tracker = XPTracker()


def get_xp_tracker():
    """Get global XP tracker instance"""
    return xp_tracker


def update_all_skills_ui(gui):
    """Update all skill-related UI elements"""
    tracker = get_xp_tracker()
    
    # Update progress bar
    if hasattr(gui, 'progress_bar'):
        tracker.update_progress_ui(
            gui.progress_bar,
            lambda tooltip: gui.progress_bar.setToolTip(tooltip) if hasattr(gui.progress_bar, 'setToolTip') else None
        )
    
    # Update cash display (includes monkey art which shows levels)
    if hasattr(gui, 'update_cash_display'):
        gui.update_cash_display()


def get_total_xp():
    """Get total XP across all skills"""
    return get_xp_tracker().get_total_xp()


def get_average_level():
    """Get average skill level"""
    return get_xp_tracker().get_average_level()


def process_trade_xp(features, result_str, confidence=None, pnl=None, log_widget=None):
    """Process trade result and award XP"""
    return get_xp_tracker().process_trade_result(features, result_str, confidence, pnl, log_widget)


def get_skill_biases():
    """Get XP-based feature biases"""
    return get_xp_tracker().get_skill_biases()


def generate_skills_report(log_widget=None):
    """Generate skills report"""
    return get_xp_tracker().generate_skill_report(log_widget)