# skills_engine.py

import json
import os
import math

SKILL_PATH = "skills.json"
SKILL_ACC_PATH = "data/skill_accuracy.json"

# == LOAD AT MODULE TOP ==
try:
    with open(SKILL_ACC_PATH, "r") as f:
        skill_accuracy = json.load(f)
except:
    skill_accuracy = {}

DEFAULT_SKILLS = {
    "volume_mastery": 0,
    "gap_instinct": 0,
    "float_accuracy": 0,
    "sentiment_iq": 0,
    "fundamental_blend": 0,
    "machine_iq": 0,
    "momentum_mastery": 0,
    "gap_fill_intuition": 0,
    "volume_surge_detector": 0,
    "after_hours_oracle": 0,
    "inspiration": 0,  # Hidden skill - only shows when activated
    "slippage_awareness": 0,  # NEW: Understanding real execution costs
    "execution_mastery": 0    # NEW: Getting better fills over time
     "candle_reading": 0,        # Ability to read candle patterns
    "tape_reading": 0,          # Volume/tape reading mastery  
    "opening_drive_mastery": 0, # Predicting opening 30-min moves
    "relative_strength": 0,     # Sector/market relative analysis
    "pattern_synthesis": 0,     # Combining multiple patterns
    "market_intuition": 0,      # Meta-skill from all analyzers
}

# RuneScape-style leveling (1-99)
def xp_for_level(level):
    """Calculate XP required for a given level using RuneScape formula"""
    if level <= 1:
        return 0
    
    total = 0
    for n in range(1, level):
        total += n + 300 * (2 ** (n / 7))
    
    return int(total / 4)

# Pre-calculate XP thresholds for levels 1-99
LEVEL_XP = {level: xp_for_level(level) for level in range(1, 100)}

def get_level_from_xp(xp):
    """Get current level from XP amount"""
    for level in range(99, 0, -1):
        if xp >= LEVEL_XP[level]:
            return level
    return 1

def get_xp_progress(xp):
    """Get progress towards next level as percentage"""
    level = get_level_from_xp(xp)
    if level >= 99:
        return 100.0
    
    current_level_xp = LEVEL_XP[level]
    next_level_xp = LEVEL_XP[level + 1]
    progress_xp = xp - current_level_xp
    needed_xp = next_level_xp - current_level_xp
    
    return (progress_xp / needed_xp) * 100

# Adjust daily XP gains to reach 99 in ~2 years (730 days)
# Level 99 requires 13,034,431 XP, so ~17,856 XP per day
# With 5-10 trades per day, that's ~2,000-3,500 XP per trade
XP_MULTIPLIER = 500  # Base XP multiplier to reach 99 in 2 years

LEVEL_THRESHOLDS = {
    "volume_mastery": 50,
    "gap_instinct": 50,
    "float_accuracy": 50,
    "sentiment_iq": 50,
    "fundamental_blend": 100,
    "machine_iq": 100
}

def load_skills():
    if os.path.exists(SKILL_PATH):
        with open(SKILL_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_SKILLS.copy()

def save_skills(skills):
    with open(SKILL_PATH, "w") as f:
        json.dump(skills, f, indent=2)

# == FUNCTION: UPDATE ACCURACY ==
def update_skill_accuracy(skill_name, feature_value, trade_result):
    if skill_name not in skill_accuracy:
        skill_accuracy[skill_name] = []
    skill_accuracy[skill_name].append((feature_value, trade_result))
    # Keep only last 500 entries for memory safety
    skill_accuracy[skill_name] = skill_accuracy[skill_name][-500:]

# == FUNCTION: GET WIN RATE ==
def get_skill_win_rate(skill_name):
    history = skill_accuracy.get(skill_name, [])
    if not history:
        return 0.0
    wins = sum(1 for val, result in history if result == "WIN")
    return round(wins / len(history), 3)

# == FUNCTION: SAVE SKILL ACCURACY ==
def save_skill_accuracy():
    os.makedirs(os.path.dirname(SKILL_ACC_PATH), exist_ok=True)
    with open(SKILL_ACC_PATH, "w") as f:
        json.dump(skill_accuracy, f, indent=2)

# == MODIFY award_xp() TO INCLUDE CONFIDENCE ==
def award_xp_new(skill_name, base_amount, result=None, confidence=None, pnl=None):
    """Smart XP system that rewards success and punishes overconfidence"""
    xp_gain = 0
    
    # Scale base amount with XP multiplier
    base_amount = base_amount * XP_MULTIPLIER
    
    if result == "WIN":
        xp_gain = base_amount
        if pnl and pnl > 100:
            xp_gain += 2 * XP_MULTIPLIER  # Big wins get bonus
    elif result == "LOSS":
        if pnl is not None and pnl > -50:  # Small loss
            xp_gain = 1 * XP_MULTIPLIER  # Grace XP - you participated
        elif confidence and confidence > 0.8:
            xp_gain = -3 * XP_MULTIPLIER  # Overconfident and wrong
        else:
            xp_gain = 0  # Regular loss, no XP
    
    xp = load_skills()
    old_xp = xp.get(skill_name, 0)
    xp[skill_name] = max(0, old_xp + xp_gain)  # Never go negative
    save_skills(xp)
    
    # Check for level up
    old_level = get_level_from_xp(old_xp)
    new_level = get_level_from_xp(xp[skill_name])
    
    if new_level > old_level:
        return xp_gain, new_level  # Return tuple with level up info
    
    return xp_gain
    
# ADD THE NEW FUNCTION HERE:
def check_enhanced_skill_unlocks(skill_name, new_level):
    """Special unlocks for enhanced analysis skills"""
    
    unlocks = {
        'candle_reading': {
            10: "Unlocked: Doji pattern recognition",
            25: "Unlocked: Hammer/Shooting Star patterns", 
            50: "Unlocked: Complex pattern combinations",
            75: "Unlocked: Master's intuition - subtle reversal hints",
            99: "Unlocked: Legendary candle whisperer status"
        },
        'tape_reading': {
            10: "Unlocked: Basic volume profile analysis",
            25: "Unlocked: Smart money detection",
            50: "Unlocked: Dark pool activity hints",
            75: "Unlocked: Institutional accumulation patterns",
            99: "Unlocked: Mind of the Market Maker"
        },
        'opening_drive_mastery': {
            10: "Unlocked: Gap-and-go prediction",
            25: "Unlocked: Opening range breakout timing",
            50: "Unlocked: First 5-minute oracle mode",
            75: "Unlocked: Pre-market to open flow analysis",
            99: "Unlocked: Opening Bell Prophet"
        },
        'relative_strength': {
            10: "Unlocked: Basic sector comparison",
            25: "Unlocked: Market beta awareness",
            50: "Unlocked: Sector rotation detection",
            75: "Unlocked: Intermarket analysis",
            99: "Unlocked: Market Symphony Conductor"
        },
        'pattern_synthesis': {
            10: "Unlocked: Dual pattern recognition",
            25: "Unlocked: Triple signal confirmation",
            50: "Unlocked: Complex pattern weaving",
            75: "Unlocked: Meta-pattern awareness",
            99: "Unlocked: Pattern Transcendence"
        },
        'market_intuition': {
            10: "Unlocked: Basic gut feeling bonus",
            25: "Unlocked: Crowd psychology sensor",
            50: "Unlocked: Market mood ring",
            75: "Unlocked: Sixth sense activation",
            99: "Unlocked: One with the Market"
        }
    }
    
    if skill_name in unlocks and new_level in unlocks[skill_name]:
        return unlocks[skill_name][new_level]
    return None

# NEW FUNCTION: Get skill-based feature modifiers
def get_enhanced_skill_modifiers(skills_xp):
    """Get feature modifiers based on enhanced skill levels"""
    modifiers = {}
    
    # Candle reading affects candle pattern confidence
    candle_level = get_level_from_xp(skills_xp.get('candle_reading', 0))
    modifiers['candle_pattern_weight'] = 1.0 + (candle_level / 100)
    
    # Tape reading affects volume analysis weight
    tape_level = get_level_from_xp(skills_xp.get('tape_reading', 0))
    modifiers['volume_analysis_weight'] = 1.0 + (tape_level / 100)
    
    # Opening drive affects first hour predictions
    opening_level = get_level_from_xp(skills_xp.get('opening_drive_mastery', 0))
    modifiers['opening_confidence_boost'] = 1.0 + (opening_level / 50)
    
    # Pattern synthesis affects combining signals
    synthesis_level = get_level_from_xp(skills_xp.get('pattern_synthesis', 0))
    modifiers['signal_combination_bonus'] = 0.1 * (synthesis_level / 25)
    
    return modifiers
    
def award_slippage_milestone_xp():
    """
    Award XP when slippage calibration milestones are reached
    
    MM-B14.1: Rewards Mon Kee for learning from real trades
    """
    skills = load_skills()
    
    # Big XP for first calibration
    skills["slippage_awareness"] = skills.get("slippage_awareness", 0) + (25 * XP_MULTIPLIER)
    skills["execution_mastery"] = skills.get("execution_mastery", 0) + (15 * XP_MULTIPLIER)
    
    save_skills(skills)
    
    # Check for level ups
    slip_level = get_level_from_xp(skills["slippage_awareness"])
    exec_level = get_level_from_xp(skills["execution_mastery"])
    
    log = f"🎯 Slippage Calibration Milestone!\n"
    log += f"• 📊 Slippage Awareness: +{25 * XP_MULTIPLIER} XP (Level {slip_level})\n"
    log += f"• ⚡ Execution Mastery: +{15 * XP_MULTIPLIER} XP (Level {exec_level})\n"
    log += f"• Mon Kee now adjusts fills based on real execution data!"
    
    return log

def update_slippage_skills(slippage_impact_percent):
    """
    Update skills based on slippage performance
    
    Args:
        slippage_impact_percent: How much slippage affected P&L (negative is bad)
    """
    skills = load_skills()
    
    # Reward for dealing with high slippage successfully
    if slippage_impact_percent < -5:  # High slippage
        # Still made profit despite slippage = good execution
        xp_gain = 5 * XP_MULTIPLIER
    elif slippage_impact_percent < -2:  # Moderate slippage
        xp_gain = 3 * XP_MULTIPLIER
    else:  # Low slippage
        xp_gain = 1 * XP_MULTIPLIER
    
    skills["execution_mastery"] = skills.get("execution_mastery", 0) + xp_gain
    save_skills(skills)
    
    return xp_gain
    # Original award_xp function for backward compatibility
def award_xp(skills, trade):
    log = "📈 Skill Report:\n"

    if trade.get("volume"):
        xp_gain = 5 * XP_MULTIPLIER
        skills["volume_mastery"] += xp_gain
        level = get_level_from_xp(skills["volume_mastery"])
        log += f"• 📊 Volume Mastery: +{xp_gain} XP (Level {level})\n"
        
        # Check for volume surge
        if trade.get("volume_surge"):
            xp_gain = 6 * XP_MULTIPLIER
            skills["volume_surge_detector"] = skills.get("volume_surge_detector", 0) + xp_gain
            level = get_level_from_xp(skills["volume_surge_detector"])
            log += f"• 🚀 Volume Surge Detector: +{xp_gain} XP (Level {level})\n"
    
    if trade.get("gap"):
        xp_gain = 3 * XP_MULTIPLIER
        skills["gap_instinct"] += xp_gain
        level = get_level_from_xp(skills["gap_instinct"])
        log += f"• ⚡ Gap % Instinct: +{xp_gain} XP (Level {level})\n"
        
        # Check for gap fill tracking
        if trade.get("gap_magnitude", 0) > 5:
            xp_gain = 3 * XP_MULTIPLIER
            skills["gap_fill_intuition"] = skills.get("gap_fill_intuition", 0) + xp_gain
            level = get_level_from_xp(skills["gap_fill_intuition"])
            log += f"• 🎯 Gap Fill Intuition: +{xp_gain} XP (Level {level})\n"
    
    if trade.get("float"):
        xp_gain = 4 * XP_MULTIPLIER
        skills["float_accuracy"] += xp_gain
        level = get_level_from_xp(skills["float_accuracy"])
        log += f"• 🪙 Float Accuracy: +{xp_gain} XP (Level {level})\n"
    
    if trade.get("sentiment"):
        xp_gain = 5 * XP_MULTIPLIER
        skills["sentiment_iq"] += xp_gain
        level = get_level_from_xp(skills["sentiment_iq"])
        log += f"• 💬 Sentiment IQ: +{xp_gain} XP (Level {level})\n"
    
    if trade.get("momentum_aligned"):
        xp_gain = 4 * XP_MULTIPLIER
        skills["momentum_mastery"] = skills.get("momentum_mastery", 0) + xp_gain
        level = get_level_from_xp(skills["momentum_mastery"])
        log += f"• 📈 Momentum Mastery: +{xp_gain} XP (Level {level})\n"

    if trade.get("volume") or trade.get("gap") or trade.get("float"):
        xp_gain = 5 * XP_MULTIPLIER
        skills["fundamental_blend"] += xp_gain
        level = get_level_from_xp(skills["fundamental_blend"])
        log += f"• 🧮 Fundamental Blend: +{xp_gain} XP (Level {level})\n"

    # Calculate Machine IQ as average of all other skills
    all_skills = ["volume_mastery", "gap_instinct", "float_accuracy", "sentiment_iq", 
                  "fundamental_blend", "momentum_mastery", "gap_fill_intuition", 
                  "volume_surge_detector", "after_hours_oracle"]
    total_xp = sum(skills.get(s, 0) for s in all_skills) // len(all_skills)
    skills["machine_iq"] = total_xp
    level = get_level_from_xp(total_xp)

    log += f"• 🧠 Machine IQ: Level {level} ({total_xp:,} XP)\n"

    save_skills(skills)
    return log