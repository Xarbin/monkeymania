# monkey_journal.py
# MM-B11: Mon Kee's trading journal with personality

import json
import os
from datetime import datetime
import random


class MonkeyJournal:
    """Mon Kee's trading journal with personality"""
    
    def __init__(self):
        self.journal_file = 'monkey_journal.json'
        self.journal = self.load_journal()
        
    def load_journal(self):
        """Load existing journal or create new"""
        if os.path.exists(self.journal_file):
            with open(self.journal_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_journal(self):
        """Save journal to file"""
        with open(self.journal_file, 'w') as f:
            json.dump(self.journal, f, indent=2)
            
    def get_monkey_mood(self, win_rate, total_pnl):
        """Generate Mon Kee's mood based on performance"""
        moods = {
            'excellent': ["Banana time! 🍌", "King of the jungle! 👑", "Swinging high! 🐒"],
            'good': ["Feeling groovy 🎵", "Nice trades today 😊", "Learning fast! 🧠"],
            'nervous': ["Bit shaky... 😰", "Need more bananas 🍌", "Hmm, tricky market 🤔"],
            'bad': ["Ouch, that hurt 😢", "Back to tree school 📚", "Tomorrow's another day 🌅"]
        }
        
        if win_rate > 0.7 and total_pnl > 100:
            mood_cat = 'excellent'
        elif win_rate > 0.5 and total_pnl > 0:
            mood_cat = 'good'
        elif win_rate > 0.3 or total_pnl > -50:
            mood_cat = 'nervous'
        else:
            mood_cat = 'bad'
            
        return random.choice(moods[mood_cat])
        
    def write_daily_summary(self, trades_today, best_predictor_stats):
        """Write daily trading summary"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate stats
        wins = sum(1 for t in trades_today if t['result'] == 'win')
        losses = len(trades_today) - wins
        total_pnl = sum(t.get('pnl', 0) for t in trades_today)
        win_rate = wins / len(trades_today) if trades_today else 0
        
        # Find best predictor of the day
        predictor_counts = {}
        for trade in trades_today:
            pred = trade.get('best_predictor', 'unknown')
            predictor_counts[pred] = predictor_counts.get(pred, 0) + 1
            
        best_predictor = max(predictor_counts.items(), key=lambda x: x[1])[0] if predictor_counts else 'none'
        
        # Generate mood and score
        mood = self.get_monkey_mood(win_rate, total_pnl)
        score = min(100, max(0, 50 + (win_rate * 50) + (total_pnl / 10)))
        
        # Create summary
        summary = {
            'date': today,
            'summary': f"{wins} wins, {losses} losses. Best predictor: {best_predictor}",
            'mood': mood,
            'score': round(score, 1),
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate * 100, 1),
            'trades': len(trades_today),
            'insights': self.generate_insights(trades_today)
        }
        
        # Add to journal
        if today not in self.journal:
            self.journal[today] = []
        self.journal[today].append(summary)
        
        self.save_journal()
        return summary
        
    def generate_insights(self, trades):
        """Generate trading insights"""
        insights = []
        
        # Time-based analysis
        morning_trades = [t for t in trades if t.get('hour', 12) < 12]
        if morning_trades:
            morning_wr = sum(1 for t in morning_trades if t['result'] == 'win') / len(morning_trades)
            if morning_wr > 0.7:
                insights.append("Morning trades crushed it! 🌅")
            elif morning_wr < 0.3:
                insights.append("Rough mornings... need coffee ☕")
                
        # Predictor analysis
        predictor_success = {}
        for trade in trades:
            pred = trade.get('best_predictor', 'unknown')
            if pred not in predictor_success:
                predictor_success[pred] = {'wins': 0, 'total': 0}
            predictor_success[pred]['total'] += 1
            if trade['result'] == 'win':
                predictor_success[pred]['wins'] += 1
                
        for pred, stats in predictor_success.items():
            wr = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            if wr > 0.8 and stats['total'] >= 3:
                insights.append(f"{pred} is on fire! 🔥")
                
        return insights

    def write_daily_summary_with_slippage(self, trades_today, best_predictor_stats, slippage_data=None):
        """
        Write daily trading summary with slippage awareness
        
        MM-B14.1: Includes slippage calibration status
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate stats
        wins = sum(1 for t in trades_today if t['result'] == 'win')
        losses = len(trades_today) - wins
        total_pnl = sum(t.get('pnl', 0) for t in trades_today)
        win_rate = wins / len(trades_today) if trades_today else 0
        
        # Find best predictor of the day
        predictor_counts = {}
        for trade in trades_today:
            pred = trade.get('best_predictor', 'unknown')
            predictor_counts[pred] = predictor_counts.get(pred, 0) + 1
            
        best_predictor = max(predictor_counts.items(), key=lambda x: x[1])[0] if predictor_counts else 'none'
        
        # Check slippage calibration status
        slippage_adjusted = False
        slippage_mood = ""
        
        if slippage_data:
            if slippage_data['sample_count'] >= 10:
                slippage_adjusted = True
                entry_slip = slippage_data['entry_slip_avg'] * 100
                exit_slip = slippage_data['exit_slip_avg'] * 100
                
                # Add slippage-aware mood
                if abs(entry_slip) + abs(exit_slip) > 3:
                    slippage_mood = "Execution felt slippery 🫣"
                else:
                    slippage_mood = "Fills were smooth today 🎯"
            else:
                remaining = 10 - slippage_data['sample_count']
                slippage_mood = f"Learning from {slippage_data['sample_count']} real trades... need {remaining} more 📝"
        
        # Generate mood and score
        base_mood = self.get_monkey_mood(win_rate, total_pnl)
        
        # Combine moods if slippage awareness active
        if slippage_adjusted:
            mood = f"{base_mood} {slippage_mood}"
        else:
            mood = base_mood
        
        score = min(100, max(0, 50 + (win_rate * 50) + (total_pnl / 10)))
        
        # Adjust score if slippage is hurting
        if slippage_adjusted and total_pnl > 0:
            # If we made money despite slippage, boost confidence
            score += 5
        
        # Create summary
        summary = {
            'date': today,
            'summary': f"{wins} wins, {losses} losses. Best predictor: {best_predictor}",
            'mood': mood,
            'score': round(score, 1),
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate * 100, 1),
            'trades': len(trades_today),
            'insights': self.generate_insights_with_slippage(trades_today, slippage_data)
        }
        
        # Add slippage data if calibrated
        if slippage_adjusted:
            summary['slippage_adjusted'] = True
            summary['slippage_entry_avg'] = round(slippage_data['entry_slip_avg'], 3)
            summary['slippage_exit_avg'] = round(slippage_data['exit_slip_avg'], 3)
            summary['slippage_sample_count'] = slippage_data['sample_count']
        
        # Add to journal
        if today not in self.journal:
            self.journal[today] = []
        self.journal[today].append(summary)
        
        self.save_journal()
        return summary

    def generate_insights_with_slippage(self, trades, slippage_data):
        """Generate trading insights including slippage awareness"""
        insights = self.generate_insights(trades)  # Original insights
        
        if slippage_data and slippage_data['sample_count'] >= 10:
            # Add slippage-specific insights
            total_slip = abs(slippage_data['entry_slip_avg']) + abs(slippage_data['exit_slip_avg'])
            
            if total_slip > 0.04:  # More than 4% round trip
                insights.append("Slippage eating profits! Need better entries 💸")
            elif total_slip < 0.02:  # Less than 2%
                insights.append("Execution is tight! Good fill quality ✨")
            
            # Check if slippage affected win rate
            slippage_trades = [t for t in trades if t.get('slippage_adjusted', False)]
            if slippage_trades:
                slip_wins = sum(1 for t in slippage_trades if t['result'] == 'win')
                slip_wr = slip_wins / len(slippage_trades) if slippage_trades else 0
                
                ideal_wins = sum(1 for t in slippage_trades if t.get('ideal_pnl', 0) > 0)
                ideal_wr = ideal_wins / len(slippage_trades) if slippage_trades else 0
                
                if ideal_wr - slip_wr > 0.1:  # 10% difference
                    insights.append(f"Slippage turned {int((ideal_wr - slip_wr) * 100)}% of winners into losers 😤")
        
        elif slippage_data:
            insights.append(f"📊 Calibrating fills... {slippage_data['sample_count']}/10 real trades logged")
        
        return insights
    
    def write_daily_summary_with_market_insights(self, trades, predictor_stats, slippage_stats, market_analysis):
        # ... existing code ...
        
        # Add market insights to mood
        if market_analysis and 'missed_opportunities' in market_analysis:
            if market_analysis['missed_opportunities']:
                top_miss = market_analysis['missed_opportunities'][0]
                mood_adjustment = f" Kicking myself for missing {top_miss['ticker']} (+{top_miss['return']:.1f}%)!"
                entry['mood'] += mood_adjustment