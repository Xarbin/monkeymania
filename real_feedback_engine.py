# real_feedback_engine.py
# MM-B14.1: Real trade logger and slippage calibration system

import json
import os
from datetime import datetime
import numpy as np


class RealFeedbackEngine:
    """Logs real trades and calculates slippage statistics"""
    
    def __init__(self):
        self.real_trades_file = 'real_trades.json'
        self.slippage_stats_file = 'slippage_stats.json'
        self.real_trades = self.load_real_trades()
        
    def load_real_trades(self):
        """Load existing real trade logs"""
        if os.path.exists(self.real_trades_file):
            with open(self.real_trades_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_real_trades(self):
        """Save real trades to file"""
        with open(self.real_trades_file, 'w') as f:
            json.dump(self.real_trades, f, indent=2)
    
    def log_real_trade(self, trade_data):
        """
        Log a real trade execution
        
        Args:
            trade_data: {
                'ticker': 'AAPL',
                'predicted_entry': 10.00,
                'actual_entry': 10.15,
                'predicted_exit': 11.00,
                'actual_exit': 10.85,
                'shares': 100,
                'date': '2025-06-05',
                'notes': 'High volume at open'
            }
        """
        # Calculate slippage percentages
        entry_slip = (trade_data['actual_entry'] - trade_data['predicted_entry']) / trade_data['predicted_entry']
        exit_slip = (trade_data['actual_exit'] - trade_data['predicted_exit']) / trade_data['predicted_exit']
        
        # Calculate P&L differences
        predicted_pnl = (trade_data['predicted_exit'] - trade_data['predicted_entry']) * trade_data['shares']
        actual_pnl = (trade_data['actual_exit'] - trade_data['actual_entry']) * trade_data['shares']
        
        # Create complete trade record
        trade_record = {
            **trade_data,
            'entry_slippage_pct': round(entry_slip * 100, 3),
            'exit_slippage_pct': round(exit_slip * 100, 3),
            'total_slippage_pct': round((entry_slip + abs(exit_slip)) * 100, 3),
            'predicted_pnl': round(predicted_pnl, 2),
            'actual_pnl': round(actual_pnl, 2),
            'slippage_cost': round(predicted_pnl - actual_pnl, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        self.real_trades.append(trade_record)
        self.save_real_trades()
        
        # Recalculate slippage stats
        self.update_slippage_stats()
        
        print(f"📝 Logged real trade: {trade_data['ticker']}")
        print(f"   Entry slip: {entry_slip*100:.2f}%")
        print(f"   Exit slip: {exit_slip*100:.2f}%")
        print(f"   Cost: ${predicted_pnl - actual_pnl:.2f}")
        
        return trade_record
    
    def get_slippage_stats(self):
        """
        Calculate average slippage statistics
        
        Returns:
            {
                "entry_slip_avg": -0.014,  # -1.4%
                "exit_slip_avg": -0.018,   # -1.8%
                "sample_count": 14,
                "total_cost": -234.50,
                "by_ticker": {...}
            }
        """
        if len(self.real_trades) == 0:
            return {
                "entry_slip_avg": 0,
                "exit_slip_avg": 0,
                "sample_count": 0,
                "total_cost": 0,
                "by_ticker": {}
            }
        
        # Calculate overall averages
        entry_slips = [t['entry_slippage_pct'] / 100 for t in self.real_trades]
        exit_slips = [t['exit_slippage_pct'] / 100 for t in self.real_trades]
        total_cost = sum(t['slippage_cost'] for t in self.real_trades)
        
        # Calculate by ticker
        by_ticker = {}
        for trade in self.real_trades:
            ticker = trade['ticker']
            if ticker not in by_ticker:
                by_ticker[ticker] = {
                    'trades': 0,
                    'avg_entry_slip': 0,
                    'avg_exit_slip': 0,
                    'total_cost': 0
                }
            
            by_ticker[ticker]['trades'] += 1
            by_ticker[ticker]['total_cost'] += trade['slippage_cost']
        
        # Calculate ticker averages
        for ticker in by_ticker:
            ticker_trades = [t for t in self.real_trades if t['ticker'] == ticker]
            by_ticker[ticker]['avg_entry_slip'] = np.mean([t['entry_slippage_pct'] / 100 for t in ticker_trades])
            by_ticker[ticker]['avg_exit_slip'] = np.mean([t['exit_slippage_pct'] / 100 for t in ticker_trades])
        
        stats = {
            "entry_slip_avg": np.mean(entry_slips),
            "exit_slip_avg": np.mean(exit_slips),
            "entry_slip_std": np.std(entry_slips),
            "exit_slip_std": np.std(exit_slips),
            "sample_count": len(self.real_trades),
            "total_cost": round(total_cost, 2),
            "by_ticker": by_ticker,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save stats
        with open(self.slippage_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def update_slippage_stats(self):
        """Recalculate and save slippage statistics"""
        stats = self.get_slippage_stats()
        
        # Award XP if we hit calibration milestone
        if stats['sample_count'] == 10:
            print("🎯 Mon Kee reached slippage calibration milestone!")
            print("   Slippage awareness activated!")
            # This would trigger XP award in the main system
        
        return stats
    
    def get_slippage_adjustment(self, price, is_entry=True):
        """
        Get price adjustment based on learned slippage
        
        Args:
            price: Original predicted price
            is_entry: True for entry, False for exit
            
        Returns:
            Adjusted price with slippage
        """
        stats = self.get_slippage_stats()
        
        # Only apply if we have enough data
        if stats['sample_count'] < 10:
            return price
        
        if is_entry:
            # Entries typically cost more (pay the ask)
            adjustment = 1 + stats['entry_slip_avg']
        else:
            # Exits typically get less (hit the bid)
            adjustment = 1 + stats['exit_slip_avg']
        
        return price * adjustment
    
    def generate_slippage_report(self):
        """Generate detailed slippage analysis report"""
        stats = self.get_slippage_stats()
        
        if stats['sample_count'] == 0:
            return "No real trades logged yet. Log at least 10 trades for calibration."
        
        report = f"""
📊 REAL TRADE SLIPPAGE ANALYSIS
==============================
Sample Size: {stats['sample_count']} trades

Average Slippage:
- Entry: {stats['entry_slip_avg']*100:.2f}% ± {stats['entry_slip_std']*100:.2f}%
- Exit: {stats['exit_slip_avg']*100:.2f}% ± {stats['exit_slip_std']*100:.2f}%
- Round Trip Impact: {(abs(stats['entry_slip_avg']) + abs(stats['exit_slip_avg']))*100:.2f}%

Total Slippage Cost: ${stats['total_cost']:.2f}

By Ticker Performance:
"""
        
        for ticker, data in stats['by_ticker'].items():
            report += f"\n{ticker}:"
            report += f"\n  Trades: {data['trades']}"
            report += f"\n  Avg Entry Slip: {data['avg_entry_slip']*100:.2f}%"
            report += f"\n  Avg Exit Slip: {data['avg_exit_slip']*100:.2f}%"
            report += f"\n  Total Cost: ${data['total_cost']:.2f}"
        
        if stats['sample_count'] < 10:
            report += f"\n\n⚠️ Need {10 - stats['sample_count']} more trades for automatic calibration."
        else:
            report += "\n\n✅ Slippage calibration ACTIVE - Mon Kee is adjusting fills!"
        
        return report
    
    def suggest_improvements(self):
        """Analyze slippage patterns and suggest improvements"""
        stats = self.get_slippage_stats()
        
        if stats['sample_count'] < 10:
            return ["Log more trades to enable slippage analysis"]
        
        suggestions = []
        
        # High entry slippage
        if stats['entry_slip_avg'] < -0.02:  # More than 2% negative
            suggestions.append("Consider using limit orders for entries - paying too much at market")
        
        # High exit slippage
        if stats['exit_slip_avg'] < -0.02:
            suggestions.append("Exit slippage high - try selling earlier or in smaller chunks")
        
        # High variance
        if stats['entry_slip_std'] > 0.03:
            suggestions.append("Entry slippage varies widely - some stocks may be too illiquid")
        
        # Ticker-specific issues
        worst_ticker = None
        worst_cost = 0
        for ticker, data in stats['by_ticker'].items():
            if data['total_cost'] < worst_cost:
                worst_cost = data['total_cost']
                worst_ticker = ticker
        
        if worst_ticker and worst_cost < -100:
            suggestions.append(f"Avoid {worst_ticker} - cost ${abs(worst_cost):.2f} in slippage")
        
        return suggestions


# Global instance
real_feedback_engine = RealFeedbackEngine()


def log_real_trade(trade_data):
    """Log a real trade execution"""
    return real_feedback_engine.log_real_trade(trade_data)


def get_slippage_stats():
    """Get current slippage statistics"""
    return real_feedback_engine.get_slippage_stats()


def get_slippage_adjustment(price, is_entry=True):
    """Get price adjustment based on learned slippage"""
    return real_feedback_engine.get_slippage_adjustment(price, is_entry)


def generate_slippage_report():
    """Generate slippage analysis report"""
    return real_feedback_engine.generate_slippage_report()