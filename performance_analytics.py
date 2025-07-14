# performance_analytics.py
# MM-B15: Performance tracking and pattern analysis system

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

class PerformanceAnalytics:
    """Track and analyze trading performance without cluttering the GUI"""
    
    def __init__(self):
        self.data_file = 'performance_data.json'
        self.pattern_stats = self.load_performance_data()
        self.session_cache = {
            'trades_analyzed': 0,
            'patterns_seen': defaultdict(int),
            'market_conditions': []
        }
        
    def load_performance_data(self):
        """Load existing performance data or create new structure"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        
        # Initialize performance tracking structure
        return {
            'pattern_performance': {
                'gap_and_go': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []},
                'momentum_surge': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []},
                'volume_spike': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []},
                'breakout': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []},
                'reversal': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []},
                'consolidation': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []},
                'unknown': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': []}
            },
            'time_performance': {
                'premarket': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'morning': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'midday': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'afternoon': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'power_hour': {'wins': 0, 'losses': 0, 'total_pnl': 0}
            },
            'market_conditions': {
                'trending_up': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'trending_down': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'choppy': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'low_volatility': {'wins': 0, 'losses': 0, 'total_pnl': 0},
                'high_volatility': {'wins': 0, 'losses': 0, 'total_pnl': 0}
            },
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'total_trades_analyzed': 0,
                'tracking_start_date': datetime.now().isoformat()
            }
        }
    
    def save_performance_data(self):
        """Save performance data to file"""
        self.pattern_stats['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.data_file, 'w') as f:
            json.dump(self.pattern_stats, f, indent=2)
    
    def record_trade_result(self, trade_data):
        """
        Record a completed trade for analysis
        
        Expected trade_data format:
        {
            'timestamp': datetime,
            'symbol': 'AAPL',
            'pattern': 'gap_and_go',
            'result': 'win' or 'loss',
            'pnl': 125.50,
            'entry_price': 150.00,
            'exit_price': 151.25,
            'shares': 100,
            'hold_time_minutes': 45,
            'market_condition': 'trending_up',
            'slippage_adjusted': True
        }
        """
        # Determine pattern category
        pattern = trade_data.get('pattern', 'unknown')
        if pattern not in self.pattern_stats['pattern_performance']:
            pattern = 'unknown'
        
        # Update pattern performance
        pattern_data = self.pattern_stats['pattern_performance'][pattern]
        if trade_data['result'] == 'win':
            pattern_data['wins'] += 1
        else:
            pattern_data['losses'] += 1
        
        pattern_data['total_pnl'] += trade_data['pnl']
        
        # Store trade summary (keep only last 100 trades per pattern)
        trade_summary = {
            'timestamp': trade_data['timestamp'].isoformat() if isinstance(trade_data['timestamp'], datetime) else trade_data['timestamp'],
            'symbol': trade_data['symbol'],
            'pnl': trade_data['pnl'],
            'hold_time': trade_data.get('hold_time_minutes', 0)
        }
        
        pattern_data['trades'].append(trade_summary)
        if len(pattern_data['trades']) > 100:
            pattern_data['trades'] = pattern_data['trades'][-100:]
        
        # Update time-based performance
        hour = trade_data['timestamp'].hour if isinstance(trade_data['timestamp'], datetime) else datetime.fromisoformat(trade_data['timestamp']).hour
        time_slot = self._get_time_slot(hour)
        
        time_data = self.pattern_stats['time_performance'][time_slot]
        if trade_data['result'] == 'win':
            time_data['wins'] += 1
        else:
            time_data['losses'] += 1
        time_data['total_pnl'] += trade_data['pnl']
        
        # Update market condition performance
        market_condition = trade_data.get('market_condition', 'unknown')
        if market_condition in self.pattern_stats['market_conditions']:
            condition_data = self.pattern_stats['market_conditions'][market_condition]
            if trade_data['result'] == 'win':
                condition_data['wins'] += 1
            else:
                condition_data['losses'] += 1
            condition_data['total_pnl'] += trade_data['pnl']
        
        # Update metadata
        self.pattern_stats['metadata']['total_trades_analyzed'] += 1
        
        # Update session cache
        self.session_cache['trades_analyzed'] += 1
        self.session_cache['patterns_seen'][pattern] += 1
        
        # Save periodically (every 10 trades)
        if self.session_cache['trades_analyzed'] % 10 == 0:
            self.save_performance_data()
    
    def _get_time_slot(self, hour):
        """Categorize trading hour into time slots"""
        if hour < 9:
            return 'premarket'
        elif 9 <= hour < 11:
            return 'morning'
        elif 11 <= hour < 14:
            return 'midday'
        elif 14 <= hour < 15:
            return 'afternoon'
        else:
            return 'power_hour'
    
    def get_pattern_stats(self, pattern=None):
        """Get performance statistics for a specific pattern or all patterns"""
        if pattern:
            if pattern in self.pattern_stats['pattern_performance']:
                return self._calculate_pattern_metrics(pattern)
            return None
        
        # Return stats for all patterns
        all_stats = {}
        for p in self.pattern_stats['pattern_performance']:
            all_stats[p] = self._calculate_pattern_metrics(p)
        return all_stats
    
    def _calculate_pattern_metrics(self, pattern):
        """Calculate detailed metrics for a pattern"""
        data = self.pattern_stats['pattern_performance'][pattern]
        total_trades = data['wins'] + data['losses']
        
        if total_trades == 0:
            return {
                'pattern': pattern,
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'expectancy': 0
            }
        
        win_rate = data['wins'] / total_trades if total_trades > 0 else 0
        avg_pnl = data['total_pnl'] / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor (gross profits / gross losses)
        winning_trades = [t['pnl'] for t in data['trades'] if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in data['trades'] if t['pnl'] < 0]
        
        gross_profits = sum(winning_trades) if winning_trades else 0
        gross_losses = abs(sum(losing_trades)) if losing_trades else 1  # Avoid division by zero
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else gross_profits
        
        # Calculate expectancy
        avg_win = statistics.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(statistics.mean(losing_trades)) if losing_trades else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'pattern': pattern,
            'total_trades': total_trades,
            'win_rate': round(win_rate * 100, 1),
            'avg_pnl': round(avg_pnl, 2),
            'total_pnl': round(data['total_pnl'], 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2),
            'wins': data['wins'],
            'losses': data['losses']
        }
    
    def get_best_trading_times(self):
        """Identify the most profitable trading time slots"""
        time_stats = []
        
        for time_slot, data in self.pattern_stats['time_performance'].items():
            total_trades = data['wins'] + data['losses']
            if total_trades > 0:
                win_rate = data['wins'] / total_trades
                avg_pnl = data['total_pnl'] / total_trades
                
                time_stats.append({
                    'time_slot': time_slot,
                    'total_trades': total_trades,
                    'win_rate': round(win_rate * 100, 1),
                    'avg_pnl': round(avg_pnl, 2),
                    'total_pnl': round(data['total_pnl'], 2)
                })
        
        # Sort by average PnL
        time_stats.sort(key=lambda x: x['avg_pnl'], reverse=True)
        return time_stats
    
    def get_market_condition_performance(self):
        """Analyze performance across different market conditions"""
        condition_stats = []
        
        for condition, data in self.pattern_stats['market_conditions'].items():
            total_trades = data['wins'] + data['losses']
            if total_trades > 0:
                win_rate = data['wins'] / total_trades
                avg_pnl = data['total_pnl'] / total_trades
                
                condition_stats.append({
                    'condition': condition,
                    'total_trades': total_trades,
                    'win_rate': round(win_rate * 100, 1),
                    'avg_pnl': round(avg_pnl, 2),
                    'total_pnl': round(data['total_pnl'], 2)
                })
        
        # Sort by win rate
        condition_stats.sort(key=lambda x: x['win_rate'], reverse=True)
        return condition_stats
    
    def get_quick_insights(self):
        """Generate quick insights for GUI display"""
        insights = []
        
        # Best performing pattern
        pattern_stats = self.get_pattern_stats()
        if pattern_stats:
            best_pattern = max(pattern_stats.items(), 
                             key=lambda x: x[1]['expectancy'] if x[1]['total_trades'] > 5 else -999)
            if best_pattern[1]['total_trades'] > 5:
                insights.append(f"Best pattern: {best_pattern[0]} ({best_pattern[1]['win_rate']}% WR)")
        
        # Best trading time
        time_stats = self.get_best_trading_times()
        if time_stats and time_stats[0]['total_trades'] > 5:
            insights.append(f"Best time: {time_stats[0]['time_slot']} (${time_stats[0]['avg_pnl']} avg)")
        
        # Warning about poor performers
        for pattern, stats in pattern_stats.items():
            if stats['total_trades'] > 10 and stats['expectancy'] < -10:
                insights.append(f"⚠️ Avoid {pattern} pattern (negative expectancy)")
        
        # Session summary
        if self.session_cache['trades_analyzed'] > 0:
            insights.append(f"Session: {self.session_cache['trades_analyzed']} trades analyzed")
        
        return insights
    
    def export_report(self, filepath=None):
        """Export detailed performance report"""
        if not filepath:
            filepath = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filepath, 'w') as f:
            f.write("=== MonkeyMania Performance Report ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Trades Analyzed: {self.pattern_stats['metadata']['total_trades_analyzed']}\n\n")
            
            # Pattern Performance
            f.write("=== Pattern Performance ===\n")
            pattern_stats = self.get_pattern_stats()
            for pattern, stats in pattern_stats.items():
                if stats['total_trades'] > 0:
                    f.write(f"\n{pattern.upper()}:\n")
                    f.write(f"  Trades: {stats['total_trades']}\n")
                    f.write(f"  Win Rate: {stats['win_rate']}%\n")
                    f.write(f"  Avg PnL: ${stats['avg_pnl']}\n")
                    f.write(f"  Total PnL: ${stats['total_pnl']}\n")
                    f.write(f"  Profit Factor: {stats['profit_factor']}\n")
                    f.write(f"  Expectancy: ${stats['expectancy']}\n")
            
            # Time Performance
            f.write("\n=== Time Slot Performance ===\n")
            time_stats = self.get_best_trading_times()
            for slot in time_stats:
                f.write(f"\n{slot['time_slot']}:\n")
                f.write(f"  Trades: {slot['total_trades']}\n")
                f.write(f"  Win Rate: {slot['win_rate']}%\n")
                f.write(f"  Avg PnL: ${slot['avg_pnl']}\n")
            
            # Market Conditions
            f.write("\n=== Market Condition Performance ===\n")
            condition_stats = self.get_market_condition_performance()
            for condition in condition_stats:
                f.write(f"\n{condition['condition']}:\n")
                f.write(f"  Trades: {condition['total_trades']}\n")
                f.write(f"  Win Rate: {condition['win_rate']}%\n")
                f.write(f"  Avg PnL: ${condition['avg_pnl']}\n")
        
        return filepath