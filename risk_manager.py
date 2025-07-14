# risk_manager.py
# MM-B12: Comprehensive risk management system

import json
import os
from datetime import datetime


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_starting_capital = initial_capital
        
        # Risk parameters
        self.stop_loss_pct = -0.05  # -5%
        self.take_profit_pct = 0.10  # +10%
        self.max_daily_drawdown = -0.20  # -20%
        self.max_position_size_pct = 0.25  # 25% per ticker
        self.max_open_positions = 5
        
        # Tracking
        self.open_positions = {}
        self.daily_pnl = 0
        self.trades_today = []
        self.trading_halted = False
        self.halt_reason = None
        
    def check_position_limits(self, ticker, position_size):
        """Check if position meets risk limits"""
        
        # Check if trading is halted
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"
            
        # Check max open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max positions ({self.max_open_positions}) reached"
            
        # Check position size limit
        position_value = position_size * self.get_current_price(ticker)
        if position_value > self.current_capital * self.max_position_size_pct:
            max_allowed = self.current_capital * self.max_position_size_pct
            return False, f"Position too large: ${position_value:.2f} > ${max_allowed:.2f}"
            
        # Check if ticker already has position
        if ticker in self.open_positions:
            current_value = self.open_positions[ticker]['value']
            total_value = current_value + position_value
            if total_value > self.current_capital * self.max_position_size_pct:
                return False, f"Would exceed position limit for {ticker}"
                
        return True, "Position approved"
        
    def get_current_price(self, ticker):
        """Get current price estimate (placeholder - would connect to real data)"""
        # In production, this would fetch real-time price
        return 10.0  # Default small-cap estimate
        
    def add_position(self, ticker, entry_price, shares, confidence):
        """Add new position with automatic stops"""
        
        # Calculate stop and target prices
        stop_price = entry_price * (1 + self.stop_loss_pct)
        target_price = entry_price * (1 + self.take_profit_pct)
        
        position = {
            'ticker': ticker,
            'entry_price': entry_price,
            'shares': shares,
            'stop_price': stop_price,
            'target_price': target_price,
            'confidence': confidence,
            'entry_time': datetime.now(),
            'value': entry_price * shares,
            'status': 'open'
        }
        
        self.open_positions[ticker] = position
        
        # Log position
        print(f"📍 Position opened: {ticker}")
        print(f"   Entry: ${entry_price:.2f} | Stop: ${stop_price:.2f} | Target: ${target_price:.2f}")
        
        return position
        
    def check_stops_and_targets(self, ticker, current_price):
        """Check if position hit stop or target"""
        
        if ticker not in self.open_positions:
            return None
            
        position = self.open_positions[ticker]
        
        if position['status'] != 'open':
            return None
            
        # Check stop loss
        if current_price <= position['stop_price']:
            return self.close_position(ticker, current_price, 'stop_loss')
            
        # Check take profit
        if current_price >= position['target_price']:
            return self.close_position(ticker, current_price, 'take_profit')
            
        return None
        
    def close_position(self, ticker, exit_price, reason):
        """Close position and calculate P&L"""
        
        if ticker not in self.open_positions:
            return None
            
        position = self.open_positions[ticker]
        
        # Calculate P&L
        entry_total = position['entry_price'] * position['shares']
        exit_total = exit_price * position['shares']
        pnl = exit_total - entry_total
        pnl_pct = (pnl / entry_total) * 100
        
        # Update position
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['pnl'] = pnl
        position['pnl_pct'] = pnl_pct
        position['exit_reason'] = reason
        position['status'] = 'closed'
        
        # Update tracking
        self.daily_pnl += pnl
        self.current_capital += pnl
        self.trades_today.append(position)
        
        # Log exit
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} {ticker} closed: {reason}")
        print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        
        # Check drawdown
        self.check_daily_drawdown()
        
        # Remove from open positions
        del self.open_positions[ticker]
        
        return position
        
    def check_daily_drawdown(self):
        """Check if daily drawdown limit hit"""
        
        daily_drawdown_pct = (self.daily_pnl / self.daily_starting_capital)
        
        if daily_drawdown_pct <= self.max_daily_drawdown:
            self.trading_halted = True
            self.halt_reason = f"Daily drawdown limit hit: {daily_drawdown_pct:.1%}"
            
            # Log to monkey journal
            self.log_trading_halt()
            
            print(f"🛑 TRADING HALTED: {self.halt_reason}")
            return True
            
        return False
        
    def log_trading_halt(self):
        """Log trading halt to monkey journal"""
        
        halt_entry = {
            'type': 'trading_halt',
            'timestamp': datetime.now().isoformat(),
            'reason': self.halt_reason,
            'daily_pnl': self.daily_pnl,
            'drawdown_pct': (self.daily_pnl / self.daily_starting_capital) * 100,
            'message': "Mon Kee is taking a break to protect the bananas 🍌"
        }
        
        # Append to journal
        journal_file = 'monkey_journal.json'
        if os.path.exists(journal_file):
            with open(journal_file, 'r') as f:
                journal = json.load(f)
        else:
            journal = {}
            
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in journal:
            journal[today] = []
        journal[today].append(halt_entry)
        
        with open(journal_file, 'w') as f:
            json.dump(journal, f, indent=2)
            
    def reset_daily_stats(self):
        """Reset stats for new trading day"""
        self.daily_starting_capital = self.current_capital
        self.daily_pnl = 0
        self.trades_today = []
        self.trading_halted = False
        self.halt_reason = None
        
    def get_risk_metrics(self):
        """Get current risk metrics for display"""
        return {
            'current_capital': self.current_capital,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.daily_starting_capital) * 100,
            'open_positions': len(self.open_positions),
            'max_positions': self.max_open_positions,
            'trading_active': not self.trading_halted,
            'halt_reason': self.halt_reason
        }