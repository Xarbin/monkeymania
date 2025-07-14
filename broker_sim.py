import json
import os

from real_feedback_engine import get_slippage_stats, get_slippage_adjustment

class BrokerSim:
    def buy_with_learned_slippage(self, ticker, predicted_price, shares):
        """
        Execute buy with slippage learned from real trades
        
        MM-B14.1: Uses real trade feedback for realistic fills
        """
        # Get slippage stats
        slippage_data = get_slippage_stats()
        
        # Apply learned slippage if we have enough data
        if slippage_data["sample_count"] >= 10:
            adjusted_entry = predicted_price * (1 + slippage_data["entry_slip_avg"])
            
            print(f"🎯 Applying learned slippage to {ticker}")
            print(f"   Predicted: ${predicted_price:.2f}")
            print(f"   Adjusted: ${adjusted_entry:.2f} ({slippage_data['entry_slip_avg']*100:.2f}% slip)")
        else:
            adjusted_entry = predicted_price
            remaining = 10 - slippage_data["sample_count"]
            print(f"📊 Need {remaining} more real trades for slippage calibration")
        
        # Execute at adjusted price
        cost = adjusted_entry * shares
        if cost > self.cash:
            raise ValueError(f"Insufficient cash to buy {shares} shares of {ticker} at ${adjusted_entry:.2f}")
        
        # Deduct cash
        self.cash -= cost
        
        # Update positions
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos['shares'] + shares
            avg_price = ((pos['shares'] * pos['avg_price']) + (shares * adjusted_entry)) / total_shares
            self.positions[ticker] = {'shares': total_shares, 'avg_price': avg_price}
        else:
            self.positions[ticker] = {'shares': shares, 'avg_price': adjusted_entry}
        if slippage_data["sample_count"] >= 10:
            total_slippage_cost = sum(t.get('slippage_cost', 0) for t in self.closed_trades[-len(self.open_trades):])
            print(f"\n💰 Day's slippage impact: ${total_slippage_cost:.2f}")
            print(f"   Based on {slippage_data['sample_count']} real trades")
        # Record as open trade with both prices
        self.open_trades.append({
            'ticker': ticker,
            'shares': shares,
            'predicted_price': predicted_price,
            'buy_price': adjusted_entry,
            'slippage_adjusted': slippage_data["sample_count"] >= 10,
            'entry_slippage': slippage_data.get("entry_slip_avg", 0),
            'status': 'open'
        })
        
        self.save_state()

    def settle_day_with_learned_slippage(self, postmarket_csv_path):
        """
        Settle trades with slippage learned from real execution
        
        MM-B14.1: Applies exit slippage from real trade data
        """
        import pandas as pd
        
        if not self.open_trades:
            raise RuntimeError("No open trades to settle")
        
        # Get slippage stats
        slippage_data = get_slippage_stats()
        
        df = pd.read_csv(postmarket_csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'symbol' in df.columns:
            df = df.rename(columns={'symbol': 'ticker'})
        
        for trade in self.open_trades:
            ticker = trade['ticker']
            shares = trade['shares']
            buy_price = trade['buy_price']
            
            # Find closing price from postmarket CSV
            row = df[df['ticker'].str.upper() == ticker.upper()]
            if row.empty:
                predicted_close = buy_price
            else:
                predicted_close = float(row.iloc[0].get('price', buy_price))
            
            # Apply learned exit slippage if available
            if slippage_data["sample_count"] >= 10:
                close_price = predicted_close * (1 + slippage_data["exit_slip_avg"])
                exit_slippage = slippage_data["exit_slip_avg"]
                
                print(f"🎯 Applying learned exit slippage to {ticker}")
                print(f"   Predicted: ${predicted_close:.2f}")
                print(f"   Adjusted: ${close_price:.2f} ({exit_slippage*100:.2f}% slip)")
            else:
                close_price = predicted_close
                exit_slippage = 0
            
            # Calculate both ideal and actual P&L
            ideal_pnl = (predicted_close - trade.get('predicted_price', buy_price)) * shares
            actual_pnl = (close_price - buy_price) * shares
            slippage_cost = ideal_pnl - actual_pnl
            
            self.cash += (close_price * shares)
            
            # Remove shares from positions
            if ticker in self.positions:
                pos = self.positions[ticker]
                pos['shares'] -= shares
                if pos['shares'] <= 0:
                    del self.positions[ticker]
                else:
                    self.positions[ticker] = pos
            
            # Mark trade closed with full slippage data
            trade['predicted_close'] = predicted_close
            trade['close_price'] = close_price
            trade['ideal_pnl'] = ideal_pnl
            trade['pnl'] = actual_pnl
            trade['slippage_cost'] = slippage_cost
            trade['exit_slippage'] = exit_slippage
            trade['status'] = 'closed'
            trade['close_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # Append to closed trades
            self.closed_trades.append(trade)
        
        # Clear open trades for next day
        self.open_trades.clear()
        self.save_state()
        
        # Generate slippage summary
    
    def __init__(self):
        self.cash = 0.0
        self.positions = {}  # ticker -> {'shares': int, 'avg_price': float}
        self.open_trades = []  # list of dicts with trade info
        self.closed_trades = []  # settled trades history

        self.load_state()

    def fund_account(self, amount):
        if amount <= 0:
            raise ValueError("Funding amount must be positive")
        self.cash += amount
        self.save_state()

    def get_cash(self):
        return self.cash

    def get_positions(self):
        return self.positions.copy()

    def buy(self, ticker, price, shares):
        if shares <= 0:
            raise ValueError("Shares must be positive")
        cost = price * shares
        if cost > self.cash:
            raise ValueError(f"Insufficient cash to buy {shares} shares of {ticker} at ${price:.2f}")
        # Deduct cash
        self.cash -= cost

        # Update positions (avg price weighted)
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos['shares'] + shares
            avg_price = ((pos['shares'] * pos['avg_price']) + (shares * price)) / total_shares
            self.positions[ticker] = {'shares': total_shares, 'avg_price': avg_price}
        else:
            self.positions[ticker] = {'shares': shares, 'avg_price': price}

        # Record as open trade (for the day)
        self.open_trades.append({
            'ticker': ticker,
            'shares': shares,
            'buy_price': price,
            'status': 'open'
        })
        self.save_state()

    def settle_day(self, postmarket_csv_path):
        import pandas as pd

        if not self.open_trades:
            raise RuntimeError("No open trades to settle")

        df = pd.read_csv(postmarket_csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'symbol' in df.columns:
            df = df.rename(columns={'symbol': 'ticker'})

        for trade in self.open_trades:
            ticker = trade['ticker']
            shares = trade['shares']
            buy_price = trade['buy_price']

            # Find closing price from postmarket CSV
            row = df[df['ticker'].str.upper() == ticker.upper()]
            if row.empty:
                # If no data, assume no price change (settle at buy price)
                close_price = buy_price
            else:
                close_price = float(row.iloc[0].get('price', buy_price))

            # Calculate pnl
            pnl = (close_price - buy_price) * shares
            self.cash += (close_price * shares)  # sell all shares at close price

            # Remove shares from positions
            if ticker in self.positions:
                pos = self.positions[ticker]
                pos['shares'] -= shares
                if pos['shares'] <= 0:
                    del self.positions[ticker]
                else:
                    self.positions[ticker] = pos

            # Mark trade closed
            trade['close_price'] = close_price
            trade['pnl'] = pnl
            trade['status'] = 'closed'
            trade['close_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

            # Append to closed trades
            self.closed_trades.append(trade)

        # Clear open trades for next day
        self.open_trades.clear()

        self.save_state()

    def get_open_trades(self):
        return list(self.open_trades)

    def get_closed_trades(self):
        return list(self.closed_trades)

    def load_state(self):
        # Load from disk, if exists
        if os.path.exists("broker_state.json"):
            with open("broker_state.json", "r") as f:
                data = json.load(f)
                self.cash = data.get("cash", 0.0)
                self.positions = data.get("positions", {})
                self.open_trades = data.get("open_trades", [])
                self.closed_trades = data.get("closed_trades", [])
        else:
            self.cash = 0.0
            self.positions = {}
            self.open_trades = []
            self.closed_trades = []

    def save_state(self):
        data = {
            "cash": self.cash,
            "positions": self.positions,
            "open_trades": self.open_trades,
            "closed_trades": self.closed_trades
        }
        with open("broker_state.json", "w") as f:
            json.dump(data, f, indent=2)