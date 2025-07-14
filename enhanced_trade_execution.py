# enhanced_trade_execution.py
"""
Mon Kee's Intelligent Trade Execution System
Adds aggressive/conservative/normal trade sizing with AI recommendations
"""

import math
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QGroupBox, QButtonGroup, QRadioButton,
                             QProgressBar, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

class MonKeeTradeExecutor(QWidget):
    """Mon Kee's intelligent trade execution widget"""
    
    trade_executed = pyqtSignal(dict)  # Emits trade details
    
    def __init__(self, performance_analytics, broker=None, parent=None):
        super().__init__(parent)
        self.performance_analytics = performance_analytics
        self.broker = broker  # Add broker reference
        self.current_pattern = None
        self.current_confidence = 0
        self.base_position_size = 1000  # Default position size
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the Mon Kee execution interface"""
        layout = QVBoxLayout()
        
        # Mon Kee's Recommendation Display
        recommendation_group = QGroupBox("🐒 Mon Kee's Analysis")
        recommendation_layout = QVBoxLayout()
        
        # Pattern confidence display
        self.confidence_label = QLabel("Analyzing pattern...")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        confidence_font = QFont()
        confidence_font.setPointSize(12)
        confidence_font.setBold(True)
        self.confidence_label.setFont(confidence_font)
        
        # Confidence meter
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("Pattern Confidence: %p%")
        
        # Mon Kee's recommendation
        self.recommendation_label = QLabel("🐒 Mon Kee is thinking...")
        self.recommendation_label.setAlignment(Qt.AlignCenter)
        self.recommendation_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        recommendation_layout.addWidget(self.confidence_label)
        recommendation_layout.addWidget(self.confidence_bar)
        recommendation_layout.addWidget(self.recommendation_label)
        recommendation_group.setLayout(recommendation_layout)
        
        # Trade Size Selection
        size_group = QGroupBox("Select Trade Size")
        size_layout = QVBoxLayout()
        
        self.size_button_group = QButtonGroup()
        
        # Create trade size options with visual indicators
        self.conservative_btn = self.create_size_button(
            "🛡️ Conservative (0.5x)", 
            "Lower risk, preserve capital",
            "#2196F3"
        )
        self.normal_btn = self.create_size_button(
            "⚖️ Normal (1x)", 
            "Standard position size",
            "#4CAF50"
        )
        self.aggressive_btn = self.create_size_button(
            "🚀 Aggressive (2x)", 
            "High conviction, larger position",
            "#FF9800"
        )
        
        self.size_button_group.addButton(self.conservative_btn, 0)
        self.size_button_group.addButton(self.normal_btn, 1)
        self.size_button_group.addButton(self.aggressive_btn, 2)
        
        # Default to normal
        self.normal_btn.setChecked(True)
        
        size_layout.addWidget(self.conservative_btn)
        size_layout.addWidget(self.normal_btn)
        size_layout.addWidget(self.aggressive_btn)
        size_group.setLayout(size_layout)
        
        # Position details display
        self.position_details = QLabel("Position Size: Calculating...")
        self.position_details.setAlignment(Qt.AlignCenter)
        self.position_details.setStyleSheet("""
            QLabel {
                background-color: #e8f5e9;
                border: 1px solid #4CAF50;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
            }
        """)
        
        # Execute button
        self.execute_btn = QPushButton("🎯 EXECUTE TRADE")
        self.execute_btn.setMinimumHeight(50)
        self.execute_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.execute_btn.clicked.connect(self.execute_trade)
        
        # Connect size selection to update position details
        self.size_button_group.buttonClicked.connect(self.update_position_details)
        
        # Layout assembly
        layout.addWidget(recommendation_group)
        layout.addWidget(size_group)
        layout.addWidget(self.position_details)
        layout.addWidget(self.execute_btn)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def create_size_button(self, text, tooltip, color):
        """Create a styled radio button for trade size"""
        btn = QRadioButton(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(f"""
            QRadioButton {{
                font-size: 14px;
                padding: 8px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}
            QRadioButton::indicator:checked {{
                background-color: {color};
                border: 2px solid {color};
                border-radius: 9px;
            }}
        """)
        return btn
        
    def analyze_pattern(self, pattern_data):
        """Mon Kee analyzes the pattern and makes recommendation"""
        self.current_pattern = pattern_data
        
        # Get pattern statistics from performance analytics
        pattern_stats = self.performance_analytics.get_pattern_stats()
        pattern_name = pattern_data.get('pattern_name', 'Unknown')
        
        # Calculate confidence based on historical performance
        if pattern_name in pattern_stats:
            stats = pattern_stats[pattern_name]
            win_rate = stats['win_rate']
            total_trades = stats['total_trades']
            avg_profit = stats.get('avg_profit', 0)
            
            # Calculate confidence score
            confidence = self.calculate_confidence(win_rate, total_trades, avg_profit)
            self.current_confidence = confidence
            
            # Update UI
            self.confidence_bar.setValue(int(confidence * 100))
            self.update_confidence_color(confidence)
            
            # Generate Mon Kee's recommendation
            recommendation = self.generate_recommendation(confidence, win_rate, total_trades)
            self.recommendation_label.setText(recommendation['text'])
            
            # Auto-select recommended size
            if recommendation['suggested_size'] == 'conservative':
                self.conservative_btn.setChecked(True)
            elif recommendation['suggested_size'] == 'aggressive':
                self.aggressive_btn.setChecked(True)
            else:
                self.normal_btn.setChecked(True)
                
            # Update position details
            self.update_position_details()
            
        else:
            # New pattern - be conservative
            self.confidence_bar.setValue(25)
            self.confidence_label.setText("⚠️ New Pattern - Limited Data")
            self.recommendation_label.setText(
                "🐒 Mon Kee says: New pattern detected! Start conservative to gather data."
            )
            self.conservative_btn.setChecked(True)
            
    def calculate_confidence(self, win_rate, total_trades, avg_profit):
        """Calculate pattern confidence score (0-1)"""
        # Weight factors
        win_rate_weight = 0.4
        sample_size_weight = 0.3
        profit_weight = 0.3
        
        # Win rate score (0-1)
        win_rate_score = min(win_rate, 1.0)
        
        # Sample size score (more trades = more confidence, caps at 50)
        sample_size_score = min(total_trades / 50, 1.0)
        
        # Profit score (positive avg profit)
        profit_score = 1.0 if avg_profit > 0 else 0.5
        
        confidence = (win_rate_score * win_rate_weight + 
                     sample_size_score * sample_size_weight + 
                     profit_score * profit_weight)
        
        return confidence
        
    def generate_recommendation(self, confidence, win_rate, total_trades):
        """Generate Mon Kee's trading recommendation"""
        if confidence > 0.8 and win_rate > 0.7:
            return {
                'suggested_size': 'aggressive',
                'text': f"🐒 Mon Kee is EXCITED! This pattern has {win_rate:.0%} win rate "
                        f"over {total_trades} trades. Time to SIZE UP! 🚀"
            }
        elif confidence > 0.6 and win_rate > 0.6:
            return {
                'suggested_size': 'normal',
                'text': f"🐒 Mon Kee likes this setup! Solid {win_rate:.0%} win rate. "
                        f"Standard position recommended. ⚖️"
            }
        elif confidence > 0.4:
            return {
                'suggested_size': 'conservative',
                'text': f"🐒 Mon Kee is cautious. {win_rate:.0%} win rate needs improvement. "
                        f"Play it safe! 🛡️"
            }
        else:
            return {
                'suggested_size': 'conservative',
                'text': f"🐒 Mon Kee says be careful! Low confidence pattern. "
                        f"Protect capital first! ⚠️"
            }
            
    def update_confidence_color(self, confidence):
        """Update confidence bar color based on level"""
        if confidence > 0.7:
            color = "#4CAF50"  # Green
        elif confidence > 0.5:
            color = "#FF9800"  # Orange
        else:
            color = "#f44336"  # Red
            
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)
        
    def update_position_details(self):
        """Update position size display based on selection"""
        # Get current balance from broker if available
        if hasattr(self, 'broker') and self.broker:
            current_balance = self.broker.get_cash()
        elif hasattr(self.performance_analytics, 'current_balance'):
            current_balance = self.performance_analytics.current_balance
        else:
            current_balance = 10000.0  # Default
            
        # Get current balance from performance analytics or use default
        if hasattr(self.performance_analytics, 'current_balance'):
            current_balance = self.performance_analytics.current_balance
        else:
            # Use default starting balance
            current_balance = 10000.0  # Your starting balance
        
        # Get selected size multiplier
        size_map = {0: 0.5, 1: 1.0, 2: 2.0}  # conservative, normal, aggressive
        selected_id = self.size_button_group.checkedId()
        multiplier = size_map.get(selected_id, 1.0)
        
        # Calculate position size using Kelly Criterion
        base_size = self.calculate_kelly_position(current_balance)
        position_size = base_size * multiplier
        
        # Calculate risk amount
        risk_amount = position_size * 0.07  # Assuming 7% stop loss
        
        # Update display
        self.position_details.setText(
            f"Position Size: ${position_size:,.2f}\n"
            f"Risk Amount: ${risk_amount:,.2f} ({(risk_amount/current_balance)*100:.1f}% of account)"
        )
        
    def calculate_kelly_position(self, balance):
        """Calculate position size using Kelly Criterion"""
        # Use default values if no performance data yet
        try:
            # Try to get pattern stats instead
            pattern_stats = self.performance_analytics.get_pattern_stats()
            
            # Calculate overall metrics from pattern stats
            total_wins = 0
            total_losses = 0
            total_win_amount = 0
            total_loss_amount = 0
            
            for pattern, stats in pattern_stats.items():
                if isinstance(stats, dict):
                    wins = stats.get('wins', 0)
                    losses = stats.get('losses', 0)
                    total_wins += wins
                    total_losses += losses
                    
                    # Estimate average win/loss (simplified)
                    if wins > 0:
                        total_win_amount += stats.get('total_pnl', 0)
            
            total_trades = total_wins + total_losses
            
            if total_trades > 0:
                win_rate = total_wins / total_trades
                avg_win = total_win_amount / total_wins if total_wins > 0 else 0.15
                avg_loss = 0.07  # Default 7% stop loss
            else:
                # No trades yet - use conservative defaults
                win_rate = 0.5
                avg_win = 0.15
                avg_loss = 0.07
                
        except:
            # If any error, use safe defaults
            win_rate = 0.5
            avg_win = 0.15
            avg_loss = 0.07
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        if avg_loss > 0:
            b = avg_win / avg_loss
            q = 1 - win_rate
            kelly = (win_rate * b - q) / b
            
            # Use 25% Kelly for safety
            safe_kelly = max(0, kelly * 0.25)
            
            # Position size as percentage of balance
            position_size = balance * safe_kelly
            
            # Cap at 10% of account max
            return min(position_size, balance * 0.1)
        else:
            # Default to 2% of balance if no loss data
            return balance * 0.02
    def execute_trade(self):
        """Execute the trade with selected size"""
        if not self.current_pattern:
            return
            
        # Get selected size
        size_map = {0: 'conservative', 1: 'normal', 2: 'aggressive'}
        selected_id = self.size_button_group.checkedId()
        trade_size = size_map.get(selected_id, 'normal')
        
        # Prepare trade data
        trade_data = {
            'pattern': self.current_pattern,
            'size_type': trade_size,
            'confidence': self.current_confidence,
            'position_size': self.calculate_kelly_position(
                self.performance_analytics.current_balance
            ) * {0: 0.5, 1: 1.0, 2: 2.0}[selected_id]
        }
        
        # Emit signal for trade execution
        self.trade_executed.emit(trade_data)
        
        # Show confirmation
        self.recommendation_label.setText(
            f"🐒 Mon Kee executed {trade_size.upper()} trade! "
            f"Position: ${trade_data['position_size']:,.2f}"
        )