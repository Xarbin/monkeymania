# compound_growth_tracker.py
"""
Compound Growth Tracking and Visualization for MonkeyMania
Shows your path to wealth with projections and milestones
"""

import math
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QGroupBox, QGridLayout, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
#import pyqtgraph as pg

class CompoundGrowthTracker(QWidget):
    """Track and visualize compound growth to wealth"""
    
    def __init__(self, performance_analytics, parent=None):
        super().__init__(parent)
        self.performance_analytics = performance_analytics
        self.wealth_milestones = [25000, 50000, 100000, 250000, 500000, 1000000]
        
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Create the compound growth interface"""
        layout = QVBoxLayout()
        
        # Current Status Box
        status_group = QGroupBox("💰 Wealth Status")
        status_layout = QGridLayout()
        
        # Large balance display
        self.balance_label = QLabel("$10,000")
        balance_font = QFont()
        balance_font.setPointSize(24)
        balance_font.setBold(True)
        self.balance_label.setFont(balance_font)
        self.balance_label.setAlignment(Qt.AlignCenter)
        
        # Growth percentage
        self.growth_label = QLabel("+0.0%")
        self.growth_label.setAlignment(Qt.AlignCenter)
        self.growth_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        
        # Starting capital reference
        self.start_label = QLabel("Started with: $10,000")
        self.start_label.setAlignment(Qt.AlignCenter)
        
        status_layout.addWidget(self.balance_label, 0, 0, 1, 2)
        status_layout.addWidget(self.growth_label, 1, 0, 1, 2)
        status_layout.addWidget(self.start_label, 2, 0, 1, 2)
        status_group.setLayout(status_layout)
        
        # Growth Metrics Box
        metrics_group = QGroupBox("📈 Growth Metrics")
        metrics_layout = QGridLayout()
        
        # Daily growth rate
        self.daily_rate_label = QLabel("Daily Growth: 0.0%")
        self.weekly_rate_label = QLabel("Weekly Growth: 0.0%")
        self.monthly_rate_label = QLabel("Monthly Growth: 0.0%")
        
        # Doubling time
        self.doubling_label = QLabel("Time to Double: Calculating...")
        self.doubling_label.setStyleSheet("font-weight: bold;")
        
        metrics_layout.addWidget(self.daily_rate_label, 0, 0)
        metrics_layout.addWidget(self.weekly_rate_label, 0, 1)
        metrics_layout.addWidget(self.monthly_rate_label, 1, 0)
        metrics_layout.addWidget(self.doubling_label, 1, 1)
        metrics_group.setLayout(metrics_layout)
        
        # Milestone Progress Box
        milestone_group = QGroupBox("🎯 Next Milestone")
        milestone_layout = QVBoxLayout()
        
        self.milestone_label = QLabel("Target: $25,000")
        self.milestone_label.setAlignment(Qt.AlignCenter)
        milestone_font = QFont()
        milestone_font.setPointSize(14)
        self.milestone_label.setFont(milestone_font)
        
        self.milestone_progress = QProgressBar()
        self.milestone_progress.setRange(0, 100)
        self.milestone_progress.setTextVisible(True)
        self.milestone_progress.setFormat("%p% to next milestone")
        
        self.milestone_eta = QLabel("ETA: Calculating...")
        self.milestone_eta.setAlignment(Qt.AlignCenter)
        
        milestone_layout.addWidget(self.milestone_label)
        milestone_layout.addWidget(self.milestone_progress)
        milestone_layout.addWidget(self.milestone_eta)
        milestone_group.setLayout(milestone_layout)
        
        # Million Dollar Projection Box
        projection_group = QGroupBox("🚀 Path to $1 Million")
        projection_layout = QVBoxLayout()
        
        self.million_eta = QLabel("Calculating trajectory...")
        self.million_eta.setAlignment(Qt.AlignCenter)
        million_font = QFont()
        million_font.setPointSize(16)
        million_font.setBold(True)
        self.million_eta.setFont(million_font)
        
        self.trades_needed = QLabel("Trades needed: Calculating...")
        self.trades_needed.setAlignment(Qt.AlignCenter)
        
        self.daily_target = QLabel("Daily profit target: Calculating...")
        self.daily_target.setAlignment(Qt.AlignCenter)
        
        projection_layout.addWidget(self.million_eta)
        projection_layout.addWidget(self.trades_needed)
        projection_layout.addWidget(self.daily_target)
        projection_group.setLayout(projection_layout)
        
        # Growth Chart
        self.growth_chart = self.create_growth_chart()
        
        # Layout assembly
        layout.addWidget(status_group)
        layout.addWidget(metrics_group)
        layout.addWidget(milestone_group)
        layout.addWidget(projection_group)
        layout.addWidget(self.growth_chart)
        
        self.setLayout(layout)
        
    def create_growth_chart(self):
        """Create a simple growth visualization"""
        # For now, return a placeholder
        # In full implementation, use pyqtgraph or matplotlib
        chart_label = QLabel("📊 Growth Chart")
        chart_label.setAlignment(Qt.AlignCenter)
        chart_label.setMinimumHeight(200)
        chart_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        return chart_label
        
    def setup_timer(self):
        """Setup periodic updates"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(5000)  # Update every 5 seconds
        
        # Initial update
        self.update_display()
        
    def update_display(self):
        """Update all growth metrics"""
        if not hasattr(self.performance_analytics, 'current_balance'):
            return
            
        # Get current data
        current_balance = self.performance_analytics.current_balance
        starting_balance = self.performance_analytics.starting_balance
        
        # Update balance display
        self.balance_label.setText(f"${current_balance:,.2f}")
        
        # Calculate and display growth
        total_growth = ((current_balance / starting_balance) - 1) * 100
        self.growth_label.setText(f"+{total_growth:.1f}%")
        
        # Update color based on performance
        if total_growth > 0:
            self.growth_label.setStyleSheet("color: #4CAF50; font-size: 18px; font-weight: bold;")
        else:
            self.growth_label.setStyleSheet("color: #f44336; font-size: 18px; font-weight: bold;")
            
        # Calculate growth rates
        self.calculate_growth_rates()
        
        # Update milestone progress
        self.update_milestone_progress(current_balance)
        
        # Project path to million
        self.project_to_million(current_balance, starting_balance)
        
    def calculate_growth_rates(self):
        """Calculate daily, weekly, monthly growth rates"""
        metrics = self.performance_analytics.calculate_metrics()
        
        # This is simplified - in production, track actual time periods
        total_days = max(1, self.performance_analytics.total_trades)  # Rough estimate
        
        if self.performance_analytics.starting_balance > 0:
            total_return = (self.performance_analytics.current_balance / 
                          self.performance_analytics.starting_balance) - 1
            
            # Daily rate (compound)
            daily_rate = (1 + total_return) ** (1/total_days) - 1
            weekly_rate = (1 + daily_rate) ** 7 - 1
            monthly_rate = (1 + daily_rate) ** 30 - 1
            
            self.daily_rate_label.setText(f"Daily Growth: {daily_rate*100:.2f}%")
            self.weekly_rate_label.setText(f"Weekly Growth: {weekly_rate*100:.1f}%")
            self.monthly_rate_label.setText(f"Monthly Growth: {monthly_rate*100:.1f}%")
            
            # Doubling time
            if daily_rate > 0:
                doubling_days = math.log(2) / math.log(1 + daily_rate)
                self.doubling_label.setText(f"Time to Double: {doubling_days:.0f} days")
            else:
                self.doubling_label.setText("Time to Double: N/A")
                
    def update_milestone_progress(self, current_balance):
        """Update progress to next milestone"""
        # Find next milestone
        next_milestone = None
        for milestone in self.wealth_milestones:
            if current_balance < milestone:
                next_milestone = milestone
                break
                
        if next_milestone:
            # Calculate progress
            prev_milestone = self.performance_analytics.starting_balance
            for milestone in self.wealth_milestones:
                if milestone < next_milestone and current_balance > milestone:
                    prev_milestone = milestone
                    
            progress = ((current_balance - prev_milestone) / 
                       (next_milestone - prev_milestone)) * 100
            
            self.milestone_label.setText(f"Target: ${next_milestone:,}")
            self.milestone_progress.setValue(int(progress))
            
            # Estimate time to milestone
            self.estimate_milestone_eta(current_balance, next_milestone)
        else:
            self.milestone_label.setText("🎉 All milestones achieved!")
            self.milestone_progress.setValue(100)
            
    def estimate_milestone_eta(self, current, target):
        """Estimate time to reach milestone"""
        # Get recent growth rate
        metrics = self.performance_analytics.calculate_metrics()
        
        if self.performance_analytics.total_trades > 5:
            # Simple estimation based on average trade profit
            avg_profit_per_trade = self.performance_analytics.total_pnl / self.performance_analytics.total_trades
            
            if avg_profit_per_trade > 0:
                trades_needed = (target - current) / avg_profit_per_trade
                
                # Assume 3 trades per day average
                days_needed = trades_needed / 3
                
                eta_date = datetime.now() + timedelta(days=days_needed)
                self.milestone_eta.setText(f"ETA: {eta_date.strftime('%B %d, %Y')}")
            else:
                self.milestone_eta.setText("ETA: Need positive returns")
        else:
            self.milestone_eta.setText("ETA: Need more data")
            
    def project_to_million(self, current_balance, starting_balance):
        """Project path to $1 million"""
        target = 1_000_000
        
        if current_balance <= starting_balance:
            self.million_eta.setText("📈 Need positive returns first!")
            return
            
        # Calculate based on current growth rate
        total_days = max(1, self.performance_analytics.total_trades)
        daily_return = (current_balance / starting_balance) ** (1/total_days) - 1
        
        if daily_return > 0:
            days_to_million = math.log(target/current_balance) / math.log(1 + daily_return)
            
            # Convert to readable format
            if days_to_million < 365:
                self.million_eta.setText(f"🚀 {days_to_million:.0f} days to $1 MILLION!")
            else:
                years = days_to_million / 365
                self.million_eta.setText(f"🚀 {years:.1f} years to $1 MILLION!")
                
            # Calculate required performance
            trades_to_million = days_to_million * 3  # Assuming 3 trades/day
            self.trades_needed.setText(f"Trades needed: ~{trades_to_million:.0f}")
            
            # Daily profit target
            daily_profit_needed = (target - current_balance) / days_to_million
            self.daily_target.setText(f"Daily profit target: ${daily_profit_needed:.2f}")
        else:
            self.million_eta.setText("📈 Achieve consistent profits first!")
            
    def celebrate_milestone(self, milestone):
        """Show celebration when milestone is hit"""
        # This could trigger a popup or special animation
        celebration_text = {
            25000: "🎉 $25K! First milestone crushed!",
            50000: "🎊 $50K! Halfway to six figures!",
            100000: "🏆 $100K! Six figures achieved!",
            250000: "💎 $250K! Quarter million!",
            500000: "👑 $500K! Half way to millionaire!",
            1000000: "🚀 $1 MILLION! Mon Kee status achieved!"
        }
        
        if milestone in celebration_text:
            # In production, show a proper celebration dialog
            print(celebration_text[milestone])