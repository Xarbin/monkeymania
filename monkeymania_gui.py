# monkeymania_gui.py - COMPLETE VERSION WITH PERFORMANCE TRACKING
# MM-B8 + MM-B15: Clean GUI with Performance Analytics

import sys
import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QProgressBar, QDialog, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# Import modular components
from gui_handlers import (
    get_csv_file, show_info, show_warning, show_error, show_question,
    log_system_message, format_currency, update_status_label,
    show_file_loaded_message, confirm_reset_action,
)
from trade_controller import create_trade_controller
from xp_tracker import get_xp_tracker, update_all_skills_ui, generate_skills_report
from vibe_analysis import generate_self_assessment
from online_learning import load_model
from broker_sim import BrokerSim
from performance_panel import PerformancePanel
from performance_analytics import PerformanceAnalytics
from enhanced_trade_execution import MonKeeTradeExecutor
from compound_growth_tracker import CompoundGrowthTracker
from skills_engine import get_xp_progress

class XPSidebarWidget(QWidget):
    """MM-B11: RuneScape-style XP progress bars"""
    
    def __init__(self, xp_tracker):
        super().__init__()
        self.xp_tracker = xp_tracker
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🐵 Mon Kee Skills")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #4CAF50;
                padding: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Skill bars container
        self.skills_container = QWidget()
        self.skills_layout = QVBoxLayout(self.skills_container)
        layout.addWidget(self.skills_container)
        
        # Add stretch at bottom
        layout.addStretch()
        
        self.setLayout(layout)
        self.setMaximumWidth(250)
        self.update_skills()
   
    
    def update_skills(self):
        # Clear existing widgets
        for i in reversed(range(self.skills_layout.count())): 
            self.skills_layout.itemAt(i).widget().setParent(None)
            
        # Get top skills
        top_skills = self.xp_tracker.get_top_skills(5)
        
        for skill in top_skills:
            skill_widget = self.create_skill_bar(skill)
            self.skills_layout.addWidget(skill_widget)
            
    def create_skill_bar(self, skill_data):
        """Create a single skill progress bar"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        
        # Skill name and level
        header = QLabel(f"{skill_data['name']} (Lvl {skill_data['level']})")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        # XP text
        xp_text = QLabel(f"{skill_data['current_xp']:,}/{skill_data['next_level_xp']:,} XP")
        xp_text.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(xp_text)
        
        # Progress bar
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(100)
        progress.setValue(int(skill_data['progress_percent']))
        progress.setTextVisible(False)
        progress.setFixedHeight(15)
        
        # RuneScape-style coloring
        progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 3px;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #7FFF00,
                    stop: 1 #228B22
                );
                border-radius: 3px;
            }
        """)
        
        layout.addWidget(progress)
        widget.setLayout(layout)
        
        # Add separator
        widget.setStyleSheet("""
            QWidget {
                border-bottom: 1px solid #333;
                padding: 5px;
            }
        """)
        
        return widget


class MonkeyManiaGUI(QWidget):
    """Streamlined MonkeyMania GUI with modular backend and performance tracking"""
    
    def __init__(self):
        super().__init__()
        self.init_core_systems()
        self.init_ui()
        self.restore_session()
    
    def init_core_systems(self):
        """Initialize core trading systems"""
        self.bank_starting_amount = 10000.0
        self.broker = BrokerSim()
        self.learning_model = load_model()
        self.xp_tracker = get_xp_tracker()
        
        # Create trade controller with Mon Kee callback
        self.trade_controller = create_trade_controller(
            self.broker, 
            self.learning_model,
            monkee_callback=self.update_monkee_with_pattern if hasattr(self, 'update_monkee_with_pattern') else None
        )
        
        self.performance_analytics = PerformanceAnalytics()  # New for MM-B15
        
        # UI state
        self.csv_path = None
        self.postmarket_csv_path = None
        self.daily_starting_balance = self.bank_starting_amount
        
        # Setup Mon Kee components (must be after performance_analytics)
        self.setup_monkee_components()
         # ... existing code ...
        self.performance_analytics = PerformanceAnalytics()
    
        # Debug: See what methods are available
        print("PerformanceAnalytics methods:", [m for m in dir(self.performance_analytics) if not m.startswith('_')])
    
    def apply_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11pt;
                background-color: #1a1a1a;
                color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 2px solid #444;
                background-color: #1a1a1a;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3b3b3b;
                border-bottom: 2px solid #4CAF50;
            }
            QLabel {
                margin: 6px 0;
                font-weight: bold;
                color: #ffffff;
                font-size: 12pt;
            }
            QPushButton {
                padding: 10px;
                font-weight: bold;
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #444;
                border-radius: 6px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #3b3b3b;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #1b1b1b;
            }
            QPushButton#buyButton {
                background-color: #2e7d32;
                border-color: #4CAF50;
            }
            QPushButton#buyButton:hover {
                background-color: #388e3c;
            }
            QPushButton#settleButton {
                background-color: #1565c0;
                border-color: #2196F3;
            }
            QPushButton#settleButton:hover {
                background-color: #1976d2;
            }
            QPushButton#dangerButton {
                background-color: #c62828;
                border-color: #f44336;
            }
            QPushButton#dangerButton:hover {
                background-color: #d32f2f;
            }
            QTextEdit {
                background-color: #0d0d0d;
                color: #00ff00;
                font-family: 'Consolas', monospace;
                font-size: 10pt;
                border: 2px solid #333;
                border-radius: 4px;
                padding: 5px;
            }
            QProgressBar {
                border: 2px solid #444;
                border-radius: 8px;
                text-align: center;
                background-color: #1a1a1a;
                color: white;
                font-weight: bold;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4CAF50, stop: 0.5 #66BB6A, stop: 1 #4CAF50);
                border-radius: 6px;
            }
            QDialog {
                background-color: #1a1a1a;
            }
        """)
    
    def init_ui(self):
        """Initialize the user interface with performance tracking"""
        self.setWindowTitle("🦍 Monkey Mania - Performance Edition")
        self.resize(1400, 900)  # Increased for tabs
        self.apply_dark_theme()
        
        # Main vertical layout
        main_layout = QVBoxLayout()
        
        # Create tab widget for main content and performance
        self.tab_widget = QTabWidget()
        
        # Trading Tab
        self.trading_tab = QWidget()
        self.create_trading_tab()
        self.tab_widget.addTab(self.trading_tab, "🦍 Trading")
        
        # Performance Tab  
        self.performance_tab = QWidget()
        self.create_performance_tab()
        self.tab_widget.addTab(self.performance_tab, "📊 Performance")
        
         # ADD NEW TAB FOR WEALTH TRACKER:
        self.wealth_tab = QWidget()
        self.create_wealth_tab()
        self.tab_widget.addTab(self.wealth_tab, "💰 Wealth Tracker")
    
        # Add tabs to main layout
        main_layout.addWidget(self.tab_widget)
        
        self.setLayout(main_layout)
        
        # Initial status
        log_system_message(self.log, "🦍 MonkeyMania started - Performance Edition loaded", "SUCCESS")
    
    def create_trading_tab(self):
        """Create the main trading tab"""
        # Create horizontal layout for content and sidebar
        content_layout = QHBoxLayout()
        
        # Left side - XP Sidebar
        self.xp_sidebar = XPSidebarWidget(self.xp_tracker)
        content_layout.addWidget(self.xp_sidebar)
        
        # Right side - Main content
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Performance widget at top
        self.create_performance_widget(right_layout)
        
        # Status and file loading section
        self.create_file_section(right_layout)
        
        # Trading action buttons
        self.create_trading_section(right_layout)
        
        # System management buttons  
        self.create_management_section(right_layout)
        
        # Progress and status displays
        self.create_status_section(right_layout)
        
        # Analysis and reporting buttons
        self.create_analysis_section(right_layout)
        
        # Cash balance display
        self.cash_balance_label = QLabel(f"Bank Balance: {format_currency(self.broker.get_cash())}")
        right_layout.addWidget(self.cash_balance_label)
        
        # Log output
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        right_layout.addWidget(self.log)
        
        right_widget.setLayout(right_layout)
        content_layout.addWidget(right_widget, 1)  # Give main content stretch priority
        
        self.trading_tab.setLayout(content_layout)
    
    def create_performance_tab(self):
        """Create the performance analytics tab"""
        layout = QVBoxLayout()
        
        # Initialize performance panel
        self.performance_panel = PerformancePanel(self.performance_tab)
        
        self.performance_tab.setLayout(layout)
    
    def create_performance_widget(self, parent_layout):
        """Add compact performance widget to main dashboard"""
        perf_frame = QWidget()
        perf_frame.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 6px;
                padding: 5px;
            }
        """)
        
        perf_layout = QHBoxLayout()
        
        # Today's stats
        self.today_label = QLabel("Today: -- trades")
        self.today_label.setStyleSheet("font-size: 11px; margin: 0 10px;")
        perf_layout.addWidget(self.today_label)
        
        self.win_rate_label = QLabel("Win Rate: ---%")
        self.win_rate_label.setStyleSheet("font-size: 11px; margin: 0 10px;")
        perf_layout.addWidget(self.win_rate_label)
        
        self.pnl_label = QLabel("P&L: $0.00")
        self.pnl_label.setStyleSheet("font-size: 11px; margin: 0 10px;")
        perf_layout.addWidget(self.pnl_label)
        
        self.best_pattern_label = QLabel("Best Pattern: ---")
        self.best_pattern_label.setStyleSheet("font-size: 11px; margin: 0 10px;")
        perf_layout.addWidget(self.best_pattern_label)
        
        perf_frame.setLayout(perf_layout)
        parent_layout.addWidget(perf_frame)
        
        # Update timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_widget)
        self.perf_timer.start(5000)  # Update every 5 seconds
    
    def update_performance_widget(self):
        """Update the performance widget with latest stats"""
        try:
            insights = self.performance_analytics.get_quick_insights()
            
            # Parse insights for display
            today_trades = self.performance_analytics.session_cache.get('trades_analyzed', 0)
            self.today_label.setText(f"Today: {today_trades} trades")
            
            # Get pattern stats
            pattern_stats = self.performance_analytics.get_pattern_stats()
            if pattern_stats:
                # Calculate overall win rate
                total_wins = 0
                total_losses = 0
                total_pnl = 0
                
                for pattern, stats in pattern_stats.items():
                    # Check if stats is a dictionary and has the expected keys
                    if isinstance(stats, dict):
                        total_wins += stats.get('wins', 0)
                        total_losses += stats.get('losses', 0)
                        total_pnl += stats.get('total_pnl', 0)
                
                total_trades = total_wins + total_losses
                if total_trades > 0:
                    overall_wr = (total_wins / total_trades) * 100
                    self.win_rate_label.setText(f"Win Rate: {overall_wr:.1f}%")
                else:
                    self.win_rate_label.setText("Win Rate: ---%")
                
                # Update PnL
                self.pnl_label.setText(f"P&L: {format_currency(total_pnl)}")
                
                # Find best pattern
                valid_patterns = [(p, s) for p, s in pattern_stats.items() 
                                if 'total_trades' in s and s['total_trades'] > 0]
                
                if valid_patterns:
                    best_pattern = max(valid_patterns, 
                                     key=lambda x: x[1].get('expectancy', -999))
                    self.best_pattern_label.setText(f"Best: {best_pattern[0].title()}")
                else:
                    self.best_pattern_label.setText("Best: ---")
            else:
                # No data yet
                self.win_rate_label.setText("Win Rate: ---%")
                self.pnl_label.setText("P&L: $0.00")
                self.best_pattern_label.setText("Best: ---")
                
        except Exception as e:
            # If any error, just show defaults
            self.today_label.setText("Today: 0 trades")
            self.win_rate_label.setText("Win Rate: ---%")
            self.pnl_label.setText("P&L: $0.00")
            self.best_pattern_label.setText("Best: ---")
          
    def setup_monkee_components(self):
        """Initialize Mon Kee's intelligent components"""
        try:
            # Only create if we have the required imports
            if 'MonKeeTradeExecutor' in globals() and 'CompoundGrowthTracker' in globals():
                # Create Mon Kee execution widget
                self.monkee_executor = MonKeeTradeExecutor(self.performance_analytics)
                self.monkee_executor.trade_executed.connect(self.handle_monkee_trade)
                
                # Create compound growth tracker
                self.growth_tracker = CompoundGrowthTracker(self.performance_analytics)
            else:
                # Placeholder until imports are added
                self.monkee_executor = None
                self.growth_tracker = None
        except Exception as e:
            # If Mon Kee components aren't available yet, continue without them
            print(f"Mon Kee components not initialized: {e}")
            self.monkee_executor = None
            self.growth_tracker = None

    def update_monkee_with_pattern(self, pattern_data):
        """Send current pattern to Mon Kee for analysis"""
        if hasattr(self, 'monkee_executor') and self.monkee_executor:
            self.monkee_executor.analyze_pattern(pattern_data)
    
    def create_file_section(self, layout):
        """Create file loading section"""
        self.status_label = QLabel("Load Premarket CSV to start trading")
        layout.addWidget(self.status_label)
        
        self.load_button = QPushButton("📂 Load Premarket CSV")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)
        
        self.aftermarket_button = QPushButton("🌙 Load Aftermarket CSV (9PM)")
        self.aftermarket_button.clicked.connect(self.load_aftermarket_csv)
        layout.addWidget(self.aftermarket_button)
    
    def create_trading_section(self, layout):
        # Always add the generate picks button
        self.score_button = QPushButton("⚙️ Generate Picks & Auto Buy")
        self.score_button.setObjectName("buyButton")
        self.score_button.clicked.connect(self.execute_daily_cycle)
        layout.addWidget(self.score_button)
        
        # Add settle button
        self.settle_button = QPushButton("💰 Settle Day & Learn")
        self.settle_button.setObjectName("settleButton")
        self.settle_button.clicked.connect(self.settle_and_learn)
        layout.addWidget(self.settle_button)
        
        # Add Mon Kee executor if available
        if hasattr(self, 'monkee_executor') and self.monkee_executor:
            layout.addWidget(self.monkee_executor)
    
    def create_management_section(self, layout):
        """Create system management buttons"""
        self.new_game_button = QPushButton("🔄 New Game (Reset Bank & Progress)")
        self.new_game_button.clicked.connect(self.new_game)
        layout.addWidget(self.new_game_button)
        
        self.fresh_state_button = QPushButton("⚠️ Fresh State (Wipe All Data)")
        self.fresh_state_button.setObjectName("dangerButton")
        self.fresh_state_button.clicked.connect(self.confirm_fresh_state)
        layout.addWidget(self.fresh_state_button)
    
    def create_status_section(self, layout):
        """Create progress and status displays"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
    
    def create_analysis_section(self, layout):
        """Create analysis and reporting buttons"""
        self.show_stats_button = QPushButton("📊 Show Stats & Skills")
        self.show_stats_button.clicked.connect(self.show_stats_dialog)
        layout.addWidget(self.show_stats_button)
        
        self.view_journal_button = QPushButton("📔 View Trade Journal")
        self.view_journal_button.clicked.connect(self.view_trade_journal)
        layout.addWidget(self.view_journal_button)
        
        self.monkey_journal_button = QPushButton("🐵 Mon Kee's Daily Reflection")
        self.monkey_journal_button.clicked.connect(self.show_monkey_journal_dialog)
        layout.addWidget(self.monkey_journal_button)
        
        self.vibe_analysis_button = QPushButton("🔮 Vibe Analysis & Self-Assessment")
        self.vibe_analysis_button.clicked.connect(self.show_vibe_analysis)
        layout.addWidget(self.vibe_analysis_button)
        
        self.enhanced_stats_button = QPushButton("🎯 Enhanced Analysis Stats")
        self.enhanced_stats_button.clicked.connect(self.show_enhanced_stats_dialog)
        layout.addWidget(self.enhanced_stats_button)
                
        self.debug_report_button = QPushButton("🔧 Generate Debug Report")
        self.debug_report_button.clicked.connect(self.generate_debug_report)
        layout.addWidget(self.debug_report_button)
    
        self.analyze_trades_button = QPushButton("📊 Analyze Today's Winners/Losers")
        self.analyze_trades_button.clicked.connect(self.analyze_daily_performance)
        layout.addWidget(self.analyze_trades_button)
        
    def analyze_daily_performance(self):
        """Analyze which stocks actually won vs what Mon Kee picked"""
        if not self.csv_path or not self.postmarket_csv_path:
            show_warning("Missing Data", "Please load both pre-market and post-market CSVs")
            return
        
        try:
            import pandas as pd
            
            # Load CSVs
            pre = pd.read_csv(self.csv_path)
            post = pd.read_csv(self.postmarket_csv_path)
            
            # Merge and analyze
            merged = pd.merge(
                pre[['Symbol', 'Pre-market Close', 'Pre-market Change %', 'Pre-market Gap %', 'Pre-market Volume']], 
                post[['Symbol', 'Price']], 
                on='Symbol'
            )
            
            # Calculate returns
            merged['Day_Return_%'] = ((merged['Price'] - merged['Pre-market Close']) / merged['Pre-market Close']) * 100
            
            # Find winners and losers
            winners = merged[merged['Day_Return_%'] > 0].sort_values('Day_Return_%', ascending=False)
            losers = merged[merged['Day_Return_%'] < 0].sort_values('Day_Return_%')
            
            # Create report
            report = []
            report.append("📊 DAILY PERFORMANCE ANALYSIS\n")
            report.append("TOP 5 WINNERS:")
            for _, row in winners.head().iterrows():
                report.append(f"  {row['Symbol']}: Gap {row['Pre-market Gap %']:.1f}% → Day Return {row['Day_Return_%']:.1f}%")
            
            report.append("\nTOP 5 LOSERS:")
            for _, row in losers.head().iterrows():
                report.append(f"  {row['Symbol']}: Gap {row['Pre-market Gap %']:.1f}% → Day Return {row['Day_Return_%']:.1f}%")
            
            # Check Mon Kee's picks
            if self.broker.closed_trades:
                report.append("\nMON KEE'S PICKS:")
                for trade in self.broker.closed_trades[-5:]:
                    symbol = trade['ticker']
                    if symbol in merged['Symbol'].values:
                        row = merged[merged['Symbol'] == symbol].iloc[0]
                        report.append(f"  {symbol}: Gap {row['Pre-market Gap %']:.1f}% → Day Return {row['Day_Return_%']:.1f}%")
            
            # Show in dialog
            show_info("Performance Analysis", "\n".join(report))
            
        except Exception as e:
            show_error("Analysis Error", f"Failed to analyze: {str(e)}")
    
    
    # ==========================================
    # FILE LOADING METHODS
    # ==========================================
    
    def load_csv(self):
        """Load premarket CSV file"""
        path = get_csv_file("Load Premarket CSV")
        if path:
            self.csv_path = path
            update_status_label(self.status_label, f"✅ Loaded: {os.path.basename(path)}")
            show_file_loaded_message(self.log, "Premarket CSV", path)
            self.save_progress()
    
    def load_aftermarket_csv(self):
        """Load aftermarket CSV file"""
        path = get_csv_file("Load Aftermarket CSV")
        if path:
            self.postmarket_csv_path = path
            show_file_loaded_message(self.log, "Aftermarket CSV", path)
            self.save_progress()
    
    # ==========================================
    # TRADING METHODS WITH PERFORMANCE TRACKING
    # ==========================================
    
    def execute_daily_cycle(self):
        """Execute daily trading cycle using trade controller"""
        if not self.csv_path:
            show_warning("No CSV", "Please load a Premarket CSV first!")
            return
        
        # Store starting balance for journal
        self.daily_starting_balance = self.broker.get_cash()
        
        # Use trade controller for the heavy lifting
        success, message = self.trade_controller.run_daily_cycle(self.csv_path, self.log)
        
        if success:
            log_system_message(self.log, "Daily trading cycle completed successfully", "SUCCESS")
        else:
            log_system_message(self.log, f"Daily cycle failed: {message}", "ERROR")
        
        self.update_displays()
    
    def settle_and_learn(self):
        """Settle trades and learn from results with performance tracking"""
        if not self.postmarket_csv_path:
            show_warning("No Data", "Please load Aftermarket CSV first!")
            return
        
        if not self.broker.get_open_trades():
            show_warning("No Trades", "No open trades to settle!")
            return
        
        # Use trade controller for settlement
        success, result = self.trade_controller.settle_trades_with_overview(self.postmarket_csv_path, self.log)
        
        if success:
            # Record trades in performance system
            self.record_trades_to_performance(result)
            
            # Save trade journal
            self.trade_controller.save_trade_journal()
            
            # Generate Mon Kee's journal entry
            self.save_monkey_journal()
            
            # Update displays
            self.update_displays()
            
            log_system_message(
                self.log, 
                f"Settlement complete: {result['win_rate']:.1f}% win rate, {format_currency(result['total_pnl'])} P&L",
                "SUCCESS"
            )
        else:
            log_system_message(self.log, f"Settlement failed: {result}", "ERROR")
    
    def record_trades_to_performance(self, settlement_result):
        """Record completed trades to performance analytics"""
        for trade in settlement_result.get('trades', []):
            # Prepare trade data for analytics
            trade_data = {
                'timestamp': datetime.now(),
                'symbol': trade['ticker'],
                'pattern': trade.get('best_predictor', 'unknown'),
                'result': 'win' if trade.get('pnl', 0) > 0 else 'loss',
                'pnl': trade.get('pnl', 0),
                'entry_price': trade.get('buy_price', 0),
                'exit_price': trade.get('close_price', 0),
                'shares': trade.get('shares', 0),
                'hold_time_minutes': 390,  # Full day for now
                'market_condition': self.detect_market_condition(),
                'slippage_adjusted': trade.get('slippage_adjusted', False)
            }
            
            # Record to performance system
            self.performance_analytics.record_trade_result(trade_data)
            if hasattr(self, 'performance_panel'):
                self.performance_panel.refresh_display()
    
    def detect_market_condition(self):
        """Simple market condition detection"""
        hour = datetime.now().hour
        
        if 9 <= hour < 10:
            return 'high_volatility'
        elif 10 <= hour < 11:
            return 'trending_up'
        elif 11 <= hour < 14:
            return 'choppy'
        elif 14 <= hour < 15:
            return 'trending_down'
        else:
            return 'low_volatility'
    
    # ==========================================
    # SYSTEM MANAGEMENT METHODS
    # ==========================================
    
    def new_game(self):
        """Start new game while preserving XP"""
        if not confirm_reset_action("New Game"):
            return
        
        # Reset broker and trading state
        self.broker.cash = self.bank_starting_amount
        self.broker.positions = {}
        self.broker.open_trades = []
        self.broker.closed_trades = []
        
        # Reset file paths
        self.csv_path = None
        self.postmarket_csv_path = None
        self.daily_starting_balance = self.bank_starting_amount
        
        # Reset trade controller
        self.trade_controller.reset_daily_data()
        
        # Update displays
        self.log.clear()
        log_system_message(self.log, f"🦍 New Game started! Bank reset to {format_currency(self.bank_starting_amount)}", "SUCCESS")
        log_system_message(self.log, f"💡 XP preserved - Total: {self.xp_tracker.get_total_xp():,}", "INFO")
        
        self.update_displays()
    
    def confirm_fresh_state(self):
        """Confirm and execute fresh state reset"""
        if not confirm_reset_action("Fresh State (Wipe All Data)"):
            return
        
        self.fresh_state()
    
    def fresh_state(self):
        """Wipe all data and start completely fresh"""
        # Remove all data files
        files_to_remove = [
            "progress.json", "broker_state.json", "trade_journal.json",
            "skills.json", "training_data.csv", "monkeymania_online_model.pkl",
            "monkey_journal.json", "data/confidence_bins.json", "data/confidence_drift.json",
            "performance_data.json"  # Add performance data
        ]
        
        for filename in files_to_remove:
            if os.path.exists(filename):
                os.remove(filename)
        
        # Reload systems
        self.xp_tracker.reload_state()
        self.learning_model = load_model()
        self.trade_controller = create_trade_controller(self.broker, self.learning_model)
        self.performance_analytics = PerformanceAnalytics()  # Reset performance
        
        # Reset UI
        self.log.clear()
        self.new_game()
        log_system_message(self.log, "⚠️ Fresh State executed: all progress and data wiped", "WARNING")
    
    # ==========================================
    # DISPLAY AND UI UPDATE METHODS
    # ==========================================
    
    def update_displays(self):
        """Update all UI displays"""
        self.update_cash_display()
        update_all_skills_ui(self)
        self.xp_sidebar.update_skills()
        self.update_performance_widget()
        self.save_progress()
    
    def update_cash_display(self):
        """Update cash balance display"""
        balance = self.broker.get_cash()
        self.cash_balance_label.setText(f"Bank Balance: {format_currency(balance)}")
    
    def get_monkey_art(self):
        """Generate dynamic monkey art based on performance"""
        balance = self.broker.get_cash()
        avg_level = self.xp_tracker.get_average_level()
        
        # Determine monkey mood based on balance
        if balance > 15000:
            mood = "$ $"
            status = "🤑 Crushing It!"
        elif balance > 10000:
            mood = "^ ^"
            status = "😊 Profitable!"
        elif balance > 8000:
            mood = "o o"
            status = "😐 Learning..."
        else:
            mood = "- -"
            status = "😔 Rough Day..."
        
        # Determine title based on average level
        if avg_level >= 75:
            title = "🏆 Grandmaster Ape"
        elif avg_level >= 50:
            title = "👑 Expert Trader"
        elif avg_level >= 25:
            title = "📊 Skilled Analyst"
        elif avg_level >= 10:
            title = "📈 Learning Monkey"
        else:
            title = "🐵 Novice Trader"
        
        # Create ASCII art
        lines = [
            "",
            "        🏢 MonkeyMania Trading Desk 🏢",
            "",
            "                    .=\"=.",
            "                  _/.-.-.\\_",
            f"                 ( ( {mood} ) )    {title}",
            f"                  |/  \"  |     📈 Avg Level: {avg_level}",
            f"                   '---'        💰 Balance: {format_currency(balance)}",
            f"                   /````\\       ⚡ Status: {status}",
            "                  /      \\",
            "",
            "        ╔════════════════════════════════╗",
            "        ║  📊 TRADING TERMINAL v3.0 📊   ║",
            "        ╚════════════════════════════════╝",
            ""
        ]
        
        return "\n".join(lines)
    
    # ==========================================
    # DIALOG AND ANALYSIS METHODS
    # ==========================================
    
    def show_stats_dialog(self):
        """Show comprehensive stats dialog"""
        dlg = QDialog(self)
        dlg.setWindowTitle("🦍 MonkeyMania Stats & Skills")
        dlg.resize(700, 600)
        
        layout = QVBoxLayout()
        
        # Create text display
        text = QTextEdit()
        text.setReadOnly(True)
        
        # Generate COMPLETE stats display
        stats_lines = []
        stats_lines.append("🦍 MONKEYMANIA COMPLETE STATS")
        stats_lines.append("=" * 50)
        
        # Overall summary
        total_xp = self.xp_tracker.get_total_xp()
        avg_level = self.xp_tracker.get_average_level()
        stats_lines.append(f"Total XP: {total_xp:,}")
        stats_lines.append(f"Average Level: {avg_level}")
        stats_lines.append("")
        
        # ALL SKILLS - No truncation
        stats_lines.append("🏆 ALL SKILLS:")
        skills_summary = self.xp_tracker.get_skills_summary()
        
        for i, skill in enumerate(skills_summary, 1):
            level = skill['level']
            xp = skill['xp']
            progress = skill['progress']
            display_name = skill['display_name']
            
            # Add skill status emoji
            if level >= 75:
                status_emoji = "🏆"
            elif level >= 50:
                status_emoji = "👑"
            elif level >= 25:
                status_emoji = "⭐"
            elif level >= 10:
                status_emoji = "📈"
            else:
                status_emoji = "🌱"
            
            stats_lines.append(
                f"{i:2}. {status_emoji} {display_name}: Level {level} "
                f"({xp:,} XP) - {progress:.1f}% to next"
            )
        
        stats_lines.append("")
        
        # Trading performance
        performance = self.trade_controller.get_recent_performance()
        if performance:
            stats_lines.append("📊 RECENT TRADING PERFORMANCE:")
            stats_lines.append(f"Recent Trades: {performance['total_trades']}")
            stats_lines.append(f"Wins/Losses: {performance['wins']}/{performance['losses']}")
            stats_lines.append(f"Win Rate: {performance['win_rate']:.1f}%")
            stats_lines.append(f"Total P&L: {format_currency(performance['total_pnl'])}")
            stats_lines.append(f"Avg P&L/Trade: {format_currency(performance['avg_pnl_per_trade'])}")
            stats_lines.append("")
        
        # Pattern performance from analytics
        pattern_stats = self.performance_analytics.get_pattern_stats()
        if any(s['total_trades'] > 0 for s in pattern_stats.values()):
            stats_lines.append("🎯 PATTERN PERFORMANCE:")
            for pattern, stats in pattern_stats.items():
                if stats['total_trades'] > 0:
                    stats_lines.append(
                        f"  {pattern.title()}: {stats['win_rate']}% WR, "
                        f"${stats['expectancy']} expectancy ({stats['total_trades']} trades)"
                    )
            stats_lines.append("")
        
        # Achievements section
        achievements = []
        for skill in skills_summary:
            level = skill['level']
            name = skill['display_name']
            if level >= 99:
                achievements.append(f"🏆 {name} - GRANDMASTER!")
            elif level >= 75:
                achievements.append(f"👑 {name} - Master Level!")
            elif level >= 50:
                achievements.append(f"⭐ {name} - Expert Level!")
            elif level >= 25:
                achievements.append(f"📈 {name} - Advanced Level!")
        
        if achievements:
            stats_lines.append("🎖️ ACHIEVEMENTS:")
            stats_lines.extend(achievements)
            stats_lines.append("")
        
        # Current status
        balance = self.broker.get_cash()
        if balance > 15000:
            monkey_status = "🤑 CRUSHING IT!"
        elif balance > 10000:
            monkey_status = "😊 PROFITABLE!"
        elif balance > 8000:
            monkey_status = "😐 LEARNING..."
        else:
            monkey_status = "😔 ROUGH PATCH..."
        
        stats_lines.append("🐵 MON KEE STATUS:")
        stats_lines.append(f"Current Balance: {format_currency(balance)}")
        stats_lines.append(f"Mood: {monkey_status}")
        
        # Join all lines
        stats_text = "\n".join(stats_lines)
        
        text.setText(stats_text)
        layout.addWidget(text)
        
        # Close button
        close_button = QPushButton("📊 Close Stats")
        close_button.clicked.connect(dlg.close)
        layout.addWidget(close_button)
        
        dlg.setLayout(layout)
        dlg.exec_()
    
    def show_enhanced_stats_dialog(self):
        """Show enhanced analyzer statistics"""
        dlg = QDialog(self)
        dlg.setWindowTitle("🎯 Enhanced Analysis Stats")
        dlg.resize(600, 400)
        
        layout = QVBoxLayout()
        text = QTextEdit()
        text.setReadOnly(True)
        
        # Get analyzer stats
        if hasattr(self.trade_controller, 'enhanced_analyzer'):
            stats = self.trade_controller.enhanced_analyzer.get_analyzer_stats()
            
            report = ["🎯 ENHANCED ANALYZER PERFORMANCE\n"]
            report.append("=" * 40)
            
            for analyzer, data in stats.items():
                report.append(f"\n{analyzer.replace('_', ' ').title()}:")
                report.append(f"  Accuracy: {data['accuracy']*100:.1f}%")
                report.append(f"  Predictions: {data['total_predictions']}")
                report.append(f"  Confidence: {data['confidence_level']}")
            
            # Add skill progress
            report.append("\n\n🆕 NEW SKILL PROGRESS:")
            new_skills = ['candle_reading', 'tape_reading', 'opening_drive_mastery']
            
            for skill in new_skills:
                level = self.xp_tracker.get_skill_level(skill)
                xp = self.xp_tracker.skills_xp.get(skill, 0)
                progress = get_xp_progress(xp)
                
                report.append(f"\n{self.xp_tracker.skill_display_names.get(skill, skill)}:")
                report.append(f"  Level: {level}")
                report.append(f"  Progress: {progress:.1f}% to next level")
            
            text.setPlainText("\n".join(report))
        
        layout.addWidget(text)
        dlg.setLayout(layout)
        dlg.exec_()

    # Add button to GUI
    self.enhanced_stats_button = QPushButton("🎯 Enhanced Analysis Stats")
    self.enhanced_stats_button.clicked.connect(self.show_enhanced_stats_dialog)
    layout.addWidget(self.enhanced_stats_button)
        
    def show_vibe_analysis(self):
        """Show vibe analysis and self-assessment"""
        dlg = QDialog(self)
        dlg.setWindowTitle("🔮 Vibe Analysis & Self-Assessment")
        dlg.resize(700, 600)
        
        layout = QVBoxLayout()
        
        # Monkey art header
        header_label = QLabel(self.get_monkey_art())
        header_label.setFont(QFont("Consolas", 8))
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("color: #4CAF50; background-color: #0a0a0a; padding: 10px;")
        layout.addWidget(header_label)
        
        # Analysis text
        analysis_text = QTextEdit()
        analysis_text.setReadOnly(True)
        
        # Generate self-assessment
        recent_trades = []
        if hasattr(self.trade_controller, 'broker') and self.trade_controller.broker.closed_trades:
            recent_trades = self.trade_controller.broker.closed_trades[-10:]  # Last 10 trades
        
        assessment = generate_self_assessment(
            self.xp_tracker.skills_xp,
            recent_trades,
            None  # Shadow results would come from last settlement
        )
        
        analysis_text.setPlainText(assessment)
        layout.addWidget(analysis_text)
        
        # Close button
        close_button = QPushButton("🔮 Close Analysis")
        close_button.clicked.connect(dlg.close)
        layout.addWidget(close_button)
        
        dlg.setLayout(layout)
        dlg.exec_()
    
    def view_trade_journal(self):
        """View trade journal"""
        journal_path = "trade_journal.json"
        if not os.path.exists(journal_path):
            show_info("Trade Journal", "No trade journal found yet.")
            return
        
        try:
            with open(journal_path, "r") as f:
                journal = json.load(f)
            
            if not journal:
                show_info("Trade Journal", "Trade journal is empty.")
                return
            
            # Create journal dialog
            dlg = QDialog(self)
            dlg.setWindowTitle("📔 Trade Journal")
            dlg.resize(700, 500)
            
            layout = QVBoxLayout()
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            
            # Format journal entries
            journal_lines = ["📔 Trade Journal Summary\n"]
            
            for entry in journal[-10:]:  # Last 10 entries
                cash_balance = entry.get('cash_balance', 0.0)
                journal_lines.append(f"Date: {entry['date']} | Balance: {format_currency(cash_balance)}")
                journal_lines.append("Trades:")
                
                for trade in entry.get('trades', []):
                    outcome = "Win" if trade.get('pnl', 0) > 0 else "Loss"
                    journal_lines.append(
                        f"  {trade['ticker']}: {trade['shares']} shares @ {format_currency(trade['buy_price'])} "
                        f"→ {format_currency(trade.get('close_price', 0))} | "
                        f"P&L: {format_currency(trade.get('pnl', 0))} ({outcome})"
                    )
                
                journal_lines.append("")
            
            text_edit.setPlainText("\n".join(journal_lines))
            layout.addWidget(text_edit)
            
            # Close button
            close_button = QPushButton("📔 Close Journal")
            close_button.clicked.connect(dlg.close)
            layout.addWidget(close_button)
            
            dlg.setLayout(layout)
            dlg.exec_()
            
        except Exception as e:
            show_error("Journal Error", f"Error reading trade journal: {str(e)}")
            
    # ==========================================
    # MON KEE INTELLIGENT TRADING METHODS
    # ==========================================
  
    def handle_monkee_trade(self, trade_data):
        """Handle trade execution from Mon Kee"""
        # Extract trade details
        pattern = trade_data['pattern']
        size_type = trade_data['size_type']
        position_size = trade_data['position_size']
        confidence = trade_data['confidence']
        
        # Log Mon Kee's decision
        log_system_message(self.log, f"🐒 Mon Kee executing {size_type} trade: ${position_size:.2f}", "INFO")
        log_system_message(self.log, f"Pattern confidence: {confidence:.1%}", "INFO")
        
        # Execute through your existing trade controller
        if hasattr(self, 'csv_path') and self.csv_path:
            self.daily_starting_balance = self.broker.get_cash()
            
            # Run the daily cycle with Mon Kee's position size
            success, message = self.trade_controller.run_daily_cycle(
                self.csv_path, 
                self.log,
                position_size=position_size,
                size_type=size_type
            )
            
            if success:
                log_system_message(self.log, f"Trade execution complete ({size_type} mode)", "SUCCESS")
            else:
                log_system_message(self.log, f"Trade execution failed: {message}", "ERROR")
            
            self.update_displays()
    def create_wealth_tab(self):
        """Create the wealth tracking tab"""
        layout = QVBoxLayout()
        layout.addWidget(self.growth_tracker)
        self.wealth_tab.setLayout(layout)

    def execute_daily_cycle_with_size(self, position_size, size_type):
        """Execute daily cycle with Mon Kee's recommended position size"""
        if not self.csv_path:
            show_warning("No CSV", "Please load a Premarket CSV first!")
            return
        
        # Store starting balance for journal
        self.daily_starting_balance = self.broker.get_cash()
        
        # Pass position size to trade controller
        # You may need to modify your trade_controller.run_daily_cycle to accept position_size
        success, message = self.trade_controller.run_daily_cycle(
            self.csv_path, 
            self.log,
            position_size=position_size,
            size_type=size_type
        )
        
        if success:
            log_system_message(self.log, f"Daily trading cycle completed ({size_type} size)", "SUCCESS")
        else:
            log_system_message(self.log, f"Daily cycle failed: {message}", "ERROR")
        
        self.update_displays()

    def update_monkee_with_pattern(self, pattern_data):
        """Send current pattern to Mon Kee for analysis"""
        if hasattr(self, 'monkee_executor') and self.monkee_executor:
            self.monkee_executor.analyze_pattern(pattern_data)
    
    def show_monkey_journal_dialog(self):
        """Show Mon Kee's personal journal"""
        journal_path = "monkey_journal.json"
        if not os.path.exists(journal_path):
            show_info("🐵 Mon Kee's Journal", "Mon Kee hasn't written any entries yet!")
            return
        
        try:
            with open(journal_path, "r") as f:
                journal = json.load(f)
            
            if not journal:
                show_info("🐵 Mon Kee's Journal", "Mon Kee's journal is empty.")
                return
            
            # Create journal dialog
            dlg = QDialog(self)
            dlg.setWindowTitle("🐵 Mon Kee's Personal Journal")
            dlg.resize(700, 500)
            
            layout = QVBoxLayout()
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            
            # Format journal entries
            journal_lines = ["🐵 Mon Kee's Personal Reflections\n"]
            
            # Get last 5 days of entries
            dates = sorted(journal.keys())[-5:]
            
            for date in dates:
                entries = journal[date]
                for entry in entries:
                    journal_lines.append(f"📅 {date}")
                    journal_lines.append(f"Summary: {entry['summary']}")
                    journal_lines.append(f"Mood: {entry['mood']}")
                    journal_lines.append(f"Score: {entry['score']}/100")
                    journal_lines.append(f"Win Rate: {entry['win_rate']}%")
                    
                    if 'insights' in entry and entry['insights']:
                        journal_lines.append("Insights:")
                        for insight in entry['insights']:
                            journal_lines.append(f"  • {insight}")
                    
                    journal_lines.append("")
            
            text_edit.setPlainText("\n".join(journal_lines))
            layout.addWidget(text_edit)
            
            # Close button
            close_button = QPushButton("🐵 Close Journal")
            close_button.clicked.connect(dlg.close)
            layout.addWidget(close_button)
            
            dlg.setLayout(layout)
            dlg.exec_()
            
        except Exception as e:
            show_error("Journal Error", f"Error reading Mon Kee's journal: {str(e)}")
    
    def show_enhanced_stats_dialog(self):
        """Show enhanced analyzer statistics"""
        dlg = QDialog(self)
        dlg.setWindowTitle("🎯 Enhanced Analysis Stats")
        dlg.resize(600, 400)
        
        layout = QVBoxLayout()
        text = QTextEdit()
        text.setReadOnly(True)
        
        # Get analyzer stats
        if hasattr(self.trade_controller, 'enhanced_analyzer'):
            stats = self.trade_controller.enhanced_analyzer.get_analyzer_stats()
            
            report = ["🎯 ENHANCED ANALYZER PERFORMANCE\n"]
            report.append("=" * 40)
            
            for analyzer, data in stats.items():
                report.append(f"\n{analyzer.replace('_', ' ').title()}:")
                report.append(f"  Accuracy: {data['accuracy']*100:.1f}%")
                report.append(f"  Predictions: {data['total_predictions']}")
                report.append(f"  Confidence: {data['confidence_level']}")
            
            # Add skill progress
            report.append("\n\n🆕 NEW SKILL PROGRESS:")
            new_skills = ['candle_reading', 'tape_reading', 'opening_drive_mastery', 
                          'relative_strength', 'pattern_synthesis', 'market_intuition']
            
            for skill in new_skills:
                level = self.xp_tracker.get_skill_level(skill)
                xp = self.xp_tracker.skills_xp.get(skill, 0)
                progress = get_xp_progress(xp)
                
                report.append(f"\n{self.xp_tracker.skill_display_names.get(skill, skill)}:")
                report.append(f"  Level: {level}")
                report.append(f"  Progress: {progress:.1f}% to next level")
            
            text.setPlainText("\n".join(report))
        else:
            text.setPlainText("Enhanced analyzer not initialized yet.")
        
        layout.addWidget(text)
        dlg.setLayout(layout)
        dlg.exec_()
    
    def generate_debug_report(self):
        """Generate debug report for analysis"""
        try:
            from monkee_debug_report import MonKeeDebugReport
            reporter = MonKeeDebugReport()
            
            # Generate main report
            filename = reporter.generate_full_report(self)
            
            # Also analyze current CSV if loaded
            if self.csv_path:
                csv_analysis = reporter.generate_csv_analysis(self.csv_path)
                csv_filename = f"csv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(csv_filename, 'w') as f:
                    json.dump(csv_analysis, f, indent=2)
                
                show_info("Debug Report", f"Reports generated:\n{filename}\n{csv_filename}")
            else:
                show_info("Debug Report", f"Report generated: {filename}")
                
        except Exception as e:
            show_error("Debug Error", f"Failed to generate report: {str(e)}")
        
    def save_monkey_journal(self):
        """Save Mon Kee's journal entry"""
        try:
            # Get today's trades
            trades_today = [t for t in self.broker.closed_trades 
                          if t.get('exit_time', '').startswith(datetime.now().strftime('%Y-%m-%d'))]
            
            if trades_today:
                # Get slippage data if available
                slippage_data = None
                if hasattr(self.broker, 'get_slippage_stats'):
                    slippage_data = self.broker.get_slippage_stats()
                
                # Create journal entry
                from monkey_journal import MonkeyJournal
                journal = MonkeyJournal()
                
                # Use slippage-aware summary if available
                if hasattr(journal, 'write_daily_summary_with_slippage'):
                    journal.write_daily_summary_with_slippage(
                        trades_today,
                        {},  # Best predictor stats
                        slippage_data
                    )
                else:
                    journal.write_daily_summary(trades_today, {})
                
                log_system_message(self.log, "📔 Mon Kee finished writing in his journal!", "INFO")
                
        except Exception as e:
            log_system_message(self.log, f"Error saving Mon Kee's journal: {str(e)}", "WARNING")
    
    # ==========================================
    # SESSION MANAGEMENT
    # ==========================================
    
    def save_progress(self):
        """Save current progress to file"""
        progress_data = {
            "csv_path": self.csv_path,
            "postmarket_csv_path": self.postmarket_csv_path,
            "daily_starting_balance": self.daily_starting_balance,
            "total_xp": self.xp_tracker.get_total_xp(),
            "average_level": self.xp_tracker.get_average_level()
        }
        
        with open("progress.json", "w") as f:
            json.dump(progress_data, f, indent=2)
        
        # Save broker state
        self.broker.save_state()
        
        # Save XP state
        self.xp_tracker.save_state()
        
        # Save performance data
        self.performance_analytics.save_performance_data()
    
    def restore_session(self):
        """Restore previous session state"""
        if os.path.exists("progress.json"):
            try:
                with open("progress.json", "r") as f:
                    data = json.load(f)
                    self.csv_path = data.get("csv_path")
                    self.postmarket_csv_path = data.get("postmarket_csv_path")
                    self.daily_starting_balance = data.get("daily_starting_balance", self.bank_starting_amount)
                    
                    if self.csv_path:
                        update_status_label(self.status_label, f"✅ Restored: {os.path.basename(self.csv_path)}")
                        
            except Exception as e:
                log_system_message(self.log, f"Error restoring session: {str(e)}", "WARNING")
        
        # Update displays
        self.update_displays()
        
        # Welcome message
        total_xp = self.xp_tracker.get_total_xp()
        if total_xp > 0:
            log_system_message(self.log, f"💡 Session restored - Total XP: {total_xp:,}", "INFO")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("MonkeyMania")
    app.setApplicationVersion("3.1 - Performance Edition")
    
    # Create and show GUI
    gui = MonkeyManiaGUI()
    gui.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()