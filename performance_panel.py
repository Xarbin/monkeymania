# performance_panel.py
# MM-B15.1: Clean performance display panel for GUI integration (PyQt5 version)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTreeWidget, QTreeWidgetItem, QDialog,
    QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from performance_analytics import PerformanceAnalytics

class PerformancePanel(QWidget):
    """Lightweight performance display panel for MonkeyMania GUI (PyQt5)"""
    
    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self.analytics = PerformanceAnalytics()
        self.create_panel()
        
    def create_panel(self):
        """Create the performance panel UI"""
        # Main layout
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("📊 Performance Analytics")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #4CAF50;
                padding: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Quick insights section
        self.insights_group = QGroupBox("Quick Insights")
        insights_layout = QVBoxLayout()
        
        self.insights_label = QLabel("Loading insights...")
        self.insights_label.setStyleSheet("font-size: 12px; padding: 5px;")
        insights_layout.addWidget(self.insights_label)
        
        self.insights_group.setLayout(insights_layout)
        layout.addWidget(self.insights_group)
        
        # Pattern performance table
        self.pattern_group = QGroupBox("Pattern Performance")
        pattern_layout = QVBoxLayout()
        
        # Create tree widget for pattern stats
        self.pattern_tree = QTreeWidget()
        self.pattern_tree.setHeaderLabels(['Pattern', 'Trades', 'Win%', 'Avg PnL', 'Expectancy'])
        self.pattern_tree.setAlternatingRowColors(True)
        
        # Set column widths
        self.pattern_tree.setColumnWidth(0, 150)
        self.pattern_tree.setColumnWidth(1, 80)
        self.pattern_tree.setColumnWidth(2, 80)
        self.pattern_tree.setColumnWidth(3, 100)
        self.pattern_tree.setColumnWidth(4, 100)
        
        pattern_layout.addWidget(self.pattern_tree)
        self.pattern_group.setLayout(pattern_layout)
        layout.addWidget(self.pattern_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.clicked.connect(self.refresh_display)
        button_layout.addWidget(self.refresh_button)
        
        self.export_button = QPushButton("📄 Export Report")
        self.export_button.clicked.connect(self.export_report)
        button_layout.addWidget(self.export_button)
        
        self.times_button = QPushButton("⏰ Best Times")
        self.times_button.clicked.connect(self.show_best_times)
        button_layout.addWidget(self.times_button)
        
        layout.addLayout(button_layout)
        
        # Set layout
        self.setLayout(layout)
        
        # Style the panel
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTreeWidget {
                background-color: #1a1a1a;
                border: 1px solid #444;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 2px;
            }
            QTreeWidget::item:selected {
                background-color: #3b3b3b;
            }
            QPushButton {
                min-width: 100px;
                padding: 5px;
            }
        """)
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds
        
        # Initial refresh
        self.refresh_display()
        
    def refresh_display(self):
        """Refresh all performance displays"""
        # Update insights
        insights = self.analytics.get_quick_insights()
        if insights:
            insight_text = "\n".join(insights[:4])  # Show top 4 insights
            self.insights_label.setText(insight_text)
        else:
            self.insights_label.setText("No trades analyzed yet")
        
        # Update pattern table
        self.pattern_tree.clear()
        
        pattern_stats = self.analytics.get_pattern_stats()
        for pattern, stats in pattern_stats.items():
            if stats['total_trades'] > 0:
                # Create tree item
                item = QTreeWidgetItem([
                    pattern.title(),
                    str(stats['total_trades']),
                    f"{stats['win_rate']}%",
                    f"${stats['avg_pnl']:.2f}",
                    f"${stats['expectancy']:.2f}"
                ])
                
                # Color code based on performance
                if stats['expectancy'] > 0:
                    for i in range(5):
                        item.setForeground(i, QColor(76, 175, 80))  # Green
                else:
                    for i in range(5):
                        item.setForeground(i, QColor(244, 67, 54))  # Red
                
                self.pattern_tree.addTopLevelItem(item)
        
    def show_best_times(self):
        """Show popup with best trading times"""
        dialog = QDialog(self.parent())
        dialog.setWindowTitle("⏰ Best Trading Times")
        dialog.resize(500, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Trading Time Performance")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Time stats display
        text_display = QTextEdit()
        text_display.setReadOnly(True)
        text_display.setFont(QFont("Consolas", 10))
        
        time_stats = self.analytics.get_best_trading_times()
        
        text_lines = ["Time Slot     | Trades | Win% | Avg PnL  | Total PnL"]
        text_lines.append("-" * 55)
        
        for slot in time_stats:
            line = f"{slot['time_slot']:13} | {slot['total_trades']:6} | {slot['win_rate']:4}% | ${slot['avg_pnl']:8.2f} | ${slot['total_pnl']:9.2f}"
            text_lines.append(line)
        
        text_display.setPlainText("\n".join(text_lines))
        layout.addWidget(text_display)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def export_report(self):
        """Export detailed performance report"""
        filepath = self.analytics.export_report()
        
        # Show confirmation dialog
        dialog = QDialog(self.parent())
        dialog.setWindowTitle("Report Exported")
        dialog.resize(400, 150)
        
        layout = QVBoxLayout()
        
        label = QLabel(f"Performance report saved to:\n{filepath}")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("padding: 20px;")
        layout.addWidget(label)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.close)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def record_trade(self, trade_data):
        """
        Record a trade result (called from main GUI after trade completion)
        
        This is the integration point - main GUI calls this method
        """
        self.analytics.record_trade_result(trade_data)
        self.refresh_display()  # Auto-refresh after recording
        
    def get_pattern_recommendation(self, pattern):
        """Get quick recommendation for a pattern (for GUI tooltips)"""
        stats = self.analytics.get_pattern_stats(pattern)
        if not stats or stats['total_trades'] < 5:
            return "Insufficient data"
        
        if stats['expectancy'] > 10:
            return f"✅ Strong pattern ({stats['win_rate']}% WR)"
        elif stats['expectancy'] > 0:
            return f"👍 Positive expectancy (${stats['expectancy']})"
        else:
            return f"⚠️ Negative expectancy (${stats['expectancy']})"