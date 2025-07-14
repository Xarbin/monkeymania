# gui_handlers.py - PyQt5 Version
# Common GUI handler functions for MonkeyMania

from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from datetime import datetime
import os

def get_csv_file(title="Select CSV File"):
    """Open file dialog to select CSV file"""
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "CSV Files (*.csv);;All Files (*.*)"
    )
    return file_path if file_path else None

def show_info(title, message):
    """Show information message box"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.exec_()

def show_warning(title, message):
    """Show warning message box"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.exec_()

def show_error(title, message):
    """Show error message box"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.exec_()

def show_question(title, message):
    """Show yes/no question dialog"""
    reply = QMessageBox.question(
        None,
        title,
        message,
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    return reply == QMessageBox.Yes

def log_system_message(log_widget, message, level="INFO"):
    """Log a system message to the log widget"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Color coding based on level
    color_map = {
        "INFO": "#00ff00",      # Green
        "WARNING": "#ffaa00",   # Orange
        "ERROR": "#ff0000",     # Red
        "SUCCESS": "#00ffff",   # Cyan
        "TRADE": "#ffff00"      # Yellow
    }
    
    color = color_map.get(level, "#ffffff")
    
    # Format the message with color
    formatted_message = f'<span style="color: {color}">[{timestamp}] {message}</span>'
    
    # Append to log (assuming QTextEdit with HTML support)
    log_widget.append(formatted_message)
    
    # Auto-scroll to bottom
    scrollbar = log_widget.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())

def format_currency(amount):
    """Format number as currency"""
    return f"${amount:,.2f}"

def update_status_label(label, text):
    """Update status label text"""
    label.setText(text)

def show_file_loaded_message(log_widget, file_type, file_path):
    """Show file loaded message in log"""
    filename = os.path.basename(file_path)
    log_system_message(log_widget, f"{file_type} loaded: {filename}", "SUCCESS")

def confirm_reset_action(action_name):
    """Confirm destructive action"""
    message = f"Are you sure you want to {action_name}?\n\nThis action cannot be undone."
    return show_question(f"Confirm {action_name}", message)

def format_xp(xp_amount):
    """Format XP amount with proper formatting"""
    if xp_amount >= 1000000:
        return f"{xp_amount/1000000:.1f}M XP"
    elif xp_amount >= 1000:
        return f"{xp_amount/1000:.1f}K XP"
    else:
        return f"{xp_amount} XP"

def format_percentage(value):
    """Format a percentage value for display"""
    # Check if it's already a percentage
    if abs(value) > 1:  # Values > 1 are likely already percentages
        return f"{value:.1f}%"
    else:  # Values < 1 are decimals needing conversion
        return f"{value * 100:.1f}%"
def get_manual_price(ticker, default_price=10.0):
    """Ask user to manually input a price for a ticker"""
    from PyQt5.QtWidgets import QInputDialog
    
    price, ok = QInputDialog.getDouble(
        None,
        f"Missing Price for {ticker}",
        f"No pre-market price found for {ticker}.\nPlease enter the correct price:",
        default_price,  # default value
        0.01,          # minimum
        1000.0,        # maximum
        2              # decimals
    )
    
    if ok:
        return price
    return default_price