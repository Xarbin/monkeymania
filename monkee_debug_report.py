# monkee_debug_report.py
"""Generate a comprehensive debug report for Mon Kee's trading system"""

import json
import pandas as pd
from datetime import datetime
import os

class MonKeeDebugReport:
    def __init__(self):
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'MonkeyMania 3.1',
            'sections': {}
        }
    
    def generate_full_report(self, gui_instance):
        """Generate complete system report"""
        print("🐒 Generating Mon Kee Debug Report...")
        
        # 1. System State
        self.report_data['sections']['system_state'] = {
            'current_balance': gui_instance.broker.get_cash(),
            'starting_balance': gui_instance.bank_starting_amount,
            'open_trades': len(gui_instance.broker.get_open_trades()),
            'total_trades': len(gui_instance.broker.closed_trades),
            'csv_loaded': {
                'premarket': gui_instance.csv_path,
                'postmarket': gui_instance.postmarket_csv_path
            }
        }
        
        # 2. Performance Metrics
        if hasattr(gui_instance, 'performance_analytics'):
            pattern_stats = gui_instance.performance_analytics.get_pattern_stats()
            self.report_data['sections']['performance'] = {
                'pattern_stats': pattern_stats,
                'quick_insights': gui_instance.performance_analytics.get_quick_insights()
            }
        
        # 3. Recent Trades
        recent_trades = gui_instance.broker.closed_trades[-10:] if gui_instance.broker.closed_trades else []
        self.report_data['sections']['recent_trades'] = recent_trades
        
        # 4. XP/Skills State
        if hasattr(gui_instance, 'xp_tracker'):
            self.report_data['sections']['xp_state'] = {
                'total_xp': gui_instance.xp_tracker.get_total_xp(),
                'skills': gui_instance.xp_tracker.get_skills_summary()
            }
        
        # 5. ML Model State
        self.report_data['sections']['ml_model'] = {
            'model_exists': os.path.exists('monkeymania_online_model.pkl'),
            'training_data_exists': os.path.exists('training_data.csv')
        }
        
        # 6. Recent Errors/Issues
        self.report_data['sections']['data_quality'] = self._check_data_quality(gui_instance)
        
        # Save report
        filename = f"monkee_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        print(f"✅ Report saved: {filename}")
        return filename
    
    def _check_data_quality(self, gui_instance):
        """Check for common data issues"""
        issues = []
        
        # Check if CSV is loaded
        if not gui_instance.csv_path:
            issues.append("No premarket CSV loaded")
        
        # Check recent trade data
        if gui_instance.broker.closed_trades:
            for trade in gui_instance.broker.closed_trades[-5:]:
                # Check for NaN or extreme values
                if trade.get('buy_price', 0) == 0:
                    issues.append(f"Zero buy price for {trade.get('ticker', 'UNKNOWN')}")
                if abs(trade.get('pnl', 0)) > 1000:
                    issues.append(f"Extreme P&L for {trade.get('ticker', 'UNKNOWN')}: ${trade.get('pnl', 0)}")
        
        return {
            'issues_found': len(issues),
            'issues': issues
        }
    
    def generate_csv_analysis(self, csv_path):
        """Analyze CSV file for issues"""
        try:
            df = pd.read_csv(csv_path)
            
            analysis = {
                'file': csv_path,
                'rows': len(df),
                'columns': list(df.columns),
                'sample_data': df.head(3).to_dict(),
                'issues': []
            }
            
            # Check for extreme values
            if 'pre-market change %' in df.columns:
                extreme_moves = df[abs(df['pre-market change %']) > 100]
                if not extreme_moves.empty:
                    analysis['issues'].append(f"Found {len(extreme_moves)} stocks with >100% moves")
                    analysis['extreme_examples'] = extreme_moves[['symbol', 'pre-market change %']].head(5).to_dict()
            
            # Check for NaN values
            nan_counts = df.isna().sum()
            if nan_counts.any():
                analysis['nan_columns'] = nan_counts[nan_counts > 0].to_dict()
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
            
    def verify_ai_functionality(self, gui_instance):
        """Verify Mon Kee's AI is actually working"""
        ai_report = {
            'is_learning': False,
            'is_adapting': False,
            'is_improving': False,
            'evidence': []
        }
        
        # 1. Check if ML model is updating
        if os.path.exists('training_data.csv'):
            df = pd.read_csv('training_data.csv')
            ai_report['training_samples'] = len(df)
            ai_report['is_learning'] = len(df) > 0
            
            # Check if predictions vary (not just returning same value)
            if hasattr(gui_instance, 'trade_controller'):
                recent_predictions = []
                for trade in gui_instance.broker.closed_trades[-10:]:
                    if 'ml_confidence' in trade:
                        recent_predictions.append(trade['ml_confidence'])
                
                if recent_predictions:
                    ai_report['prediction_variance'] = np.std(recent_predictions)
                    ai_report['is_adapting'] = np.std(recent_predictions) > 0.01
                    ai_report['recent_predictions'] = recent_predictions
        
        # 2. Check if pattern performance is being tracked
        if hasattr(gui_instance, 'performance_analytics'):
            pattern_stats = gui_instance.performance_analytics.get_pattern_stats()
            patterns_with_data = sum(1 for p in pattern_stats.values() if p.get('total_trades', 0) > 0)
            ai_report['patterns_tracked'] = patterns_with_data
            ai_report['is_pattern_learning'] = patterns_with_data > 0
        
        # 3. Check if win rate is improving over time
        if len(gui_instance.broker.closed_trades) >= 20:
            # Compare first 10 trades vs last 10 trades
            first_10 = gui_instance.broker.closed_trades[:10]
            last_10 = gui_instance.broker.closed_trades[-10:]
            
            first_wins = sum(1 for t in first_10 if t.get('pnl', 0) > 0)
            last_wins = sum(1 for t in last_10 if t.get('pnl', 0) > 0)
            
            ai_report['first_10_win_rate'] = first_wins / 10
            ai_report['last_10_win_rate'] = last_wins / 10
            ai_report['is_improving'] = last_wins > first_wins
        
        # 4. Check if position sizing varies based on confidence
        position_sizes = []
        for trade in gui_instance.broker.closed_trades[-10:]:
            if 'shares' in trade and 'buy_price' in trade:
                position_sizes.append(trade['shares'] * trade['buy_price'])
        
        if position_sizes:
            ai_report['position_size_variance'] = np.std(position_sizes)
            ai_report['is_sizing_dynamic'] = np.std(position_sizes) > 10
        
        # 5. Generate AI verdict
        ai_score = sum([
            ai_report['is_learning'],
            ai_report['is_adapting'],
            ai_report.get('is_pattern_learning', False),
            ai_report.get('is_sizing_dynamic', False)
        ])
        
        ai_report['ai_score'] = f"{ai_score}/4"
        ai_report['verdict'] = self._get_ai_verdict(ai_score)
        
        return ai_report

    def _get_ai_verdict(self, score):
        """Generate human-readable AI verdict"""
        if score == 4:
            return "✅ FULLY FUNCTIONAL AI - Mon Kee is learning, adapting, and improving!"
        elif score >= 3:
            return "🟨 MOSTLY FUNCTIONAL - AI is working but could be better optimized"
        elif score >= 2:
            return "🟧 PARTIALLY FUNCTIONAL - Some AI features working, others need attention"
        else:
            return "🟥 LIMITED FUNCTION - AI components need troubleshooting"
    
    def generate_full_report(self, gui_instance):
    # ... existing code ...
    
    # Add this new section:
    # Market Analysis Data
    if hasattr(gui_instance.trade_controller, 'market_analyzer'):
        sections['market_analysis'] = {
            'last_analysis': getattr(gui_instance.trade_controller, 'last_market_analysis', {}),
            'market_insights': getattr(gui_instance.trade_controller.market_analyzer, 'market_insights', {})
        }