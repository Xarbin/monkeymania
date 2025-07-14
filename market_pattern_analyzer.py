# market_pattern_analyzer_fixed.py
# Robust version that handles all column name variations

import pandas as pd
import numpy as np
from datetime import datetime
from gui_handlers import log_system_message

class MarketPatternAnalyzer:
    """Analyzes patterns across the entire market to learn from stocks Mon Kee didn't trade"""
    
    def __init__(self, pattern_identifier_func):
        self.identify_pattern = pattern_identifier_func
        self.market_insights = {}
        self.data_quality_threshold = 0.8
        
    def analyze_full_market(self, premarket_df, postmarket_df, traded_tickers, log_widget=None):
        """Analyze all stocks in the market to find patterns in winners/losers"""
        # Clean and validate data first
        cleaned_pre = self._clean_dataframe(premarket_df, 'premarket')
        cleaned_post = self._clean_dataframe(postmarket_df, 'postmarket')
        
        # Ensure both dataframes have required columns
        if 'ticker' not in cleaned_pre.columns or 'ticker' not in cleaned_post.columns:
            if log_widget:
                log_system_message(log_widget, "❌ Missing ticker column in data", "ERROR")
            return {}
        
        if 'close_price' not in cleaned_post.columns:
            if log_widget:
                log_system_message(log_widget, "❌ No price column found in postmarket data", "ERROR")
            return {}
        
        # Merge on ticker
        try:
            # Select only needed columns for merge
            post_subset = cleaned_post[['ticker', 'close_price']]
            merged = pd.merge(cleaned_pre, post_subset, on='ticker', how='inner')
        except Exception as e:
            if log_widget:
                log_system_message(log_widget, f"❌ Failed to merge market data: {str(e)}", "ERROR")
            return {}
        
        # Validate merged data
        valid_data = self._validate_merged_data(merged, log_widget)
        if valid_data.empty:
            if log_widget:
                log_system_message(log_widget, "❌ No valid market data to analyze", "ERROR")
            return {}
        
        # Calculate returns only for valid rows
        if 'premarket_close' in valid_data.columns:
            valid_data['day_return'] = ((valid_data['close_price'] - valid_data['premarket_close']) / 
                                       valid_data['premarket_close']) * 100
        else:
            # If no premarket close, estimate from gap
            if 'gap_pct' in valid_data.columns:
                # Reverse engineer premarket price from gap
                valid_data['estimated_premarket'] = valid_data['close_price'] / (1 + valid_data['gap_pct']/100)
                valid_data['day_return'] = valid_data['gap_pct']  # Use gap as proxy for return
            else:
                if log_widget:
                    log_system_message(log_widget, "⚠️ No premarket prices, using price change as return", "WARNING")
                valid_data['day_return'] = 0  # Default to 0 if we can't calculate
        
        # Additional sanity check - remove unrealistic returns
        valid_data = valid_data[(valid_data['day_return'] > -99) & (valid_data['day_return'] < 1000)]
        
        # Log data quality
        if log_widget:
            self._log_data_quality(len(merged), len(valid_data), log_widget)
        
        # Analyze patterns
        market_patterns = self._analyze_patterns(valid_data, log_widget)
        
        # Find missed opportunities
        missed_opps = self._find_missed_opportunities(valid_data, traded_tickers, log_widget)
        
        return {
            'patterns': market_patterns,
            'missed_opportunities': missed_opps,
            'data_quality': {
                'total_stocks': len(merged),
                'valid_stocks': len(valid_data),
                'quality_rate': len(valid_data) / len(merged) if len(merged) > 0 else 0
            }
        }
    
    def _find_column_case_insensitive(self, df, target_names):
        """Find a column name regardless of case"""
        # Create lowercase mapping
        lowercase_cols = {col.lower(): col for col in df.columns}
        
        # Check each target name
        for target in target_names:
            if target.lower() in lowercase_cols:
                return lowercase_cols[target.lower()]
        
        return None
    
    def _clean_dataframe(self, df, source):
        """Clean and standardize dataframe columns"""
        # Create a copy to avoid modifying original
        clean_df = df.copy()
        
        # Define column mappings with multiple possible names
        column_mappings = {
            'ticker': ['symbol', 'ticker', 'sym', 'stock'],
            'close_price': ['price', 'close', 'last', 'close price', 'closing price', 
                           'last price', 'current price', 'market price', 'post-market close',
                           'postmarket close', 'post market close'],
            'premarket_close': ['pre-market close', 'premarket close', 'pre market close',
                               'pm close', 'morning close'],
            'gap_pct': ['pre-market gap %', 'premarket gap %', 'gap %', 'gap percent',
                       'gap percentage', 'pre-market gap', 'premarket gap'],
            'premarket_volume': ['pre-market volume', 'premarket volume', 'pre market volume',
                                'pm volume', 'morning volume'],
            'premarket_change': ['pre-market change %', 'premarket change %', 'pre market change %',
                                'pm change %', 'morning change %', 'pre-market change', 'premarket change']
        }
        
        # Apply mappings
        for standard_name, possible_names in column_mappings.items():
            actual_col = self._find_column_case_insensitive(clean_df, possible_names)
            if actual_col:
                clean_df[standard_name] = clean_df[actual_col]
        
        # Ensure ticker column exists and is cleaned
        if 'ticker' in clean_df.columns:
            clean_df['ticker'] = clean_df['ticker'].astype(str).str.upper().str.strip()
            # Remove invalid tickers
            clean_df = clean_df[
                (clean_df['ticker'] != 'NAN') & 
                (clean_df['ticker'] != '') &
                (clean_df['ticker'].str.len() <= 5)
            ]
        
        # Convert numeric columns
        numeric_columns = ['close_price', 'premarket_close', 'gap_pct', 'premarket_volume', 'premarket_change']
        for col in numeric_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        
        return clean_df
    
    def _validate_merged_data(self, merged_df, log_widget):
        """Validate merged data and remove rows with critical NaN values"""
        initial_count = len(merged_df)
        
        # Required columns - be flexible about premarket_close
        required_cols = ['ticker', 'close_price']
        
        # Remove rows with NaN in critical columns
        valid_df = merged_df.dropna(subset=required_cols)
        
        # Additional validation - only check close_price
        valid_df = valid_df[
            (valid_df['close_price'] > 0) & 
            (valid_df['close_price'] < 10000)
        ]
        
        # If we have premarket_close, validate it too
        if 'premarket_close' in valid_df.columns:
            valid_df = valid_df[
                (valid_df['premarket_close'] > 0) & 
                (valid_df['premarket_close'] < 10000)
            ]
        
        dropped_count = initial_count - len(valid_df)
        if dropped_count > 0 and log_widget:
            log_system_message(
                log_widget, 
                f"⚠️ Dropped {dropped_count} stocks with invalid/missing data", 
                "WARNING"
            )
        
        return valid_df
    
    def _analyze_patterns(self, valid_data, log_widget):
        """Analyze patterns in validated market data"""
        market_patterns = {}
        
        for _, stock in valid_data.iterrows():
            # Create features dict with safe defaults
            features = {
                'gap_pct': float(stock.get('gap_pct', 0)) if pd.notna(stock.get('gap_pct')) else 0,
                'premarket_volume': float(stock.get('premarket_volume', 100000)) if pd.notna(stock.get('premarket_volume')) else 100000,
                'early_premarket_move': float(stock.get('premarket_change', 0)) if pd.notna(stock.get('premarket_change')) else 0,
                'float': 10000000,  # Default
                'volume_change_pct': 0
            }
            
            # Identify pattern
            try:
                pattern = self.identify_pattern(features, stock)
            except Exception as e:
                pattern = 'unknown'
            
            # Track pattern performance
            if pattern not in market_patterns:
                market_patterns[pattern] = {
                    'total': 0,
                    'winners': 0,
                    'total_return': 0,
                    'big_winners': [],
                    'big_losers': []
                }
            
            market_patterns[pattern]['total'] += 1
            day_return = stock.get('day_return', 0)
            
            if day_return > 0:
                market_patterns[pattern]['winners'] += 1
            market_patterns[pattern]['total_return'] += day_return
            
            # Track big movers
            if day_return > 10:
                market_patterns[pattern]['big_winners'].append({
                    'ticker': stock['ticker'],
                    'return': day_return,
                    'gap': features['gap_pct']
                })
            elif day_return < -10:
                market_patterns[pattern]['big_losers'].append({
                    'ticker': stock['ticker'],
                    'return': day_return,
                    'gap': features['gap_pct']
                })
        
        # Log pattern insights
        if log_widget:
            self._log_pattern_insights(market_patterns, log_widget)
        
        return market_patterns
    
    def _log_pattern_insights(self, market_patterns, log_widget):
        """Log insights about market patterns"""
        log_system_message(log_widget, "\n🌍 MARKET-WIDE PATTERN PERFORMANCE:", "INFO")
        
        # Sort patterns by performance
        pattern_performance = []
        for pattern, stats in market_patterns.items():
            if stats['total'] > 0:
                win_rate = (stats['winners'] / stats['total']) * 100
                avg_return = stats['total_return'] / stats['total']
                pattern_performance.append((pattern, win_rate, avg_return, stats['total']))
        
        # Sort by average return
        pattern_performance.sort(key=lambda x: x[2], reverse=True)
        
        for pattern, win_rate, avg_return, total in pattern_performance:
            emoji = "🎯" if avg_return > 5 else "📈" if avg_return > 0 else "📉"
            log_system_message(
                log_widget,
                f"{emoji} {pattern}: {win_rate:.0f}% WR, {avg_return:+.1f}% avg return ({total} stocks)",
                "INFO"
            )
            
            # Show examples of big winners
            stats = market_patterns[pattern]
            if stats['big_winners']:
                best = max(stats['big_winners'], key=lambda x: x['return'])
                log_system_message(
                    log_widget,
                    f"   Best: {best['ticker']} +{best['return']:.1f}% (gap: {best['gap']:.1f}%)",
                    "SUCCESS"
                )
    
    def _find_missed_opportunities(self, valid_data, traded_tickers, log_widget):
        """Find big winners Mon Kee missed"""
        # Get winners from validated data only
        winners = valid_data[valid_data['day_return'] > 10].sort_values('day_return', ascending=False)
        
        # Find ones we didn't trade
        missed = winners[~winners['ticker'].isin(traded_tickers)]
        
        missed_list = []
        
        if len(missed) > 0 and log_widget:
            log_system_message(log_widget, "\n🙈 BIG WINNERS MON KEE MISSED:", "WARNING")
            
            for _, stock in missed.head(5).iterrows():
                # Get pattern with safe feature extraction
                features = {
                    'gap_pct': float(stock.get('gap_pct', 0)) if pd.notna(stock.get('gap_pct')) else 0,
                    'premarket_volume': float(stock.get('premarket_volume', 100000)) if pd.notna(stock.get('premarket_volume')) else 100000,
                    'early_premarket_move': float(stock.get('premarket_change', 0)) if pd.notna(stock.get('premarket_change')) else 0,
                    'float': 10000000,
                    'volume_change_pct': 0
                }
                
                try:
                    pattern = self.identify_pattern(features, stock)
                except:
                    pattern = 'unknown'
                
                log_system_message(
                    log_widget,
                    f"  {stock['ticker']}: +{stock['day_return']:.1f}% (Pattern: {pattern}, Gap: {features['gap_pct']:.1f}%)",
                    "WARNING"
                )
                
                missed_list.append({
                    'ticker': stock['ticker'],
                    'return': stock['day_return'],
                    'pattern': pattern,
                    'gap': features['gap_pct']
                })
        
        return missed_list
    
    def _log_data_quality(self, total_stocks, valid_stocks, log_widget):
        """Log data quality metrics"""
        quality_rate = (valid_stocks / total_stocks * 100) if total_stocks > 0 else 0
        
        log_system_message(log_widget, "\n📊 MARKET DATA QUALITY:", "INFO")
        log_system_message(log_widget, f"Total stocks in dataset: {total_stocks}", "INFO")
        log_system_message(log_widget, f"Valid stocks for analysis: {valid_stocks}", "INFO")
        log_system_message(log_widget, f"Data quality rate: {quality_rate:.1f}%", "INFO")
        
        if quality_rate < 50:
            log_system_message(
                log_widget, 
                "⚠️ Low data quality - results may not be representative", 
                "WARNING"
            )
    
    def generate_learning_summary(self, market_analysis):
        """Generate a summary of what Mon Kee should learn from market analysis"""
        if not market_analysis or 'patterns' not in market_analysis:
            return {}
        
        patterns = market_analysis['patterns']
        summary = {
            'best_pattern': None,
            'worst_pattern': None,
            'insights': []
        }
        
        # Find best and worst patterns by average return
        best_return = -999
        worst_return = 999
        
        for pattern, stats in patterns.items():
            if stats['total'] >= 5:  # Need at least 5 examples
                avg_return = stats['total_return'] / stats['total']
                if avg_return > best_return:
                    best_return = avg_return
                    summary['best_pattern'] = (pattern, avg_return, stats['total'])
                if avg_return < worst_return:
                    worst_return = avg_return
                    summary['worst_pattern'] = (pattern, avg_return, stats['total'])
        
        # Generate insights
        if summary['best_pattern']:
            pattern, ret, count = summary['best_pattern']
            summary['insights'].append(
                f"Market favored {pattern} patterns today (+{ret:.1f}% avg on {count} stocks)"
            )
        
        if summary['worst_pattern']:
            pattern, ret, count = summary['worst_pattern']
            summary['insights'].append(
                f"Avoid {pattern} patterns ({ret:.1f}% avg on {count} stocks)"
            )
        
        return summary