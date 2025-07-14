# trade_controller.py
# MM-B8: Core Trading Logic Controller
# UPDATED: MM-B11-B13 implementations
# UPDATED: MM-B14.1 Real trade slippage calibration

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from online_learning import predict_success_prob, train_or_update_model, append_trade_data
from confidence_calibrator import calibrate_confidence, log_confidence_result, log_confidence_drift
from skills_engine import get_level_from_xp, award_slippage_milestone_xp
from broker_sim import BrokerSim
from trade_memory import log_trade_result
from xp_tracker import get_xp_tracker, process_trade_xp, get_skill_biases
from vibe_analysis import analyze_vibe_effects, calculate_vibe_confidence_boost
from gui_handlers import log_system_message, format_currency, format_percentage

# Import new modules for MM-B12, MM-B13, and MM-B14.1
from risk_manager import RiskManager
from trade_narrator import TradeNarrator
from monkey_journal import MonkeyJournal
from real_feedback_engine import get_slippage_stats, log_real_trade
from gui_handlers import (
    log_system_message, format_currency, format_percentage,
    show_warning, show_info, show_error,
    get_manual_price  # ADD THIS
)
from market_pattern_analyzer import MarketPatternAnalyzer
from analysis.enhanced_features import EnhancedAnalyzer
from learning.enhanced_ml_features import EnhancedMLFeatures

class TradeController:
    """Centralized trading logic controller"""
    
    def __init__(self, broker, learning_model, monkee_callback=None):
        self.broker = broker
        self.learning_model = learning_model
        self.daily_picks_data = []
        self.daily_shadow_data = []
        self.xp_tracker = get_xp_tracker()
        self.monkee_callback = monkee_callback  # For sending patterns to Mon Kee
        
        # Initialize new components for MM-B12 and MM-B13
        self.risk_manager = RiskManager(initial_capital=10000)
        self.trade_narrator = TradeNarrator()
        self.monkey_journal = MonkeyJournal()
        
        #neural link
        self.pattern_memory = self.load_pattern_memory()
        self.pattern_confidence = {}
        
        # Initialize market analyzer
        self.market_analyzer = MarketPatternAnalyzer(self.identify_trade_pattern)
        self.enhanced_analyzer = EnhancedAnalyzer()    
        
    def load_pattern_memory(self):
        """Load historical pattern performance"""
        import json
        import os
    
        if os.path.exists('pattern_memory.json'):
            with open('pattern_memory.json', 'r') as f:
                return json.load(f)
        
         # Initialize empty memory
        return {
            'gap_and_go': {'total_trades': 0, 'wins': 0, 'total_pnl': 0},
            'reversal': {'total_trades': 0, 'wins': 0, 'total_pnl': 0},
            'volume_spike': {'total_trades': 0, 'wins': 0, 'total_pnl': 0},
            'momentum_surge': {'total_trades': 0, 'wins': 0, 'total_pnl': 0},
            'breakout': {'total_trades': 0, 'wins': 0, 'total_pnl': 0},
            'consolidation': {'total_trades': 0, 'wins': 0, 'total_pnl': 0},
            'unknown': {'total_trades': 0, 'wins': 0, 'total_pnl': 0}
        }

    def save_pattern_memory(self):
        """Save pattern memory to disk"""
        import json
        with open('pattern_memory.json', 'w') as f:
            json.dump(self.pattern_memory, f, indent=2)
    
    def identify_trade_pattern(self, features, ticker_data):
        """Identify what pattern this trade represents based on features"""
        gap = features.get('gap_pct', 0)
        volume = features.get('premarket_volume', 0)
        volume_change = features.get('volume_change_pct', 0)
        early_move = features.get('early_premarket_move', 0)
        
        # Pattern identification logic
        if gap < -2 and volume > 1000000:
            return 'reversal'  # Negative gap reversal setup
        elif gap > 5 and early_move > 10:
            return 'gap_and_go'  # Large gap continuation
        elif volume_change > 500:
            return 'volume_spike'  # Volume-driven move
        elif gap > 20:
            return 'momentum_surge'  # Extreme momentum
        elif abs(gap) < 2 and volume > 2000000:
            return 'breakout'  # Flat with high volume
        elif -2 < gap < 2:
            return 'consolidation'  # Range-bound
        else:
            return 'unknown'
    
    def update_pattern_knowledge(self, session_performance):
        """Update pattern memory with today's results"""
        for pattern, stats in session_performance.items():
            if pattern not in self.pattern_memory:
                self.pattern_memory[pattern] = {'total_trades': 0, 'wins': 0, 'total_pnl': 0}
            
            self.pattern_memory[pattern]['total_trades'] += stats['trades']
            self.pattern_memory[pattern]['wins'] += stats['wins']
            self.pattern_memory[pattern]['total_pnl'] += stats['total_pnl']
        
        # ADD THIS SECTION TO CALCULATE CONFIDENCE:
        # Calculate pattern confidence scores
        for pattern, memory in self.pattern_memory.items():
            if memory['total_trades'] >= 3:  # Need at least 3 trades
                win_rate = memory['wins'] / memory['total_trades']
                avg_pnl = memory['total_pnl'] / memory['total_trades']
                
                # Confidence based on win rate and profitability
                confidence = (win_rate * 0.7) + (min(avg_pnl / 100, 0.3))
                self.pattern_confidence[pattern] = confidence
                
                # Mon Kee speaks!
                if memory['total_trades'] == 3:
                    print(f"🧠 Mon Kee discovered: {pattern} pattern! Initial results...")
                elif memory['total_trades'] % 10 == 0:
                    if win_rate > 0.6:
                        print(f"🎯 Mon Kee mastered: {pattern}! {win_rate*100:.0f}% win rate!")
                    elif win_rate < 0.4:
                        print(f"🙈 Mon Kee avoiding: {pattern}. Only {win_rate*100:.0f}% wins...")
    
        # Save updated memory
        self.save_pattern_memory()


    def determine_best_predictor(self, ticker, actual_move, predictions):
        """
        MM-B11: Determine which feature was the best predictor for this trade
        
        Args:
            ticker: Stock ticker
            actual_move: Actual price movement percentage
            predictions: Dict of feature predictions
            
        Returns:
            Tuple of (best_predictor_name, accuracy_score)
        """
        best_predictor = None
        best_accuracy = 0
        
        # Check each predictor's accuracy
        predictor_scores = {
            'volume_pressure': predictions.get('volume_score', 0),
            'float_sense': predictions.get('float_rank', 0),
            'sentiment_spike': predictions.get('sentiment_score', 0),
            'gap_momentum': predictions.get('gap_percent', 0),
            'ml_confidence': predictions.get('ml_prediction', 0)
        }
        
        # Calculate accuracy for each predictor
        for predictor, score in predictor_scores.items():
            # Convert score to expected move
            if predictor == 'gap_momentum':
                expected_move = score  # Already a percentage
            else:
                expected_move = (score - 0.5) * 20  # Convert 0-1 to -10% to +10%
            
            # Calculate accuracy (inverse of error)
            error = abs(expected_move - actual_move)
            accuracy = max(0, 1 - (error / 20))  # Normalize to 0-1
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_predictor = predictor
        
        return best_predictor, best_accuracy

    def settle_and_learn(self, ticker, entry_price, exit_price, predictions):
        """MM-B11: Enhanced settle_and_learn with smart attribution"""
        # Calculate actual move
        actual_move = ((exit_price - entry_price) / entry_price) * 100
        
        # Determine best predictor
        best_predictor, accuracy = self.determine_best_predictor(
            ticker, actual_move, predictions
        )
        
        # Calculate XP award
        base_xp = 10 if actual_move > 0 else -5
        xp_awarded = int(base_xp * accuracy)
        
        # Award XP to the best predictor skill
        if hasattr(self, 'xp_tracker'):
            skill_map = {
                'volume_pressure': 'market_sense',
                'float_sense': 'float_mastery',
                'sentiment_spike': 'sentiment_radar',
                'gap_momentum': 'gap_hunter',
                'ml_confidence': 'pattern_recognition'
            }
            skill_name = skill_map.get(best_predictor, 'general_trading')
            self.xp_tracker.add_xp(skill_name, xp_awarded)
        
        # Log with attribution
        trade_result = {
            "ticker": ticker,
            "result": "win" if actual_move > 0 else "loss",
            "confidence": predictions.get('final_confidence', 0),
            "best_predictor": best_predictor,
            "predictor_accuracy": accuracy,
            "xp_awarded": xp_awarded,
            "actual_move": actual_move,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update trade memory with attribution - REMOVE THIS LINE
        # if hasattr(self, 'trade_memory'):
        #     self.trade_memory.log_trade_with_attribution(trade_result)
        
        return trade_result
    
    def log_real_execution(self, ticker, predicted_entry, actual_entry, 
                          predicted_exit, actual_exit, shares, notes=""):
        """
        MM-B14.1: Log a real trade execution for slippage learning
        
        Call this after you manually execute trades
        """
        trade_data = {
            'ticker': ticker,
            'predicted_entry': predicted_entry,
            'actual_entry': actual_entry,
            'predicted_exit': predicted_exit,
            'actual_exit': actual_exit,
            'shares': shares,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'notes': notes
        }
        
        # Log the trade
        result = log_real_trade(trade_data)
        
        # Check if we hit calibration milestone
        stats = get_slippage_stats()
        if stats['sample_count'] == 10:
            # Award milestone XP
            xp_log = award_slippage_milestone_xp()
            print(xp_log)
            
            # Update monkey journal
            self.monkey_journal.mood = "I'm learning how the real market works! 🎓"
        
        return result
    
    def run_daily_cycle(self, csv_path, log_widget=None, position_size=None, size_type='normal'):
        """Execute the complete daily trading cycle"""
        try:
            log_system_message(log_widget, f"Starting daily trading cycle ({size_type} mode)", "INFO")
            
            self.csv_path = csv_path
            
            # DEBUG: Check CSV columns
            import pandas as pd
            df = pd.read_csv(csv_path)
            log_system_message(log_widget, f"CSV columns: {list(df.columns)[:5]}...", "INFO")
            if 'Pre-market Close' in df.columns:
                non_null_count = df['Pre-market Close'].notna().sum()
                log_system_message(log_widget, f"Pre-market Close: {non_null_count} valid prices out of {len(df)} rows", "INFO")
            
            # ADD THIS: STANDARDIZE COLUMN NAMES
            if 'Symbol' in df.columns and 'ticker' not in df.columns:
                df['ticker'] = df['Symbol']
                log_system_message(log_widget, "✅ Created 'ticker' column from 'Symbol'", "INFO")
            
            # Save the standardized DataFrame for later use
            #df.to_csv(csv_path + '.temp', index=False)
            
            # Reset daily stats for risk manager
            self.risk_manager.reset_daily_stats()
            
            # Store Mon Kee's recommendations
            self.monkee_position_size = position_size
            self.monkee_size_type = size_type
            
            # Get slippage calibration status (MM-B14.1)
            slippage_stats = get_slippage_stats()
            if slippage_stats['sample_count'] > 0:
                log_system_message(
                    log_widget, 
                    f"📊 Slippage calibration: {slippage_stats['sample_count']}/10 real trades logged"
                )
            
            # Continue with the rest of the method...
            
            # Load and process CSV data
            df = self._load_and_clean_csv(csv_path)
            if df is None:
                return False, "Failed to load CSV data"
            
            # Score tickers with ML model
            scored_tickers = self._score_all_tickers(df, log_widget)
            
            # Select picks and execute trades
            picks_executed = self._execute_trades(scored_tickers, log_widget)
            
            if picks_executed == 0:
                return False, "No trades executed"
            
            log_system_message(log_widget, f"Daily cycle complete: {picks_executed} trades executed", "SUCCESS")
            return True, f"Successfully executed {picks_executed} trades"
            
        except Exception as e:
            error_msg = f"Error in daily cycle: {str(e)}"
            log_system_message(log_widget, error_msg, "ERROR")
            return False, error_msg
    
    def _load_and_clean_csv(self, csv_path):
        """Load and clean CSV data"""
        try:
            df = pd.read_csv(csv_path)
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Handle column variations and map to standard names
            column_mapping = {
                'symbol': 'ticker',
                'pre-market change %': 'early_premarket_move',
                'post-market change %': 'aftermarket_move',
                'pre-market gap %': 'gap_pct',
                'pre-market volume': 'premarket_volume',
                'gap % 1 day': 'gap_pct_alt',
                'volume 1 day': 'volume',
                'volume change % 1 day': 'volume_change_pct'
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Clean data
            df = df.dropna(subset=['ticker'])
            df['ticker'] = df['ticker'].str.upper().str.strip()
            
            return df
            
        except Exception as e:
            return None
    
    def _score_all_tickers(self, df, log_widget=None):
    """Score all tickers using ML model and XP biases"""
    skills_xp = self.xp_tracker.skills_xp
    biases = get_skill_biases()
    
    # Apply inspiration bonus if unlocked
    inspiration_level = get_level_from_xp(skills_xp.get('inspiration', 0))
    inspiration_bonus = 1.0
    if inspiration_level >= 10:
        inspiration_bonus = 1 + (inspiration_level / 200)
        if log_widget:
            log_system_message(
                log_widget, 
                f"Inspiration Level {inspiration_level} active! ({(inspiration_bonus-1)*100:.1f}% intuition boost)"
            )
    
    # Calculate adaptive threshold based on Machine IQ
    machine_iq = skills_xp.get("machine_iq", 0)
    score_threshold = 0.6 - min(machine_iq / 500, 0.1)
    
    if log_widget:
        log_system_message(log_widget, f"Using adaptive threshold: {score_threshold:.3f} (Machine IQ: {machine_iq})")
        log_system_message(log_widget, "Applying XP-based biases and vibe analysis")
    
    scored_tickers = []
    
    for _, row in df.iterrows():
        ticker = row.get('ticker', '').upper()
        if not ticker:
            continue
        
        # Extract and process features
        features = self._extract_features(row, biases)
        
         enhanced_features = self.enhanced_analyzer.enhance_ticker_analysis(
            features, 
            row.to_dict()
        )
        features = enhanced_features  # Use enhanced features
        
        # Get ML prediction
        success_prob = predict_success_prob(self.learning_model, features)
        
        # Identify potential pattern
        potential_pattern = self.identify_trade_pattern(features, row)

        # Apply pattern learning if available
        if potential_pattern in self.pattern_confidence:
            pattern_conf = self.pattern_confidence[potential_pattern]
            
            # Blend ML prediction with pattern knowledge
            if self.pattern_memory[potential_pattern]['total_trades'] >= 5:
                pattern_weight = min(0.5, self.pattern_memory[potential_pattern]['total_trades'] / 20)
                ml_weight = 1 - pattern_weight
                success_prob = (success_prob * ml_weight) + (pattern_conf * pattern_weight)
        
        # Apply vibe-based confidence modifiers
        success_prob = self._apply_vibe_modifiers(success_prob, features)
        
        # Apply inspiration bonus to borderline picks
        if inspiration_level >= 10 and 0.45 <= success_prob <= 0.6:
            success_prob *= inspiration_bonus
        
        # Ensure probability stays in valid range
        success_prob = min(success_prob, 0.95)
        
        # Calibrate the raw confidence
        calibrated_prob = calibrate_confidence(success_prob)
        
        # ENHANCED PRICE EXTRACTION WITH FALLBACKS
        # Try multiple price sources in order of preference
        premarket_close = None
        
        # 1. Try Pre-market Close (original column name with exact case)
        if 'Pre-market Close' in row.index:
            val = row['Pre-market Close']
            if pd.notna(val) and val > 0:
                premarket_close = float(val)
        
        # 2. Try lowercase version
        if premarket_close is None and 'pre-market close' in row.index:
            val = row['pre-market close']
            if pd.notna(val) and val > 0:
                premarket_close = float(val)
        
        # 3. Fallback to regular Price column
        if premarket_close is None and 'Price' in row.index:
            val = row['Price']
            if pd.notna(val) and val > 0:
                premarket_close = float(val)
        
        # 4. Fallback to lowercase price
        if premarket_close is None and 'price' in row.index:
            val = row['price']
            if pd.notna(val) and val > 0:
                premarket_close = float(val)
        
        # 5. If still no price, calculate estimate based on gap
        if premarket_close is None or premarket_close <= 0:
            # Use a base price and adjust by gap percentage
            gap = features.get('gap_pct', 0)
            if gap != 0:
                # Estimate price based on typical small-cap price and gap
                premarket_close = 5.0 * (1 + gap / 100)
            else:
                premarket_close = None  # Will be handled later
        
        scored_tickers.append({
            'ticker': ticker,
            'score': calibrated_prob,
            'raw_score': success_prob,
            'features': features,
            'premarket_close': premarket_close,
            'pattern': potential_pattern
        })
    
    # Sort by score
    scored_tickers.sort(key=lambda x: x['score'], reverse=True)
    
    # After scoring, send top pattern to Mon Kee if available
    if scored_tickers and hasattr(self, 'monkee_callback') and self.monkee_callback:
        top_pattern = {
            'pattern_name': self._identify_pattern_type(scored_tickers[0]),
            'symbol': scored_tickers[0]['ticker'],
            'confidence': scored_tickers[0]['score'],
            'features': scored_tickers[0]['features']
        }
        self.monkee_callback(top_pattern)
    
    return scored_tickers
    
    def _extract_features(self, row, biases):
        """Extract and bias features from CSV row"""
        # Extract traditional features
        gap_pct = float(row.get('gap_pct', row.get('pre-market gap %', row.get('gap', 0))))
        volume = float(row.get('premarket_volume', row.get('pre-market volume', row.get('volume', 100000))))
        float_val = float(row.get('float', row.get('shares_float', 10000000)))
        
        # Extract vibe features
        aftermarket_move = float(row.get('aftermarket_move', row.get('post-market change %', 0)))
        early_premarket_move = float(row.get('early_premarket_move', row.get('pre-market change %', 0)))
        
        # FIX PERCENTAGE VALUES HERE
        # If the values are already percentages (like 3789.5 instead of 37.895)
        if abs(early_premarket_move) > 100:
            early_premarket_move = early_premarket_move / 100
        if abs(aftermarket_move) > 100:
            aftermarket_move = aftermarket_move / 100
        if abs(gap_pct) > 100:
            gap_pct = gap_pct / 100
        
        # Sanity caps - nothing should be over 100% realistically
        early_premarket_move = max(-99, min(99, early_premarket_move))
        aftermarket_move = max(-99, min(99, aftermarket_move))
        gap_pct = max(-99, min(99, gap_pct))
        
        # Extract additional features
        volume_change_pct = float(row.get('volume_change_pct', row.get('volume change % 1 day', 0)))
        price_change_1d = float(row.get('price change % 1 day', 0))
        
        # Fix these too if needed
        if abs(volume_change_pct) > 1000:
            volume_change_pct = volume_change_pct / 100
        if abs(price_change_1d) > 100:
            price_change_1d = price_change_1d / 100
        
        # Calculate advanced features
        momentum_aligned = np.sign(early_premarket_move) == np.sign(price_change_1d) if price_change_1d != 0 else False
        volume_surge = volume_change_pct > 500  # 5x normal volume
        gap_magnitude = abs(gap_pct)
        
        # Apply XP-based biases
        biased_gap_pct = gap_pct * biases['gap_bias']
        biased_volume = volume * biases['volume_bias']
        biased_float = float_val * biases['float_bias']
        biased_aftermarket = aftermarket_move * biases['aftermarket_bias']
        biased_premarket = early_premarket_move * biases['premarket_bias']
        
        # Create feature row for ML prediction
        return {
            'gap_pct': biased_gap_pct,
            'premarket_volume': biased_volume,
            'float': biased_float,
            'sentiment_score': 0,  # Placeholder
            'aftermarket_move': biased_aftermarket,
            'early_premarket_move': biased_premarket,
            'momentum_aligned': 1.0 if momentum_aligned else 0.0,
            'volume_surge': 1.0 if volume_surge else 0.0,
            'gap_magnitude': gap_magnitude,
            'volume_change_pct': volume_change_pct
        }
    
    def _apply_vibe_modifiers(self, success_prob, features):
        """Apply vibe-based confidence modifiers"""
        aftermarket = features.get('aftermarket_move', 0)
        premarket = features.get('early_premarket_move', 0)
        
        # MM-B11: Apply freshness adjustment
        from vibe_analysis import VibeAnalyzer
        vibe_analyzer = VibeAnalyzer()
        
        # Apply freshness penalty if we have the method
        if hasattr(vibe_analyzer, 'apply_freshness_adjustment'):
            success_prob = vibe_analyzer.apply_freshness_adjustment(success_prob, 'scanner')
        
        # Strong aftermarket signal
        if abs(aftermarket) > 5:
            if aftermarket > 0:
                success_prob *= 1.15  # 15% boost for positive aftermarket
            else:
                success_prob *= 0.95  # 5% penalty for negative aftermarket
        
        # Strong premarket signal  
        if abs(premarket) > 3:
            if premarket > 0:
                success_prob *= 1.10  # 10% boost for positive premarket
            else:
                success_prob *= 0.97  # 3% penalty for negative premarket
        
        # Vibe alignment bonus
        vibe_confidence = calculate_vibe_confidence_boost(aftermarket, premarket)
        if vibe_confidence > 0.7:
            success_prob *= 1.05  # Small boost for aligned vibes
        elif vibe_confidence < 0.4:
            success_prob *= 0.95  # Small penalty for conflicted vibes
        
        return success_prob
    
    def _execute_trades(self, scored_tickers, log_widget=None):
        """Execute trades based on scored tickers with risk management"""
        # Determine picks using adaptive threshold
        skills_xp = self.xp_tracker.skills_xp
        machine_iq = skills_xp.get("machine_iq", 0)
        score_threshold = 0.6 - min(machine_iq / 500, 0.1)
        
        # Get main picks (top 5 above threshold)
        top_picks = [t for t in scored_tickers if t['score'] >= score_threshold][:5]
        
        if not top_picks:
            if log_widget:
                log_system_message(log_widget, f"No tickers met {score_threshold:.3f} threshold. Taking top 5 anyway...")
            top_picks = scored_tickers[:5]

        # Check which picks still need manual prices
        picks_needing_prices = []
        for pick in top_picks:
            ticker = pick['ticker']
            current_price = pick.get('premarket_close', None)
            
            if current_price is None or pd.isna(current_price) or current_price <= 0:
                picks_needing_prices.append(pick)

        # If any picks need prices, ask user once
        if picks_needing_prices and log_widget:
            log_system_message(log_widget, f"⚠️ Need prices for {len(picks_needing_prices)} stocks Mon Kee wants to trade", "WARNING")
            
            for pick in picks_needing_prices:
                ticker = pick['ticker']
                # Provide a smarter default based on typical penny stock prices
                default_price = 3.0 if pick['features'].get('gap_pct', 0) < -20 else 5.0
                manual_price = get_manual_price(ticker, default_price)
                pick['premarket_close'] = manual_price
                log_system_message(log_widget, f"✅ Set {ticker} price to ${manual_price:.2f}", "INFO")
        
        # Get shadow picks for learning
        shadow_candidates = [t for t in scored_tickers if t not in top_picks]
        learning_zone_picks = [t for t in shadow_candidates if 0.4 <= t['score'] < score_threshold]
        other_picks = [t for t in shadow_candidates if t not in learning_zone_picks]
        shadow_picks = (learning_zone_picks[:5] + other_picks[:5])[:5]
        
        # Log shadow picks
        if shadow_picks and log_widget:
            log_system_message(log_widget, f"Tracking {len(shadow_picks)} shadow picks for learning")
        
        # Execute main trades
        cash_available = self.broker.get_cash()
        total_trade_bank = cash_available * 0.20  # 20% allocation
        
        successful_trades = 0
        raw_confidences = []
        calibrated_confidences = []
        
        for pick in top_picks:
            ticker = pick['ticker']
            score = pick['score']
            raw_confidences.append(pick['raw_score'])
            calibrated_confidences.append(score)
            
            # Calculate position size
            weight = self._calculate_position_weight(pick, top_picks)
            allocation = weight * total_trade_bank
            
            # Get the price (should now always be valid)
            current_price = pick.get('premarket_close', 10.0)
            
            # Final safety check
            if pd.isna(current_price) or current_price <= 0:
                if log_widget:
                    log_system_message(log_widget, f"⚠️ Skipping {ticker}: Still invalid price after fixes", "ERROR")
                continue

            shares_to_buy = int(allocation / current_price)
            
            # Check risk limits before executing
            can_trade, reason = self.risk_manager.check_position_limits(ticker, shares_to_buy)
            
            if not can_trade:
                if log_widget:
                    log_system_message(log_widget, f"Trade rejected for {ticker}: {reason}", "WARNING")
                continue
            
            if shares_to_buy > 0:
                try:
                    # Use slippage-aware buy method
                    self.broker.buy_with_learned_slippage(ticker, current_price, shares_to_buy)
                    
                    # Add position to risk manager
                    position = self.risk_manager.add_position(
                        ticker, current_price, shares_to_buy, score
                    )
                    
                    # Generate and log trade justification
                    justification = self.trade_narrator.generate_trade_justification(
                        ticker, "BUY", current_price, pick['features']
                    )
                    
                    # Log trade details
                    if log_widget:
                        log_system_message(log_widget, justification)
                        self._log_trade_execution(log_widget, ticker, shares_to_buy, current_price, score, pick)
                    
                    # Identify the pattern
                    pattern = pick.get('pattern', self.identify_trade_pattern(pick['features'], pick))
                    
                    # Store pick data for learning
                    self.daily_picks_data.append({
                        'ticker': ticker,
                        'features': pick['features'],
                        'buy_price': current_price,
                        'shares': shares_to_buy,
                        'predicted_prob': score,
                        'raw_prob': pick['raw_score'],
                        'pattern': pattern,
                        'vibe_analysis': analyze_vibe_effects(pick['features']),
                        'best_predictor': None  # Will be determined at settlement
                    })
                    
                    # Log the pattern
                    if log_widget:
                        log_system_message(log_widget, f"Pattern identified: {pattern}", "INFO")
                   
                    successful_trades += 1
                    
                except Exception as e:
                    if log_widget:
                        log_system_message(log_widget, f"Failed to buy {ticker}: {str(e)}", "ERROR")
        
        # Store shadow picks for learning
        for pick in shadow_picks:
            ticker = pick['ticker']
            # For shadows, we can use estimated prices since we're not actually trading
            shadow_price = pick.get('premarket_close', 10.0)
            if pd.isna(shadow_price) or shadow_price <= 0:
                gap_pct = pick['features'].get('gap_pct', 0)
                shadow_price = 10.0 * (1 + gap_pct / 100)
            
            self.daily_shadow_data.append({
                'ticker': ticker,
                'features': pick['features'],
                'buy_price': shadow_price,
                'predicted_prob': pick['score'],
                'raw_prob': pick['raw_score']
            })
        
        # Log confidence drift
        if raw_confidences and calibrated_confidences:
            log_confidence_drift(raw_confidences, calibrated_confidences)
        
        if log_widget:
            log_system_message(
                log_widget, 
                f"Trade execution complete: {successful_trades} trades, {format_currency(self.broker.get_cash())} remaining"
            )
        
        return successful_trades
    
    def _calculate_position_weight(self, pick, all_picks):
        """Calculate position weight based on confidence"""
        if not all_picks:
            return 1.0
        
        min_score = min(p['score'] for p in all_picks)
        max_score = max(p['score'] for p in all_picks)
        score_range = max_score - min_score if max_score != min_score else 1
        
        normalized = (pick['score'] - min_score) / score_range
        weight = 1 + (normalized - 0.5) * 0.2  # ranges from 0.9 to 1.1
        
        # Normalize across all picks
        total_weight = sum(1 + ((p['score'] - min_score) / score_range - 0.5) * 0.2 for p in all_picks)
        return weight / total_weight
    
    def _log_trade_execution(self, log_widget, ticker, shares, price, score, pick):
        """Log trade execution details"""
        features = pick['features']
        aftermarket = features.get('aftermarket_move', 0)
        premarket = features.get('early_premarket_move', 0)
        
        # Main trade log
        log_system_message(
            log_widget, 
            f"BOUGHT {ticker}: {shares} shares @ {format_currency(price)}"
        )
        
        # Score details
        log_system_message(
            log_widget, 
            f"ML Score: Raw {pick['raw_score']:.3f} → Calibrated {score:.3f}"
        )
        
        # Vibe details
        if aftermarket != 0 or premarket != 0:
            log_system_message(
                log_widget, 
                f"Vibes: Aftermarket {format_percentage(aftermarket)}, Premarket {format_percentage(premarket)}"
            )
    
    def check_intraday_stops(self, current_prices, log_widget=None):
        """MM-B12: Check and execute stop losses/take profits"""
        for ticker, current_price in current_prices.items():
            exit_result = self.risk_manager.check_stops_and_targets(ticker, current_price)
            
            if exit_result:
                # Find original pick data
                pick_data = next((p for p in self.daily_picks_data if p['ticker'] == ticker), None)
                if pick_data:
                    # MM-B13: Log trade error if stop hit
                    if exit_result['exit_reason'] == 'stop_loss':
                        expected_move = (exit_result['target_price'] - exit_result['entry_price']) / exit_result['entry_price'] * 100
                        actual_move = (exit_result['exit_price'] - exit_result['entry_price']) / exit_result['entry_price'] * 100
                        
                        self.trade_narrator.log_trade_error(
                            ticker, expected_move, actual_move, pick_data['features']
                        )
    
    def settle_trades_with_overview(self, postmarket_csv_path, log_widget=None):
        """Settle trades and learn from results"""
        try:
            if not self.broker.get_open_trades():
                return False, "No open trades to settle"
            
            log_system_message(log_widget, "Starting trade settlement and learning", "INFO")
            
            # MM-B14.1: Use slippage-aware settlement
            self.broker.settle_day_with_learned_slippage(postmarket_csv_path)
            
            # Get slippage stats for journal
            slippage_stats = get_slippage_stats()
            
            # Load postmarket data for analysis
            post_df = pd.read_csv(postmarket_csv_path)
            post_df.columns = [col.strip().lower() for col in post_df.columns]
            if 'symbol' in post_df.columns:
                post_df = post_df.rename(columns={'symbol': 'ticker'})
            
            # Process trades for learning
            training_data = []
            total_pnl = 0
            wins = 0
            daily_trades = []
            pattern_performance = {} 
            
            recent_trades = self.broker.closed_trades[-len(self.daily_picks_data):]
            
            for i, pick_data in enumerate(self.daily_picks_data):
                if i < len(recent_trades):
                    trade = recent_trades[i]
                    pattern = pick_data.get('pattern', 'unknown')
                    
                    # MM-B11: Determine best predictor
                    actual_move = ((trade['close_price'] - trade['buy_price']) / trade['buy_price']) * 100
                    predictions = {
                        'volume_score': pick_data['features'].get('premarket_volume', 0) / 1e6,
                        'float_rank': 10 / max(pick_data['features'].get('float', 1e6) / 1e6, 1),
                        'sentiment_score': pick_data['features'].get('sentiment_score', 0.5),
                        'gap_percent': pick_data['features'].get('gap_pct', 0),
                        'ml_prediction': pick_data['predicted_prob'],
                        'final_confidence': pick_data['predicted_prob']
                    }
                    
                    best_predictor, accuracy = self.determine_best_predictor(
                        trade['ticker'], actual_move, predictions
                    )
                    
                    # Create training record
                    features = pick_data['features'].copy()
                    result_str = "WIN" if trade['pnl'] > 0 else "LOSS"
                    features['trade_outcome'] = 1 if trade['pnl'] > 0 else 0
                    
                    training_data.append(features)
                    total_pnl += trade['pnl']
                    if trade['pnl'] > 0:
                        wins += 1
                    
                    # Process XP and skill updates
                    confidence = pick_data.get('predicted_prob', 0.5)
                    pnl = trade['pnl']
                    
                    # Award XP using centralized system
                    xp_logs = process_trade_xp(features, result_str, confidence, pnl, log_widget)
                    
                    # Log confidence result for calibration
                    log_confidence_result(confidence, result_str)
                    
                    # Enhanced trade memory data with attribution
                    trade_memory_data = {
                        "ticker": trade['ticker'],
                        "entry_price": trade['buy_price'],
                        "exit_price": trade['close_price'],
                        "shares": trade['shares'],
                        "pnl": trade['pnl'],
                        "result": result_str,
                        "predicted_prob": pick_data.get('predicted_prob', 0.5),
                        "raw_prob": pick_data.get('raw_prob', 0.5),
                        "gap_pct": features.get("gap_pct", 0),
                        "premarket_volume": features.get("premarket_volume", 0),
                        "float": features.get("float", 0),
                        "sentiment_score": features.get("sentiment_score", 0),
                        "aftermarket_move": features.get("aftermarket_move", 0),
                        "early_premarket_move": features.get("early_premarket_move", 0),
                        "pattern": pattern,
                        "best_predictor": best_predictor,
                        "predictor_accuracy": accuracy,
                        "xp_awarded": 10 * accuracy if actual_move > 0 else -5 * accuracy
                    }
                    log_trade_result(trade_memory_data)
                    
                    # MM-B13: Log trade error analysis
                    if result_str == "LOSS":
                        expected_move = (pick_data['predicted_prob'] - 0.5) * 20
                        self.trade_narrator.log_trade_error(
                            trade['ticker'], expected_move, actual_move, pick_data['features']
                        )
                    
                    daily_trades.append(trade_memory_data)
                    
                    # Track pattern performance
                    if pattern not in pattern_performance:
                        pattern_performance[pattern] = {
                            'trades': 0,
                            'wins': 0,
                            'losses': 0,
                            'total_pnl': 0,
                            'examples': []
                        }
                    
                    pattern_performance[pattern]['trades'] += 1
                    if trade['pnl'] > 0:
                        pattern_performance[pattern]['wins'] += 1
                    else:
                        pattern_performance[pattern]['losses'] += 1
                    pattern_performance[pattern]['total_pnl'] += trade['pnl']
                    pattern_performance[pattern]['examples'].append({
                        'ticker': trade['ticker'],
                        'pnl': trade['pnl'],
                        'gap': pick_data['features'].get('gap_pct', 0)
                    })
                    
                    if log_widget:
                        log_system_message(
                            log_widget, 
                            f"{trade['ticker']}: {result_str} ({format_currency(pnl)}) - Pattern: {pattern}, Best predictor: {best_predictor}"
                        )
        
            # Update ML model with new data
            if training_data:
                append_trade_data(training_data)
                self.learning_model = train_or_update_model(self.learning_model, pd.DataFrame(training_data))
                if log_widget:
                    log_system_message(log_widget, f"ML Model updated with {len(training_data)} trades")
            
            # Process shadow picks for learning
            shadow_results = self._process_shadow_picks(post_df, log_widget)
            
            # Check for inspiration unlock
            inspiration_unlocked, inspiration_reason = self.xp_tracker.check_inspiration_unlock(shadow_results)
            if inspiration_unlocked:
                if log_widget:
                    log_system_message(log_widget, f"✨ INSPIRATION UNLOCKED! {inspiration_reason}", "SUCCESS")
            
            # MM-B11: Write Mon Kee's journal entry with slippage awareness
            predictor_stats = {}
            for trade in daily_trades:
                pred = trade.get('best_predictor', 'unknown')
                if pred not in predictor_stats:
                    predictor_stats[pred] = 0
                predictor_stats[pred] += 1
            
            # MM-B14.1: Use slippage-aware journal writing
            journal_summary = self.monkey_journal.write_daily_summary_with_slippage(
                daily_trades, 
                predictor_stats,
                slippage_stats
            )
            
            # Update pattern knowledge
            self.update_pattern_knowledge(pattern_performance)
            
            # Log pattern performance
            if log_widget and pattern_performance:
                log_system_message(log_widget, "\n📊 PATTERN PERFORMANCE THIS SESSION:", "INFO")
                for pattern, stats in pattern_performance.items():
                    if stats['trades'] > 0:
                        win_rate = (stats['wins'] / stats['trades']) * 100
                        avg_pnl = stats['total_pnl'] / stats['trades']
                        log_system_message(
                            log_widget, 
                            f"{pattern}: {win_rate:.0f}% WR, ${avg_pnl:.2f} avg P&L ({stats['trades']} trades)",
                            "INFO"
                        )
            
            if log_widget:
                log_system_message(log_widget, "\n🔍 Analyzing market-wide patterns...", "INFO")

            market_analysis = self.analyze_market_patterns(
                self.csv_path,
                postmarket_csv_path,
                log_widget
            )
            
            # Generate summary
            total_trades = len(recent_trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            summary = {
                'total_trades': len(recent_trades),  # ← CORRECT: using :
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'wins': wins,
                'final_balance': self.broker.get_cash(),
                'risk_metrics': self.risk_manager.get_risk_metrics(),
                'slippage_calibrated': slippage_stats['sample_count'] >= 10
            }
            
            # Clear daily data for next cycle
            self.daily_picks_data = []
            self.daily_shadow_data = []
            
            if log_widget:
                log_system_message(
                    log_widget, 
                    f"Settlement complete: {win_rate:.1f}% win rate, {format_currency(total_pnl)} P&L",
                    "SUCCESS"
                )
                log_system_message(
                    log_widget,
                    f"Mon Kee's mood: {journal_summary['mood']}",
                    "INFO"
                )
                self.risk_manager.open_positions = {}
                self.risk_manager.daily_pnl = 0
                
            return True, summary
            
        except Exception as e:
            error_msg = f"Error settling trades: {str(e)}"
            if log_widget:
                log_system_message(log_widget, error_msg, "ERROR")
            return False, error_msg
    
    def _process_shadow_picks(self, post_df, log_widget=None):
        """Process shadow picks for learning"""
        if not self.daily_shadow_data:
            return None
        
        shadow_training_data = []
        shadow_wins = 0
        shadow_pnl = 0
        big_winner_missed = False
        big_winner_reason = ""
        
        for shadow in self.daily_shadow_data:
            ticker = shadow['ticker']
            buy_price = shadow['buy_price']
            
            # Find closing price
            row = post_df[post_df['ticker'].str.upper() == ticker.upper()]
            if not row.empty:
                close_price = float(row.iloc[0].get('price', buy_price))
                shadow_profit_pct = (close_price - buy_price) / buy_price * 100
                shadow_profit_dollars = (close_price - buy_price) * 100  # Assume 100 shares
                
                result = "WIN" if close_price > buy_price else "LOSS"
                if result == "WIN":
                    shadow_wins += 1
                
                shadow_pnl += shadow_profit_dollars
                
                # Check for big winner
                if shadow_profit_pct > 15:
                    big_winner_missed = True
                    big_winner_reason = f"missed a {shadow_profit_pct:.1f}% winner with {ticker}!"
                
                # Add to training data
                features = shadow['features'].copy()
                features['trade_outcome'] = 1 if result == "WIN" else 0
                shadow_training_data.append(features)
                
                # Track confidence accuracy for shadows
                log_confidence_result(shadow['predicted_prob'], result)
                
                if log_widget:
                    log_system_message(
                        log_widget, 
                        f"Shadow {ticker}: {result} ({format_percentage(shadow_profit_pct)})"
                    )
        
        # Update model with shadow results
        if shadow_training_data:
            append_trade_data(shadow_training_data)
            self.learning_model = train_or_update_model(self.learning_model, pd.DataFrame(shadow_training_data))
            
            if log_widget:
                shadow_win_rate = shadow_wins / len(self.daily_shadow_data) if self.daily_shadow_data else 0
                log_system_message(
                    log_widget, 
                    f"Shadow Learning: {shadow_win_rate*100:.1f}% win rate, {format_currency(shadow_pnl)} hypothetical P&L"
                )
        
        return {
            'win_rate': shadow_wins / len(self.daily_shadow_data) if self.daily_shadow_data else 0,
            'total_pnl': shadow_pnl,
            'big_winner_missed': big_winner_missed,
            'big_winner_reason': big_winner_reason,
            'total_picks': len(self.daily_shadow_data)
        }
    
    def save_trade_journal(self):
        """Save trade history to journal"""
        journal_path = "trade_journal.json"
        
        # Load existing journal
        if os.path.exists(journal_path):
            with open(journal_path, "r") as f:
                journal = json.load(f)
        else:
            journal = []
        
        # Add today's entry
        if self.broker.closed_trades:            
            recent_trades = self.broker.closed_trades[-len(self.daily_picks_data):] if self.daily_picks_data else self.broker.closed_trades[-5:]
            entry = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'cash_balance': self.broker.get_cash(),
                'trades': recent_trades,
                'total_picks': len(self.daily_picks_data),
                'shadow_picks': len(self.daily_shadow_data),
                'market_analysis': getattr(self, 'last_market_analysis', {})
            }
            journal.append(entry)
            
            # Save updated journal
            with open(journal_path, "w") as f:
                json.dump(journal, f, indent=2)
    
    def get_daily_summary(self):
        """Get summary of today's trading activity"""
        open_trades = len(self.broker.get_open_trades())
        main_picks = len(self.daily_picks_data)
        shadow_picks = len(self.daily_shadow_data)
        cash_balance = self.broker.get_cash()
        
        return {
            'open_trades': open_trades,
            'main_picks': main_picks,
            'shadow_picks': shadow_picks,
            'cash_balance': cash_balance,
            'has_activity': main_picks > 0 or shadow_picks > 0
        }
    
    def reset_daily_data(self):
        """Reset daily trading data"""
        self.daily_picks_data = []
        self.daily_shadow_data = []
    
    def get_recent_performance(self, days=10):
        """Get recent trading performance"""
        if not self.broker.closed_trades:
            return None
        
        recent_trades = self.broker.closed_trades[-days:] if len(self.broker.closed_trades) >= days else self.broker.closed_trades
        
        total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
        wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        win_rate = (wins / len(recent_trades) * 100) if recent_trades else 0
        
        return {
            'total_trades': len(recent_trades),
            'wins': wins,
            'losses': len(recent_trades) - wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / len(recent_trades) if recent_trades else 0
        }
    def get_recent_performance(self, days=10):
        """Get recent trading performance"""
        if not self.broker.closed_trades:
            return None
        
        recent_trades = self.broker.closed_trades[-days:] if len(self.broker.closed_trades) >= days else self.broker.closed_trades
        
        total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
        wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        win_rate = (wins / len(recent_trades) * 100) if recent_trades else 0
        
        return {
            'total_trades': len(recent_trades),
            'wins': wins,
            'losses': len(recent_trades) - wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / len(recent_trades) if recent_trades else 0
        }
    
    def get_current_pattern_data(self):
        """Get current pattern data for Mon Kee's analysis"""
        if self.daily_picks_data:
            # Get the most recent/best pattern
            best_pick = max(self.daily_picks_data, key=lambda x: x.get('predicted_prob', 0))
            return {
                'pattern_name': best_pick.get('best_predictor', 'mixed_signals'),
                'symbol': best_pick['ticker'],
                'entry_price': best_pick['buy_price'],
                'confidence': best_pick['predicted_prob'],
                'features': best_pick['features']
            }
        return None
    def analyze_market_patterns(self, premarket_csv_path, postmarket_csv_path, log_widget=None):
        """Analyze full market patterns to accelerate learning"""
        try:
            # Load CSVs
            pre_df = pd.read_csv(premarket_csv_path)
            post_df = pd.read_csv(postmarket_csv_path)
            
            # Get list of tickers we traded
            traded_tickers = [pick['ticker'] for pick in self.daily_picks_data]
            
            # Run market analysis
            market_analysis = self.market_analyzer.analyze_full_market(
                pre_df, post_df, traded_tickers, log_widget
            )
            
            # Get learning summary
            learning_summary = self.market_analyzer.generate_learning_summary(market_analysis)
            
            # Update pattern knowledge with market insights
            if 'patterns' in market_analysis:
                self._update_from_market_patterns(market_analysis['patterns'])
            
            # Update journal with missed opportunities
            if hasattr(self, 'monkey_journal') and market_analysis.get('missed_opportunities'):
                top_miss = market_analysis['missed_opportunities'][0] if market_analysis['missed_opportunities'] else None
                if top_miss:
                    self.monkey_journal.regrets = f"Missed {top_miss['ticker']} (+{top_miss['return']:.1f}%)"
            
             # Store the analysis for export
            self.last_market_analysis = market_analysis
            
            return market_analysis
            
        except Exception as e:
            if log_widget:
                log_system_message(log_widget, f"Error analyzing market patterns: {str(e)}", "ERROR")
            return {}

    def _update_from_market_patterns(self, market_patterns):
        """Update pattern confidence based on market-wide performance"""
        # Weight market observations less than actual trades
        MARKET_WEIGHT = 0.2

        for pattern, stats in market_patterns.items():
            if stats['total'] >= 5:  # Need meaningful sample size
                # Calculate market performance
                market_win_rate = stats['winners'] / stats['total']
                market_avg_return = stats['total_return'] / stats['total']
                
                # Convert to confidence score (0-1)
                # Positive returns boost confidence, negative reduce it
                market_confidence = 0.5 + (market_avg_return / 100)  # +10% return = 0.6 confidence
                market_confidence = max(0.1, min(0.9, market_confidence))  # Clamp to reasonable range
                
                # Update or create pattern confidence
                if pattern not in self.pattern_confidence:
                    self.pattern_confidence[pattern] = market_confidence
                else:
                    # Blend with existing confidence
                    current = self.pattern_confidence[pattern]
                    self.pattern_confidence[pattern] = (current * (1 - MARKET_WEIGHT)) + (market_confidence * MARKET_WEIGHT)

        # Save updated confidence
        self.save_pattern_memory()
        
    def _identify_pattern_type(self, ticker_data):
        """Identify the dominant pattern type for Mon Kee"""
        features = ticker_data['features']
        
        # Simple pattern identification based on strongest signal
        patterns = {
            'gap_momentum': abs(features.get('gap_pct', 0)),
            'volume_surge': features.get('volume_change_pct', 0) / 100,
            'float_play': 10000000 / max(features.get('float', 10000000), 1),
            'sentiment_spike': features.get('sentiment_score', 0.5)
        }
        
        return max(patterns.items(), key=lambda x: x[1])[0]


    # Factory function to create trade controller
def create_trade_controller(broker, learning_model, monkee_callback=None):
    """Create a new trade controller instance"""
    return TradeController(broker, learning_model, monkee_callback)