"""
Multi-Timeframe Validator Module

This module implements a Multi-Timeframe Validator that works as an additional filter/overlay
before trade execution. It validates trade signals across multiple timeframes (1m, 5m, 15m)
using the EXACT SAME indicators and ML signals from the 1-minute engine.

The validator:
1. Takes 1m trade signals from the main algorithm
2. Computes the same indicators (RSI, CCI, ADX, WT, momentum) on 5m and 15m timeframes
3. Uses the same Lorentzian distance classifier for trend analysis
4. Returns trade modifiers (lot size, TP adjustments, or block signals)
5. Fails safe to the original algorithm if data is missing or validation fails
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter

# Import the exact same indicator functions from the main algorithm
from core.signals import (
    series_from, calculate_rsi_pine, calculate_williams_r_pine, 
    calculate_cci_pine, calculate_adx_pine, get_lorentzian_distance,
    FeatureSeries, FeatureArrays, Settings, Label
)

logger = logging.getLogger(__name__)

@dataclass
class MTFValidationResult:
    """Result of multi-timeframe validation"""
    allow_trade: bool
    lot_multiplier: float
    tp_multiplier: float
    confidence_boost: float
    reasoning: str
    timeframe_alignment: Dict[str, str]  # '1m': 'bullish', '5m': 'bullish', '15m': 'neutral'
    validation_score: float  # 0.0 to 1.0
    scenario_label: str  # Descriptive label like 'pullback_countertrend', 'aligned_strong', etc.

class MultiTimeframeValidator:
    """
    Multi-Timeframe Validator that validates trade signals across 1m, 5m, and 15m timeframes.
    
    This validator uses the EXACT SAME indicators and ML signals from the 1-minute engine:
    - RSI(14,1), WT(10,11), CCI(20,1), ADX(20,2), RSI(9,1)
    - Lorentzian distance classifier
    - Same feature parameters and calculations
    
    It works as an overlay on top of the existing Lorentzian ML algorithm,
    providing additional intelligence for trade execution without modifying core logic.
    """
    
    def __init__(self, broker_adapter=None):
        """
        Initialize the Multi-Timeframe Validator
        
        Args:
            broker_adapter: Broker adapter for fetching historical data
        """
        self.broker_adapter = broker_adapter
        self.logger = logging.getLogger(__name__)
        
        # Use the EXACT SAME feature parameters as the main algorithm
        self.feature_params = {
            'f1': {'string': 'RSI', 'param_a': 14, 'param_b': 1},    # RSI(14,1)
            'f2': {'string': 'WT', 'param_a': 10, 'param_b': 11},    # WT(10,11) - Williams %R
            'f3': {'string': 'CCI', 'param_a': 20, 'param_b': 1},    # CCI(20,1)
            'f4': {'string': 'ADX', 'param_a': 20, 'param_b': 2},    # ADX(20,2)
            'f5': {'string': 'RSI', 'param_a': 9, 'param_b': 1}      # RSI(9,1)
        }
        
        # Use the EXACT SAME settings as the main algorithm
        self.settings = Settings(
            source='close',
            neighbors_count=8,
            max_bars_back=2000,
            feature_count=5,
            color_compression=1,
            show_exits=False,
            use_dynamic_exits=True
        )
        
        # Trade modifiers
        self.full_conviction_multiplier = 1.2
        self.partial_conviction_multiplier = 0.8
        self.reduced_conviction_multiplier = 0.6
        self.block_threshold = 0.3
        
    def validate_trade_signal(self, symbol: str, signal: int, current_1m_data: Dict, 
                            historical_data: List[Dict]) -> MTFValidationResult:
        """
        Validate a trade signal across multiple timeframes
        
        Args:
            symbol: Trading symbol
            signal: Trade signal from main algorithm (1 for long, -1 for short)
            current_1m_data: Current 1-minute OHLC data
            historical_data: Historical data for the symbol
            
        Returns:
            MTFValidationResult with validation outcome and trade modifiers
        """
        try:
            self.logger.info(f"[MTF_VALIDATOR] Validating {symbol} signal: {signal}")
            
            # Get multi-timeframe data
            mtf_data = self._get_mtf_data(symbol, historical_data)
            if not mtf_data:
                return self._create_fail_safe_result("No MTF data available")
            
            # Analyze each timeframe
            tf_analysis = self._analyze_timeframes(symbol, mtf_data)
            
            # Detailed debug logging - consolidated timeframe analysis
            self._log_consolidated_timeframe_analysis(symbol, tf_analysis)
            
            # Determine trade validation
            validation_result = self._determine_validation(signal, tf_analysis)
            
            self.logger.info(f"[MTF_VALIDATOR] {symbol} validation: {validation_result.reasoning}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error validating {symbol}: {e}")
            return self._create_fail_safe_result(f"Validation error: {e}")
    
    def _get_mtf_data(self, symbol: str, historical_data: List[Dict]) -> Optional[Dict]:
        """
        Get REAL multi-timeframe data (1m, 5m, 15m) from MT5
        
        Args:
            symbol: Trading symbol
            historical_data: Historical data from main algorithm
            
        Returns:
            Dictionary with 1m, 5m, and 15m data or None if unavailable
        """
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, timedelta
            
            # Use existing historical data as 1m data
            if not historical_data or len(historical_data) < 50:
                self.logger.warning(f"[MTF_VALIDATOR] Insufficient 1m data for {symbol}")
                return None
            
            df_1m = pd.DataFrame(historical_data[-100:])  # Last 100 bars for 1m
            
            # 🚨 CRITICAL FIX: Get REAL higher timeframe data from MT5
            current_time = datetime.now()
            
            # Track if we're using real MT5 data
            using_real_data = False
            
            # Check if MT5 is initialized
            if not mt5.initialize():
                self.logger.warning(f"[MTF_VALIDATOR] MT5 not initialized, using resampled data for {symbol}")
                df_5m = df_1m.iloc[::5].copy()
                df_15m = df_1m.iloc[::15].copy()
            else:
                # Get real 5m data from MT5 - Look back 1000 bars for proper trend analysis
                try:
                    rates_5m = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1000)
                    if rates_5m is not None and len(rates_5m) > 0:
                        df_5m = pd.DataFrame(rates_5m)
                        df_5m['time'] = pd.to_datetime(df_5m['time'], unit='s')
                        self.logger.info(f"[MTF_VALIDATOR] Got REAL 5m data for {symbol}: {len(df_5m)} bars (1000 requested)")
                        using_real_data = True
                    else:
                        self.logger.warning(f"[MTF_VALIDATOR] No 5m data from MT5 for {symbol}, using resampled")
                        df_5m = df_1m.iloc[::5].copy()
                except Exception as e:
                    self.logger.warning(f"[MTF_VALIDATOR] Error getting 5m data for {symbol}: {e}, using resampled")
                    df_5m = df_1m.iloc[::5].copy()
                
                # Get real 15m data from MT5 - Look back 1000 bars for proper trend analysis
                try:
                    rates_15m = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1000)
                    if rates_15m is not None and len(rates_15m) > 0:
                        df_15m = pd.DataFrame(rates_15m)
                        df_15m['time'] = pd.to_datetime(df_15m['time'], unit='s')
                        self.logger.info(f"[MTF_VALIDATOR] Got REAL 15m data for {symbol}: {len(df_15m)} bars (1000 requested)")
                        using_real_data = True
                    else:
                        self.logger.warning(f"[MTF_VALIDATOR] No 15m data from MT5 for {symbol}, using resampled")
                        df_15m = df_1m.iloc[::15].copy()
                except Exception as e:
                    self.logger.warning(f"[MTF_VALIDATOR] Error getting 15m data for {symbol}: {e}, using resampled")
                    df_15m = df_1m.iloc[::15].copy()
            
            # Set flag for logging
            self._using_real_mtf_data = using_real_data
            
            return {
                '1m': df_1m,
                '5m': df_5m,
                '15m': df_15m
            }
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error getting MTF data for {symbol}: {e}")
            return None
    
    def _analyze_timeframes(self, symbol: str, mtf_data: Dict) -> Dict[str, Dict]:
        """
        Analyze each timeframe using the EXACT SAME indicators and ML logic as the 1-minute engine
        
        Args:
            symbol: Trading symbol
            mtf_data: Multi-timeframe data dictionary
            
        Returns:
            Dictionary with analysis for each timeframe
        """
        analysis = {'symbol': symbol}  # Include symbol for debug logging
        
        for tf_name, df in mtf_data.items():
            # Enhanced data validation
            if df is None or len(df) < 20:  # Need minimum data for indicators
                self.logger.warning(f"[MTF_VALIDATOR] {tf_name.upper()} data insufficient: {len(df) if df is not None else 'None'} bars")
                analysis[tf_name] = {
                    'direction': 'neutral', 
                    'strength': 0.0, 
                    'feature_series': None,
                    'ml_signal': 0,
                    'confidence': 0.0
                }
                continue
            
            # Validate DataFrame structure
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                self.logger.warning(f"[MTF_VALIDATOR] {tf_name.upper()} missing required columns: {required_columns}")
                analysis[tf_name] = {
                    'direction': 'neutral', 
                    'strength': 0.0, 
                    'feature_series': None,
                    'ml_signal': 0,
                    'confidence': 0.0
                }
                continue
            
            # Log data info for debugging
            self.logger.debug(f"[MTF_VALIDATOR] {tf_name.upper()} data: {len(df)} bars, columns: {list(df.columns)}")
                
            # Calculate features using the EXACT SAME logic as the main algorithm
            feature_series = self._calculate_features_same_as_main(df)
            
            # Calculate ML signal using Lorentzian distance classifier
            ml_signal, confidence = self._calculate_ml_signal_same_as_main(df, feature_series)
            
            # Determine trend direction from ML signal and indicators (OLD METHOD)
            old_direction = self._determine_trend_from_ml_signal(ml_signal, feature_series)
            
            # NEW METHOD: Robust trend detection using EMA + price action + indicators
            new_direction, new_confidence = self.detect_trend_robust(df, lookback=5)
            
            # Log comparison between old and new methods
            self.logger.info(f"[TREND_CHECK] {symbol} {tf_name.upper()}: old={old_direction}, new={new_direction}(conf={new_confidence:.3f}), last_closes={df['close'].iloc[-6:].to_list()}")
            
            if old_direction != new_direction:
                self.logger.warning(f"[TREND_MISMATCH] {symbol} {tf_name.upper()}: old={old_direction} vs new={new_direction} - investigate.")
                # Optional CSV write for debugging
                try:
                    df.tail(50).to_csv(f"/tmp/{symbol}_{tf_name}_debug.csv")
                except:
                    pass  # Ignore CSV write errors
            
            # Use the new robust trend detection
            direction = new_direction
            
            # Calculate trend strength from confidence and indicator alignment
            strength = self._calculate_trend_strength_from_ml(confidence, feature_series)
            
            analysis[tf_name] = {
                'direction': direction,
                'strength': strength,
                'feature_series': feature_series,
                'ml_signal': ml_signal,
                'confidence': confidence
            }
            
            # Debug logging for each timeframe
            self.logger.info(f"[MTF_DEBUG] {tf_name.upper()}: signal={ml_signal}, confidence={confidence:.3f}, direction={direction}")
            
        # Mapping sanity check to detect sign->label inconsistencies
        # Note: We now use robust trend detection, so we check the actual direction vs what the ML signal would suggest
        _signal_to_label = {1: "bullish", 0: "neutral", -1: "bearish"}
        
        for tf_name, tf_data in analysis.items():
            if tf_name == 'symbol':  # Skip the symbol entry
                continue
                
            sig = tf_data.get("ml_signal", 0)
            logged_dir = tf_data.get("direction", "neutral")
            expected_from_ml = _signal_to_label.get(sig, "unknown")
            
            # Log the comparison but don't treat it as an error since we're using robust detection
            self.logger.debug(f"[MAPPING_CHECK] {symbol} {tf_name.upper()}: ML_signal={sig}->{expected_from_ml}, robust_direction={logged_dir}")
            
            # Only flag as error if there's a clear contradiction (e.g., ML says bullish but robust says bearish)
            if (sig == 1 and logged_dir == "bearish") or (sig == -1 and logged_dir == "bullish"):
                self.logger.warning(f"[MAPPING_CONFLICT] {symbol} {tf_name.upper()}: ML_signal={sig}->{expected_from_ml} conflicts with robust_direction={logged_dir}")
            
        return analysis
    
    def _calculate_features_same_as_main(self, df: pd.DataFrame) -> FeatureSeries:
        """
        Calculate features using the EXACT SAME logic as the main algorithm
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            FeatureSeries with the same features as the main algorithm
        """
        try:
            # Get the most recent bar data
            latest_bar = df.iloc[-1]
            close = float(latest_bar['close'])
            high = float(latest_bar['high'])
            low = float(latest_bar['low'])
            hlc3 = (high + low + close) / 3
            
            # Debug logging for price data
            self.logger.debug(f"[FEATURE_CALC] Price data - Close: {close}, High: {high}, Low: {low}, HLC3: {hlc3}")
            
            # Calculate features using the EXACT SAME series_from function
            f1 = series_from(self.feature_params['f1']['string'], close, high, low, hlc3,
                            self.feature_params['f1']['param_a'], self.feature_params['f1']['param_b'])
            f2 = series_from(self.feature_params['f2']['string'], close, high, low, hlc3,
                            self.feature_params['f2']['param_a'], self.feature_params['f2']['param_b'])
            f3 = series_from(self.feature_params['f3']['string'], close, high, low, hlc3,
                            self.feature_params['f3']['param_a'], self.feature_params['f3']['param_b'])
            f4 = series_from(self.feature_params['f4']['string'], close, high, low, hlc3,
                            self.feature_params['f4']['param_a'], self.feature_params['f4']['param_b'])
            f5 = series_from(self.feature_params['f5']['string'], close, high, low, hlc3,
                            self.feature_params['f5']['param_a'], self.feature_params['f5']['param_b'])
            
            # Debug logging for calculated features
            self.logger.debug(f"[FEATURE_CALC] Features - F1: {f1}, F2: {f2}, F3: {f3}, F4: {f4}, F5: {f5}")
            
            return FeatureSeries(f1=f1, f2=f2, f3=f3, f4=f4, f5=f5)
            
        except TypeError as e:
            if "object of type 'float' has no len()" in str(e):
                self.logger.warning(f"[FEATURE_CALC] NEUTRAL due to float error: {e}")
            else:
                self.logger.warning(f"[FEATURE_CALC] TypeError in feature calculation: {e}")
            return FeatureSeries(f1=50.0, f2=-50.0, f3=0.0, f4=20.0, f5=50.0)
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error calculating features: {e}")
            return FeatureSeries(f1=50.0, f2=-50.0, f3=0.0, f4=20.0, f5=50.0)
    
    def _calculate_ml_signal_same_as_main(self, df: pd.DataFrame, feature_series: FeatureSeries) -> Tuple[int, float]:
        """
        Calculate ML signal using the EXACT SAME Lorentzian distance classifier logic
        
        Args:
            df: DataFrame with OHLC data
            feature_series: Calculated feature series
            
        Returns:
            Tuple of (ml_signal, confidence)
        """
        try:
            # Build feature arrays from historical data for Lorentzian distance calculation
            feature_arrays = self._build_feature_arrays_from_dataframe(df)
            
            # Build training labels from price movements
            training_labels = self._build_training_labels_from_dataframe(df)
            
            if len(training_labels) < self.settings.neighbors_count:
                # Not enough data for ML prediction
                return 0, 0.0
            
            # Use the EXACT SAME Lorentzian distance classifier logic
            prediction = self._approximate_nearest_neighbors_same_as_main(
                feature_series, feature_arrays, training_labels
            )
            
            # Apply the EXACT SAME filters as the main algorithm
            filter_all = self._apply_filters_same_as_main(prediction, feature_series)
            
            if not filter_all:
                return 0, 0.0  # Filtered out
            
            # Generate signal using the EXACT SAME logic as main algorithm
            if prediction > 0:
                ml_signal = 1  # Long
            elif prediction < 0:
                ml_signal = -1  # Short
            else:
                ml_signal = 0  # Neutral
            
            # Calculate confidence using the EXACT SAME method as main algorithm
            confidence = abs(prediction) / self.settings.neighbors_count
            
            return ml_signal, confidence
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error calculating ML signal: {e}")
            return 0, 0.0
    
    def _determine_trend_from_ml_signal(self, ml_signal: int, feature_series: FeatureSeries) -> str:
        """
        Determine trend direction from ML signal and feature series with enhanced trend detection
        
        Args:
            ml_signal: ML signal from Lorentzian classifier (1, -1, 0)
            feature_series: Feature series with indicator values
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        # Use ML signal as primary indicator
        if ml_signal == 1:
            return 'bullish'
        elif ml_signal == -1:
            return 'bearish'
        else:
            # ML signal is 0 (neutral), but check indicators for trend bias
            return self._detect_trend_from_indicators(feature_series)
    
    def _detect_trend_from_indicators(self, feature_series: FeatureSeries) -> str:
        """
        Detect trend direction from indicator values when ML signal is neutral
        
        Args:
            feature_series: Feature series with indicator values
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        try:
            # Get the indicator values (FeatureSeries contains single float values, not arrays)
            # Add explicit type checking and conversion to handle any data type issues
            rsi_14 = float(feature_series.f1) if isinstance(feature_series.f1, (int, float)) else 50.0
            wt_10 = float(feature_series.f2) if isinstance(feature_series.f2, (int, float)) else 0.0
            cci_20 = float(feature_series.f3) if isinstance(feature_series.f3, (int, float)) else 0.0
            adx_20 = float(feature_series.f4) if isinstance(feature_series.f4, (int, float)) else 0.0
            rsi_9 = float(feature_series.f5) if isinstance(feature_series.f5, (int, float)) else 50.0
            
            # Debug logging for indicator values and data types
            self.logger.debug(f"[TREND_DETECTION] RSI14={rsi_14:.2f}, WT10={wt_10:.2f}, CCI20={cci_20:.2f}, ADX20={adx_20:.2f}, RSI9={rsi_9:.2f}")
            self.logger.debug(f"[TREND_DETECTION] Data types - f1: {type(feature_series.f1)}, f2: {type(feature_series.f2)}, f3: {type(feature_series.f3)}, f4: {type(feature_series.f4)}, f5: {type(feature_series.f5)}")
            
            # Calculate trend score based on multiple indicators
            bullish_score = 0
            bearish_score = 0
            
            # RSI(14) trend bias
            if rsi_14 > 60:
                bullish_score += 1
            elif rsi_14 < 40:
                bearish_score += 1
            
            # RSI(9) trend bias  
            if rsi_9 > 60:
                bullish_score += 1
            elif rsi_9 < 40:
                bearish_score += 1
                
            # CCI trend bias
            if cci_20 > 100:
                bullish_score += 1
            elif cci_20 < -100:
                bearish_score += 1
                
            # Williams %R trend bias (WT)
            if wt_10 > -20:
                bullish_score += 1
            elif wt_10 < -80:
                bearish_score += 1
                
            # ADX strength filter (only consider trend if ADX > 25)
            if adx_20 > 25:
                # Strong trend - weight the signals more
                if bullish_score > bearish_score:
                    self.logger.debug(f"[TREND_DETECTION] Strong bullish trend detected: Bull={bullish_score}, Bear={bearish_score}, ADX={adx_20:.2f}")
                    return 'bullish'
                elif bearish_score > bullish_score:
                    self.logger.debug(f"[TREND_DETECTION] Strong bearish trend detected: Bull={bullish_score}, Bear={bearish_score}, ADX={adx_20:.2f}")
                    return 'bearish'
            else:
                # Weak trend - only return direction if there's a clear bias
                if bullish_score >= 3:
                    self.logger.debug(f"[TREND_DETECTION] Weak bullish trend detected: Bull={bullish_score}, Bear={bearish_score}, ADX={adx_20:.2f}")
                    return 'bullish'
                elif bearish_score >= 3:
                    self.logger.debug(f"[TREND_DETECTION] Weak bearish trend detected: Bull={bullish_score}, Bear={bearish_score}, ADX={adx_20:.2f}")
                    return 'bearish'
            
            self.logger.debug(f"[TREND_DETECTION] No clear trend: Bull={bullish_score}, Bear={bearish_score}, ADX={adx_20:.2f}")
            return 'neutral'
            
        except TypeError as e:
            if "object of type 'float' has no len()" in str(e):
                self.logger.warning(f"[TREND_DETECTION] NEUTRAL due to float error: {e}")
            else:
                self.logger.warning(f"[TREND_DETECTION] TypeError in trend detection: {e}")
            return 'neutral'
        except Exception as e:
            self.logger.warning(f"[TREND_DETECTION] Error in trend detection from indicators: {e}")
            return 'neutral'
    
    def detect_trend_robust(self, df: pd.DataFrame, lookback: int = 5, ema_fast: int = 20, ema_slow: int = 50) -> Tuple[str, float]:
        """
        Robust trend detection using EMA + price action + indicators
        
        Args:
            df: pandas DataFrame with columns ['open','high','low','close'] indexed oldest->newest
            lookback: Number of candles to look back for price action analysis
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            
        Returns:
            Tuple of (trend_label, confidence_float) where trend_label in {"bullish","neutral","bearish"}
        """
        try:
            if len(df) < max(ema_slow, lookback) + 1:
                return "neutral", 0.0

            # EMAs
            ema_fast_series = df['close'].ewm(span=ema_fast, adjust=False).mean()
            ema_slow_series = df['close'].ewm(span=ema_slow, adjust=False).mean()
            ema_trend = "bullish" if ema_fast_series.iloc[-1] > ema_slow_series.iloc[-1] else "bearish"

            # Price-action swing: check monotonic highs/lows over lookback
            highs = df['high'].iloc[-(lookback+1):].to_numpy()
            lows = df['low'].iloc[-(lookback+1):].to_numpy()

            # require at least 2 sequential moves to count as HH/HL or LH/LL
            hh = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
            hl = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
            ll = all(lows[i] < lows[i-1] for i in range(1, len(lows)))
            lh = all(highs[i] < highs[i-1] for i in range(1, len(highs)))

            if hh and hl:
                pa_trend = "bullish"
            elif lh and ll:
                pa_trend = "bearish"
            else:
                pa_trend = "neutral"

            # Simple indicator aggregation: check RSI/ADX/CCI sign over last candle
            indicator_votes = []
            
            # Calculate indicators if not present
            try:
                rsi_series = calculate_rsi_pine(series_from(df['close']), 14, 1)
                last_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50.0
                if last_rsi >= 60:
                    indicator_votes.append("bullish")
                elif last_rsi <= 40:
                    indicator_votes.append("bearish")
                else:
                    indicator_votes.append("neutral")
            except:
                pass

            try:
                adx_series = calculate_adx_pine(series_from(df['high']), series_from(df['low']), series_from(df['close']), 20, 2)
                last_adx = adx_series.iloc[-1] if len(adx_series) > 0 else 25.0
                # For ADX, we need to check if it's strong enough to consider trend
                if last_adx >= 25:
                    # If ADX is strong, use EMA trend as the directional component
                    if ema_trend == "bullish":
                        indicator_votes.append("bullish")
                    elif ema_trend == "bearish":
                        indicator_votes.append("bearish")
                    else:
                        indicator_votes.append("neutral")
                else:
                    indicator_votes.append("neutral")
            except:
                pass

            try:
                cci_series = calculate_cci_pine(series_from(df['high']), series_from(df['low']), series_from(df['close']), 20, 1)
                last_cci = cci_series.iloc[-1] if len(cci_series) > 0 else 0.0
                if last_cci >= 100:
                    indicator_votes.append("bullish")
                elif last_cci <= -100:
                    indicator_votes.append("bearish")
                else:
                    indicator_votes.append("neutral")
            except:
                pass

            # Build votes: ema_trend, pa_trend, indicator_votes...
            votes = [ema_trend, pa_trend] + indicator_votes
            vote_counts = Counter(votes)
            most_common, count = vote_counts.most_common(1)[0]

            # Confidence scaled 0..1 by fraction of votes for the winner
            confidence = count / max(1, len(votes))

            # If the results are a tie or low confidence, report neutral
            if confidence < 0.5:
                return "neutral", confidence

            return most_common, float(confidence)
            
        except Exception as e:
            self.logger.warning(f"[ROBUST_TREND] Error in robust trend detection: {e}")
            return "neutral", 0.0
    
    def _calculate_trend_strength_from_ml(self, confidence: float, feature_series: FeatureSeries) -> float:
        """
        Calculate trend strength from ML confidence and feature series
        
        Args:
            confidence: ML confidence score
            feature_series: Feature series with indicator values
            
        Returns:
            Trend strength as float between 0.0 and 1.0
        """
        # Use ML confidence as primary strength indicator
        ml_strength = confidence
        
        # Add ADX strength as confirmation
        adx_strength = min(feature_series.f4 / 50.0, 1.0)  # Normalize ADX
        
        # Combine ML confidence with ADX strength
        combined_strength = (ml_strength + adx_strength) / 2
        
        return min(combined_strength, 1.0)
    
    def _determine_validation(self, signal: int, tf_analysis: Dict) -> MTFValidationResult:
        """
        Determine trade validation based on equal weighting (33-33-33) and majority voting
        
        Rules:
        1. All instruments use equal weighting: 1m=33%, 5m=33%, 15m=33%
        2. Majority voting: 3/3 = strong, 2/3 = moderate, 1/3 = blocked
        3. Redistribute weights among active timeframes when neutrals present
        4. Block all counter-trend trades
        
        Args:
            signal: Trade signal (1 for long, -1 for short)
            tf_analysis: Timeframe analysis results
            
        Returns:
            MTFValidationResult with validation outcome
        """
        # Get timeframe signals and confidence for debug logging
        signal_1m = tf_analysis.get('1m', {}).get('ml_signal', 0)
        confidence_1m = tf_analysis.get('1m', {}).get('confidence', 0.0)
        signal_5m = tf_analysis.get('5m', {}).get('ml_signal', 0)
        confidence_5m = tf_analysis.get('5m', {}).get('confidence', 0.0)
        signal_15m = tf_analysis.get('15m', {}).get('ml_signal', 0)
        confidence_15m = tf_analysis.get('15m', {}).get('confidence', 0.0)
        
        # Get timeframe directions
        tf_1m = tf_analysis.get('1m', {}).get('direction', 'neutral')
        tf_5m = tf_analysis.get('5m', {}).get('direction', 'neutral')
        tf_15m = tf_analysis.get('15m', {}).get('direction', 'neutral')
        
        # Get timeframe strengths
        strength_1m = tf_analysis.get('1m', {}).get('strength', 0.5)
        strength_5m = tf_analysis.get('5m', {}).get('strength', 0.5)
        strength_15m = tf_analysis.get('15m', {}).get('strength', 0.5)
        
        # Use robust trend detection results for all timeframes
        # The robust detection gives us the actual trend direction based on price action + indicators
        signal_direction = 'bullish' if signal > 0 else 'bearish'
        
        # Enhanced HTF vs LTF logging with data source info
        ltf_signal = f"1m={tf_1m.upper()}"  # Use robust detection result for 1m too
        htf_5m = f"5m={tf_5m.upper()}" if tf_5m != 'neutral' else "5m=NEUTRAL"
        htf_15m = f"15m={tf_15m.upper()}" if tf_15m != 'neutral' else "15m=NEUTRAL"
        
        # Check if we're using real MT5 data or resampled data
        data_source = "REAL_MT5" if hasattr(self, '_using_real_mtf_data') and self._using_real_mtf_data else "RESAMPLED"
        
        self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | LTF: {ltf_signal} | HTF: {htf_5m}, {htf_15m} | Data: {data_source}")
        self.logger.info(f"[MTF_DEBUG] Symbol: {tf_analysis.get('symbol', 'UNKNOWN')} | 1m: {signal_1m}({confidence_1m:.3f}), 5m: {signal_5m}({confidence_5m:.3f}), 15m: {signal_15m}({confidence_15m:.3f}), Final Action: {signal_direction}")
        
        # EQUAL WEIGHTING (33-33-33) MAJORITY VOTING SYSTEM
        # Use robust trend detection results for alignment checking
        
        # Count timeframes by alignment with the main signal direction
        aligned_count = 0
        opposing_count = 0
        neutral_count = 0
        
        # Check 1m alignment (use robust detection result)
        if tf_1m == signal_direction:
            aligned_count += 1
        elif tf_1m == 'neutral':
            neutral_count += 1
        else:
            opposing_count += 1
        
        # Check 5m alignment (use robust detection result)
        if tf_5m == signal_direction:
            aligned_count += 1
        elif tf_5m == 'neutral':
            neutral_count += 1
        else:
            opposing_count += 1
        
        # Check 15m alignment (use robust detection result)
        if tf_15m == signal_direction:
            aligned_count += 1
        elif tf_15m == 'neutral':
            neutral_count += 1
        else:
            opposing_count += 1
        
        # Calculate active timeframes (non-neutral)
        active_count = aligned_count + opposing_count
        total_count = 3
        
        # RULE 1: Perfect alignment (3/3) - Strong signal
        if aligned_count == 3:
            self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | ALLOWED: 3/3 alignment → strong signal")
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=1.3,  # Maximum position for perfect alignment
                tp_multiplier=1.2,   # Extended TP for strong trend
                confidence_boost=0.3,  # 100% confidence (3/3 = 100%)
                reasoning=f"Perfect alignment - all 3 timeframes {signal_direction} (100% confidence)",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=1.0,  # 100% score
                scenario_label="perfect_alignment"
            )
        
        # RULE 2: Majority support (2/3) - Moderate signal
        elif aligned_count == 2 and opposing_count == 0:
            # Redistribute weights: 2 active timeframes = 50/50
            self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | ALLOWED: 2/3 alignment → majority support")
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=1.0,  # Standard position for majority support
                tp_multiplier=1.0,   # Standard TP
                confidence_boost=0.2,  # 66% confidence (2/3 = 66%)
                reasoning=f"Majority support - 2/3 timeframes {signal_direction} (66% confidence)",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.67,  # 66% score
                scenario_label="majority_support"
            )
        
        # RULE 3: Counter-trend detected - Block trade
        elif opposing_count > 0:
            self.logger.warning(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | BLOCKED: counter-trend detected")
            return MTFValidationResult(
                allow_trade=False,
                lot_multiplier=0.0,
                tp_multiplier=0.0,
                confidence_boost=0.0,
                reasoning=f"BLOCKED: Counter-trend detected - {opposing_count} timeframes oppose {signal_direction} signal",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.0,
                scenario_label="blocked_countertrend"
            )
        
        # RULE 4: Only 1/3 aligned - Block trade
        elif aligned_count == 1:
            self.logger.warning(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | BLOCKED: only 1/3 aligned")
            return MTFValidationResult(
                allow_trade=False,
                lot_multiplier=0.0,
                tp_multiplier=0.0,
                confidence_boost=0.0,
                reasoning=f"BLOCKED: Insufficient support - only 1/3 timeframes {signal_direction} (33% confidence)",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.33,  # 33% score
                scenario_label="blocked_insufficient_support"
            )
        
        # RULE 5: All neutral - Block trade
        else:
            self.logger.warning(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | BLOCKED: all timeframes neutral")
            return MTFValidationResult(
                allow_trade=False,
                lot_multiplier=0.0,
                tp_multiplier=0.0,
                confidence_boost=0.0,
                reasoning=f"BLOCKED: All timeframes neutral - no clear direction",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.0,
                scenario_label="blocked_all_neutral"
            )
        
    
    def _create_fail_safe_result(self, reason: str) -> MTFValidationResult:
        """
        Create a fail-safe result that allows the original algorithm to proceed
        
        Args:
            reason: Reason for fail-safe
            
        Returns:
            MTFValidationResult that allows trade with original parameters
        """
        return MTFValidationResult(
            allow_trade=True,
            lot_multiplier=1.0,
            tp_multiplier=1.0,
            confidence_boost=0.0,
            reasoning=f"Fail-safe: {reason}",
            timeframe_alignment={'1m': 'unknown', '5m': 'unknown', '15m': 'unknown'},
            validation_score=0.5,
            scenario_label="fail_safe"
        )
    
    def _build_feature_arrays_from_dataframe(self, df: pd.DataFrame) -> FeatureArrays:
        """
        Build feature arrays from DataFrame using the EXACT SAME logic as main algorithm
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            FeatureArrays with calculated features
        """
        try:
            f1_list = []
            f2_list = []
            f3_list = []
            f4_list = []
            f5_list = []
            
            # Calculate features for each bar in the DataFrame
            for i in range(len(df)):
                bar = df.iloc[i]
                close = bar['close']
                high = bar['high']
                low = bar['low']
                hlc3 = (high + low + close) / 3
                
                # Calculate features using the EXACT SAME series_from function
                f1 = series_from(self.feature_params['f1']['string'], close, high, low, hlc3,
                               self.feature_params['f1']['param_a'], self.feature_params['f1']['param_b'])
                f2 = series_from(self.feature_params['f2']['string'], close, high, low, hlc3,
                               self.feature_params['f2']['param_a'], self.feature_params['f2']['param_b'])
                f3 = series_from(self.feature_params['f3']['string'], close, high, low, hlc3,
                               self.feature_params['f3']['param_a'], self.feature_params['f3']['param_b'])
                f4 = series_from(self.feature_params['f4']['string'], close, high, low, hlc3,
                               self.feature_params['f4']['param_a'], self.feature_params['f4']['param_b'])
                f5 = series_from(self.feature_params['f5']['string'], close, high, low, hlc3,
                               self.feature_params['f5']['param_a'], self.feature_params['f5']['param_b'])
                
                f1_list.append(f1)
                f2_list.append(f2)
                f3_list.append(f3)
                f4_list.append(f4)
                f5_list.append(f5)
            
            return FeatureArrays(
                f1=f1_list,
                f2=f2_list,
                f3=f3_list,
                f4=f4_list,
                f5=f5_list
            )
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error building feature arrays: {e}")
            return FeatureArrays(f1=[], f2=[], f3=[], f4=[], f5=[])
    
    def _build_training_labels_from_dataframe(self, df: pd.DataFrame) -> List[int]:
        """
        Build training labels from DataFrame using the EXACT SAME logic as main algorithm
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of training labels
        """
        try:
            training_labels = []
            
            # Calculate training labels based on price movements
            for i in range(1, len(df)):
                current_price = df.iloc[i]['close']
                future_price = df.iloc[i-1]['close']  # Previous bar as "future" for training
                
                # Use the EXACT SAME training label calculation as main algorithm
                if future_price < current_price:
                    training_labels.append(-1)  # Short
                elif future_price > current_price:
                    training_labels.append(1)   # Long
                else:
                    training_labels.append(0)   # Neutral
            
            return training_labels
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error building training labels: {e}")
            return []
    
    def _approximate_nearest_neighbors_same_as_main(self, feature_series: FeatureSeries, 
                                                   feature_arrays: FeatureArrays, 
                                                   training_labels: List[int]) -> float:
        """
        Approximate Nearest Neighbors Search using the EXACT SAME Lorentzian distance logic
        
        Args:
            feature_series: Current feature series
            feature_arrays: Historical feature arrays
            training_labels: Historical training labels
            
        Returns:
            Prediction score
        """
        try:
            last_distance = -1.0
            size = min(self.settings.max_bars_back - 1, len(training_labels) - 1)
            size_loop = min(self.settings.max_bars_back - 1, size)
            
            distances = []
            predictions = []
            
            # Use the EXACT SAME Lorentzian distance algorithm as main algorithm
            for i in range(size_loop):
                d = get_lorentzian_distance(i, self.settings.feature_count, 
                                          feature_series, feature_arrays)
                
                # Use the EXACT SAME logic: only process every 4th bar for chronological spacing
                if d >= last_distance and i % 4 == 0:
                    last_distance = d
                    distances.append(d)
                    predictions.append(round(training_labels[i]))
                    
                    # Use the EXACT SAME logic: maintain neighbors count limit
                    if len(predictions) > self.settings.neighbors_count:
                        # Use the EXACT SAME logic: boost accuracy by using lower 25% distance
                        last_distance = distances[int(self.settings.neighbors_count * 3 / 4)]
                        distances.pop(0)
                        predictions.pop(0)
            
            # Use the EXACT SAME logic: sum predictions for final signal
            return sum(predictions)
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error in approximate nearest neighbors: {e}")
            return 0.0
    
    def _apply_filters_same_as_main(self, prediction: float, feature_series: FeatureSeries) -> bool:
        """
        Apply filters using the EXACT SAME logic as main algorithm
        
        Args:
            prediction: ML prediction
            feature_series: Feature series
            
        Returns:
            True if filters pass, False if blocked
        """
        try:
            # Use the EXACT SAME filter logic as main algorithm
            filter_all = True
            
            # Volatility filter - uses Williams %R (f2) as volatility proxy
            volatility = abs(feature_series.f2)  # Use Williams %R as volatility proxy
            if volatility < 20:  # Low volatility threshold
                filter_all = False
                self.logger.debug(f"Volatility filter blocked: volatility={volatility:.2f} < 20")
            
            # Regime filter - uses CCI (f3) as regime indicator
            if feature_series.f3 < -0.1:  # Regime threshold
                filter_all = False
                self.logger.debug(f"Regime filter blocked: CCI={feature_series.f3:.2f} < -0.1")
            
            # ADX filter - uses ADX (f4) for trend strength
            if feature_series.f4 < 20:  # ADX threshold
                filter_all = False
                self.logger.debug(f"ADX filter blocked: ADX={feature_series.f4:.2f} < 20")
            
            return filter_all
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error applying filters: {e}")
            return True  # Fail safe - allow trade if filter fails
    
    def _log_consolidated_timeframe_analysis(self, symbol: str, tf_analysis: Dict):
        """
        Log consolidated timeframe analysis in the requested format
        
        Args:
            symbol: Trading symbol
            tf_analysis: Timeframe analysis results
        """
        try:
            # Get 1m analysis
            tf_1m = tf_analysis.get('1m', {})
            direction_1m = tf_1m.get('direction', 'neutral')
            confidence_1m = tf_1m.get('confidence', 0.0)
            
            # Get 5m analysis
            tf_5m = tf_analysis.get('5m', {})
            direction_5m = tf_5m.get('direction', 'neutral')
            confidence_5m = tf_5m.get('confidence', 0.0)
            
            # Get 15m analysis
            tf_15m = tf_analysis.get('15m', {})
            direction_15m = tf_15m.get('direction', 'neutral')
            confidence_15m = tf_15m.get('confidence', 0.0)
            
            # Log consolidated analysis
            self.logger.info(f"[MTF_DEBUG] {symbol} | 1m: {direction_1m}(conf={confidence_1m:.3f}), 5m: {direction_5m}(conf={confidence_5m:.3f}), 15m: {direction_15m}(conf={confidence_15m:.3f})")
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error logging consolidated analysis: {e}")
