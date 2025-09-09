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
            if len(df) < 20:  # Need minimum data for indicators
                analysis[tf_name] = {
                    'direction': 'neutral', 
                    'strength': 0.0, 
                    'feature_series': None,
                    'ml_signal': 0,
                    'confidence': 0.0
                }
                continue
                
            # Calculate features using the EXACT SAME logic as the main algorithm
            feature_series = self._calculate_features_same_as_main(df)
            
            # Calculate ML signal using Lorentzian distance classifier
            ml_signal, confidence = self._calculate_ml_signal_same_as_main(df, feature_series)
            
            # Determine trend direction from ML signal and indicators
            direction = self._determine_trend_from_ml_signal(ml_signal, feature_series)
            
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
            close = latest_bar['close']
            high = latest_bar['high']
            low = latest_bar['low']
            hlc3 = (high + low + close) / 3
            
            # Calculate features using the EXACT SAME series_from function
            return FeatureSeries(
                f1=series_from(self.feature_params['f1']['string'], close, high, low, hlc3,
                              self.feature_params['f1']['param_a'], self.feature_params['f1']['param_b']),
                f2=series_from(self.feature_params['f2']['string'], close, high, low, hlc3,
                              self.feature_params['f2']['param_a'], self.feature_params['f2']['param_b']),
                f3=series_from(self.feature_params['f3']['string'], close, high, low, hlc3,
                              self.feature_params['f3']['param_a'], self.feature_params['f3']['param_b']),
                f4=series_from(self.feature_params['f4']['string'], close, high, low, hlc3,
                              self.feature_params['f4']['param_a'], self.feature_params['f4']['param_b']),
                f5=series_from(self.feature_params['f5']['string'], close, high, low, hlc3,
                              self.feature_params['f5']['param_a'], self.feature_params['f5']['param_b'])
            )
            
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
        Determine trend direction from ML signal and feature series
        
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
            return 'neutral'
    
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
        Determine trade validation based on intelligent pullback detection and weighted timeframe analysis
        
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
        
        # Determine signal direction
        signal_direction = 'bullish' if signal > 0 else 'bearish'
        
        # Enhanced HTF vs LTF logging with data source info
        ltf_signal = f"1m={signal_direction.upper()}"
        htf_5m = f"5m={tf_5m.upper()}" if tf_5m != 'neutral' else "5m=NEUTRAL"
        htf_15m = f"15m={tf_15m.upper()}" if tf_15m != 'neutral' else "15m=NEUTRAL"
        
        # Check if we're using real MT5 data or resampled data
        data_source = "REAL_MT5" if hasattr(self, '_using_real_mtf_data') and self._using_real_mtf_data else "RESAMPLED"
        
        self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | LTF: {ltf_signal} | HTF: {htf_5m}, {htf_15m} | Data: {data_source}")
        self.logger.info(f"[MTF_DEBUG] Symbol: {tf_analysis.get('symbol', 'UNKNOWN')} | 1m: {signal_1m}({confidence_1m:.3f}), 5m: {signal_5m}({confidence_5m:.3f}), 15m: {signal_15m}({confidence_15m:.3f}), Final Action: {signal_direction}")
        
        # PULLBACK DETECTION LOGIC
        # Check for bullish pullback: 15M bullish, 5M+1M bearish
        if tf_15m == 'bullish' and tf_5m == 'bearish' and tf_1m == 'bearish':
            if signal_direction == 'bearish':  # Countertrend short in bullish pullback
                return MTFValidationResult(
                    allow_trade=True,
                    lot_multiplier=0.4,  # Small countertrend position
                    tp_multiplier=0.6,   # Tighter TP for quick scalp
                    confidence_boost=0.0,
                    reasoning="Bullish pullback - countertrend short scalp",
                    timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                    validation_score=0.3,
                    scenario_label="pullback_countertrend"
                )
            else:  # Long signal in bullish pullback - trend continuation
                return MTFValidationResult(
                    allow_trade=True,
                    lot_multiplier=1.1,  # Larger position for trend continuation
                    tp_multiplier=1.0,   # Full TP targets
                    confidence_boost=0.15,
                    reasoning="Bullish pullback - trend continuation long",
                    timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                    validation_score=0.8,
                    scenario_label="pullback_continuation"
                )
        
        # Check for bearish pullback: 15M bearish, 5M+1M bullish
        elif tf_15m == 'bearish' and tf_5m == 'bullish' and tf_1m == 'bullish':
            if signal_direction == 'bullish':  # Countertrend long in bearish pullback
                return MTFValidationResult(
                    allow_trade=True,
                    lot_multiplier=0.4,  # Small countertrend position
                    tp_multiplier=0.6,   # Tighter TP for quick scalp
                    confidence_boost=0.0,
                    reasoning="Bearish pullback - countertrend long scalp",
                    timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                    validation_score=0.3,
                    scenario_label="pullback_countertrend"
                )
            else:  # Short signal in bearish pullback - trend continuation
                return MTFValidationResult(
                    allow_trade=True,
                    lot_multiplier=1.1,  # Larger position for trend continuation
                    tp_multiplier=1.0,   # Full TP targets
                    confidence_boost=0.15,
                    reasoning="Bearish pullback - trend continuation short",
                    timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                    validation_score=0.8,
                    scenario_label="pullback_continuation"
                )
        
        # 🚀 AGGRESSIVE COUNTER-TREND FILTERING WITH NEW WEIGHTING
        # New weighting: 15m=50%, 5m=40%, 1m=10%
        
        # First, check for counter-trend scenarios
        is_counter_trend = False
        htf_opposition_strength = 0.0
        
        # Determine if this is a counter-trend trade
        if signal_direction == 'bullish':
            # Long signal - check if HTF opposes
            if tf_15m == 'bearish' or tf_5m == 'bearish':
                is_counter_trend = True
                # Calculate HTF opposition strength
                if tf_15m == 'bearish':
                    htf_opposition_strength += 0.5 * strength_15m  # 15m weight
                if tf_5m == 'bearish':
                    htf_opposition_strength += 0.4 * strength_5m   # 5m weight
        else:  # signal_direction == 'bearish'
            # Short signal - check if HTF opposes
            if tf_15m == 'bullish' or tf_5m == 'bullish':
                is_counter_trend = True
                # Calculate HTF opposition strength
                if tf_15m == 'bullish':
                    htf_opposition_strength += 0.5 * strength_15m  # 15m weight
                if tf_5m == 'bullish':
                    htf_opposition_strength += 0.4 * strength_5m   # 5m weight
        
        # AGGRESSIVE COUNTER-TREND FILTERING
        if is_counter_trend:
            # Check if HTF strongly opposes the 1m signal
            if htf_opposition_strength > 0.6:  # Strong HTF opposition
                self.logger.warning(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | BLOCKED: Counter-trend trade blocked - HTF strongly opposes 1m signal | HTF Opposition: {htf_opposition_strength:.2f} | LTF={ltf_signal} vs HTF={htf_5m},{htf_15m}")
                return MTFValidationResult(
                    allow_trade=False,
                    lot_multiplier=0.0,
                    tp_multiplier=0.0,
                    confidence_boost=0.0,
                    reasoning=f"BLOCKED: Counter-trend trade - HTF strongly opposes 1m signal (opposition: {htf_opposition_strength:.2f})",
                    timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                    validation_score=0.0,
                    scenario_label="blocked_countertrend_htf_opposition"
                )
            else:
                # Allow counter-trend trade with temporary HTF support
                self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | ALLOWED: Counter-trend trade - temporary HTF support | HTF Opposition: {htf_opposition_strength:.2f} | LTF={ltf_signal} vs HTF={htf_5m},{htf_15m}")
                return MTFValidationResult(
                    allow_trade=True,
                    lot_multiplier=0.3,  # Small position for counter-trend
                    tp_multiplier=0.6,   # Tighter TP for quick exit
                    confidence_boost=0.0,
                    reasoning=f"Counter-trend trade allowed - temporary HTF support (opposition: {htf_opposition_strength:.2f})",
                    timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                    validation_score=0.3,
                    scenario_label="countertrend_temporary_support"
                )
        
        # WEIGHTED DECISION SYSTEM - Dynamic weights based on symbol type
        # Forex pairs: 5M=40%, 15M=40%, 1M=20% (smoother flow for majors)
        # Other assets: 15M=50%, 5M=40%, 1M=10% (original weights)
        
        # Determine if this is a forex pair
        forex_pairs = [
            'EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'USDCADm', 'USDCHFm', 'NZDUSDm',
            'EURJPYm', 'GBPJPYm', 'CADJPYm', 'EURAUDm', 'EURGBPm', 'GBPCHFm', 'GBPNZDm',
            'EURNZDm', 'AUDCHFm', 'AUDCADm', 'EURCHFm'
        ]
        
        symbol = tf_analysis.get('symbol', 'UNKNOWN')
        is_forex = symbol in forex_pairs  # Direct comparison, no case conversion needed
        
        # Debug logging (can be removed in production)
        # self.logger.info(f"[MTF_DEBUG] Symbol: {symbol}, is_forex: {is_forex}")
        
        if is_forex:
            # Forex pairs: 15M=40%, 5M=40%, 1M=20%
            tf_15m_weight = 0.4
            tf_5m_weight = 0.4
            tf_1m_weight = 0.2
            weight_desc = "15m=40%, 5m=40%, 1m=20%"
        else:
            # Other assets: 15M=50%, 5M=40%, 1M=10%
            tf_5m_weight = 0.4
            tf_15m_weight = 0.5
            tf_1m_weight = 0.1
            weight_desc = "15m=50%, 5m=40%, 1m=10%"
        
        # Calculate weighted score based on timeframe alignment
        weighted_score = 0.0
        max_score = 0.0
        
        # 15M timeframe weight (highest priority for both forex and other assets)
        if tf_15m == signal_direction:
            weighted_score += tf_15m_weight * strength_15m
        elif tf_15m == 'neutral':
            weighted_score += (tf_15m_weight / 2) * strength_15m  # Half weight for neutral
        max_score += tf_15m_weight
        
        # 5M timeframe weight (equal to 15m for forex, secondary for others)
        if tf_5m == signal_direction:
            weighted_score += tf_5m_weight * strength_5m
        elif tf_5m == 'neutral':
            weighted_score += (tf_5m_weight / 2) * strength_5m  # Half weight for neutral
        max_score += tf_5m_weight
        
        # 1M timeframe weight (lowest priority for both)
        if tf_1m == signal_direction:
            weighted_score += tf_1m_weight * strength_1m
        elif tf_1m == 'neutral':
            weighted_score += (tf_1m_weight / 2) * strength_1m  # Half weight for neutral
        max_score += tf_1m_weight
        
        # Normalize weighted score
        if max_score > 0:
            normalized_score = weighted_score / max_score
        else:
            normalized_score = 0.0
        
        # RISK SCALING BASED ON DYNAMIC WEIGHTED SCORE
        if normalized_score >= 0.8:  # Strong alignment
            self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | PASSED: LTF={ltf_signal} vs HTF={htf_5m},{htf_15m} → Score {normalized_score:.2f} (STRONG) | Weights: {weight_desc}")
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=1.2,  # Large position
                tp_multiplier=1.0,   # Full TP targets
                confidence_boost=0.2,
                reasoning=f"Strong timeframe alignment (score: {normalized_score:.2f}) - Weights: {weight_desc}",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=normalized_score,
                scenario_label="aligned_strong"
            )
        
        elif normalized_score >= 0.6:  # Good alignment
            self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | PASSED: LTF={ltf_signal} vs HTF={htf_5m},{htf_15m} → Score {normalized_score:.2f} (GOOD) | Weights: {weight_desc}")
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=1.0,  # Normal position
                tp_multiplier=1.0,   # Full TP targets
                confidence_boost=0.1,
                reasoning=f"Good timeframe alignment (score: {normalized_score:.2f}) - Weights: {weight_desc}",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=normalized_score,
                scenario_label="aligned_good"
            )
        
        elif normalized_score >= 0.4:  # Moderate alignment
            self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | PASSED: LTF={ltf_signal} vs HTF={htf_5m},{htf_15m} → Score {normalized_score:.2f} (MODERATE) | Weights: {weight_desc}")
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=0.7,  # Reduced position
                tp_multiplier=0.9,   # Slightly tighter TP
                confidence_boost=0.0,
                reasoning=f"Moderate timeframe alignment (score: {normalized_score:.2f}) - Weights: {weight_desc}",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=normalized_score,
                scenario_label="aligned_moderate"
            )
        
        elif normalized_score >= 0.2:  # Weak alignment
            self.logger.info(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | PASSED: LTF={ltf_signal} vs HTF={htf_5m},{htf_15m} → Score {normalized_score:.2f} (WEAK) | Weights: {weight_desc}")
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=0.4,  # Small position
                tp_multiplier=0.7,   # Tighter TP
                confidence_boost=0.0,
                reasoning=f"Weak timeframe alignment (score: {normalized_score:.2f}) - Weights: {weight_desc}",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=normalized_score,
                scenario_label="aligned_weak"
            )
        
        else:  # Very weak alignment or conflicting signals
            # 🚨 CRITICAL: Block trades when HTF strongly opposes LTF signal
            if normalized_score < self.block_threshold:  # Default 0.3
                self.logger.warning(f"[MTF_VALIDATOR] {tf_analysis.get('symbol', 'UNKNOWN')} | BLOCKED: LTF={ltf_signal} vs HTF={htf_5m},{htf_15m} → Score {normalized_score:.2f} < {self.block_threshold} | Weights: {weight_desc}")
                return MTFValidationResult(
                    allow_trade=False,
                    lot_multiplier=0.0,
                    tp_multiplier=0.0,
                    confidence_boost=0.0,
                    reasoning=f"BLOCKED: HTF strongly opposes LTF signal (score: {normalized_score:.2f} < {self.block_threshold}) - Weights: {weight_desc}",
                    timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                    validation_score=normalized_score,
                    scenario_label="blocked_htf_opposition"
                )
            
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=0.3,  # Minimal position
                tp_multiplier=0.6,   # Tight TP for quick exit
                confidence_boost=0.0,
                reasoning=f"Minimal timeframe alignment (score: {normalized_score:.2f}) - Weights: {weight_desc}",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=normalized_score,
                scenario_label="aligned_minimal"
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
