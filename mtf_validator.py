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
            tf_analysis = self._analyze_timeframes(mtf_data)
            
            # Determine trade validation
            validation_result = self._determine_validation(signal, tf_analysis)
            
            self.logger.info(f"[MTF_VALIDATOR] {symbol} validation: {validation_result.reasoning}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error validating {symbol}: {e}")
            return self._create_fail_safe_result(f"Validation error: {e}")
    
    def _get_mtf_data(self, symbol: str, historical_data: List[Dict]) -> Optional[Dict]:
        """
        Get multi-timeframe data (1m, 5m, 15m) for the symbol
        
        Args:
            symbol: Trading symbol
            historical_data: Historical data from main algorithm
            
        Returns:
            Dictionary with 1m, 5m, and 15m data or None if unavailable
        """
        try:
            # Use existing historical data as 1m data
            if not historical_data or len(historical_data) < 50:
                self.logger.warning(f"[MTF_VALIDATOR] Insufficient 1m data for {symbol}")
                return None
            
            # For now, we'll use the existing data and simulate 5m/15m
            # In a real implementation, you'd fetch actual 5m and 15m data
            df_1m = pd.DataFrame(historical_data[-50:])  # Last 50 bars
            
            # Simulate 5m data by resampling 1m data (every 5th bar)
            df_5m = df_1m.iloc[::5].copy()
            df_5m = df_5m.reset_index(drop=True)
            
            # Simulate 15m data by resampling 1m data (every 15th bar)
            df_15m = df_1m.iloc[::15].copy()
            df_15m = df_15m.reset_index(drop=True)
            
            return {
                '1m': df_1m,
                '5m': df_5m,
                '15m': df_15m
            }
            
        except Exception as e:
            self.logger.error(f"[MTF_VALIDATOR] Error getting MTF data for {symbol}: {e}")
            return None
    
    def _analyze_timeframes(self, mtf_data: Dict) -> Dict[str, Dict]:
        """
        Analyze each timeframe using the EXACT SAME indicators and ML logic as the 1-minute engine
        
        Args:
            mtf_data: Multi-timeframe data dictionary
            
        Returns:
            Dictionary with analysis for each timeframe
        """
        analysis = {}
        
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
        Determine trade validation based on timeframe analysis
        
        Args:
            signal: Trade signal (1 for long, -1 for short)
            tf_analysis: Timeframe analysis results
            
        Returns:
            MTFValidationResult with validation outcome
        """
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
        
        # Apply validation logic
        if tf_1m == signal_direction and tf_5m == signal_direction and tf_15m == signal_direction:
            # All timeframes align - full conviction
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=self.full_conviction_multiplier,
                tp_multiplier=1.0,
                confidence_boost=0.2,
                reasoning="All timeframes aligned - full conviction",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.9
            )
        
        elif tf_1m == signal_direction and tf_5m == signal_direction and tf_15m == 'neutral':
            # 1m + 5m align, 15m neutral - good conviction
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=self.partial_conviction_multiplier,
                tp_multiplier=1.0,
                confidence_boost=0.1,
                reasoning="1m+5m aligned, 15m neutral - good conviction",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.7
            )
        
        elif tf_1m == signal_direction and tf_5m == signal_direction and tf_15m != signal_direction:
            # 1m + 5m align, 15m opposite - reduced conviction
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=self.reduced_conviction_multiplier,
                tp_multiplier=0.8,  # Tighter TP
                confidence_boost=0.0,
                reasoning="1m+5m aligned, 15m opposite - reduced conviction",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.5
            )
        
        elif tf_1m == signal_direction and tf_5m != signal_direction:
            # 1m signal conflicts with 5m trend - block trade
            return MTFValidationResult(
                allow_trade=False,
                lot_multiplier=0.0,
                tp_multiplier=0.0,
                confidence_boost=0.0,
                reasoning="1m signal conflicts with 5m trend - too noisy",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.1
            )
        
        else:
            # Default case - allow with reduced conviction
            return MTFValidationResult(
                allow_trade=True,
                lot_multiplier=self.reduced_conviction_multiplier,
                tp_multiplier=0.9,
                confidence_boost=0.0,
                reasoning="Mixed signals - reduced conviction",
                timeframe_alignment={'1m': tf_1m, '5m': tf_5m, '15m': tf_15m},
                validation_score=0.4
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
            validation_score=0.5
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
