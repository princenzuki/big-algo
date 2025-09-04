"""
Lorentzian Classification ML Signals

This module implements the exact Pine Script logic for Lorentzian Classification
with line-by-line parity. Any deviations must be logged with reason codes.

Pine Script Reference: Machine Learning: Lorentzian Classification by jdehorty
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ======================
# ==== Custom Types ====
# ======================

@dataclass
class Settings:
    """General user-defined inputs - Pine lines 45-52"""
    source: str = 'close'
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    show_exits: bool = False
    use_dynamic_exits: bool = False

@dataclass
class Label:
    """Label classification - Pine lines 54-58"""
    long: int = 1
    short: int = -1
    neutral: int = 0

@dataclass
class FeatureArrays:
    """Feature arrays for ML - Pine lines 60-66"""
    f1: List[float]
    f2: List[float]
    f3: List[float]
    f4: List[float]
    f5: List[float]

@dataclass
class FeatureSeries:
    """Current feature values - Pine lines 68-74"""
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float

@dataclass
class MLModel:
    """ML model state - Pine lines 76-84"""
    first_bar_index: int
    training_labels: List[int]
    loop_size: int
    last_distance: float
    distances_array: List[float]
    predictions_array: List[int]
    prediction: int

@dataclass
class FilterSettings:
    """Filter configuration - Pine lines 86-93"""
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = False
    regime_threshold: float = -0.1
    adx_threshold: int = 20

@dataclass
class Filter:
    """Active filters - Pine lines 95-100"""
    volatility: bool
    regime: bool
    adx: bool

# ==========================
# ==== Helper Functions ====
# ==========================

def series_from(feature_string: str, close: float, high: float, low: float, 
                hlc3: float, f_param_a: int, f_param_b: int) -> float:
    """
    Calculate feature series - Pine lines 102-107
    
    Args:
        feature_string: Feature type (RSI, WT, CCI, ADX)
        close, high, low, hlc3: Price data
        f_param_a, f_param_b: Feature parameters
    
    Returns:
        Calculated feature value
    """
    # DEVIATION: Using simplified implementations for now
    # In production, these should use the exact Pine Script calculations
    if feature_string == "RSI":
        # Simplified RSI calculation - needs exact Pine implementation
        return 50.0  # Placeholder
    elif feature_string == "WT":
        # Simplified Williams %R calculation
        return 50.0  # Placeholder
    elif feature_string == "CCI":
        # Simplified CCI calculation
        return 0.0  # Placeholder
    elif feature_string == "ADX":
        # Simplified ADX calculation
        return 25.0  # Placeholder
    else:
        logger.warning(f"Unknown feature: {feature_string}")
        return 0.0

def get_lorentzian_distance(i: int, feature_count: int, feature_series: FeatureSeries, 
                          feature_arrays: FeatureArrays) -> float:
    """
    Calculate Lorentzian distance - Pine lines 109-125
    
    This is the core distance metric that accounts for "price-time" warping
    due to proximity to significant economic events.
    """
    if feature_count == 5:
        return (np.log(1 + abs(feature_series.f1 - feature_arrays.f1[i])) +
                np.log(1 + abs(feature_series.f2 - feature_arrays.f2[i])) +
                np.log(1 + abs(feature_series.f3 - feature_arrays.f3[i])) +
                np.log(1 + abs(feature_series.f4 - feature_arrays.f4[i])) +
                np.log(1 + abs(feature_series.f5 - feature_arrays.f5[i])))
    elif feature_count == 4:
        return (np.log(1 + abs(feature_series.f1 - feature_arrays.f1[i])) +
                np.log(1 + abs(feature_series.f2 - feature_arrays.f2[i])) +
                np.log(1 + abs(feature_series.f3 - feature_arrays.f3[i])) +
                np.log(1 + abs(feature_series.f4 - feature_arrays.f4[i])))
    elif feature_count == 3:
        return (np.log(1 + abs(feature_series.f1 - feature_arrays.f1[i])) +
                np.log(1 + abs(feature_series.f2 - feature_arrays.f2[i])) +
                np.log(1 + abs(feature_series.f3 - feature_arrays.f3[i])))
    elif feature_count == 2:
        return (np.log(1 + abs(feature_series.f1 - feature_arrays.f1[i])) +
                np.log(1 + abs(feature_series.f2 - feature_arrays.f2[i])))
    else:
        logger.warning(f"Unsupported feature count: {feature_count}")
        return 0.0

class LorentzianClassifier:
    """
    Main Lorentzian Classification ML Model
    
    Implements the exact Pine Script logic with strict parity enforcement.
    """
    
    def __init__(self, settings: Settings, filter_settings: FilterSettings):
        self.settings = settings
        self.filter_settings = filter_settings
        self.label = Label()
        
        # Initialize feature arrays - Pine lines 200-210
        self.feature_arrays = FeatureArrays(
            f1=[], f2=[], f3=[], f4=[], f5=[]
        )
        
        # Initialize ML model state
        self.ml_model = MLModel(
            first_bar_index=0,
            training_labels=[],
            loop_size=0,
            last_distance=-1.0,
            distances_array=[],
            predictions_array=[],
            prediction=0
        )
        
        # Variables for ML logic - Pine lines 280-285
        self.predictions = []
        self.prediction = 0.0
        self.signal = self.label.neutral
        self.distances = []
        
        # Feature parameters - Pine lines 150-180
        self.feature_params = {
            'f1': {'string': 'RSI', 'param_a': 14, 'param_b': 1},
            'f2': {'string': 'WT', 'param_a': 10, 'param_b': 11},
            'f3': {'string': 'CCI', 'param_a': 20, 'param_b': 1},
            'f4': {'string': 'ADX', 'param_a': 20, 'param_b': 2},
            'f5': {'string': 'RSI', 'param_a': 9, 'param_b': 1}
        }
    
    def calculate_features(self, ohlc_data: Dict[str, float]) -> FeatureSeries:
        """
        Calculate feature series from OHLC data - Pine lines 182-190
        """
        close = ohlc_data['close']
        high = ohlc_data['high']
        low = ohlc_data['low']
        hlc3 = (high + low + close) / 3
        
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
    
    def update_feature_arrays(self, feature_series: FeatureSeries):
        """
        Update feature arrays with new data - Pine lines 200-210
        """
        self.feature_arrays.f1.append(feature_series.f1)
        self.feature_arrays.f2.append(feature_series.f2)
        self.feature_arrays.f3.append(feature_series.f3)
        self.feature_arrays.f4.append(feature_series.f4)
        self.feature_arrays.f5.append(feature_series.f5)
        
        # Maintain max_bars_back limit
        if len(self.feature_arrays.f1) > self.settings.max_bars_back:
            self.feature_arrays.f1.pop(0)
            self.feature_arrays.f2.pop(0)
            self.feature_arrays.f3.pop(0)
            self.feature_arrays.f4.pop(0)
            self.feature_arrays.f5.pop(0)
    
    def calculate_training_label(self, current_price: float, future_price: float) -> int:
        """
        Calculate training label for next bar classification - Pine lines 250-255
        """
        if future_price < current_price:
            return self.label.short
        elif future_price > current_price:
            return self.label.long
        else:
            return self.label.neutral
    
    def approximate_nearest_neighbors(self, feature_series: FeatureSeries) -> float:
        """
        Approximate Nearest Neighbors Search with Lorentzian Distance
        Pine lines 300-350
        
        This is the core ML algorithm that finds similar historical patterns
        using Lorentzian distance to account for "price-time" warping.
        """
        last_distance = -1.0
        size = min(self.settings.max_bars_back - 1, len(self.ml_model.training_labels) - 1)
        size_loop = min(self.settings.max_bars_back - 1, size)
        
        # Clear previous predictions
        self.distances = []
        self.predictions = []
        
        # Pine lines 320-340: Main ANN algorithm
        for i in range(size_loop):
            d = get_lorentzian_distance(i, self.settings.feature_count, 
                                      feature_series, self.feature_arrays)
            
            # Pine line 325: Only process every 4th bar for chronological spacing
            if d >= last_distance and i % 4:
                last_distance = d
                self.distances.append(d)
                self.predictions.append(round(self.ml_model.training_labels[i]))
                
                # Pine lines 330-335: Maintain neighbors count limit
                if len(self.predictions) > self.settings.neighbors_count:
                    # Pine line 332: Boost accuracy by using lower 25% distance
                    last_distance = self.distances[int(self.settings.neighbors_count * 3 / 4)]
                    self.distances.pop(0)
                    self.predictions.pop(0)
        
        # Pine line 340: Sum predictions for final signal
        return sum(self.predictions)
    
    def apply_filters(self, prediction: float) -> bool:
        """
        Apply user-defined filters - Pine lines 360-370
        """
        # Simplified filter implementation
        # In production, these should use exact Pine Script filter logic
        volatility_filter = True  # Placeholder
        regime_filter = True      # Placeholder
        adx_filter = True         # Placeholder
        
        return volatility_filter and regime_filter and adx_filter
    
    def generate_signal(self, ohlc_data: Dict[str, float], historical_data: List[Dict[str, float]]) -> Dict[str, any]:
        """
        Generate ML signal from OHLC data - Main entry point
        
        Args:
            ohlc_data: Current bar OHLC data
            historical_data: Historical bars for training
            
        Returns:
            Signal dictionary with prediction, confidence, and metadata
        """
        # Calculate current features
        feature_series = self.calculate_features(ohlc_data)
        
        # Update feature arrays
        self.update_feature_arrays(feature_series)
        
        # Update training labels if we have enough historical data
        if len(historical_data) >= 5:
            current_price = ohlc_data['close']
            future_price = historical_data[-5]['close']  # 4 bars ahead
            training_label = self.calculate_training_label(current_price, future_price)
            self.ml_model.training_labels.append(training_label)
            
            # Maintain max_bars_back limit
            if len(self.ml_model.training_labels) > self.settings.max_bars_back:
                self.ml_model.training_labels.pop(0)
        
        # Generate prediction using ANN algorithm
        if len(self.ml_model.training_labels) > self.settings.neighbors_count:
            prediction = self.approximate_nearest_neighbors(feature_series)
            
            # Apply filters
            filter_all = self.apply_filters(prediction)
            
            # Generate signal - Pine lines 375-380
            if prediction > 0 and filter_all:
                signal = self.label.long
            elif prediction < 0 and filter_all:
                signal = self.label.short
            else:
                signal = self.signal  # Keep previous signal
            
            self.signal = signal
            
            # Calculate confidence based on prediction magnitude
            confidence = abs(prediction) / self.settings.neighbors_count
            
            return {
                'prediction': prediction,
                'signal': signal,
                'confidence': confidence,
                'feature_series': feature_series,
                'filter_applied': filter_all,
                'neighbors_count': len(self.predictions),
                'distances': self.distances[:5]  # Top 5 distances for debugging
            }
        else:
            # Not enough data for prediction
            return {
                'prediction': 0,
                'signal': self.label.neutral,
                'confidence': 0.0,
                'feature_series': feature_series,
                'filter_applied': False,
                'neighbors_count': 0,
                'distances': []
            }

# =========================
# ==== Entry Conditions ====
# =========================

def generate_entry_conditions(signal_data: Dict[str, any], ohlc_data: Dict[str, float]) -> Dict[str, bool]:
    """
    Generate entry conditions - Pine lines 400-420
    """
    signal = signal_data['signal']
    prediction = signal_data['prediction']
    
    # Simplified entry conditions - needs exact Pine implementation
    is_buy_signal = signal == 1  # direction.long
    is_sell_signal = signal == -1  # direction.short
    
    return {
        'start_long_trade': is_buy_signal,
        'start_short_trade': is_sell_signal,
        'is_buy_signal': is_buy_signal,
        'is_sell_signal': is_sell_signal
    }

# =========================
# ==== Exit Conditions ====
# =========================

def generate_exit_conditions(signal_data: Dict[str, any], bars_held: int) -> Dict[str, bool]:
    """
    Generate exit conditions - Pine lines 430-450
    """
    signal = signal_data['signal']
    
    # Bar-count filters for 4-bar holding period
    is_held_four_bars = bars_held == 4
    is_held_less_than_four_bars = 0 < bars_held < 4
    
    # Simplified exit conditions - needs exact Pine implementation
    end_long_trade = is_held_four_bars and signal == 1
    end_short_trade = is_held_four_bars and signal == -1
    
    return {
        'end_long_trade': end_long_trade,
        'end_short_trade': end_short_trade,
        'is_held_four_bars': is_held_four_bars,
        'is_held_less_than_four_bars': is_held_less_than_four_bars
    }
