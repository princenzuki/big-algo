"""
Smart Take Profit System

Implements intelligent take profit logic with:
- ATR-based baseline TP
- Dynamic momentum adjustment using ADX, CCI, WaveTrend
- Partial take profits
- Trailing TP for remaining position
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SmartTPConfig:
    """Configuration for Smart Take Profit system"""
    atr_period: int = 14
    baseline_tp_multiplier: float = 2.0
    strong_trend_multiplier: float = 2.5
    weak_trend_multiplier: float = 1.0
    partial_tp_ratio: float = 0.5  # Take 50% at first TP
    partial_tp_rr: float = 1.5  # 1.5:1 R:R for partial TP
    adx_period: int = 14
    adx_strong_threshold: float = 25.0
    cci_period: int = 14
    cci_strong_threshold: float = 100.0
    cci_weak_threshold: float = -100.0

@dataclass
class SmartTPResult:
    """Result of Smart Take Profit calculation"""
    partial_tp_price: float
    full_tp_price: float
    momentum_strength: str  # 'strong', 'medium', 'weak'
    tp_multiplier: float
    should_take_partial: bool
    trailing_enabled: bool

class SmartTakeProfit:
    """Smart Take Profit system with momentum-based adjustments"""
    
    def __init__(self, config: SmartTPConfig = None):
        self.config = config or SmartTPConfig()
        self.active_positions: Dict[str, Dict] = {}  # Track partial TP status
    
    def calculate_smart_tp(self, symbol: str, entry_price: float, side: str, 
                          historical_data: List[Dict], stop_loss: float) -> SmartTPResult:
        """
        Calculate smart take profit levels based on momentum and volatility
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price of the trade
            side: 'buy' or 'sell'
            historical_data: Historical OHLC data
            stop_loss: Stop loss price
            
        Returns:
            SmartTPResult with TP levels and recommendations
        """
        if not historical_data or len(historical_data) < 20:
            # Fallback to simple ATR-based TP
            return self._fallback_tp(entry_price, side, stop_loss)
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Calculate ATR baseline
        atr = self._calculate_atr(df)
        logger.debug(f"Smart TP for {symbol}: ATR={atr:.6f}")
        
        if atr <= 0:
            logger.warning(f"⚠️ ATR calculation failed for {symbol}, using fallback TP")
            return self._fallback_tp(entry_price, side, stop_loss)
        
        # Calculate momentum indicators
        momentum_strength = self._calculate_momentum_strength(df)
        logger.debug(f"Smart TP for {symbol}: Momentum={momentum_strength}")
        
        # Determine TP multiplier based on momentum
        if momentum_strength == 'strong':
            tp_multiplier = self.config.strong_trend_multiplier
        elif momentum_strength == 'weak':
            tp_multiplier = self.config.weak_trend_multiplier
        else:
            tp_multiplier = self.config.baseline_tp_multiplier
        
        # Calculate TP distances
        tp_distance = atr * tp_multiplier
        logger.debug(f"Smart TP for {symbol}: ATR={atr:.6f}, Multiplier={tp_multiplier}, Distance={tp_distance:.6f}")
        
        # Calculate partial and full TP prices
        if side == 'buy':
            partial_tp_price = entry_price + (tp_distance * self.config.partial_tp_rr)
            full_tp_price = entry_price + tp_distance
        else:
            partial_tp_price = entry_price - (tp_distance * self.config.partial_tp_rr)
            full_tp_price = entry_price - tp_distance
        
        # HYBRID TP SYSTEM: Ensure minimum 1:1 RR
        stop_distance = abs(entry_price - stop_loss)
        min_tp_distance = stop_distance  # Minimum 1:1 RR
        
        # Check if ATR-based TP meets minimum RR requirement
        if tp_distance < min_tp_distance:
            logger.warning(f"⚠️ ATR TP distance {tp_distance:.6f} < minimum RR {min_tp_distance:.6f}, using fallback RR TP")
            tp_distance = min_tp_distance
            
            # Recalculate TP prices with minimum RR
            if side == 'buy':
                partial_tp_price = entry_price + (tp_distance * self.config.partial_tp_rr)
                full_tp_price = entry_price + tp_distance
            else:
                partial_tp_price = entry_price - (tp_distance * self.config.partial_tp_rr)
                full_tp_price = entry_price - tp_distance
            
            tp_source = "FALLBACK_RR"
        else:
            tp_source = "ATR_BASED"
        
        # Log TP decision
        actual_rr = tp_distance / stop_distance if stop_distance > 0 else 0
        logger.info(f"✅ TP chosen: {tp_source}, distance={tp_distance:.6f}, RR={actual_rr:.2f}")
        
        # Determine if we should take partial TP
        should_take_partial = momentum_strength in ['strong', 'medium']
        
        # Enable trailing for remaining position
        trailing_enabled = should_take_partial
        
        return SmartTPResult(
            partial_tp_price=partial_tp_price,
            full_tp_price=full_tp_price,
            momentum_strength=momentum_strength,
            tp_multiplier=tp_multiplier,
            should_take_partial=should_take_partial,
            trailing_enabled=trailing_enabled
        )
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            
            # Calculate ATR with proper handling of NaN values
            atr_series = tr.rolling(window=self.config.atr_period).mean()
            
            # Get the last valid ATR value
            atr = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else 0.0
            
            logger.debug(f"ATR calculation: period={self.config.atr_period}, atr={atr:.6f}")
            return atr if not pd.isna(atr) and atr > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def _calculate_momentum_strength(self, df: pd.DataFrame) -> str:
        """Calculate momentum strength using ADX and CCI"""
        try:
            # Calculate ADX
            adx = self._calculate_adx(df)
            
            # Calculate CCI
            cci = self._calculate_cci(df)
            
            # Determine momentum strength
            if adx > self.config.adx_strong_threshold and abs(cci) > self.config.cci_strong_threshold:
                return 'strong'
            elif adx < 15 or abs(cci) < 50:
                return 'weak'
            else:
                return 'medium'
                
        except Exception as e:
            logger.warning(f"⚠️ Momentum calculation failed: {e}, defaulting to medium")
            return 'medium'  # Default to medium
    
    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calculate Average Directional Index"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            
            # Calculate Directional Movement
            dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
            dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
            
            # Smooth the values
            tr_smooth = tr.rolling(window=self.config.adx_period).mean()
            dm_plus_smooth = pd.Series(dm_plus).rolling(window=self.config.adx_period).mean()
            dm_minus_smooth = pd.Series(dm_minus).rolling(window=self.config.adx_period).mean()
            
            # Calculate DI+ and DI- with division by zero protection
            di_plus = 100 * (dm_plus_smooth / tr_smooth.replace(0, np.nan))
            di_minus = 100 * (dm_minus_smooth / tr_smooth.replace(0, np.nan))
            
            # Calculate DX with division by zero protection
            denominator = di_plus + di_minus
            dx = 100 * abs(di_plus - di_minus) / denominator.replace(0, np.nan)
            
            # Calculate ADX
            adx = dx.rolling(window=self.config.adx_period).mean().iloc[-1]
            
            return adx if not pd.isna(adx) else 20.0
            
        except Exception as e:
            logger.warning(f"⚠️ ADX calculation failed: {e}, defaulting to 20.0")
            return 20.0  # Default neutral ADX
    
    def _calculate_cci(self, df: pd.DataFrame) -> float:
        """Calculate Commodity Channel Index"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate Typical Price
            tp = (high + low + close) / 3
            
            # Calculate Simple Moving Average of TP
            sma_tp = tp.rolling(window=self.config.cci_period).mean()
            
            # Calculate Mean Deviation
            mean_dev = tp.rolling(window=self.config.cci_period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            # Calculate CCI
            cci = (tp - sma_tp) / (0.015 * mean_dev)
            
            return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ CCI calculation failed: {e}, defaulting to 0.0")
            return 0.0  # Default neutral CCI
    
    def _fallback_tp(self, entry_price: float, side: str, stop_loss: float) -> SmartTPResult:
        """Fallback to simple TP calculation"""
        # Simple 2:1 R:R ratio
        stop_distance = abs(entry_price - stop_loss)
        if side == 'buy':
            tp_price = entry_price + (2 * stop_distance)
        else:
            tp_price = entry_price - (2 * stop_distance)
        
        logger.warning(f"⚠️ Using fallback TP: {tp_price:.5f} (2:1 R:R, distance: {2 * stop_distance:.5f})")
        
        return SmartTPResult(
            partial_tp_price=tp_price,
            full_tp_price=tp_price,
            momentum_strength='medium',
            tp_multiplier=2.0,
            should_take_partial=False,
            trailing_enabled=False
        )
    
    def update_trailing_tp(self, symbol: str, current_price: float, 
                          historical_data: List[Dict]) -> Optional[float]:
        """
        Update trailing take profit for remaining position
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            historical_data: Historical OHLC data
            
        Returns:
            New trailing TP price or None if no update needed
        """
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        if not position.get('trailing_enabled', False):
            return None
        
        # Calculate current ATR
        if not historical_data or len(historical_data) < 14:
            return None
        
        df = pd.DataFrame(historical_data)
        atr = self._calculate_atr(df)
        
        if atr <= 0:
            return None
        
        # Calculate new trailing TP
        if position['side'] == 'buy':
            new_tp = current_price - (atr * 0.5)  # Trail at 0.5 ATR
            # Only update if new TP is better (higher)
            if new_tp > position.get('current_tp', 0):
                return new_tp
        else:
            new_tp = current_price + (atr * 0.5)  # Trail at 0.5 ATR
            # Only update if new TP is better (lower)
            if new_tp < position.get('current_tp', float('inf')):
                return new_tp
        
        return None
    
    def register_position(self, symbol: str, side: str, entry_price: float, 
                         tp_result: SmartTPResult):
        """Register a new position for tracking"""
        self.active_positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'partial_tp_price': tp_result.partial_tp_price,
            'full_tp_price': tp_result.full_tp_price,
            'current_tp': tp_result.full_tp_price,
            'trailing_enabled': tp_result.trailing_enabled,
            'partial_taken': False,
            'momentum_strength': tp_result.momentum_strength
        }
    
    def check_partial_tp(self, symbol: str, current_price: float) -> bool:
        """Check if partial TP should be taken"""
        if symbol not in self.active_positions:
            return False
        
        position = self.active_positions[symbol]
        if position.get('partial_taken', False):
            return False
        
        if position['side'] == 'buy':
            return current_price >= position['partial_tp_price']
        else:
            return current_price <= position['partial_tp_price']
    
    def take_partial_tp(self, symbol: str):
        """Mark partial TP as taken"""
        if symbol in self.active_positions:
            self.active_positions[symbol]['partial_taken'] = True
            logger.info(f"[SMART_TP] Partial TP taken for {symbol}")
    
    def close_position(self, symbol: str):
        """Remove position from tracking"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info(f"[SMART_TP] Position closed for {symbol}")
