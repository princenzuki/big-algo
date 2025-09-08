"""
Risk Management Module

Implements strict risk controls including:
- 10% account risk cap
- Position sizing based on confidence
- Cooldown periods
- One trade per symbol
- Spread-aware stop loss placement
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ===============================
# Symbol-Aware Pip Value System
# ===============================

def get_pip_value(symbol: str) -> float:
    """
    Get the pip multiplier for a given trading symbol.
    
    This function returns the correct pip value based on the instrument type:
    - 0.0001 for standard forex pairs (EURUSD, GBPUSD, etc.)
    - 0.01 for JPY pairs (USDJPY, EURJPY, etc.)
    - 0.01 for gold (XAUUSD) and oil (USOIL, UKOIL, etc.)
    - 1.0 for Bitcoin and other cryptos (BTCUSD, BTCUSDm, etc.)
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD', 'BTCUSDm')
        
    Returns:
        float: Pip multiplier for the symbol
        
    Examples:
        >>> get_pip_value('EURUSD')
        0.0001
        >>> get_pip_value('USDJPY')
        0.01
        >>> get_pip_value('XAUUSD')
        0.01
        >>> get_pip_value('BTCUSDm')
        1.0
    """
    symbol_upper = symbol.upper()
    
    # Bitcoin and other cryptocurrencies
    if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'DOT']):
        return 1.0
    
    # Gold and precious metals
    if any(metal in symbol_upper for metal in ['XAU', 'GOLD']):
        return 0.01
    
    # Oil and energy commodities
    if any(oil in symbol_upper for oil in ['OIL', 'USOIL', 'UKOIL', 'BRENT', 'WTI']):
        return 0.01
    
    # Stock indices (if they use different pip values)
    if any(index in symbol_upper for index in ['US30', 'US500', 'USTEC', 'NAS100', 'SPX500']):
        return 0.01  # Most indices use 0.01
    
    # JPY pairs - special case for forex
    if 'JPY' in symbol_upper:
        return 0.01
    
    # Default to standard forex pip value (0.0001)
    # This covers EURUSD, GBPUSD, AUDUSD, USDCAD, etc.
    return 0.0001

def get_pip_distance_in_price(symbol: str, distance_pips: float) -> float:
    """
    Convert pip distance to actual price distance for a symbol.
    
    Args:
        symbol: Trading symbol
        distance_pips: Distance in pips
        
    Returns:
        float: Price distance
    """
    pip_value = get_pip_value(symbol)
    return distance_pips * pip_value

def get_price_distance_in_pips(symbol: str, price_distance: float) -> float:
    """
    Convert price distance to pip distance for a symbol.
    
    Args:
        symbol: Trading symbol
        price_distance: Price distance
        
    Returns:
        float: Distance in pips
    """
    pip_value = get_pip_value(symbol)
    return price_distance / pip_value if pip_value > 0 else 0.0

# ===============================
# Dynamic Spread & Smart Stop Logic
# ===============================

def get_dynamic_spread(symbol_data: List[Dict], base_min_spread: float, spread_multiplier: float = 1.5) -> float:
    """
    Calculate intelligent spread per symbol based on recent market conditions.
    
    Args:
        symbol_data: List of dicts with 'high', 'low', 'open', 'close' columns
        base_min_spread: float, minimum allowed spread in pips (0.0 means no base)
        spread_multiplier: float, multiplies average spread for breathing room

    Returns:
        max_allowed_spread: float
    """
    if not symbol_data or len(symbol_data) < 20:
        # If no base spread specified, return a minimal fallback
        return base_min_spread if base_min_spread > 0 else 1.0
    
    # Convert to DataFrame for easier calculation
    df = pd.DataFrame(symbol_data)
    
    # Estimate spread from high-low range (proxy for volatility)
    # This is a reasonable approximation since we don't have actual bid/ask data
    df['estimated_spread'] = (df['high'] - df['low']) * 10000  # Convert to pips
    
    # Calculate rolling average of last 20 bars
    recent_avg = df['estimated_spread'].rolling(window=20).mean().iloc[-1]
    
    # Calculate dynamic spread
    dynamic_spread = recent_avg * spread_multiplier
    
    # If no base spread specified, use only dynamic spread
    if base_min_spread <= 0:
        max_allowed_spread = dynamic_spread
    else:
        # Return max of base minimum or calculated dynamic spread
        max_allowed_spread = max(base_min_spread, dynamic_spread)
    
    logger.debug(f"Dynamic spread calculation: base={base_min_spread}, recent_avg={recent_avg:.2f}, multiplier={spread_multiplier}, result={max_allowed_spread:.2f}")
    
    return max_allowed_spread


def get_intelligent_sl(symbol_data: List[Dict], atr_multiplier: float = 2.0) -> float:
    """
    Place smart stops using ATR-based volatility.
    
    Args:
        symbol_data: List of dicts with 'high', 'low', 'close' columns
        atr_multiplier: float, how many ATRs to use for stop

    Returns:
        atr_sl: float, stop distance in pips
    """
    if not symbol_data or len(symbol_data) < 14:
        return 20.0  # Default fallback
    
    # Convert to DataFrame for easier calculation
    df = pd.DataFrame(symbol_data)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    
    # Calculate 14-period ATR
    atr = tr.rolling(window=14).mean().iloc[-1]
    
    # Convert to pips (assuming 4-digit forex pairs)
    atr_sl = atr * atr_multiplier * 10000
    
    logger.debug(f"ATR-based SL calculation: ATR={atr:.5f}, multiplier={atr_multiplier}, result={atr_sl:.2f} pips")
    
    return atr_sl


def calculate_hybrid_stop_loss(symbol: str, entry_price: float, side: str, 
                              historical_data: List[Dict], spread_pips: float,
                              atr_multiplier: float = 1.5, min_distance_pips: float = 10.0) -> Tuple[float, str]:
    """
    Calculate hybrid ATR-based stop loss with spread buffer and minimum distance.
    
    Args:
        symbol: Trading symbol
        entry_price: Entry price of the trade
        side: 'buy' or 'sell'
        historical_data: Historical OHLC data for ATR calculation
        spread_pips: Current spread in pips
        atr_multiplier: ATR multiplier for stop distance
        min_distance_pips: Minimum stop distance in pips
        
    Returns:
        Tuple of (stop_loss_price, calculation_method)
    """
    logger.info(f"[HYBRID_SL] Calculating hybrid SL for {symbol} {side} @ {entry_price:.5f}")
    
    # Step 1: Calculate ATR-based stop distance
    atr_distance_pips = 0.0
    atr_calculation_success = False
    
    if historical_data and len(historical_data) >= 14:
        try:
            df = pd.DataFrame(historical_data)
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            
            # Calculate 14-period ATR
            atr_series = tr.rolling(window=14).mean()
            atr = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else 0.0
            
            if not pd.isna(atr) and atr > 0:
                # Convert ATR to pips using symbol-aware pip value
                pip_value = get_pip_value(symbol)
                atr_distance_pips = (atr * atr_multiplier) / pip_value
                
                atr_calculation_success = True
                logger.info(f"[HYBRID_SL] ATR calculation: ATR={atr:.6f}, multiplier={atr_multiplier}, distance={atr_distance_pips:.2f} pips")
            else:
                logger.warning(f"[HYBRID_SL] ATR calculation failed: invalid ATR value {atr}")
        except Exception as e:
            logger.error(f"[HYBRID_SL] ATR calculation error: {e}")
    else:
        logger.warning(f"[HYBRID_SL] Insufficient data for ATR: {len(historical_data) if historical_data else 0} bars")
    
    # Step 2: Add spread buffer
    spread_buffer_pips = spread_pips + 5.0  # Spread + 5 pips buffer
    logger.info(f"[HYBRID_SL] Spread buffer: {spread_pips:.1f} + 5.0 = {spread_buffer_pips:.1f} pips")
    
    # Step 3: Determine final stop distance
    if atr_calculation_success:
        # Use ATR-based distance + spread buffer
        final_distance_pips = atr_distance_pips + spread_buffer_pips
        calculation_method = "ATR_BASED"
        logger.info(f"[HYBRID_SL] ATR-based distance: {atr_distance_pips:.1f} + spread buffer: {spread_buffer_pips:.1f} = {final_distance_pips:.1f} pips")
    else:
        # Fallback to static distance + spread buffer
        final_distance_pips = min_distance_pips + spread_buffer_pips
        calculation_method = "FALLBACK_STATIC"
        logger.warning(f"[HYBRID_SL] ATR failed, using fallback: {min_distance_pips:.1f} + spread buffer: {spread_buffer_pips:.1f} = {final_distance_pips:.1f} pips")
    
    # Step 4: Enforce minimum distance
    if final_distance_pips < min_distance_pips:
        final_distance_pips = min_distance_pips
        calculation_method = "MINIMUM_ENFORCED"
        logger.warning(f"[HYBRID_SL] Enforcing minimum distance: {min_distance_pips:.1f} pips")
    
    # Step 5: Convert pips to price distance using symbol-aware pip value
    pip_value = get_pip_value(symbol)
    price_distance = get_pip_distance_in_price(symbol, final_distance_pips)
    logger.info(f"[HYBRID_SL] {symbol}: {final_distance_pips:.1f} pips × {pip_value:.5f} = {price_distance:.5f}")
    
    # Step 6: Calculate stop loss price
    if side == 'buy':
        stop_loss_price = entry_price - price_distance
    else:
        stop_loss_price = entry_price + price_distance
    
    # Step 7: Safety validation
    if side == 'buy' and stop_loss_price >= entry_price:
        logger.error(f"[HYBRID_SL] ❌ Invalid SL for buy: {stop_loss_price:.5f} >= {entry_price:.5f}")
        # Force correct SL
        stop_loss_price = entry_price - price_distance
    elif side == 'sell' and stop_loss_price <= entry_price:
        logger.error(f"[HYBRID_SL] ❌ Invalid SL for sell: {stop_loss_price:.5f} <= {entry_price:.5f}")
        # Force correct SL
        stop_loss_price = entry_price + price_distance
    
    logger.info(f"[HYBRID_SL] [OK] Final SL: {stop_loss_price:.5f} (method: {calculation_method}, distance: {final_distance_pips:.1f} pips, pip_mult: {pip_value:.5f})")
    
    return stop_loss_price, calculation_method


@dataclass
class RiskSettings:
    """Risk management configuration"""
    max_risk_per_trade: float = 0.05  # 5% per trade (confidence-based: 1% to 5%)
    max_total_risk: float = 0.10  # 10% total account risk (unlimited trades per day)
    min_lot_size: float = 0.01
    max_concurrent_trades: int = 5
    cooldown_minutes: int = 10
    max_spread_pips: float = 3.0
    min_stop_distance_pips: float = 5.0
    one_trade_per_symbol: bool = True
    # Legacy field for backward compatibility
    max_account_risk_percent: float = 10.0  # 10% max risk

@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    lot_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_amount: float
    opened_at: datetime
    close_time: Optional[datetime] = None
    status: str = 'open'  # 'open', 'closed', 'stopped'
    # Trailing stop fields
    original_stop_loss: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_distance: float = 0.0
    trailing_enabled: bool = False
    best_price: float = 0.0
    atr_distance: float = 0.0
    break_even_triggered: bool = False
    
    # 🚀 ENHANCED SL/TP MANAGEMENT FIELDS
    # Multi-tier SL levels
    micro_trailing_enabled: bool = False  # For noise protection
    main_trailing_enabled: bool = False   # For risk control
    micro_trailing_distance: float = 0.0  # Smaller distance for noise
    main_trailing_distance: float = 0.0   # Main ATR distance
    last_micro_update: Optional[datetime] = None
    last_main_update: Optional[datetime] = None
    
    # Partial TP management
    partial_tp_taken: bool = False
    partial_tp_percentage: float = 0.5  # 50% at baseline
    partial_tp_price: float = 0.0
    remaining_lot_size: float = 0.0
    trailing_tp_enabled: bool = False
    trailing_tp_distance: float = 0.0
    best_tp_price: float = 0.0
    last_tp_update: Optional[datetime] = None
    
    # Trend-aware adjustments
    htf_support_level: float = 0.0  # 5m/15m support level
    htf_resistance_level: float = 0.0  # 5m/15m resistance level
    trend_strength: float = 0.0  # Combined HTF trend strength


def calculate_trailing_stop(position: Position, current_price: float, current_atr: float, 
                          historical_data: List[Dict]) -> Tuple[float, bool, str]:
    """
    Calculate intelligent trailing stop based on ATR and market conditions.
    
    Args:
        position: Current position object
        current_price: Current market price
        current_atr: Current ATR value
        historical_data: Historical data for ATR calculation
        
    Returns:
        Tuple of (new_stop_loss, stop_updated, log_message)
    """
    if position.status != 'open':
        return position.stop_loss, False, "Position not open"
    
    # Initialize trailing stop if not enabled yet
    if not position.trailing_enabled:
        # Check if trade moved in favor by at least 0.5 × ATR
        if position.side == 'buy':
            profit_distance = current_price - position.entry_price
            trigger_distance = current_atr * 0.5
            if profit_distance >= trigger_distance:
                position.trailing_enabled = True
                position.original_stop_loss = position.stop_loss
                position.best_price = current_price
                position.atr_distance = current_atr
                new_stop = current_price - current_atr
                return new_stop, True, f"Trailing enabled: {profit_distance:.5f} >= {trigger_distance:.5f}"
        else:  # sell
            profit_distance = position.entry_price - current_price
            trigger_distance = current_atr * 0.5
            if profit_distance >= trigger_distance:
                position.trailing_enabled = True
                position.original_stop_loss = position.stop_loss
                position.best_price = current_price
                position.atr_distance = current_atr
                new_stop = current_price + current_atr
                return new_stop, True, f"Trailing enabled: {profit_distance:.5f} >= {trigger_distance:.5f}"
        
        return position.stop_loss, False, "Trailing not triggered yet"
    
    # Trailing stop is enabled, update it
    updated = False
    log_message = ""
    
    if position.side == 'buy':
        # Update best price if current price is higher
        if current_price > position.best_price:
            position.best_price = current_price
            
            # Calculate new trailing stop
            new_stop = current_price - current_atr
            
            # Don't move below break-even
            break_even_stop = position.entry_price
            new_stop = max(new_stop, break_even_stop)
            
            # Only update if new stop is better (higher)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                position.atr_distance = current_atr
                updated = True
                log_message = f"Trailing updated: {new_stop:.5f} (BE: {break_even_stop:.5f})"
    
    else:  # sell
        # Update best price if current price is lower
        if current_price < position.best_price:
            position.best_price = current_price
            
            # Calculate new trailing stop
            new_stop = current_price + current_atr
            
            # Don't move above break-even
            break_even_stop = position.entry_price
            new_stop = min(new_stop, break_even_stop)
            
            # Only update if new stop is better (lower)
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
                position.atr_distance = current_atr
                updated = True
                log_message = f"Trailing updated: {new_stop:.5f} (BE: {break_even_stop:.5f})"
    
    return position.stop_loss, updated, log_message


@dataclass
class AccountInfo:
    """Account information for risk calculations"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str = 'USD'
    leverage: int = 100
    server: str = 'TestServer'
    margin_level: float = 1000.0

class RiskManager:
    """
    Central risk management system
    
    Enforces all risk rules and position sizing logic.
    """
    
    def __init__(self, settings: RiskSettings):
        self.settings = settings
        self.positions: Dict[str, Position] = {}  # symbol -> position
        self.cooldowns: Dict[str, datetime] = {}  # symbol -> cooldown_end_time
        self.account_info: Optional[AccountInfo] = None
        self._position_locks: Dict[str, bool] = {}  # symbol -> is_locked (prevents race conditions)
        self._starting_balance: Optional[float] = None  # Track starting balance for drawdown calculation
        self._max_equity_peak: Optional[float] = None  # Track highest equity reached
        self._last_trade_close_time: Optional[datetime] = None  # Track last trade close for global cooldown
        
    def update_account_info(self, account_info: AccountInfo):
        """Update account information for risk calculations"""
        self.account_info = account_info
        
        # Initialize starting balance and equity peak tracking
        if self._starting_balance is None:
            self._starting_balance = account_info.balance
            self._max_equity_peak = account_info.equity
            logger.info(f"🎯 DRAWDOWN TRACKING INITIALIZED: Starting Balance=${self._starting_balance:.2f}, Starting Equity=${self._max_equity_peak:.2f}")
        
        # Update equity peak if current equity is higher
        if self._max_equity_peak is None or account_info.equity > self._max_equity_peak:
            self._max_equity_peak = account_info.equity
        
        # Calculate current drawdown
        current_drawdown = self._calculate_actual_drawdown()
        
        logger.info(f"Account updated: Balance={account_info.balance:.2f}, "
                   f"Equity={account_info.equity:.2f}, Free Margin={account_info.free_margin:.2f}, "
                   f"Drawdown={current_drawdown:.2f}%")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, confidence: float) -> Tuple[float, float]:
        """
        Calculate position size based on confidence and risk rules
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: ML confidence (0-1)
            
        Returns:
            Tuple of (lot_size, risk_amount)
        """
        if not self.account_info:
            logger.error("Account info not available for position sizing")
            return 0.0, 0.0
        
        # Calculate risk per trade based on confidence
        # Base risk: 2% per trade
        # Confidence scaling: 1% to 5% based on confidence (0.0 to 1.0)
        # Formula: risk = 1% + (confidence * 4%) = 1% to 5% range
        base_risk_percent = 1.0  # Minimum 1% risk
        confidence_boost = confidence * 4.0  # Up to 4% additional risk for high confidence
        risk_percent = base_risk_percent + confidence_boost  # 1% to 5% range
        
        # Calculate risk amount in account currency
        risk_amount = self.account_info.equity * (risk_percent / 100.0)
        
        # Calculate pip value and stop distance using symbol-aware system
        pip_value = get_pip_value(symbol)
        stop_distance_pips = get_price_distance_in_pips(symbol, abs(entry_price - stop_loss))
        
        if stop_distance_pips < self.settings.min_stop_distance_pips:
            logger.warning(f"Stop distance too small: {stop_distance_pips:.1f} pips")
            return 0.0, 0.0
        
        # Calculate lot size
        # For standard lots: 1 lot = 100,000 units
        # Risk per pip = lot_size * 100,000 * pip_value
        # So: lot_size = risk_amount / (stop_distance_pips * pip_value * 100000)
        lot_size = risk_amount / (stop_distance_pips * pip_value * 100000)
        
        # Apply minimum lot size
        lot_size = max(lot_size, self.settings.min_lot_size)
        
        # Note: Lot size will be properly rounded by the broker adapter
        # using the actual lot_step from symbol_info
        
        logger.info(f"Position sizing for {symbol}: Confidence={confidence:.3f}, "
                   f"Risk={risk_percent:.1f}% (${risk_amount:.2f}), "
                   f"Lots={lot_size:.2f}, Stop={stop_distance_pips:.1f} pips")
        
        return lot_size, risk_amount
    
    def can_open_position(self, symbol: str, spread_pips: float, new_trade_risk: float = None, side: str = None) -> Tuple[bool, str]:
        """
        Check if position can be opened based on risk rules
        
        Args:
            symbol: Trading symbol
            spread_pips: Current spread in pips
            new_trade_risk: Risk amount for the new trade (optional)
            side: Trade direction ('buy' or 'sell') for duplicate direction check
            
        Returns:
            Tuple of (can_open, reason)
        """
        # Check for position lock (prevents race conditions)
        if symbol in self._position_locks and self._position_locks[symbol]:
            return False, "POSITION_LOCKED"
        
        # Check for duplicate positions using actual MT5 positions
        try:
            import MetaTrader5 as mt5
            mt5_positions = mt5.positions_get(symbol=symbol)
            
            if mt5_positions is not None and len(mt5_positions) > 0:
                for mt5_position in mt5_positions:
                    # Check for same-direction duplicate trades
                    if side:
                        # Convert MT5 position type to our side format
                        mt5_side = 'buy' if mt5_position.type == 0 else 'sell'  # 0=BUY, 1=SELL
                        if mt5_side == side:
                            return False, f"DUPLICATE_{side.upper()}_POSITION"
                    # General duplicate check (any open position on same symbol)
                    return False, "DUPLICATE_SYMBOL"
                    
        except Exception as e:
            logger.warning(f"Failed to check MT5 positions for {symbol}: {e}")
            # Fallback to internal positions check if MT5 check fails
            if symbol in self.positions:
                position = self.positions[symbol]
                if position.status == 'open':
                    if side and position.side == side:
                        return False, f"DUPLICATE_{side.upper()}_POSITION"
                    return False, "DUPLICATE_SYMBOL"
        
        # Check cooldown
        if symbol in self.cooldowns:
            if datetime.now() < self.cooldowns[symbol]:
                remaining = (self.cooldowns[symbol] - datetime.now()).total_seconds() / 60
                return False, f"COOLDOWN_ACTIVE_{remaining:.1f}m"
        
        # Check max concurrent trades
        open_positions = sum(1 for p in self.positions.values() if p.status == 'open')
        if open_positions >= self.settings.max_concurrent_trades:
            return False, "MAX_TRADES_REACHED"
        
        # Check spread
        if spread_pips > self.settings.max_spread_pips:
            return False, f"SPREAD_TOO_WIDE_{spread_pips:.1f}pips"
        
        # Check account risk
        if not self.account_info:
            return False, "NO_ACCOUNT_INFO"
        
        # 🚨 CRITICAL: Check actual account drawdown first (10% max drawdown limit)
        current_drawdown = self._calculate_actual_drawdown()
        max_drawdown_limit = self.settings.max_account_risk_percent  # 10% from settings
        
        if current_drawdown >= max_drawdown_limit:
            logger.warning(f"🚨 DRAWDOWN LIMIT EXCEEDED: {current_drawdown:.2f}% >= {max_drawdown_limit:.2f}% - BLOCKING ALL NEW TRADES")
            return False, f"DRAWDOWN_LIMIT_EXCEEDED_{current_drawdown:.1f}%_>=_{max_drawdown_limit:.1f}%"
        
        # Calculate current risk from open positions
        current_risk_amount = sum(p.risk_amount for p in self.positions.values() if p.status == 'open')
        
        # If new trade risk is provided, check if total would exceed limit
        if new_trade_risk is not None:
            total_risk_amount = current_risk_amount + new_trade_risk
            max_total_risk_amount = self.account_info.equity * self.settings.max_total_risk
            
            if total_risk_amount > max_total_risk_amount:
                current_risk_pct = (current_risk_amount / self.account_info.equity) * 100
                new_risk_pct = (new_trade_risk / self.account_info.equity) * 100
                total_risk_pct = (total_risk_amount / self.account_info.equity) * 100
                return False, f"TOTAL_RISK_EXCEEDED_{total_risk_pct:.1f}%_({current_risk_pct:.1f}%_current_+_{new_risk_pct:.1f}%_new)"
        
        # Fallback: check current risk percentage (for backward compatibility)
        current_risk_pct = (current_risk_amount / self.account_info.equity) * 100
        if current_risk_pct >= self.settings.max_total_risk * 100:
            return False, f"RISK_CAP_REACHED_{current_risk_pct:.1f}%"
        
        return True, "OK"
    
    def open_position(self, symbol: str, side: str, entry_price: float, 
                     stop_loss: float, take_profit: float, confidence: float,
                     spread_pips: float) -> Optional[Position]:
        """
        Open a new position with risk checks
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: ML confidence
            spread_pips: Current spread
            
        Returns:
            Position object if successful, None if rejected
        """
        # Calculate position size first to get risk amount
        lot_size, risk_amount = self.calculate_position_size(
            symbol, entry_price, stop_loss, confidence
        )
        
        if lot_size <= 0:
            logger.warning(f"Invalid lot size calculated for {symbol}: {lot_size}")
            return None
        
        # Check if position can be opened with the calculated risk amount
        can_open, reason = self.can_open_position(symbol, spread_pips, risk_amount)
        if not can_open:
            logger.warning(f"Position rejected for {symbol}: {reason}")
            return None
        
        # Lock the symbol to prevent race conditions
        self._position_locks[symbol] = True
        
        try:
            # Double-check for duplicates after acquiring lock
            if symbol in self.positions and self.positions[symbol].status == 'open':
                logger.warning(f"Duplicate position detected for {symbol} after lock acquisition")
                return None
            
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                lot_size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                risk_amount=risk_amount,
                opened_at=datetime.now()
            )
            
            # Add to positions
            self.positions[symbol] = position
            
            logger.info(f"Position opened: {symbol} {side} {lot_size} lots @ {entry_price}, "
                       f"SL={stop_loss}, TP={take_profit}, Risk={risk_amount:.2f}")
            
            return position
            
        finally:
            # Always unlock the symbol
            self._position_locks[symbol] = False
    
    def close_position(self, symbol: str, close_price: float, reason: str = "manual"):
        """
        Close a position and start cooldown
        
        Args:
            symbol: Trading symbol
            close_price: Close price
            reason: Close reason
        """
        # 🚀 CRITICAL FIX: Always set global cooldown regardless of position tracking
        self._last_trade_close_time = datetime.now()
        logger.info(f"[COOLDOWN] Global 10-minute cooldown started at {self._last_trade_close_time.strftime('%H:%M:%S')}")
        
        if symbol not in self.positions:
            logger.warning(f"Position not found for {symbol} - but cooldown still activated")
            return None
        
        position = self.positions[symbol]
        position.status = 'closed'
        position.close_time = datetime.now()
        
        # Calculate P&L
        if position.side == 'long':
            pnl = (close_price - position.entry_price) * position.lot_size * 100000
        else:
            pnl = (position.entry_price - close_price) * position.lot_size * 100000
        
        # Start per-symbol cooldown
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.settings.cooldown_minutes)
        
        logger.info(f"Position closed: {symbol} @ {close_price}, P&L={pnl:.2f}, Reason={reason}")
        
        return pnl
    
    def is_in_global_cooldown(self) -> tuple[bool, str]:
        """
        Check if the bot is in global cooldown period (10 minutes after last trade close)
        
        Returns:
            Tuple of (is_in_cooldown, message)
        """
        if self._last_trade_close_time is None:
            return False, "No previous trades"
        
        current_time = datetime.now()
        time_since_close = current_time - self._last_trade_close_time
        cooldown_duration = timedelta(minutes=10)
        
        if time_since_close < cooldown_duration:
            remaining_time = cooldown_duration - time_since_close
            remaining_minutes = int(remaining_time.total_seconds() / 60)
            remaining_seconds = int(remaining_time.total_seconds() % 60)
            return True, f"still in cooldown window ({remaining_minutes}m {remaining_seconds}s left)"
        
        return False, "cooldown expired"
    
    def update_position_prices(self, symbol: str, current_price: float, historical_data: List[Dict] = None, 
                              signal_data_1m: Dict = None, bars_held: int = None, 
                              signal_data_5m: Dict = None, signal_data_15m: Dict = None):
        """
        Update position with current market price for monitoring and trailing stops.
        Now includes momentum-based exit conditions from generate_exit_conditions().
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            historical_data: Historical OHLC data for ATR calculation
            signal_data: Current signal data for momentum exit checks
            bars_held: Number of bars position has been held
        """
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            return
        
        position = self.positions[symbol]
        
        # 🎯 MULTI-TIMEFRAME MOMENTUM EXIT CHECKS (ENHANCED - HTF Protection)
        if signal_data_1m and bars_held is not None:
            try:
                from core.signals import generate_exit_conditions
                
                # Generate 1m momentum exit conditions
                exit_conditions_1m = generate_exit_conditions(signal_data_1m, bars_held)
                
                # 🚀 NEW: Check HTF momentum alignment before allowing exit
                should_exit = False
                exit_reason = "unknown"
                
                if position.side == 'buy' and exit_conditions_1m.get('end_long_trade', False):
                    # 1m wants to exit long - check if HTF still supports
                    htf_alignment = self._check_htf_momentum_alignment(signal_data_5m, signal_data_15m, 'long')
                    
                    if htf_alignment['should_exit']:
                        should_exit = True
                        exit_reason = f"1m_momentum_{htf_alignment['reason']}"
                        logger.info(f"🎯 [MTF_EXIT] {symbol}: Long position closed - 1m+5m+15m all oppose: {htf_alignment['details']}")
                    else:
                        logger.info(f"🛡️ [HTF_PROTECTION] {symbol}: Long position HELD - HTF still supports: {htf_alignment['details']}")
                        
                elif position.side == 'sell' and exit_conditions_1m.get('end_short_trade', False):
                    # 1m wants to exit short - check if HTF still supports
                    htf_alignment = self._check_htf_momentum_alignment(signal_data_5m, signal_data_15m, 'short')
                    
                    if htf_alignment['should_exit']:
                        should_exit = True
                        exit_reason = f"1m_momentum_{htf_alignment['reason']}"
                        logger.info(f"🎯 [MTF_EXIT] {symbol}: Short position closed - 1m+5m+15m all oppose: {htf_alignment['details']}")
                    else:
                        logger.info(f"🛡️ [HTF_PROTECTION] {symbol}: Short position HELD - HTF still supports: {htf_alignment['details']}")
                
                # Execute exit if all timeframes agree
                if should_exit:
                    self.close_position(symbol, current_price, f"mtf_momentum_exit_{exit_reason}")
                    return  # Exit immediately, don't check other conditions
                    
            except Exception as e:
                logger.warning(f"⚠️ [MTF_MOMENTUM_EXIT] Error checking multi-timeframe momentum exits for {symbol}: {e}")
        
        # 🚀 ENHANCED SL/TP MANAGEMENT (Multi-tier + Trend-aware)
        if historical_data:
            current_atr = self._calculate_current_atr(historical_data)
            if current_atr > 0:
                # Enhanced trailing stop with HTF awareness
                new_stop, stop_updated, stop_log = self._calculate_enhanced_trailing_stop(
                    position, current_price, current_atr, signal_data_5m, signal_data_15m
                )
                if stop_updated:
                    logger.info(f"📈 [ENHANCED_SL] {symbol}: {stop_log}")
                    # Update the stop loss in MT5
                    self._update_stop_loss_in_mt5(symbol, new_stop)
                
                # Enhanced take profit with partial TP and trend awareness
                new_tp, tp_updated, tp_log = self._calculate_enhanced_take_profit(
                    position, current_price, current_atr, signal_data_5m, signal_data_15m
                )
                if tp_updated:
                    logger.info(f"🎯 [ENHANCED_TP] {symbol}: {tp_log}")
                    # Update the take profit in MT5
                    self._update_take_profit_in_mt5(symbol, new_tp)
        
        # 🛑 ATR STOP LOSS & TAKE PROFIT CHECKS (EXISTING - Preserved)
        if position.side == 'buy':
            if current_price <= position.stop_loss:
                logger.info(f"🛑 [ATR_STOP_LOSS] {symbol}: Long position closed at stop loss")
                self.close_position(symbol, current_price, "atr_stop_loss")
            elif current_price >= position.take_profit:
                logger.info(f"🎯 [ATR_TAKE_PROFIT] {symbol}: Long position closed at take profit")
                self.close_position(symbol, current_price, "atr_take_profit")
        else:  # sell
            if current_price >= position.stop_loss:
                logger.info(f"🛑 [ATR_STOP_LOSS] {symbol}: Short position closed at stop loss")
                self.close_position(symbol, current_price, "atr_stop_loss")
            elif current_price <= position.take_profit:
                logger.info(f"🎯 [ATR_TAKE_PROFIT] {symbol}: Short position closed at take profit")
                self.close_position(symbol, current_price, "atr_take_profit")
    
    def _check_htf_momentum_alignment(self, signal_data_5m: Dict, signal_data_15m: Dict, position_side: str) -> Dict:
        """
        Check if higher timeframe momentum still supports the position
        
        Args:
            signal_data_5m: 5m signal data (can be None)
            signal_data_15m: 15m signal data (can be None)
            position_side: 'long' or 'short'
            
        Returns:
            Dict with 'should_exit', 'reason', and 'details'
        """
        try:
            # If no HTF data available, allow exit (fallback to 1m only)
            if not signal_data_5m and not signal_data_15m:
                return {
                    'should_exit': True,
                    'reason': 'no_htf_data',
                    'details': 'No 5m/15m data available - using 1m only'
                }
            
            # Check 5m momentum
            momentum_5m = 'neutral'
            if signal_data_5m:
                signal_5m = signal_data_5m.get('signal', 0)
                if signal_5m > 0:
                    momentum_5m = 'bullish'
                elif signal_5m < 0:
                    momentum_5m = 'bearish'
            
            # Check 15m momentum
            momentum_15m = 'neutral'
            if signal_data_15m:
                signal_15m = signal_data_15m.get('signal', 0)
                if signal_15m > 0:
                    momentum_15m = 'bullish'
                elif signal_15m < 0:
                    momentum_15m = 'bearish'
            
            # Determine if HTF momentum opposes the position
            if position_side == 'long':
                # For long positions, check if HTF momentum is bearish
                opposes_5m = momentum_5m == 'bearish'
                opposes_15m = momentum_15m == 'bearish'
                
                # Exit only if BOTH 5m and 15m oppose (or if one is missing and the other opposes)
                if (opposes_5m and opposes_15m) or (not signal_data_5m and opposes_15m) or (not signal_data_15m and opposes_5m):
                    return {
                        'should_exit': True,
                        'reason': 'htf_opposes',
                        'details': f'5m={momentum_5m}, 15m={momentum_15m} - both oppose long'
                    }
                else:
                    return {
                        'should_exit': False,
                        'reason': 'htf_supports',
                        'details': f'5m={momentum_5m}, 15m={momentum_15m} - HTF still supports long'
                    }
                    
            else:  # position_side == 'short'
                # For short positions, check if HTF momentum is bullish
                opposes_5m = momentum_5m == 'bullish'
                opposes_15m = momentum_15m == 'bullish'
                
                # Exit only if BOTH 5m and 15m oppose (or if one is missing and the other opposes)
                if (opposes_5m and opposes_15m) or (not signal_data_5m and opposes_15m) or (not signal_data_15m and opposes_5m):
                    return {
                        'should_exit': True,
                        'reason': 'htf_opposes',
                        'details': f'5m={momentum_5m}, 15m={momentum_15m} - both oppose short'
                    }
                else:
                    return {
                        'should_exit': False,
                        'reason': 'htf_supports',
                        'details': f'5m={momentum_5m}, 15m={momentum_15m} - HTF still supports short'
                    }
                    
        except Exception as e:
            logger.warning(f"Error checking HTF momentum alignment: {e}")
            return {
                'should_exit': True,
                'reason': 'error',
                'details': f'Error checking HTF alignment: {e}'
            }

    def _determine_momentum_exit_reason(self, signal_data: Dict, bars_held: int, side: str) -> str:
        """
        Determine the specific reason for momentum-based exit
        
        Args:
            signal_data: Current signal data
            bars_held: Number of bars position has been held
            side: 'long' or 'short'
            
        Returns:
            String describing the specific exit reason
        """
        try:
            confidence = signal_data.get('confidence', 0.0)
            feature_series = signal_data.get('feature_series')
            
            # Check confidence-based exit first
            if confidence < 0.1:
                return f"low_confidence_{confidence:.3f}"
            
            # Check time-based exit
            if bars_held == 4:
                return f"time_based_4_bars"
            
            # Check momentum indicator exits
            if feature_series and bars_held >= 2:
                if side == 'long':
                    # Check for overbought conditions
                    rsi_f1 = feature_series.f1 if hasattr(feature_series, 'f1') else 50.0
                    williams_r = feature_series.f2 if hasattr(feature_series, 'f2') else -50.0
                    cci = feature_series.f3 if hasattr(feature_series, 'f3') else 0.0
                    
                    if rsi_f1 > 70:
                        return f"rsi_overbought_{rsi_f1:.1f}"
                    elif williams_r > -20:
                        return f"williams_overbought_{williams_r:.1f}"
                    elif cci > 100:
                        return f"cci_overbought_{cci:.1f}"
                        
                else:  # short
                    # Check for oversold conditions
                    rsi_f1 = feature_series.f1 if hasattr(feature_series, 'f1') else 50.0
                    williams_r = feature_series.f2 if hasattr(feature_series, 'f2') else -50.0
                    cci = feature_series.f3 if hasattr(feature_series, 'f3') else 0.0
                    
                    if rsi_f1 < 30:
                        return f"rsi_oversold_{rsi_f1:.1f}"
                    elif williams_r < -80:
                        return f"williams_oversold_{williams_r:.1f}"
                    elif cci < -100:
                        return f"cci_oversold_{cci:.1f}"
            
            # Default fallback
            return f"momentum_reversal_bars_{bars_held}"
            
        except Exception as e:
            logger.warning(f"Error determining momentum exit reason: {e}")
            return f"momentum_exit_error_{bars_held}_bars"
    
    def _calculate_current_atr(self, historical_data: List[Dict]) -> float:
        """Calculate current ATR from historical data"""
        if not historical_data or len(historical_data) < 14:
            return 0.0
        
        df = pd.DataFrame(historical_data)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        
        # Calculate 14-period ATR
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def _update_stop_loss_in_mt5(self, symbol: str, new_stop_loss: float):
        """Update stop loss in MT5 - ACTUALLY SEND TO MT5"""
        try:
            # Get the position to find the ticket
            if symbol not in self.positions:
                logger.warning(f"Cannot update SL for {symbol}: position not found")
                return False
            
            position = self.positions[symbol]
            if position.status != 'open':
                logger.warning(f"Cannot update SL for {symbol}: position not open")
                return False
            
            # Import MT5 adapter to modify position
            from adapters.mt5_adapter import MT5Adapter
            mt5_adapter = MT5Adapter()
            
            # Get current position ticket from MT5
            import MetaTrader5 as mt5
            mt5_positions = mt5.positions_get(symbol=symbol)
            
            if not mt5_positions or len(mt5_positions) == 0:
                logger.warning(f"Cannot update SL for {symbol}: no MT5 position found")
                return False
            
            # Use the first position found for this symbol
            mt5_position = mt5_positions[0]
            ticket = mt5_position.ticket
            
            # Update the stop loss in MT5
            success = mt5_adapter.modify_position(
                ticket=ticket,
                stop_loss=new_stop_loss,
                take_profit=None  # Keep existing TP
            )
            
            if success:
                # Update our internal position tracking
                position.stop_loss = new_stop_loss
                logger.info(f"✅ [MT5_UPDATE] {symbol}: Stop loss updated to {new_stop_loss:.5f} (Ticket: {ticket})")
                return True
            else:
                logger.error(f"❌ [MT5_UPDATE] {symbol}: Failed to update stop loss to {new_stop_loss:.5f}")
                return False
                
        except Exception as e:
            logger.error(f"❌ [MT5_UPDATE] Error updating stop loss for {symbol}: {e}")
            return False
    
    def _update_take_profit_in_mt5(self, symbol: str, new_take_profit: float):
        """Update take profit in MT5 - ACTUALLY SEND TO MT5"""
        try:
            # Get the position to find the ticket
            if symbol not in self.positions:
                logger.warning(f"Cannot update TP for {symbol}: position not found")
                return False
            
            position = self.positions[symbol]
            if position.status != 'open':
                logger.warning(f"Cannot update TP for {symbol}: position not open")
                return False
            
            # Import MT5 adapter to modify position
            from adapters.mt5_adapter import MT5Adapter
            mt5_adapter = MT5Adapter()
            
            # Get current position ticket from MT5
            import MetaTrader5 as mt5
            mt5_positions = mt5.positions_get(symbol=symbol)
            
            if not mt5_positions or len(mt5_positions) == 0:
                logger.warning(f"Cannot update TP for {symbol}: no MT5 position found")
                return False
            
            # Use the first position found for this symbol
            mt5_position = mt5_positions[0]
            ticket = mt5_position.ticket
            
            # Update the take profit in MT5
            success = mt5_adapter.modify_position(
                ticket=ticket,
                stop_loss=None,  # Keep existing SL
                take_profit=new_take_profit
            )
            
            if success:
                # Update our internal position tracking
                position.take_profit = new_take_profit
                logger.info(f"✅ [MT5_UPDATE] {symbol}: Take profit updated to {new_take_profit:.5f} (Ticket: {ticket})")
                return True
            else:
                logger.error(f"❌ [MT5_UPDATE] {symbol}: Failed to update take profit to {new_take_profit:.5f}")
                return False
                
        except Exception as e:
            logger.error(f"❌ [MT5_UPDATE] Error updating take profit for {symbol}: {e}")
            return False
    
    def _calculate_enhanced_trailing_stop(self, position: Position, current_price: float, 
                                        current_atr: float, signal_data_5m: Dict = None, 
                                        signal_data_15m: Dict = None) -> Tuple[float, bool, str]:
        """
        Enhanced multi-tier trailing stop with trend-aware adjustments
        
        Args:
            position: Current position
            current_price: Current market price
            current_atr: Current ATR value
            signal_data_5m: 5m signal data for trend awareness
            signal_data_15m: 15m signal data for trend awareness
            
        Returns:
            Tuple of (new_stop_loss, stop_updated, log_message)
        """
        if position.status != 'open':
            return position.stop_loss, False, "Position not open"
        
        # Calculate trend strength from HTF data
        trend_strength = self._calculate_trend_strength(signal_data_5m, signal_data_15m, position.side)
        position.trend_strength = trend_strength
        
        # Determine if HTF supports the position
        htf_supports = self._check_htf_support(signal_data_5m, signal_data_15m, position.side)
        
        # Calculate micro and main trailing distances
        micro_distance = current_atr * 0.3  # 30% of ATR for noise protection
        main_distance = current_atr * 1.0   # Full ATR for risk control
        
        # Adjust distances based on trend strength
        if trend_strength > 0.7:  # Strong trend
            micro_distance *= 0.8  # Tighter micro trailing
            main_distance *= 1.2   # Wider main trailing
        elif trend_strength < 0.3:  # Weak trend
            micro_distance *= 1.2  # Wider micro trailing
            main_distance *= 0.9   # Tighter main trailing
        
        position.micro_trailing_distance = micro_distance
        position.main_trailing_distance = main_distance
        
        updated = False
        log_message = ""
        
        if position.side == 'buy':
            # Update best price if current price is higher
            if current_price > position.best_price:
                position.best_price = current_price
                
                # Calculate micro trailing stop (for noise protection)
                micro_stop = current_price - micro_distance
                
                # Calculate main trailing stop (for risk control)
                main_stop = current_price - main_distance
                
                # Don't move below break-even
                break_even_stop = position.entry_price
                micro_stop = max(micro_stop, break_even_stop)
                main_stop = max(main_stop, break_even_stop)
                
                # Determine which stop to use based on HTF support
                if htf_supports and trend_strength > 0.5:
                    # Use micro trailing when HTF strongly supports
                    new_stop = micro_stop
                    stop_type = "micro"
                    position.micro_trailing_enabled = True
                    position.last_micro_update = datetime.now()
                else:
                    # Use main trailing for risk control
                    new_stop = main_stop
                    stop_type = "main"
                    position.main_trailing_enabled = True
                    position.last_main_update = datetime.now()
                
                # Only update if new stop is better (higher)
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    updated = True
                    log_message = f"Enhanced trailing ({stop_type}): {new_stop:.5f} | Trend: {trend_strength:.2f} | HTF: {'supports' if htf_supports else 'neutral'}"
        
        else:  # sell
            # Update best price if current price is lower
            if current_price < position.best_price:
                position.best_price = current_price
                
                # Calculate micro trailing stop (for noise protection)
                micro_stop = current_price + micro_distance
                
                # Calculate main trailing stop (for risk control)
                main_stop = current_price + main_distance
                
                # Don't move above break-even
                break_even_stop = position.entry_price
                micro_stop = min(micro_stop, break_even_stop)
                main_stop = min(main_stop, break_even_stop)
                
                # Determine which stop to use based on HTF support
                if htf_supports and trend_strength > 0.5:
                    # Use micro trailing when HTF strongly supports
                    new_stop = micro_stop
                    stop_type = "micro"
                    position.micro_trailing_enabled = True
                    position.last_micro_update = datetime.now()
                else:
                    # Use main trailing for risk control
                    new_stop = main_stop
                    stop_type = "main"
                    position.main_trailing_enabled = True
                    position.last_main_update = datetime.now()
                
                # Only update if new stop is better (lower)
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    updated = True
                    log_message = f"Enhanced trailing ({stop_type}): {new_stop:.5f} | Trend: {trend_strength:.2f} | HTF: {'supports' if htf_supports else 'neutral'}"
        
        return position.stop_loss, updated, log_message
    
    def _calculate_enhanced_take_profit(self, position: Position, current_price: float, 
                                      current_atr: float, signal_data_5m: Dict = None, 
                                      signal_data_15m: Dict = None) -> Tuple[float, bool, str]:
        """
        Enhanced take profit with partial TP and trend-aware adjustments
        
        Args:
            position: Current position
            current_price: Current market price
            current_atr: Current ATR value
            signal_data_5m: 5m signal data for trend awareness
            signal_data_15m: 15m signal data for trend awareness
            
        Returns:
            Tuple of (new_take_profit, tp_updated, log_message)
        """
        if position.status != 'open':
            return position.take_profit, False, "Position not open"
        
        # Calculate trend strength from HTF data
        trend_strength = self._calculate_trend_strength(signal_data_5m, signal_data_15m, position.side)
        
        # Determine if HTF supports the position
        htf_supports = self._check_htf_support(signal_data_5m, signal_data_15m, position.side)
        
        updated = False
        log_message = ""
        
        # Check for partial TP opportunity
        if not position.partial_tp_taken:
            # Calculate baseline TP (50% of original TP distance)
            if position.side == 'buy':
                original_tp_distance = position.take_profit - position.entry_price
                baseline_tp = position.entry_price + (original_tp_distance * 0.5)
                
                if current_price >= baseline_tp:
                    # Take partial TP
                    partial_lot_size = position.lot_size * position.partial_tp_percentage
                    success = self._execute_partial_tp(position.symbol, partial_lot_size, current_price)
                    
                    if success:
                        position.partial_tp_taken = True
                        position.partial_tp_price = current_price
                        position.remaining_lot_size = position.lot_size - partial_lot_size
                        position.lot_size = position.remaining_lot_size
                        
                        # Enable trailing TP for remaining position
                        position.trailing_tp_enabled = True
                        position.trailing_tp_distance = current_atr * 0.8
                        position.best_tp_price = current_price
                        
                        log_message = f"Partial TP taken: {partial_lot_size:.3f} lots @ {current_price:.5f} | Remaining: {position.remaining_lot_size:.3f} lots"
                        updated = True
                        logger.info(f"🎯 [PARTIAL_TP] {position.symbol}: {log_message}")
        
        # Handle trailing TP for remaining position
        if position.trailing_tp_enabled and position.partial_tp_taken:
            if position.side == 'buy':
                if current_price > position.best_tp_price:
                    position.best_tp_price = current_price
                    
                    # Calculate new TP based on trend strength
                    if htf_supports and trend_strength > 0.6:
                        # Extend TP further when HTF strongly supports
                        new_tp = current_price + (current_atr * 1.5)
                        tp_type = "extended"
                    else:
                        # Standard trailing TP
                        new_tp = current_price + position.trailing_tp_distance
                        tp_type = "trailing"
                    
                    # Only update if new TP is better (higher)
                    if new_tp > position.take_profit:
                        position.take_profit = new_tp
                        position.last_tp_update = datetime.now()
                        updated = True
                        log_message = f"Trailing TP ({tp_type}): {new_tp:.5f} | Trend: {trend_strength:.2f} | HTF: {'supports' if htf_supports else 'neutral'}"
            
            else:  # sell
                if current_price < position.best_tp_price:
                    position.best_tp_price = current_price
                    
                    # Calculate new TP based on trend strength
                    if htf_supports and trend_strength > 0.6:
                        # Extend TP further when HTF strongly supports
                        new_tp = current_price - (current_atr * 1.5)
                        tp_type = "extended"
                    else:
                        # Standard trailing TP
                        new_tp = current_price - position.trailing_tp_distance
                        tp_type = "trailing"
                    
                    # Only update if new TP is better (lower)
                    if new_tp < position.take_profit:
                        position.take_profit = new_tp
                        position.last_tp_update = datetime.now()
                        updated = True
                        log_message = f"Trailing TP ({tp_type}): {new_tp:.5f} | Trend: {trend_strength:.2f} | HTF: {'supports' if htf_supports else 'neutral'}"
        
        return position.take_profit, updated, log_message
    
    def _calculate_trend_strength(self, signal_data_5m: Dict, signal_data_15m: Dict, position_side: str) -> float:
        """Calculate combined trend strength from 5m and 15m data"""
        if not signal_data_5m and not signal_data_15m:
            return 0.5  # Neutral if no HTF data
        
        strength_5m = 0.0
        strength_15m = 0.0
        
        if signal_data_5m:
            signal_5m = signal_data_5m.get('signal', 0)
            confidence_5m = signal_data_5m.get('confidence', 0.0)
            
            if position_side == 'buy' and signal_5m > 0:
                strength_5m = confidence_5m
            elif position_side == 'sell' and signal_5m < 0:
                strength_5m = confidence_5m
            else:
                strength_5m = 0.0
        
        if signal_data_15m:
            signal_15m = signal_data_15m.get('signal', 0)
            confidence_15m = signal_data_15m.get('confidence', 0.0)
            
            if position_side == 'buy' and signal_15m > 0:
                strength_15m = confidence_15m
            elif position_side == 'sell' and signal_15m < 0:
                strength_15m = confidence_15m
            else:
                strength_15m = 0.0
        
        # Weight 15m more heavily (0.6) than 5m (0.4)
        if signal_data_5m and signal_data_15m:
            return (strength_5m * 0.4) + (strength_15m * 0.6)
        elif signal_data_5m:
            return strength_5m
        elif signal_data_15m:
            return strength_15m
        else:
            return 0.5
    
    def _check_htf_support(self, signal_data_5m: Dict, signal_data_15m: Dict, position_side: str) -> bool:
        """Check if HTF data supports the position"""
        if not signal_data_5m and not signal_data_15m:
            return False  # No HTF data, assume no support
        
        supports_5m = False
        supports_15m = False
        
        if signal_data_5m:
            signal_5m = signal_data_5m.get('signal', 0)
            if position_side == 'buy' and signal_5m > 0:
                supports_5m = True
            elif position_side == 'sell' and signal_5m < 0:
                supports_5m = True
        
        if signal_data_15m:
            signal_15m = signal_data_15m.get('signal', 0)
            if position_side == 'buy' and signal_15m > 0:
                supports_15m = True
            elif position_side == 'sell' and signal_15m < 0:
                supports_15m = True
        
        # Support if at least one HTF supports (or both if both available)
        if signal_data_5m and signal_data_15m:
            return supports_5m and supports_15m
        else:
            return supports_5m or supports_15m
    
    def _execute_partial_tp(self, symbol: str, partial_lot_size: float, current_price: float) -> bool:
        """Execute partial take profit in MT5"""
        try:
            # Import MT5 adapter
            from adapters.mt5_adapter import MT5Adapter
            mt5_adapter = MT5Adapter()
            
            # Get current position ticket from MT5
            import MetaTrader5 as mt5
            mt5_positions = mt5.positions_get(symbol=symbol)
            
            if not mt5_positions or len(mt5_positions) == 0:
                logger.warning(f"Cannot execute partial TP for {symbol}: no MT5 position found")
                return False
            
            # Use the first position found for this symbol
            mt5_position = mt5_positions[0]
            ticket = mt5_position.ticket
            
            # Execute partial close in MT5
            success = mt5_adapter.close_position_partial(ticket, partial_lot_size)
            
            if success:
                logger.info(f"✅ [PARTIAL_TP_EXECUTED] {symbol}: {partial_lot_size:.3f} lots closed @ {current_price:.5f} (Ticket: {ticket})")
                return True
            else:
                logger.error(f"❌ [PARTIAL_TP_FAILED] {symbol}: Failed to close {partial_lot_size:.3f} lots")
                return False
                
        except Exception as e:
            logger.error(f"❌ [PARTIAL_TP_ERROR] Error executing partial TP for {symbol}: {e}")
            return False
    
    def _calculate_current_risk(self) -> float:
        """Calculate current account risk percentage from open positions"""
        if not self.account_info:
            return 0.0
        
        total_risk = sum(p.risk_amount for p in self.positions.values() if p.status == 'open')
        return (total_risk / self.account_info.equity) * 100.0
    
    def _calculate_actual_drawdown(self) -> float:
        """
        Calculate actual account drawdown percentage from peak equity.
        
        This is the REAL drawdown calculation that tracks actual losses,
        not just potential risk amounts from open positions.
        
        Returns:
            float: Current drawdown percentage (0.0 = no drawdown, 10.0 = 10% drawdown)
        """
        if not self.account_info or self._max_equity_peak is None:
            return 0.0
        
        # Calculate drawdown from peak equity
        if self._max_equity_peak <= 0:
            return 0.0
        
        current_equity = self.account_info.equity
        drawdown_amount = self._max_equity_peak - current_equity
        drawdown_percentage = (drawdown_amount / self._max_equity_peak) * 100.0
        
        # Ensure drawdown is never negative (can't be more than 100% either)
        drawdown_percentage = max(0.0, min(100.0, drawdown_percentage))
        
        return drawdown_percentage
    
    def _get_pip_value(self, symbol: str, price: float) -> float:
        """Get pip value for symbol in account currency"""
        # This method should return the monetary value of 1 pip
        # For now, using a simplified calculation - in production this should
        # use exact broker pip values based on account currency and lot size
        pip_multiplier = get_pip_value(symbol)
        # Simplified: assume $10 per pip for standard lot (this should be calculated properly)
        return 10.0  # Placeholder - should calculate based on account currency
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol using symbol-aware system"""
        return get_pip_value(symbol)
    
    def get_risk_summary(self) -> Dict[str, any]:
        """Get current risk status summary"""
        if not self.account_info:
            return {"error": "No account info available"}
        
        open_positions = [p for p in self.positions.values() if p.status == 'open']
        current_risk = self._calculate_current_risk()
        current_drawdown = self._calculate_actual_drawdown()
        
        return {
            "account_balance": self.account_info.balance,
            "account_equity": self.account_info.equity,
            "starting_balance": self._starting_balance,
            "max_equity_peak": self._max_equity_peak,
            "current_risk_percent": current_risk,
            "current_drawdown_percent": current_drawdown,
            "max_drawdown_percent": self.settings.max_account_risk_percent,
            "max_risk_percent": self.settings.max_account_risk_percent,
            "drawdown_limit_exceeded": current_drawdown >= self.settings.max_account_risk_percent,
            "open_positions": len(open_positions),
            "max_positions": self.settings.max_concurrent_trades,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "lot_size": p.lot_size,
                    "entry_price": p.entry_price,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "confidence": p.confidence,
                    "risk_amount": p.risk_amount,
                    "opened_at": p.opened_at.isoformat()
                }
                for p in open_positions
            ],
            "cooldowns": {
                symbol: end_time.isoformat() 
                for symbol, end_time in self.cooldowns.items()
                if end_time > datetime.now()
            }
        }
    
    def get_trade_stats(self) -> Dict[str, any]:
        """Get trading statistics"""
        all_positions = list(self.positions.values())
        closed_positions = [p for p in all_positions if p.status == 'closed']
        
        if not closed_positions:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0
            }
        
        # Calculate P&L for closed positions (simplified)
        pnls = []
        for position in closed_positions:
            # This would need actual close prices from trade history
            pnl = 0.0  # Placeholder
            pnls.append(pnl)
        
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        return {
            "total_trades": len(closed_positions),
            "win_rate": len(wins) / len(closed_positions) * 100 if closed_positions else 0,
            "total_pnl": sum(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0
        }
