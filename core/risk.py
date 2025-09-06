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
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_daily_risk: float = 0.05  # 5% daily
    max_total_risk: float = 0.10  # 10% total
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
    status: str = 'open'  # 'open', 'closed', 'stopped'
    # Trailing stop fields
    original_stop_loss: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_distance: float = 0.0
    trailing_enabled: bool = False
    best_price: float = 0.0
    atr_distance: float = 0.0
    break_even_triggered: bool = False


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
        
    def update_account_info(self, account_info: AccountInfo):
        """Update account information for risk calculations"""
        self.account_info = account_info
        logger.info(f"Account updated: Balance={account_info.balance:.2f}, "
                   f"Equity={account_info.equity:.2f}, Free Margin={account_info.free_margin:.2f}")
    
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
        base_risk_percent = self.settings.max_risk_per_trade * 100  # Convert to percentage
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
        risk_percent = base_risk_percent * confidence_multiplier
        
        # Calculate risk amount in account currency
        risk_amount = self.account_info.equity * (risk_percent / 100.0)
        
        # Calculate pip value and stop distance using symbol-aware system
        pip_value = get_pip_value(symbol)
        stop_distance_pips = get_price_distance_in_pips(symbol, abs(entry_price - stop_loss))
        
        if stop_distance_pips < self.settings.min_stop_distance_pips:
            logger.warning(f"Stop distance too small: {stop_distance_pips:.1f} pips")
            return 0.0, 0.0
        
        # Calculate lot size
        lot_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Apply minimum lot size
        lot_size = max(lot_size, self.settings.min_lot_size)
        
        # Note: Lot size will be properly rounded by the broker adapter
        # using the actual lot_step from symbol_info
        
        logger.info(f"Position sizing for {symbol}: Risk={risk_amount:.2f}, "
                   f"Lots={lot_size:.2f}, Stop={stop_distance_pips:.1f} pips")
        
        return lot_size, risk_amount
    
    def can_open_position(self, symbol: str, spread_pips: float) -> Tuple[bool, str]:
        """
        Check if position can be opened based on risk rules
        
        Args:
            symbol: Trading symbol
            spread_pips: Current spread in pips
            
        Returns:
            Tuple of (can_open, reason)
        """
        # Check if symbol already has open position
        if symbol in self.positions and self.positions[symbol].status == 'open':
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
        
        current_risk = self._calculate_current_risk()
        if current_risk >= self.settings.max_total_risk * 100:
            return False, f"RISK_CAP_REACHED_{current_risk:.1f}%"
        
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
        # Check if position can be opened
        can_open, reason = self.can_open_position(symbol, spread_pips)
        if not can_open:
            logger.warning(f"Position rejected for {symbol}: {reason}")
            return None
        
        # Calculate position size
        lot_size, risk_amount = self.calculate_position_size(
            symbol, entry_price, stop_loss, confidence
        )
        
        if lot_size <= 0:
            logger.warning(f"Invalid lot size for {symbol}: {lot_size}")
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
    
    def close_position(self, symbol: str, close_price: float, reason: str = "manual"):
        """
        Close a position and start cooldown
        
        Args:
            symbol: Trading symbol
            close_price: Close price
            reason: Close reason
        """
        if symbol not in self.positions:
            logger.warning(f"Position not found for {symbol}")
            return
        
        position = self.positions[symbol]
        position.status = 'closed'
        
        # Calculate P&L
        if position.side == 'long':
            pnl = (close_price - position.entry_price) * position.lot_size * 100000
        else:
            pnl = (position.entry_price - close_price) * position.lot_size * 100000
        
        # Start cooldown
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.settings.cooldown_minutes)
        
        logger.info(f"Position closed: {symbol} @ {close_price}, P&L={pnl:.2f}, Reason={reason}")
        
        return pnl
    
    def update_position_prices(self, symbol: str, current_price: float, historical_data: List[Dict] = None):
        """
        Update position with current market price for monitoring and trailing stops
        """
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            return
        
        position = self.positions[symbol]
        
        # Update trailing stop if historical data is available
        if historical_data:
            current_atr = self._calculate_current_atr(historical_data)
            if current_atr > 0:
                new_stop, updated, log_msg = calculate_trailing_stop(position, current_price, current_atr, historical_data)
                if updated:
                    logger.info(f"[TRAILING] {symbol}: {log_msg}")
                    # Update the stop loss in MT5 if needed
                    self._update_stop_loss_in_mt5(symbol, new_stop)
        
        # Check for stop loss or take profit
        if position.side == 'buy':
            if current_price <= position.stop_loss:
                self.close_position(symbol, current_price, "stop_loss")
            elif current_price >= position.take_profit:
                self.close_position(symbol, current_price, "take_profit")
        else:  # sell
            if current_price >= position.stop_loss:
                self.close_position(symbol, current_price, "stop_loss")
            elif current_price <= position.take_profit:
                self.close_position(symbol, current_price, "take_profit")
    
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
        """Update stop loss in MT5 (placeholder for now)"""
        # This would integrate with MT5 adapter to modify the stop loss
        # For now, we'll just log the update
        logger.debug(f"MT5 Stop Loss Update: {symbol} -> {new_stop_loss:.5f}")
    
    def _calculate_current_risk(self) -> float:
        """Calculate current account risk percentage"""
        if not self.account_info:
            return 0.0
        
        total_risk = sum(p.risk_amount for p in self.positions.values() if p.status == 'open')
        return (total_risk / self.account_info.equity) * 100.0
    
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
        
        return {
            "account_balance": self.account_info.balance,
            "account_equity": self.account_info.equity,
            "current_risk_percent": current_risk,
            "max_risk_percent": self.settings.max_account_risk_percent,
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
