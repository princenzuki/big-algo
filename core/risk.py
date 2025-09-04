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
# Dynamic Spread & Smart Stop Logic
# ===============================

def get_dynamic_spread(symbol_data: List[Dict], base_min_spread: float, spread_multiplier: float = 1.5) -> float:
    """
    Calculate intelligent spread per symbol based on recent market conditions.
    
    Args:
        symbol_data: List of dicts with 'high', 'low', 'open', 'close' columns
        base_min_spread: float, minimum allowed spread in pips
        spread_multiplier: float, multiplies average spread for breathing room

    Returns:
        max_allowed_spread: float
    """
    if not symbol_data or len(symbol_data) < 20:
        return base_min_spread
    
    # Convert to DataFrame for easier calculation
    df = pd.DataFrame(symbol_data)
    
    # Estimate spread from high-low range (proxy for volatility)
    # This is a reasonable approximation since we don't have actual bid/ask data
    df['estimated_spread'] = (df['high'] - df['low']) * 10000  # Convert to pips
    
    # Calculate rolling average of last 20 bars
    recent_avg = df['estimated_spread'].rolling(window=20).mean().iloc[-1]
    
    # Return max of base minimum or calculated dynamic spread
    max_allowed_spread = max(base_min_spread, recent_avg * spread_multiplier)
    
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

@dataclass
class RiskSettings:
    """Risk management configuration"""
    max_account_risk_percent: float = 10.0  # 10% max risk
    min_lot_size: float = 0.01
    max_concurrent_trades: int = 5
    cooldown_minutes: int = 10
    max_spread_pips: float = 3.0
    min_stop_distance_pips: float = 5.0

@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    lot_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_amount: float
    opened_at: datetime
    status: str = 'open'  # 'open', 'closed', 'stopped'

@dataclass
class AccountInfo:
    """Account information for risk calculations"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str = 'USD'

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
        base_risk_percent = self.settings.max_account_risk_percent / self.settings.max_concurrent_trades
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
        risk_percent = base_risk_percent * confidence_multiplier
        
        # Calculate risk amount in account currency
        risk_amount = self.account_info.equity * (risk_percent / 100.0)
        
        # Calculate pip value and stop distance
        pip_value = self._get_pip_value(symbol, entry_price)
        stop_distance_pips = abs(entry_price - stop_loss) / self._get_pip_size(symbol)
        
        if stop_distance_pips < self.settings.min_stop_distance_pips:
            logger.warning(f"Stop distance too small: {stop_distance_pips:.1f} pips")
            return 0.0, 0.0
        
        # Calculate lot size
        lot_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Apply minimum lot size
        lot_size = max(lot_size, self.settings.min_lot_size)
        
        # Round to broker precision (assuming 0.01)
        lot_size = round(lot_size, 2)
        
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
        if current_risk >= self.settings.max_account_risk_percent:
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
    
    def update_position_prices(self, symbol: str, current_price: float):
        """
        Update position with current market price for monitoring
        """
        if symbol not in self.positions or self.positions[symbol].status != 'open':
            return
        
        position = self.positions[symbol]
        
        # Check for stop loss or take profit
        if position.side == 'long':
            if current_price <= position.stop_loss:
                self.close_position(symbol, current_price, "stop_loss")
            elif current_price >= position.take_profit:
                self.close_position(symbol, current_price, "take_profit")
        else:
            if current_price >= position.stop_loss:
                self.close_position(symbol, current_price, "stop_loss")
            elif current_price <= position.take_profit:
                self.close_position(symbol, current_price, "take_profit")
    
    def _calculate_current_risk(self) -> float:
        """Calculate current account risk percentage"""
        if not self.account_info:
            return 0.0
        
        total_risk = sum(p.risk_amount for p in self.positions.values() if p.status == 'open')
        return (total_risk / self.account_info.equity) * 100.0
    
    def _get_pip_value(self, symbol: str, price: float) -> float:
        """Get pip value for symbol"""
        # Simplified pip value calculation
        # In production, this should use exact broker pip values
        if 'USD' in symbol:
            return 10.0  # $10 per pip for standard lot
        else:
            return 10.0  # Placeholder
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol"""
        # Simplified pip size calculation
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
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
