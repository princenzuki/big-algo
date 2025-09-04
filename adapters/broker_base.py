"""
Base broker interface

Defines the standard interface that all broker adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AccountInfo:
    """Account information"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str
    leverage: int
    server: str

@dataclass
class SymbolInfo:
    """Symbol information"""
    symbol: str
    bid: float
    ask: float
    spread: float
    point: float
    digits: int
    lot_min: float
    lot_max: float
    lot_step: float
    margin_required: float

@dataclass
class OrderRequest:
    """Order request"""
    symbol: str
    order_type: str  # 'buy', 'sell'
    lot_size: float
    price: Optional[float] = None  # None for market orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "Lorentzian ML"

@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[int]
    error_code: Optional[int]
    error_message: Optional[str]
    price: Optional[float]
    deviation: Optional[float]

@dataclass
class Position:
    """Open position"""
    ticket: int
    symbol: str
    type: str  # 'buy', 'sell'
    volume: float
    price_open: float
    price_current: float
    stop_loss: float
    take_profit: float
    profit: float
    swap: float
    time: datetime
    comment: str

class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters
    
    All broker implementations must inherit from this class and implement
    all abstract methods.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from broker
        
        Returns:
            True if disconnection successful
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to broker
        
        Returns:
            True if connected
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account information
        
        Returns:
            AccountInfo object or None if failed
        """
        pass
    
    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get symbol information
        
        Args:
            symbol: Trading symbol
            
        Returns:
            SymbolInfo object or None if failed
        """
        pass
    
    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place an order
        
        Args:
            request: Order request
            
        Returns:
            OrderResult object
        """
        pass
    
    @abstractmethod
    def close_position(self, ticket: int) -> bool:
        """
        Close a position
        
        Args:
            ticket: Position ticket number
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def modify_position(self, ticket: int, stop_loss: Optional[float] = None, 
                       take_profit: Optional[float] = None) -> bool:
        """
        Modify position stop loss or take profit
        
        Args:
            ticket: Position ticket number
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all open positions
        
        Returns:
            List of Position objects
        """
        pass
    
    @abstractmethod
    def get_position(self, ticket: int) -> Optional[Position]:
        """
        Get specific position
        
        Args:
            ticket: Position ticket number
            
        Returns:
            Position object or None if not found
        """
        pass
    
    @abstractmethod
    def get_rates(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """
        Get historical rates
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            count: Number of bars
            
        Returns:
            List of rate dictionaries with OHLC data
        """
        pass
    
    @abstractmethod
    def get_current_rates(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current rates for symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbol to rate data
        """
        pass
    
    def validate_order_request(self, request: OrderRequest) -> Tuple[bool, str]:
        """
        Validate order request
        
        Args:
            request: Order request to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation
        if not request.symbol:
            return False, "Symbol is required"
        
        if request.lot_size <= 0:
            return False, "Lot size must be positive"
        
        if request.order_type not in ['buy', 'sell']:
            return False, "Order type must be 'buy' or 'sell'"
        
        # Get symbol info for additional validation
        symbol_info = self.get_symbol_info(request.symbol)
        if not symbol_info:
            return False, f"Symbol {request.symbol} not found"
        
        # Validate lot size
        if request.lot_size < symbol_info.lot_min:
            return False, f"Lot size {request.lot_size} below minimum {symbol_info.lot_min}"
        
        if request.lot_size > symbol_info.lot_max:
            return False, f"Lot size {request.lot_size} above maximum {symbol_info.lot_max}"
        
        # Validate lot step
        if (request.lot_size - symbol_info.lot_min) % symbol_info.lot_step != 0:
            return False, f"Lot size {request.lot_size} not aligned with step {symbol_info.lot_step}"
        
        return True, "OK"
    
    def round_lot_size(self, lot_size: float, symbol: str) -> float:
        """
        Round lot size to broker precision
        
        Args:
            lot_size: Desired lot size
            symbol: Trading symbol
            
        Returns:
            Rounded lot size
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"Symbol info not available for {symbol}, using fallback rounding")
            return round(lot_size, 2)  # Fallback to 0.01 precision
        
        # Ensure lot size is not negative
        if lot_size <= 0:
            logger.warning(f"Invalid lot size {lot_size} for {symbol}, using minimum")
            return symbol_info.lot_min
        
        # Round to lot step
        steps = round((lot_size - symbol_info.lot_min) / symbol_info.lot_step)
        rounded = symbol_info.lot_min + (steps * symbol_info.lot_step)
        
        # Ensure within bounds
        rounded = max(rounded, symbol_info.lot_min)
        rounded = min(rounded, symbol_info.lot_max)
        
        # Log if significant rounding occurred
        if abs(rounded - lot_size) > symbol_info.lot_step * 0.5:
            logger.debug(f"Lot size rounded for {symbol}: {lot_size:.3f} -> {rounded:.3f} "
                        f"(min: {symbol_info.lot_min:.3f}, step: {symbol_info.lot_step:.3f})")
        
        return rounded
    
    def calculate_spread_pips(self, symbol: str) -> float:
        """
        Calculate spread in pips
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Spread in pips
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return 0.0
        
        spread_points = symbol_info.ask - symbol_info.bid
        spread_pips = spread_points / symbol_info.point
        
        return spread_pips
