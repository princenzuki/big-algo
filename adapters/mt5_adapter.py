"""
MetaTrader 5 Adapter

Implements the broker interface for MetaTrader 5.
Handles all MT5-specific operations and deviation logging.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import MetaTrader5 as mt5
from .broker_base import (
    BrokerAdapter, AccountInfo, SymbolInfo, OrderRequest, OrderResult, Position
)

logger = logging.getLogger(__name__)

class MT5Adapter(BrokerAdapter):
    """
    MetaTrader 5 broker adapter
    
    Implements all broker operations for MT5 with deviation logging.
    """
    
    def __init__(self, login: int = None, password: str = None, server: str = None):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        
        # Deviation tracking
        self.deviations = []
    
    def connect(self) -> bool:
        """Connect to existing MT5 terminal"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Check if terminal is already logged in (no need for new login)
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("MT5 terminal not logged in, but continuing with limited functionality")
                # Don't return False - allow API to work without full MT5 connection
            else:
                logger.info(f"Connected to existing MT5 terminal (Account: {account_info.login})")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from MT5 terminal"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5 terminal")
            return True
        except Exception as e:
            logger.error(f"MT5 disconnection error: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to MT5"""
        return self.connected and mt5.terminal_info() is not None
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get MT5 account information"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None
            
            return AccountInfo(
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.margin_free,
                currency=account_info.currency,
                leverage=account_info.leverage,
                server=account_info.server
            )
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get MT5 symbol information"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found: {mt5.last_error()}")
                return None
            
            # Calculate current spread
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick for {symbol}: {mt5.last_error()}")
                return None
            
            spread = tick.ask - tick.bid
            
            return SymbolInfo(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                spread=spread,
                point=symbol_info.point,
                digits=symbol_info.digits,
                lot_min=symbol_info.volume_min,
                lot_max=symbol_info.volume_max,
                lot_step=symbol_info.volume_step,
                margin_required=symbol_info.margin_initial
            )
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            raise  # Re-raise the exception so it can be caught by place_order
    
    def get_symbol_point(self, symbol: str) -> float:
        """Get symbol point value for accurate price calculations"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol {symbol} not found: {mt5.last_error()}")
                return 0.0001  # Default fallback
            return symbol_info.point
        except Exception as e:
            logger.error(f"Error getting symbol point for {symbol}: {e}")
            return 0.0001  # Default fallback
    
    def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place order in MT5
        
        Logs any deviations from requested parameters.
        """
        try:
            # Get symbol info first
            try:
                symbol_info = self.get_symbol_info(request.symbol)
                if not symbol_info:
                    return OrderResult(
                        success=False,
                        order_id=None,
                        error_code=None,
                        error_message=f"Symbol {request.symbol} not found",
                        price=None,
                        deviation=None
                    )
            except Exception as e:
                return OrderResult(
                    success=False,
                    order_id=None,
                    error_code=None,
                    error_message=str(e),
                    price=None,
                    deviation=None
                )
            
            # Round lot size to broker precision
            original_lot_size = request.lot_size
            rounded_lot_size = self.round_lot_size(request.lot_size, request.symbol)
            
            # Use current price for market orders
            if request.price is None:
                if request.order_type == 'buy':
                    request.price = symbol_info.ask
                else:
                    request.price = symbol_info.bid
            
            # Check and adjust stop levels to meet broker minimum distance requirements
            adjusted_request = self._adjust_stop_levels(request, symbol_info)
            
            # Validate request with rounded lot size
            request_copy = adjusted_request
            request_copy.lot_size = rounded_lot_size
            is_valid, error_msg = self.validate_order_request(request_copy)
            if not is_valid:
                return OrderResult(
                    success=False,
                    order_id=None,
                    error_code=None,
                    error_message=error_msg,
                    price=None,
                    deviation=None
                )
            
            # Log deviation if lot size was rounded
            if abs(rounded_lot_size - original_lot_size) > 0.001:
                deviation = {
                    'type': 'DEVIATION_BROKER_CONSTRAINT',
                    'parameter': 'lot_size',
                    'requested': original_lot_size,
                    'actual': rounded_lot_size,
                    'reason': 'Broker lot step constraint',
                    'timestamp': datetime.now().isoformat()
                }
                self.deviations.append(deviation)
                logger.warning(f"Lot size deviation: {original_lot_size} -> {rounded_lot_size}")
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if request.order_type == 'buy' else mt5.ORDER_TYPE_SELL
            
            # Prepare order structure
            order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": adjusted_request.symbol,
                "volume": rounded_lot_size,
                "type": order_type,
                "price": adjusted_request.price,
                "deviation": 20,  # 20 points deviation allowed
                "magic": 12345,   # Magic number for identification
                "comment": adjusted_request.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if specified
            if adjusted_request.stop_loss:
                order_request["sl"] = adjusted_request.stop_loss
            if adjusted_request.take_profit:
                order_request["tp"] = adjusted_request.take_profit
            
            # Send order
            result = mt5.order_send(order_request)
            
            if result is None:
                return OrderResult(
                    success=False,
                    order_id=None,
                    error_code=None,
                    error_message="Order send failed - no result",
                    price=None,
                    deviation=None
                )
            
            # Check result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return OrderResult(
                    success=False,
                    order_id=result.order,
                    error_code=result.retcode,
                    error_message=result.comment,
                    price=result.price,
                    deviation=None
                )
            
            # Log successful order
            logger.info(f"Order placed successfully: {request.symbol} {request.order_type} "
                       f"{rounded_lot_size} lots @ {result.price}")
            
            return OrderResult(
                success=True,
                order_id=result.order,
                error_code=None,
                error_message=None,
                price=result.price,
                deviation=None
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                error_code=None,
                error_message=str(e),
                price=None,
                deviation=None
            )
    
    def close_position(self, ticket: int) -> bool:
        """Close position in MT5"""
        try:
            # Get position info
            position = self.get_position(ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.type == 'buy' else mt5.ORDER_TYPE_BUY
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "deviation": 20,
                "magic": 12345,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(close_request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close position {ticket}: {result.comment if result else 'No result'}")
                return False
            
            logger.info(f"Position {ticket} closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def modify_position(self, ticket: int, stop_loss: Optional[float] = None, 
                       take_profit: Optional[float] = None) -> bool:
        """Modify position stop loss or take profit"""
        try:
            # Get position info
            position = self.get_position(ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            # Prepare modify request
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": stop_loss if stop_loss is not None else position.stop_loss,
                "tp": take_profit if take_profit is not None else position.take_profit,
            }
            
            # Send modify request
            result = mt5.order_send(modify_request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to modify position {ticket}: {result.comment if result else 'No result'}")
                return False
            
            logger.info(f"Position {ticket} modified: SL={stop_loss}, TP={take_profit}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying position {ticket}: {e}")
            return False
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append(Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type='buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    profit=pos.profit,
                    swap=pos.swap,
                    time=datetime.fromtimestamp(pos.time),
                    comment=pos.comment
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, ticket: int) -> Optional[Position]:
        """Get specific position"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return None
            
            pos = positions[0]
            return Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                type='buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                volume=pos.volume,
                price_open=pos.price_open,
                price_current=pos.price_current,
                stop_loss=pos.sl,
                take_profit=pos.tp,
                profit=pos.profit,
                swap=pos.swap,
                time=datetime.fromtimestamp(pos.time),
                comment=pos.comment
            )
            
        except Exception as e:
            logger.error(f"Error getting position {ticket}: {e}")
            return None
    
    def get_rates(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """Get historical rates"""
        try:
            # Convert timeframe string to MT5 constant
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            if timeframe not in tf_map:
                logger.error(f"Unsupported timeframe: {timeframe}")
                return []
            
            rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
            if rates is None:
                logger.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
                return []
            
            result = []
            for rate in rates:
                result.append({
                    'time': datetime.fromtimestamp(rate['time']),
                    'open': rate['open'],
                    'high': rate['high'],
                    'low': rate['low'],
                    'close': rate['close'],
                    'volume': rate['tick_volume']
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting rates for {symbol}: {e}")
            return []
    
    def get_current_rates(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get current rates for symbols"""
        try:
            result = {}
            for symbol in symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    result[symbol] = {
                        'time': datetime.fromtimestamp(tick.time),
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'last': tick.last,
                        'volume': tick.volume
                    }
                else:
                    logger.warning(f"Failed to get tick for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting current rates: {e}")
            return {}
    
    def get_deviations(self) -> List[Dict]:
        """Get list of deviations from requested parameters"""
        return self.deviations.copy()
    
    def clear_deviations(self):
        """Clear deviation history"""
        self.deviations.clear()
        logger.info("Deviation history cleared")
    
    def _adjust_stop_levels(self, request, symbol_info) -> 'OrderRequest':
        """
        Adjust stop loss and take profit levels to meet broker minimum distance requirements
        
        Args:
            request: Original order request
            symbol_info: MT5 symbol information
            
        Returns:
            Adjusted order request with valid stop levels
        """
        from adapters.broker_base import OrderRequest
        
        # Create a copy of the request
        adjusted_request = OrderRequest(
            symbol=request.symbol,
            order_type=request.order_type,
            lot_size=request.lot_size,
            price=request.price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=request.comment
        )
        
        # Get minimum stop level from symbol info
        # Try different possible attribute names for minimum stop level
        min_stop_level = 0
        if hasattr(symbol_info, 'trade_stops_level'):
            min_stop_level = symbol_info.trade_stops_level
            logger.debug(f"[STOP_ADJUST] Using trade_stops_level: {min_stop_level}")
        elif hasattr(symbol_info, 'stops_level'):
            min_stop_level = symbol_info.stops_level
            logger.debug(f"[STOP_ADJUST] Using stops_level: {min_stop_level}")
        else:
            # Use a default minimum stop level (e.g., 10 points)
            min_stop_level = 10
            logger.debug(f"[STOP_ADJUST] Using default stops_level: {min_stop_level}")
            
        point = symbol_info.point
        
        if min_stop_level <= 0:
            # Use a default minimum stop level
            min_stop_level = 10
        
        # Calculate minimum distance in price units
        min_distance = min_stop_level * point
        logger.debug(f"[STOP_ADJUST] Min stop level: {min_stop_level}, point: {point}, min_distance: {min_distance}")
        
        # Check if price is valid before adjusting stops
        if adjusted_request.price is None:
            logger.warning(f"[STOP_ADJUST] Cannot adjust stops for {request.symbol}: price is None")
            return adjusted_request
        
        # Adjust stop loss if needed
        if adjusted_request.stop_loss is not None:
            logger.debug(f"[STOP_ADJUST] Checking SL: {adjusted_request.stop_loss}, order_type: {adjusted_request.order_type}")
            if adjusted_request.order_type == 'buy':
                # For buy orders, SL should be below entry price
                min_sl = adjusted_request.price - min_distance
                logger.debug(f"[STOP_ADJUST] Buy order - min_sl: {min_sl}, current_sl: {adjusted_request.stop_loss}")
                if adjusted_request.stop_loss > min_sl:
                    adjusted_request.stop_loss = min_sl
                    logger.info(f"[STOP_ADJUST] SL adjusted for {request.symbol}: {request.stop_loss:.5f} -> {min_sl:.5f} (min distance: {min_stop_level} points)")
                else:
                    logger.debug(f"[STOP_ADJUST] SL already far enough: {adjusted_request.stop_loss} <= {min_sl}")
            else:
                # For sell orders, SL should be above entry price
                max_sl = adjusted_request.price + min_distance
                logger.debug(f"[STOP_ADJUST] Sell order - max_sl: {max_sl}, current_sl: {adjusted_request.stop_loss}")
                if adjusted_request.stop_loss < max_sl:
                    adjusted_request.stop_loss = max_sl
                    logger.info(f"[STOP_ADJUST] SL adjusted for {request.symbol}: {request.stop_loss:.5f} -> {max_sl:.5f} (min distance: {min_stop_level} points)")
                else:
                    logger.debug(f"[STOP_ADJUST] SL already far enough: {adjusted_request.stop_loss} >= {max_sl}")
        
        # Adjust take profit if needed
        if adjusted_request.take_profit is not None:
            if adjusted_request.order_type == 'buy':
                # For buy orders, TP should be above entry price
                min_tp = adjusted_request.price + min_distance
                if adjusted_request.take_profit < min_tp:
                    adjusted_request.take_profit = min_tp
                    logger.info(f"[STOP_ADJUST] TP adjusted for {request.symbol}: {request.take_profit:.5f} -> {min_tp:.5f} (min distance: {min_stop_level} points)")
            else:
                # For sell orders, TP should be below entry price
                max_tp = adjusted_request.price - min_distance
                if adjusted_request.take_profit > max_tp:
                    adjusted_request.take_profit = max_tp
                    logger.info(f"[STOP_ADJUST] TP adjusted for {request.symbol}: {request.take_profit:.5f} -> {max_tp:.5f} (min distance: {min_stop_level} points)")
        
        return adjusted_request
    
    def validate_lot_size(self, lot_size: float, symbol_info: SymbolInfo) -> bool:
        """Validate lot size against symbol limits"""
        if not symbol_info:
            logger.warning(f"Symbol info not available for validation")
            return False
            
        # Check if lot size is within bounds
        if not (symbol_info.lot_min <= lot_size <= symbol_info.lot_max):
            return False
        
        # Check if lot size aligns with step (handle floating point precision)
        steps = lot_size / symbol_info.lot_step
        return abs(steps - round(steps)) < 1e-10
    
    def close_position_partial(self, ticket: int, partial_lot_size: float) -> bool:
        """
        Close a portion of a position in MT5
        
        Args:
            ticket: Position ticket
            partial_lot_size: Lot size to close
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get position information
            position = mt5.positions_get(ticket=ticket)
            if not position or len(position) == 0:
                logger.error(f"Position with ticket {ticket} not found")
                return False
            
            position = position[0]
            
            # Validate partial lot size
            if partial_lot_size >= position.volume:
                logger.error(f"Partial lot size {partial_lot_size} >= position volume {position.volume}")
                return False
            
            # Create close request for partial position
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": partial_lot_size,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": position.magic,
                "comment": f"Partial close {partial_lot_size} lots",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(close_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close partial position: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"✅ Partial position closed: {partial_lot_size} lots of ticket {ticket}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing partial position {ticket}: {e}")
            return False