"""
Structured logging configuration

Provides structured JSON logging for the trading bot.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import structlog

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup structured logging for the trading bot
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return structlog.get_logger()

class TradingLogger:
    """
    Specialized logger for trading operations
    
    Provides structured logging with trading-specific context.
    """
    
    def __init__(self, name: str = "trading_bot"):
        self.logger = structlog.get_logger(name)
    
    def log_signal(self, symbol: str, signal: int, confidence: float, 
                   prediction: float, features: Dict[str, float]):
        """Log ML signal generation"""
        self.logger.info(
            "ML signal generated",
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            prediction=prediction,
            features=features,
            event_type="signal_generated"
        )
    
    def log_trade_opened(self, symbol: str, side: str, lot_size: float, 
                        entry_price: float, stop_loss: float, take_profit: float,
                        confidence: float, risk_amount: float):
        """Log trade opening"""
        self.logger.info(
            "Trade opened",
            symbol=symbol,
            side=side,
            lot_size=lot_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_amount=risk_amount,
            event_type="trade_opened"
        )
    
    def log_trade_closed(self, symbol: str, side: str, pnl: float, 
                        exit_reason: str, duration_hours: float):
        """Log trade closing"""
        self.logger.info(
            "Trade closed",
            symbol=symbol,
            side=side,
            pnl=pnl,
            exit_reason=exit_reason,
            duration_hours=duration_hours,
            event_type="trade_closed"
        )
    
    def log_risk_event(self, event_type: str, symbol: str, reason: str, 
                      current_risk: float, max_risk: float):
        """Log risk management events"""
        self.logger.warning(
            "Risk event",
            event_type=event_type,
            symbol=symbol,
            reason=reason,
            current_risk=current_risk,
            max_risk=max_risk,
            event_type="risk_event"
        )
    
    def log_session_event(self, event_type: str, symbol: str, session: str, 
                         can_trade: bool, reason: str):
        """Log session management events"""
        self.logger.info(
            "Session event",
            event_type=event_type,
            symbol=symbol,
            session=session,
            can_trade=can_trade,
            reason=reason,
            event_type="session_event"
        )
    
    def log_broker_event(self, event_type: str, symbol: str, order_id: Optional[int],
                        success: bool, error_message: Optional[str] = None):
        """Log broker interaction events"""
        level = "info" if success else "error"
        self.logger.log(
            level,
            "Broker event",
            event_type=event_type,
            symbol=symbol,
            order_id=order_id,
            success=success,
            error_message=error_message,
            event_type="broker_event"
        )
    
    def log_deviation(self, deviation_type: str, parameter: str, requested: Any,
                     actual: Any, reason: str):
        """Log deviations from requested parameters"""
        self.logger.warning(
            "Parameter deviation",
            deviation_type=deviation_type,
            parameter=parameter,
            requested=requested,
            actual=actual,
            reason=reason,
            event_type="deviation"
        )
    
    def log_parity_test(self, test_name: str, pine_value: float, python_value: float,
                       delta: float, passed: bool):
        """Log parity test results"""
        level = "info" if passed else "error"
        self.logger.log(
            level,
            "Parity test",
            test_name=test_name,
            pine_value=pine_value,
            python_value=python_value,
            delta=delta,
            passed=passed,
            event_type="parity_test"
        )
    
    def log_system_event(self, event_type: str, message: str, **kwargs):
        """Log system events"""
        self.logger.info(
            "System event",
            event_type=event_type,
            message=message,
            **kwargs,
            event_type="system_event"
        )
    
    def log_error(self, error_type: str, message: str, symbol: Optional[str] = None,
                 error_code: Optional[str] = None, **kwargs):
        """Log errors with context"""
        self.logger.error(
            "Error occurred",
            error_type=error_type,
            message=message,
            symbol=symbol,
            error_code=error_code,
            **kwargs,
            event_type="error"
        )

# Global logger instance
trading_logger = TradingLogger()
