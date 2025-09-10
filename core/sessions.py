"""
Session Management Module

Handles trading sessions, weekend blocking, and timezone management.
All time logic uses Africa/Nairobi timezone as specified.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import pytz

logger = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    """Session configuration for a symbol"""
    symbol: str
    allow_weekend: bool = False  # Only BTCUSD should be True
    london_session: bool = True
    new_york_session: bool = True
    asia_session: bool = True
    custom_hours: Optional[Tuple[time, time]] = None  # (start, end) in Nairobi time

class SessionManager:
    """
    Manages trading sessions and weekend blocking
    
    Implements the weekend block rule: No trading (except BTCUSD) 
    from Friday 23:55 to Monday 00:05 Kenya time.
    """
    
    def __init__(self):
        # Africa/Nairobi timezone
        self.timezone = pytz.timezone('Africa/Nairobi')
        
        # Weekend block times (Nairobi time)
        self.weekend_start = time(23, 55)  # Friday 23:55
        self.weekend_end = time(0, 5)      # Monday 00:05
        
        # Session times (Nairobi time)
        self.london_start = time(10, 0)    # 10:00 AM
        self.london_end = time(19, 0)      # 7:00 PM
        self.new_york_start = time(15, 0)  # 3:00 PM
        self.new_york_end = time(23, 59)   # 11:59 PM (spans until midnight)
        self.asia_start = time(2, 0)       # 2:00 AM
        self.asia_end = time(11, 0)        # 11:00 AM
        
        # Default session configs
        self.session_configs: Dict[str, SessionConfig] = {
            'BTCUSD': SessionConfig('BTCUSD', allow_weekend=True),
            'EURUSD': SessionConfig('EURUSD'),
            'GBPUSD': SessionConfig('GBPUSD'),
            'USDJPY': SessionConfig('USDJPY'),
            'AUDUSD': SessionConfig('AUDUSD'),
            'USDCAD': SessionConfig('USDCAD'),
            'NZDUSD': SessionConfig('NZDUSD'),
            'USDCHF': SessionConfig('USDCHF'),
        }
    
    def get_current_time(self) -> datetime:
        """Get current time in Nairobi timezone"""
        return datetime.now(self.timezone)
    
    def is_weekend_blocked(self, symbol: str, symbol_config: Dict = None) -> Tuple[bool, str]:
        """
        Check if trading is blocked due to weekend rules
        
        Args:
            symbol: Trading symbol
            symbol_config: Symbol configuration from YAML (optional)
            
        Returns:
            Tuple of (is_blocked, reason)
        """
        # 24/7 TRADING ENABLED - No weekend restrictions
        # All symbols can trade 24/7 including weekends
        return False, "OK"
    
    def get_current_session(self) -> str:
        """
        Get current trading session
        
        Returns:
            Session name: 'london', 'new_york', 'asia', or 'closed'
        """
        current_time = self.get_current_time()
        current_time_only = current_time.time()
        
        # Check London session
        if self.london_start <= current_time_only < self.london_end:
            return 'london'
        
        # Check New York session (3:00 PM - 11:59 PM)
        if self.new_york_start <= current_time_only <= self.new_york_end:
            return 'new_york'
        
        # Check Asia session
        if self.asia_start <= current_time_only < self.asia_end:
            return 'asia'
        
        return 'closed'
    
    def is_session_allowed(self, symbol: str, session: str) -> bool:
        """
        Check if symbol is allowed to trade in current session
        
        Args:
            symbol: Trading symbol
            session: Session name
            
        Returns:
            True if trading is allowed
        """
        if symbol not in self.session_configs:
            # Default to allowing all sessions for unknown symbols
            return True
        
        config = self.session_configs[symbol]
        
        if session == 'london':
            return config.london_session
        elif session == 'new_york':
            return config.new_york_session
        elif session == 'asia':
            return config.asia_session
        else:
            return False
    
    def can_trade_symbol(self, symbol: str, symbol_config: Dict = None) -> Tuple[bool, str]:
        """
        Check if symbol can be traded right now
        
        Args:
            symbol: Trading symbol
            symbol_config: Symbol configuration from YAML (optional)
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # 24/7 TRADING ENABLED - No session restrictions
        # Only check weekend blocking for non-crypto symbols
        is_blocked, weekend_reason = self.is_weekend_blocked(symbol, symbol_config)
        if is_blocked:
            return False, weekend_reason
        
        # All other checks removed - bot runs 24/7
        return True, "OK"
    
    def get_session_info(self) -> Dict[str, any]:
        """
        Get current session information
        
        Returns:
            Dictionary with session details
        """
        current_time = self.get_current_time()
        current_session = self.get_current_session()
        
        return {
            "current_time": current_time.isoformat(),
            "timezone": "Africa/Nairobi",
            "current_session": current_session,
            "weekday": current_time.strftime("%A"),
            "is_weekend": current_time.weekday() >= 5,
            "session_times": {
                "london": f"{self.london_start.strftime('%H:%M')} - {self.london_end.strftime('%H:%M')}",
                "new_york": f"{self.new_york_start.strftime('%H:%M')} - {self.new_york_end.strftime('%H:%M')}",
                "asia": f"{self.asia_start.strftime('%H:%M')} - {self.asia_end.strftime('%H:%M')}"
            },
            "weekend_block": {
                "start": self.weekend_start.strftime('%H:%M'),
                "end": self.weekend_end.strftime('%H:%M'),
                "applies_to": "All symbols except BTCUSD"
            }
        }
    
    def get_symbol_session_status(self, symbol: str) -> Dict[str, any]:
        """
        Get session status for a specific symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol session details
        """
        can_trade, reason = self.can_trade_symbol(symbol)
        current_session = self.get_current_session()
        
        config = self.session_configs.get(symbol, SessionConfig(symbol))
        
        return {
            "symbol": symbol,
            "can_trade": can_trade,
            "reason": reason,
            "current_session": current_session,
            "allow_weekend": config.allow_weekend,
            "allowed_sessions": {
                "london": config.london_session,
                "new_york": config.new_york_session,
                "asia": config.asia_session
            }
        }
    
    def update_session_config(self, symbol: str, config: SessionConfig):
        """
        Update session configuration for a symbol
        
        Args:
            symbol: Trading symbol
            config: New session configuration
        """
        self.session_configs[symbol] = config
        logger.info(f"Updated session config for {symbol}: weekend={config.allow_weekend}")
    
    def get_next_session_change(self) -> Dict[str, any]:
        """
        Get information about when the next session change will occur
        
        Returns:
            Dictionary with next session change details
        """
        current_time = self.get_current_time()
        current_time_only = current_time.time()
        
        # Calculate next session change
        next_change = None
        next_session = None
        
        # Check London session
        if current_time_only < self.london_start:
            next_change = current_time.replace(
                hour=self.london_start.hour,
                minute=self.london_start.minute,
                second=0,
                microsecond=0
            )
            next_session = 'london'
        elif current_time_only < self.london_end:
            next_change = current_time.replace(
                hour=self.london_end.hour,
                minute=self.london_end.minute,
                second=0,
                microsecond=0
            )
            next_session = 'new_york'
        elif current_time_only < self.new_york_start:
            next_change = current_time.replace(
                hour=self.new_york_start.hour,
                minute=self.new_york_start.minute,
                second=0,
                microsecond=0
            )
            next_session = 'new_york'
        elif current_time_only < time(0, 0):
            # New York session ends at midnight
            next_change = current_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            next_session = 'asia'
        else:
            # Asia session
            if current_time_only < self.asia_start:
                next_change = current_time.replace(
                    hour=self.asia_start.hour,
                    minute=self.asia_start.minute,
                    second=0,
                    microsecond=0
                )
                next_session = 'asia'
            elif current_time_only < self.asia_end:
                next_change = current_time.replace(
                    hour=self.asia_end.hour,
                    minute=self.asia_end.minute,
                    second=0,
                    microsecond=0
                )
                next_session = 'london'
            else:
                # Next day London session
                next_change = current_time.replace(
                    hour=self.london_start.hour,
                    minute=self.london_start.minute,
                    second=0,
                    microsecond=0
                ) + timedelta(days=1)
                next_session = 'london'
        
        if next_change:
            time_until = (next_change - current_time).total_seconds() / 60  # minutes
            
            return {
                "next_session": next_session,
                "next_change_time": next_change.isoformat(),
                "minutes_until": time_until,
                "hours_until": time_until / 60
            }
        
        return {"error": "Could not calculate next session change"}
    
    def get_weekend_block_info(self) -> Dict[str, any]:
        """
        Get weekend block information
        
        Returns:
            Dictionary with weekend block details
        """
        current_time = self.get_current_time()
        current_weekday = current_time.weekday()
        current_time_only = current_time.time()
        
        # Calculate next weekend block
        days_until_friday = (4 - current_weekday) % 7
        if days_until_friday == 0 and current_time_only >= self.weekend_start:
            days_until_friday = 7  # Next Friday
        
        next_weekend_start = current_time.replace(
            hour=self.weekend_start.hour,
            minute=self.weekend_start.minute,
            second=0,
            microsecond=0
        ) + timedelta(days=days_until_friday)
        
        # Calculate weekend block duration
        weekend_duration = timedelta(days=2, hours=0, minutes=10)  # Fri 23:55 to Sun 00:05
        
        return {
            "current_weekday": current_time.strftime("%A"),
            "is_weekend_blocked": self.is_weekend_blocked("EURUSD")[0],  # Use EURUSD as example
            "next_weekend_start": next_weekend_start.isoformat(),
            "weekend_duration_hours": weekend_duration.total_seconds() / 3600,
            "weekend_start": self.weekend_start.strftime('%H:%M'),
            "weekend_end": self.weekend_end.strftime('%H:%M'),
            "exceptions": ["BTCUSD"]
        }
