"""
Timezone and time utilities

Handles timezone conversions and time-related operations for the trading bot.
All time operations use Africa/Nairobi timezone as specified.
"""

import pytz
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default timezone
DEFAULT_TIMEZONE = pytz.timezone('Africa/Nairobi')

def get_current_time(timezone: Optional[pytz.timezone] = None) -> datetime:
    """
    Get current time in specified timezone
    
    Args:
        timezone: Target timezone (defaults to Africa/Nairobi)
        
    Returns:
        Current datetime in specified timezone
    """
    if timezone is None:
        timezone = DEFAULT_TIMEZONE
    
    return datetime.now(timezone)

def convert_timezone(dt: datetime, from_tz: pytz.timezone, to_tz: pytz.timezone) -> datetime:
    """
    Convert datetime from one timezone to another
    
    Args:
        dt: Datetime to convert
        from_tz: Source timezone
        to_tz: Target timezone
        
    Returns:
        Converted datetime
    """
    if dt.tzinfo is None:
        dt = from_tz.localize(dt)
    
    return dt.astimezone(to_tz)

def to_nairobi_time(dt: datetime) -> datetime:
    """
    Convert datetime to Nairobi timezone
    
    Args:
        dt: Datetime to convert
        
    Returns:
        Datetime in Nairobi timezone
    """
    if dt.tzinfo is None:
        # Assume UTC if no timezone info
        dt = pytz.UTC.localize(dt)
    
    return dt.astimezone(DEFAULT_TIMEZONE)

def from_nairobi_time(dt: datetime, target_tz: pytz.timezone) -> datetime:
    """
    Convert datetime from Nairobi timezone to target timezone
    
    Args:
        dt: Datetime in Nairobi timezone
        target_tz: Target timezone
        
    Returns:
        Datetime in target timezone
    """
    if dt.tzinfo is None:
        dt = DEFAULT_TIMEZONE.localize(dt)
    
    return dt.astimezone(target_tz)

def is_weekend(dt: Optional[datetime] = None) -> bool:
    """
    Check if given datetime is weekend (Saturday or Sunday)
    
    Args:
        dt: Datetime to check (defaults to current time)
        
    Returns:
        True if weekend
    """
    if dt is None:
        dt = get_current_time()
    
    return dt.weekday() >= 5  # 5=Saturday, 6=Sunday

def is_weekend_blocked(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Check if trading is blocked due to weekend rules
    
    Args:
        dt: Datetime to check (defaults to current time)
        
    Returns:
        Tuple of (is_blocked, reason)
    """
    if dt is None:
        dt = get_current_time()
    
    weekday = dt.weekday()
    time_only = dt.time()
    
    # Weekend block: Friday 23:55 to Monday 00:05
    if weekday == 4:  # Friday
        if time_only >= dt.replace(hour=23, minute=55, second=0, microsecond=0).time():
            return True, "WEEKEND_BLOCK_FRIDAY"
    elif weekday == 5:  # Saturday
        return True, "WEEKEND_BLOCK_SATURDAY"
    elif weekday == 6:  # Sunday
        return True, "WEEKEND_BLOCK_SUNDAY"
    elif weekday == 0:  # Monday
        if time_only <= dt.replace(hour=0, minute=5, second=0, microsecond=0).time():
            return True, "WEEKEND_BLOCK_MONDAY"
    
    return False, "OK"

def get_trading_session(dt: Optional[datetime] = None) -> str:
    """
    Get current trading session
    
    Args:
        dt: Datetime to check (defaults to current time)
        
    Returns:
        Session name: 'london', 'new_york', 'asia', or 'closed'
    """
    if dt is None:
        dt = get_current_time()
    
    time_only = dt.time()
    
    # London session: 10:00 - 19:00
    if dt.replace(hour=10, minute=0, second=0, microsecond=0).time() <= time_only < dt.replace(hour=19, minute=0, second=0, microsecond=0).time():
        return 'london'
    
    # New York session: 15:00 - 00:00 (next day)
    if time_only >= dt.replace(hour=15, minute=0, second=0, microsecond=0).time() or time_only < dt.replace(hour=0, minute=0, second=0, microsecond=0).time():
        return 'new_york'
    
    # Asia session: 02:00 - 11:00
    if dt.replace(hour=2, minute=0, second=0, microsecond=0).time() <= time_only < dt.replace(hour=11, minute=0, second=0, microsecond=0).time():
        return 'asia'
    
    return 'closed'

def get_next_session_change(dt: Optional[datetime] = None) -> Tuple[datetime, str]:
    """
    Get when the next session change will occur
    
    Args:
        dt: Current datetime (defaults to current time)
        
    Returns:
        Tuple of (next_change_time, next_session)
    """
    if dt is None:
        dt = get_current_time()
    
    current_session = get_trading_session(dt)
    time_only = dt.time()
    
    # Calculate next session change
    if current_session == 'london':
        # London ends at 19:00, New York starts at 15:00 (already started)
        next_change = dt.replace(hour=19, minute=0, second=0, microsecond=0)
        next_session = 'new_york'
    elif current_session == 'new_york':
        # New York ends at 00:00, Asia starts at 02:00
        next_change = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        next_change = next_change.replace(hour=2, minute=0, second=0, microsecond=0)
        next_session = 'asia'
    elif current_session == 'asia':
        # Asia ends at 11:00, London starts at 10:00 (already started)
        next_change = dt.replace(hour=11, minute=0, second=0, microsecond=0)
        next_session = 'london'
    else:  # closed
        # Next London session
        next_change = dt.replace(hour=10, minute=0, second=0, microsecond=0)
        if time_only >= dt.replace(hour=10, minute=0, second=0, microsecond=0).time():
            next_change += timedelta(days=1)
        next_session = 'london'
    
    return next_change, next_session

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def get_market_hours(session: str) -> Tuple[datetime.time, datetime.time]:
    """
    Get market hours for a trading session
    
    Args:
        session: Session name ('london', 'new_york', 'asia')
        
    Returns:
        Tuple of (start_time, end_time)
    """
    session_hours = {
        'london': (datetime.time(10, 0), datetime.time(19, 0)),
        'new_york': (datetime.time(15, 0), datetime.time(0, 0)),  # Ends at midnight
        'asia': (datetime.time(2, 0), datetime.time(11, 0))
    }
    
    return session_hours.get(session, (datetime.time(0, 0), datetime.time(0, 0)))

def is_market_open(session: str, dt: Optional[datetime] = None) -> bool:
    """
    Check if a specific market session is open
    
    Args:
        session: Session name ('london', 'new_york', 'asia')
        dt: Datetime to check (defaults to current time)
        
    Returns:
        True if market is open
    """
    if dt is None:
        dt = get_current_time()
    
    current_session = get_trading_session(dt)
    return current_session == session

def get_time_until_next_session(dt: Optional[datetime] = None) -> timedelta:
    """
    Get time until next session change
    
    Args:
        dt: Current datetime (defaults to current time)
        
    Returns:
        Time until next session change
    """
    if dt is None:
        dt = get_current_time()
    
    next_change, _ = get_next_session_change(dt)
    return next_change - dt

def log_time_info():
    """Log current time information for debugging"""
    current_time = get_current_time()
    current_session = get_trading_session(current_time)
    is_blocked, reason = is_weekend_blocked(current_time)
    next_change, next_session = get_next_session_change(current_time)
    time_until = get_time_until_next_session(current_time)
    
    logger.info(
        "Time information",
        current_time=current_time.isoformat(),
        current_session=current_session,
        is_weekend_blocked=is_blocked,
        block_reason=reason,
        next_session=next_session,
        next_change=next_change.isoformat(),
        time_until_next=format_duration(time_until.total_seconds())
    )
