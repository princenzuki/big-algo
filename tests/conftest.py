"""
Pytest configuration and shared fixtures for MT5 trading bot tests
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_historical_data():
    """Create mock historical data for testing"""
    np.random.seed(42)  # For reproducible tests
    data = []
    base_price = 1.2000
    
    for i in range(100):
        # Simulate realistic price movement
        price_change = np.random.normal(0, 0.001)
        base_price += price_change
        
        high = base_price + abs(np.random.normal(0, 0.0005))
        low = base_price - abs(np.random.normal(0, 0.0005))
        open_price = base_price + np.random.normal(0, 0.0002)
        close = base_price + np.random.normal(0, 0.0003)
        
        data.append({
            'time': i,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    return data

@pytest.fixture
def mock_symbol_info():
    """Create mock symbol info for testing"""
    return MagicMock(
        symbol="EURUSD",
        point=0.00001,
        trade_tick_value=1.0,
        trade_tick_size=0.00001,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        trade_stops_level=10,
        ask=1.2000,
        bid=1.1999
    )

@pytest.fixture
def mock_account_info():
    """Create mock account info for testing"""
    return MagicMock(
        balance=10000.0,
        equity=10000.0,
        free_margin=10000.0,
        margin_level=1000.0
    )

@pytest.fixture
def mock_risk_settings():
    """Create mock risk settings for testing"""
    from core.risk import RiskSettings
    
    return RiskSettings(
        max_risk_per_trade=0.02,  # 2%
        max_daily_risk=0.05,      # 5%
        max_total_risk=0.10,      # 10%
        cooldown_minutes=30,
        one_trade_per_symbol=True
    )

@pytest.fixture
def mock_smart_tp_config():
    """Create mock Smart TP config for testing"""
    from core.smart_tp import SmartTPConfig
    
    return SmartTPConfig(
        atr_period=14,
        baseline_tp_multiplier=2.0,
        strong_trend_multiplier=2.5,
        weak_trend_multiplier=1.0,
        partial_tp_ratio=0.5,
        partial_tp_rr=1.5,
        adx_period=14,
        adx_strong_threshold=25.0,
        cci_period=14,
        cci_strong_threshold=100.0,
        cci_weak_threshold=-100.0
    )

@pytest.fixture
def mock_mt5_success_response():
    """Create mock successful MT5 response"""
    response = MagicMock()
    response.retcode = 10009  # TRADE_RETCODE_DONE
    response.order = 12345
    response.price = 1.2000
    response.volume = 0.1
    response.comment = "Test Order"
    response.request_id = 1
    return response

@pytest.fixture
def mock_mt5_failure_response():
    """Create mock failed MT5 response"""
    response = MagicMock()
    response.retcode = 10004  # TRADE_RETCODE_REJECT
    response.order = 0
    response.price = 0.0
    response.volume = 0.0
    response.comment = "Order rejected"
    response.request_id = 1
    return response

@pytest.fixture
def mock_position():
    """Create mock position for testing"""
    from core.risk import Position
    
    return Position(
        symbol="EURUSD",
        side="buy",
        entry_price=1.2000,
        lot_size=0.1,
        stop_loss=1.1950,
        take_profit=1.2050,
        confidence=0.8,
        risk_amount=20.0,
        trailing_stop_price=None,
        trailing_stop_distance=None,
        break_even_triggered=False
    )

@pytest.fixture
def mock_order_request():
    """Create mock order request for testing"""
    from adapters.mt5_adapter import OrderRequest
    
    return OrderRequest(
        symbol="EURUSD",
        side="buy",
        lot_size=0.1,
        price=1.2000,
        stop_loss=1.1950,
        take_profit=1.2050,
        comment="Test Order"
    )

@pytest.fixture
def mock_smart_tp_result():
    """Create mock Smart TP result for testing"""
    from core.smart_tp import SmartTPResult
    
    return SmartTPResult(
        partial_tp_price=1.2015,
        full_tp_price=1.2030,
        momentum_strength='strong',
        tp_multiplier=2.5,
        should_take_partial=True,
        trailing_enabled=True
    )

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit marker to all tests in test_* files
        if "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might be slow
        if "integration" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.integration)
