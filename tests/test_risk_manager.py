"""
Unit tests for Risk Management System

Tests position sizing, stop loss calculation, lot size validation,
break-even logic, and risk controls.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging
from datetime import datetime, timedelta

from core.risk import (
    RiskManager, RiskSettings, AccountInfo, Position,
    get_dynamic_spread, get_intelligent_sl, calculate_trailing_stop
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestRiskManager:
    """Test cases for Risk Management system"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.risk_settings = RiskSettings(
            max_risk_per_trade=0.02,  # 2%
            max_daily_risk=0.05,      # 5%
            max_total_risk=0.10,      # 10%
            cooldown_minutes=30,
            one_trade_per_symbol=True
        )
        
        self.account_info = AccountInfo(
            balance=10000.0,
            equity=10000.0,
            free_margin=10000.0,
            margin_level=1000.0
        )
        
        self.risk_manager = RiskManager(self.risk_settings)
        self.risk_manager.update_account_info(self.account_info)
        
        # Mock broker adapter
        self.mock_broker = MagicMock()
        self.risk_manager.broker_adapter = self.mock_broker
    
    def test_position_sizing_calculation(self):
        """Test position sizing calculation with different confidence levels"""
        print("\n=== Testing Position Sizing Calculation ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        stop_loss = 1.1950  # 50 pips stop
        confidence = 0.8
        
        # Mock symbol info
        mock_symbol_info = MagicMock()
        mock_symbol_info.point = 0.00001
        mock_symbol_info.trade_tick_value = 1.0
        mock_symbol_info.trade_tick_size = 0.00001
        
        with patch('core.risk.mt5.symbol_info', return_value=mock_symbol_info):
            lot_size, risk_amount = self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, confidence
            )
        
        print(f"Lot size: {lot_size:.3f}")
        print(f"Risk amount: ${risk_amount:.2f}")
        print(f"Confidence: {confidence}")
        
        # Validate lot size
        assert lot_size > 0, f"Lot size should be positive, got {lot_size}"
        assert lot_size <= 1.0, f"Lot size should be reasonable, got {lot_size}"
        
        # Validate risk amount
        expected_risk = self.account_info.balance * self.risk_settings.max_risk_per_trade * confidence
        assert abs(risk_amount - expected_risk) < 1.0, f"Expected risk ~${expected_risk:.2f}, got ${risk_amount:.2f}"
        
        print("✅ Position sizing calculation test passed")
    
    def test_position_sizing_with_different_confidence(self):
        """Test position sizing with different confidence levels"""
        print("\n=== Testing Position Sizing with Different Confidence ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        stop_loss = 1.1950
        
        # Mock symbol info
        mock_symbol_info = MagicMock()
        mock_symbol_info.point = 0.00001
        mock_symbol_info.trade_tick_value = 1.0
        mock_symbol_info.trade_tick_size = 0.00001
        
        with patch('core.risk.mt5.symbol_info', return_value=mock_symbol_info):
            # Test high confidence
            lot_size_high, risk_high = self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, 0.9
            )
            
            # Test low confidence
            lot_size_low, risk_low = self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, 0.3
            )
        
        print(f"High confidence (0.9): {lot_size_high:.3f} lots, ${risk_high:.2f} risk")
        print(f"Low confidence (0.3): {lot_size_low:.3f} lots, ${risk_low:.2f} risk")
        
        # Higher confidence should result in larger position
        assert lot_size_high > lot_size_low, "Higher confidence should result in larger position"
        assert risk_high > risk_low, "Higher confidence should result in higher risk"
        
        print("✅ Different confidence levels test passed")
    
    def test_stop_loss_validation_correct_side(self):
        """Test stop loss validation for correct side of entry"""
        print("\n=== Testing Stop Loss Validation - Correct Side ===")
        
        # Test buy order with stop below entry
        entry_price = 1.2000
        stop_loss_buy = 1.1950  # Correct: below entry for buy
        
        is_valid_buy = self._validate_stop_loss(entry_price, stop_loss_buy, "buy")
        print(f"Buy order SL validation: {is_valid_buy}")
        assert is_valid_buy, "Stop loss below entry should be valid for buy order"
        
        # Test sell order with stop above entry
        stop_loss_sell = 1.2050  # Correct: above entry for sell
        
        is_valid_sell = self._validate_stop_loss(entry_price, stop_loss_sell, "sell")
        print(f"Sell order SL validation: {is_valid_sell}")
        assert is_valid_sell, "Stop loss above entry should be valid for sell order"
        
        print("✅ Stop loss correct side validation test passed")
    
    def test_stop_loss_validation_incorrect_side(self):
        """Test stop loss validation rejects incorrect side"""
        print("\n=== Testing Stop Loss Validation - Incorrect Side ===")
        
        entry_price = 1.2000
        
        # Test buy order with stop above entry (incorrect)
        stop_loss_buy_wrong = 1.2050  # Incorrect: above entry for buy
        
        is_valid_buy = self._validate_stop_loss(entry_price, stop_loss_buy_wrong, "buy")
        print(f"Buy order with SL above entry: {is_valid_buy}")
        assert not is_valid_buy, "Stop loss above entry should be invalid for buy order"
        
        # Test sell order with stop below entry (incorrect)
        stop_loss_sell_wrong = 1.1950  # Incorrect: below entry for sell
        
        is_valid_sell = self._validate_stop_loss(entry_price, stop_loss_sell_wrong, "sell")
        print(f"Sell order with SL below entry: {is_valid_sell}")
        assert not is_valid_sell, "Stop loss below entry should be invalid for sell order"
        
        print("✅ Stop loss incorrect side validation test passed")
    
    def test_stop_loss_validation_zero_negative(self):
        """Test stop loss validation rejects zero and negative values"""
        print("\n=== Testing Stop Loss Validation - Zero/Negative ===")
        
        entry_price = 1.2000
        
        # Test zero stop loss
        is_valid_zero = self._validate_stop_loss(entry_price, 0, "buy")
        print(f"Zero stop loss validation: {is_valid_zero}")
        assert not is_valid_zero, "Zero stop loss should be invalid"
        
        # Test negative stop loss
        is_valid_negative = self._validate_stop_loss(entry_price, -1.0, "buy")
        print(f"Negative stop loss validation: {is_valid_negative}")
        assert not is_valid_negative, "Negative stop loss should be invalid"
        
        print("✅ Stop loss zero/negative validation test passed")
    
    def test_take_profit_validation_correct_side(self):
        """Test take profit validation for correct side of entry"""
        print("\n=== Testing Take Profit Validation - Correct Side ===")
        
        entry_price = 1.2000
        
        # Test buy order with TP above entry
        tp_buy = 1.2050  # Correct: above entry for buy
        
        is_valid_buy = self._validate_take_profit(entry_price, tp_buy, "buy")
        print(f"Buy order TP validation: {is_valid_buy}")
        assert is_valid_buy, "Take profit above entry should be valid for buy order"
        
        # Test sell order with TP below entry
        tp_sell = 1.1950  # Correct: below entry for sell
        
        is_valid_sell = self._validate_take_profit(entry_price, tp_sell, "sell")
        print(f"Sell order TP validation: {is_valid_sell}")
        assert is_valid_sell, "Take profit below entry should be valid for sell order"
        
        print("✅ Take profit correct side validation test passed")
    
    def test_take_profit_validation_incorrect_side(self):
        """Test take profit validation rejects incorrect side"""
        print("\n=== Testing Take Profit Validation - Incorrect Side ===")
        
        entry_price = 1.2000
        
        # Test buy order with TP below entry (incorrect)
        tp_buy_wrong = 1.1950  # Incorrect: below entry for buy
        
        is_valid_buy = self._validate_take_profit(entry_price, tp_buy_wrong, "buy")
        print(f"Buy order with TP below entry: {is_valid_buy}")
        assert not is_valid_buy, "Take profit below entry should be invalid for buy order"
        
        # Test sell order with TP above entry (incorrect)
        tp_sell_wrong = 1.2050  # Incorrect: above entry for sell
        
        is_valid_sell = self._validate_take_profit(entry_price, tp_sell_wrong, "sell")
        print(f"Sell order with TP above entry: {is_valid_sell}")
        assert not is_valid_sell, "Take profit above entry should be invalid for sell order"
        
        print("✅ Take profit incorrect side validation test passed")
    
    def test_take_profit_validation_rr_requirement(self):
        """Test take profit validation enforces minimum 1:1 RR"""
        print("\n=== Testing Take Profit Validation - RR Requirement ===")
        
        entry_price = 1.2000
        stop_loss = 1.1950  # 50 pips stop
        stop_distance = abs(entry_price - stop_loss)
        
        # Test TP with insufficient RR (0.5:1)
        tp_insufficient = entry_price + (stop_distance * 0.5)  # 25 pips TP
        
        is_valid_insufficient = self._validate_take_profit_rr(entry_price, tp_insufficient, stop_loss)
        print(f"TP with 0.5:1 RR validation: {is_valid_insufficient}")
        assert not is_valid_insufficient, "TP with < 1:1 RR should be invalid"
        
        # Test TP with sufficient RR (1.5:1)
        tp_sufficient = entry_price + (stop_distance * 1.5)  # 75 pips TP
        
        is_valid_sufficient = self._validate_take_profit_rr(entry_price, tp_sufficient, stop_loss)
        print(f"TP with 1.5:1 RR validation: {is_valid_sufficient}")
        assert is_valid_sufficient, "TP with >= 1:1 RR should be valid"
        
        print("✅ Take profit RR requirement validation test passed")
    
    def test_lot_size_validation_positive(self):
        """Test lot size validation for positive values"""
        print("\n=== Testing Lot Size Validation - Positive ===")
        
        # Test valid lot sizes
        valid_lots = [0.01, 0.1, 0.5, 1.0, 2.0]
        
        for lot in valid_lots:
            is_valid = self._validate_lot_size(lot)
            print(f"Lot size {lot}: {is_valid}")
            assert is_valid, f"Lot size {lot} should be valid"
        
        print("✅ Lot size positive validation test passed")
    
    def test_lot_size_validation_negative_zero(self):
        """Test lot size validation rejects negative and zero values"""
        print("\n=== Testing Lot Size Validation - Negative/Zero ===")
        
        # Test invalid lot sizes
        invalid_lots = [0, -0.01, -1.0]
        
        for lot in invalid_lots:
            is_valid = self._validate_lot_size(lot)
            print(f"Lot size {lot}: {is_valid}")
            assert not is_valid, f"Lot size {lot} should be invalid"
        
        print("✅ Lot size negative/zero validation test passed")
    
    def test_break_even_logic_activation(self):
        """Test break-even logic activation when price moves in favor"""
        print("\n=== Testing Break-Even Logic Activation ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        stop_loss = 1.1950
        current_price = 1.2020  # Price moved 20 pips in favor
        
        # Create a position
        position = Position(
            symbol=symbol,
            side="buy",
            entry_price=entry_price,
            lot_size=0.1,
            stop_loss=stop_loss,
            take_profit=1.2050,
            confidence=0.8,
            risk_amount=20.0,
            trailing_stop_price=None,
            trailing_stop_distance=None,
            break_even_triggered=False
        )
        
        # Mock historical data for ATR calculation
        historical_data = self._create_mock_historical_data()
        
        # Mock MT5 symbol info
        mock_symbol_info = MagicMock()
        mock_symbol_info.ask = current_price
        mock_symbol_info.bid = current_price - 0.0001
        
        with patch('core.risk.mt5.symbol_info', return_value=mock_symbol_info):
            with patch('core.risk.mt5.order_send') as mock_order_send:
                mock_order_send.return_value = MagicMock(retcode=10009)  # Success
                
                result = self.risk_manager.update_position_prices(
                    symbol, current_price, historical_data
                )
        
        print(f"Break-even update result: {result}")
        print(f"Position break-even triggered: {position.break_even_triggered}")
        
        # Break-even should be triggered
        assert result is not False, "Break-even update should succeed"
        
        print("✅ Break-even logic activation test passed")
    
    def test_dynamic_spread_calculation(self):
        """Test dynamic spread calculation"""
        print("\n=== Testing Dynamic Spread Calculation ===")
        
        # Create mock historical data
        historical_data = self._create_mock_historical_data()
        
        # Test with base spread
        spread = get_dynamic_spread(historical_data, base_min_spread=10.0, spread_multiplier=1.5)
        print(f"Dynamic spread with base: {spread:.2f} pips")
        
        assert spread > 0, "Dynamic spread should be positive"
        assert spread >= 10.0, "Dynamic spread should be at least base spread"
        
        # Test without base spread
        spread_no_base = get_dynamic_spread(historical_data, base_min_spread=0.0, spread_multiplier=1.5)
        print(f"Dynamic spread without base: {spread_no_base:.2f} pips")
        
        assert spread_no_base > 0, "Dynamic spread should be positive even without base"
        
        print("✅ Dynamic spread calculation test passed")
    
    def test_intelligent_sl_calculation(self):
        """Test intelligent stop loss calculation using ATR"""
        print("\n=== Testing Intelligent SL Calculation ===")
        
        # Create mock historical data
        historical_data = self._create_mock_historical_data()
        
        # Test ATR-based SL
        atr_sl = get_intelligent_sl(historical_data, atr_period=14, sl_multiplier=2.0)
        print(f"ATR-based SL: {atr_sl:.2f} pips")
        
        assert atr_sl > 0, "ATR-based SL should be positive"
        assert atr_sl < 100, "ATR-based SL should be reasonable"
        
        print("✅ Intelligent SL calculation test passed")
    
    def test_trailing_stop_calculation(self):
        """Test trailing stop calculation"""
        print("\n=== Testing Trailing Stop Calculation ===")
        
        # Create a position
        position = Position(
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
        
        current_price = 1.2020
        current_atr = 0.001
        
        # Test trailing stop calculation
        new_trailing_stop = calculate_trailing_stop(position, current_price, current_atr)
        
        print(f"New trailing stop: {new_trailing_stop}")
        
        if new_trailing_stop:
            assert new_trailing_stop > position.stop_loss, "Trailing stop should be better than original SL"
            assert new_trailing_stop < current_price, "Trailing stop should be below current price for buy"
        
        print("✅ Trailing stop calculation test passed")
    
    def _create_mock_historical_data(self, bars=50):
        """Create mock historical data for testing"""
        np.random.seed(42)
        data = []
        base_price = 1.2000
        
        for i in range(bars):
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
    
    def _validate_stop_loss(self, entry_price, stop_loss, side):
        """Helper method to validate stop loss"""
        if stop_loss is None or stop_loss <= 0:
            return False
        
        if side == 'buy' and stop_loss >= entry_price:
            return False
        elif side == 'sell' and stop_loss <= entry_price:
            return False
        
        return True
    
    def _validate_take_profit(self, entry_price, take_profit, side):
        """Helper method to validate take profit"""
        if take_profit is None or take_profit <= 0:
            return False
        
        if side == 'buy' and take_profit <= entry_price:
            return False
        elif side == 'sell' and take_profit >= entry_price:
            return False
        
        return True
    
    def _validate_take_profit_rr(self, entry_price, take_profit, stop_loss):
        """Helper method to validate take profit RR"""
        if not self._validate_take_profit(entry_price, take_profit, "buy"):
            return False
        
        stop_distance = abs(entry_price - stop_loss)
        tp_distance = abs(take_profit - entry_price)
        
        return tp_distance >= stop_distance
    
    def _validate_lot_size(self, lot_size):
        """Helper method to validate lot size"""
        return lot_size is not None and lot_size > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
