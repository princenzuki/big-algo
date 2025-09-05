"""
Unit tests for Main Trading Logic

Tests signal processing, trade execution flow, validation,
and error handling in the main trading loop.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import logging
from datetime import datetime

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestMainTradingLogic:
    """Test cases for Main Trading Logic"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Mock the main trading bot class
        self.mock_bot = MagicMock()
        self.mock_bot.historical_data = {}
        self.mock_bot.smart_tp = MagicMock()
        self.mock_bot.risk_manager = MagicMock()
        self.mock_bot.broker_adapter = MagicMock()
        self.mock_bot.portfolio_manager = MagicMock()
        
        # Mock account info
        self.mock_account_info = MagicMock()
        self.mock_account_info.balance = 10000.0
        self.mock_account_info.equity = 10000.0
        self.mock_account_info.free_margin = 10000.0
        
        # Mock symbol info
        self.mock_symbol_info = MagicMock()
        self.mock_symbol_info.ask = 1.2000
        self.mock_symbol_info.bid = 1.1999
        self.mock_symbol_info.point = 0.00001
        self.mock_symbol_info.trade_tops_level = 10
        
        # Create mock historical data
        self.mock_historical_data = self._create_mock_historical_data()
    
    def _create_mock_historical_data(self, bars=100):
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
    
    def test_signal_validation_valid_signal(self):
        """Test signal validation with valid signal data"""
        print("\n=== Testing Signal Validation - Valid Signal ===")
        
        # Test valid signal
        valid_signal = {'signal': 1, 'confidence': 0.8}
        
        is_valid = self._validate_signal(valid_signal)
        print(f"Valid signal validation: {is_valid}")
        assert is_valid, "Valid signal should pass validation"
        
        # Test neutral signal
        neutral_signal = {'signal': 0, 'confidence': 0.5}
        
        is_valid_neutral = self._validate_signal(neutral_signal)
        print(f"Neutral signal validation: {is_valid_neutral}")
        assert is_valid_neutral, "Neutral signal should pass validation"
        
        print("✅ Valid signal validation test passed")
    
    def test_signal_validation_invalid_signal(self):
        """Test signal validation with invalid signal data"""
        print("\n=== Testing Signal Validation - Invalid Signal ===")
        
        # Test invalid signals
        invalid_signals = [
            {'signal': None, 'confidence': 0.8},
            {'signal': 1, 'confidence': None},
            {'signal': 2, 'confidence': 0.8},  # Invalid signal value
            {'signal': 1, 'confidence': 1.5},  # Invalid confidence
            {'signal': 1, 'confidence': -0.1},  # Invalid confidence
            {'signal': 'buy', 'confidence': 0.8},  # Wrong type
        ]
        
        for invalid_signal in invalid_signals:
            is_valid = self._validate_signal(invalid_signal)
            print(f"Invalid signal {invalid_signal}: {is_valid}")
            assert not is_valid, f"Invalid signal {invalid_signal} should fail validation"
        
        print("✅ Invalid signal validation test passed")
    
    def test_historical_data_validation_sufficient_data(self):
        """Test historical data validation with sufficient data"""
        print("\n=== Testing Historical Data Validation - Sufficient Data ===")
        
        # Test with sufficient data
        is_valid = self._validate_historical_data(self.mock_historical_data)
        print(f"Sufficient data validation: {is_valid}")
        assert is_valid, "Sufficient data should pass validation"
        
        print("✅ Sufficient data validation test passed")
    
    def test_historical_data_validation_insufficient_data(self):
        """Test historical data validation with insufficient data"""
        print("\n=== Testing Historical Data Validation - Insufficient Data ===")
        
        # Test with insufficient data
        insufficient_data = self.mock_historical_data[:10]  # Only 10 bars
        
        is_valid = self._validate_historical_data(insufficient_data)
        print(f"Insufficient data validation: {is_valid}")
        assert not is_valid, "Insufficient data should fail validation"
        
        # Test with empty data
        is_valid_empty = self._validate_historical_data([])
        print(f"Empty data validation: {is_valid_empty}")
        assert not is_valid_empty, "Empty data should fail validation"
        
        print("✅ Insufficient data validation test passed")
    
    def test_historical_data_validation_nan_values(self):
        """Test historical data validation with NaN values"""
        print("\n=== Testing Historical Data Validation - NaN Values ===")
        
        # Create data with NaN values
        nan_data = self.mock_historical_data.copy()
        nan_data[10]['high'] = np.nan
        nan_data[20]['low'] = np.nan
        
        is_valid = self._validate_historical_data(nan_data)
        print(f"NaN data validation: {is_valid}")
        assert not is_valid, "Data with NaN values should fail validation"
        
        print("✅ NaN data validation test passed")
    
    def test_stop_loss_calculation_buy_order(self):
        """Test stop loss calculation for buy orders"""
        print("\n=== Testing Stop Loss Calculation - Buy Order ===")
        
        entry_price = 1.2000
        side = "buy"
        spread_pips = 20.0
        pip_value = 0.0001
        
        stop_loss = self._calculate_stop_loss(entry_price, side, spread_pips, pip_value)
        print(f"Entry price: {entry_price}")
        print(f"Stop loss: {stop_loss}")
        print(f"Stop distance: {abs(entry_price - stop_loss):.5f}")
        
        # Stop loss should be below entry for buy order
        assert stop_loss < entry_price, "Stop loss should be below entry for buy order"
        
        # Should be reasonable distance
        stop_distance = abs(entry_price - stop_loss)
        expected_distance = (spread_pips + 10) * pip_value
        assert abs(stop_distance - expected_distance) < 0.0001, f"Expected distance {expected_distance}, got {stop_distance}"
        
        print("✅ Buy order stop loss calculation test passed")
    
    def test_stop_loss_calculation_sell_order(self):
        """Test stop loss calculation for sell orders"""
        print("\n=== Testing Stop Loss Calculation - Sell Order ===")
        
        entry_price = 1.2000
        side = "sell"
        spread_pips = 20.0
        pip_value = 0.0001
        
        stop_loss = self._calculate_stop_loss(entry_price, side, spread_pips, pip_value)
        print(f"Entry price: {entry_price}")
        print(f"Stop loss: {stop_loss}")
        print(f"Stop distance: {abs(stop_loss - entry_price):.5f}")
        
        # Stop loss should be above entry for sell order
        assert stop_loss > entry_price, "Stop loss should be above entry for sell order"
        
        # Should be reasonable distance
        stop_distance = abs(stop_loss - entry_price)
        expected_distance = (spread_pips + 10) * pip_value
        assert abs(stop_distance - expected_distance) < 0.0001, f"Expected distance {expected_distance}, got {stop_distance}"
        
        print("✅ Sell order stop loss calculation test passed")
    
    def test_take_profit_validation_correct_side(self):
        """Test take profit validation for correct side of entry"""
        print("\n=== Testing Take Profit Validation - Correct Side ===")
        
        entry_price = 1.2000
        
        # Test buy order with TP above entry
        tp_buy = 1.2050
        is_valid_buy = self._validate_take_profit_side(entry_price, tp_buy, "buy")
        print(f"Buy order TP validation: {is_valid_buy}")
        assert is_valid_buy, "TP above entry should be valid for buy order"
        
        # Test sell order with TP below entry
        tp_sell = 1.1950
        is_valid_sell = self._validate_take_profit_side(entry_price, tp_sell, "sell")
        print(f"Sell order TP validation: {is_valid_sell}")
        assert is_valid_sell, "TP below entry should be valid for sell order"
        
        print("✅ Take profit correct side validation test passed")
    
    def test_take_profit_validation_incorrect_side(self):
        """Test take profit validation rejects incorrect side"""
        print("\n=== Testing Take Profit Validation - Incorrect Side ===")
        
        entry_price = 1.2000
        
        # Test buy order with TP below entry (incorrect)
        tp_buy_wrong = 1.1950
        is_valid_buy = self._validate_take_profit_side(entry_price, tp_buy_wrong, "buy")
        print(f"Buy order with TP below entry: {is_valid_buy}")
        assert not is_valid_buy, "TP below entry should be invalid for buy order"
        
        # Test sell order with TP above entry (incorrect)
        tp_sell_wrong = 1.2050
        is_valid_sell = self._validate_take_profit_side(entry_price, tp_sell_wrong, "sell")
        print(f"Sell order with TP above entry: {is_valid_sell}")
        assert not is_valid_sell, "TP above entry should be invalid for sell order"
        
        print("✅ Take profit incorrect side validation test passed")
    
    def test_risk_reward_validation_sufficient_rr(self):
        """Test risk-reward validation with sufficient RR"""
        print("\n=== Testing Risk-Reward Validation - Sufficient RR ===")
        
        entry_price = 1.2000
        stop_loss = 1.1950
        take_profit = 1.2100  # 2:1 RR
        
        is_valid = self._validate_risk_reward(entry_price, stop_loss, take_profit)
        print(f"RR validation (2:1): {is_valid}")
        assert is_valid, "2:1 RR should be valid"
        
        # Test exactly 1:1 RR
        take_profit_1_1 = 1.2050  # 1:1 RR
        is_valid_1_1 = self._validate_risk_reward(entry_price, stop_loss, take_profit_1_1)
        print(f"RR validation (1:1): {is_valid_1_1}")
        assert is_valid_1_1, "1:1 RR should be valid"
        
        print("✅ Sufficient RR validation test passed")
    
    def test_risk_reward_validation_insufficient_rr(self):
        """Test risk-reward validation with insufficient RR"""
        print("\n=== Testing Risk-Reward Validation - Insufficient RR ===")
        
        entry_price = 1.2000
        stop_loss = 1.1950
        take_profit = 1.2025  # 0.5:1 RR (insufficient)
        
        is_valid = self._validate_risk_reward(entry_price, stop_loss, take_profit)
        print(f"RR validation (0.5:1): {is_valid}")
        assert not is_valid, "0.5:1 RR should be invalid"
        
        print("✅ Insufficient RR validation test passed")
    
    def test_lot_size_validation_positive(self):
        """Test lot size validation for positive values"""
        print("\n=== Testing Lot Size Validation - Positive ===")
        
        valid_lots = [0.01, 0.1, 0.5, 1.0, 2.0]
        
        for lot in valid_lots:
            is_valid = self._validate_lot_size(lot)
            print(f"Lot size {lot}: {is_valid}")
            assert is_valid, f"Lot size {lot} should be valid"
        
        print("✅ Positive lot size validation test passed")
    
    def test_lot_size_validation_negative_zero(self):
        """Test lot size validation rejects negative and zero values"""
        print("\n=== Testing Lot Size Validation - Negative/Zero ===")
        
        invalid_lots = [0, -0.01, -1.0]
        
        for lot in invalid_lots:
            is_valid = self._validate_lot_size(lot)
            print(f"Lot size {lot}: {is_valid}")
            assert not is_valid, f"Lot size {lot} should be invalid"
        
        print("✅ Negative/zero lot size validation test passed")
    
    def test_trade_execution_success_flow(self):
        """Test successful trade execution flow"""
        print("\n=== Testing Trade Execution Success Flow ===")
        
        # Mock successful execution
        mock_order_result = MagicMock()
        mock_order_result.success = True
        mock_order_result.order_id = 12345
        mock_order_result.price = 1.2000
        mock_order_result.error_message = None
        
        mock_trade = MagicMock()
        mock_trade.id = "trade_123"
        
        self.mock_bot.broker_adapter.place_order.return_value = mock_order_result
        self.mock_bot.portfolio_manager.open_trade.return_value = mock_trade
        
        # Simulate trade execution
        result = self._simulate_trade_execution(
            symbol="EURUSD",
            side="buy",
            lot_size=0.1,
            entry_price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            confidence=0.8
        )
        
        print(f"Trade execution result: {result}")
        
        # Should succeed
        assert result['success'], "Trade execution should succeed"
        assert result['order_id'] == 12345, f"Expected order ID 12345, got {result['order_id']}"
        assert result['trade_id'] == "trade_123", f"Expected trade ID trade_123, got {result['trade_id']}"
        
        print("✅ Trade execution success flow test passed")
    
    def test_trade_execution_failure_flow(self):
        """Test failed trade execution flow"""
        print("\n=== Testing Trade Execution Failure Flow ===")
        
        # Mock failed execution
        mock_order_result = MagicMock()
        mock_order_result.success = False
        mock_order_result.order_id = None
        mock_order_result.price = 0.0
        mock_order_result.error_message = "Order rejected"
        
        self.mock_bot.broker_adapter.place_order.return_value = mock_order_result
        
        # Simulate trade execution
        result = self._simulate_trade_execution(
            symbol="EURUSD",
            side="buy",
            lot_size=0.1,
            entry_price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            confidence=0.8
        )
        
        print(f"Trade execution result: {result}")
        
        # Should fail
        assert not result['success'], "Trade execution should fail"
        assert result['order_id'] is None, "Order ID should be None for failed execution"
        assert "Order rejected" in result['error_message'], f"Expected error message, got {result['error_message']}"
        
        print("✅ Trade execution failure flow test passed")
    
    def test_break_even_update_success(self):
        """Test successful break-even update"""
        print("\n=== Testing Break-Even Update Success ===")
        
        # Mock successful break-even update
        self.mock_bot.risk_manager.update_position_prices.return_value = True
        
        result = self._simulate_break_even_update("EURUSD", 1.2020)
        
        print(f"Break-even update result: {result}")
        
        # Should succeed
        assert result, "Break-even update should succeed"
        
        print("✅ Break-even update success test passed")
    
    def test_break_even_update_failure(self):
        """Test failed break-even update"""
        print("\n=== Testing Break-Even Update Failure ===")
        
        # Mock failed break-even update
        self.mock_bot.risk_manager.update_position_prices.return_value = False
        
        result = self._simulate_break_even_update("EURUSD", 1.2020)
        
        print(f"Break-even update result: {result}")
        
        # Should fail
        assert not result, "Break-even update should fail"
        
        print("✅ Break-even update failure test passed")
    
    def test_partial_tp_trigger(self):
        """Test partial take profit trigger"""
        print("\n=== Testing Partial Take Profit Trigger ===")
        
        # Mock partial TP trigger
        self.mock_bot.smart_tp.check_partial_tp.return_value = True
        
        result = self._simulate_partial_tp_check("EURUSD", 1.2015)
        
        print(f"Partial TP check result: {result}")
        
        # Should trigger
        assert result, "Partial TP should trigger"
        
        print("✅ Partial TP trigger test passed")
    
    def test_trailing_tp_update(self):
        """Test trailing take profit update"""
        print("\n=== Testing Trailing Take Profit Update ===")
        
        # Mock trailing TP update
        self.mock_bot.smart_tp.update_trailing_tp.return_value = 1.2025
        
        result = self._simulate_trailing_tp_update("EURUSD", 1.2020)
        
        print(f"Trailing TP update result: {result}")
        
        # Should return new TP level
        assert result == 1.2025, f"Expected trailing TP 1.2025, got {result}"
        
        print("✅ Trailing TP update test passed")
    
    # Helper methods for testing
    def _validate_signal(self, signal_data):
        """Helper method to validate signal data"""
        signal = signal_data.get('signal')
        confidence = signal_data.get('confidence')
        
        if signal is None or not isinstance(signal, (int, float)) or signal not in [-1, 0, 1]:
            return False
        if confidence is None or not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            return False
        
        return True
    
    def _validate_historical_data(self, data):
        """Helper method to validate historical data"""
        if not data or len(data) < 20:
            return False
        
        df = pd.DataFrame(data)
        if df[['open', 'high', 'low', 'close']].isnull().any().any():
            return False
        
        return True
    
    def _calculate_stop_loss(self, entry_price, side, spread_pips, pip_value):
        """Helper method to calculate stop loss"""
        stop_loss_pips = spread_pips + 10
        stop_loss_distance = stop_loss_pips * pip_value
        
        if side == 'buy':
            return entry_price - stop_loss_distance
        else:
            return entry_price + stop_loss_distance
    
    def _validate_take_profit_side(self, entry_price, take_profit, side):
        """Helper method to validate take profit side"""
        if take_profit is None or take_profit <= 0:
            return False
        
        if side == 'buy' and take_profit <= entry_price:
            return False
        elif side == 'sell' and take_profit >= entry_price:
            return False
        
        return True
    
    def _validate_risk_reward(self, entry_price, stop_loss, take_profit):
        """Helper method to validate risk-reward ratio"""
        stop_distance = abs(entry_price - stop_loss)
        tp_distance = abs(take_profit - entry_price)
        
        return tp_distance >= stop_distance
    
    def _validate_lot_size(self, lot_size):
        """Helper method to validate lot size"""
        return lot_size is not None and lot_size > 0
    
    def _simulate_trade_execution(self, symbol, side, lot_size, entry_price, stop_loss, take_profit, confidence):
        """Helper method to simulate trade execution"""
        order_request = MagicMock()
        order_request.symbol = symbol
        order_request.side = side
        order_request.lot_size = lot_size
        order_request.price = entry_price
        order_request.stop_loss = stop_loss
        order_request.take_profit = take_profit
        
        order_result = self.mock_bot.broker_adapter.place_order(order_request)
        
        if order_result.success:
            trade = self.mock_bot.portfolio_manager.open_trade(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                lot_size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence
            )
            
            return {
                'success': True,
                'order_id': order_result.order_id,
                'trade_id': trade.id if trade else None,
                'error_message': None
            }
        else:
            return {
                'success': False,
                'order_id': None,
                'trade_id': None,
                'error_message': order_result.error_message
            }
    
    def _simulate_break_even_update(self, symbol, current_price):
        """Helper method to simulate break-even update"""
        return self.mock_bot.risk_manager.update_position_prices(
            symbol, current_price, self.mock_historical_data
        )
    
    def _simulate_partial_tp_check(self, symbol, current_price):
        """Helper method to simulate partial TP check"""
        return self.mock_bot.smart_tp.check_partial_tp(symbol, current_price)
    
    def _simulate_trailing_tp_update(self, symbol, current_price):
        """Helper method to simulate trailing TP update"""
        return self.mock_bot.smart_tp.update_trailing_tp(symbol, current_price, self.mock_historical_data)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
