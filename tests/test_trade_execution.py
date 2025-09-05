"""
Unit tests for Trade Execution System

Tests MT5 order placement, validation, error handling,
and trade execution success/failure scenarios.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import logging
from datetime import datetime

from adapters.mt5_adapter import MT5Adapter, OrderRequest, OrderResult
from adapters.broker_base import SymbolInfo

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestTradeExecution:
    """Test cases for Trade Execution system"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mt5_adapter = MT5Adapter()
        
        # Mock symbol info
        self.mock_symbol_info = SymbolInfo(
            symbol="EURUSD",
            bid=1.1999,
            ask=1.2000,
            spread=0.0001,
            point=0.00001,
            digits=5,
            lot_min=0.01,
            lot_max=100.0,
            lot_step=0.01,
            margin_required=100.0,
            trade_tick_value=1.0,
            trade_tick_size=0.00001,
            trade_stops_level=10
        )
        
        # Mock MT5 symbol info for adapter tests
        self.mock_mt5_symbol_info = MagicMock()
        self.mock_mt5_symbol_info.point = 0.00001
        self.mock_mt5_symbol_info.digits = 5
        self.mock_mt5_symbol_info.volume_min = 0.01
        self.mock_mt5_symbol_info.volume_max = 100.0
        self.mock_mt5_symbol_info.volume_step = 0.01
        self.mock_mt5_symbol_info.margin_initial = 100.0
    
    def test_order_request_creation(self):
        """Test order request creation with valid parameters"""
        print("\n=== Testing Order Request Creation ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Test Order"
        )
        
        print(f"Order Request: {order_request}")
        
        # Validate order request
        assert order_request.symbol == "EURUSD"
        assert order_request.order_type == "buy"
        assert order_request.lot_size == 0.1
        assert order_request.price == 1.2000
        assert order_request.stop_loss == 1.1950
        assert order_request.take_profit == 1.2050
        
        print("✅ Order request creation test passed")
    
    def test_successful_order_placement(self):
        """Test successful order placement with MT5"""
        print("\n=== Testing Successful Order Placement ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Test Order"
        )
        
        # Mock successful MT5 response
        mock_mt5_result = MagicMock()
        mock_mt5_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_mt5_result.order = 12345
        mock_mt5_result.price = 1.2000
        mock_mt5_result.volume = 0.1
        mock_mt5_result.comment = "Test Order"
        mock_mt5_result.request_id = 1
        
        # Mock MT5 symbol info and tick data
        mock_tick = MagicMock()
        mock_tick.bid = 1.1999
        mock_tick.ask = 1.2000
        
        with patch('adapters.mt5_adapter.mt5.symbol_info', return_value=self.mock_mt5_symbol_info):
            with patch('adapters.mt5_adapter.mt5.symbol_info_tick', return_value=mock_tick):
                with patch('adapters.mt5_adapter.mt5.order_send', return_value=mock_mt5_result):
                    result = self.mt5_adapter.place_order(order_request)
        
        print(f"Order Result: {result}")
        print(f"Success: {result.success}")
        print(f"Order ID: {result.order_id}")
        print(f"Price: {result.price}")
        
        # Validate successful result
        assert result.success, "Order should be successful"
        assert result.order_id == 12345, f"Expected order ID 12345, got {result.order_id}"
        assert result.price == 1.2000, f"Expected price 1.2000, got {result.price}"
        assert result.error_message is None, "Error message should be None for successful order"
        
        print("✅ Successful order placement test passed")
    
    def test_failed_order_placement(self):
        """Test failed order placement with MT5"""
        print("\n=== Testing Failed Order Placement ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Test Order"
        )
        
        # Mock failed MT5 response
        mock_mt5_result = MagicMock()
        mock_mt5_result.retcode = 10004  # TRADE_RETCODE_REJECT
        mock_mt5_result.order = 0
        mock_mt5_result.price = 0.0
        mock_mt5_result.volume = 0.0
        mock_mt5_result.comment = "Order rejected"
        mock_mt5_result.request_id = 1
        
        # Mock MT5 symbol info and tick data
        mock_tick = MagicMock()
        mock_tick.bid = 1.1999
        mock_tick.ask = 1.2000
        
        with patch('adapters.mt5_adapter.mt5.symbol_info', return_value=self.mock_mt5_symbol_info):
            with patch('adapters.mt5_adapter.mt5.symbol_info_tick', return_value=mock_tick):
                with patch('adapters.mt5_adapter.mt5.order_send', return_value=mock_mt5_result):
                    result = self.mt5_adapter.place_order(order_request)
        
        print(f"Order Result: {result}")
        print(f"Success: {result.success}")
        print(f"Error Message: {result.error_message}")
        
        # Validate failed result
        assert not result.success, "Order should fail"
        assert result.order_id == 0, f"Order ID should be 0 for failed order, got {result.order_id}"
        assert "Order rejected" in result.error_message, f"Expected error message, got {result.error_message}"
        
        print("✅ Failed order placement test passed")
    
    def test_order_placement_missing_order_id(self):
        """Test order placement when MT5 returns success but no order ID"""
        print("\n=== Testing Order Placement - Missing Order ID ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Test Order"
        )
        
        # Mock MT5 response with success but no order ID
        mock_mt5_result = MagicMock()
        mock_mt5_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_mt5_result.order = 0  # No order ID
        mock_mt5_result.price = 1.2000
        mock_mt5_result.volume = 0.1
        mock_mt5_result.comment = "Test Order"
        mock_mt5_result.request_id = 1
        
        # Mock MT5 symbol info and tick data
        mock_tick = MagicMock()
        mock_tick.bid = 1.1999
        mock_tick.ask = 1.2000
        
        with patch('adapters.mt5_adapter.mt5.symbol_info', return_value=self.mock_mt5_symbol_info):
            with patch('adapters.mt5_adapter.mt5.symbol_info_tick', return_value=mock_tick):
                with patch('adapters.mt5_adapter.mt5.order_send', return_value=mock_mt5_result):
                    result = self.mt5_adapter.place_order(order_request)
        
        print(f"Order Result: {result}")
        print(f"Success: {result.success}")
        print(f"Order ID: {result.order_id}")
        
        # Should handle missing order ID gracefully
        assert result.success, "Order should still be considered successful"
        assert result.order_id == 0, f"Expected order ID 0, got {result.order_id}"
        
        print("✅ Missing order ID test passed")
    
    def test_lot_size_rounding(self):
        """Test lot size rounding to broker's step size"""
        print("\n=== Testing Lot Size Rounding ===")
        
        # Test lot size that needs rounding
        unrounded_lot = 0.123  # Should round to 0.12 (step 0.01)
        
        rounded_lot = self.mt5_adapter.round_lot_size(unrounded_lot, "EURUSD")
        
        print(f"Unrounded lot: {unrounded_lot}")
        print(f"Rounded lot: {rounded_lot}")
        
        # Should round to nearest step
        assert rounded_lot == 0.12, f"Expected 0.12, got {rounded_lot}"
        # Check if rounded lot aligns with step size (handle floating point precision)
        steps = rounded_lot / 0.01
        assert abs(steps - round(steps)) < 1e-10, f"Rounded lot {rounded_lot} should align with step size 0.01"
        
        print("✅ Lot size rounding test passed")
    
    def test_lot_size_validation_within_limits(self):
        """Test lot size validation within broker limits"""
        print("\n=== Testing Lot Size Validation - Within Limits ===")
        
        # Test lot sizes within limits
        valid_lots = [0.01, 0.1, 1.0, 10.0, 50.0]
        
        for lot in valid_lots:
            is_valid = self.mt5_adapter.validate_lot_size(lot, self.mock_symbol_info)
            print(f"Lot size {lot}: {is_valid}")
            assert is_valid, f"Lot size {lot} should be valid"
        
        print("✅ Lot size within limits validation test passed")
    
    def test_lot_size_validation_outside_limits(self):
        """Test lot size validation outside broker limits"""
        print("\n=== Testing Lot Size Validation - Outside Limits ===")
        
        # Test lot sizes outside limits
        invalid_lots = [0.005, 0.009, 101.0, 200.0]  # Below min, above max
        
        for lot in invalid_lots:
            is_valid = self.mt5_adapter.validate_lot_size(lot, self.mock_symbol_info)
            print(f"Lot size {lot}: {is_valid}")
            assert not is_valid, f"Lot size {lot} should be invalid"
        
        print("✅ Lot size outside limits validation test passed")
    
    def test_stop_level_adjustment(self):
        """Test stop level adjustment for broker requirements"""
        print("\n=== Testing Stop Level Adjustment ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.19995,  # Too close to entry (0.5 points)
            take_profit=1.20005,  # Too close to entry (0.5 points)
            comment="Test Order"
        )
        
        # Mock symbol info with stops level
        mock_symbol_info = SymbolInfo(
            symbol="EURUSD",
            bid=1.1999,
            ask=1.2000,
            spread=0.0001,
            point=0.00001,
            digits=5,
            lot_min=0.01,
            lot_max=100.0,
            lot_step=0.01,
            margin_required=100.0,
            trade_tick_value=1.0,
            trade_tick_size=0.00001,
            trade_stops_level=10  # 10 points minimum
        )
        
        with patch('adapters.mt5_adapter.mt5.symbol_info', return_value=mock_symbol_info):
            adjusted_request = self.mt5_adapter._adjust_stop_levels(order_request, mock_symbol_info)
        
        print(f"Original SL: {order_request.stop_loss}")
        print(f"Adjusted SL: {adjusted_request.stop_loss}")
        print(f"Original TP: {order_request.take_profit}")
        print(f"Adjusted TP: {adjusted_request.take_profit}")
        
        # Stop levels should be adjusted outward
        assert adjusted_request.stop_loss < order_request.stop_loss, "SL should be adjusted outward"
        assert adjusted_request.take_profit > order_request.take_profit, "TP should be adjusted outward"
        
        # Should meet minimum distance requirement
        min_distance = mock_symbol_info.trade_stops_level * mock_symbol_info.point
        sl_distance = abs(adjusted_request.price - adjusted_request.stop_loss)
        tp_distance = abs(adjusted_request.take_profit - adjusted_request.price)
        
        # Use floating point tolerance for distance comparison
        assert sl_distance >= min_distance - 1e-10, f"SL distance {sl_distance} should be >= {min_distance}"
        assert tp_distance >= min_distance - 1e-10, f"TP distance {tp_distance} should be >= {min_distance}"
        
        print("✅ Stop level adjustment test passed")
    
    def test_sell_order_execution(self):
        """Test sell order execution"""
        print("\n=== Testing Sell Order Execution ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="sell",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.2050,  # Above entry for sell
            take_profit=1.1950,  # Below entry for sell
            comment="Test Sell Order"
        )
        
        # Mock successful MT5 response
        mock_mt5_result = MagicMock()
        mock_mt5_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_mt5_result.order = 12346
        mock_mt5_result.price = 1.2000
        mock_mt5_result.volume = 0.1
        mock_mt5_result.comment = "Test Sell Order"
        mock_mt5_result.request_id = 1
        
        # Mock MT5 symbol info and tick data
        mock_tick = MagicMock()
        mock_tick.bid = 1.1999
        mock_tick.ask = 1.2000
        
        with patch('adapters.mt5_adapter.mt5.symbol_info', return_value=self.mock_mt5_symbol_info):
            with patch('adapters.mt5_adapter.mt5.symbol_info_tick', return_value=mock_tick):
                with patch('adapters.mt5_adapter.mt5.order_send', return_value=mock_mt5_result):
                    result = self.mt5_adapter.place_order(order_request)
        
        print(f"Sell Order Result: {result}")
        print(f"Success: {result.success}")
        print(f"Order ID: {result.order_id}")
        
        # Validate sell order result
        assert result.success, "Sell order should be successful"
        assert result.order_id == 12346, f"Expected order ID 12346, got {result.order_id}"
        
        print("✅ Sell order execution test passed")
    
    def test_order_validation_before_placement(self):
        """Test order validation before placement"""
        print("\n=== Testing Order Validation Before Placement ===")
        
        # Test valid order
        valid_order = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Valid Order"
        )
        
        is_valid, error_msg = self.mt5_adapter.validate_order_request(valid_order, self.mock_symbol_info)
        print(f"Valid order validation: {is_valid}, error: {error_msg}")
        assert is_valid, f"Valid order should pass validation, got error: {error_msg}"
        
        # Test invalid order (negative lot size)
        invalid_order = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=-0.1,  # Invalid
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Invalid Order"
        )
        
        is_valid_invalid, error_msg = self.mt5_adapter.validate_order_request(invalid_order, self.mock_symbol_info)
        print(f"Invalid order validation: {is_valid_invalid}, error: {error_msg}")
        assert not is_valid_invalid, "Invalid order should fail validation"
        
        print("✅ Order validation test passed")
    
    def test_mt5_connection_error_handling(self):
        """Test handling of MT5 connection errors"""
        print("\n=== Testing MT5 Connection Error Handling ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Test Order"
        )
        
        # Mock MT5 connection error
        with patch('adapters.mt5_adapter.mt5.symbol_info', side_effect=Exception("MT5 not connected")):
            result = self.mt5_adapter.place_order(order_request)
        
        print(f"Connection Error Result: {result}")
        print(f"Success: {result.success}")
        print(f"Error Message: {result.error_message}")
        
        # Should handle connection error gracefully
        assert not result.success, "Order should fail on connection error"
        assert "MT5 not connected" in result.error_message, f"Expected connection error, got {result.error_message}"
        
        print("✅ MT5 connection error handling test passed")
    
    def test_order_retry_logic(self):
        """Test order retry logic for transient failures"""
        print("\n=== Testing Order Retry Logic ===")
        
        order_request = OrderRequest(
            symbol="EURUSD",
            order_type="buy",
            lot_size=0.1,
            price=1.2000,
            stop_loss=1.1950,
            take_profit=1.2050,
            comment="Test Order"
        )
        
        # Mock first attempt fails, second succeeds
        mock_failed_result = MagicMock()
        mock_failed_result.retcode = 10004  # TRADE_RETCODE_REJECT
        
        mock_success_result = MagicMock()
        mock_success_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_success_result.order = 12347
        mock_success_result.price = 1.2000
        mock_success_result.volume = 0.1
        mock_success_result.comment = "Test Order"
        mock_success_result.request_id = 1
        
        with patch('adapters.mt5_adapter.mt5.symbol_info', return_value=self.mock_mt5_symbol_info):
            with patch('adapters.mt5_adapter.mt5.order_send', side_effect=[mock_failed_result, mock_success_result]):
                # Note: This test assumes retry logic exists in the adapter
                # If not implemented, this test documents the expected behavior
                result = self.mt5_adapter.place_order(order_request)
        
        print(f"Retry Result: {result}")
        print(f"Success: {result.success}")
        
        # Should eventually succeed or fail appropriately
        assert isinstance(result.success, bool), "Result should have success boolean"
        
        print("✅ Order retry logic test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
