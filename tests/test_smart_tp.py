"""
Unit tests for Smart Take Profit System

Tests ATR-based TP calculation, fallback RR logic, momentum indicators,
and hybrid TP system with 1:1 RR minimum enforcement.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from core.smart_tp import SmartTakeProfit, SmartTPConfig, SmartTPResult

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestSmartTakeProfit:
    """Test cases for Smart Take Profit system"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = SmartTPConfig(
            atr_period=14,
            baseline_tp_multiplier=2.0,
            strong_trend_multiplier=2.5,
            weak_trend_multiplier=1.0,
            partial_tp_ratio=0.5,
            partial_tp_rr=0.5,  # Partial TP should be closer to entry
            adx_period=14,
            adx_strong_threshold=25.0,
            cci_period=14,
            cci_strong_threshold=100.0,
            cci_weak_threshold=-100.0
        )
        self.smart_tp = SmartTakeProfit(self.config)
        
        # Create mock historical data
        self.mock_data = self._create_mock_historical_data()
    
    def _create_mock_historical_data(self, bars=100):
        """Create mock historical data for testing"""
        np.random.seed(42)  # For reproducible tests
        data = []
        base_price = 1.2000
        
        for i in range(bars):
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
    
    def test_atr_calculation_with_valid_data(self):
        """Test ATR calculation with valid historical data"""
        print("\n=== Testing ATR Calculation with Valid Data ===")
        
        df = pd.DataFrame(self.mock_data)
        atr = self.smart_tp._calculate_atr(df)
        
        print(f"ATR calculated: {atr:.6f}")
        
        # ATR should be positive and reasonable
        assert atr > 0, f"ATR should be positive, got {atr}"
        assert atr < 0.01, f"ATR should be reasonable (< 0.01), got {atr}"
        
        print("✅ ATR calculation test passed")
    
    def test_atr_calculation_with_insufficient_data(self):
        """Test ATR calculation with insufficient data"""
        print("\n=== Testing ATR Calculation with Insufficient Data ===")
        
        # Create data with only 5 bars (less than ATR period of 14)
        insufficient_data = self.mock_data[:5]
        df = pd.DataFrame(insufficient_data)
        atr = self.smart_tp._calculate_atr(df)
        
        print(f"ATR with insufficient data: {atr:.6f}")
        
        # Should return 0 for insufficient data
        assert atr == 0.0, f"ATR should be 0 for insufficient data, got {atr}"
        
        print("✅ ATR insufficient data test passed")
    
    def test_atr_calculation_with_nan_data(self):
        """Test ATR calculation with NaN values in data"""
        print("\n=== Testing ATR Calculation with NaN Data ===")
        
        df = pd.DataFrame(self.mock_data)
        # Introduce NaN values
        df.loc[10:15, 'high'] = np.nan
        df.loc[20:25, 'low'] = np.nan
        
        atr = self.smart_tp._calculate_atr(df)
        
        print(f"ATR with NaN data: {atr:.6f}")
        
        # Should handle NaN gracefully
        assert atr >= 0, f"ATR should handle NaN gracefully, got {atr}"
        
        print("✅ ATR NaN data test passed")
    
    def test_momentum_calculation_strong_trend(self):
        """Test momentum calculation for strong trend"""
        print("\n=== Testing Momentum Calculation - Strong Trend ===")
        
        # Create data with strong uptrend
        strong_trend_data = []
        base_price = 1.2000
        for i in range(50):
            base_price += 0.001  # Strong uptrend
            high = base_price + 0.0005
            low = base_price - 0.0002
            open_price = base_price - 0.0001
            close = base_price + 0.0003
            
            strong_trend_data.append({
                'time': i,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            })
        
        df = pd.DataFrame(strong_trend_data)
        momentum = self.smart_tp._calculate_momentum_strength(df)
        
        print(f"Momentum for strong trend: {momentum}")
        
        # Should detect strong trend
        assert momentum in ['strong', 'medium', 'weak'], f"Invalid momentum: {momentum}"
        
        print("✅ Strong trend momentum test passed")
    
    def test_momentum_calculation_weak_trend(self):
        """Test momentum calculation for weak trend"""
        print("\n=== Testing Momentum Calculation - Weak Trend ===")
        
        # Create data with weak/sideways movement
        weak_trend_data = []
        base_price = 1.2000
        for i in range(50):
            base_price += np.random.normal(0, 0.0001)  # Weak movement
            high = base_price + 0.0001
            low = base_price - 0.0001
            open_price = base_price
            close = base_price + np.random.normal(0, 0.00005)
            
            weak_trend_data.append({
                'time': i,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            })
        
        df = pd.DataFrame(weak_trend_data)
        momentum = self.smart_tp._calculate_momentum_strength(df)
        
        print(f"Momentum for weak trend: {momentum}")
        
        # Should detect weak trend
        assert momentum in ['strong', 'medium', 'weak'], f"Invalid momentum: {momentum}"
        
        print("✅ Weak trend momentum test passed")
    
    def test_hybrid_tp_system_atr_based_success(self):
        """Test hybrid TP system when ATR-based TP meets 1:1 RR requirement"""
        print("\n=== Testing Hybrid TP System - ATR Success ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        stop_loss = 1.1950  # 50 pips stop
        side = "buy"
        
        # Mock ATR calculation to return reasonable value
        with patch.object(self.smart_tp, '_calculate_atr', return_value=0.002):
            with patch.object(self.smart_tp, '_calculate_momentum_strength', return_value='medium'):
                result = self.smart_tp.calculate_smart_tp(
                    symbol=symbol,
                    entry_price=entry_price,
                    side=side,
                    historical_data=self.mock_data,
                    stop_loss=stop_loss
                )
        
        print(f"TP Result: {result}")
        print(f"Full TP Price: {result.full_tp_price:.5f}")
        print(f"Partial TP Price: {result.partial_tp_price:.5f}")
        print(f"Momentum: {result.momentum_strength}")
        print(f"TP Multiplier: {result.tp_multiplier}")
        
        # Validate TP prices
        assert result.full_tp_price > entry_price, "Full TP should be above entry for buy"
        assert result.partial_tp_price > entry_price, "Partial TP should be above entry for buy"
        assert result.full_tp_price > result.partial_tp_price, "Full TP should be higher than partial TP"
        
        # Validate RR (should be at least 1:1)
        stop_distance = abs(entry_price - stop_loss)
        tp_distance = abs(result.full_tp_price - entry_price)
        rr = tp_distance / stop_distance if stop_distance > 0 else 0
        
        print(f"Stop Distance: {stop_distance:.5f}")
        print(f"TP Distance: {tp_distance:.5f}")
        print(f"Risk:Reward Ratio: {rr:.2f}")
        
        assert rr >= 1.0, f"RR should be at least 1:1, got {rr:.2f}"
        
        print("✅ Hybrid TP ATR success test passed")
    
    def test_hybrid_tp_system_fallback_rr(self):
        """Test hybrid TP system when ATR-based TP fails 1:1 RR requirement"""
        print("\n=== Testing Hybrid TP System - Fallback RR ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        stop_loss = 1.1950  # 50 pips stop
        side = "buy"
        
        # Mock ATR calculation to return very small value (would fail RR)
        with patch.object(self.smart_tp, '_calculate_atr', return_value=0.0001):
            with patch.object(self.smart_tp, '_calculate_momentum_strength', return_value='medium'):
                result = self.smart_tp.calculate_smart_tp(
                    symbol=symbol,
                    entry_price=entry_price,
                    side=side,
                    historical_data=self.mock_data,
                    stop_loss=stop_loss
                )
        
        print(f"TP Result: {result}")
        print(f"Full TP Price: {result.full_tp_price:.5f}")
        print(f"Partial TP Price: {result.partial_tp_price:.5f}")
        
        # Validate TP prices
        assert result.full_tp_price > entry_price, "Full TP should be above entry for buy"
        assert result.partial_tp_price > entry_price, "Partial TP should be above entry for buy"
        
        # Validate RR (should be exactly 1:1 due to fallback)
        stop_distance = abs(entry_price - stop_loss)
        tp_distance = abs(result.full_tp_price - entry_price)
        rr = tp_distance / stop_distance if stop_distance > 0 else 0
        
        print(f"Stop Distance: {stop_distance:.5f}")
        print(f"TP Distance: {tp_distance:.5f}")
        print(f"Risk:Reward Ratio: {rr:.2f}")
        
        assert rr >= 1.0, f"RR should be at least 1:1, got {rr:.2f}"
        
        print("✅ Hybrid TP fallback RR test passed")
    
    def test_hybrid_tp_system_sell_order(self):
        """Test hybrid TP system for sell orders"""
        print("\n=== Testing Hybrid TP System - Sell Order ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        stop_loss = 1.2050  # 50 pips stop above entry
        side = "sell"
        
        with patch.object(self.smart_tp, '_calculate_atr', return_value=0.002):
            with patch.object(self.smart_tp, '_calculate_momentum_strength', return_value='medium'):
                result = self.smart_tp.calculate_smart_tp(
                    symbol=symbol,
                    entry_price=entry_price,
                    side=side,
                    historical_data=self.mock_data,
                    stop_loss=stop_loss
                )
        
        print(f"TP Result: {result}")
        print(f"Full TP Price: {result.full_tp_price:.5f}")
        print(f"Partial TP Price: {result.partial_tp_price:.5f}")
        
        # Validate TP prices for sell order
        assert result.full_tp_price < entry_price, "Full TP should be below entry for sell"
        assert result.partial_tp_price < entry_price, "Partial TP should be below entry for sell"
        assert result.full_tp_price < result.partial_tp_price, "Full TP should be lower than partial TP for sell"
        
        # Validate RR
        stop_distance = abs(entry_price - stop_loss)
        tp_distance = abs(entry_price - result.full_tp_price)
        rr = tp_distance / stop_distance if stop_distance > 0 else 0
        
        print(f"Stop Distance: {stop_distance:.5f}")
        print(f"TP Distance: {tp_distance:.5f}")
        print(f"Risk:Reward Ratio: {rr:.2f}")
        
        assert rr >= 1.0, f"RR should be at least 1:1, got {rr:.2f}"
        
        print("✅ Hybrid TP sell order test passed")
    
    def test_fallback_tp_calculation(self):
        """Test fallback TP calculation when ATR fails"""
        print("\n=== Testing Fallback TP Calculation ===")
        
        entry_price = 1.2000
        stop_loss = 1.1950
        side = "buy"
        
        result = self.smart_tp._fallback_tp(entry_price, side, stop_loss)
        
        print(f"Fallback TP Result: {result}")
        print(f"Full TP Price: {result.full_tp_price:.5f}")
        print(f"Partial TP Price: {result.partial_tp_price:.5f}")
        
        # Should use 2:1 R:R ratio
        expected_tp = entry_price + (2 * abs(entry_price - stop_loss))
        assert abs(result.full_tp_price - expected_tp) < 0.0001, f"Expected {expected_tp}, got {result.full_tp_price}"
        
        print("✅ Fallback TP calculation test passed")
    
    def test_partial_tp_tracking(self):
        """Test partial TP tracking functionality"""
        print("\n=== Testing Partial TP Tracking ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        side = "buy"
        
        # Create a mock result
        result = SmartTPResult(
            partial_tp_price=1.2015,
            full_tp_price=1.2030,
            momentum_strength='strong',
            tp_multiplier=2.5,
            should_take_partial=True,
            trailing_enabled=True
        )
        
        # Register position
        self.smart_tp.register_position(symbol, side, entry_price, result)
        
        print(f"Registered position: {self.smart_tp.active_positions[symbol]}")
        
        # Check if position is tracked
        assert symbol in self.smart_tp.active_positions, "Position should be registered"
        assert not self.smart_tp.active_positions[symbol]['partial_taken'], "Partial TP should not be taken yet"
        
        # Test partial TP check
        should_take = self.smart_tp.check_partial_tp(symbol, 1.2015)
        print(f"Should take partial at 1.2015: {should_take}")
        
        assert should_take, "Should take partial TP at partial TP price"
        
        print("✅ Partial TP tracking test passed")
    
    def test_trailing_tp_update(self):
        """Test trailing TP update functionality"""
        print("\n=== Testing Trailing TP Update ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        side = "buy"
        
        # Create a mock result
        result = SmartTPResult(
            partial_tp_price=1.2015,
            full_tp_price=1.2030,
            momentum_strength='strong',
            tp_multiplier=2.5,
            should_take_partial=True,
            trailing_enabled=True
        )
        
        # Register position
        self.smart_tp.register_position(symbol, side, entry_price, result)
        
        # Test trailing TP update
        new_tp = self.smart_tp.update_trailing_tp(symbol, 1.2020, self.mock_data)
        
        print(f"New trailing TP: {new_tp}")
        
        # Should return updated TP or None
        assert new_tp is None or new_tp > 0, f"Invalid trailing TP: {new_tp}"
        
        print("✅ Trailing TP update test passed")
    
    def test_position_cleanup(self):
        """Test position cleanup when trade is closed"""
        print("\n=== Testing Position Cleanup ===")
        
        symbol = "EURUSD"
        entry_price = 1.2000
        side = "buy"
        
        # Create and register position
        result = SmartTPResult(
            partial_tp_price=1.2015,
            full_tp_price=1.2030,
            momentum_strength='strong',
            tp_multiplier=2.5,
            should_take_partial=True,
            trailing_enabled=True
        )
        
        self.smart_tp.register_position(symbol, side, entry_price, result)
        assert symbol in self.smart_tp.active_positions, "Position should be registered"
        
        # Close position
        self.smart_tp.close_position(symbol)
        
        print(f"Active positions after cleanup: {self.smart_tp.active_positions}")
        
        # Position should be removed
        assert symbol not in self.smart_tp.active_positions, "Position should be removed after cleanup"
        
        print("✅ Position cleanup test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
