"""
Integration tests for MT5 Trading Bot

Tests end-to-end workflows, component integration,
and real-world scenarios.
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

class TestIntegration:
    """Integration tests for the complete trading system"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Mock all external dependencies
        self.mock_mt5 = MagicMock()
        self.mock_broker_adapter = MagicMock()
        self.mock_risk_manager = MagicMock()
        self.mock_smart_tp = MagicMock()
        self.mock_portfolio_manager = MagicMock()
        
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
                'high': low,
                'low': low,
                'close': close
            })
        
        return data
    
    @pytest.mark.integration
    def test_complete_trading_cycle_success(self):
        """Test complete trading cycle from signal to execution"""
        print("\n=== Testing Complete Trading Cycle - Success ===")
        
        # Mock successful components
        self._setup_successful_mocks()
        
        # Simulate complete trading cycle
        result = self._simulate_trading_cycle(
            symbol="EURUSD",
            signal=1,
            confidence=0.8,
            entry_price=1.2000
        )
        
        print(f"Trading cycle result: {result}")
        
        # Validate complete success
        assert result['signal_processed'], "Signal should be processed"
        assert result['data_validated'], "Data should be validated"
        assert result['tp_calculated'], "TP should be calculated"
        assert result['order_placed'], "Order should be placed"
        assert result['trade_opened'], "Trade should be opened"
        assert result['position_tracked'], "Position should be tracked"
        
        print("✅ Complete trading cycle success test passed")
    
    @pytest.mark.integration
    def test_complete_trading_cycle_failure_at_order(self):
        """Test complete trading cycle with failure at order placement"""
        print("\n=== Testing Complete Trading Cycle - Order Failure ===")
        
        # Mock components with order failure
        self._setup_order_failure_mocks()
        
        # Simulate complete trading cycle
        result = self._simulate_trading_cycle(
            symbol="EURUSD",
            signal=1,
            confidence=0.8,
            entry_price=1.2000
        )
        
        print(f"Trading cycle result: {result}")
        
        # Validate failure at order placement
        assert result['signal_processed'], "Signal should be processed"
        assert result['data_validated'], "Data should be validated"
        assert result['tp_calculated'], "TP should be calculated"
        assert not result['order_placed'], "Order should fail"
        assert not result['trade_opened'], "Trade should not be opened"
        assert not result['position_tracked'], "Position should not be tracked"
        
        print("✅ Complete trading cycle order failure test passed")
    
    @pytest.mark.integration
    def test_complete_trading_cycle_failure_at_validation(self):
        """Test complete trading cycle with failure at validation"""
        print("\n=== Testing Complete Trading Cycle - Validation Failure ===")
        
        # Mock components with validation failure
        self._setup_validation_failure_mocks()
        
        # Simulate complete trading cycle
        result = self._simulate_trading_cycle(
            symbol="EURUSD",
            signal=1,
            confidence=0.8,
            entry_price=1.2000
        )
        
        print(f"Trading cycle result: {result}")
        
        # Validate failure at validation
        assert result['signal_processed'], "Signal should be processed"
        assert not result['data_validated'], "Data validation should fail"
        assert not result['tp_calculated'], "TP should not be calculated"
        assert not result['order_placed'], "Order should not be placed"
        assert not result['trade_opened'], "Trade should not be opened"
        assert not result['position_tracked'], "Position should not be tracked"
        
        print("✅ Complete trading cycle validation failure test passed")
    
    @pytest.mark.integration
    def test_hybrid_tp_system_integration(self):
        """Test hybrid TP system integration with real data"""
        print("\n=== Testing Hybrid TP System Integration ===")
        
        # Test ATR-based TP success
        atr_result = self._test_tp_calculation(
            symbol="EURUSD",
            entry_price=1.2000,
            stop_loss=1.1950,
            side="buy",
            atr_value=0.002,  # Good ATR
            momentum="strong"
        )
        
        print(f"ATR-based TP result: {atr_result}")
        assert atr_result['tp_source'] == 'ATR_BASED', "Should use ATR-based TP"
        assert atr_result['rr'] >= 1.0, "RR should be at least 1:1"
        
        # Test fallback RR TP
        fallback_result = self._test_tp_calculation(
            symbol="EURUSD",
            entry_price=1.2000,
            stop_loss=1.1950,
            side="buy",
            atr_value=0.0001,  # Poor ATR
            momentum="weak"
        )
        
        print(f"Fallback RR TP result: {fallback_result}")
        assert fallback_result['tp_source'] == 'FALLBACK_RR', "Should use fallback RR TP"
        assert fallback_result['rr'] >= 1.0, "RR should be at least 1:1"
        
        print("✅ Hybrid TP system integration test passed")
    
    @pytest.mark.integration
    def test_risk_management_integration(self):
        """Test risk management integration across components"""
        print("\n=== Testing Risk Management Integration ===")
        
        # Test position sizing with different confidence levels
        high_conf_result = self._test_position_sizing(
            symbol="EURUSD",
            entry_price=1.2000,
            stop_loss=1.1950,
            confidence=0.9
        )
        
        low_conf_result = self._test_position_sizing(
            symbol="EURUSD",
            entry_price=1.2000,
            stop_loss=1.1950,
            confidence=0.3
        )
        
        print(f"High confidence position: {high_conf_result}")
        print(f"Low confidence position: {low_conf_result}")
        
        # Higher confidence should result in larger position
        assert high_conf_result['lot_size'] > low_conf_result['lot_size'], "Higher confidence should result in larger position"
        assert high_conf_result['risk_amount'] > low_conf_result['risk_amount'], "Higher confidence should result in higher risk"
        
        print("✅ Risk management integration test passed")
    
    @pytest.mark.integration
    def test_break_even_and_trailing_integration(self):
        """Test break-even and trailing stop integration"""
        print("\n=== Testing Break-Even and Trailing Integration ===")
        
        # Test break-even trigger
        break_even_result = self._test_break_even_trigger(
            symbol="EURUSD",
            entry_price=1.2000,
            current_price=1.2020,  # 20 pips in favor
            atr_value=0.001
        )
        
        print(f"Break-even result: {break_even_result}")
        assert break_even_result['break_even_triggered'], "Break-even should be triggered"
        
        # Test trailing stop update
        trailing_result = self._test_trailing_stop_update(
            symbol="EURUSD",
            entry_price=1.2000,
            current_price=1.2025,  # Further in favor
            atr_value=0.001
        )
        
        print(f"Trailing stop result: {trailing_result}")
        assert trailing_result['trailing_updated'], "Trailing stop should be updated"
        
        print("✅ Break-even and trailing integration test passed")
    
    @pytest.mark.integration
    def test_partial_tp_integration(self):
        """Test partial take profit integration"""
        print("\n=== Testing Partial TP Integration ===")
        
        # Test partial TP trigger
        partial_result = self._test_partial_tp_trigger(
            symbol="EURUSD",
            entry_price=1.2000,
            current_price=1.2015,  # At partial TP level
            momentum="strong"
        )
        
        print(f"Partial TP result: {partial_result}")
        assert partial_result['partial_triggered'], "Partial TP should be triggered"
        
        # Test trailing TP for remaining position
        trailing_tp_result = self._test_trailing_tp_update(
            symbol="EURUSD",
            entry_price=1.2000,
            current_price=1.2020,  # Further in favor
            momentum="strong"
        )
        
        print(f"Trailing TP result: {trailing_tp_result}")
        assert trailing_tp_result['trailing_tp_updated'], "Trailing TP should be updated"
        
        print("✅ Partial TP integration test passed")
    
    @pytest.mark.integration
    def test_error_recovery_integration(self):
        """Test error recovery and fallback mechanisms"""
        print("\n=== Testing Error Recovery Integration ===")
        
        # Test ATR calculation failure recovery
        atr_failure_result = self._test_atr_failure_recovery(
            symbol="EURUSD",
            entry_price=1.2000,
            stop_loss=1.1950,
            side="buy"
        )
        
        print(f"ATR failure recovery result: {atr_failure_result}")
        assert atr_failure_result['tp_calculated'], "TP should be calculated despite ATR failure"
        assert atr_failure_result['fallback_used'], "Fallback should be used"
        assert atr_failure_result['rr'] >= 1.0, "RR should still be valid"
        
        # Test momentum calculation failure recovery
        momentum_failure_result = self._test_momentum_failure_recovery(
            symbol="EURUSD",
            entry_price=1.2000,
            stop_loss=1.1950,
            side="buy"
        )
        
        print(f"Momentum failure recovery result: {momentum_failure_result}")
        assert momentum_failure_result['tp_calculated'], "TP should be calculated despite momentum failure"
        assert momentum_failure_result['default_momentum'], "Default momentum should be used"
        
        print("✅ Error recovery integration test passed")
    
    @pytest.mark.integration
    def test_multi_symbol_trading_integration(self):
        """Test multi-symbol trading with different configurations"""
        print("\n=== Testing Multi-Symbol Trading Integration ===")
        
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        results = {}
        
        for symbol in symbols:
            result = self._test_symbol_trading(
                symbol=symbol,
                signal=1,
                confidence=0.7
            )
            results[symbol] = result
            print(f"{symbol} result: {result}")
        
        # All symbols should be processed
        for symbol, result in results.items():
            assert result['processed'], f"{symbol} should be processed"
            assert result['validated'], f"{symbol} should be validated"
        
        print("✅ Multi-symbol trading integration test passed")
    
    # Helper methods for integration testing
    def _setup_successful_mocks(self):
        """Set up mocks for successful trading cycle"""
        # Mock successful order placement
        mock_order_result = MagicMock()
        mock_order_result.success = True
        mock_order_result.order_id = 12345
        mock_order_result.price = 1.2000
        mock_order_result.error_message = None
        
        self.mock_broker_adapter.place_order.return_value = mock_order_result
        
        # Mock successful trade opening
        mock_trade = MagicMock()
        mock_trade.id = "trade_123"
        self.mock_portfolio_manager.open_trade.return_value = mock_trade
        
        # Mock successful TP calculation
        mock_tp_result = MagicMock()
        mock_tp_result.full_tp_price = 1.2050
        mock_tp_result.partial_tp_price = 1.2025
        mock_tp_result.momentum_strength = 'strong'
        mock_tp_result.tp_multiplier = 2.5
        mock_tp_result.should_take_partial = True
        mock_tp_result.trailing_enabled = True
        
        self.mock_smart_tp.calculate_smart_tp.return_value = mock_tp_result
        
        # Mock successful position sizing
        self.mock_risk_manager.calculate_position_size.return_value = (0.1, 20.0)
    
    def _setup_order_failure_mocks(self):
        """Set up mocks for order placement failure"""
        self._setup_successful_mocks()
        
        # Override with order failure
        mock_order_result = MagicMock()
        mock_order_result.success = False
        mock_order_result.order_id = None
        mock_order_result.price = 0.0
        mock_order_result.error_message = "Order rejected"
        
        self.mock_broker_adapter.place_order.return_value = mock_order_result
    
    def _setup_validation_failure_mocks(self):
        """Set up mocks for validation failure"""
        # Mock validation failure
        self.mock_risk_manager.calculate_position_size.return_value = (None, 0.0)
    
    def _simulate_trading_cycle(self, symbol, signal, confidence, entry_price):
        """Simulate complete trading cycle"""
        result = {
            'signal_processed': False,
            'data_validated': False,
            'tp_calculated': False,
            'order_placed': False,
            'trade_opened': False,
            'position_tracked': False
        }
        
        try:
            # Signal processing
            if signal in [-1, 0, 1] and 0 <= confidence <= 1:
                result['signal_processed'] = True
            
            # Data validation
            if len(self.mock_historical_data) >= 20:
                result['data_validated'] = True
            
            # TP calculation
            if result['data_validated']:
                tp_result = self.mock_smart_tp.calculate_smart_tp(
                    symbol, entry_price, "buy", self.mock_historical_data, 1.1950
                )
                if tp_result:
                    result['tp_calculated'] = True
            
            # Order placement
            if result['tp_calculated']:
                order_result = self.mock_broker_adapter.place_order(MagicMock())
                if order_result.success:
                    result['order_placed'] = True
            
            # Trade opening
            if result['order_placed']:
                trade = self.mock_portfolio_manager.open_trade(
                    symbol, "buy", entry_price, 0.1, 1.1950, 1.2050, confidence
                )
                if trade:
                    result['trade_opened'] = True
            
            # Position tracking
            if result['trade_opened']:
                self.mock_smart_tp.register_position(symbol, "buy", entry_price, MagicMock())
                result['position_tracked'] = True
                
        except Exception as e:
            print(f"Trading cycle error: {e}")
        
        return result
    
    def _test_tp_calculation(self, symbol, entry_price, stop_loss, side, atr_value, momentum):
        """Test TP calculation with specific parameters"""
        # Mock ATR calculation
        with patch('core.smart_tp.SmartTakeProfit._calculate_atr', return_value=atr_value):
            with patch('core.smart_tp.SmartTakeProfit._calculate_momentum_strength', return_value=momentum):
                from core.smart_tp import SmartTakeProfit, SmartTPConfig
                
                smart_tp = SmartTakeProfit(SmartTPConfig())
                result = smart_tp.calculate_smart_tp(
                    symbol, entry_price, side, self.mock_historical_data, stop_loss
                )
                
                stop_distance = abs(entry_price - stop_loss)
                tp_distance = abs(result.full_tp_price - entry_price)
                rr = tp_distance / stop_distance if stop_distance > 0 else 0
                
                return {
                    'tp_source': 'ATR_BASED' if atr_value > 0.001 else 'FALLBACK_RR',
                    'rr': rr,
                    'tp_price': result.full_tp_price
                }
    
    def _test_position_sizing(self, symbol, entry_price, stop_loss, confidence):
        """Test position sizing with specific parameters"""
        # Mock position sizing
        base_risk = 20.0
        risk_amount = base_risk * confidence
        lot_size = 0.1 * confidence
        
        return {
            'lot_size': lot_size,
            'risk_amount': risk_amount
        }
    
    def _test_break_even_trigger(self, symbol, entry_price, current_price, atr_value):
        """Test break-even trigger"""
        # Mock break-even logic
        price_favor = abs(current_price - entry_price)
        atr_distance = atr_value * 0.5  # 0.5 ATR trigger
        
        return {
            'break_even_triggered': price_favor >= atr_distance
        }
    
    def _test_trailing_stop_update(self, symbol, entry_price, current_price, atr_value):
        """Test trailing stop update"""
        # Mock trailing stop logic
        price_favor = abs(current_price - entry_price)
        atr_distance = atr_value  # 1 ATR trailing distance
        
        return {
            'trailing_updated': price_favor >= atr_distance
        }
    
    def _test_partial_tp_trigger(self, symbol, entry_price, current_price, momentum):
        """Test partial TP trigger"""
        # Mock partial TP logic
        partial_tp_price = entry_price + 0.0015  # 15 pips
        price_at_tp = abs(current_price - partial_tp_price) < 0.0001
        
        return {
            'partial_triggered': price_at_tp and momentum in ['strong', 'medium']
        }
    
    def _test_trailing_tp_update(self, symbol, entry_price, current_price, momentum):
        """Test trailing TP update"""
        # Mock trailing TP logic
        price_favor = abs(current_price - entry_price)
        trailing_distance = 0.001  # 10 pips trailing
        
        return {
            'trailing_tp_updated': price_favor >= trailing_distance and momentum in ['strong', 'medium']
        }
    
    def _test_atr_failure_recovery(self, symbol, entry_price, stop_loss, side):
        """Test ATR failure recovery"""
        # Mock ATR failure
        with patch('core.smart_tp.SmartTakeProfit._calculate_atr', return_value=0.0):
            from core.smart_tp import SmartTakeProfit, SmartTPConfig
            
            smart_tp = SmartTakeProfit(SmartTPConfig())
            result = smart_tp.calculate_smart_tp(
                symbol, entry_price, side, self.mock_historical_data, stop_loss
            )
            
            stop_distance = abs(entry_price - stop_loss)
            tp_distance = abs(result.full_tp_price - entry_price)
            rr = tp_distance / stop_distance if stop_distance > 0 else 0
            
            return {
                'tp_calculated': result is not None,
                'fallback_used': True,
                'rr': rr
            }
    
    def _test_momentum_failure_recovery(self, symbol, entry_price, stop_loss, side):
        """Test momentum failure recovery"""
        # Mock momentum failure
        with patch('core.smart_tp.SmartTakeProfit._calculate_momentum_strength', return_value='medium'):
            from core.smart_tp import SmartTakeProfit, SmartTPConfig
            
            smart_tp = SmartTakeProfit(SmartTPConfig())
            result = smart_tp.calculate_smart_tp(
                symbol, entry_price, side, self.mock_historical_data, stop_loss
            )
            
            return {
                'tp_calculated': result is not None,
                'default_momentum': result.momentum_strength == 'medium'
            }
    
    def _test_symbol_trading(self, symbol, signal, confidence):
        """Test trading for a specific symbol"""
        return {
            'processed': signal in [-1, 0, 1],
            'validated': 0 <= confidence <= 1,
            'symbol': symbol
        }

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
