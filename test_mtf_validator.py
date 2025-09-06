"""
Test script for Multi-Timeframe Validator

This script demonstrates how the MTF validator works with sample data.
"""

import numpy as np
import pandas as pd
from mtf_validator import MultiTimeframeValidator, MTFValidationResult

def create_sample_data():
    """Create sample historical data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Create 100 bars of sample OHLC data
    base_price = 1.2000
    prices = [base_price]
    
    for i in range(99):
        # Random walk with slight upward bias
        change = np.random.normal(0.0001, 0.001)
        new_price = prices[-1] + change
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i, close in enumerate(prices[1:], 1):
        high = close + abs(np.random.normal(0, 0.0005))
        low = close - abs(np.random.normal(0, 0.0005))
        open_price = prices[i-1]
        volume = np.random.randint(1000, 5000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return data

def test_mtf_validator():
    """Test the MTF validator with sample data"""
    print("🧪 Testing Multi-Timeframe Validator")
    print("=" * 50)
    
    # Create validator
    validator = MultiTimeframeValidator()
    
    # Create sample data
    historical_data = create_sample_data()
    
    # Test scenarios
    test_cases = [
        {"signal": 1, "description": "Long signal (bullish)"},
        {"signal": -1, "description": "Short signal (bearish)"},
        {"signal": 0, "description": "Neutral signal"}
    ]
    
    for test_case in test_cases:
        signal = test_case["signal"]
        description = test_case["description"]
        
        print(f"\n📊 Testing: {description}")
        print("-" * 30)
        
        # Get current 1m data
        current_1m_data = historical_data[-1]
        
        # Validate signal
        result = validator.validate_trade_signal(
            symbol="EURUSD",
            signal=signal,
            current_1m_data=current_1m_data,
            historical_data=historical_data
        )
        
        # Display results
        print(f"Signal: {signal}")
        print(f"Allow Trade: {result.allow_trade}")
        print(f"Lot Multiplier: {result.lot_multiplier:.2f}")
        print(f"TP Multiplier: {result.tp_multiplier:.2f}")
        print(f"Confidence Boost: {result.confidence_boost:.2f}")
        print(f"Validation Score: {result.validation_score:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Timeframe Alignment: {result.timeframe_alignment}")

def test_validation_scenarios():
    """Test different validation scenarios"""
    print("\n🎯 Testing Validation Scenarios")
    print("=" * 50)
    
    validator = MultiTimeframeValidator()
    
    # Create sample data with different characteristics
    scenarios = [
        {
            "name": "Strong Bullish Trend",
            "data": create_trending_data("bullish"),
            "signal": 1
        },
        {
            "name": "Strong Bearish Trend", 
            "data": create_trending_data("bearish"),
            "signal": -1
        },
        {
            "name": "Mixed Signals",
            "data": create_mixed_data(),
            "signal": 1
        },
        {
            "name": "Weak Trend",
            "data": create_weak_trend_data(),
            "signal": 1
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📈 Scenario: {scenario['name']}")
        print("-" * 30)
        
        result = validator.validate_trade_signal(
            symbol="EURUSD",
            signal=scenario["signal"],
            current_1m_data=scenario["data"][-1],
            historical_data=scenario["data"]
        )
        
        print(f"Result: {'✅ ALLOW' if result.allow_trade else '❌ BLOCK'}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Score: {result.validation_score:.2f}")

def create_trending_data(direction: str):
    """Create data with strong trend"""
    np.random.seed(42)
    base_price = 1.2000
    prices = [base_price]
    
    trend_factor = 0.001 if direction == "bullish" else -0.001
    
    for i in range(99):
        change = np.random.normal(trend_factor, 0.0005)
        new_price = prices[-1] + change
        prices.append(new_price)
    
    data = []
    for i, close in enumerate(prices[1:], 1):
        high = close + abs(np.random.normal(0, 0.0003))
        low = close - abs(np.random.normal(0, 0.0003))
        open_price = prices[i-1]
        volume = np.random.randint(2000, 6000)  # Higher volume for trends
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return data

def create_mixed_data():
    """Create data with mixed signals"""
    np.random.seed(123)
    base_price = 1.2000
    prices = [base_price]
    
    for i in range(99):
        # Alternating trend
        trend_factor = 0.0005 if i % 20 < 10 else -0.0005
        change = np.random.normal(trend_factor, 0.001)
        new_price = prices[-1] + change
        prices.append(new_price)
    
    data = []
    for i, close in enumerate(prices[1:], 1):
        high = close + abs(np.random.normal(0, 0.0008))
        low = close - abs(np.random.normal(0, 0.0008))
        open_price = prices[i-1]
        volume = np.random.randint(1000, 3000)  # Lower volume for mixed signals
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return data

def create_weak_trend_data():
    """Create data with weak trend"""
    np.random.seed(456)
    base_price = 1.2000
    prices = [base_price]
    
    for i in range(99):
        # Very small trend with high noise
        change = np.random.normal(0.00005, 0.002)
        new_price = prices[-1] + change
        prices.append(new_price)
    
    data = []
    for i, close in enumerate(prices[1:], 1):
        high = close + abs(np.random.normal(0, 0.001))
        low = close - abs(np.random.normal(0, 0.001))
        open_price = prices[i-1]
        volume = np.random.randint(800, 2000)  # Low volume
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return data

if __name__ == "__main__":
    print("🚀 Multi-Timeframe Validator Test Suite")
    print("=" * 60)
    
    # Run basic tests
    test_mtf_validator()
    
    # Run scenario tests
    test_validation_scenarios()
    
    print("\n✅ All tests completed!")
    print("\n📝 Integration Notes:")
    print("- The MTF validator is designed to be non-intrusive")
    print("- It only adds one call before trade execution")
    print("- If validation fails, it fails safe to original algorithm")
    print("- All existing core logic remains unchanged")
