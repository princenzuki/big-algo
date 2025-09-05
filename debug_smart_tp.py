#!/usr/bin/env python3
"""
Debug script to test Smart Take Profit ATR calculations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.smart_tp import SmartTakeProfit, SmartTPConfig
from core.risk import get_intelligent_sl

def test_atr_calculation():
    """Test ATR calculation with sample data"""
    
    # Create sample historical data with different volatility patterns
    np.random.seed(42)
    
    # Low volatility data
    low_vol_data = []
    base_price = 1.2000
    for i in range(100):
        price_change = np.random.normal(0, 0.0005)  # Low volatility
        base_price += price_change
        high = base_price + abs(np.random.normal(0, 0.0002))
        low = base_price - abs(np.random.normal(0, 0.0002))
        close = base_price + np.random.normal(0, 0.0001)
        low_vol_data.append({
            'open': base_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    # High volatility data
    high_vol_data = []
    base_price = 1.2000
    for i in range(100):
        price_change = np.random.normal(0, 0.002)  # High volatility
        base_price += price_change
        high = base_price + abs(np.random.normal(0, 0.001))
        low = base_price - abs(np.random.normal(0, 0.001))
        close = base_price + np.random.normal(0, 0.0005)
        high_vol_data.append({
            'open': base_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    # Initialize Smart TP
    smart_tp = SmartTakeProfit(SmartTPConfig())
    
    print("=== ATR CALCULATION TEST ===")
    print()
    
    # Test low volatility
    print("1. LOW VOLATILITY DATA:")
    atr_low = smart_tp._calculate_atr(pd.DataFrame(low_vol_data))
    print(f"   ATR: {atr_low:.6f}")
    
    momentum_low = smart_tp._calculate_momentum_strength(pd.DataFrame(low_vol_data))
    print(f"   Momentum: {momentum_low}")
    
    # Test high volatility
    print("\n2. HIGH VOLATILITY DATA:")
    atr_high = smart_tp._calculate_atr(pd.DataFrame(high_vol_data))
    print(f"   ATR: {atr_high:.6f}")
    
    momentum_high = smart_tp._calculate_momentum_strength(pd.DataFrame(high_vol_data))
    print(f"   Momentum: {momentum_high}")
    
    # Test Smart TP calculation
    print("\n3. SMART TP CALCULATION TEST:")
    
    # Low volatility trade
    entry_price = 1.2000
    stop_loss = 1.1950
    
    result_low = smart_tp.calculate_smart_tp(
        symbol="EURUSDm",
        entry_price=entry_price,
        side="buy",
        historical_data=low_vol_data,
        stop_loss=stop_loss
    )
    
    print(f"   LOW VOLATILITY TRADE:")
    print(f"   - Entry: {entry_price:.5f}")
    print(f"   - Stop Loss: {stop_loss:.5f}")
    print(f"   - ATR: {atr_low:.6f}")
    print(f"   - Momentum: {result_low.momentum_strength}")
    print(f"   - TP Multiplier: {result_low.tp_multiplier}")
    print(f"   - Full TP: {result_low.full_tp_price:.5f}")
    print(f"   - Partial TP: {result_low.partial_tp_price:.5f}")
    print(f"   - TP Distance: {result_low.full_tp_price - entry_price:.5f}")
    
    # High volatility trade
    result_high = smart_tp.calculate_smart_tp(
        symbol="EURUSDm",
        entry_price=entry_price,
        side="buy",
        historical_data=high_vol_data,
        stop_loss=stop_loss
    )
    
    print(f"\n   HIGH VOLATILITY TRADE:")
    print(f"   - Entry: {entry_price:.5f}")
    print(f"   - Stop Loss: {stop_loss:.5f}")
    print(f"   - ATR: {atr_high:.6f}")
    print(f"   - Momentum: {result_high.momentum_strength}")
    print(f"   - TP Multiplier: {result_high.tp_multiplier}")
    print(f"   - Full TP: {result_high.full_tp_price:.5f}")
    print(f"   - Partial TP: {result_high.partial_tp_price:.5f}")
    print(f"   - TP Distance: {result_high.full_tp_price - entry_price:.5f}")
    
    # Compare results
    print(f"\n4. COMPARISON:")
    print(f"   ATR Ratio (High/Low): {atr_high/atr_low:.2f}x")
    print(f"   TP Distance Ratio: {(result_high.full_tp_price - entry_price)/(result_low.full_tp_price - entry_price):.2f}x")
    
    # Test with real data from logs
    print(f"\n5. REAL DATA SIMULATION:")
    print("   Simulating EURAUDm trade from logs...")
    
    # Simulate the EURAUDm data from your logs
    real_data = []
    base_price = 1.78788
    for i in range(100):
        # Simulate realistic EURAUD volatility
        price_change = np.random.normal(0, 0.0008)
        base_price += price_change
        high = base_price + abs(np.random.normal(0, 0.0004))
        low = base_price - abs(np.random.normal(0, 0.0004))
        close = base_price + np.random.normal(0, 0.0002)
        real_data.append({
            'open': base_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    atr_real = smart_tp._calculate_atr(pd.DataFrame(real_data))
    result_real = smart_tp.calculate_smart_tp(
        symbol="EURAUDm",
        entry_price=1.78788,
        side="buy",
        historical_data=real_data,
        stop_loss=1.78348
    )
    
    print(f"   - ATR: {atr_real:.6f}")
    print(f"   - Momentum: {result_real.momentum_strength}")
    print(f"   - TP Multiplier: {result_real.tp_multiplier}")
    print(f"   - Full TP: {result_real.full_tp_price:.5f}")
    print(f"   - TP Distance: {result_real.full_tp_price - 1.78788:.5f}")
    print(f"   - Expected from logs: 1.78841")
    print(f"   - Difference: {abs(result_real.full_tp_price - 1.78841):.5f}")

if __name__ == "__main__":
    test_atr_calculation()
