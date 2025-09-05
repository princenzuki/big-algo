#!/usr/bin/env python3
"""
Simple test to verify ATR calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.smart_tp import SmartTakeProfit, SmartTPConfig

def test_atr():
    """Test ATR calculation with known data"""
    
    # Create test data with known volatility
    data = []
    base_price = 1.2000
    
    # Create 50 bars with increasing volatility
    for i in range(50):
        if i < 25:
            # Low volatility
            volatility = 0.0005
        else:
            # High volatility
            volatility = 0.0020
            
        price_change = np.random.normal(0, volatility)
        base_price += price_change
        
        high = base_price + abs(np.random.normal(0, volatility/2))
        low = base_price - abs(np.random.normal(0, volatility/2))
        close = base_price + np.random.normal(0, volatility/4)
        
        data.append({
            'open': base_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    # Test Smart TP
    smart_tp = SmartTakeProfit(SmartTPConfig())
    df = pd.DataFrame(data)
    
    # Calculate ATR
    atr = smart_tp._calculate_atr(df)
    print(f"ATR calculated: {atr:.6f}")
    
    # Test momentum
    momentum = smart_tp._calculate_momentum_strength(df)
    print(f"Momentum: {momentum}")
    
    # Test full calculation
    result = smart_tp.calculate_smart_tp(
        symbol="TEST",
        entry_price=1.2000,
        side="buy",
        historical_data=data,
        stop_loss=1.1950
    )
    
    print(f"TP Multiplier: {result.tp_multiplier}")
    print(f"Full TP: {result.full_tp_price:.5f}")
    print(f"TP Distance: {result.full_tp_price - 1.2000:.5f}")
    print(f"Using ATR: {atr > 0}")

if __name__ == "__main__":
    test_atr()
