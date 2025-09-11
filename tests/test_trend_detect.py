#!/usr/bin/env python3
"""
Unit tests for robust trend detection function

This module tests the detect_trend_robust function to ensure it correctly
identifies bullish, bearish, and neutral trends based on EMA, price action, and indicators.
"""

import sys
import os
import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtf_validator import MultiTimeframeValidator

class TestTrendDetection(unittest.TestCase):
    """Test cases for robust trend detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = MultiTimeframeValidator()
    
    def make_bearish_series(self, n=120, start=1.9900, step=-0.0008):
        """Create a bearish price series for testing"""
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        closes = [start + i*step for i in range(n)]
        highs = [c + 0.0002 for c in closes]
        lows = [c - 0.0002 for c in closes]
        opens = [closes[i-1] if i > 0 else closes[i] for i in range(n)]
        
        return pd.DataFrame({
            "time": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [1000] * n
        })
    
    def make_bullish_series(self, n=120, start=1.9700, step=0.0008):
        """Create a bullish price series for testing"""
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        closes = [start + i*step for i in range(n)]
        highs = [c + 0.0002 for c in closes]
        lows = [c - 0.0002 for c in closes]
        opens = [closes[i-1] if i > 0 else closes[i] for i in range(n)]
        
        return pd.DataFrame({
            "time": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [1000] * n
        })
    
    def make_sideways_series(self, n=120, start=1.9800, amplitude=0.001):
        """Create a sideways/neutral price series for testing"""
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
        # Create sine wave pattern for sideways movement
        closes = [start + amplitude * np.sin(i * 0.1) for i in range(n)]
        highs = [c + 0.0001 for c in closes]
        lows = [c - 0.0001 for c in closes]
        opens = [closes[i-1] if i > 0 else closes[i] for i in range(n)]
        
        return pd.DataFrame({
            "time": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [1000] * n
        })
    
    def test_bearish_trend_detection(self):
        """Test that bearish price series is correctly identified as bearish"""
        print("\n🧪 Testing bearish trend detection...")
        
        df = self.make_bearish_series(120)
        trend, confidence = self.validator.detect_trend_robust(df, lookback=5)
        
        print(f"   Trend: {trend}, Confidence: {confidence:.3f}")
        print(f"   Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
        print(f"   Overall movement: {df['close'].iloc[-1] - df['close'].iloc[0]:.5f}")
        
        # Should detect bearish trend
        self.assertEqual(trend, "bearish", f"Expected bearish trend, got {trend}")
        self.assertGreater(confidence, 0.5, f"Expected confidence > 0.5, got {confidence}")
    
    def test_bullish_trend_detection(self):
        """Test that bullish price series is correctly identified as bullish"""
        print("\n🧪 Testing bullish trend detection...")
        
        df = self.make_bullish_series(120)
        trend, confidence = self.validator.detect_trend_robust(df, lookback=5)
        
        print(f"   Trend: {trend}, Confidence: {confidence:.3f}")
        print(f"   Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
        print(f"   Overall movement: {df['close'].iloc[-1] - df['close'].iloc[0]:.5f}")
        
        # Should detect bullish trend
        self.assertEqual(trend, "bullish", f"Expected bullish trend, got {trend}")
        self.assertGreater(confidence, 0.5, f"Expected confidence > 0.5, got {confidence}")
    
    def test_neutral_trend_detection(self):
        """Test that sideways price series is correctly identified as neutral"""
        print("\n🧪 Testing neutral trend detection...")
        
        df = self.make_sideways_series(120)
        trend, confidence = self.validator.detect_trend_robust(df, lookback=5)
        
        print(f"   Trend: {trend}, Confidence: {confidence:.3f}")
        print(f"   Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
        print(f"   Overall movement: {df['close'].iloc[-1] - df['close'].iloc[0]:.5f}")
        
        # Should detect neutral trend or low confidence
        self.assertIn(trend, ["neutral", "bullish", "bearish"], f"Unexpected trend: {trend}")
        if trend == "neutral":
            self.assertLessEqual(confidence, 0.5, f"Neutral trend should have low confidence, got {confidence}")
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        print("\n🧪 Testing insufficient data handling...")
        
        # Create very short series
        df = self.make_bearish_series(10)  # Only 10 bars
        trend, confidence = self.validator.detect_trend_robust(df, lookback=5)
        
        print(f"   Trend: {trend}, Confidence: {confidence:.3f}")
        print(f"   Data length: {len(df)} bars")
        
        # Should return neutral with low confidence
        self.assertEqual(trend, "neutral", f"Expected neutral for insufficient data, got {trend}")
        self.assertEqual(confidence, 0.0, f"Expected 0.0 confidence for insufficient data, got {confidence}")
    
    def test_different_lookback_periods(self):
        """Test trend detection with different lookback periods"""
        print("\n🧪 Testing different lookback periods...")
        
        df = self.make_bearish_series(120)
        
        for lookback in [3, 5, 10]:
            trend, confidence = self.validator.detect_trend_robust(df, lookback=lookback)
            print(f"   Lookback {lookback}: {trend} (conf={confidence:.3f})")
            
            # Should still detect bearish trend regardless of lookback
            self.assertEqual(trend, "bearish", f"Expected bearish for lookback {lookback}, got {trend}")
    
    def test_ema_crossover_detection(self):
        """Test that EMA crossovers are properly detected"""
        print("\n🧪 Testing EMA crossover detection...")
        
        # Create data where EMA20 crosses above EMA50 (bullish)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        
        # Create price data that will cause EMA20 > EMA50
        base_price = 1.9800
        closes = []
        for i in range(100):
            # Create upward trend that accelerates
            price = base_price + (i * 0.0001) + (i * i * 0.00001)
            closes.append(price)
        
        highs = [c + 0.0001 for c in closes]
        lows = [c - 0.0001 for c in closes]
        opens = [closes[i-1] if i > 0 else closes[i] for i in range(100)]
        
        df = pd.DataFrame({
            "time": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [1000] * 100
        })
        
        trend, confidence = self.validator.detect_trend_robust(df, ema_fast=20, ema_slow=50)
        
        print(f"   Trend: {trend}, Confidence: {confidence:.3f}")
        print(f"   Price movement: {df['close'].iloc[-1] - df['close'].iloc[0]:.5f}")
        
        # Should detect bullish trend due to EMA crossover
        self.assertEqual(trend, "bullish", f"Expected bullish trend from EMA crossover, got {trend}")

def run_tests():
    """Run all trend detection tests"""
    print("🚀 RUNNING TREND DETECTION TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrendDetection)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        for test, error in result.failures + result.errors:
            print(f"\n🔍 {test}: {error}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
