#!/usr/bin/env python3
"""
Test ML Parity between Pine Script and Python implementations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pine_script_reference import PineScriptReference, PineScriptSettings, PineScriptFilterSettings
from core.signals import LorentzianClassifier, Settings, FilterSettings
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_ml_parity():
    """Test ML parity between Pine Script and Python implementations"""
    print("=== Testing ML Parity ===")
    
    # Test data
    ohlc_data = {'open': 1.2000, 'high': 1.2010, 'low': 1.1990, 'close': 1.2005}
    historical_data = [{'open': 1.1995, 'high': 1.2005, 'low': 1.1990, 'close': 1.2000}]
    
    # Python implementation
    print("\n--- Python Implementation ---")
    python_classifier = LorentzianClassifier(Settings(), FilterSettings())
    python_result = python_classifier.generate_signal(ohlc_data, historical_data)
    print(f"Python Result: {python_result}")
    
    # Pine Script reference
    print("\n--- Pine Script Reference ---")
    pine_classifier = PineScriptReference(PineScriptSettings(), PineScriptFilterSettings())
    ohlc_arrays = {'open': [1.2000], 'high': [1.2010], 'low': [1.1990], 'close': [1.2005]}
    pine_result = pine_classifier.generate_signal(ohlc_arrays, historical_data)
    print(f"Pine Script Result: {pine_result}")
    
    # Parity check
    print("\n--- Parity Check ---")
    prediction_match = python_result['prediction'] == pine_result['prediction']
    signal_match = python_result['signal'] == pine_result['signal']
    confidence_match = abs(python_result['confidence'] - pine_result['confidence']) < 0.001
    neighbors_match = python_result['neighbors_count'] == pine_result['neighbors_count']
    
    print(f"Prediction Match: {prediction_match} ({python_result['prediction']} vs {pine_result['prediction']})")
    print(f"Signal Match: {signal_match} ({python_result['signal']} vs {pine_result['signal']})")
    print(f"Confidence Match: {confidence_match} ({python_result['confidence']:.3f} vs {pine_result['confidence']:.3f})")
    print(f"Neighbors Match: {neighbors_match} ({python_result['neighbors_count']} vs {pine_result['neighbors_count']})")
    
    overall_match = prediction_match and signal_match and confidence_match and neighbors_match
    print(f"\nOverall Parity: {'✅ PASS' if overall_match else '❌ FAIL'}")
    
    return overall_match

if __name__ == "__main__":
    test_ml_parity()
