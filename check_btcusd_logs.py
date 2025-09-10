#!/usr/bin/env python3
"""
Check BTCUSD Trading Logs

This script will help identify why BTCUSD isn't taking trades by checking:
1. Symbol processing in the main loop
2. Signal generation
3. Filter blocking
4. Configuration issues
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_btcusd_configuration():
    """Check BTCUSD configuration details"""
    print("=" * 60)
    print("🔍 CHECKING BTCUSD CONFIGURATION")
    print("=" * 60)
    
    # Check if BTCUSDm is in the enabled symbols
    enabled_symbols = settings_manager.get_enabled_symbols()
    print(f"📋 Total enabled symbols: {len(enabled_symbols)}")
    
    if 'BTCUSDm' in enabled_symbols:
        print("✅ BTCUSDm is enabled")
    else:
        print("❌ BTCUSDm is NOT in enabled symbols list")
        print(f"   Enabled symbols: {enabled_symbols}")
    
    # Check BTCUSDm specific config
    btc_config = settings_manager.get_symbol_config('BTCUSDm')
    if btc_config:
        print("\n📋 BTCUSDm Configuration:")
        for key, value in btc_config.items():
            print(f"   {key}: {value}")
        
        # Check critical settings
        min_confidence = btc_config.get('min_confidence', 0.4)
        max_spread = btc_config.get('max_spread_pips', 50.0)
        enabled = btc_config.get('enabled', True)
        
        print(f"\n🎯 Critical Settings:")
        print(f"   Enabled: {enabled}")
        print(f"   Min Confidence: {min_confidence}")
        print(f"   Max Spread: {max_spread} pips")
        
        if not enabled:
            print("❌ BTCUSDm is DISABLED in configuration!")
        else:
            print("✅ BTCUSDm is enabled in configuration")
    else:
        print("❌ No configuration found for BTCUSDm")

def check_symbol_processing():
    """Check how symbols are processed in the main loop"""
    print("\n" + "=" * 60)
    print("🔄 CHECKING SYMBOL PROCESSING")
    print("=" * 60)
    
    # Check the main trading loop logic
    print("🔍 Looking for symbol processing logic...")
    
    # Read main.py to see how symbols are processed
    try:
        with open('main.py', 'r') as f:
            content = f.read()
            
        # Look for symbol processing
        if 'BTCUSDm' in content:
            print("✅ BTCUSDm found in main.py")
        else:
            print("❌ BTCUSDm NOT found in main.py")
            
        # Look for symbol iteration
        if 'for symbol in' in content:
            print("✅ Found symbol iteration loop")
        else:
            print("❌ No symbol iteration found")
            
        # Look for enabled symbols check
        if 'get_enabled_symbols' in content:
            print("✅ Found enabled symbols check")
        else:
            print("❌ No enabled symbols check found")
            
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")

def check_common_blocking_reasons():
    """Check common reasons why BTCUSD might be blocked"""
    print("\n" + "=" * 60)
    print("🚦 CHECKING COMMON BLOCKING REASONS")
    print("=" * 60)
    
    # Check if it's a weekend and weekend trading is allowed
    is_weekend = datetime.now().weekday() >= 5
    btc_config = settings_manager.get_symbol_config('BTCUSDm')
    allow_weekend = btc_config.get('allow_weekend', False) if btc_config else False
    
    print(f"📅 Current day: {datetime.now().strftime('%A')}")
    print(f"📅 Is weekend: {is_weekend}")
    print(f"📅 Allow weekend trading: {allow_weekend}")
    
    if is_weekend and not allow_weekend:
        print("❌ Weekend trading not allowed for BTCUSDm")
    else:
        print("✅ Weekend trading allowed")
    
    # Check minimum confidence threshold
    min_confidence = btc_config.get('min_confidence', 0.4) if btc_config else 0.4
    print(f"📊 Minimum confidence required: {min_confidence}")
    print("💡 If ML confidence is below this, trades will be blocked")
    
    # Check spread limits
    max_spread = btc_config.get('max_spread_pips', 50.0) if btc_config else 50.0
    print(f"📊 Maximum spread allowed: {max_spread} pips")
    print("💡 If current spread exceeds this, trades will be blocked")

def check_mtf_validator():
    """Check if MTF validator might be blocking BTCUSD"""
    print("\n" + "=" * 60)
    print("🧠 CHECKING MTF VALIDATOR")
    print("=" * 60)
    
    # Check if BTCUSDm is in the forex pairs list for MTF weighting
    try:
        with open('mtf_validator.py', 'r') as f:
            content = f.read()
            
        if 'BTCUSDm' in content:
            print("✅ BTCUSDm found in MTF validator")
        else:
            print("❌ BTCUSDm NOT found in MTF validator")
            print("💡 This might cause MTF validation to fail")
            
        # Look for forex pairs list
        if 'forex_pairs' in content:
            print("✅ Found forex pairs list in MTF validator")
        else:
            print("❌ No forex pairs list found")
            
    except Exception as e:
        print(f"❌ Error reading mtf_validator.py: {e}")

def check_momentum_filter():
    """Check if momentum filter might be blocking BTCUSD"""
    print("\n" + "=" * 60)
    print("⚡ CHECKING MOMENTUM FILTER")
    print("=" * 60)
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
            
        if '_validate_momentum_ignition' in content:
            print("✅ Momentum ignition filter found")
        else:
            print("❌ Momentum ignition filter not found")
            
        if 'MOMENTUM_FILTER' in content:
            print("✅ Momentum filter logging found")
        else:
            print("❌ No momentum filter logging found")
            
    except Exception as e:
        print(f"❌ Error checking momentum filter: {e}")

def main():
    """Main diagnostic function"""
    print("🚀 BTCUSD TRADING LOG CHECKER")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now()}")
    
    # Check configuration
    check_btcusd_configuration()
    
    # Check symbol processing
    check_symbol_processing()
    
    # Check common blocking reasons
    check_common_blocking_reasons()
    
    # Check MTF validator
    check_mtf_validator()
    
    # Check momentum filter
    check_momentum_filter()
    
    print("\n" + "=" * 60)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    print("\n💡 NEXT STEPS:")
    print("1. Check your trading logs for BTCUSD processing")
    print("2. Look for messages like '[PROCESS] Processing signal for BTCUSDm'")
    print("3. Check for blocking messages like '[MTF_VALIDATION] BLOCKED' or '[MOMENTUM_FILTER] BLOCKED'")
    print("4. Verify BTCUSDm is in the enabled symbols list")
    print("5. Check if MT5 can fetch data for BTCUSDm")

if __name__ == "__main__":
    main()
