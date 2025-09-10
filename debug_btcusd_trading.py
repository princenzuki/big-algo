#!/usr/bin/env python3
"""
Debug BTCUSD Trading Issues

This script will diagnose why BTCUSD isn't taking trades by checking:
1. Symbol configuration and mapping
2. Data availability
3. Signal generation
4. Filter conditions
5. MT5 connection status
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings_manager
from adapters.mt5_adapter import MT5Adapter
from core.signals import LorentzianClassifier, Settings, FilterSettings
from core.risk import RiskManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_symbol_configuration():
    """Check BTCUSD symbol configuration"""
    print("=" * 60)
    print("🔍 CHECKING BTCUSD SYMBOL CONFIGURATION")
    print("=" * 60)
    
    # Check symbols.yaml
    symbols_config = settings_manager.symbol_configs
    print(f"📋 Total symbols in config: {len(symbols_config)}")
    
    # Look for BTCUSD variants
    btc_variants = [symbol for symbol in symbols_config.keys() if 'BTC' in symbol.upper()]
    print(f"🔍 BTC variants found: {btc_variants}")
    
    # Check BTCUSDm specifically
    if 'BTCUSDm' in symbols_config:
        btc_config = symbols_config['BTCUSDm']
        print(f"✅ BTCUSDm configuration found:")
        for key, value in btc_config.items():
            print(f"   {key}: {value}")
    else:
        print("❌ BTCUSDm not found in symbols.yaml")
    
    # Check if BTCUSD (without 'm') exists
    if 'BTCUSD' in symbols_config:
        btc_config = symbols_config['BTCUSD']
        print(f"✅ BTCUSD configuration found:")
        for key, value in btc_config.items():
            print(f"   {key}: {value}")
    else:
        print("❌ BTCUSD not found in symbols.yaml")
    
    return btc_variants

def check_mt5_connection():
    """Check MT5 connection and symbol availability"""
    print("\n" + "=" * 60)
    print("🔌 CHECKING MT5 CONNECTION & SYMBOL AVAILABILITY")
    print("=" * 60)
    
    # Check MT5 installation
    print("🔍 Checking MT5 installation...")
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 module imported successfully")
    except ImportError as e:
        print(f"❌ MetaTrader5 module not found: {e}")
        return False
    
    # Try to initialize MT5
    print("🔄 Attempting MT5 initialization...")
    if not mt5.initialize():
        error_code = mt5.last_error()
        print(f"❌ MT5 initialization failed")
        print(f"   Error code: {error_code}")
        print(f"   Error description: {mt5.last_error()}")
        
        # Check common issues
        print("\n🔍 Common MT5 issues:")
        print("   1. MT5 terminal not running")
        print("   2. Wrong terminal path")
        print("   3. Terminal not logged in")
        print("   4. Wrong account credentials")
        print("   5. Terminal version incompatible")
        
        return False
    
    print("✅ MT5 initialized successfully")
    
    # Check account info
    account_info = mt5.account_info()
    if account_info:
        print(f"📊 Account: {account_info.login}, Balance: ${account_info.balance:.2f}")
    else:
        print("❌ Could not get account info")
    
    # Check BTCUSD symbol variants
    btc_symbols = ['BTCUSD', 'BTCUSDm', 'BTCUSD#', 'BTCUSD.']
    
    for symbol in btc_symbols:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            print(f"✅ {symbol} found in MT5:")
            print(f"   Bid: {symbol_info.bid}")
            print(f"   Ask: {symbol_info.ask}")
            print(f"   Spread: {symbol_info.spread}")
            print(f"   Trade mode: {symbol_info.trade_mode}")
            print(f"   Visible: {symbol_info.visible}")
        else:
            print(f"❌ {symbol} not found in MT5")
    
    return True

def check_data_availability():
    """Check if historical data is available for BTCUSD"""
    print("\n" + "=" * 60)
    print("📊 CHECKING HISTORICAL DATA AVAILABILITY")
    print("=" * 60)
    
    adapter = MT5Adapter()
    
    # Test different symbol names
    test_symbols = ['BTCUSD', 'BTCUSDm']
    
    for symbol in test_symbols:
        try:
            print(f"\n🔍 Testing {symbol}:")
            
            # Get recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            data = adapter.get_historical_data(symbol, 'M1', start_time, end_time, 1000)
            
            if data and len(data) > 0:
                print(f"   ✅ Data available: {len(data)} bars")
                print(f"   📅 Latest bar: {data[-1]['time']}")
                print(f"   💰 Latest close: {data[-1]['close']}")
            else:
                print(f"   ❌ No data available for {symbol}")
                
        except Exception as e:
            print(f"   ❌ Error fetching data for {symbol}: {e}")

def test_signal_generation():
    """Test signal generation for BTCUSD"""
    print("\n" + "=" * 60)
    print("🧠 TESTING SIGNAL GENERATION")
    print("=" * 60)
    
    adapter = MT5Adapter()
    
    # Test with BTCUSDm (the configured symbol)
    symbol = 'BTCUSDm'
    
    try:
        # Get historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        historical_data = adapter.get_historical_data(symbol, 'M1', start_time, end_time, 1000)
        
        if not historical_data or len(historical_data) < 100:
            print(f"❌ Insufficient data for {symbol}: {len(historical_data) if historical_data else 0} bars")
            return
        
        print(f"✅ Got {len(historical_data)} bars of data for {symbol}")
        
        # Initialize ML classifier
        settings = Settings()
        filter_settings = FilterSettings()
        classifier = LorentzianClassifier(settings, filter_settings)
        
        # Process historical data to build training set
        print("🔄 Building training set...")
        for i, bar in enumerate(historical_data[:-1]):  # Exclude last bar for prediction
            ohlc_data = {
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'time': bar['time']
            }
            classifier.add_training_data(ohlc_data)
        
        print(f"📚 Training set size: {len(classifier.ml_model.training_labels)}")
        
        # Generate signal for the last bar
        last_bar = historical_data[-1]
        ohlc_data = {
            'open': last_bar['open'],
            'high': last_bar['high'],
            'low': last_bar['low'],
            'close': last_bar['close'],
            'time': last_bar['time']
        }
        
        signal_data = classifier.generate_signal(ohlc_data, historical_data)
        
        print(f"🎯 Signal generation result:")
        print(f"   Signal: {signal_data['signal']}")
        print(f"   Confidence: {signal_data['confidence']:.3f}")
        print(f"   Prediction: {signal_data['prediction']:.3f}")
        print(f"   Neighbors: {signal_data['neighbors_count']}")
        print(f"   Filter applied: {signal_data['filter_applied']}")
        
        # Check if signal meets minimum confidence
        symbol_config = settings_manager.get_symbol_config(symbol)
        min_confidence = symbol_config.get('min_confidence', 0.3)
        
        if signal_data['confidence'] >= min_confidence:
            print(f"✅ Signal meets minimum confidence ({min_confidence})")
        else:
            print(f"❌ Signal below minimum confidence ({min_confidence})")
            
    except Exception as e:
        print(f"❌ Error in signal generation: {e}")
        import traceback
        traceback.print_exc()

def check_trading_conditions():
    """Check various trading conditions that might block BTCUSD"""
    print("\n" + "=" * 60)
    print("🚦 CHECKING TRADING CONDITIONS")
    print("=" * 60)
    
    symbol = 'BTCUSDm'
    from config.settings import settings_manager
    symbol_config = settings_manager.get_symbol_config(symbol)
    
    if not symbol_config:
        print(f"❌ No configuration found for {symbol}")
        return
    
    print(f"📋 Configuration for {symbol}:")
    for key, value in symbol_config.items():
        print(f"   {key}: {value}")
    
    # Check if symbol is enabled
    if not symbol_config.get('enabled', True):
        print("❌ Symbol is disabled")
    else:
        print("✅ Symbol is enabled")
    
    # Check weekend trading
    is_weekend = datetime.now().weekday() >= 5
    allow_weekend = symbol_config.get('allow_weekend', False)
    
    if is_weekend and not allow_weekend:
        print("❌ Weekend trading not allowed")
    else:
        print("✅ Weekend trading allowed")
    
    # Check risk manager cooldown
    try:
        from config.settings import settings_manager
        risk_manager = RiskManager(settings_manager.global_settings)
        if risk_manager.is_in_global_cooldown():
            print("❌ Global cooldown is active")
        else:
            print("✅ No global cooldown")
    except Exception as e:
        print(f"⚠️ Could not check cooldown: {e}")

def main():
    """Main diagnostic function"""
    print("🚀 BTCUSD TRADING DIAGNOSTIC")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now()}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check symbol configuration
    btc_variants = check_symbol_configuration()
    
    # Check MT5 connection
    mt5_ok = check_mt5_connection()
    
    if mt5_ok:
        # Check data availability
        check_data_availability()
        
        # Test signal generation
        test_signal_generation()
    
    # Check trading conditions
    check_trading_conditions()
    
    print("\n" + "=" * 60)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    # Final recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not btc_variants:
        print("   - Add BTCUSD configuration to symbols.yaml")
    if not mt5_ok:
        print("   - Fix MT5 connection issues")
    print("   - Check if BTCUSDm is the correct symbol name in your broker")
    print("   - Verify minimum confidence threshold (currently 0.4 for BTCUSDm)")
    print("   - Check if spread is within limits (max 50 pips for BTCUSDm)")

if __name__ == "__main__":
    main()
