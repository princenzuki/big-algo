#!/usr/bin/env python3
"""
Debug BTCUSD Specific Issues

This script focuses on BTCUSD-specific problems since other symbols are working.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_btcusd_symbol_variants():
    """Check all possible BTCUSD symbol variants in MT5"""
    print("=" * 60)
    print("🔍 CHECKING BTCUSD SYMBOL VARIANTS IN MT5")
    print("=" * 60)
    
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return []
    
    print("✅ MT5 initialized successfully")
    
    # Get all symbols from MT5
    symbols = mt5.symbols_get()
    if not symbols:
        print("❌ Could not get symbols from MT5")
        return []
    
    print(f"📊 Total symbols available in MT5: {len(symbols)}")
    
    # Find all BTC-related symbols
    btc_symbols = []
    for symbol in symbols:
        symbol_name = symbol.name
        if 'BTC' in symbol_name.upper() or 'BITCOIN' in symbol_name.upper():
            btc_symbols.append(symbol)
            print(f"🔍 Found BTC symbol: {symbol_name}")
            print(f"   Bid: {symbol.bid}")
            print(f"   Ask: {symbol.ask}")
            print(f"   Spread: {symbol.spread}")
            print(f"   Trade mode: {symbol.trade_mode}")
            print(f"   Visible: {symbol.visible}")
            print(f"   Trade stops level: {symbol.trade_stops_level}")
            print(f"   Point: {symbol.point}")
            print(f"   Digits: {symbol.digits}")
            print()
    
    return btc_symbols

def test_btcusd_data_fetching():
    """Test data fetching for different BTCUSD variants"""
    print("=" * 60)
    print("📊 TESTING BTCUSD DATA FETCHING")
    print("=" * 60)
    
    # Test different symbol names
    test_symbols = [
        'BTCUSD',
        'BTCUSDm', 
        'BTCUSD#',
        'BTCUSD.',
        'BTCUSD_',
        'BTCUSD-',
        'BTCUSD1',
        'BTCUSD2'
    ]
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}:")
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"   ❌ Symbol {symbol} not found")
            continue
        
        print(f"   ✅ Symbol {symbol} found")
        print(f"   📊 Bid: {symbol_info.bid}, Ask: {symbol_info.ask}")
        print(f"   📈 Spread: {symbol_info.spread} points")
        print(f"   🔄 Trade mode: {symbol_info.trade_mode}")
        
        # Try to get historical data
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_time, end_time)
            
            if rates is not None and len(rates) > 0:
                print(f"   ✅ Historical data: {len(rates)} bars")
                print(f"   📅 Latest: {rates[-1]['time']} = {rates[-1]['close']}")
            else:
                print(f"   ❌ No historical data available")
                
        except Exception as e:
            print(f"   ❌ Error fetching data: {e}")

def check_btcusd_trading_conditions():
    """Check BTCUSD-specific trading conditions"""
    print("=" * 60)
    print("🚦 CHECKING BTCUSD TRADING CONDITIONS")
    print("=" * 60)
    
    # Check symbol configuration
    symbol_config = settings_manager.get_symbol_config('BTCUSDm')
    
    if not symbol_config:
        print("❌ No configuration found for BTCUSDm")
        return
    
    print("📋 BTCUSDm Configuration:")
    for key, value in symbol_config.items():
        print(f"   {key}: {value}")
    
    # Check if symbol is enabled
    if not symbol_config.get('enabled', True):
        print("❌ BTCUSDm is disabled in configuration")
    else:
        print("✅ BTCUSDm is enabled")
    
    # Check minimum confidence
    min_confidence = symbol_config.get('min_confidence', 0.4)
    print(f"📊 Minimum confidence required: {min_confidence}")
    
    # Check spread limits
    max_spread = symbol_config.get('max_spread_pips', 50.0)
    print(f"📊 Maximum spread allowed: {max_spread} pips")
    
    # Check if we can get current spread
    try:
        symbol_info = mt5.symbol_info('BTCUSDm')
        if symbol_info:
            current_spread = symbol_info.spread * symbol_info.point
            spread_pips = current_spread / symbol_info.point
            print(f"📊 Current spread: {spread_pips:.1f} pips")
            
            if spread_pips > max_spread:
                print(f"❌ Spread too high: {spread_pips:.1f} > {max_spread}")
            else:
                print(f"✅ Spread within limits: {spread_pips:.1f} <= {max_spread}")
        else:
            print("❌ Could not get current spread for BTCUSDm")
    except Exception as e:
        print(f"❌ Error checking spread: {e}")

def check_btcusd_signal_generation():
    """Test signal generation specifically for BTCUSD"""
    print("=" * 60)
    print("🧠 TESTING BTCUSD SIGNAL GENERATION")
    print("=" * 60)
    
    # First, find the correct BTCUSD symbol
    symbols = mt5.symbols_get()
    btc_symbol = None
    
    for symbol in symbols:
        if 'BTC' in symbol.name.upper():
            btc_symbol = symbol.name
            print(f"✅ Using BTC symbol: {btc_symbol}")
            break
    
    if not btc_symbol:
        print("❌ No BTC symbol found in MT5")
        return
    
    # Get historical data
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        rates = mt5.copy_rates_range(btc_symbol, mt5.TIMEFRAME_M1, start_time, end_time)
        
        if rates is None or len(rates) < 100:
            print(f"❌ Insufficient data: {len(rates) if rates is not None else 0} bars")
            return
        
        print(f"✅ Got {len(rates)} bars of data")
        
        # Convert to our format
        historical_data = []
        for rate in rates:
            historical_data.append({
                'time': datetime.fromtimestamp(rate['time']),
                'open': rate['open'],
                'high': rate['high'],
                'low': rate['low'],
                'close': rate['close']
            })
        
        # Test signal generation
        from core.signals import LorentzianClassifier, Settings, FilterSettings
        
        settings = Settings()
        filter_settings = FilterSettings()
        classifier = LorentzianClassifier(settings, filter_settings)
        
        # Build training set
        print("🔄 Building training set...")
        for bar in historical_data[:-1]:  # Exclude last bar
            ohlc_data = {
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'time': bar['time']
            }
            classifier.add_training_data(ohlc_data)
        
        print(f"📚 Training set size: {len(classifier.ml_model.training_labels)}")
        
        # Generate signal
        last_bar = historical_data[-1]
        ohlc_data = {
            'open': last_bar['open'],
            'high': last_bar['high'],
            'low': last_bar['low'],
            'close': last_bar['close'],
            'time': last_bar['time']
        }
        
        signal_data = classifier.generate_signal(ohlc_data, historical_data)
        
        print(f"🎯 Signal result:")
        print(f"   Signal: {signal_data['signal']}")
        print(f"   Confidence: {signal_data['confidence']:.3f}")
        print(f"   Prediction: {signal_data['prediction']:.3f}")
        print(f"   Neighbors: {signal_data['neighbors_count']}")
        print(f"   Filter applied: {signal_data['filter_applied']}")
        
        # Check against minimum confidence
        min_confidence = 0.4  # BTCUSDm minimum
        if signal_data['confidence'] >= min_confidence:
            print(f"✅ Signal meets minimum confidence ({min_confidence})")
        else:
            print(f"❌ Signal below minimum confidence ({min_confidence})")
            
    except Exception as e:
        print(f"❌ Error in signal generation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main diagnostic function"""
    print("🚀 BTCUSD SPECIFIC DIAGNOSTIC")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now()}")
    
    # Check MT5 connection first
    if not mt5.initialize():
        print("❌ MT5 initialization failed - cannot proceed")
        return
    
    print("✅ MT5 connected successfully")
    
    # Check BTCUSD symbol variants
    btc_symbols = check_btcusd_symbol_variants()
    
    # Test data fetching
    test_btcusd_data_fetching()
    
    # Check trading conditions
    check_btcusd_trading_conditions()
    
    # Test signal generation
    check_btcusd_signal_generation()
    
    print("\n" + "=" * 60)
    print("🏁 BTCUSD DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    if not btc_symbols:
        print("\n💡 RECOMMENDATIONS:")
        print("   - BTCUSD symbol not found in MT5")
        print("   - Check if your broker offers BTCUSD trading")
        print("   - Verify the correct symbol name with your broker")
        print("   - Check if BTCUSD trading is enabled in your account")
    else:
        print(f"\n✅ Found {len(btc_symbols)} BTC symbols in MT5")
        print("   - Check if the bot is using the correct symbol name")
        print("   - Verify symbol configuration matches MT5 symbol")

if __name__ == "__main__":
    main()
