#!/usr/bin/env python3
"""
System Test Script - Tests all core components
"""

def test_configuration():
    """Test configuration loading"""
    print("🧪 Testing Configuration...")
    try:
        from config.settings import settings_manager
        settings = settings_manager.get_all_settings()
        print("✅ Configuration loaded successfully")
        print(f"   Neighbors: {settings.neighbors_count}")
        print(f"   Features: {settings.feature_count}")
        print(f"   Max bars back: {settings.max_bars_back}")
        print(f"   Use dynamic exits: {settings.use_dynamic_exits}")
        print(f"   Use volatility filter: {settings.use_volatility_filter}")
        print(f"   Regime threshold: {settings.regime_threshold}")
        print(f"   Feature 1: {settings.feature_params['f1']}")
        print(f"   Feature 2: {settings.feature_params['f2']}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_core_components():
    """Test core components"""
    print("\n🧪 Testing Core Components...")
    
    # Test risk management
    try:
        from core.risk import RiskManager
        rm = RiskManager()
        print("✅ Risk Manager loaded")
        print(f"   Max risk: {rm.max_risk_percent}%")
        print(f"   Max trades: {rm.max_concurrent_trades}")
    except Exception as e:
        print(f"❌ Risk Manager test failed: {e}")
        return False
    
    # Test session management
    try:
        from core.sessions import SessionManager
        sm = SessionManager()
        print("✅ Session Manager loaded")
        print(f"   Timezone: {sm.timezone}")
        current_session = sm.get_current_session()
        print(f"   Current session: {current_session}")
    except Exception as e:
        print(f"❌ Session Manager test failed: {e}")
        return False
    
    # Test portfolio management
    try:
        from core.portfolio import PortfolioManager
        pm = PortfolioManager()
        print("✅ Portfolio Manager loaded")
        open_trades = pm.get_open_trades()
        print(f"   Open trades: {len(open_trades)}")
    except Exception as e:
        print(f"❌ Portfolio Manager test failed: {e}")
        return False
    
    return True

def test_utilities():
    """Test utility functions"""
    print("\n🧪 Testing Utilities...")
    
    # Test time utilities
    try:
        from utils.time_utils import get_kenya_time, is_weekend_blocked
        kenya_time = get_kenya_time()
        print("✅ Time utilities loaded")
        print(f"   Kenya time: {kenya_time}")
        print(f"   Weekend blocked: {is_weekend_blocked()}")
    except Exception as e:
        print(f"❌ Time utilities test failed: {e}")
        return False
    
    # Test logging
    try:
        from utils.logging import setup_logging
        logger = setup_logging()
        print("✅ Logging system loaded")
        logger.info("Test log message")
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False
    
    return True

def test_api_components():
    """Test API components"""
    print("\n🧪 Testing API Components...")
    
    try:
        from app_api.models import TradeResponse, RiskStatus
        print("✅ API models loaded")
    except Exception as e:
        print(f"❌ API models test failed: {e}")
        return False
    
    try:
        from app_api.main import app
        print("✅ FastAPI app loaded")
    except Exception as e:
        print(f"❌ FastAPI app test failed: {e}")
        return False
    
    return True

def test_main_bot():
    """Test main bot initialization"""
    print("\n🧪 Testing Main Bot...")
    
    try:
        from main import TradingBot
        bot = TradingBot()
        print("✅ Trading Bot initialized")
        print("   Bot ready for testing")
    except Exception as e:
        print(f"❌ Trading Bot test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Lorentzian Trading Bot - System Test")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_core_components,
        test_utilities,
        test_api_components,
        test_main_bot
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for trading.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()
