#!/usr/bin/env python3
"""
Trading Bot Startup Script

Starts the Lorentzian Trading Bot with proper configuration.
"""

import sys
import os
import asyncio
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import main
from utils.logging import setup_logging
from config.settings import settings_manager

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if config files exist
    config_dir = project_root / "config"
    if not config_dir.exists():
        print("❌ Config directory not found. Creating default config...")
        config_dir.mkdir(exist_ok=True)
    
    # Check if symbols.yaml exists
    symbols_file = config_dir / "symbols.yaml"
    if not symbols_file.exists():
        print("❌ symbols.yaml not found. Creating from example...")
        example_file = config_dir / "symbols.yaml.example"
        if example_file.exists():
            import shutil
            shutil.copy(example_file, symbols_file)
        else:
            print("⚠️  symbols.yaml.example not found. Using default config.")
    
    # Check if environment file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found. Using default settings.")
        print("   Copy env.example to .env and configure your settings.")
    
    # Create necessary directories
    data_dir = project_root / "data"
    logs_dir = project_root / "logs"
    exports_dir = project_root / "exports"
    
    for directory in [data_dir, logs_dir, exports_dir]:
        directory.mkdir(exist_ok=True)
    
    print("✅ Requirements check completed")

def main_startup():
    """Main startup function"""
    print("🚀 Lorentzian Trading Bot - Starting Up")
    print("=" * 50)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check requirements
    check_requirements()
    
    # Setup logging
    print("📝 Setting up logging...")
    logger = setup_logging(
        log_level=settings_manager.global_settings.log_level,
        log_file=settings_manager.global_settings.log_file
    )
    
    logger.info("Lorentzian Trading Bot starting up")
    logger.info(f"Timezone: {settings_manager.global_settings.timezone}")
    logger.info(f"Loop interval: {settings_manager.global_settings.loop_interval_minutes} minutes")
    logger.info(f"Max risk: {settings_manager.global_settings.max_account_risk_percent}%")
    
    # Print configuration summary
    print("\n📋 Configuration Summary:")
    print(f"   Timezone: {settings_manager.global_settings.timezone}")
    print(f"   Loop Interval: {settings_manager.global_settings.loop_interval_minutes} minutes")
    print(f"   Cooldown: {settings_manager.global_settings.cooldown_minutes} minutes")
    print(f"   Max Risk: {settings_manager.global_settings.max_account_risk_percent}%")
    print(f"   Max Trades: {settings_manager.global_settings.max_concurrent_trades}")
    print(f"   Neighbors Count: {settings_manager.global_settings.neighbors_count}")
    print(f"   Feature Count: {settings_manager.global_settings.feature_count}")
    
    # Check broker configuration
    if not settings_manager.global_settings.broker_login:
        print("⚠️  Warning: No broker login configured")
        print("   Set BROKER_LOGIN environment variable or update config")
    
    print("\n🎯 Starting trading bot...")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Run the main bot
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trading bot stopped by user")
    except Exception as e:
        print(f"\n💥 Trading bot crashed: {e}")
        logger.error(f"Trading bot crashed: {e}", exc_info=True)
        return 1
    
    print("👋 Trading bot shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main_startup())
