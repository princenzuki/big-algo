#!/usr/bin/env python3
"""
API Server Startup Script

Starts the FastAPI server for the web dashboard.
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings_manager

def main():
    """Start the API server"""
    print("🌐 Lorentzian Trading Bot - API Server")
    print("=" * 50)
    
    # Print configuration
    print("📋 API Configuration:")
    print(f"   Host: 0.0.0.0")
    print(f"   Port: 8000")
    print(f"   Database: {settings_manager.global_settings.database_path}")
    print(f"   Log Level: {settings_manager.global_settings.log_level}")
    
    print("\n🚀 Starting API server...")
    print("   Dashboard will be available at: http://localhost:3000")
    print("   API docs will be available at: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "app_api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level=settings_manager.global_settings.log_level.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 API server stopped by user")
    except Exception as e:
        print(f"\n💥 API server crashed: {e}")
        return 1
    
    print("👋 API server shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
