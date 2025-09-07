#!/usr/bin/env python3
"""
Simple API Server Startup Script
Avoids import issues by running from project root
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the FastAPI app
from app_api.main import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting BigAlgo FinTech API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📊 Dashboard API docs: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "app_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )