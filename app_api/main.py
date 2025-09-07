"""
FastAPI Main Application

Provides REST API endpoints for the trading bot web dashboard.
Focuses on P&L analytics and algo health monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path

from .models import *
from core.portfolio import PortfolioManager
from core.risk import RiskManager, RiskSettings
from core.sessions import SessionManager
from core.signals import LorentzianClassifier, Settings, FilterSettings
from adapters.mt5_adapter import MT5Adapter
from config.settings import settings_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lorentzian Trading Bot API",
    description="API for Lorentzian Classification Trading Bot with P&L Analytics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (in production, use dependency injection)
portfolio_manager = PortfolioManager()
risk_manager = RiskManager(RiskSettings())
session_manager = SessionManager()
broker_adapter = MT5Adapter()
classifier = LorentzianClassifier(Settings(), FilterSettings())

# Health tracking
bot_start_time = datetime.now()
system_status = {
    "bot_running": True,
    "broker_connected": False,
    "database_connected": True,
    "last_heartbeat": datetime.now(),
    "active_symbols": [],
    "total_signals_processed": 0,
    "errors_last_hour": 0
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Lorentzian Trading Bot API")
    
    # Connect to broker
    if broker_adapter.connect():
        system_status["broker_connected"] = True
        logger.info("Connected to broker")
    else:
        logger.warning("Failed to connect to broker")
    
    # Start background tasks
    asyncio.create_task(heartbeat_task())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Lorentzian Trading Bot API")
    broker_adapter.disconnect()

async def heartbeat_task():
    """Background heartbeat task"""
    while True:
        system_status["last_heartbeat"] = datetime.now()
        await asyncio.sleep(30)  # Update every 30 seconds

# ===================
# ==== Dashboard ====
# ===================

@app.get("/api/dashboard", response_model=DashboardDataModel)
async def get_dashboard_data():
    """Get complete dashboard data"""
    try:
        # Get portfolio data
        portfolio_data = portfolio_manager.get_dashboard_data()
        
        # Get risk summary
        risk_summary = risk_manager.get_risk_summary()
        
        # Get session info
        session_info = session_manager.get_session_info()
        
        # Get algo health
        algo_health = get_algo_health()
        
        # Get system status
        system_status_data = get_system_status()
        
        # Handle risk summary error case
        if "error" in risk_summary:
            risk_summary_model = RiskSummaryModel(
                account_balance=0.0,
                account_equity=0.0,
                current_risk_percent=0.0,
                max_risk_percent=8.0,
                open_positions=0,
                max_positions=5,
                positions=[],
                cooldowns={}
            )
        else:
            risk_summary_model = RiskSummaryModel(**risk_summary)
        
        return DashboardDataModel(
            portfolio_stats=PortfolioStatsModel(**portfolio_data["portfolio_stats"]),
            open_trades=[TradeModel(**trade) for trade in portfolio_data["open_trades"]],
            recent_trades=[TradeModel(**trade) for trade in portfolio_data["recent_trades"]],
            pnl_by_period=portfolio_data["pnl_by_period"],
            confidence_distribution=ConfidenceDistributionModel(**portfolio_data["confidence_distribution"]),
            symbol_performance=[SymbolPerformanceModel(**perf) for perf in portfolio_data["symbol_performance"].values()],
            risk_summary=risk_summary_model,
            session_info=SessionInfoModel(**session_info),
            algo_health=algo_health,
            system_status=system_status_data
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================
# ==== P&L Data ====
# =================

@app.get("/api/pnl/daily", response_model=PnLChartDataModel)
async def get_daily_pnl():
    """Get daily P&L data"""
    try:
        pnl_data = portfolio_manager.get_pnl_by_period('daily')
        
        chart_data = [
            {"date": date, "pnl": pnl}
            for date, pnl in pnl_data.items()
        ]
        
        return PnLChartDataModel(period="daily", data=chart_data)
        
    except Exception as e:
        logger.error(f"Error getting daily P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pnl/weekly", response_model=PnLChartDataModel)
async def get_weekly_pnl():
    """Get weekly P&L data"""
    try:
        pnl_data = portfolio_manager.get_pnl_by_period('weekly')
        
        chart_data = [
            {"week": week, "pnl": pnl}
            for week, pnl in pnl_data.items()
        ]
        
        return PnLChartDataModel(period="weekly", data=chart_data)
        
    except Exception as e:
        logger.error(f"Error getting weekly P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pnl/monthly", response_model=PnLChartDataModel)
async def get_monthly_pnl():
    """Get monthly P&L data"""
    try:
        pnl_data = portfolio_manager.get_pnl_by_period('monthly')
        
        chart_data = [
            {"month": month, "pnl": pnl}
            for month, pnl in pnl_data.items()
        ]
        
        return PnLChartDataModel(period="monthly", data=chart_data)
        
    except Exception as e:
        logger.error(f"Error getting monthly P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pnl/quarterly", response_model=PnLChartDataModel)
async def get_quarterly_pnl():
    """Get quarterly P&L data"""
    try:
        pnl_data = portfolio_manager.get_pnl_by_period('quarterly')
        
        chart_data = [
            {"quarter": quarter, "pnl": pnl}
            for quarter, pnl in pnl_data.items()
        ]
        
        return PnLChartDataModel(period="quarterly", data=chart_data)
        
    except Exception as e:
        logger.error(f"Error getting quarterly P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pnl/yearly", response_model=PnLChartDataModel)
async def get_yearly_pnl():
    """Get yearly P&L data"""
    try:
        pnl_data = portfolio_manager.get_pnl_by_period('yearly')
        
        chart_data = [
            {"year": year, "pnl": pnl}
            for year, pnl in pnl_data.items()
        ]
        
        return PnLChartDataModel(period="yearly", data=chart_data)
        
    except Exception as e:
        logger.error(f"Error getting yearly P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================
# ==== Trade Data ====
# ===================

@app.get("/api/trades/open", response_model=List[TradeModel])
async def get_open_trades():
    """Get all open trades"""
    try:
        open_trades = portfolio_manager.get_open_trades()
        return [TradeModel(**trade.to_dict()) for trade in open_trades]
        
    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades/closed", response_model=List[TradeModel])
async def get_closed_trades(limit: Optional[int] = 50):
    """Get closed trades"""
    try:
        closed_trades = portfolio_manager.get_closed_trades(limit)
        return [TradeModel(**trade.to_dict()) for trade in closed_trades]
        
    except Exception as e:
        logger.error(f"Error getting closed trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades/symbol/{symbol}", response_model=List[TradeModel])
async def get_trades_by_symbol(symbol: str):
    """Get trades for specific symbol"""
    try:
        trades = portfolio_manager.get_trades_by_symbol(symbol)
        return [TradeModel(**trade.to_dict()) for trade in trades]
        
    except Exception as e:
        logger.error(f"Error getting trades for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades/stats", response_model=TradeStatsModel)
async def get_trade_stats():
    """Get trade statistics"""
    try:
        stats = portfolio_manager.get_portfolio_stats()
        return TradeStatsModel(
            total_trades=stats.total_trades,
            win_rate=stats.win_rate,
            total_pnl=stats.total_pnl,
            avg_win=stats.avg_win,
            avg_loss=stats.avg_loss,
            largest_win=stats.largest_win,
            largest_loss=stats.largest_loss,
            profit_factor=stats.profit_factor,
            max_drawdown=stats.max_drawdown
        )
        
    except Exception as e:
        logger.error(f"Error getting trade stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================
# ==== Risk Data ====
# =================

@app.get("/api/risk/summary", response_model=RiskSummaryModel)
async def get_risk_summary():
    """Get risk summary"""
    try:
        risk_data = risk_manager.get_risk_summary()
        
        # Handle case when no account info is available
        if "error" in risk_data:
            return RiskSummaryModel(
                account_balance=0.0,
                account_equity=0.0,
                current_risk_percent=0.0,
                max_risk_percent=8.0,
                open_positions=0,
                max_positions=5,
                positions=[],
                cooldowns={}
            )
        
        return RiskSummaryModel(**risk_data)
        
    except Exception as e:
        logger.error(f"Error getting risk summary: {e}")
        # Return default values instead of raising exception
        return RiskSummaryModel(
            account_balance=0.0,
            account_equity=0.0,
            current_risk_percent=0.0,
            max_risk_percent=8.0,
            open_positions=0,
            max_positions=5,
            positions=[],
            cooldowns={}
        )

@app.get("/api/risk/metrics", response_model=RiskMetricsModel)
async def get_risk_metrics():
    """Get risk metrics for dashboard"""
    try:
        risk_data = risk_manager.get_risk_summary()
        
        # Handle case when no account info is available
        if "error" in risk_data:
            return RiskMetricsModel(
                current_risk_percent=0.0,
                max_risk_percent=8.0,
                open_positions=0,
                max_positions=5,
                risk_thermometer=0,
                cooldown_symbols=[]
            )
        
        # Calculate risk thermometer (0-100)
        risk_percent = risk_data["current_risk_percent"]
        max_risk_percent = risk_data["max_risk_percent"]
        risk_thermometer = int((risk_percent / max_risk_percent) * 100)
        
        # Get cooldown symbols
        cooldowns = risk_data.get("cooldowns", {})
        cooldown_symbols = list(cooldowns.keys())
        
        return RiskMetricsModel(
            current_risk_percent=risk_percent,
            max_risk_percent=max_risk_percent,
            open_positions=risk_data["open_positions"],
            max_positions=risk_data["max_positions"],
            risk_thermometer=risk_thermometer,
            cooldown_symbols=cooldown_symbols
        )
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        # Return default values instead of raising exception
        return RiskMetricsModel(
            current_risk_percent=0.0,
            max_risk_percent=8.0,
            open_positions=0,
            max_positions=5,
            risk_thermometer=0,
            cooldown_symbols=[]
        )

# ===================
# ==== Session Data ====
# ===================

@app.get("/api/session/info", response_model=SessionInfoModel)
async def get_session_info():
    """Get session information"""
    try:
        session_data = session_manager.get_session_info()
        return SessionInfoModel(**session_data)
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/symbol/{symbol}", response_model=SymbolSessionStatusModel)
async def get_symbol_session_status(symbol: str):
    """Get session status for specific symbol"""
    try:
        status_data = session_manager.get_symbol_session_status(symbol)
        return SymbolSessionStatusModel(**status_data)
        
    except Exception as e:
        logger.error(f"Error getting session status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================
# ==== Health Data ====
# ===================

@app.get("/api/health/algo", response_model=AlgoHealthModel)
async def get_algo_health():
    """Get algorithm health metrics"""
    try:
        return get_algo_health()
        
    except Exception as e:
        logger.error(f"Error getting algo health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health/system", response_model=SystemStatusModel)
async def get_system_status():
    """Get system status"""
    try:
        return get_system_status()
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================
# ==== Configuration ====
# ===================

@app.get("/api/config/settings", response_model=SettingsModel)
async def get_settings():
    """Get current settings"""
    try:
        settings_data = settings_manager.get_all_settings()
        return SettingsModel(**settings_data)
        
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/symbols", response_model=Dict[str, SymbolConfigModel])
async def get_symbol_configs():
    """Get symbol configurations"""
    try:
        symbol_configs = settings_manager.symbol_configs
        return {
            symbol: SymbolConfigModel(**config)
            for symbol, config in symbol_configs.items()
        }
        
    except Exception as e:
        logger.error(f"Error getting symbol configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================
# ==== Export ====
# ===================

@app.post("/api/export/trades", response_model=ExportResponseModel)
async def export_trades(request: ExportRequestModel, background_tasks: BackgroundTasks):
    """Export trades to file"""
    try:
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trades_export_{timestamp}.{request.format}"
        filepath = f"exports/{filename}"
        
        # Create exports directory
        Path("exports").mkdir(exist_ok=True)
        
        # Export data
        if request.format == "csv":
            portfolio_manager.export_trades_csv(filepath)
            record_count = len(portfolio_manager.get_closed_trades())
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        return ExportResponseModel(
            success=True,
            file_path=filepath,
            record_count=record_count
        )
        
    except Exception as e:
        logger.error(f"Error exporting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/download/{filename}")
async def download_export(filename: str):
    """Download exported file"""
    try:
        filepath = f"exports/{filename}"
        if not Path(filepath).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(filepath, filename=filename)
        
    except Exception as e:
        logger.error(f"Error downloading export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================
# ==== Helper Functions ====
# ===================

def get_algo_health() -> AlgoHealthModel:
    """Calculate algorithm health metrics"""
    try:
        # Calculate uptime
        uptime_hours = (datetime.now() - bot_start_time).total_seconds() / 3600
        
        # Get trade stats
        trade_stats = risk_manager.get_trade_stats()
        
        # Calculate success rate (simplified)
        total_trades = trade_stats["total_trades"]
        success_rate = 95.0  # Placeholder - would calculate from actual data
        
        # Calculate risk exposure
        risk_data = risk_manager.get_risk_summary()
        risk_exposure = risk_data.get("current_risk_percent", 0.0)
        
        # Calculate average confidence
        portfolio_data = portfolio_manager.get_dashboard_data()
        confidence_dist = portfolio_data["confidence_distribution"]
        total_confidence_trades = confidence_dist["low"] + confidence_dist["medium"] + confidence_dist["high"]
        
        if total_confidence_trades > 0:
            avg_confidence = (
                confidence_dist["low"] * 0.2 +
                confidence_dist["medium"] * 0.5 +
                confidence_dist["high"] * 0.8
            ) / total_confidence_trades
        else:
            avg_confidence = 0.0
        
        # Calculate health score (0-100)
        health_score = 100
        if risk_exposure > 8.0:  # Over 80% of max risk
            health_score -= 20
        if success_rate < 80.0:
            health_score -= 30
        if not system_status["broker_connected"]:
            health_score -= 40
        if system_status["errors_last_hour"] > 5:
            health_score -= 20
        
        # Determine status
        if health_score >= 80:
            status = HealthStatus.HEALTHY
        elif health_score >= 60:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        
        return AlgoHealthModel(
            uptime_hours=uptime_hours,
            trade_execution_success_rate=success_rate,
            broker_errors=system_status["errors_last_hour"],
            risk_exposure_percent=risk_exposure,
            avg_trade_confidence=avg_confidence,
            health_score=health_score,
            status=status,
            last_update=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error calculating algo health: {e}")
        return AlgoHealthModel(
            uptime_hours=0.0,
            trade_execution_success_rate=0.0,
            broker_errors=0,
            risk_exposure_percent=0.0,
            avg_trade_confidence=0.0,
            health_score=0,
            status=HealthStatus.CRITICAL,
            last_update=datetime.now()
        )

def get_system_status() -> SystemStatusModel:
    """Get system status"""
    return SystemStatusModel(
        bot_running=system_status["bot_running"],
        broker_connected=system_status["broker_connected"],
        database_connected=system_status["database_connected"],
        last_heartbeat=system_status["last_heartbeat"],
        active_symbols=system_status["active_symbols"],
        total_signals_processed=system_status["total_signals_processed"],
        errors_last_hour=system_status["errors_last_hour"]
    )

# ===================
# ==== Root Endpoint ====
# ===================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Lorentzian Trading Bot API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_hours": (datetime.now() - bot_start_time).total_seconds() / 3600
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
