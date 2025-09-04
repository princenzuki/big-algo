"""
Pydantic models for API responses

Defines data transfer objects for the web dashboard.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TradeSide(str, Enum):
    """Trade side enumeration"""
    LONG = "long"
    SHORT = "short"

class TradeStatus(str, Enum):
    """Trade status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"

class SessionType(str, Enum):
    """Trading session enumeration"""
    LONDON = "london"
    NEW_YORK = "new_york"
    ASIA = "asia"
    CLOSED = "closed"

class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"

# Portfolio Models
class TradeModel(BaseModel):
    """Trade model for API responses"""
    id: str
    symbol: str
    side: TradeSide
    entry_price: float
    exit_price: Optional[float] = None
    lot_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_amount: float
    pnl: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    status: TradeStatus

class PortfolioStatsModel(BaseModel):
    """Portfolio statistics model"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    profit_factor: float
    avg_trade_duration: float
    total_risk_taken: float

class PnLPeriodModel(BaseModel):
    """P&L by period model"""
    period: str
    pnl: float

class ConfidenceDistributionModel(BaseModel):
    """Confidence distribution model"""
    low: int
    medium: int
    high: int

class SymbolPerformanceModel(BaseModel):
    """Symbol performance model"""
    symbol: str
    total_trades: int
    winning_trades: int
    total_pnl: float
    win_rate: float
    avg_confidence: float
    total_risk: float

# Risk Models
class RiskSummaryModel(BaseModel):
    """Risk summary model"""
    account_balance: float
    account_equity: float
    current_risk_percent: float
    max_risk_percent: float
    open_positions: int
    max_positions: int
    positions: List[Dict[str, Any]]
    cooldowns: Dict[str, str]

class PositionModel(BaseModel):
    """Position model"""
    symbol: str
    side: TradeSide
    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_amount: float
    opened_at: datetime

# Session Models
class SessionInfoModel(BaseModel):
    """Session information model"""
    current_time: str
    timezone: str
    current_session: SessionType
    weekday: str
    is_weekend: bool
    session_times: Dict[str, str]
    weekend_block: Dict[str, Any]

class SymbolSessionStatusModel(BaseModel):
    """Symbol session status model"""
    symbol: str
    can_trade: bool
    reason: str
    current_session: SessionType
    allow_weekend: bool
    allowed_sessions: Dict[str, bool]

# Health Models
class AlgoHealthModel(BaseModel):
    """Algorithm health model"""
    uptime_hours: float
    trade_execution_success_rate: float
    broker_errors: int
    risk_exposure_percent: float
    avg_trade_confidence: float
    health_score: int
    status: HealthStatus
    last_update: datetime

class SystemStatusModel(BaseModel):
    """System status model"""
    bot_running: bool
    broker_connected: bool
    database_connected: bool
    last_heartbeat: datetime
    active_symbols: List[str]
    total_signals_processed: int
    errors_last_hour: int

# Dashboard Models
class DashboardDataModel(BaseModel):
    """Complete dashboard data model"""
    portfolio_stats: PortfolioStatsModel
    open_trades: List[TradeModel]
    recent_trades: List[TradeModel]
    pnl_by_period: Dict[str, Dict[str, float]]
    confidence_distribution: ConfidenceDistributionModel
    symbol_performance: List[SymbolPerformanceModel]
    risk_summary: RiskSummaryModel
    session_info: SessionInfoModel
    algo_health: AlgoHealthModel
    system_status: SystemStatusModel

# API Response Models
class ApiResponseModel(BaseModel):
    """Standard API response model"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class PnLChartDataModel(BaseModel):
    """P&L chart data model"""
    period: str
    data: List[Dict[str, Any]]  # [{date: str, pnl: float}, ...]

class TradeStatsModel(BaseModel):
    """Trade statistics model"""
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    max_drawdown: float

class RiskMetricsModel(BaseModel):
    """Risk metrics model"""
    current_risk_percent: float
    max_risk_percent: float
    open_positions: int
    max_positions: int
    risk_thermometer: int  # 0-100 scale
    cooldown_symbols: List[str]

class SignalModel(BaseModel):
    """Signal model"""
    symbol: str
    signal: int  # -1, 0, 1
    confidence: float
    prediction: float
    timestamp: datetime
    features: Dict[str, float]
    filters_applied: bool

class LogEntryModel(BaseModel):
    """Log entry model"""
    timestamp: datetime
    level: str
    message: str
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    error_code: Optional[str] = None

# Configuration Models
class SettingsModel(BaseModel):
    """Settings model"""
    global_settings: Dict[str, Any]
    symbol_configs: Dict[str, Any]

class SymbolConfigModel(BaseModel):
    """Symbol configuration model"""
    enabled: bool
    allow_weekend: bool
    min_confidence: float
    max_spread_pips: float
    atr_period: int
    sl_multiplier: float
    tp_multiplier: float
    sessions: Dict[str, bool]

# Export Models
class ExportRequestModel(BaseModel):
    """Export request model"""
    format: str  # 'csv', 'json'
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: Optional[List[str]] = None
    include_open_trades: bool = True
    include_closed_trades: bool = True

class ExportResponseModel(BaseModel):
    """Export response model"""
    success: bool
    file_path: Optional[str] = None
    record_count: int
    error: Optional[str] = None
