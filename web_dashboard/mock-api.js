const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 8000;

app.use(cors());
app.use(express.json());

// Mock data for your live algorithm
const mockData = {
  metrics: {
    total_trades: 47,
    win_rate: 68.1,
    profit_factor: 1.85,
    net_profit: 1247.50
  },
  
  algo_health: {
    health_score: 87,
    status: 'healthy',
    uptime_hours: 72.5,
    trade_execution_success_rate: 94.2,
    avg_trade_confidence: 0.73,
    risk_exposure_percent: 4.2
  },
  
  system_status: {
    bot_running: true,
    broker_connected: true,
    database_connected: true,
    last_heartbeat: new Date().toISOString(),
    total_signals_processed: 1247,
    active_symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD'],
    errors_last_hour: 0,
    bot_state: 'running',
    last_signal_time: new Date(Date.now() - 30000).toISOString(),
    restart_count: 2
  },
  
  risk_summary: {
    account_balance: 10000.00,
    account_equity: 11247.50,
    current_risk_percent: 4.2,
    max_risk_percent: 8.0,
    open_positions: 3,
    max_positions: 5,
    positions: [
      {
        symbol: 'EURUSD',
        side: 'long',
        lot_size: 0.1,
        entry_price: 1.0856,
        stop_loss: 1.0820,
        take_profit: 1.0920,
        risk_amount: 36.00,
        confidence: 0.78,
        opened_at: new Date(Date.now() - 3600000).toISOString()
      },
      {
        symbol: 'GBPUSD',
        side: 'short',
        lot_size: 0.05,
        entry_price: 1.2645,
        stop_loss: 1.2680,
        take_profit: 1.2580,
        risk_amount: 17.50,
        confidence: 0.72,
        opened_at: new Date(Date.now() - 7200000).toISOString()
      },
      {
        symbol: 'BTCUSD',
        side: 'long',
        lot_size: 0.01,
        entry_price: 43250.00,
        stop_loss: 42800.00,
        take_profit: 44500.00,
        risk_amount: 45.00,
        confidence: 0.85,
        opened_at: new Date(Date.now() - 1800000).toISOString()
      }
    ],
    cooldowns: []
  },
  
  current_trades: [
    {
      id: 1,
      symbol: 'EURUSD',
      type: 'BUY',
      volume: 0.1,
      entry_price: 1.0856,
      current_price: 1.0872,
      stop_loss: 1.0820,
      take_profit: 1.0920,
      pnl: 16.00,
      confidence: 0.78,
      status: 'OPEN',
      entry_time: new Date(Date.now() - 3600000).toISOString()
    },
    {
      id: 2,
      symbol: 'GBPUSD',
      type: 'SELL',
      volume: 0.05,
      entry_price: 1.2645,
      current_price: 1.2628,
      stop_loss: 1.2680,
      take_profit: 1.2580,
      pnl: 8.50,
      confidence: 0.72,
      status: 'OPEN',
      entry_time: new Date(Date.now() - 7200000).toISOString()
    },
    {
      id: 3,
      symbol: 'BTCUSD',
      type: 'BUY',
      volume: 0.01,
      entry_price: 43250.00,
      current_price: 43580.00,
      stop_loss: 42800.00,
      take_profit: 44500.00,
      pnl: 33.00,
      confidence: 0.85,
      status: 'OPEN',
      entry_time: new Date(Date.now() - 1800000).toISOString()
    }
  ],
  
  ai_insights: {
    last_prediction: {
      signal: 'long',
      confidence: 0.78,
      top_features: [
        { name: 'RSI', impact: 0.234 },
        { name: 'Williams %R', impact: 0.198 },
        { name: 'CCI', impact: 0.156 },
        { name: 'ADX', impact: 0.134 },
        { name: 'Momentum', impact: 0.112 }
      ]
    },
    nearest_neighbors: [
      { similarity: 0.9234 },
      { similarity: 0.9156 },
      { similarity: 0.9087 },
      { similarity: 0.9012 },
      { similarity: 0.8945 }
    ],
    signals_queued: 2,
    signals_executed: 47,
    execution_rate: 95.9,
    alerts: [
      {
        type: 'High Volatility',
        message: 'BTCUSD showing increased volatility',
        severity: 'medium',
        timestamp: new Date(Date.now() - 600000).toISOString()
      }
    ]
  },
  
  market_regime: {
    current_regime: 'trending',
    trend_strength: 0.73,
    volatility_level: 0.45,
    direction: 'bullish',
    regime_duration_hours: 4.2,
    regime_probabilities: {
      trending: 0.73,
      ranging: 0.18,
      volatile: 0.09
    }
  },
  
  confidence_distribution: {
    high_confidence_count: 28,
    medium_confidence_count: 15,
    low_confidence_count: 4,
    high_confidence_percentage: 0.596,
    medium_confidence_percentage: 0.319,
    low_confidence_percentage: 0.085
  },
  
  performance_score: {
    signal_execution_success_rate: 94.2,
    break_even_hits: 8,
    stop_loss_hits: 12
  },
  
  latency_metrics: {
    avg_signal_latency_ms: 45,
    max_signal_latency_ms: 127
  },
  
  error_logs: [
    {
      type: 'Connection Timeout',
      message: 'Brief connection timeout to broker',
      details: 'Retry successful after 2 seconds',
      timestamp: new Date(Date.now() - 1800000).toISOString()
    },
    {
      type: 'Spread Warning',
      message: 'Spread exceeded threshold for EURUSD',
      details: 'Spread: 3.2 pips, Max allowed: 3.0 pips',
      timestamp: new Date(Date.now() - 3600000).toISOString()
    }
  ]
};

// API Routes
app.get('/metrics', (req, res) => {
  res.json(mockData.metrics);
});

app.get('/algo-health', (req, res) => {
  res.json(mockData.algo_health);
});

app.get('/system-status', (req, res) => {
  res.json(mockData.system_status);
});

app.get('/risk-summary', (req, res) => {
  res.json(mockData.risk_summary);
});

app.get('/trades', (req, res) => {
  res.json(mockData.current_trades);
});

app.get('/ai-insights', (req, res) => {
  res.json(mockData.ai_insights);
});

app.get('/market-regime', (req, res) => {
  res.json(mockData.market_regime);
});

app.get('/confidence-distribution', (req, res) => {
  res.json(mockData.confidence_distribution);
});

app.get('/performance-score', (req, res) => {
  res.json(mockData.performance_score);
});

app.get('/latency-metrics', (req, res) => {
  res.json(mockData.latency_metrics);
});

app.get('/error-logs', (req, res) => {
  res.json(mockData.error_logs);
});

app.get('/api/dashboard', (req, res) => {
  res.json({
    metrics: mockData.metrics,
    algo_health: mockData.algo_health,
    system_status: mockData.system_status,
    risk_summary: mockData.risk_summary,
    current_trades: mockData.current_trades
  });
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`🚀 Mock API server running on http://localhost:${PORT}`);
  console.log(`📊 Dashboard should be available at http://localhost:3000`);
  console.log(`🔗 API endpoints available at http://localhost:${PORT}/`);
});
