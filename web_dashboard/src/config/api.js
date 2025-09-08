// API Configuration
// Change this to your AWS RDP IP when deploying
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

export const API_ENDPOINTS = {
  // Dashboard data
  DASHBOARD: `${API_BASE_URL}/api/dashboard`,
  
  // Trading data
  TRADES_STATS: `${API_BASE_URL}/api/trades/stats`,
  OPEN_TRADES: `${API_BASE_URL}/api/trades/open`,
  CLOSED_TRADES: `${API_BASE_URL}/api/trades/closed`,
  
  // Health & system
  ALGO_HEALTH: `${API_BASE_URL}/api/health/algo`,
  SYSTEM_STATUS: `${API_BASE_URL}/api/health/system`,
  
  // Risk & portfolio
  RISK_SUMMARY: `${API_BASE_URL}/api/risk/summary`,
  RISK_METRICS: `${API_BASE_URL}/api/risk/metrics`,
  
  // AI insights
  AI_INSIGHTS: `${API_BASE_URL}/api/ai-insights`,
  
  // P&L data
  DAILY_PNL: `${API_BASE_URL}/api/pnl/daily`,
  WEEKLY_PNL: `${API_BASE_URL}/api/pnl/weekly`,
  MONTHLY_PNL: `${API_BASE_URL}/api/pnl/monthly`,
  
  // WebSocket
  WEBSOCKET: API_BASE_URL.replace('http', 'ws') + '/ws'
};

export default API_ENDPOINTS;
