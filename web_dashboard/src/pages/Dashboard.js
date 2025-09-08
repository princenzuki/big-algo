import React, { useEffect, useState } from 'react';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  TrophyIcon,
  HeartIcon,
  ClockIcon,
  ArrowPathIcon,
  CheckIcon,
  ExclamationTriangleIcon,
  ArrowUpIcon,
  ArrowDownIcon,
} from '@heroicons/react/24/outline';
import API_ENDPOINTS from '../config/api';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [metrics, setMetrics] = useState({
    total_trades: 0,
    win_rate: 0,
    profit_factor: 0,
    net_profit: 0,
    total_pnl: 0,
    avg_trade_duration: 0,
    max_drawdown: 0,
    sharpe_ratio: 0,
  });

  const [algo_health, setAlgoHealth] = useState({
    health_score: 0,
    status: 'loading',
    uptime_hours: 0,
    last_signal_time: null,
    restart_count: 0,
    error_count: 0,
  });

  const [account_data, setAccountData] = useState({
    account_balance: 0,
    account_equity: 0,
    margin_level: 0,
    free_margin: 0,
    used_margin: 0,
  });

  const [live_trades, setLiveTrades] = useState([]);
  const [recent_errors, setRecentErrors] = useState([]);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [wsConnected, setWsConnected] = useState(false);

  const fetchData = async (showToast = false) => {
    try {
      if (showToast) {
        setRefreshing(true);
        toast.loading('Refreshing live data...', { id: 'refresh' });
      } else {
        setLoading(true);
      }
      
      // Fetch all live data in parallel
      const [
        metricsRes, 
        healthRes, 
        accountRes, 
        tradesRes, 
        errorsRes
      ] = await Promise.all([
        fetch(API_ENDPOINTS.TRADES_STATS),
        fetch(API_ENDPOINTS.ALGO_HEALTH),
        fetch(API_ENDPOINTS.RISK_SUMMARY),
        fetch(API_ENDPOINTS.OPEN_TRADES),
        fetch(API_ENDPOINTS.SYSTEM_STATUS)
      ]);
      
      const [
        metricsData, 
        healthData, 
        accountData, 
        tradesData, 
        errorsData
      ] = await Promise.all([
        metricsRes.json(),
        healthRes.json(),
        accountRes.json(),
        tradesRes.json(),
        errorsRes.json()
      ]);
      
      // Update all state with live data
      setMetrics({
        total_trades: metricsData.total_trades || 0,
        win_rate: metricsData.win_rate || 0,
        profit_factor: metricsData.profit_factor || 0,
        net_profit: metricsData.net_profit || 0,
        total_pnl: metricsData.total_pnl || 0,
        avg_trade_duration: metricsData.avg_trade_duration || 0,
        max_drawdown: metricsData.max_drawdown || 0,
        sharpe_ratio: metricsData.sharpe_ratio || 0,
      });
      
      setAlgoHealth({
        health_score: healthData.health_score || 0,
        status: healthData.status || 'unknown',
        uptime_hours: healthData.uptime_hours || 0,
        last_signal_time: healthData.last_signal_time || null,
        restart_count: healthData.restart_count || 0,
        error_count: healthData.error_count || 0,
      });
      
      setAccountData({
        account_balance: accountData.account_balance || 0,
        account_equity: accountData.account_equity || 0,
        margin_level: accountData.margin_level || 0,
        free_margin: accountData.free_margin || 0,
        used_margin: accountData.used_margin || 0,
      });
      
      setLiveTrades(tradesData || []);
      setRecentErrors(errorsData.recent_errors || []);
      setLastUpdate(new Date());
      
      if (showToast) {
        toast.success(`Live data updated! ${tradesData?.length || 0} open trades`, { id: 'refresh' });
      }
    } catch (err) {
      console.error('Data fetch error:', err);
      if (showToast) {
        toast.error('Failed to refresh live data', { id: 'refresh' });
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(() => fetchData(), 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // WebSocket connection for live updates
  useEffect(() => {
    const wsUrl = API_ENDPOINTS.WEBSOCKET;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsConnected(true);
      toast.success('Live updates connected', { duration: 2000 });
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.error) {
          console.error('WebSocket error:', data.error);
          toast.error(`Live update error: ${data.error}`, { duration: 3000 });
          return;
        }
        
        // Update state with live data
        if (data.pnl) {
          setMetrics(prev => ({
            ...prev,
            total_trades: data.pnl.total_trades || prev.total_trades,
            win_rate: data.pnl.win_rate || prev.win_rate,
            profit_factor: data.pnl.profit_factor || prev.profit_factor,
            net_profit: data.pnl.net_profit || prev.net_profit,
            total_pnl: data.pnl.total_pnl || prev.total_pnl,
            max_drawdown: data.pnl.max_drawdown || prev.max_drawdown
          }));
        }
        
        if (data.health) {
          setAlgoHealth(prev => ({
            ...prev,
            health_score: data.health.health_score || prev.health_score,
            status: data.health.status || prev.status,
            uptime_hours: data.health.uptime_hours || prev.uptime_hours,
            last_signal_time: data.health.last_signal_time || prev.last_signal_time,
            restart_count: data.health.restart_count || prev.restart_count,
            error_count: data.health.error_count || prev.error_count
          }));
        }
        
        if (data.account) {
          setAccountData(prev => ({
            ...prev,
            account_balance: data.account.account_balance || prev.account_balance,
            account_equity: data.account.account_equity || prev.account_equity,
            margin_level: data.account.margin_level || prev.margin_level,
            free_margin: data.account.free_margin || prev.free_margin,
            used_margin: data.account.used_margin || prev.used_margin
          }));
        }
        
        if (data.trades) {
          setLiveTrades(data.trades);
        }
        
        // Update last update time
        setLastUpdate(new Date());
        
      } catch (error) {
        console.error('Error parsing WebSocket data:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);
      toast.error('Live updates disconnected', { duration: 3000 });
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
      toast.error('WebSocket connection error', { duration: 3000 });
    };
    
    return () => {
      ws.close();
    };
  }, []);

  const formatLastUpdate = (date) => {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient">Trading Dashboard</h1>
          <p className="text-text-muted mt-2">Real-time monitoring of your BigAlgo FinTech trading system</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm text-text-muted bg-dark-800/50 px-3 py-2 rounded-lg border border-dark-600 flex items-center gap-2">
            <ClockIcon className="h-4 w-4" />
            Last updated: {formatLastUpdate(lastUpdate)}
          </div>
          <div className={`text-sm px-3 py-2 rounded-lg border flex items-center gap-2 ${
            wsConnected 
              ? 'text-green-400 bg-green-900/20 border-green-600' 
              : 'text-red-400 bg-red-900/20 border-red-600'
          }`}>
            <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            {wsConnected ? 'Live Updates' : 'Offline'}
          </div>
          <button
            onClick={() => fetchData(true)}
            disabled={loading || refreshing}
            className="btn btn-primary"
          >
            {refreshing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Refreshing...
              </>
            ) : (
              <>
                <ArrowPathIcon className="h-4 w-4" />
                Refresh
              </>
            )}
          </button>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <MetricCard
          title="Account Balance"
          value={`$${account_data.account_balance?.toLocaleString() || '0'}`}
          change="neutral"
          icon={CurrencyDollarIcon}
          delay="0s"
          clickable={true}
          subtitle="Click to view risk details"
        />

        <MetricCard
          title="Total Trades"
          value={metrics.total_trades}
          change={metrics.total_trades > 0 ? 'positive' : 'neutral'}
          icon={ChartBarIcon}
          delay="0.1s"
          clickable={true}
          subtitle="Click to view all trades"
        />

        <MetricCard
          title="Win Rate"
          value={`${metrics.win_rate}%`}
          change={metrics.win_rate >= 50 ? 'positive' : 'negative'}
          icon={metrics.win_rate >= 50 ? ArrowTrendingUpIcon : ArrowTrendingDownIcon}
          delay="0.2s"
          clickable={true}
          subtitle="Click to view trade analysis"
        />

        <MetricCard
          title="Profit Factor"
          value={metrics.profit_factor}
          change={metrics.profit_factor >= 1.5 ? 'positive' : 'negative'}
          icon={metrics.profit_factor >= 1.5 ? ArrowTrendingUpIcon : ArrowTrendingDownIcon}
          delay="0.3s"
          clickable={true}
          subtitle="Click to view performance"
        />

        <MetricCard
          title="Net Profit"
          value={`$${metrics.net_profit?.toLocaleString() || '0'}`}
          change={metrics.net_profit >= 0 ? 'positive' : 'negative'}
          icon={metrics.net_profit >= 0 ? CurrencyDollarIcon : ArrowTrendingDownIcon}
          delay="0.4s"
          clickable={true}
          subtitle="Click to view P&L details"
        />
      </div>

      {/* Health Score Card */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricCard
          title="System Health"
          value={`${algo_health.health_score}/100`}
          change={
            algo_health.health_score >= 80
              ? 'positive'
              : algo_health.health_score >= 60
              ? 'warning'
              : 'negative'
          }
          icon={HeartIcon}
          delay="0.5s"
          large={true}
          clickable={true}
          subtitle="Click to view detailed health metrics"
        />
        
        <div className="card bounce-in" style={{ animationDelay: '0.6s' }}>
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="card-title">System Status</h3>
              <div className="p-3 bg-primary-500/20 rounded-xl border border-primary-500/30">
                <TrophyIcon className="h-6 w-6 text-primary-400" />
              </div>
            </div>
            <p className="card-subtitle">Overall system performance</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-text-muted">Algorithm Status</span>
              <span className={`status-badge ${
                algo_health.status === 'healthy' ? 'positive' : 
                algo_health.status === 'warning' ? 'warning' : 'danger'
              }`}>
                {algo_health.status}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-text-muted">Health Score</span>
              <span className="text-text-primary font-mono font-bold">
                {algo_health.health_score}/100
              </span>
            </div>
            <div className="w-full bg-dark-700 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-1000 ${
                  algo_health.health_score >= 80 ? 'bg-success-500' :
                  algo_health.health_score >= 60 ? 'bg-warning-500' : 'bg-danger-500'
                }`}
                style={{ width: `${algo_health.health_score}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Live Trades Section */}
      {live_trades.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <ChartBarIcon className="h-6 w-6 text-primary-400" />
                <h3 className="card-title">Live Open Trades</h3>
                <span className="badge badge-info">{live_trades.length}</span>
              </div>
              <button
                onClick={() => window.location.href = '/trades'}
                className="btn btn-sm btn-secondary"
              >
                View All Trades
              </button>
            </div>
            <p className="card-subtitle">Real-time trading positions</p>
          </div>
          <div className="overflow-x-auto">
            <table className="table">
              <thead className="table-header">
                <tr>
                  <th className="table-header-cell">Symbol</th>
                  <th className="table-header-cell">Side</th>
                  <th className="table-header-cell">Size</th>
                  <th className="table-header-cell">Entry Price</th>
                  <th className="table-header-cell">Current P&L</th>
                  <th className="table-header-cell">Confidence</th>
                  <th className="table-header-cell">Time</th>
                </tr>
              </thead>
              <tbody className="table-body">
                {live_trades.slice(0, 5).map((trade) => (
                  <tr key={trade.id} className="table-row">
                    <td className="table-cell font-medium">{trade.symbol}</td>
                    <td className="table-cell">
                      <div className="flex items-center">
                        {trade.side === 'long' ? (
                          <ArrowUpIcon className="h-4 w-4 text-success-500" />
                        ) : (
                          <ArrowDownIcon className="h-4 w-4 text-danger-500" />
                        )}
                        <span className="ml-2 capitalize">{trade.side}</span>
                      </div>
                    </td>
                    <td className="table-cell">{trade.lot_size}</td>
                    <td className="table-cell">{trade.entry_price?.toFixed(5) || 'N/A'}</td>
                    <td className={`table-cell font-medium ${trade.pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                      {trade.pnl ? `$${trade.pnl.toFixed(2)}` : 'N/A'}
                    </td>
                    <td className="table-cell">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${(trade.confidence || 0) * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-500">
                          {trade.confidence ? (trade.confidence * 100).toFixed(0) : '0'}%
                        </span>
                      </div>
                    </td>
                    <td className="table-cell text-sm text-gray-500">
                      {new Date(trade.entry_time).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent Errors Section */}
      {recent_errors.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <ExclamationTriangleIcon className="h-6 w-6 text-warning-400" />
              <h3 className="card-title">Recent Errors</h3>
              <span className="badge badge-warning">{recent_errors.length}</span>
            </div>
            <p className="card-subtitle">Latest system errors and warnings</p>
          </div>
          <div className="space-y-3">
            {recent_errors.slice(0, 3).map((error, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-warning-500/10 rounded-lg border border-warning-500/20">
                <ExclamationTriangleIcon className="h-5 w-5 text-warning-500 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-warning-800">{error.message || 'Unknown error'}</p>
                  <p className="text-xs text-warning-600 mt-1">
                    {new Date(error.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const MetricCard = ({ title, value, change, icon: Icon, delay = "0s", large = false }) => {
  const changeColors = {
    positive: 'text-success-400',
    negative: 'text-danger-400',
    warning: 'text-warning-400',
    neutral: 'text-text-muted',
  };

  const iconColors = {
    positive: 'text-success-400',
    negative: 'text-danger-400',
    warning: 'text-warning-400',
    neutral: 'text-text-muted',
  };

  const iconBgColors = {
    positive: 'bg-success-500/20 border-success-500/30',
    negative: 'bg-danger-500/20 border-danger-500/30',
    warning: 'bg-warning-500/20 border-warning-500/30',
    neutral: 'bg-dark-700/50 border-dark-600',
  };

  return (
    <div className={`card slide-in-right ${large ? 'lg:col-span-1' : ''}`} style={{ animationDelay: delay }}>
      <div className="card-header">
        <div className="flex items-center justify-between">
          <h3 className="card-title">{title}</h3>
          <div className={`p-3 rounded-xl border ${iconBgColors[change] || iconBgColors.neutral}`}>
            <Icon className={`h-6 w-6 ${iconColors[change] || iconColors.neutral}`} />
          </div>
        </div>
        <p className="card-subtitle">
          {change === 'positive'
            ? 'Improving'
            : change === 'negative'
            ? 'Declining'
            : change === 'warning'
            ? 'At Risk'
            : 'Stable'}
        </p>
      </div>
      <div className={`font-mono font-bold ${large ? 'text-5xl' : 'text-4xl'} text-text-primary`}>
        {value}
      </div>
      <div className="mt-4">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            change === 'positive' ? 'bg-success-500 animate-pulse' :
            change === 'negative' ? 'bg-danger-500' :
            change === 'warning' ? 'bg-warning-500' : 'bg-text-muted'
          }`} />
          <span className={`text-sm font-medium ${changeColors[change] || changeColors.neutral}`}>
            {change === 'positive' ? 'Trending Up' :
             change === 'negative' ? 'Trending Down' :
             change === 'warning' ? 'Caution' : 'Neutral'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
