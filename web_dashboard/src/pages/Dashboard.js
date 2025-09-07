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
} from '@heroicons/react/24/outline';
import API_ENDPOINTS from '../config/api';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [metrics, setMetrics] = useState({
    total_trades: 0,
    win_rate: 0,
    profit_factor: 0,
    net_profit: 0,
  });

  const [algo_health, setAlgoHealth] = useState({
    health_score: 0,
    status: 'loading',
  });

  const [account_data, setAccountData] = useState({
    account_balance: 0,
    account_equity: 0,
  });

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const fetchData = async (showToast = false) => {
    try {
      if (showToast) {
        setRefreshing(true);
        toast.loading('Refreshing data...', { id: 'refresh' });
      } else {
        setLoading(true);
      }
      
      const [metricsRes, healthRes, accountRes] = await Promise.all([
        fetch(API_ENDPOINTS.TRADES_STATS),
        fetch(API_ENDPOINTS.ALGO_HEALTH),
        fetch(API_ENDPOINTS.RISK_SUMMARY)
      ]);
      
      const [metricsData, healthData, accountData] = await Promise.all([
        metricsRes.json(),
        healthRes.json(),
        accountRes.json()
      ]);
      
      setMetrics(metricsData);
      setAlgoHealth(healthData);
      setAccountData(accountData);
      setLastUpdate(new Date());
      
      if (showToast) {
        toast.success('Data refreshed successfully!', { id: 'refresh' });
      }
    } catch (err) {
      console.error('Data fetch error:', err);
      if (showToast) {
        toast.error('Failed to refresh data', { id: 'refresh' });
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
          value={`$${metrics.net_profit}`}
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
