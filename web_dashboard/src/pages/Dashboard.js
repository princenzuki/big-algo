import React, { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { 
  TrendingUpIcon, 
  TrendingDownIcon, 
  CurrencyDollarIcon,
  ChartBarIcon,
  ClockIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import MetricCard from '../components/MetricCard';
import LoadingSpinner from '../components/LoadingSpinner';

const Dashboard = () => {
  const { dashboardData, loading, error, fetchPnLData } = useApi();
  const [selectedPeriod, setSelectedPeriod] = useState('daily');
  const [pnlData, setPnlData] = useState(null);
  const [pnlLoading, setPnlLoading] = useState(false);

  const periods = [
    { value: 'daily', label: 'Daily' },
    { value: 'weekly', label: 'Weekly' },
    { value: 'monthly', label: 'Monthly' },
    { value: 'quarterly', label: 'Quarterly' },
    { value: 'yearly', label: 'Yearly' }
  ];

  const COLORS = {
    profit: '#22c55e',
    loss: '#ef4444',
    neutral: '#6b7280'
  };

  useEffect(() => {
    const loadPnLData = async () => {
      setPnlLoading(true);
      try {
        const data = await fetchPnLData(selectedPeriod);
        setPnlData(data);
      } catch (err) {
        console.error('Error loading P&L data:', err);
      } finally {
        setPnlLoading(false);
      }
    };

    loadPnLData();
  }, [selectedPeriod, fetchPnLData]);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-danger-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Error Loading Dashboard</h3>
          <p className="text-gray-500">{error}</p>
        </div>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Data Available</h3>
          <p className="text-gray-500">Dashboard data is not available at the moment.</p>
        </div>
      </div>
    );
  }

  const { portfolio_stats, algo_health, risk_summary, session_info } = dashboardData;

  // Prepare P&L chart data
  const preparePnLChartData = (data) => {
    if (!data || !data.data) return [];
    
    return data.data.map(item => ({
      ...item,
      pnl: parseFloat(item.pnl || 0)
    }));
  };

  // Prepare win/loss pie chart data
  const winLossData = [
    { name: 'Wins', value: portfolio_stats.winning_trades, color: COLORS.profit },
    { name: 'Losses', value: portfolio_stats.losing_trades, color: COLORS.loss }
  ];

  // Prepare confidence distribution pie chart data
  const confidenceData = [
    { name: 'High', value: dashboardData.confidence_distribution.high, color: COLORS.profit },
    { name: 'Medium', value: dashboardData.confidence_distribution.medium, color: COLORS.neutral },
    { name: 'Low', value: dashboardData.confidence_distribution.low, color: COLORS.loss }
  ];

  // Calculate equity curve data
  const equityCurveData = preparePnLChartData(pnlData).map((item, index) => ({
    ...item,
    cumulative: preparePnLChartData(pnlData)
      .slice(0, index + 1)
      .reduce((sum, item) => sum + item.pnl, 0)
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Trading Dashboard</h1>
          <p className="text-gray-500">Monitor your Lorentzian ML trading performance</p>
        </div>
        <div className="flex items-center space-x-2">
          <ClockIcon className="h-5 w-5 text-gray-400" />
          <span className="text-sm text-gray-500">
            {session_info.current_session} session
          </span>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total P&L"
          value={`$${portfolio_stats.total_pnl.toFixed(2)}`}
          change={portfolio_stats.total_pnl >= 0 ? 'positive' : 'negative'}
          icon={CurrencyDollarIcon}
        />
        <MetricCard
          title="Win Rate"
          value={`${portfolio_stats.win_rate.toFixed(1)}%`}
          change={portfolio_stats.win_rate >= 50 ? 'positive' : 'negative'}
          icon={TrendingUpIcon}
        />
        <MetricCard
          title="Total Trades"
          value={portfolio_stats.total_trades.toString()}
          change="neutral"
          icon={ChartBarIcon}
        />
        <MetricCard
          title="Health Score"
          value={`${algo_health.health_score}/100`}
          change={algo_health.health_score >= 80 ? 'positive' : algo_health.health_score >= 60 ? 'warning' : 'negative'}
          icon={algo_health.status === 'healthy' ? TrendingUpIcon : TrendingDownIcon}
        />
      </div>

      {/* P&L Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Curve */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="card-title">Equity Curve</h3>
                <p className="card-subtitle">Cumulative P&L over time</p>
              </div>
              <select
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
                className="select text-sm"
              >
                {periods.map(period => (
                  <option key={period.value} value={period.value}>
                    {period.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="h-80">
            {pnlLoading ? (
              <div className="flex items-center justify-center h-full">
                <LoadingSpinner />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={equityCurveData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey={selectedPeriod === 'daily' ? 'date' : selectedPeriod === 'weekly' ? 'week' : selectedPeriod === 'monthly' ? 'month' : selectedPeriod === 'quarterly' ? 'quarter' : 'year'}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    formatter={(value) => [`$${value.toFixed(2)}`, 'Cumulative P&L']}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="cumulative" 
                    stroke={COLORS.profit} 
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* P&L Bar Chart */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">P&L by Period</h3>
            <p className="card-subtitle">Profit/Loss breakdown</p>
          </div>
          <div className="h-80">
            {pnlLoading ? (
              <div className="flex items-center justify-center h-full">
                <LoadingSpinner />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={preparePnLChartData(pnlData)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey={selectedPeriod === 'daily' ? 'date' : selectedPeriod === 'weekly' ? 'week' : selectedPeriod === 'monthly' ? 'month' : selectedPeriod === 'quarterly' ? 'quarter' : 'year'}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    formatter={(value) => [`$${value.toFixed(2)}`, 'P&L']}
                    labelStyle={{ color: '#374151' }}
                  />
                  <Bar 
                    dataKey="pnl" 
                    fill={(entry) => entry.pnl >= 0 ? COLORS.profit : COLORS.loss}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Win/Loss and Confidence Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Win/Loss Ratio */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Win/Loss Ratio</h3>
            <p className="card-subtitle">Trade outcome distribution</p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={winLossData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {winLossData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [value, 'Trades']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confidence Distribution */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Confidence Distribution</h3>
            <p className="card-subtitle">ML confidence levels</p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={confidenceData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {confidenceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [value, 'Trades']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Performance Metrics</h3>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Avg Win</span>
              <span className="text-sm font-medium text-success-600">
                ${portfolio_stats.avg_win.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Avg Loss</span>
              <span className="text-sm font-medium text-danger-600">
                ${portfolio_stats.avg_loss.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Profit Factor</span>
              <span className="text-sm font-medium text-gray-900">
                {portfolio_stats.profit_factor.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Max Drawdown</span>
              <span className="text-sm font-medium text-danger-600">
                ${portfolio_stats.max_drawdown.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Risk Status</h3>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Current Risk</span>
              <span className="text-sm font-medium text-gray-900">
                {risk_summary.current_risk_percent.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Max Risk</span>
              <span className="text-sm font-medium text-gray-900">
                {risk_summary.max_risk_percent.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Open Positions</span>
              <span className="text-sm font-medium text-gray-900">
                {risk_summary.open_positions}/{risk_summary.max_positions}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Account Equity</span>
              <span className="text-sm font-medium text-gray-900">
                ${risk_summary.account_equity.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Algo Health</h3>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Uptime</span>
              <span className="text-sm font-medium text-gray-900">
                {algo_health.uptime_hours.toFixed(1)}h
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Success Rate</span>
              <span className="text-sm font-medium text-gray-900">
                {algo_health.trade_execution_success_rate.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Avg Confidence</span>
              <span className="text-sm font-medium text-gray-900">
                {(algo_health.avg_trade_confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Status</span>
              <span className={`text-sm font-medium ${
                algo_health.status === 'healthy' ? 'text-success-600' :
                algo_health.status === 'warning' ? 'text-warning-600' :
                'text-danger-600'
              }`}>
                {algo_health.status.charAt(0).toUpperCase() + algo_health.status.slice(1)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
