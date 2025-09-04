import React, { useState, useEffect } from 'react';
import { 
  ShieldCheckIcon, 
  ExclamationTriangleIcon,
  ClockIcon,
  CurrencyDollarIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';

const Risk = () => {
  const { fetchRiskSummary, fetchRiskMetrics } = useApi();
  const [riskSummary, setRiskSummary] = useState(null);
  const [riskMetrics, setRiskMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRiskData();
  }, []);

  const loadRiskData = async () => {
    setLoading(true);
    try {
      const [summary, metrics] = await Promise.all([
        fetchRiskSummary(),
        fetchRiskMetrics()
      ]);
      setRiskSummary(summary);
      setRiskMetrics(metrics);
    } catch (err) {
      console.error('Error loading risk data:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const getRiskLevel = (riskPercent) => {
    if (riskPercent < 5) return { level: 'Low', color: 'text-success-600', bgColor: 'bg-success-100' };
    if (riskPercent < 8) return { level: 'Medium', color: 'text-warning-600', bgColor: 'bg-warning-100' };
    return { level: 'High', color: 'text-danger-600', bgColor: 'bg-danger-100' };
  };

  const getRiskThermometerColor = (value) => {
    if (value < 30) return 'bg-success-500';
    if (value < 70) return 'bg-warning-500';
    return 'bg-danger-500';
  };

  if (loading) {
    return <LoadingSpinner size="large" className="h-64" />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Risk Management</h1>
          <p className="text-gray-500">Monitor risk exposure and position limits</p>
        </div>
        <div className="flex items-center space-x-2">
          <ShieldCheckIcon className="h-6 w-6 text-primary-500" />
          <span className="text-sm font-medium text-gray-700">Risk Controls Active</span>
        </div>
      </div>

      {/* Risk Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Thermometer */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Risk Exposure</h3>
            <p className="card-subtitle">Current vs Maximum Risk</p>
          </div>
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-900 mb-2">
                {riskSummary?.current_risk_percent.toFixed(1)}%
              </div>
              <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                getRiskLevel(riskSummary?.current_risk_percent || 0).bgColor
              } ${getRiskLevel(riskSummary?.current_risk_percent || 0).color}`}>
                {getRiskLevel(riskSummary?.current_risk_percent || 0).level} Risk
              </div>
            </div>
            
            {/* Risk Thermometer */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-500">
                <span>0%</span>
                <span>Max: {riskSummary?.max_risk_percent.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all duration-300 ${getRiskThermometerColor(riskMetrics?.risk_thermometer || 0)}`}
                  style={{ width: `${Math.min(riskMetrics?.risk_thermometer || 0, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Account Information */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Account Status</h3>
            <p className="card-subtitle">Account balance and equity</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <CurrencyDollarIcon className="h-5 w-5 text-gray-400 mr-2" />
                <span className="text-sm text-gray-500">Balance</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {formatCurrency(riskSummary?.account_balance || 0)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <ChartBarIcon className="h-5 w-5 text-gray-400 mr-2" />
                <span className="text-sm text-gray-500">Equity</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {formatCurrency(riskSummary?.account_equity || 0)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <ShieldCheckIcon className="h-5 w-5 text-gray-400 mr-2" />
                <span className="text-sm text-gray-500">Free Margin</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {formatCurrency((riskSummary?.account_equity || 0) - (riskSummary?.account_balance || 0))}
              </span>
            </div>
          </div>
        </div>

        {/* Position Limits */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Position Limits</h3>
            <p className="card-subtitle">Open positions and limits</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">Open Positions</span>
              <span className="text-sm font-medium text-gray-900">
                {riskSummary?.open_positions || 0} / {riskSummary?.max_positions || 0}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-primary-600 h-2 rounded-full"
                style={{ 
                  width: `${((riskSummary?.open_positions || 0) / (riskSummary?.max_positions || 1)) * 100}%` 
                }}
              />
            </div>
            <div className="text-xs text-gray-500">
              {riskSummary?.max_positions - (riskSummary?.open_positions || 0)} positions available
            </div>
          </div>
        </div>
      </div>

      {/* Open Positions */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Open Positions</h3>
          <p className="card-subtitle">Current risk exposure by position</p>
        </div>
        <div className="overflow-x-auto">
          <table className="table">
            <thead className="table-header">
              <tr>
                <th className="table-header-cell">Symbol</th>
                <th className="table-header-cell">Side</th>
                <th className="table-header-cell">Size</th>
                <th className="table-header-cell">Entry Price</th>
                <th className="table-header-cell">Stop Loss</th>
                <th className="table-header-cell">Take Profit</th>
                <th className="table-header-cell">Risk Amount</th>
                <th className="table-header-cell">Confidence</th>
                <th className="table-header-cell">Opened</th>
              </tr>
            </thead>
            <tbody className="table-body">
              {riskSummary?.positions?.length === 0 ? (
                <tr>
                  <td colSpan="9" className="table-cell text-center py-8 text-gray-500">
                    No open positions
                  </td>
                </tr>
              ) : (
                riskSummary?.positions?.map((position, index) => (
                  <tr key={index} className="table-row">
                    <td className="table-cell font-medium">{position.symbol}</td>
                    <td className="table-cell">
                      <span className={`badge ${position.side === 'long' ? 'badge-success' : 'badge-danger'}`}>
                        {position.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="table-cell">{position.lot_size}</td>
                    <td className="table-cell">{position.entry_price.toFixed(5)}</td>
                    <td className="table-cell">{position.stop_loss.toFixed(5)}</td>
                    <td className="table-cell">{position.take_profit.toFixed(5)}</td>
                    <td className="table-cell font-medium text-danger-600">
                      {formatCurrency(position.risk_amount)}
                    </td>
                    <td className="table-cell">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${position.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-500">
                          {(position.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="table-cell text-sm text-gray-500">
                      {new Date(position.opened_at).toLocaleString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cooldowns */}
      {riskMetrics?.cooldown_symbols?.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Cooldown Periods</h3>
            <p className="card-subtitle">Symbols in cooldown period</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {riskMetrics.cooldown_symbols.map((symbol) => (
              <div key={symbol} className="flex items-center p-3 bg-warning-50 rounded-lg">
                <ClockIcon className="h-5 w-5 text-warning-500 mr-3" />
                <div>
                  <div className="text-sm font-medium text-gray-900">{symbol}</div>
                  <div className="text-xs text-gray-500">In cooldown</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk Warnings */}
      {riskSummary?.current_risk_percent > 8 && (
        <div className="card border-l-4 border-danger-500 bg-danger-50">
          <div className="flex items-center p-4">
            <ExclamationTriangleIcon className="h-6 w-6 text-danger-500 mr-3" />
            <div>
              <h3 className="text-sm font-medium text-danger-800">High Risk Warning</h3>
              <p className="text-sm text-danger-700">
                Current risk exposure ({riskSummary.current_risk_percent.toFixed(1)}%) is approaching the maximum limit.
                Consider reducing position sizes or closing some positions.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Risk;
