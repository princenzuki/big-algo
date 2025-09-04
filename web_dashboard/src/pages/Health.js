import React, { useState, useEffect } from 'react';
import { 
  HeartIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ChartBarIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';

const Health = () => {
  const { fetchAlgoHealth, fetchSystemStatus } = useApi();
  const [algoHealth, setAlgoHealth] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHealthData();
    // Refresh every 30 seconds
    const interval = setInterval(loadHealthData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadHealthData = async () => {
    setLoading(true);
    try {
      const [health, status] = await Promise.all([
        fetchAlgoHealth(),
        fetchSystemStatus()
      ]);
      setAlgoHealth(health);
      setSystemStatus(status);
    } catch (err) {
      console.error('Error loading health data:', err);
    } finally {
      setLoading(false);
    }
  };

  const getHealthStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'text-success-600 bg-success-100';
      case 'warning':
        return 'text-warning-600 bg-warning-100';
      case 'critical':
        return 'text-danger-600 bg-danger-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getHealthScoreColor = (score) => {
    if (score >= 80) return 'text-success-600';
    if (score >= 60) return 'text-warning-600';
    return 'text-danger-600';
  };

  const getHealthScoreBg = (score) => {
    if (score >= 80) return 'bg-success-500';
    if (score >= 60) return 'bg-warning-500';
    return 'bg-danger-500';
  };

  const formatUptime = (hours) => {
    if (hours < 1) return `${(hours * 60).toFixed(0)} minutes`;
    if (hours < 24) return `${hours.toFixed(1)} hours`;
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    return `${days}d ${remainingHours.toFixed(1)}h`;
  };

  if (loading) {
    return <LoadingSpinner size="large" className="h-64" />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Algorithm Health</h1>
          <p className="text-gray-500">Monitor system performance and health metrics</p>
        </div>
        <div className="flex items-center space-x-2">
          <HeartIcon className="h-6 w-6 text-primary-500" />
          <span className="text-sm font-medium text-gray-700">Health Monitor</span>
        </div>
      </div>

      {/* Health Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Overall Health Score */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Health Score</h3>
            <p className="card-subtitle">Overall system health</p>
          </div>
          <div className="text-center">
            <div className={`text-4xl font-bold mb-2 ${getHealthScoreColor(algoHealth?.health_score || 0)}`}>
              {algoHealth?.health_score || 0}
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
              <div
                className={`h-3 rounded-full transition-all duration-300 ${getHealthScoreBg(algoHealth?.health_score || 0)}`}
                style={{ width: `${algoHealth?.health_score || 0}%` }}
              />
            </div>
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getHealthStatusColor(algoHealth?.status || 'unknown')}`}>
              {algoHealth?.status?.charAt(0).toUpperCase() + algoHealth?.status?.slice(1) || 'Unknown'}
            </div>
          </div>
        </div>

        {/* Uptime */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Uptime</h3>
            <p className="card-subtitle">System running time</p>
          </div>
          <div className="flex items-center">
            <ClockIcon className="h-8 w-8 text-primary-500" />
            <div className="ml-4">
              <p className="text-2xl font-bold text-gray-900">
                {formatUptime(algoHealth?.uptime_hours || 0)}
              </p>
              <p className="text-sm text-gray-500">Since last restart</p>
            </div>
          </div>
        </div>

        {/* Success Rate */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Success Rate</h3>
            <p className="card-subtitle">Trade execution success</p>
          </div>
          <div className="flex items-center">
            <CheckCircleIcon className="h-8 w-8 text-success-500" />
            <div className="ml-4">
              <p className="text-2xl font-bold text-gray-900">
                {algoHealth?.trade_execution_success_rate?.toFixed(1) || 0}%
              </p>
              <p className="text-sm text-gray-500">Execution success</p>
            </div>
          </div>
        </div>

        {/* Avg Confidence */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Avg Confidence</h3>
            <p className="card-subtitle">ML model confidence</p>
          </div>
          <div className="flex items-center">
            <ChartBarIcon className="h-8 w-8 text-primary-500" />
            <div className="ml-4">
              <p className="text-2xl font-bold text-gray-900">
                {((algoHealth?.avg_trade_confidence || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-gray-500">ML confidence</p>
            </div>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Components */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">System Components</h3>
            <p className="card-subtitle">Component status and connectivity</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <CpuChipIcon className="h-5 w-5 text-gray-400 mr-3" />
                <span className="text-sm text-gray-500">Trading Bot</span>
              </div>
              <div className="flex items-center">
                <div className={`w-2 h-2 rounded-full mr-2 ${systemStatus?.bot_running ? 'bg-success-400' : 'bg-danger-400'}`} />
                <span className="text-sm font-medium text-gray-900">
                  {systemStatus?.bot_running ? 'Running' : 'Stopped'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <ChartBarIcon className="h-5 w-5 text-gray-400 mr-3" />
                <span className="text-sm text-gray-500">Broker Connection</span>
              </div>
              <div className="flex items-center">
                <div className={`w-2 h-2 rounded-full mr-2 ${systemStatus?.broker_connected ? 'bg-success-400' : 'bg-danger-400'}`} />
                <span className="text-sm font-medium text-gray-900">
                  {systemStatus?.broker_connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <HeartIcon className="h-5 w-5 text-gray-400 mr-3" />
                <span className="text-sm text-gray-500">Database</span>
              </div>
              <div className="flex items-center">
                <div className={`w-2 h-2 rounded-full mr-2 ${systemStatus?.database_connected ? 'bg-success-400' : 'bg-danger-400'}`} />
                <span className="text-sm font-medium text-gray-900">
                  {systemStatus?.database_connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <ClockIcon className="h-5 w-5 text-gray-400 mr-3" />
                <span className="text-sm text-gray-500">Last Heartbeat</span>
              </div>
              <span className="text-sm font-medium text-gray-900">
                {systemStatus?.last_heartbeat ? 
                  new Date(systemStatus.last_heartbeat).toLocaleTimeString() : 
                  'Never'
                }
              </span>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Performance Metrics</h3>
            <p className="card-subtitle">System performance indicators</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">Signals Processed</span>
              <span className="text-sm font-medium text-gray-900">
                {systemStatus?.total_signals_processed || 0}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">Active Symbols</span>
              <span className="text-sm font-medium text-gray-900">
                {systemStatus?.active_symbols?.length || 0}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">Errors (Last Hour)</span>
              <span className={`text-sm font-medium ${(systemStatus?.errors_last_hour || 0) > 5 ? 'text-danger-600' : 'text-gray-900'}`}>
                {systemStatus?.errors_last_hour || 0}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500">Risk Exposure</span>
              <span className="text-sm font-medium text-gray-900">
                {algoHealth?.risk_exposure_percent?.toFixed(1) || 0}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Health Alerts */}
      {algoHealth?.status !== 'healthy' && (
        <div className="card border-l-4 border-danger-500 bg-danger-50">
          <div className="flex items-center p-4">
            <ExclamationTriangleIcon className="h-6 w-6 text-danger-500 mr-3" />
            <div>
              <h3 className="text-sm font-medium text-danger-800">Health Alert</h3>
              <p className="text-sm text-danger-700">
                {algoHealth?.status === 'warning' 
                  ? 'System health is degraded. Monitor closely for potential issues.'
                  : 'System health is critical. Immediate attention required.'
                }
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Active Symbols */}
      {systemStatus?.active_symbols?.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Active Symbols</h3>
            <p className="card-subtitle">Currently monitored trading symbols</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {systemStatus.active_symbols.map((symbol) => (
              <div key={symbol} className="flex items-center justify-center p-3 bg-primary-50 rounded-lg">
                <span className="text-sm font-medium text-primary-900">{symbol}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Health History Chart would go here */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Health History</h3>
          <p className="card-subtitle">Health score over time</p>
        </div>
        <div className="h-64 flex items-center justify-center text-gray-500">
          <div className="text-center">
            <ChartBarIcon className="h-12 w-12 mx-auto mb-2 text-gray-400" />
            <p>Health history chart coming soon</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Health;
