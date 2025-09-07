import React, { useState, useEffect } from 'react';
import { 
  CpuChipIcon,
  ChartBarIcon,
  LightBulbIcon,
  EyeIcon,
  BoltIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ClockIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';

const AIInsights = () => {
  const { fetchAIInsights, fetchMarketRegime, fetchConfidenceDistribution } = useApi();
  const [aiInsights, setAiInsights] = useState(null);
  const [marketRegime, setMarketRegime] = useState(null);
  const [confidenceDistribution, setConfidenceDistribution] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAIInsights();
    // Refresh every 15 seconds for real-time AI data
    const interval = setInterval(loadAIInsights, 15000);
    return () => clearInterval(interval);
  }, []);

  const loadAIInsights = async () => {
    setLoading(true);
    try {
      const [insights, regime, confidence] = await Promise.all([
        fetchAIInsights(),
        fetchMarketRegime(),
        fetchConfidenceDistribution()
      ]);
      setAiInsights(insights);
      setMarketRegime(regime);
      setConfidenceDistribution(confidence);
    } catch (err) {
      console.error('Error loading AI insights:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRegimeColor = (regime) => {
    switch (regime) {
      case 'trending': return 'text-success-400 bg-success-500/20 border-success-500/30';
      case 'ranging': return 'text-warning-400 bg-warning-500/20 border-warning-500/30';
      case 'volatile': return 'text-danger-400 bg-danger-500/20 border-danger-500/30';
      default: return 'text-text-muted bg-dark-700/50 border-dark-600';
    }
  };

  const getConfidenceColor = (level) => {
    switch (level) {
      case 'high': return 'text-success-400 bg-success-500/20';
      case 'medium': return 'text-warning-400 bg-warning-500/20';
      case 'low': return 'text-danger-400 bg-danger-500/20';
      default: return 'text-text-muted bg-dark-700/50';
    }
  };

  if (loading) {
    return <LoadingSpinner size="large" className="h-64" />;
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient">AI Brain Scanner</h1>
          <p className="text-text-muted mt-2">Real-time AI insights and machine learning analysis</p>
        </div>
        <div className="flex items-center space-x-2">
          <CpuChipIcon className="h-6 w-6 text-primary-500" />
          <span className="text-sm font-medium text-text-secondary">AI Monitor</span>
        </div>
      </div>

      {/* Market Regime Classification */}
      <div className="card slide-in-right">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h3 className="card-title">Market Regime Analysis</h3>
            <div className={`px-3 py-1 rounded-lg border ${getRegimeColor(marketRegime?.current_regime)}`}>
              <span className="text-sm font-medium capitalize">
                {marketRegime?.current_regime || 'Unknown'}
              </span>
            </div>
          </div>
          <p className="card-subtitle">Current market state classification</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center gap-3">
                <ArrowTrendingUpIcon className="h-6 w-6 text-success-400" />
                <div>
                  <p className="text-sm text-text-muted">Trend Strength</p>
                  <p className="text-lg font-bold text-text-primary font-mono">
                    {marketRegime?.trend_strength?.toFixed(2) || 0}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center gap-3">
                <BoltIcon className="h-6 w-6 text-warning-400" />
                <div>
                  <p className="text-sm text-text-muted">Volatility</p>
                  <p className="text-lg font-bold text-text-primary font-mono">
                    {marketRegime?.volatility_level?.toFixed(2) || 0}
                  </p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center gap-3">
                <ChartBarIcon className="h-6 w-6 text-primary-400" />
                <div>
                  <p className="text-sm text-text-muted">Market Direction</p>
                  <p className="text-lg font-bold text-text-primary font-mono">
                    {marketRegime?.direction === 'bullish' ? '↗' : 
                     marketRegime?.direction === 'bearish' ? '↘' : '↔'}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center gap-3">
                <ClockIcon className="h-6 w-6 text-purple-400" />
                <div>
                  <p className="text-sm text-text-muted">Regime Duration</p>
                  <p className="text-lg font-bold text-text-primary font-mono">
                    {marketRegime?.regime_duration_hours?.toFixed(1) || 0}h
                  </p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <h4 className="text-sm font-medium text-text-primary mb-3">Regime Confidence</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Trending</span>
                  <span className="text-text-secondary font-mono">
                    {(marketRegime?.regime_probabilities?.trending * 100)?.toFixed(1) || 0}%
                  </span>
                </div>
                <div className="w-full bg-dark-700 rounded-full h-2">
                  <div 
                    className="bg-success-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(marketRegime?.regime_probabilities?.trending * 100) || 0}%` }}
                  />
                </div>
                
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Ranging</span>
                  <span className="text-text-secondary font-mono">
                    {(marketRegime?.regime_probabilities?.ranging * 100)?.toFixed(1) || 0}%
                  </span>
                </div>
                <div className="w-full bg-dark-700 rounded-full h-2">
                  <div 
                    className="bg-warning-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(marketRegime?.regime_probabilities?.ranging * 100) || 0}%` }}
                  />
                </div>
                
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Volatile</span>
                  <span className="text-text-secondary font-mono">
                    {(marketRegime?.regime_probabilities?.volatile * 100)?.toFixed(1) || 0}%
                  </span>
                </div>
                <div className="w-full bg-dark-700 rounded-full h-2">
                  <div 
                    className="bg-danger-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(marketRegime?.regime_probabilities?.volatile * 100) || 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Confidence Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card bounce-in">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="card-title">Confidence Distribution</h3>
              <div className="flex items-center gap-2">
                <LightBulbIcon className="h-5 w-5 text-primary-400" />
                <span className="text-sm text-text-muted">Live</span>
              </div>
            </div>
            <p className="card-subtitle">ML model confidence levels</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-success-500 animate-pulse" />
                <span className="text-sm text-text-muted">High Confidence</span>
              </div>
              <div className="text-right">
                <p className="text-lg font-bold text-success-400 font-mono">
                  {confidenceDistribution?.high_confidence_count || 0}
                </p>
                <p className="text-xs text-text-muted">
                  {(confidenceDistribution?.high_confidence_percentage * 100)?.toFixed(1) || 0}%
                </p>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-warning-500" />
                <span className="text-sm text-text-muted">Medium Confidence</span>
              </div>
              <div className="text-right">
                <p className="text-lg font-bold text-warning-400 font-mono">
                  {confidenceDistribution?.medium_confidence_count || 0}
                </p>
                <p className="text-xs text-text-muted">
                  {(confidenceDistribution?.medium_confidence_percentage * 100)?.toFixed(1) || 0}%
                </p>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-danger-500" />
                <span className="text-sm text-text-muted">Low Confidence</span>
              </div>
              <div className="text-right">
                <p className="text-lg font-bold text-danger-400 font-mono">
                  {confidenceDistribution?.low_confidence_count || 0}
                </p>
                <p className="text-xs text-text-muted">
                  {(confidenceDistribution?.low_confidence_percentage * 100)?.toFixed(1) || 0}%
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Last Prediction Details */}
        <div className="card bounce-in" style={{ animationDelay: '0.2s' }}>
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="card-title">Last Prediction</h3>
              <div className="flex items-center gap-2">
                <EyeIcon className="h-5 w-5 text-primary-400" />
                <span className="text-sm text-text-muted">Details</span>
              </div>
            </div>
            <p className="card-subtitle">Most recent ML prediction analysis</p>
          </div>
          <div className="space-y-4">
            <div className="p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-text-primary">Prediction</span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  aiInsights?.last_prediction?.signal === 'long' ? 'bg-success-500/20 text-success-400' :
                  aiInsights?.last_prediction?.signal === 'short' ? 'bg-danger-500/20 text-danger-400' :
                  'bg-gray-500/20 text-gray-400'
                }`}>
                  {aiInsights?.last_prediction?.signal?.toUpperCase() || 'NONE'}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-text-muted">Confidence</span>
                  <span className="text-text-secondary font-mono">
                    {(aiInsights?.last_prediction?.confidence * 100)?.toFixed(1) || 0}%
                  </span>
                </div>
                <div className="w-full bg-dark-700 rounded-full h-2">
                  <div 
                    className="bg-primary-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(aiInsights?.last_prediction?.confidence * 100) || 0}%` }}
                  />
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <h4 className="text-sm font-medium text-text-primary mb-3">Top Contributing Features</h4>
              <div className="space-y-2">
                {aiInsights?.last_prediction?.top_features?.slice(0, 3).map((feature, index) => (
                  <div key={index} className="flex justify-between text-sm">
                    <span className="text-text-muted">{feature.name}</span>
                    <span className="text-text-secondary font-mono">
                      {feature.impact?.toFixed(3) || 0}
                    </span>
                  </div>
                )) || (
                  <p className="text-text-muted text-sm">No feature data available</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Nearest Neighbors & Similarity */}
      <div className="card fade-in-up">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <h3 className="card-title">Lorentzian Distance Analysis</h3>
            <div className="flex items-center gap-2">
              <CpuChipIcon className="h-5 w-5 text-primary-400" />
              <span className="text-sm text-text-muted">k-NN Similarity</span>
            </div>
          </div>
          <p className="card-subtitle">Nearest neighbors and similarity scores</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <h4 className="text-sm font-medium text-text-primary mb-3">Nearest Neighbors</h4>
              <div className="space-y-2">
                {aiInsights?.nearest_neighbors?.slice(0, 5).map((neighbor, index) => (
                  <div key={index} className="flex justify-between text-sm">
                    <span className="text-text-muted">Neighbor {index + 1}</span>
                    <span className="text-text-secondary font-mono">
                      {neighbor.similarity?.toFixed(4) || 0}
                    </span>
                  </div>
                )) || (
                  <p className="text-text-muted text-sm">No neighbor data available</p>
                )}
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="p-4 bg-dark-800/50 rounded-lg border border-dark-600">
              <h4 className="text-sm font-medium text-text-primary mb-3">Signal Queue Status</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-muted">Signals Queued</span>
                  <span className="text-lg font-bold text-warning-400 font-mono">
                    {aiInsights?.signals_queued || 0}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-muted">Signals Executed</span>
                  <span className="text-lg font-bold text-success-400 font-mono">
                    {aiInsights?.signals_executed || 0}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-muted">Execution Rate</span>
                  <span className="text-lg font-bold text-primary-400 font-mono">
                    {aiInsights?.execution_rate?.toFixed(1) || 0}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AI Alerts & Unusual Behavior */}
      {aiInsights?.alerts?.length > 0 && (
        <div className="card fade-in-up">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="card-title">AI Alerts</h3>
              <div className="flex items-center gap-2">
                <ExclamationTriangleIcon className="h-5 w-5 text-warning-400" />
                <span className="text-sm text-text-muted">Unusual Behavior</span>
              </div>
            </div>
            <p className="card-subtitle">AI-detected anomalies and alerts</p>
          </div>
          <div className="space-y-3">
            {aiInsights.alerts.map((alert, index) => (
              <div key={index} className={`p-4 rounded-lg border ${
                alert.severity === 'high' ? 'bg-danger-500/10 border-danger-500/30' :
                alert.severity === 'medium' ? 'bg-warning-500/10 border-warning-500/30' :
                'bg-primary-500/10 border-primary-500/30'
              }`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-sm font-medium text-text-primary">{alert.type}</p>
                    <p className="text-xs text-text-muted mt-1">{alert.message}</p>
                  </div>
                  <div className="text-xs text-text-muted ml-3">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AIInsights;
