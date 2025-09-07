import React, { useState, useEffect } from 'react';
import { 
  ArrowUpIcon, 
  ArrowDownIcon, 
  ClockIcon,
  CurrencyDollarIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';

const Trades = () => {
  const { fetchTrades, fetchTradesBySymbol, fetchTradeStats } = useApi();
  const [activeTab, setActiveTab] = useState('open');
  const [trades, setTrades] = useState([]);
  const [tradeStats, setTradeStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState('all');

  const tabs = [
    { id: 'open', name: 'Open Trades', count: 0 },
    { id: 'closed', name: 'Closed Trades', count: 0 },
    { id: 'analyzing', name: 'Analyzing', count: 0 }
  ];

  useEffect(() => {
    loadTrades();
    loadTradeStats();
  }, [activeTab, selectedSymbol]);

  const loadTrades = async () => {
    setLoading(true);
    try {
      let data = [];
      if (selectedSymbol === 'all') {
        data = await fetchTrades(activeTab, 100);
      } else {
        data = await fetchTradesBySymbol(selectedSymbol);
        // Filter by status if needed
        if (activeTab !== 'all') {
          data = data.filter(trade => trade.status === activeTab);
        }
      }
      setTrades(data);
    } catch (err) {
      console.error('Error loading trades:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadTradeStats = async () => {
    try {
      const stats = await fetchTradeStats();
      setTradeStats(stats);
    } catch (err) {
      console.error('Error loading trade stats:', err);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      open: { color: 'badge-info', text: 'Open' },
      closed: { color: 'badge-gray', text: 'Closed' },
      stopped: { color: 'badge-danger', text: 'Stopped' }
    };
    
    const config = statusConfig[status] || { color: 'badge-gray', text: status };
    return <span className={`badge ${config.color}`}>{config.text}</span>;
  };

  const getSideIcon = (side) => {
    return side === 'long' ? (
      <ArrowUpIcon className="h-4 w-4 text-success-500" />
    ) : (
      <ArrowDownIcon className="h-4 w-4 text-danger-500" />
    );
  };

  const getPnLColor = (pnl) => {
    if (!pnl) return 'text-gray-500';
    return pnl >= 0 ? 'text-success-600' : 'text-danger-600';
  };

  if (loading) {
    return <LoadingSpinner size="large" className="h-64" />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Trade Management</h1>
          <p className="text-gray-500">Monitor and analyze your trading positions</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="select"
          >
            <option value="all">All Symbols</option>
            <option value="EURUSD">EURUSD</option>
            <option value="GBPUSD">GBPUSD</option>
            <option value="USDJPY">USDJPY</option>
            <option value="BTCUSD">BTCUSD</option>
          </select>
        </div>
      </div>

      {/* Trade Stats */}
      {tradeStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card p-4">
            <div className="flex items-center">
              <ChartBarIcon className="h-8 w-8 text-primary-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Trades</p>
                <p className="text-2xl font-bold text-gray-900">{tradeStats.total_trades}</p>
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center">
              <CurrencyDollarIcon className="h-8 w-8 text-success-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total P&L</p>
                <p className={`text-2xl font-bold ${getPnLColor(tradeStats.total_pnl)}`}>
                  {formatCurrency(tradeStats.total_pnl)}
                </p>
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center">
              <ArrowUpIcon className="h-8 w-8 text-success-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Win Rate</p>
                <p className="text-2xl font-bold text-gray-900">{tradeStats?.win_rate?.toFixed(1) || '0.0'}%</p>
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center">
              <ClockIcon className="h-8 w-8 text-warning-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Avg Duration</p>
                <p className="text-2xl font-bold text-gray-900">{tradeStats?.avg_trade_duration?.toFixed(1) || '0.0'}h</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.name}
              {tab.count > 0 && (
                <span className="ml-2 bg-gray-100 text-gray-600 py-0.5 px-2 rounded-full text-xs">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Trades Table */}
      <div className="card">
        <div className="overflow-x-auto">
          <table className="table">
            <thead className="table-header">
              <tr>
                <th className="table-header-cell">Symbol</th>
                <th className="table-header-cell">Side</th>
                <th className="table-header-cell">Size</th>
                <th className="table-header-cell">Entry Price</th>
                <th className="table-header-cell">Current Price</th>
                <th className="table-header-cell">Stop Loss</th>
                <th className="table-header-cell">Take Profit</th>
                <th className="table-header-cell">P&L</th>
                <th className="table-header-cell">Confidence</th>
                <th className="table-header-cell">Status</th>
                <th className="table-header-cell">Time</th>
              </tr>
            </thead>
            <tbody className="table-body">
              {trades.length === 0 ? (
                <tr>
                  <td colSpan="11" className="table-cell text-center py-8 text-gray-500">
                    No {activeTab} trades found
                  </td>
                </tr>
              ) : (
                trades.map((trade) => (
                  <tr key={trade.id} className="table-row">
                    <td className="table-cell font-medium">{trade.symbol}</td>
                    <td className="table-cell">
                      <div className="flex items-center">
                        {getSideIcon(trade.side)}
                        <span className="ml-2 capitalize">{trade.side}</span>
                      </div>
                    </td>
                    <td className="table-cell">{trade.lot_size}</td>
                    <td className="table-cell">{trade.entry_price?.toFixed(5) || 'N/A'}</td>
                    <td className="table-cell">
                      {trade.exit_price ? trade.exit_price.toFixed(5) : 'N/A'}
                    </td>
                    <td className="table-cell">{trade.stop_loss?.toFixed(5) || 'N/A'}</td>
                    <td className="table-cell">{trade.take_profit?.toFixed(5) || 'N/A'}</td>
                    <td className={`table-cell font-medium ${getPnLColor(trade.pnl)}`}>
                      {trade.pnl ? formatCurrency(trade.pnl) : 'N/A'}
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
                    <td className="table-cell">{getStatusBadge(trade.status)}</td>
                    <td className="table-cell text-sm text-gray-500">
                      {formatDate(trade.entry_time)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Trade Details Modal would go here */}
    </div>
  );
};

export default Trades;
