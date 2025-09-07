import React, { useState, useEffect } from 'react';
import { 
  ArrowUpIcon, 
  ArrowDownIcon, 
  ClockIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  EyeIcon,
  XMarkIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';
import toast from 'react-hot-toast';

const Trades = () => {
  const { fetchTrades, fetchTradesBySymbol, fetchTradeStats } = useApi();
  const [activeTab, setActiveTab] = useState('open');
  const [trades, setTrades] = useState([]);
  const [tradeStats, setTradeStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState('all');
  const [selectedTrade, setSelectedTrade] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

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

  const handleTradeAction = async (tradeId, action) => {
    setActionLoading(true);
    try {
      const response = await fetch(`/api/trades/${tradeId}/${action}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        toast.success(`Trade ${action} successful!`);
        loadTrades(); // Refresh trades
      } else {
        throw new Error(`Failed to ${action} trade`);
      }
    } catch (error) {
      console.error(`Error ${action}ing trade:`, error);
      toast.error(`Failed to ${action} trade. Please try again.`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleTradeClick = (trade) => {
    setSelectedTrade(selectedTrade?.id === trade.id ? null : trade);
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
                <th className="table-header-cell">Actions</th>
              </tr>
            </thead>
            <tbody className="table-body">
              {trades.length === 0 ? (
                <tr>
                  <td colSpan="12" className="table-cell text-center py-8 text-gray-500">
                    No {activeTab} trades found
                  </td>
                </tr>
              ) : (
                trades.map((trade) => (
                  <tr 
                    key={trade.id} 
                    className={`table-row cursor-pointer hover:bg-gray-50 transition-colors ${
                      selectedTrade?.id === trade.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                    }`}
                    onClick={() => handleTradeClick(trade)}
                  >
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
                    <td className="table-cell">
                      <div className="flex items-center space-x-2">
                        <button
                          className="p-1 text-blue-600 hover:text-blue-800 transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleTradeClick(trade);
                          }}
                          title="View Details"
                        >
                          <EyeIcon className="h-4 w-4" />
                        </button>
                        {trade.status === 'open' && (
                          <>
                            <button
                              className="p-1 text-red-600 hover:text-red-800 transition-colors"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleTradeAction(trade.id, 'close');
                              }}
                              disabled={actionLoading}
                              title="Close Trade"
                            >
                              <XMarkIcon className="h-4 w-4" />
                            </button>
                            <button
                              className="p-1 text-green-600 hover:text-green-800 transition-colors"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleTradeAction(trade.id, 'modify');
                              }}
                              disabled={actionLoading}
                              title="Modify Trade"
                            >
                              <ArrowPathIcon className="h-4 w-4" />
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Trade Details Panel */}
      {selectedTrade && (
        <div className="card mt-6">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <h3 className="card-title">Trade Details</h3>
              <button
                onClick={() => setSelectedTrade(null)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div>
                <h4 className="text-sm font-medium text-gray-500 mb-2">Basic Info</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Symbol:</span>
                    <span className="font-medium">{selectedTrade.symbol}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Side:</span>
                    <span className="font-medium capitalize">{selectedTrade.side}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Size:</span>
                    <span className="font-medium">{selectedTrade.lot_size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status:</span>
                    <span>{getStatusBadge(selectedTrade.status)}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-500 mb-2">Prices</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Entry:</span>
                    <span className="font-medium">{selectedTrade.entry_price?.toFixed(5) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Current:</span>
                    <span className="font-medium">{selectedTrade.exit_price?.toFixed(5) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Stop Loss:</span>
                    <span className="font-medium">{selectedTrade.stop_loss?.toFixed(5) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Take Profit:</span>
                    <span className="font-medium">{selectedTrade.take_profit?.toFixed(5) || 'N/A'}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-500 mb-2">Performance</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">P&L:</span>
                    <span className={`font-medium ${getPnLColor(selectedTrade.pnl)}`}>
                      {selectedTrade.pnl ? formatCurrency(selectedTrade.pnl) : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Confidence:</span>
                    <span className="font-medium">
                      {selectedTrade.confidence ? (selectedTrade.confidence * 100).toFixed(0) : '0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Entry Time:</span>
                    <span className="font-medium text-sm">{formatDate(selectedTrade.entry_time)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Trades;
