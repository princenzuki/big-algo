import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

const ApiContext = createContext();

export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

export const ApiProvider = ({ children }) => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // API base URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

  // Create axios instance
  const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 10000,
  });

  // Request interceptor
  api.interceptors.request.use(
    (config) => {
      // Add auth headers if needed
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor
  api.interceptors.response.use(
    (response) => {
      return response;
    },
    (error) => {
      const message = error.response?.data?.detail || error.message || 'An error occurred';
      toast.error(message);
      return Promise.reject(error);
    }
  );

  // Fetch dashboard data
  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await api.get('/dashboard');
      setDashboardData(response.data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch P&L data by period
  const fetchPnLData = async (period) => {
    try {
      const response = await api.get(`/pnl/${period}`);
      return response.data;
    } catch (err) {
      console.error(`Error fetching ${period} P&L data:`, err);
      throw err;
    }
  };

  // Fetch trades
  const fetchTrades = async (type = 'open', limit = 50) => {
    try {
      const response = await api.get(`/trades/${type}`, {
        params: { limit }
      });
      return response.data;
    } catch (err) {
      console.error(`Error fetching ${type} trades:`, err);
      throw err;
    }
  };

  // Fetch trades by symbol
  const fetchTradesBySymbol = async (symbol) => {
    try {
      const response = await api.get(`/trades/symbol/${symbol}`);
      return response.data;
    } catch (err) {
      console.error(`Error fetching trades for ${symbol}:`, err);
      throw err;
    }
  };

  // Fetch trade stats
  const fetchTradeStats = async () => {
    try {
      const response = await api.get('/trades/stats');
      return response.data;
    } catch (err) {
      console.error('Error fetching trade stats:', err);
      throw err;
    }
  };

  // Fetch risk summary
  const fetchRiskSummary = async () => {
    try {
      const response = await api.get('/risk/summary');
      return response.data;
    } catch (err) {
      console.error('Error fetching risk summary:', err);
      throw err;
    }
  };

  // Fetch risk metrics
  const fetchRiskMetrics = async () => {
    try {
      const response = await api.get('/risk/metrics');
      return response.data;
    } catch (err) {
      console.error('Error fetching risk metrics:', err);
      throw err;
    }
  };

  // Fetch session info
  const fetchSessionInfo = async () => {
    try {
      const response = await api.get('/session/info');
      return response.data;
    } catch (err) {
      console.error('Error fetching session info:', err);
      throw err;
    }
  };

  // Fetch symbol session status
  const fetchSymbolSessionStatus = async (symbol) => {
    try {
      const response = await api.get(`/session/symbol/${symbol}`);
      return response.data;
    } catch (err) {
      console.error(`Error fetching session status for ${symbol}:`, err);
      throw err;
    }
  };

  // Fetch algo health
  const fetchAlgoHealth = async () => {
    try {
      const response = await api.get('/health/algo');
      return response.data;
    } catch (err) {
      console.error('Error fetching algo health:', err);
      throw err;
    }
  };

  // Fetch system status
  const fetchSystemStatus = async () => {
    try {
      const response = await api.get('/health/system');
      return response.data;
    } catch (err) {
      console.error('Error fetching system status:', err);
      throw err;
    }
  };

  // Fetch settings
  const fetchSettings = async () => {
    try {
      const response = await api.get('/config/settings');
      return response.data;
    } catch (err) {
      console.error('Error fetching settings:', err);
      throw err;
    }
  };

  // Fetch symbol configs
  const fetchSymbolConfigs = async () => {
    try {
      const response = await api.get('/config/symbols');
      return response.data;
    } catch (err) {
      console.error('Error fetching symbol configs:', err);
      throw err;
    }
  };

  // Export trades
  const exportTrades = async (format = 'csv', options = {}) => {
    try {
      const response = await api.post('/export/trades', {
        format,
        ...options
      });
      return response.data;
    } catch (err) {
      console.error('Error exporting trades:', err);
      throw err;
    }
  };

  // Download export file
  const downloadExport = async (filename) => {
    try {
      const response = await api.get(`/export/download/${filename}`, {
        responseType: 'blob'
      });
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success('Export downloaded successfully');
    } catch (err) {
      console.error('Error downloading export:', err);
      throw err;
    }
  };

  // Auto-refresh dashboard data
  useEffect(() => {
    fetchDashboardData();
    
    // Set up auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const value = {
    // State
    dashboardData,
    loading,
    error,
    lastUpdate,
    
    // API methods
    api,
    fetchDashboardData,
    fetchPnLData,
    fetchTrades,
    fetchTradesBySymbol,
    fetchTradeStats,
    fetchRiskSummary,
    fetchRiskMetrics,
    fetchSessionInfo,
    fetchSymbolSessionStatus,
    fetchAlgoHealth,
    fetchSystemStatus,
    fetchSettings,
    fetchSymbolConfigs,
    exportTrades,
    downloadExport,
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
};
