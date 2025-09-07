import React, { useState, useEffect } from 'react';
import {
  CogIcon,
  ArrowPathIcon,
  ArrowDownTrayIcon,
  ShieldCheckIcon,
  CpuChipIcon,
  ChartBarIcon,
  BellIcon,
  GlobeAltIcon,
  ClockIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';
import toast from 'react-hot-toast';

const Settings = () => {
  const { getSettings, updateSettings } = useApi();
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Default settings structure
  const defaultSettings = {
    timezone: 'UTC',
    loop_interval: 60,
    log_level: 'INFO',
    cooldown_period: 15,
    max_risk_percent: 5.0,
    max_concurrent_trades: 5,
    min_lot_size: 0.01,
    max_lot_size: 1.0,
    max_spread_pips: 3,
    neighbors_count: 8,
    max_bars_back: 100,
    feature_count: 5,
    color_compression: 0.1,
    min_confidence_threshold: 0.6,
    min_volatility: 0.5,
    min_adx: 25,
    allow_weekend_trading: false,
    enable_news_filter: false,
    symbols: {
      EURUSD: { min_confidence: 0.7, enabled: true },
      GBPUSD: { min_confidence: 0.7, enabled: true },
      BTCUSD: { min_confidence: 0.6, enabled: true },
      USDJPY: { min_confidence: 0.7, enabled: true },
      AUDUSD: { min_confidence: 0.7, enabled: true }
    },
    notifications: {
      email: false,
      telegram: false,
      dashboard: true
    },
    telegram_bot_token: '',
    telegram_chat_id: ''
  };

  const loadSettings = async () => {
    setLoading(true);
    try {
      const data = await getSettings();
      // Merge with defaults to ensure all fields exist
      setSettings({ ...defaultSettings, ...data });
      setHasChanges(false);
    } catch (error) {
      console.error('Error loading settings:', error);
      // Use defaults if API fails
      setSettings(defaultSettings);
      toast.error('Using default settings - API unavailable');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateSettings(settings);
      setHasChanges(false);
      toast.success('Settings saved successfully');
    } catch (error) {
      console.error('Error saving settings:', error);
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleSettingChange = (path, value) => {
    const newSettings = { ...settings };
    const keys = path.split('.');
    let current = newSettings;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
    setSettings(newSettings);
    setHasChanges(true);
  };

  const resetToDefaults = () => {
    setSettings(defaultSettings);
    setHasChanges(true);
    toast.success('Settings reset to defaults');
  };

  useEffect(() => {
    loadSettings();
  }, []);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!settings) {
    return (
      <div className="p-6">
        <p className="text-red-500">Failed to load settings.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient">Algorithm Settings</h1>
          <p className="text-text-muted mt-2">Configure all trading algorithm parameters</p>
          {settings && (
            <div className="mt-2 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${hasChanges ? 'bg-warning-400' : 'bg-success-400'}`}></div>
              <span className="text-sm text-text-muted">
                {hasChanges ? 'Unsaved changes' : 'All changes saved'}
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={resetToDefaults}
            className="btn btn-warning"
            disabled={loading || saving}
          >
            <ExclamationTriangleIcon className="h-4 w-4" />
            Reset to Defaults
          </button>
          <button
            onClick={loadSettings}
            className="btn btn-secondary"
            disabled={loading || saving}
          >
            <ArrowPathIcon className="h-4 w-4" />
            Refresh
          </button>
          <button
            onClick={handleSave}
            className={`btn ${hasChanges ? 'btn-primary' : 'btn-secondary'}`}
            disabled={saving || !hasChanges}
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            {saving ? 'Saving...' : hasChanges ? 'Save Changes' : 'No Changes'}
          </button>
        </div>
      </div>

      {/* Settings Tabs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* General Settings */}
        <div className="card slide-in-right">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <GlobeAltIcon className="h-6 w-6 text-primary-400" />
              <h3 className="card-title">General Settings</h3>
            </div>
            <p className="card-subtitle">Basic algorithm configuration</p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Timezone</label>
              <select
                value={settings?.timezone || 'UTC'}
                onChange={(e) => handleSettingChange('timezone', e.target.value)}
                className="select"
              >
                <option value="UTC">UTC</option>
                <option value="America/New_York">New York (EST)</option>
                <option value="Europe/London">London (GMT)</option>
                <option value="Asia/Tokyo">Tokyo (JST)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Loop Interval (seconds)</label>
              <input
                type="number"
                value={settings?.loop_interval || 60}
                onChange={(e) => setSettings({ ...settings, loop_interval: parseInt(e.target.value) })}
                className="select"
                min="1"
                max="3600"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Log Level</label>
              <select
                value={settings?.log_level || 'INFO'}
                onChange={(e) => setSettings({ ...settings, log_level: e.target.value })}
                className="select"
              >
                <option value="DEBUG">DEBUG</option>
                <option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Cooldown Period (minutes)</label>
              <input
                type="number"
                value={settings?.cooldown_period || 15}
                onChange={(e) => setSettings({ ...settings, cooldown_period: parseInt(e.target.value) })}
                className="select"
                min="1"
                max="1440"
              />
            </div>
          </div>
        </div>

        {/* Risk Management */}
        <div className="card slide-in-right" style={{ animationDelay: '0.1s' }}>
          <div className="card-header">
            <div className="flex items-center gap-3">
              <ShieldCheckIcon className="h-6 w-6 text-success-400" />
              <h3 className="card-title">Risk Management</h3>
            </div>
            <p className="card-subtitle">Risk and position limits</p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Max Account Risk %</label>
              <input
                type="number"
                value={settings?.max_risk_percent || 5}
                onChange={(e) => setSettings({ ...settings, max_risk_percent: parseFloat(e.target.value) })}
                className="select"
                min="0.1"
                max="20"
                step="0.1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Max Concurrent Trades</label>
              <input
                type="number"
                value={settings?.max_concurrent_trades || 5}
                onChange={(e) => setSettings({ ...settings, max_concurrent_trades: parseInt(e.target.value) })}
                className="select"
                min="1"
                max="20"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Min Lot Size</label>
              <input
                type="number"
                value={settings?.min_lot_size || 0.01}
                onChange={(e) => setSettings({ ...settings, min_lot_size: parseFloat(e.target.value) })}
                className="select"
                min="0.01"
                max="1"
                step="0.01"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Max Lot Size</label>
              <input
                type="number"
                value={settings?.max_lot_size || 1.0}
                onChange={(e) => setSettings({ ...settings, max_lot_size: parseFloat(e.target.value) })}
                className="select"
                min="0.01"
                max="10"
                step="0.01"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Max Spread (pips)</label>
              <input
                type="number"
                value={settings?.max_spread_pips || 3}
                onChange={(e) => setSettings({ ...settings, max_spread_pips: parseInt(e.target.value) })}
                className="select"
                min="1"
                max="20"
              />
            </div>
          </div>
        </div>

        {/* ML/AI Settings */}
        <div className="card slide-in-right" style={{ animationDelay: '0.2s' }}>
          <div className="card-header">
            <div className="flex items-center gap-3">
              <CpuChipIcon className="h-6 w-6 text-purple-400" />
              <h3 className="card-title">ML/AI Configuration</h3>
            </div>
            <p className="card-subtitle">Machine learning parameters</p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Neighbors Count (k)</label>
              <input
                type="number"
                value={settings?.neighbors_count || 8}
                onChange={(e) => setSettings({ ...settings, neighbors_count: parseInt(e.target.value) })}
                className="select"
                min="3"
                max="20"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Max Bars Back</label>
              <input
                type="number"
                value={settings?.max_bars_back || 100}
                onChange={(e) => setSettings({ ...settings, max_bars_back: parseInt(e.target.value) })}
                className="select"
                min="50"
                max="500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Feature Count</label>
              <input
                type="number"
                value={settings?.feature_count || 5}
                onChange={(e) => setSettings({ ...settings, feature_count: parseInt(e.target.value) })}
                className="select"
                min="3"
                max="10"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Color Compression</label>
              <input
                type="number"
                value={settings?.color_compression || 0.1}
                onChange={(e) => setSettings({ ...settings, color_compression: parseFloat(e.target.value) })}
                className="select"
                min="0.01"
                max="1"
                step="0.01"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Min Confidence Threshold</label>
              <input
                type="number"
                value={settings?.min_confidence_threshold || 0.6}
                onChange={(e) => setSettings({ ...settings, min_confidence_threshold: parseFloat(e.target.value) })}
                className="select"
                min="0.1"
                max="1"
                step="0.1"
              />
            </div>
          </div>
        </div>

        {/* Signal Filters */}
        <div className="card slide-in-right" style={{ animationDelay: '0.3s' }}>
          <div className="card-header">
            <div className="flex items-center gap-3">
              <ChartBarIcon className="h-6 w-6 text-warning-400" />
              <h3 className="card-title">Signal Filters</h3>
            </div>
            <p className="card-subtitle">Trading signal filters</p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Min Volatility</label>
              <input
                type="number"
                value={settings?.min_volatility || 0.5}
                onChange={(e) => setSettings({ ...settings, min_volatility: parseFloat(e.target.value) })}
                className="select"
                min="0.1"
                max="5"
                step="0.1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Min ADX</label>
              <input
                type="number"
                value={settings?.min_adx || 25}
                onChange={(e) => setSettings({ ...settings, min_adx: parseInt(e.target.value) })}
                className="select"
                min="10"
                max="100"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-text-secondary">Enable Weekend Trading</label>
              <input
                type="checkbox"
                checked={settings?.allow_weekend_trading || false}
                onChange={(e) => setSettings({ ...settings, allow_weekend_trading: e.target.checked })}
                className="w-4 h-4 text-primary-600 bg-dark-700 border-dark-600 rounded focus:ring-primary-500"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-text-secondary">Enable News Filter</label>
              <input
                type="checkbox"
                checked={settings?.enable_news_filter || false}
                onChange={(e) => setSettings({ ...settings, enable_news_filter: e.target.checked })}
                className="w-4 h-4 text-primary-600 bg-dark-700 border-dark-600 rounded focus:ring-primary-500"
              />
            </div>
          </div>
        </div>

        {/* Symbol-Specific Settings */}
        <div className="card slide-in-right" style={{ animationDelay: '0.4s' }}>
          <div className="card-header">
            <div className="flex items-center gap-3">
              <ExclamationTriangleIcon className="h-6 w-6 text-danger-400" />
              <h3 className="card-title">Symbol Configuration</h3>
            </div>
            <p className="card-subtitle">Per-symbol trading settings</p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">EURUSD Min Confidence</label>
              <input
                type="number"
                value={settings?.symbols?.EURUSD?.min_confidence || 0.7}
                onChange={(e) => setSettings({ 
                  ...settings, 
                  symbols: { 
                    ...settings.symbols, 
                    EURUSD: { ...settings.symbols?.EURUSD, min_confidence: parseFloat(e.target.value) }
                  }
                })}
                className="select"
                min="0.1"
                max="1"
                step="0.1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">GBPUSD Min Confidence</label>
              <input
                type="number"
                value={settings?.symbols?.GBPUSD?.min_confidence || 0.7}
                onChange={(e) => setSettings({ 
                  ...settings, 
                  symbols: { 
                    ...settings.symbols, 
                    GBPUSD: { ...settings.symbols?.GBPUSD, min_confidence: parseFloat(e.target.value) }
                  }
                })}
                className="select"
                min="0.1"
                max="1"
                step="0.1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">BTCUSD Min Confidence</label>
              <input
                type="number"
                value={settings?.symbols?.BTCUSD?.min_confidence || 0.6}
                onChange={(e) => handleSettingChange('symbols.BTCUSD.min_confidence', parseFloat(e.target.value))}
                className="select"
                min="0.1"
                max="1"
                step="0.1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">USDJPY Min Confidence</label>
              <input
                type="number"
                value={settings?.symbols?.USDJPY?.min_confidence || 0.7}
                onChange={(e) => handleSettingChange('symbols.USDJPY.min_confidence', parseFloat(e.target.value))}
                className="select"
                min="0.1"
                max="1"
                step="0.1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">AUDUSD Min Confidence</label>
              <input
                type="number"
                value={settings?.symbols?.AUDUSD?.min_confidence || 0.7}
                onChange={(e) => handleSettingChange('symbols.AUDUSD.min_confidence', parseFloat(e.target.value))}
                className="select"
                min="0.1"
                max="1"
                step="0.1"
              />
            </div>
            
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-text-secondary">Symbol Status</h4>
              {Object.entries(settings?.symbols || {}).map(([symbol, config]) => (
                <div key={symbol} className="flex items-center justify-between">
                  <label className="text-sm text-text-muted">{symbol}</label>
                  <input
                    type="checkbox"
                    checked={config?.enabled || false}
                    onChange={(e) => handleSettingChange(`symbols.${symbol}.enabled`, e.target.checked)}
                    className="w-4 h-4 text-primary-600 bg-dark-700 border-dark-600 rounded focus:ring-primary-500"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Alerts & Notifications */}
        <div className="card slide-in-right" style={{ animationDelay: '0.5s' }}>
          <div className="card-header">
            <div className="flex items-center gap-3">
              <BellIcon className="h-6 w-6 text-primary-400" />
              <h3 className="card-title">Alerts & Notifications</h3>
            </div>
            <p className="card-subtitle">Notification preferences</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-text-secondary">Email Notifications</label>
              <input
                type="checkbox"
                checked={settings?.notifications?.email || false}
                onChange={(e) => setSettings({ 
                  ...settings, 
                  notifications: { ...settings.notifications, email: e.target.checked }
                })}
                className="w-4 h-4 text-primary-600 bg-dark-700 border-dark-600 rounded focus:ring-primary-500"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-text-secondary">Telegram Notifications</label>
              <input
                type="checkbox"
                checked={settings?.notifications?.telegram || false}
                onChange={(e) => setSettings({ 
                  ...settings, 
                  notifications: { ...settings.notifications, telegram: e.target.checked }
                })}
                className="w-4 h-4 text-primary-600 bg-dark-700 border-dark-600 rounded focus:ring-primary-500"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-text-secondary">Dashboard Popups</label>
              <input
                type="checkbox"
                checked={settings?.notifications?.dashboard || true}
                onChange={(e) => setSettings({ 
                  ...settings, 
                  notifications: { ...settings.notifications, dashboard: e.target.checked }
                })}
                className="w-4 h-4 text-primary-600 bg-dark-700 border-dark-600 rounded focus:ring-primary-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Telegram Bot Token</label>
              <input
                type="password"
                value={settings?.telegram_bot_token || ''}
                onChange={(e) => setSettings({ ...settings, telegram_bot_token: e.target.value })}
                className="select"
                placeholder="Enter Telegram bot token"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Telegram Chat ID</label>
              <input
                type="text"
                value={settings?.telegram_chat_id || ''}
                onChange={(e) => setSettings({ ...settings, telegram_chat_id: e.target.value })}
                className="select"
                placeholder="Enter Telegram chat ID"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
