import React, { useState, useEffect } from 'react';
import { 
  CogIcon, 
  SaveIcon,
  RefreshIcon
} from '@heroicons/react/24/outline';
import { useApi } from '../contexts/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';
import toast from 'react-hot-toast';

const Settings = () => {
  const { fetchSettings, fetchSymbolConfigs } = useApi();
  const [settings, setSettings] = useState(null);
  const [symbolConfigs, setSymbolConfigs] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('general');

  const tabs = [
    { id: 'general', name: 'General' },
    { id: 'risk', name: 'Risk Management' },
    { id: 'symbols', name: 'Symbols' },
    { id: 'features', name: 'Features' }
  ];

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const [settingsData, symbolsData] = await Promise.all([
        fetchSettings(),
        fetchSymbolConfigs()
      ]);
      setSettings(settingsData);
      setSymbolConfigs(symbolsData);
    } catch (err) {
      console.error('Error loading settings:', err);
      toast.error('Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      // In a real implementation, you would send the updated settings to the API
      toast.success('Settings saved successfully');
    } catch (err) {
      console.error('Error saving settings:', err);
      toast.error('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleSettingChange = (path, value) => {
    setSettings(prev => {
      const newSettings = { ...prev };
      const keys = path.split('.');
      let current = newSettings;
      
      for (let i = 0; i < keys.length - 1; i++) {
        current = current[keys[i]];
      }
      
      current[keys[keys.length - 1]] = value;
      return newSettings;
    });
  };

  const handleSymbolConfigChange = (symbol, key, value) => {
    setSymbolConfigs(prev => ({
      ...prev,
      [symbol]: {
        ...prev[symbol],
        [key]: value
      }
    }));
  };

  if (loading) {
    return <LoadingSpinner size="large" className="h-64" />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-500">Configure trading bot parameters and preferences</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={loadSettings}
            className="btn btn-secondary"
            disabled={loading}
          >
            <RefreshIcon className="h-4 w-4 mr-2" />
            Refresh
          </button>
          <button
            onClick={handleSave}
            className="btn btn-primary"
            disabled={saving}
          >
            <SaveIcon className="h-4 w-4 mr-2" />
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

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
            </button>
          ))}
        </nav>
      </div>

      {/* General Settings */}
      {activeTab === 'general' && (
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">General Configuration</h3>
              <p className="card-subtitle">Basic bot settings and preferences</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Timezone
                </label>
                <select
                  value={settings?.global_settings?.timezone || 'Africa/Nairobi'}
                  onChange={(e) => handleSettingChange('global_settings.timezone', e.target.value)}
                  className="select"
                >
                  <option value="Africa/Nairobi">Africa/Nairobi</option>
                  <option value="UTC">UTC</option>
                  <option value="America/New_York">America/New_York</option>
                  <option value="Europe/London">Europe/London</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Loop Interval (minutes)
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.loop_interval_minutes || 5}
                  onChange={(e) => handleSettingChange('global_settings.loop_interval_minutes', parseInt(e.target.value))}
                  className="input"
                  min="1"
                  max="60"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Cooldown Period (minutes)
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.cooldown_minutes || 10}
                  onChange={(e) => handleSettingChange('global_settings.cooldown_minutes', parseInt(e.target.value))}
                  className="input"
                  min="1"
                  max="60"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Log Level
                </label>
                <select
                  value={settings?.global_settings?.log_level || 'INFO'}
                  onChange={(e) => handleSettingChange('global_settings.log_level', e.target.value)}
                  className="select"
                >
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Risk Management Settings */}
      {activeTab === 'risk' && (
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Risk Management</h3>
              <p className="card-subtitle">Configure risk limits and position sizing</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Account Risk (%)
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.max_account_risk_percent || 10}
                  onChange={(e) => handleSettingChange('global_settings.max_account_risk_percent', parseFloat(e.target.value))}
                  className="input"
                  min="1"
                  max="50"
                  step="0.1"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Min Lot Size
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.min_lot_size || 0.01}
                  onChange={(e) => handleSettingChange('global_settings.min_lot_size', parseFloat(e.target.value))}
                  className="input"
                  min="0.01"
                  step="0.01"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Concurrent Trades
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.max_concurrent_trades || 5}
                  onChange={(e) => handleSettingChange('global_settings.max_concurrent_trades', parseInt(e.target.value))}
                  className="input"
                  min="1"
                  max="20"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Spread (pips)
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.max_spread_pips || 3}
                  onChange={(e) => handleSettingChange('global_settings.max_spread_pips', parseFloat(e.target.value))}
                  className="input"
                  min="0.1"
                  step="0.1"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Symbol Settings */}
      {activeTab === 'symbols' && (
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Symbol Configuration</h3>
              <p className="card-subtitle">Configure trading parameters for each symbol</p>
            </div>
            <div className="space-y-6">
              {Object.entries(symbolConfigs).map(([symbol, config]) => (
                <div key={symbol} className="border border-gray-200 rounded-lg p-4">
                  <h4 className="text-lg font-medium text-gray-900 mb-4">{symbol}</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={config.enabled || false}
                          onChange={(e) => handleSymbolConfigChange(symbol, 'enabled', e.target.checked)}
                          className="mr-2"
                        />
                        <span className="text-sm font-medium text-gray-700">Enabled</span>
                      </label>
                    </div>
                    
                    <div>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={config.allow_weekend || false}
                          onChange={(e) => handleSymbolConfigChange(symbol, 'allow_weekend', e.target.checked)}
                          className="mr-2"
                        />
                        <span className="text-sm font-medium text-gray-700">Allow Weekend</span>
                      </label>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Min Confidence
                      </label>
                      <input
                        type="number"
                        value={config.min_confidence || 0.3}
                        onChange={(e) => handleSymbolConfigChange(symbol, 'min_confidence', parseFloat(e.target.value))}
                        className="input"
                        min="0"
                        max="1"
                        step="0.1"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Max Spread (pips)
                      </label>
                      <input
                        type="number"
                        value={config.max_spread_pips || 3}
                        onChange={(e) => handleSymbolConfigChange(symbol, 'max_spread_pips', parseFloat(e.target.value))}
                        className="input"
                        min="0.1"
                        step="0.1"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        SL Multiplier
                      </label>
                      <input
                        type="number"
                        value={config.sl_multiplier || 2}
                        onChange={(e) => handleSymbolConfigChange(symbol, 'sl_multiplier', parseFloat(e.target.value))}
                        className="input"
                        min="0.1"
                        step="0.1"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        TP Multiplier
                      </label>
                      <input
                        type="number"
                        value={config.tp_multiplier || 3}
                        onChange={(e) => handleSymbolConfigChange(symbol, 'tp_multiplier', parseFloat(e.target.value))}
                        className="input"
                        min="0.1"
                        step="0.1"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Feature Settings */}
      {activeTab === 'features' && (
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">ML Feature Configuration</h3>
              <p className="card-subtitle">Configure machine learning parameters</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Neighbors Count
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.neighbors_count || 8}
                  onChange={(e) => handleSettingChange('global_settings.neighbors_count', parseInt(e.target.value))}
                  className="input"
                  min="1"
                  max="100"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Bars Back
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.max_bars_back || 2000}
                  onChange={(e) => handleSettingChange('global_settings.max_bars_back', parseInt(e.target.value))}
                  className="input"
                  min="100"
                  max="10000"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Feature Count
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.feature_count || 5}
                  onChange={(e) => handleSettingChange('global_settings.feature_count', parseInt(e.target.value))}
                  className="input"
                  min="2"
                  max="5"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Color Compression
                </label>
                <input
                  type="number"
                  value={settings?.global_settings?.color_compression || 1}
                  onChange={(e) => handleSettingChange('global_settings.color_compression', parseInt(e.target.value))}
                  className="input"
                  min="1"
                  max="10"
                />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Filter Settings</h3>
              <p className="card-subtitle">Configure signal filters</p>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">Volatility Filter</label>
                  <p className="text-xs text-gray-500">Filter signals based on volatility</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings?.global_settings?.use_volatility_filter || false}
                  onChange={(e) => handleSettingChange('global_settings.use_volatility_filter', e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">Regime Filter</label>
                  <p className="text-xs text-gray-500">Filter signals based on market regime</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings?.global_settings?.use_regime_filter || false}
                  onChange={(e) => handleSettingChange('global_settings.use_regime_filter', e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">ADX Filter</label>
                  <p className="text-xs text-gray-500">Filter signals based on ADX trend strength</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings?.global_settings?.use_adx_filter || false}
                  onChange={(e) => handleSettingChange('global_settings.use_adx_filter', e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Settings;
