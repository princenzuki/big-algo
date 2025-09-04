"""
Global Settings Configuration

Defines global settings for the trading bot including timezone, risk caps,
overlay flags, and other system-wide parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
from pathlib import Path

@dataclass
class GlobalSettings:
    """Global system settings"""
    # Timezone and scheduling
    timezone: str = "Africa/Nairobi"
    loop_interval_minutes: int = 5
    cooldown_minutes: int = 10
    
    # Risk management
    max_account_risk_percent: float = 10.0
    min_lot_size: float = 0.01
    max_concurrent_trades: int = 5
    max_spread_pips: float = 3.0
    min_stop_distance_pips: float = 5.0
    
    # Pine Script settings (EXACT TradingView settings)
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    show_exits: bool = False  # "Show Default Exits" - NOT ticked
    use_dynamic_exits: bool = True  # "Use Dynamic Exits" - IS ticked
    
    # Feature parameters (EXACT TradingView settings)
    feature_params: Dict[str, Dict[str, any]] = None
    
    # Filter settings (EXACT TradingView settings)
    use_volatility_filter: bool = True  # "Use Volatility Filter" - IS ticked
    use_regime_filter: bool = True  # "Use Regime Filter" - IS ticked
    use_adx_filter: bool = False  # "Use ADX Filter" - NOT ticked
    regime_threshold: float = -0.1  # "Regime Filter Threshold" - IS ticked, value -0.1
    adx_threshold: int = 20  # Not used since ADX filter is off
    
    # EMA/SMA filters (EXACT TradingView settings)
    use_ema_filter: bool = False  # "Use EMA Filter" - NOT ticked
    ema_period: int = 200  # Not used since EMA filter is off
    use_sma_filter: bool = False  # "Use SMA Filter" - NOT ticked
    sma_period: int = 200  # Not used since SMA filter is off
    
    # Kernel regression settings (EXACT TradingView settings)
    use_kernel_filter: bool = True  # "Trade with Kernel" - IS ticked
    show_kernel_estimate: bool = True  # "Show Kernel Estimate" - IS ticked
    use_kernel_smoothing: bool = False  # "Enhance Kernel Smoothing" - NOT ticked
    kernel_lookback_window: int = 8  # "Lookback Window" - value 8
    kernel_relative_weighting: float = 8.0  # "Relative Weighting" - value 8
    kernel_regression_level: int = 25  # "Regression Level" - value 25
    kernel_lag: int = 2  # "Lag" - value 2 (not used since smoothing is off)
    
    # Display settings
    show_bar_colors: bool = True
    show_bar_predictions: bool = True
    use_atr_offset: bool = False
    bar_predictions_offset: float = 0.0
    
    # Trading settings (EXACT TradingView settings)
    show_trade_stats: bool = False  # "Show Trade Stats" - NOT ticked
    use_worst_case: bool = False  # "Use Worst Case Estimates" - NOT ticked
    
    # Broker settings
    broker_login: Optional[int] = None
    broker_password: Optional[str] = None
    broker_server: Optional[str] = None
    
    # Database settings
    database_path: str = "trading_portfolio.db"
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"
    
    def __post_init__(self):
        """Initialize default feature parameters if not provided"""
        if self.feature_params is None:
            # EXACT TradingView Feature Engineering settings
            self.feature_params = {
                'f1': {'string': 'RSI', 'param_a': 14, 'param_b': 1},  # Feature 1: RSI, A=14, B=1
                'f2': {'string': 'WT', 'param_a': 10, 'param_b': 11},  # Feature 2: WT, A=10, B=11
                'f3': {'string': 'CCI', 'param_a': 20, 'param_b': 1},  # Feature 3: CCI, A=20, B=1
                'f4': {'string': 'ADX', 'param_a': 20, 'param_b': 2},  # Feature 4: ADX, A=20, B=2
                'f5': {'string': 'RSI', 'param_a': 9, 'param_b': 1}    # Feature 5: RSI, A=9, B=1
            }

class SettingsManager:
    """
    Manages global settings and per-symbol configurations
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.global_settings = GlobalSettings()
        self.symbol_configs = {}
        
        self._load_settings()
        self._load_symbol_configs()
    
    def _load_settings(self):
        """Load global settings from file or environment"""
        settings_file = self.config_dir / "settings.yaml"
        
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings_data = yaml.safe_load(f)
                if settings_data:
                    self._update_settings_from_dict(settings_data)
        
        # Override with environment variables
        self._load_from_environment()
    
    def _update_settings_from_dict(self, data: Dict):
        """Update settings from dictionary"""
        for key, value in data.items():
            if hasattr(self.global_settings, key):
                setattr(self.global_settings, key, value)
    
    def _load_from_environment(self):
        """Load settings from environment variables"""
        env_mappings = {
            'TRADING_TIMEZONE': 'timezone',
            'LOOP_INTERVAL_MINUTES': 'loop_interval_minutes',
            'COOLDOWN_MINUTES': 'cooldown_minutes',
            'MAX_ACCOUNT_RISK_PERCENT': 'max_account_risk_percent',
            'MIN_LOT_SIZE': 'min_lot_size',
            'MAX_CONCURRENT_TRADES': 'max_concurrent_trades',
            'MAX_SPREAD_PIPS': 'max_spread_pips',
            'MIN_STOP_DISTANCE_PIPS': 'min_stop_distance_pips',
            'NEIGHBORS_COUNT': 'neighbors_count',
            'MAX_BARS_BACK': 'max_bars_back',
            'FEATURE_COUNT': 'feature_count',
            'BROKER_LOGIN': 'broker_login',
            'BROKER_PASSWORD': 'broker_password',
            'BROKER_SERVER': 'broker_server',
            'DATABASE_PATH': 'database_path',
            'LOG_LEVEL': 'log_level',
            'LOG_FILE': 'log_file'
        }
        
        for env_var, setting_name in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if setting_name in ['loop_interval_minutes', 'cooldown_minutes', 'max_concurrent_trades', 
                                  'neighbors_count', 'max_bars_back', 'feature_count', 'color_compression',
                                  'adx_threshold', 'ema_period', 'sma_period', 'kernel_lookback_window',
                                  'kernel_regression_level', 'kernel_lag', 'broker_login']:
                    value = int(value)
                elif setting_name in ['max_account_risk_percent', 'min_lot_size', 'max_spread_pips',
                                    'min_stop_distance_pips', 'regime_threshold', 'kernel_relative_weighting',
                                    'bar_predictions_offset']:
                    value = float(value)
                elif setting_name in ['show_exits', 'use_dynamic_exits', 'use_volatility_filter',
                                    'use_regime_filter', 'use_adx_filter', 'use_ema_filter', 'use_sma_filter',
                                    'use_kernel_filter', 'show_kernel_estimate', 'use_kernel_smoothing',
                                    'show_bar_colors', 'show_bar_predictions', 'use_atr_offset',
                                    'show_trade_stats', 'use_worst_case']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                setattr(self.global_settings, setting_name, value)
    
    def _load_symbol_configs(self):
        """Load per-symbol configurations"""
        symbols_file = self.config_dir / "symbols.yaml"
        
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                self.symbol_configs = yaml.safe_load(f) or {}
        else:
            # Create default symbol configs
            self._create_default_symbol_configs()
    
    def _create_default_symbol_configs(self):
        """Create default symbol configurations"""
        default_symbols = [
            'BTCUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
            'USDCAD', 'NZDUSD', 'USDCHF'
        ]
        
        for symbol in default_symbols:
            self.symbol_configs[symbol] = {
                'enabled': True,
                'allow_weekend': symbol == 'BTCUSD',
                'min_confidence': 0.3,
                'max_spread_pips': 3.0,
                'atr_period': 14,
                'sl_multiplier': 2.0,
                'tp_multiplier': 3.0,
                'sessions': {
                    'london': True,
                    'new_york': True,
                    'asia': True
                }
            }
        
        self.save_symbol_configs()
    
    def get_symbol_config(self, symbol: str) -> Dict:
        """Get configuration for a specific symbol"""
        return self.symbol_configs.get(symbol, {})
    
    def update_symbol_config(self, symbol: str, config: Dict):
        """Update configuration for a specific symbol"""
        self.symbol_configs[symbol] = config
        self.save_symbol_configs()
    
    def get_enabled_symbols(self) -> List[str]:
        """Get list of enabled symbols"""
        return [
            symbol for symbol, config in self.symbol_configs.items()
            if config.get('enabled', True)
        ]
    
    def save_settings(self):
        """Save global settings to file"""
        settings_file = self.config_dir / "settings.yaml"
        
        settings_data = {
            'timezone': self.global_settings.timezone,
            'loop_interval_minutes': self.global_settings.loop_interval_minutes,
            'cooldown_minutes': self.global_settings.cooldown_minutes,
            'max_account_risk_percent': self.global_settings.max_account_risk_percent,
            'min_lot_size': self.global_settings.min_lot_size,
            'max_concurrent_trades': self.global_settings.max_concurrent_trades,
            'max_spread_pips': self.global_settings.max_spread_pips,
            'min_stop_distance_pips': self.global_settings.min_stop_distance_pips,
            'neighbors_count': self.global_settings.neighbors_count,
            'max_bars_back': self.global_settings.max_bars_back,
            'feature_count': self.global_settings.feature_count,
            'color_compression': self.global_settings.color_compression,
            'show_exits': self.global_settings.show_exits,
            'use_dynamic_exits': self.global_settings.use_dynamic_exits,
            'feature_params': self.global_settings.feature_params,
            'use_volatility_filter': self.global_settings.use_volatility_filter,
            'use_regime_filter': self.global_settings.use_regime_filter,
            'use_adx_filter': self.global_settings.use_adx_filter,
            'regime_threshold': self.global_settings.regime_threshold,
            'adx_threshold': self.global_settings.adx_threshold,
            'use_ema_filter': self.global_settings.use_ema_filter,
            'ema_period': self.global_settings.ema_period,
            'use_sma_filter': self.global_settings.use_sma_filter,
            'sma_period': self.global_settings.sma_period,
            'use_kernel_filter': self.global_settings.use_kernel_filter,
            'show_kernel_estimate': self.global_settings.show_kernel_estimate,
            'use_kernel_smoothing': self.global_settings.use_kernel_smoothing,
            'kernel_lookback_window': self.global_settings.kernel_lookback_window,
            'kernel_relative_weighting': self.global_settings.kernel_relative_weighting,
            'kernel_regression_level': self.global_settings.kernel_regression_level,
            'kernel_lag': self.global_settings.kernel_lag,
            'show_bar_colors': self.global_settings.show_bar_colors,
            'show_bar_predictions': self.global_settings.show_bar_predictions,
            'use_atr_offset': self.global_settings.use_atr_offset,
            'bar_predictions_offset': self.global_settings.bar_predictions_offset,
            'show_trade_stats': self.global_settings.show_trade_stats,
            'use_worst_case': self.global_settings.use_worst_case,
            'database_path': self.global_settings.database_path,
            'log_level': self.global_settings.log_level,
            'log_file': self.global_settings.log_file
        }
        
        with open(settings_file, 'w') as f:
            yaml.dump(settings_data, f, default_flow_style=False, indent=2)
    
    def save_symbol_configs(self):
        """Save symbol configurations to file"""
        symbols_file = self.config_dir / "symbols.yaml"
        
        with open(symbols_file, 'w') as f:
            yaml.dump(self.symbol_configs, f, default_flow_style=False, indent=2)
    
    def get_all_settings(self) -> Dict:
        """Get all settings as dictionary"""
        return {
            'global': {
                'timezone': self.global_settings.timezone,
                'loop_interval_minutes': self.global_settings.loop_interval_minutes,
                'cooldown_minutes': self.global_settings.cooldown_minutes,
                'max_account_risk_percent': self.global_settings.max_account_risk_percent,
                'min_lot_size': self.global_settings.min_lot_size,
                'max_concurrent_trades': self.global_settings.max_concurrent_trades,
                'max_spread_pips': self.global_settings.max_spread_pips,
                'min_stop_distance_pips': self.global_settings.min_stop_distance_pips,
                'neighbors_count': self.global_settings.neighbors_count,
                'max_bars_back': self.global_settings.max_bars_back,
                'feature_count': self.global_settings.feature_count,
                'color_compression': self.global_settings.color_compression,
                'show_exits': self.global_settings.show_exits,
                'use_dynamic_exits': self.global_settings.use_dynamic_exits,
                'feature_params': self.global_settings.feature_params,
                'use_volatility_filter': self.global_settings.use_volatility_filter,
                'use_regime_filter': self.global_settings.use_regime_filter,
                'use_adx_filter': self.global_settings.use_adx_filter,
                'regime_threshold': self.global_settings.regime_threshold,
                'adx_threshold': self.global_settings.adx_threshold,
                'use_ema_filter': self.global_settings.use_ema_filter,
                'ema_period': self.global_settings.ema_period,
                'use_sma_filter': self.global_settings.use_sma_filter,
                'sma_period': self.global_settings.sma_period,
                'use_kernel_filter': self.global_settings.use_kernel_filter,
                'show_kernel_estimate': self.global_settings.show_kernel_estimate,
                'use_kernel_smoothing': self.global_settings.use_kernel_smoothing,
                'kernel_lookback_window': self.global_settings.kernel_lookback_window,
                'kernel_relative_weighting': self.global_settings.kernel_relative_weighting,
                'kernel_regression_level': self.global_settings.kernel_regression_level,
                'kernel_lag': self.global_settings.kernel_lag,
                'show_bar_colors': self.global_settings.show_bar_colors,
                'show_bar_predictions': self.global_settings.show_bar_predictions,
                'use_atr_offset': self.global_settings.use_atr_offset,
                'bar_predictions_offset': self.global_settings.bar_predictions_offset,
                'show_trade_stats': self.global_settings.show_trade_stats,
                'use_worst_case': self.global_settings.use_worst_case,
                'database_path': self.global_settings.database_path,
                'log_level': self.global_settings.log_level,
                'log_file': self.global_settings.log_file
            },
            'symbols': self.symbol_configs
        }

# Global settings instance
settings_manager = SettingsManager()
