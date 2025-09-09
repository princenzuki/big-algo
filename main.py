"""
Lorentzian Trading Bot - Main Entry Point

This is the main entry point for the Lorentzian Classification Trading Bot.
It orchestrates all components and runs the main trading loop.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import pandas as pd

from core.signals import LorentzianClassifier, Settings, FilterSettings
from core.risk import RiskManager, RiskSettings, AccountInfo, get_dynamic_spread, get_intelligent_sl, calculate_trailing_stop, calculate_hybrid_stop_loss
from core.sessions import SessionManager
from core.portfolio import PortfolioManager
from core.smart_tp import SmartTakeProfit, SmartTPConfig
from adapters.mt5_adapter import MT5Adapter
from config.settings import settings_manager
from mtf_validator import MultiTimeframeValidator

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings_manager.global_settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings_manager.global_settings.log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class LorentzianTradingBot:
    """
    Main trading bot class
    
    Orchestrates all components and runs the main trading loop.
    """
    
    def __init__(self):
        self.settings = settings_manager.global_settings
        self.symbol_configs = settings_manager.symbol_configs
        
        # Initialize components
        self.classifier = LorentzianClassifier(
            Settings(
                neighbors_count=self.settings.neighbors_count,
                max_bars_back=self.settings.max_bars_back,
                feature_count=self.settings.feature_count,
                color_compression=self.settings.color_compression,
                show_exits=self.settings.show_exits,
                use_dynamic_exits=self.settings.use_dynamic_exits
            ),
            FilterSettings(
                use_volatility_filter=self.settings.use_volatility_filter,
                use_regime_filter=self.settings.use_regime_filter,
                use_adx_filter=self.settings.use_adx_filter,
                regime_threshold=self.settings.regime_threshold,
                adx_threshold=self.settings.adx_threshold
            )
        )
        
        self.risk_manager = RiskManager(RiskSettings(
            max_account_risk_percent=self.settings.max_account_risk_percent,
            min_lot_size=self.settings.min_lot_size,
            max_concurrent_trades=self.settings.max_concurrent_trades,
            cooldown_minutes=self.settings.cooldown_minutes,
            max_spread_pips=self.settings.max_spread_pips,
            min_stop_distance_pips=self.settings.min_stop_distance_pips
        ))
        
        self.session_manager = SessionManager()
        self.portfolio_manager = PortfolioManager(self.settings.database_path)
        
        # Initialize broker adapter first
        self.broker_adapter = MT5Adapter(
            login=self.settings.broker_login,
            password=self.settings.broker_password,
            server=self.settings.broker_server
        )
        
        # Initialize Smart Take Profit system
        self.smart_tp = SmartTakeProfit(SmartTPConfig())
        
        # Initialize Multi-Timeframe Validator
        self.mtf_validator = MultiTimeframeValidator(self.broker_adapter)
        
        # Bot state
        self.running = False
        self.last_heartbeat = datetime.now()
        self.signals_processed = 0
        self.errors_count = 0
        
        # Historical data cache
        self.historical_data: Dict[str, List[Dict]] = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def initialize(self) -> bool:
        """
        Initialize the trading bot
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Lorentzian Trading Bot...")
            
            # Connect to broker
            if not self.broker_adapter.connect():
                logger.error("Failed to connect to broker")
                return False
            
            # Get account info
            account_info = self.broker_adapter.get_account_info()
            if not account_info:
                logger.error("Failed to get account information")
                return False
            
            self.risk_manager.update_account_info(account_info)
            logger.info(f"Connected to account: {account_info.balance:.2f} {account_info.currency}")
            
            # Load historical data for enabled symbols
            await self._load_historical_data()
            
            logger.info("Trading bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def _load_historical_data(self):
        """Load historical data for all enabled symbols"""
        enabled_symbols = settings_manager.get_enabled_symbols()
        
        for symbol in enabled_symbols:
            try:
                # Get historical rates (last 1000 bars) - using 1-minute timeframe to match Pine Script
                settings = settings_manager.get_all_settings()
                timeframe = settings['global']['trading_timeframe']
                rates = self.broker_adapter.get_rates(symbol, timeframe, 1000)
                
                if rates:
                    self.historical_data[symbol] = rates
                    logger.info(f"Loaded {len(rates)} historical bars for {symbol}")
                else:
                    logger.warning(f"No historical data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
    
    async def run(self):
        """Main trading loop"""
        if not await self.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        self.running = True
        logger.info("Starting main trading loop...")
        
        try:
            while self.running:
                await self._trading_cycle()
                await asyncio.sleep(self.settings.loop_interval_minutes * 60)
                
        except Exception as e:
            logger.error(f"Error in main trading loop: {e}")
        finally:
            await self.shutdown()
    
    async def _trading_cycle(self):
        """Single trading cycle"""
        try:
            self.last_heartbeat = datetime.now()
            
            # Update account info
            account_info = self.broker_adapter.get_account_info()
            if account_info:
                self.risk_manager.update_account_info(account_info)
            
            # Get current positions
            current_positions = self.broker_adapter.get_positions()
            
            # Update portfolio with current positions
            await self._update_portfolio_positions(current_positions)
            
            # Process each enabled symbol
            enabled_symbols = settings_manager.get_enabled_symbols()
            
            # Initialize cycle tracking
            cycle_stats = {
                'signals_processed': 0,
                'trades_executed': 0,
                'trades_skipped': 0,
                'skip_reasons': {}
            }
            
            logger.info(f"[CYCLE] Starting trading cycle - Processing {len(enabled_symbols)} symbols")
            logger.info(f"[SYMBOLS] Enabled symbols: {enabled_symbols[:5]}...")  # Show first 5 symbols
            
            for symbol in enabled_symbols:
                try:
                    logger.info(f"[PROCESS] Processing symbol: {symbol}")
                    result = await self._process_symbol(symbol, cycle_stats)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    self.errors_count += 1
            
            # Monitor open positions for trailing stops
            await self._monitor_positions()
            
            # Log cycle completion with detailed stats
            logger.info(f"[SUMMARY] Trading cycle completed:")
            logger.info(f"   - Signals processed: {cycle_stats['signals_processed']}")
            logger.info(f"   - Trades executed: {cycle_stats['trades_executed']}")
            logger.info(f"   - Trades skipped: {cycle_stats['trades_skipped']}")
            
            if cycle_stats['skip_reasons']:
                logger.info(f"   - Skip reasons:")
                for reason, count in cycle_stats['skip_reasons'].items():
                    logger.info(f"     * {reason}: {count}")
            
            self.signals_processed += cycle_stats['signals_processed']
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.errors_count += 1
    
    async def _process_symbol(self, symbol: str, cycle_stats: Dict):
        """Process a single symbol with detailed logging"""
        try:
            logger.info(f"[ANALYZE] Analyzing {symbol}...")
            
            # Get symbol configuration first
            symbol_config = self.symbol_configs.get(symbol, {})
            
            # Check if symbol can be traded (pass symbol config for weekend rules)
            can_trade, reason = self.session_manager.can_trade_symbol(symbol, symbol_config)
            if not can_trade:
                logger.info(f"   [SKIP] {reason}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons'][reason] = cycle_stats['skip_reasons'].get(reason, 0) + 1
                return
            if not symbol_config.get('enabled', True):
                logger.info(f"   [SKIP] Symbol disabled in configuration")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Symbol disabled'] = cycle_stats['skip_reasons'].get('Symbol disabled', 0) + 1
                return
            
            # Get current rates
            current_rates = self.broker_adapter.get_current_rates([symbol])
            if symbol not in current_rates:
                logger.info(f"   [SKIP] No current rates available")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['No current rates'] = cycle_stats['skip_reasons'].get('No current rates', 0) + 1
                return
            
            current_rate = current_rates[symbol]
            
            # Prepare OHLC data
            ohlc_data = {
                'open': current_rate['bid'],  # Simplified - would use proper OHLC
                'high': current_rate['ask'],
                'low': current_rate['bid'],
                'close': current_rate['bid']
            }
            
            # Get historical data for this symbol
            historical_data = self.historical_data.get(symbol, [])
            logger.info(f"   [DATA] Historical data: {len(historical_data)} bars")
            
            # Debug: Check if we have enough data for ML
            if len(historical_data) < 8:
                logger.warning(f"   [WARNING] Insufficient data for ML: {len(historical_data)} bars (need at least 8)")
            
            # Generate ML signal (Pine Script logic - no minimum data requirement)
            logger.info(f"   [ML] Generating ML signal...")
            signal_data = self.classifier.generate_signal(ohlc_data, historical_data)
            
            if not signal_data:
                logger.info(f"   [SKIP] No signal generated")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['No signal'] = cycle_stats['skip_reasons'].get('No signal', 0) + 1
                return
            
            cycle_stats['signals_processed'] += 1
            signal = signal_data.get('signal', 'NEUTRAL')
            confidence = signal_data.get('confidence', 0.0)
            
            logger.info(f"   [SIGNAL] Signal: {signal} | Confidence: {confidence:.3f}")
            logger.info(f"   [DEBUG] Signal data: prediction={signal_data.get('prediction', 'N/A')}, filter_applied={signal_data.get('filter_applied', 'N/A')}")
            
            # NOTE: Pine Script doesn't use confidence threshold for entry!
            # Confidence is only used for exit conditions (when confidence < 0.1)
            # The signal generation already includes all necessary filters
            logger.info(f"   [DEBUG] Confidence: {confidence:.3f} (not used for entry in Pine Script)")
            
            # REMOVED: Confidence threshold check - Pine Script doesn't use this for entry
            # The signal generation already includes all necessary filters
            
            # Check if signal is neutral (Pine Script uses 0 for neutral, not 'NEUTRAL')
            if signal == 0 or signal == 'NEUTRAL':
                logger.info(f"   [SKIP] Neutral signal (value: {signal})")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Neutral signal'] = cycle_stats['skip_reasons'].get('Neutral signal', 0) + 1
                return
            
            # Debug: Check signal values
            logger.info(f"   [DEBUG] Signal type: {type(signal)}, value: {signal}")
            if signal not in [1, -1]:
                logger.warning(f"   [WARNING] Unexpected signal value: {signal}")
            
            # Check spread
            symbol_info = self.broker_adapter.get_symbol_info(symbol)
            if not symbol_info:
                logger.info(f"   [SKIP] No symbol info available")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['No symbol info'] = cycle_stats['skip_reasons'].get('No symbol info', 0) + 1
                return
            
            # Get current spread for logging (no restrictions)
            spread_pips = self.broker_adapter.calculate_spread_pips(symbol)
            logger.info(f"   [SPREAD] {symbol}: Current spread: {spread_pips:.1f} pips")
            
            # Process signal
            await self._process_signal(symbol, signal_data, symbol_info, symbol_config, cycle_stats)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            self.errors_count += 1
    
    async def _monitor_positions(self):
        """Monitor open positions for trailing stops and updates"""
        try:
            # Get current positions from risk manager
            open_positions = [pos for pos in self.risk_manager.positions.values() if pos.status == 'open']
            
            if not open_positions:
                return
            
            logger.debug(f"[MONITOR] Checking {len(open_positions)} open positions for trailing stops")
            
            for position in open_positions:
                try:
                    # Get current market price
                    symbol_info = self.broker_adapter.get_symbol_info(position.symbol)
                    if not symbol_info:
                        continue
                    
                    current_price = symbol_info.ask if position.side == 'buy' else symbol_info.bid
                    
                    # Get historical data for ATR calculation
                    historical_data = self.historical_data.get(position.symbol, [])
                    
                    # 🎯 ENHANCED: Generate signal data for ALL timeframes (1m, 5m, 15m) for momentum exit checks
                    signal_data_1m = None
                    signal_data_5m = None
                    signal_data_15m = None
                    bars_held = None
                    
                    if historical_data and len(historical_data) >= 20:
                        try:
                            # Generate 1m signal data (existing logic)
                            signal_data_1m = self.classifier.generate_signal(historical_data[-1:], historical_data)
                            
                            # 🚀 NEW: Fetch and generate 5m signal data
                            import MetaTrader5 as mt5
                            if mt5.initialize():
                                try:
                                    # Get 5m historical data
                                    rates_5m = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_M5, 0, 100)
                                    if rates_5m is not None and len(rates_5m) > 20:
                                        # Convert to DataFrame format
                                        df_5m = pd.DataFrame(rates_5m)
                                        historical_data_5m = []
                                        for _, row in df_5m.iterrows():
                                            historical_data_5m.append({
                                                'time': row['time'],
                                                'open': row['open'],
                                                'high': row['high'],
                                                'low': row['low'],
                                                'close': row['close'],
                                                'volume': row['tick_volume']
                                            })
                                        
                                        # Generate 5m signal data
                                        signal_data_5m = self.classifier.generate_signal(historical_data_5m[-1:], historical_data_5m)
                                        logger.debug(f"[HTF_MOMENTUM] {position.symbol}: 5m signal data generated")
                                    else:
                                        logger.warning(f"[HTF_MOMENTUM] {position.symbol}: No 5m data available")
                                except Exception as e:
                                    logger.warning(f"[HTF_MOMENTUM] {position.symbol}: Error getting 5m data: {e}")
                                
                                try:
                                    # Get 15m historical data
                                    rates_15m = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_M15, 0, 100)
                                    if rates_15m is not None and len(rates_15m) > 20:
                                        # Convert to DataFrame format
                                        df_15m = pd.DataFrame(rates_15m)
                                        historical_data_15m = []
                                        for _, row in df_15m.iterrows():
                                            historical_data_15m.append({
                                                'time': row['time'],
                                                'open': row['open'],
                                                'high': row['high'],
                                                'low': row['low'],
                                                'close': row['close'],
                                                'volume': row['tick_volume']
                                            })
                                        
                                        # Generate 15m signal data
                                        signal_data_15m = self.classifier.generate_signal(historical_data_15m[-1:], historical_data_15m)
                                        logger.debug(f"[HTF_MOMENTUM] {position.symbol}: 15m signal data generated")
                                    else:
                                        logger.warning(f"[HTF_MOMENTUM] {position.symbol}: No 15m data available")
                                except Exception as e:
                                    logger.warning(f"[HTF_MOMENTUM] {position.symbol}: Error getting 15m data: {e}")
                            else:
                                logger.warning(f"[HTF_MOMENTUM] {position.symbol}: MT5 not initialized, using 1m only")
                            
                            # Calculate bars held (time since position opened)
                            from datetime import datetime, timedelta
                            time_held = datetime.now() - position.opened_at
                            # Assuming 1 bar = 1 minute for M1 timeframe
                            bars_held = int(time_held.total_seconds() / 60)
                            
                            logger.debug(f"[HTF_MOMENTUM] {position.symbol}: bars_held={bars_held}, 1m={signal_data_1m is not None}, 5m={signal_data_5m is not None}, 15m={signal_data_15m is not None}")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ [HTF_MOMENTUM] Error generating signal data for {position.symbol}: {e}")
                    
                    # Update position with trailing stop logic AND multi-timeframe momentum exit checks
                    self.risk_manager.update_position_prices(
                        position.symbol, 
                        current_price, 
                        historical_data,
                        signal_data_1m,  # 1m signal data for momentum exits
                        bars_held,       # Bars held for time-based exits
                        signal_data_5m,  # NEW: 5m signal data for HTF momentum
                        signal_data_15m  # NEW: 15m signal data for HTF momentum
                    )
                    
                    # Position monitoring completed (momentum exits and ATR logic handled in update_position_prices)
                    
                    # Check Smart Take Profit conditions
                    await self._check_smart_tp(position.symbol, current_price, historical_data)
                    
                except Exception as e:
                    logger.error(f"Error monitoring position {position.symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in position monitoring: {e}")
    
    async def _check_smart_tp(self, symbol: str, current_price: float, historical_data: List[Dict]):
        """Check and handle Smart Take Profit conditions"""
        try:
            # Check if partial TP should be taken
            if self.smart_tp.check_partial_tp(symbol, current_price):
                logger.info(f"✅ Partial TP triggered for {symbol} at {current_price:.5f}")
                await self._take_partial_profit(symbol, current_price)
            else:
                logger.debug(f"✅ Partial TP check passed for {symbol}")
            
            # Update trailing TP for remaining position
            new_trailing_tp = self.smart_tp.update_trailing_tp(symbol, current_price, historical_data)
            if new_trailing_tp:
                logger.info(f"✅ Trailing TP updated for {symbol}: {new_trailing_tp:.5f}")
                # Here you would update the TP in MT5 if needed
            else:
                logger.debug(f"✅ Trailing TP check passed for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error in Smart TP check for {symbol}: {e}")
    
    async def _take_partial_profit(self, symbol: str, current_price: float):
        """Take partial profit at the partial TP level - ACTUALLY EXECUTE IN MT5"""
        try:
            # Get position info
            if symbol not in self.risk_manager.positions:
                logger.warning(f"Cannot take partial TP for {symbol}: position not found")
                return
            
            position = self.risk_manager.positions[symbol]
            if position.status != 'open':
                logger.warning(f"Cannot take partial TP for {symbol}: position not open")
                return
            
            # Get MT5 position ticket
            import MetaTrader5 as mt5
            mt5_positions = mt5.positions_get(symbol=symbol)
            
            if not mt5_positions or len(mt5_positions) == 0:
                logger.warning(f"Cannot take partial TP for {symbol}: no MT5 position found")
                return
            
            mt5_position = mt5_positions[0]
            ticket = mt5_position.ticket
            
            # Calculate partial close amount (50% of position)
            partial_volume = position.lot_size * 0.5  # Take 50% profit
            
            # Execute partial close in MT5
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": ticket,
                "volume": partial_volume,
                "type": mt5.ORDER_TYPE_SELL if position.side == 'buy' else mt5.ORDER_TYPE_BUY,
                "price": current_price,
                "comment": "Partial TP",
            }
            
            result = mt5.order_send(close_request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Mark partial TP as taken
                self.smart_tp.take_partial_tp(symbol)
                
                # Update position lot size
                position.lot_size = position.lot_size - partial_volume
                
                logger.info(f"✅ [PARTIAL_TP] {symbol}: Partial profit taken - {partial_volume:.3f} lots at {current_price:.5f} (Ticket: {ticket})")
                logger.info(f"   [REMAINING] {symbol}: {position.lot_size:.3f} lots remaining")
            else:
                logger.error(f"❌ [PARTIAL_TP] {symbol}: Failed to execute partial close - {result.comment if result else 'No result'}")
                
        except Exception as e:
            logger.error(f"❌ [PARTIAL_TP] Error taking partial profit for {symbol}: {e}")
            
            # Note: In a full implementation, you would:
            # 1. Calculate the partial lot size (e.g., 50% of position)
            # 2. Close that portion of the position
            # 3. Update the remaining position's TP to trailing mode
            
        except Exception as e:
            logger.error(f"Error taking partial profit for {symbol}: {e}")
    
    def _validate_momentum_ignition(self, symbol: str, signal: int, historical_data: List[Dict]) -> Dict:
        """
        Validate momentum ignition before allowing trade entry.
        
        Requirements:
        1. Trigger candle: close > last swing high (5m) for buys, close < last swing low (5m) for sells
        2. Strong body filter: body size >= 60% of total candle range
        3. Momentum confirmation: RSI/WaveTrend/ADX must show acceleration
        4. Multi-timeframe validation: 5m trigger only if 15m bias matches
        5. Smart delay: Optional 5-15s delay after new candle open
        
        Args:
            symbol: Trading symbol
            signal: ML signal (1 for buy, -1 for sell)
            historical_data: Historical OHLC data
            
        Returns:
            Dict with 'allow_trade' (bool) and 'reason' (str)
        """
        try:
            if not historical_data or len(historical_data) < 50:
                return {
                    'allow_trade': False,
                    'reason': f'Insufficient data: {len(historical_data) if historical_data else 0} bars (need 50+)'
                }
            
            # Get recent data for analysis
            recent_data = historical_data[-20:]  # Last 20 bars for swing analysis
            current_candle = recent_data[-1]
            
            # 1. TRIGGER CANDLE VALIDATION
            logger.info(f"[MOMENTUM_FILTER] {symbol} | Analyzing trigger candle...")
            
            # Calculate swing highs/lows from 5m perspective (using 5 bars = 5 minutes)
            swing_period = 5
            if len(recent_data) < swing_period * 2:
                return {
                    'allow_trade': False,
                    'reason': f'Insufficient data for swing analysis: {len(recent_data)} bars'
                }
            
            # Find last swing high and low
            highs = [bar['high'] for bar in recent_data[-swing_period*2:]]
            lows = [bar['low'] for bar in recent_data[-swing_period*2:]]
            
            last_swing_high = max(highs[:-swing_period])  # Exclude current period
            last_swing_low = min(lows[:-swing_period])    # Exclude current period
            
            current_close = current_candle['close']
            current_open = current_candle['open']
            current_high = current_candle['high']
            current_low = current_candle['low']
            
            # Check trigger candle condition
            if signal > 0:  # Buy signal
                trigger_condition = current_close > last_swing_high
                trigger_desc = f"close {current_close:.5f} > swing high {last_swing_high:.5f}"
            else:  # Sell signal
                trigger_condition = current_close < last_swing_low
                trigger_desc = f"close {current_close:.5f} < swing low {last_swing_low:.5f}"
            
            if not trigger_condition:
                return {
                    'allow_trade': False,
                    'reason': f'No trigger candle: {trigger_desc}'
                }
            
            logger.info(f"[MOMENTUM_FILTER] {symbol} | [OK] Trigger candle: {trigger_desc}")
            
            # 2. STRONG BODY FILTER
            logger.info(f"[MOMENTUM_FILTER] {symbol} | Analyzing candle body strength...")
            
            candle_range = current_high - current_low
            body_size = abs(current_close - current_open)
            
            if candle_range == 0:
                return {
                    'allow_trade': False,
                    'reason': 'Zero candle range (doji candle)'
                }
            
            body_percentage = (body_size / candle_range) * 100
            min_body_percentage = 60.0
            
            if body_percentage < min_body_percentage:
                return {
                    'allow_trade': False,
                    'reason': f'Weak body: {body_percentage:.1f}% < {min_body_percentage}%'
                }
            
            logger.info(f"[MOMENTUM_FILTER] {symbol} | [OK] Strong body: {body_percentage:.1f}% >= {min_body_percentage}%")
            
            # 3. MOMENTUM CONFIRMATION
            logger.info(f"[MOMENTUM_FILTER] {symbol} | Analyzing momentum acceleration...")
            
            # Calculate RSI for momentum analysis
            import pandas as pd
            import numpy as np
            
            df = pd.DataFrame(recent_data)
            closes = df['close'].values
            
            # Simple RSI calculation
            def calculate_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return None
                
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gains = np.mean(gains[-period:])
                avg_losses = np.mean(losses[-period:])
                
                if avg_losses == 0:
                    return 100
                
                rs = avg_gains / avg_losses
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            # Calculate RSI for last 3 candles
            rsi_values = []
            for i in range(3, 0, -1):
                if len(closes) >= 14 + i:
                    rsi = calculate_rsi(closes[:-i], 14)
                    if rsi is not None:
                        rsi_values.append(rsi)
            
            if len(rsi_values) < 3:
                return {
                    'allow_trade': False,
                    'reason': f'Insufficient data for RSI calculation: {len(rsi_values)} values'
                }
            
            # Check RSI momentum (rising for buys, falling for sells)
            rsi_current = rsi_values[0]
            rsi_prev = rsi_values[1]
            rsi_prev2 = rsi_values[2]
            
            if signal > 0:  # Buy signal
                rsi_momentum = rsi_current > rsi_prev > rsi_prev2
                momentum_desc = f"RSI rising: {rsi_prev2:.1f} -> {rsi_prev:.1f} -> {rsi_current:.1f}"
            else:  # Sell signal
                rsi_momentum = rsi_current < rsi_prev < rsi_prev2
                momentum_desc = f"RSI falling: {rsi_prev2:.1f} -> {rsi_prev:.1f} -> {rsi_current:.1f}"
            
            if not rsi_momentum:
                return {
                    'allow_trade': False,
                    'reason': f'No momentum acceleration: {momentum_desc}'
                }
            
            logger.info(f"[MOMENTUM_FILTER] {symbol} | [OK] Momentum confirmed: {momentum_desc}")
            
            # 4. MULTI-TIMEFRAME VALIDATION (15m bias check)
            logger.info(f"[MOMENTUM_FILTER] {symbol} | Validating 15m bias alignment...")
            
            # Use 15 bars to represent 15m timeframe (15 minutes = 15 bars)
            tf_15m_data = historical_data[-15:] if len(historical_data) >= 15 else historical_data
            
            if len(tf_15m_data) < 10:
                return {
                    'allow_trade': False,
                    'reason': f'Insufficient 15m data: {len(tf_15m_data)} bars'
                }
            
            # Calculate 15m trend using simple moving average
            tf_15m_closes = [bar['close'] for bar in tf_15m_data]
            tf_15m_sma = np.mean(tf_15m_closes[-5:])  # Last 5 bars average
            tf_15m_prev_sma = np.mean(tf_15m_closes[-10:-5])  # Previous 5 bars average
            
            tf_15m_bullish = tf_15m_sma > tf_15m_prev_sma
            tf_15m_bearish = tf_15m_sma < tf_15m_prev_sma
            
            # Check if 15m bias matches signal direction
            if signal > 0 and not tf_15m_bullish:
                return {
                    'allow_trade': False,
                    'reason': f'15m bias conflict: signal BUY but 15m trend bearish ({tf_15m_sma:.5f} < {tf_15m_prev_sma:.5f})'
                }
            elif signal < 0 and not tf_15m_bearish:
                return {
                    'allow_trade': False,
                    'reason': f'15m bias conflict: signal SELL but 15m trend bullish ({tf_15m_sma:.5f} > {tf_15m_prev_sma:.5f})'
                }
            
            logger.info(f"[MOMENTUM_FILTER] {symbol} | [OK] 15m bias aligned: {tf_15m_sma:.5f} vs {tf_15m_prev_sma:.5f}")
            
            # 5. SMART DELAY (Optional - can be implemented with timestamp checking)
            # For now, we'll skip the delay as it requires real-time timestamp management
            
            # All conditions passed
            logger.info(f"[MOMENTUM_FILTER] {symbol} | [OK] ALL CONDITIONS PASSED - Momentum ignition confirmed!")
            logger.info(f"[MOMENTUM_FILTER] {symbol} | Summary:")
            logger.info(f"   - Trigger: {trigger_desc}")
            logger.info(f"   - Body: {body_percentage:.1f}% (strong)")
            logger.info(f"   - Momentum: {momentum_desc}")
            logger.info(f"   - 15m Bias: Aligned")
            
            return {
                'allow_trade': True,
                'reason': 'Momentum ignition confirmed',
                'details': {
                    'trigger_condition': trigger_desc,
                    'body_percentage': body_percentage,
                    'rsi_momentum': momentum_desc,
                    'tf_15m_aligned': True
                }
            }
            
        except Exception as e:
            logger.error(f"[MOMENTUM_FILTER] {symbol} | Error in momentum validation: {e}")
            return {
                'allow_trade': False,
                'reason': f'Momentum validation error: {str(e)}'
            }

    async def _process_signal(self, symbol: str, signal_data: Dict, symbol_info, symbol_config: Dict, cycle_stats: Dict):
        """Process a trading signal with detailed logging"""
        try:
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # ✅ ML Signal Validation
            if signal is None or not isinstance(signal, (int, float)) or signal not in [-1, 0, 1]:
                logger.error(f"❌ Invalid signal generated: {signal}")
                return
            if confidence is None or not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logger.error(f"❌ Invalid confidence: {confidence}")
                return
            
            logger.info(f"[OK] Signal generated: {signal}, confidence: {confidence:.3f}")
            
            logger.info(f"   [PROCESS] Processing {signal} signal for {symbol}...")
            
            # Signal validation already done in main loop, proceed with processing
            
            # Get historical data for MTF validation
            historical_data = self.historical_data.get(symbol, [])
            
            # ✅ Momentum Ignition Filter - Wait for actual momentum, not just direction bias
            logger.info(f"[MOMENTUM_FILTER] Starting momentum ignition validation for {symbol}...")
            momentum_result = self._validate_momentum_ignition(
                symbol=symbol,
                signal=signal,
                historical_data=historical_data
            )
            
            # Check if momentum ignition allows the trade
            if not momentum_result['allow_trade']:
                logger.info(f"[MOMENTUM_FILTER] {symbol} | BLOCKED: {momentum_result['reason']}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Momentum Filter'] = cycle_stats['skip_reasons'].get('Momentum Filter', 0) + 1
                return
            
            # ✅ MTF Validation - Validate signal across multiple timeframes
            logger.info(f"[MTF_VALIDATION] Starting multi-timeframe validation for {symbol}...")
            mtf_result = self.mtf_validator.validate_trade_signal(
                symbol=symbol,
                signal=signal,
                current_1m_data=historical_data[-1] if historical_data else {},
                historical_data=historical_data
            )
            
            # Check if MTF validation allows the trade
            if not mtf_result.allow_trade:
                logger.info(f"[MTF_VALIDATION] {symbol} | BLOCKED: {mtf_result.reasoning}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['MTF Validation'] = cycle_stats['skip_reasons'].get('MTF Validation', 0) + 1
                return
            
            # Apply MTF confidence boost
            original_confidence = confidence
            confidence = min(confidence + mtf_result.confidence_boost, 1.0)
            
            # Log MTF validation results
            tf_1m = mtf_result.timeframe_alignment.get('1m', 'unknown')
            tf_5m = mtf_result.timeframe_alignment.get('5m', 'unknown')
            tf_15m = mtf_result.timeframe_alignment.get('15m', 'unknown')
            logger.info(f"[MTF_VALIDATION] {symbol} | 1m: {tf_1m} | 5m: {tf_5m} | 15m: {tf_15m} | Final Action: ALLOWED")
            logger.info(f"[MTF_VALIDATION] {symbol} | Confidence: {original_confidence:.3f} -> {confidence:.3f} (+{mtf_result.confidence_boost:.3f})")
            logger.info(f"[MTF_VALIDATION] {symbol} | Lot Multiplier: {mtf_result.lot_multiplier:.2f} | TP Multiplier: {mtf_result.tp_multiplier:.2f}")
            
            # Log MTF decision with scenario label
            confidence_str = f"+{mtf_result.confidence_boost:.1f}" if mtf_result.confidence_boost > 0 else f"{mtf_result.confidence_boost:.1f}"
            logger.info(f"[MTF Decision] {mtf_result.scenario_label} -> Lot={mtf_result.lot_multiplier:.1f}x, TP={mtf_result.tp_multiplier:.1f}x, Confidence={confidence_str}")
            
            # Determine trade side
            side = 'buy' if signal > 0 else 'sell'
            
            # 🚨 CRITICAL DEBUG: Signal Direction Verification
            logger.info(f"[SIGNAL_DEBUG] {symbol} | ML Signal: {signal} | Mapped Side: {side.upper()}")
            logger.info(f"[SIGNAL_DEBUG] {symbol} | Signal > 0 = {signal > 0} | Side = 'buy' if signal > 0 else 'sell'")
            
            # Calculate entry price
            entry_price = symbol_info.ask if side == 'buy' else symbol_info.bid
            logger.info(f"[SIGNAL_DEBUG] {symbol} | Entry Price: {entry_price:.5f} ({'ASK' if side == 'buy' else 'BID'})")
            
            # ✅ Data Validation
            if not historical_data or len(historical_data) < 20:
                logger.error(f"❌ Insufficient historical data for {symbol}: {len(historical_data) if historical_data else 0} bars")
                return
            
            # Check for NaN values in OHLC data
            import pandas as pd
            df_check = pd.DataFrame(historical_data)
            if df_check[['open', 'high', 'low', 'close']].isnull().any().any():
                logger.error(f"❌ NaN values found in historical data for {symbol}")
                return
            
            logger.info(f"[OK] Data validated: {len(historical_data)} bars, no NaNs")
            
            # Get current spread for hybrid SL calculation
            current_spread_pips = self.broker_adapter.calculate_spread_pips(symbol)
            logger.info(f"[SPREAD] {symbol}: Current spread: {current_spread_pips:.1f} pips")
            
            # Calculate hybrid ATR-based stop loss
            stop_loss, sl_method = calculate_hybrid_stop_loss(
                symbol=symbol,
                entry_price=entry_price,
                side=side,
                historical_data=historical_data,
                spread_pips=current_spread_pips,
                atr_multiplier=1.5,  # 1.5x ATR for stop distance
                min_distance_pips=10.0  # Minimum 10 pips
            )
            
            # ✅ Stop Loss Validation
            if stop_loss is None or stop_loss <= 0:
                logger.error(f"❌ Invalid stop loss calculated: {stop_loss}")
                return
            
            # Validate stop loss is on correct side of entry
            if side == 'buy' and stop_loss >= entry_price:
                logger.error(f"❌ Stop loss {stop_loss:.5f} not below entry {entry_price:.5f} for buy order")
                return
            elif side == 'sell' and stop_loss <= entry_price:
                logger.error(f"❌ Stop loss {stop_loss:.5f} not above entry {entry_price:.5f} for sell order")
                return
            
            stop_distance = abs(entry_price - stop_loss)
            from core.risk import get_pip_value, get_price_distance_in_pips
            pip_multiplier = get_pip_value(symbol)
            stop_distance_pips = get_price_distance_in_pips(symbol, stop_distance)
            logger.info(f"[OK] SL validated: {stop_loss:.5f}, distance: {stop_distance:.5f} ({stop_distance_pips:.1f} pips), method: {sl_method}, pip_mult: {pip_multiplier:.5f}")
            
            # Calculate Smart Take Profit levels
            logger.info(f"[HYBRID_TP] Calculating hybrid TP for {symbol} {side} @ {entry_price:.5f}")
            smart_tp_result = self.smart_tp.calculate_smart_tp(
                symbol=symbol,
                entry_price=entry_price,
                side=side,
                historical_data=historical_data,
                stop_loss=stop_loss
            )
            
            # Use the full TP price for initial order
            take_profit = smart_tp_result.full_tp_price
            
            # Apply MTF TP multiplier
            original_tp = take_profit
            if side == 'buy':
                # For buy orders, adjust TP distance based on multiplier
                tp_distance = take_profit - entry_price
                if mtf_result.tp_multiplier >= 1.0:
                    # Increase TP distance for higher conviction
                    take_profit = entry_price + (tp_distance * mtf_result.tp_multiplier)
                else:
                    # For lower conviction, keep original TP (don't reduce it)
                    take_profit = original_tp
            else:
                # For sell orders, adjust TP distance based on multiplier
                tp_distance = entry_price - take_profit
                if mtf_result.tp_multiplier >= 1.0:
                    # Increase TP distance for higher conviction
                    take_profit = entry_price - (tp_distance * mtf_result.tp_multiplier)
                else:
                    # For lower conviction, keep original TP (don't reduce it)
                    take_profit = original_tp
            
            logger.info(f"[MTF_VALIDATION] {symbol} | TP: {original_tp:.5f} -> {take_profit:.5f} (x{mtf_result.tp_multiplier:.2f})")
            
            # Debug TP calculation
            logger.info(f"[HYBRID_TP] TP calculation results:")
            logger.info(f"   - Momentum strength: {smart_tp_result.momentum_strength}")
            logger.info(f"   - TP multiplier: {smart_tp_result.tp_multiplier:.1f}")
            logger.info(f"   - Partial TP: {smart_tp_result.partial_tp_price:.5f}")
            logger.info(f"   - Full TP: {smart_tp_result.full_tp_price:.5f}")
            logger.info(f"   - Should take partial: {smart_tp_result.should_take_partial}")
            logger.info(f"   - Trailing enabled: {smart_tp_result.trailing_enabled}")
            
            # ✅ Take Profit Validation
            if take_profit is None or take_profit <= 0:
                logger.error(f"❌ Invalid take profit calculated: {take_profit}")
                return
            
            # Validate take profit is on correct side of entry
            if side == 'buy' and take_profit <= entry_price:
                logger.error(f"❌ Take profit {take_profit:.5f} not above entry {entry_price:.5f} for buy order")
                return
            elif side == 'sell' and take_profit >= entry_price:
                logger.error(f"❌ Take profit {take_profit:.5f} not below entry {entry_price:.5f} for sell order")
                return
            
            # Validate minimum RR (1:1)
            tp_distance = abs(take_profit - entry_price)
            if tp_distance < stop_distance:
                logger.error(f"[ERROR] Invalid RR: TP distance {tp_distance:.5f} < SL distance {stop_distance:.5f}")
                return
            
            actual_rr = tp_distance / stop_distance if stop_distance > 0 else 0
            tp_distance_pips = get_price_distance_in_pips(symbol, tp_distance)
            logger.info(f"[OK] TP validated: {take_profit:.5f}, distance: {tp_distance:.5f} ({tp_distance_pips:.1f} pips), RR: {actual_rr:.2f}, pip_mult: {pip_multiplier:.5f}")
            
            # Register position for Smart TP tracking
            self.smart_tp.register_position(symbol, side, entry_price, smart_tp_result)
            
            # Comprehensive debug summary
            logger.info(f"[TRADE_SUMMARY] {symbol} {side.upper()} @ {entry_price:.5f}")
            logger.info(f"   [SL] Method: {sl_method}, Price: {stop_loss:.5f}, Distance: {stop_distance:.5f}")
            logger.info(f"   [TP] Momentum: {smart_tp_result.momentum_strength}, Multiplier: {smart_tp_result.tp_multiplier:.1f}")
            logger.info(f"   [TP] Full: {take_profit:.5f}, Partial: {smart_tp_result.partial_tp_price:.5f}")
            logger.info(f"   [RR] Actual: {actual_rr:.2f}:1, Partial RR: {self.smart_tp.config.partial_tp_rr:.1f}:1")
            logger.info(f"   [FEATURES] Partial TP: {smart_tp_result.should_take_partial}, Trailing: {smart_tp_result.trailing_enabled}")
            
            # ✅ Safety Check: Ensure both SL and TP are valid
            if stop_loss is None or stop_loss <= 0:
                logger.error(f"❌ SAFETY CHECK FAILED: Invalid stop loss {stop_loss}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Invalid stop loss'] = cycle_stats['skip_reasons'].get('Invalid stop loss', 0) + 1
                return
            
            if take_profit is None or take_profit <= 0:
                logger.error(f"❌ SAFETY CHECK FAILED: Invalid take profit {take_profit}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Invalid take profit'] = cycle_stats['skip_reasons'].get('Invalid take profit', 0) + 1
                return
            
            # Validate SL/TP logical relationship
            if side == 'buy' and (stop_loss >= entry_price or take_profit <= entry_price):
                logger.error(f"❌ SAFETY CHECK FAILED: Invalid SL/TP for buy - SL: {stop_loss:.5f}, Entry: {entry_price:.5f}, TP: {take_profit:.5f}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Invalid SL/TP relationship'] = cycle_stats['skip_reasons'].get('Invalid SL/TP relationship', 0) + 1
                return
            
            if side == 'sell' and (stop_loss <= entry_price or take_profit >= entry_price):
                logger.error(f"❌ SAFETY CHECK FAILED: Invalid SL/TP for sell - SL: {stop_loss:.5f}, Entry: {entry_price:.5f}, TP: {take_profit:.5f}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Invalid SL/TP relationship'] = cycle_stats['skip_reasons'].get('Invalid SL/TP relationship', 0) + 1
                return
            
            logger.info(f"[OK] SAFETY CHECK PASSED: Both SL and TP are valid and logically placed")
            
            # Calculate position size first to get risk amount
            lot_size, risk_amount = self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, confidence
            )
            
            if lot_size is None or lot_size <= 0:
                logger.error(f"   [SKIP] Invalid lot size calculated: {lot_size}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Invalid lot size'] = cycle_stats['skip_reasons'].get('Invalid lot size', 0) + 1
                return
            
            # 🚀 CRITICAL FIX: Check global cooldown FIRST (10 minutes after last trade close)
            is_in_cooldown, cooldown_message = self.risk_manager.is_in_global_cooldown()
            if is_in_cooldown:
                logger.info(f"   [COOLDOWN] Trade blocked: {cooldown_message}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Global cooldown'] = cycle_stats['skip_reasons'].get('Global cooldown', 0) + 1
                return
            
            # Check if we can open a position with the calculated risk amount (includes per-symbol cooldown)
            can_open, reason = self.risk_manager.can_open_position(symbol, symbol_info.spread, risk_amount, side)
            if not can_open:
                logger.info(f"   [SKIP] {reason}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons'][reason] = cycle_stats['skip_reasons'].get(reason, 0) + 1
                return
            
            # Apply MTF lot multiplier
            original_lot_size = lot_size
            lot_size = lot_size * mtf_result.lot_multiplier
            logger.info(f"[MTF_VALIDATION] {symbol} | Lot Size: {original_lot_size:.3f} -> {lot_size:.3f} (x{mtf_result.lot_multiplier:.2f})")
            
            # ✅ Lot Size Validation
            if lot_size is None or lot_size <= 0:
                logger.error(f"❌ Invalid lot size calculated: {lot_size}")
                return
            
            logger.info(f"✅ Lot size calculated: {lot_size:.3f}, risk: ${risk_amount:.2f}")
            
            # Round lot size to broker's step size
            rounded_lot_size = self.broker_adapter.round_lot_size(lot_size, symbol)
            
            logger.info(f"   [SIZE] Position size: {lot_size:.3f} lots -> {rounded_lot_size:.3f} lots | Risk: ${risk_amount:.2f}")
            
            # Validate lot size after rounding
            if rounded_lot_size <= 0:
                logger.info(f"   [SKIP] Invalid lot size after rounding ({rounded_lot_size:.3f})")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Invalid lot size'] = cycle_stats['skip_reasons'].get('Invalid lot size', 0) + 1
                return
            
            # Check if lot size is too small (less than minimum)
            symbol_info = self.broker_adapter.get_symbol_info(symbol)
            if symbol_info and rounded_lot_size < symbol_info.lot_min:
                logger.info(f"   [SKIP] Lot size too small ({rounded_lot_size:.3f} < {symbol_info.lot_min:.3f})")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Lot size too small'] = cycle_stats['skip_reasons'].get('Lot size too small', 0) + 1
                return
            
            # Place order
            from adapters.broker_base import OrderRequest
            
            order_request = OrderRequest(
                symbol=symbol,
                order_type=side,
                lot_size=rounded_lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment="Lorentzian ML"
            )
            
            # 🚨 CRITICAL DEBUG: Order Request Verification
            logger.info(f"[ORDER_DEBUG] {symbol} | OrderRequest Details:")
            logger.info(f"   - Symbol: {order_request.symbol}")
            logger.info(f"   - Order Type: {order_request.order_type} (should match side: {side})")
            logger.info(f"   - Lot Size: {order_request.lot_size:.3f}")
            logger.info(f"   - Stop Loss: {order_request.stop_loss:.5f}")
            logger.info(f"   - Take Profit: {order_request.take_profit:.5f}")
            logger.info(f"   - Comment: {order_request.comment}")
            
            logger.info(f"   [ORDER] Placing {side} order: {rounded_lot_size:.3f} lots @ {entry_price:.5f}")
            logger.info(f"   [LEVELS] Stop Loss: {stop_loss:.5f} ({sl_method}) | Take Profit: {take_profit:.5f} ({smart_tp_result.momentum_strength})")
            logger.info(f"   [HYBRID] SL Distance: {stop_distance:.5f} | TP Distance: {tp_distance:.5f} | RR: {actual_rr:.2f}:1")
            
            order_result = self.broker_adapter.place_order(order_request)
            
            # ✅ Trade Execution Validation
            if order_result.success:
                if order_result.order_id is None:
                    logger.error(f"❌ Order succeeded but no order ID returned")
                    return
                
                logger.info(f"✅ MT5 order confirmed: ticket={order_result.order_id}, price={order_result.price:.5f}")
                
                # Create trade record
                trade = self.portfolio_manager.open_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    lot_size=rounded_lot_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    risk_amount=risk_amount
                )
                
                if trade is None:
                    logger.error(f"❌ Failed to open trade in portfolio")
                    return
                
                logger.info(f"✅ Trade opened: {trade.id}")
                logger.info(f"   [EXECUTED] TRADE EXECUTED: {symbol} {side.upper()} {rounded_lot_size:.3f} lots @ {entry_price:.5f}")
                logger.info(f"   [DETAILS] Confidence: {confidence:.3f} | Risk: ${risk_amount:.2f}")
                logger.info(f"   [LEVELS] SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                
                cycle_stats['trades_executed'] += 1
                
                # Update risk manager
                self.risk_manager.open_position(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    spread_pips=symbol_info.spread
                )
                
            else:
                logger.info(f"   [FAILED] ORDER FAILED: {order_result.error_message}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Order failed'] = cycle_stats['skip_reasons'].get('Order failed', 0) + 1
                
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
            self.errors_count += 1
    
    async def _update_portfolio_positions(self, current_positions: List):
        """Update portfolio with current broker positions"""
        try:
            # Get open trades from portfolio
            open_trades = self.portfolio_manager.get_open_trades()
            
            # 🚨 CRITICAL FIX: Only process if we have open trades AND current positions
            # This prevents false closing of trades when broker positions are empty at startup
            if not open_trades:
                logger.debug("No open trades to sync")
                return
                
            if not current_positions:
                logger.debug("No current broker positions - skipping position sync to avoid false closes")
                return
            
            logger.debug(f"Syncing {len(open_trades)} open trades with {len(current_positions)} broker positions")
            
            # Check for closed positions
            for trade in open_trades:
                position_found = False
                
                for position in current_positions:
                    if position.symbol == trade.symbol and position.type == trade.side:
                        position_found = True
                        break
                
                if not position_found:
                    # Position was closed by broker
                    logger.info(f"Position closed by broker: {trade.symbol} {trade.side}")
                    # Clean up Smart TP tracking
                    self.smart_tp.close_position(trade.symbol)
                    # Note: In a real implementation, we'd need to get the actual close price
                    # For now, we'll mark it as closed with a placeholder
                    self.portfolio_manager.close_trade(trade.id, 0.0, "position_closed", self.risk_manager)
            
        except Exception as e:
            logger.error(f"Error updating portfolio positions: {e}")
    
    async def shutdown(self):
        """Shutdown the trading bot"""
        try:
            logger.info("Shutting down trading bot...")
            
            # Disconnect from broker
            self.broker_adapter.disconnect()
            
            # Log final statistics
            logger.info(f"Final statistics:")
            logger.info(f"  Signals processed: {self.signals_processed}")
            logger.info(f"  Errors: {self.errors_count}")
            logger.info(f"  Uptime: {datetime.now() - self.last_heartbeat}")
            
            logger.info("Trading bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    """Main entry point"""
    try:
        # Create and run trading bot
        bot = LorentzianTradingBot()
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())
