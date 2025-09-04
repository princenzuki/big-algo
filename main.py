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

from core.signals import LorentzianClassifier, Settings, FilterSettings
from core.risk import RiskManager, RiskSettings, AccountInfo, get_dynamic_spread, get_intelligent_sl, calculate_trailing_stop
from core.sessions import SessionManager
from core.portfolio import PortfolioManager
from core.smart_tp import SmartTakeProfit, SmartTPConfig
from adapters.mt5_adapter import MT5Adapter
from config.settings import settings_manager

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
        
        # Initialize Smart Take Profit system
        self.smart_tp = SmartTakeProfit(SmartTPConfig())
        
        # Initialize broker adapter
        self.broker_adapter = MT5Adapter(
            login=self.settings.broker_login,
            password=self.settings.broker_password,
            server=self.settings.broker_server
        )
        
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
                # Get historical rates (last 1000 bars) - using 5-minute timeframe to match Pine Script
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
            
            # Check if symbol can be traded
            can_trade, reason = self.session_manager.can_trade_symbol(symbol)
            if not can_trade:
                logger.info(f"   [SKIP] {reason}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons'][reason] = cycle_stats['skip_reasons'].get(reason, 0) + 1
                return
            
            # Get symbol configuration
            symbol_config = self.symbol_configs.get(symbol, {})
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
            
            # Check confidence threshold
            min_confidence = symbol_config.get('min_confidence', 0.3)
            if confidence < min_confidence:
                logger.info(f"   [SKIP] Confidence too low ({confidence:.3f} < {min_confidence:.3f})")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Low confidence'] = cycle_stats['skip_reasons'].get('Low confidence', 0) + 1
                return
            
            # Check if signal is neutral
            if signal == 'NEUTRAL':
                logger.info(f"   [SKIP] Neutral signal")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Neutral signal'] = cycle_stats['skip_reasons'].get('Neutral signal', 0) + 1
                return
            
            # Check spread
            symbol_info = self.broker_adapter.get_symbol_info(symbol)
            if not symbol_info:
                logger.info(f"   [SKIP] No symbol info available")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['No symbol info'] = cycle_stats['skip_reasons'].get('No symbol info', 0) + 1
                return
            
            # Get current spread for logging (no restrictions)
            spread_pips = self.broker_adapter.calculate_spread_pips(symbol)
            logger.info(f"   [SPREAD] Current spread: {spread_pips:.1f} pips")
            
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
                    
                    # Update position with trailing stop logic
                    self.risk_manager.update_position_prices(
                        position.symbol, 
                        current_price, 
                        historical_data
                    )
                    
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
                await self._take_partial_profit(symbol, current_price)
            
            # Update trailing TP for remaining position
            new_trailing_tp = self.smart_tp.update_trailing_tp(symbol, current_price, historical_data)
            if new_trailing_tp:
                logger.info(f"[SMART_TP] Trailing TP updated for {symbol}: {new_trailing_tp:.5f}")
                # Here you would update the TP in MT5 if needed
                
        except Exception as e:
            logger.error(f"Error in Smart TP check for {symbol}: {e}")
    
    async def _take_partial_profit(self, symbol: str, current_price: float):
        """Take partial profit at the partial TP level"""
        try:
            # Mark partial TP as taken
            self.smart_tp.take_partial_tp(symbol)
            
            # Here you would implement the actual partial close in MT5
            # For now, we'll just log it
            logger.info(f"[SMART_TP] Partial profit taken for {symbol} at {current_price:.5f}")
            
            # Note: In a full implementation, you would:
            # 1. Calculate the partial lot size (e.g., 50% of position)
            # 2. Close that portion of the position
            # 3. Update the remaining position's TP to trailing mode
            
        except Exception as e:
            logger.error(f"Error taking partial profit for {symbol}: {e}")
    
    async def _process_signal(self, symbol: str, signal_data: Dict, symbol_info, symbol_config: Dict, cycle_stats: Dict):
        """Process a trading signal with detailed logging"""
        try:
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            logger.info(f"   [PROCESS] Processing {signal} signal for {symbol}...")
            
            if signal == 0:  # Neutral signal
                return
            
            # Determine trade side
            side = 'buy' if signal > 0 else 'sell'
            
            # Calculate stop loss and take profit based on real-time spread
            entry_price = symbol_info.ask if side == 'buy' else symbol_info.bid
            
            # Get real-time spread and calculate spread-adaptive stop loss
            current_spread_pips = self.broker_adapter.calculate_spread_pips(symbol)
            stop_loss_pips = current_spread_pips + 10  # Spread + 10 pips buffer
            
            # Get symbol info for accurate pip-to-price conversion
            symbol_info_mt5 = self.broker_adapter.get_symbol_info(symbol)
            if not symbol_info_mt5:
                logger.error(f"Could not get symbol info for {symbol}")
                return
            
            # Calculate pip value in price units
            # For most pairs: 1 pip = 0.0001 (4-digit) or 0.00001 (5-digit)
            # But we need to account for JPY pairs where 1 pip = 0.01
            if 'JPY' in symbol:
                pip_value = 0.01  # JPY pairs: 1 pip = 0.01
            else:
                pip_value = 0.0001  # Most pairs: 1 pip = 0.0001
            
            stop_loss_price_distance = stop_loss_pips * pip_value
            
            # Calculate Smart Take Profit
            historical_data = self.historical_data.get(symbol, [])
            
            # First calculate stop loss
            if side == 'buy':
                stop_loss = entry_price - stop_loss_price_distance
            else:
                stop_loss = entry_price + stop_loss_price_distance
            
            # Calculate Smart Take Profit levels
            smart_tp_result = self.smart_tp.calculate_smart_tp(
                symbol=symbol,
                entry_price=entry_price,
                side=side,
                historical_data=historical_data,
                stop_loss=stop_loss
            )
            
            # Use the full TP price for initial order
            take_profit = smart_tp_result.full_tp_price
            
            # Register position for Smart TP tracking
            self.smart_tp.register_position(symbol, side, entry_price, smart_tp_result)
            
            logger.info(f"   [STOP] Spread: {current_spread_pips:.1f} pips, SL distance: {stop_loss_pips:.1f} pips")
            logger.info(f"   [CALC] Pip value: {pip_value:.5f}, Price distance: {stop_loss_price_distance:.5f}")
            logger.info(f"   [SMART_TP] Momentum: {smart_tp_result.momentum_strength}, TP multiplier: {smart_tp_result.tp_multiplier:.1f}")
            if smart_tp_result.should_take_partial:
                logger.info(f"   [SMART_TP] Partial TP: {smart_tp_result.partial_tp_price:.5f} (R:R {self.smart_tp.config.partial_tp_rr:.1f})")
            
            # Check if we can open a position
            can_open, reason = self.risk_manager.can_open_position(symbol, symbol_info.spread)
            if not can_open:
                logger.info(f"   [SKIP] {reason}")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons'][reason] = cycle_stats['skip_reasons'].get(reason, 0) + 1
                return
            
            # Calculate position size
            lot_size, risk_amount = self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, confidence
            )
            
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
            
            logger.info(f"   [ORDER] Placing {side} order: {rounded_lot_size:.3f} lots @ {entry_price:.5f}")
            logger.info(f"   [LEVELS] Stop Loss: {stop_loss:.5f} | Take Profit: {take_profit:.5f}")
            
            order_result = self.broker_adapter.place_order(order_request)
            
            if order_result.success:
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
            
            # Check for closed positions
            for trade in open_trades:
                position_found = False
                
                for position in current_positions:
                    if position.symbol == trade.symbol and position.type == trade.side:
                        position_found = True
                        break
                
                if not position_found:
                    # Position was closed
                    logger.info(f"Position closed: {trade.symbol} {trade.side}")
                    # Clean up Smart TP tracking
                    self.smart_tp.close_position(trade.symbol)
                    # Note: In a real implementation, we'd need to get the actual close price
                    # For now, we'll mark it as closed with a placeholder
                    self.portfolio_manager.close_trade(trade.id, 0.0, "position_closed")
            
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
