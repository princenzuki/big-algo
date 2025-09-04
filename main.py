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
from core.risk import RiskManager, RiskSettings, AccountInfo
from core.sessions import SessionManager
from core.portfolio import PortfolioManager
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
            
            spread_pips = self.broker_adapter.calculate_spread_pips(symbol)
            max_spread = symbol_config.get('max_spread_pips', 3.0)
            
            logger.info(f"   [SPREAD] Spread: {spread_pips:.1f} pips (max: {max_spread:.1f})")
            
            if spread_pips > max_spread:
                logger.info(f"   [SKIP] Spread too wide ({spread_pips:.1f} > {max_spread:.1f} pips)")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Spread too wide'] = cycle_stats['skip_reasons'].get('Spread too wide', 0) + 1
                return
            
            # Process signal
            await self._process_signal(symbol, signal_data, symbol_info, symbol_config, cycle_stats)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            self.errors_count += 1
    
    async def _process_signal(self, symbol: str, signal_data: Dict, symbol_info, symbol_config: Dict, cycle_stats: Dict):
        """Process a trading signal with detailed logging"""
        try:
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            logger.info(f"   [PROCESS] Processing {signal} signal for {symbol}...")
            
            if signal == 0:  # Neutral signal
                return
            
            # Determine trade side
            side = 'long' if signal > 0 else 'short'
            
            # Calculate stop loss and take profit
            entry_price = symbol_info.ask if side == 'long' else symbol_info.bid
            
            # Use ATR for stop loss/take profit calculation
            atr_period = symbol_config.get('atr_period', 14)
            sl_multiplier = symbol_config.get('sl_multiplier', 2.0)
            tp_multiplier = symbol_config.get('tp_multiplier', 3.0)
            
            # Simplified ATR calculation (would use proper ATR)
            atr = 0.001  # Placeholder
            
            if side == 'long':
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
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
            
            logger.info(f"   [SIZE] Position size: {lot_size:.2f} lots | Risk: ${risk_amount:.2f}")
            
            if lot_size <= 0:
                logger.info(f"   [SKIP] Invalid lot size ({lot_size:.2f})")
                cycle_stats['trades_skipped'] += 1
                cycle_stats['skip_reasons']['Invalid lot size'] = cycle_stats['skip_reasons'].get('Invalid lot size', 0) + 1
                return
            
            # Place order
            from adapters.broker_base import OrderRequest
            
            order_request = OrderRequest(
                symbol=symbol,
                order_type=side,
                lot_size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment="Lorentzian ML"
            )
            
            logger.info(f"   [ORDER] Placing {side} order: {lot_size:.2f} lots @ {entry_price:.5f}")
            logger.info(f"   [LEVELS] Stop Loss: {stop_loss:.5f} | Take Profit: {take_profit:.5f}")
            
            order_result = self.broker_adapter.place_order(order_request)
            
            if order_result.success:
                # Create trade record
                trade = self.portfolio_manager.open_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    lot_size=lot_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    risk_amount=risk_amount
                )
                
                logger.info(f"   [EXECUTED] TRADE EXECUTED: {symbol} {side.upper()} {lot_size:.2f} lots @ {entry_price:.5f}")
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
