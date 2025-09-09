"""
Portfolio Management Module

Tracks positions, trades, and P&L analytics.
Maintains trade history and provides statistics for the web dashboard.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    lot_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_amount: float
    pnl: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_reason: Optional[str]  # 'stop_loss', 'take_profit', 'manual', 'timeout'
    status: str  # 'open', 'closed', 'stopped'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        data['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return data

@dataclass
class PortfolioStats:
    """Portfolio statistics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    profit_factor: float
    avg_trade_duration: float
    total_risk_taken: float

class PortfolioManager:
    """
    Manages portfolio, trades, and P&L analytics
    
    Provides comprehensive tracking and statistics for the web dashboard.
    """
    
    def __init__(self, db_path: str = "trading_portfolio.db"):
        self.db_path = db_path
        self.trades: Dict[str, Trade] = {}
        self._init_database()
        self._load_trades()
    
    def _init_database(self):
        """Initialize SQLite database for trade persistence"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    lot_size REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    confidence REAL NOT NULL,
                    risk_amount REAL NOT NULL,
                    pnl REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    status TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
            """)
    
    def _load_trades(self):
        """Load trades from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM trades")
            rows = cursor.fetchall()
            
            for row in rows:
                trade = Trade(
                    id=row[0],
                    symbol=row[1],
                    side=row[2],
                    entry_price=row[3],
                    exit_price=row[4],
                    lot_size=row[5],
                    stop_loss=row[6],
                    take_profit=row[7],
                    confidence=row[8],
                    risk_amount=row[9],
                    pnl=row[10],
                    entry_time=datetime.fromisoformat(row[11]),
                    exit_time=datetime.fromisoformat(row[12]) if row[12] else None,
                    exit_reason=row[13],
                    status=row[14]
                )
                self.trades[trade.id] = trade
        
        logger.info(f"Loaded {len(self.trades)} trades from database")
    
    def _save_trade(self, trade: Trade):
        """Save trade to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades 
                (id, symbol, side, entry_price, exit_price, lot_size, stop_loss, 
                 take_profit, confidence, risk_amount, pnl, entry_time, exit_time, 
                 exit_reason, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.id, trade.symbol, trade.side, trade.entry_price, trade.exit_price,
                trade.lot_size, trade.stop_loss, trade.take_profit, trade.confidence,
                trade.risk_amount, trade.pnl, trade.entry_time.isoformat(),
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_reason, trade.status
            ))
    
    def open_trade(self, symbol: str, side: str, entry_price: float, 
                   lot_size: float, stop_loss: float, take_profit: float,
                   confidence: float, risk_amount: float) -> Trade:
        """
        Open a new trade
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            lot_size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: ML confidence
            risk_amount: Risk amount in account currency
            
        Returns:
            Trade object
        """
        trade_id = f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade = Trade(
            id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=None,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_amount=risk_amount,
            pnl=None,
            entry_time=datetime.now(),
            exit_time=None,
            exit_reason=None,
            status='open'
        )
        
        self.trades[trade_id] = trade
        self._save_trade(trade)
        
        logger.info(f"Trade opened: {trade_id} {side} {lot_size} lots @ {entry_price}")
        
        return trade
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str, risk_manager=None) -> Optional[float]:
        """
        Close a trade
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: Reason for closing
            risk_manager: Optional risk manager to update global cooldown
            
        Returns:
            P&L amount or None if trade not found
        """
        if trade_id not in self.trades:
            logger.warning(f"Trade not found: {trade_id}")
            return None
        
        trade = self.trades[trade_id]
        
        # 🚀 CRITICAL FIX: Update global cooldown if risk manager is provided
        if risk_manager is not None:
            risk_manager._last_trade_close_time = datetime.now()
            logger.info(f"[COOLDOWN] Global 10-minute cooldown started at {risk_manager._last_trade_close_time.strftime('%H:%M:%S')} (via portfolio close)")
        
        # Calculate P&L
        if trade.side == 'long':
            pnl = (exit_price - trade.entry_price) * trade.lot_size * 100000
        else:
            pnl = (trade.entry_price - exit_price) * trade.lot_size * 100000
        
        # Update trade
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.exit_time = datetime.now()
        trade.exit_reason = exit_reason
        trade.status = 'closed'
        
        self._save_trade(trade)
        
        logger.info(f"Trade closed: {trade_id} @ {exit_price}, P&L={pnl:.2f}, Reason={exit_reason}")
        
        return pnl
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        return [trade for trade in self.trades.values() if trade.status == 'open']
    
    def get_closed_trades(self, limit: Optional[int] = None) -> List[Trade]:
        """Get closed trades, optionally limited"""
        closed_trades = [trade for trade in self.trades.values() if trade.status == 'closed']
        closed_trades.sort(key=lambda x: x.exit_time, reverse=True)
        
        if limit:
            return closed_trades[:limit]
        return closed_trades
    
    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """Get all trades for a specific symbol"""
        return [trade for trade in self.trades.values() if trade.symbol == symbol]
    
    def get_portfolio_stats(self) -> PortfolioStats:
        """Calculate comprehensive portfolio statistics"""
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return PortfolioStats(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
                largest_win=0.0, largest_loss=0.0, max_drawdown=0.0,
                profit_factor=0.0, avg_trade_duration=0.0, total_risk_taken=0.0
            )
        
        # Basic stats
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in closed_trades if t.pnl and t.pnl < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L stats
        pnls = [t.pnl for t in closed_trades if t.pnl is not None]
        total_pnl = sum(pnls)
        
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Drawdown calculation
        cumulative_pnl = []
        running_total = 0
        for pnl in pnls:
            running_total += pnl
            cumulative_pnl.append(running_total)
        
        max_drawdown = 0
        peak = 0
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average trade duration
        durations = []
        for trade in closed_trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
        
        # Total risk taken
        total_risk_taken = sum(t.risk_amount for t in closed_trades)
        
        return PortfolioStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            total_risk_taken=total_risk_taken
        )
    
    def get_pnl_by_period(self, period: str = 'daily') -> Dict[str, float]:
        """
        Get P&L breakdown by time period
        
        Args:
            period: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
            
        Returns:
            Dictionary with period -> P&L mapping
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return {}
        
        pnl_by_period = {}
        
        for trade in closed_trades:
            if not trade.exit_time or not trade.pnl:
                continue
            
            if period == 'daily':
                key = trade.exit_time.strftime('%Y-%m-%d')
            elif period == 'weekly':
                # Get Monday of the week
                monday = trade.exit_time - timedelta(days=trade.exit_time.weekday())
                key = monday.strftime('%Y-W%U')
            elif period == 'monthly':
                key = trade.exit_time.strftime('%Y-%m')
            elif period == 'quarterly':
                quarter = (trade.exit_time.month - 1) // 3 + 1
                key = f"{trade.exit_time.year}-Q{quarter}"
            elif period == 'yearly':
                key = str(trade.exit_time.year)
            else:
                continue
            
            if key not in pnl_by_period:
                pnl_by_period[key] = 0
            pnl_by_period[key] += trade.pnl
        
        return pnl_by_period
    
    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of trades by confidence level"""
        closed_trades = self.get_closed_trades()
        
        distribution = {
            'low': 0,    # 0.0 - 0.33
            'medium': 0, # 0.33 - 0.66
            'high': 0    # 0.66 - 1.0
        }
        
        for trade in closed_trades:
            if trade.confidence < 0.33:
                distribution['low'] += 1
            elif trade.confidence < 0.66:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1
        
        return distribution
    
    def get_symbol_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by symbol"""
        closed_trades = self.get_closed_trades()
        
        symbol_stats = {}
        
        for trade in closed_trades:
            if trade.symbol not in symbol_stats:
                symbol_stats[trade.symbol] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0,
                    'avg_confidence': 0,
                    'total_risk': 0
                }
            
            stats = symbol_stats[trade.symbol]
            stats['total_trades'] += 1
            stats['total_pnl'] += trade.pnl or 0
            stats['total_risk'] += trade.risk_amount
            
            if trade.pnl and trade.pnl > 0:
                stats['winning_trades'] += 1
        
        # Calculate averages and win rates
        for symbol, stats in symbol_stats.items():
            if stats['total_trades'] > 0:
                stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
                
                # Calculate average confidence for this symbol
                symbol_trades = [t for t in closed_trades if t.symbol == symbol]
                stats['avg_confidence'] = sum(t.confidence for t in symbol_trades) / len(symbol_trades)
        
        return symbol_stats
    
    def export_trades_csv(self, filepath: str):
        """Export trades to CSV file"""
        import csv
        
        closed_trades = self.get_closed_trades()
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'id', 'symbol', 'side', 'entry_price', 'exit_price', 'lot_size',
                'stop_loss', 'take_profit', 'confidence', 'risk_amount', 'pnl',
                'entry_time', 'exit_time', 'exit_reason', 'status'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for trade in closed_trades:
                row = trade.to_dict()
                writer.writerow(row)
        
        logger.info(f"Exported {len(closed_trades)} trades to {filepath}")
    
    def get_dashboard_data(self) -> Dict[str, any]:
        """Get comprehensive data for web dashboard"""
        stats = self.get_portfolio_stats()
        open_trades = self.get_open_trades()
        recent_trades = self.get_closed_trades(limit=10)
        
        return {
            'portfolio_stats': asdict(stats),
            'open_trades': [trade.to_dict() for trade in open_trades],
            'recent_trades': [trade.to_dict() for trade in recent_trades],
            'pnl_by_period': {
                'daily': self.get_pnl_by_period('daily'),
                'weekly': self.get_pnl_by_period('weekly'),
                'monthly': self.get_pnl_by_period('monthly'),
                'quarterly': self.get_pnl_by_period('quarterly'),
                'yearly': self.get_pnl_by_period('yearly')
            },
            'confidence_distribution': self.get_confidence_distribution(),
            'symbol_performance': self.get_symbol_performance()
        }
