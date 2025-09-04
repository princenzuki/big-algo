# Lorentzian Classification Trading Bot

A Python implementation of the Pine Script "Machine Learning: Lorentzian Classification" indicator with strict parity enforcement and web dashboard for P&L analytics.

## 🎯 Key Features

- **Strict Pine Parity**: Line-by-line translation of Pine Script logic
- **Risk Management**: 10% account risk cap, position sizing, cooldowns
- **Session Awareness**: Weekend blocking (except BTCUSD), timezone handling
- **MT5 Integration**: Broker adapter with deviation logging
- **Web Dashboard**: P&L analytics, algo health monitoring
- **Modular Design**: Extensible for future AI roles and features

## 🏗️ Architecture

```
project_root/
├── core/                    # Core trading logic (Pine parity)
│   ├── signals.py          # ML signals (line-by-line Pine port)
│   ├── risk.py             # Risk management
│   ├── portfolio.py        # Position tracking
│   └── sessions.py         # Session management
├── adapters/               # Broker connectors
│   ├── broker_base.py      # Standard interface
│   └── mt5_adapter.py      # MT5 implementation
├── app_api/                # Web API
│   ├── main.py             # FastAPI app
│   ├── models.py           # Pydantic models
│   └── routes/             # API endpoints
├── roles/                  # AI role stubs
│   ├── ai_quant.py
│   ├── ai_cto.py
│   ├── ai_engineer.py
│   └── ai_reporter.py
├── config/                 # Configuration
│   ├── settings.py         # Global settings
│   └── symbols.yaml        # Per-symbol config
├── utils/                  # Utilities
│   ├── time_utils.py       # Timezone handling
│   ├── logging.py          # Structured logging
│   └── helpers.py          # General helpers
├── tests/                  # Testing
│   ├── parity_test.py      # Pine parity tests
│   └── unit_tests/         # Unit tests
├── web_dashboard/          # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Dashboard pages
│   │   └── utils/          # Frontend utilities
│   └── package.json
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## ⚖️ Non-Negotiable Rules

1. **Python must mirror Pine Script exactly**
2. **Adjustments are optional overlays (default OFF)**
3. **Any deviation must be logged with reason codes**
4. **Parity testing is mandatory**
5. **Risk, concurrency, and session rules must be enforced**

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for web dashboard)
- MetaTrader 5 terminal
- MT5 account with API access

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd lorentzian-trading-bot
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Node.js dependencies**
```bash
cd web_dashboard
npm install
cd ..
```

4. **Configure the bot**
```bash
# Copy environment template
cp env.example .env

# Edit .env with your settings
nano .env

# Copy symbol configuration
cp config/symbols.yaml.example config/symbols.yaml
```

5. **Run parity tests**
```bash
python run_tests.py
```

### Running the Bot

#### Option 1: Manual Start
```bash
# Start trading bot
python start_bot.py

# In another terminal, start API server
python start_api.py

# In another terminal, start web dashboard
cd web_dashboard && npm start
```

#### Option 2: Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Access Points

- **Web Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Endpoints**: http://localhost:8000/api

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Broker Settings
BROKER_LOGIN=123456
BROKER_PASSWORD=your_password
BROKER_SERVER=your_server

# Risk Management
MAX_ACCOUNT_RISK_PERCENT=10.0
MIN_LOT_SIZE=0.01
MAX_CONCURRENT_TRADES=5

# Pine Script Settings
NEIGHBORS_COUNT=8
MAX_BARS_BACK=2000
FEATURE_COUNT=5

# Timezone
TRADING_TIMEZONE=Africa/Nairobi
```

### Symbol Configuration

Edit `config/symbols.yaml` to configure per-symbol settings:

```yaml
EURUSD:
  enabled: true
  allow_weekend: false
  min_confidence: 0.3
  max_spread_pips: 2.0
  atr_period: 14
  sl_multiplier: 2.0
  tp_multiplier: 3.0
  sessions:
    london: true
    new_york: true
    asia: false
```

## 🧪 Testing

### Run All Tests
```bash
python run_tests.py
```

### Individual Test Suites
```bash
# Parity tests
python -m pytest tests/parity_test.py -v

# Unit tests
python -m pytest tests/unit_tests/ -v

# Linting
python -m flake8 . --count --select=E9,F63,F7,F82

# Type checking
python -m mypy . --ignore-missing-imports
```

### Parity Testing

The parity testing framework ensures exact Pine Script compatibility:

- **Lorentzian Distance**: Core distance metric calculation
- **Feature Calculations**: RSI, WT, CCI, ADX indicators
- **ML Predictions**: Signal generation and confidence scoring
- **Risk Calculations**: Position sizing and risk management
- **Session Management**: Weekend blocking and timezone handling

## 📊 Web Dashboard

The React dashboard provides comprehensive P&L analytics:

### Dashboard Features
- **P&L Charts**: Daily, weekly, monthly, quarterly, yearly
- **Trade Management**: Open/closed trades, performance metrics
- **Risk Monitoring**: Real-time risk exposure, position limits
- **Algo Health**: System status, uptime, success rates
- **Settings**: Configuration management

### Dashboard Pages
- **Dashboard**: Overview with key metrics and charts
- **Trades**: Trade management and history
- **Risk**: Risk monitoring and position tracking
- **Health**: Algorithm health and system status
- **Settings**: Configuration and parameters

## 🔧 Development

### Project Structure
- `core/`: Core trading logic with Pine Script parity
- `adapters/`: Broker integration (MT5, future brokers)
- `app_api/`: FastAPI backend for web dashboard
- `web_dashboard/`: React frontend
- `tests/`: Comprehensive testing suite
- `config/`: Configuration management
- `utils/`: Utility functions and helpers
- `roles/`: AI role stubs for future expansion

### Adding New Features

1. **Core Logic**: Add to `core/` modules
2. **API Endpoints**: Add to `app_api/main.py`
3. **Frontend**: Add to `web_dashboard/src/`
4. **Tests**: Add to `tests/`
5. **Configuration**: Update `config/` files

### Code Standards

- **Type Hints**: All functions must have type hints
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for all new features
- **Parity**: Maintain Pine Script compatibility
- **Logging**: Structured logging for all operations

## 🚨 Risk Management

### Built-in Protections
- **10% Account Risk Cap**: Maximum risk exposure
- **Position Sizing**: Confidence-based lot sizing
- **Cooldown Periods**: 10-minute cooldown between trades
- **One Trade Per Symbol**: No duplicate positions
- **Spread Filtering**: Reject trades with wide spreads
- **Weekend Blocking**: No trading except BTCUSD on weekends

### Risk Monitoring
- **Real-time Risk Exposure**: Live risk percentage tracking
- **Position Limits**: Maximum concurrent trades
- **Stop Loss Enforcement**: Automatic SL placement
- **Deviation Logging**: All parameter deviations logged

## 🌍 Session Management

### Trading Sessions
- **London**: 10:00 - 19:00 (Nairobi time)
- **New York**: 15:00 - 00:00 (Nairobi time)
- **Asia**: 02:00 - 11:00 (Nairobi time)

### Weekend Rules
- **Block Period**: Friday 23:55 - Sunday 00:05
- **Exception**: BTCUSD trades 24/7
- **Timezone**: All times in Africa/Nairobi

## 📈 Performance Monitoring

### Key Metrics
- **Health Score**: 0-100 system health rating
- **Success Rate**: Trade execution success percentage
- **Uptime**: System running time
- **Risk Exposure**: Current vs maximum risk
- **Confidence**: Average ML model confidence

### Alerts
- **High Risk**: Risk approaching limits
- **System Errors**: Broker connection issues
- **Health Degradation**: Performance issues
- **Session Changes**: Trading session transitions

## 🔮 Future Roadmap

### AI Roles (Stubs Implemented)
- **AI Quant**: Strategy optimization and analysis
- **AI CTO**: System architecture and performance
- **AI Engineer**: Code optimization and testing
- **AI Reporter**: Automated reporting and insights

### Planned Features
- **Multi-Broker Support**: Additional broker adapters
- **Advanced Analytics**: Machine learning insights
- **Mobile App**: React Native mobile dashboard
- **Cloud Deployment**: AWS/Azure deployment options
- **Backtesting Engine**: Historical strategy testing

## 📝 License

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use at your own risk.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure parity compliance
6. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test suite
- Examine the parity reports
