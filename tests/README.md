# MT5 Trading Bot Test Suite

Comprehensive unit and integration tests for the MT5 trading bot, focusing on risk management, TP/SL logic, and silent failure prevention.

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_smart_tp.py           # Smart Take Profit system tests
├── test_risk_manager.py       # Risk management system tests
├── test_trade_execution.py    # Trade execution and MT5 adapter tests
├── test_main_trading_logic.py # Main trading logic and validation tests
├── test_integration.py        # End-to-end integration tests
├── run_tests.py              # Test runner with various configurations
└── README.md                 # This file
```

## Test Categories

### Unit Tests
- **Smart Take Profit System** (`test_smart_tp.py`)
  - ATR calculation with valid/invalid data
  - Momentum calculation (ADX, CCI)
  - Hybrid TP system with 1:1 RR minimum
  - Fallback TP when ATR fails
  - Partial TP and trailing TP logic

- **Risk Management** (`test_risk_manager.py`)
  - Position sizing with different confidence levels
  - Stop loss validation (correct side, non-zero, positive)
  - Take profit validation (correct side, RR >= 1:1)
  - Lot size validation (positive, within limits)
  - Break-even logic activation
  - Dynamic spread calculation

- **Trade Execution** (`test_trade_execution.py`)
  - Order request creation and validation
  - Successful/failed order placement
  - MT5 response handling
  - Lot size rounding and validation
  - Stop level adjustment
  - Error handling and recovery

- **Main Trading Logic** (`test_main_trading_logic.py`)
  - Signal validation (ML output)
  - Historical data validation
  - Stop loss/take profit calculation
  - Risk-reward validation
  - Trade execution flow
  - Break-even and trailing updates

### Integration Tests
- **End-to-End Workflows** (`test_integration.py`)
  - Complete trading cycle from signal to execution
  - Hybrid TP system integration
  - Risk management across components
  - Break-even and trailing stop integration
  - Partial TP integration
  - Error recovery and fallback mechanisms
  - Multi-symbol trading

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_tests.py all

# Run unit tests only
python tests/run_tests.py unit

# Run integration tests only
python tests/run_tests.py integration

# Run fast tests (exclude slow tests)
python tests/run_tests.py fast

# Run with coverage report
python tests/run_tests.py coverage

# Run specific test file
python tests/run_tests.py specific test_smart_tp.py
```

### Using pytest directly
```bash
# Run all tests
pytest tests/ -v -s

# Run specific test file
pytest tests/test_smart_tp.py -v -s

# Run tests with coverage
pytest tests/ --cov=core --cov=adapters --cov-report=html

# Run only unit tests
pytest tests/ -m unit -v -s

# Run only integration tests
pytest tests/ -m integration -v -s

# Run fast tests only
pytest tests/ -m "not slow" -v -s
```

## Test Features

### Comprehensive Validation
- ✅ **ML Signal Validation** - Ensures signals are valid (-1, 0, 1) with confidence [0,1]
- ✅ **Data Validation** - Checks for sufficient bars and NaN values
- ✅ **Stop Loss Validation** - Verifies correct side of entry and positive values
- ✅ **Take Profit Validation** - Enforces minimum 1:1 RR and correct side
- ✅ **Lot Size Validation** - Ensures positive values within broker limits
- ✅ **Trade Execution Validation** - Confirms MT5 order success and order ID

### Silent Failure Prevention
- ✅ **ATR Calculation** - Handles insufficient data, NaN values, and calculation errors
- ✅ **Momentum Indicators** - Graceful fallback when ADX/CCI calculations fail
- ✅ **Hybrid TP System** - Ensures 1:1 RR minimum with clear fallback logging
- ✅ **Error Recovery** - Comprehensive error handling with detailed logging

### Mock Data and Dependencies
- ✅ **Historical Data** - Realistic OHLC data generation for testing
- ✅ **MT5 Responses** - Mock successful and failed order responses
- ✅ **Symbol Info** - Mock broker symbol information
- ✅ **Account Info** - Mock account balance and margin data

### Debug Logging
- ✅ **Checkpoint Logging** - Every major step is logged with ✅/⚠️/❌ indicators
- ✅ **Fallback Transparency** - Clear logging when fallbacks are used
- ✅ **Error Context** - Detailed error messages with full context
- ✅ **Test Output** - Each test prints its progress and results

## Test Coverage

The test suite covers:

1. **Smart Take Profit System**
   - ATR-based TP calculation
   - Momentum-based adjustments
   - Hybrid TP with 1:1 RR minimum
   - Partial TP and trailing TP logic
   - Fallback mechanisms

2. **Risk Management**
   - Position sizing algorithms
   - Stop loss and take profit validation
   - Lot size validation and rounding
   - Break-even logic
   - Dynamic spread calculation

3. **Trade Execution**
   - Order placement and validation
   - MT5 adapter functionality
   - Error handling and recovery
   - Stop level adjustments

4. **Main Trading Logic**
   - Signal processing and validation
   - Data validation and error handling
   - Trade execution flow
   - Position monitoring and updates

5. **Integration Scenarios**
   - End-to-end trading cycles
   - Component interaction
   - Error recovery workflows
   - Multi-symbol trading

## Expected Test Results

When running the tests, you should see:

- ✅ **All unit tests pass** - Core functionality works correctly
- ✅ **All integration tests pass** - Components work together properly
- ✅ **Comprehensive logging** - Clear indication of what's being tested
- ✅ **Error handling** - Graceful handling of edge cases and failures
- ✅ **Fallback mechanisms** - Proper fallbacks when primary methods fail

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is in Python path
   - Check that all dependencies are installed

2. **Mock Failures**
   - Verify mock data matches expected format
   - Check mock return values are properly configured

3. **Test Failures**
   - Check test output for specific error messages
   - Verify mock data is realistic and valid
   - Ensure all dependencies are properly mocked

### Debug Mode

To run tests in debug mode with maximum logging:

```bash
pytest tests/ -v -s --log-cli-level=DEBUG
```

This will show all debug messages and help identify issues in the test execution.

## Contributing

When adding new tests:

1. Follow the existing naming convention: `test_<functionality>_<scenario>()`
2. Use descriptive test names that explain what is being tested
3. Include print statements to show test progress
4. Use appropriate assertions with clear error messages
5. Mock external dependencies appropriately
6. Test both success and failure scenarios
7. Include edge cases and error conditions

## Test Data

The test suite uses realistic mock data:

- **Historical Data**: 100 bars of OHLC data with realistic price movements
- **Symbol Info**: Standard EURUSD configuration with typical broker settings
- **Account Info**: $10,000 balance with standard margin requirements
- **Risk Settings**: 2% per trade, 5% daily, 10% total risk limits

This ensures tests are representative of real-world trading scenarios while remaining fast and reliable.
