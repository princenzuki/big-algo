"""
Pine Script Parity Testing Framework

Compares Python implementation outputs with Pine Script reference outputs.
Must fail loudly if mismatched beyond floating tolerance.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import csv
import json
import logging

from core.signals import LorentzianClassifier, Settings, FilterSettings
from core.risk import RiskManager, RiskSettings
from core.sessions import SessionManager
from core.portfolio import PortfolioManager

logger = logging.getLogger(__name__)

class ParityTestResult:
    """Result of a parity test"""
    def __init__(self, variable_name: str, pine_value: float, python_value: float, 
                 tolerance: float = 1e-6):
        self.variable_name = variable_name
        self.pine_value = pine_value
        self.python_value = python_value
        self.tolerance = tolerance
        self.delta = abs(pine_value - python_value)
        self.passed = self.delta <= tolerance
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'variable': self.variable_name,
            'pine_value': self.pine_value,
            'python_value': self.python_value,
            'delta': self.delta,
            'tolerance': self.tolerance,
            'status': 'PASS' if self.passed else 'FAIL'
        }

class ParityTestSuite:
    """
    Comprehensive parity testing suite
    
    Tests all critical Pine Script calculations against reference outputs.
    """
    
    def __init__(self, reference_data_path: str = "tests/reference_data/"):
        self.reference_data_path = reference_data_path
        self.results: List[ParityTestResult] = []
        self.tolerance = 1e-6
        
        # Initialize components
        self.settings = Settings()
        self.filter_settings = FilterSettings()
        self.classifier = LorentzianClassifier(self.settings, self.filter_settings)
        self.risk_manager = RiskManager(RiskSettings())
        self.session_manager = SessionManager()
        self.portfolio_manager = PortfolioManager()
        
        # Initialize Pine Script reference classifier (persistent across tests)
        from core.pine_script_reference import PineScriptReference, PineScriptSettings, PineScriptFilterSettings
        self.pine_settings = PineScriptSettings()
        self.pine_filter_settings = PineScriptFilterSettings()
        self.pine_classifier = PineScriptReference(self.pine_settings, self.pine_filter_settings)
    
    def load_reference_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load Pine Script reference data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            
        Returns:
            DataFrame with Pine Script outputs
        """
        file_path = f"{self.reference_data_path}{symbol}_{timeframe}_pine_outputs.csv"
        
        try:
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'])
            return df
        except FileNotFoundError:
            logger.warning(f"Reference data not found: {file_path}")
            return pd.DataFrame()
    
    def generate_test_data(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic test data for parity testing
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            bars: Number of bars to generate
            
        Returns:
            DataFrame with OHLC data
        """
        # Generate realistic OHLC data
        np.random.seed(42)  # For reproducible tests
        
        base_price = 1.1000 if 'USD' in symbol else 100.0
        prices = [base_price]
        
        for i in range(bars - 1):
            # Random walk with slight upward bias
            change = np.random.normal(0, 0.001)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Generate OHLC from prices
        data = []
        for i, price in enumerate(prices):
            # Generate realistic OHLC spread
            spread = np.random.uniform(0.0001, 0.0005)
            high = price + np.random.uniform(0, spread)
            low = price - np.random.uniform(0, spread)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            data.append({
                'time': datetime.now() - timedelta(minutes=i*5),  # 5-minute bars
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(data)
    
    def test_lorentzian_distance(self, feature_series, feature_arrays, feature_count: int):
        """
        Test Lorentzian distance calculation
        
        This is the core distance metric that must match Pine Script exactly.
        """
        from core.signals import get_lorentzian_distance
        
        # Test with known values
        test_cases = [
            {
                'i': 0,
                'feature_count': feature_count,
                'feature_series': feature_series,
                'feature_arrays': feature_arrays,
                'expected': 0.0  # Distance to self should be 0
            }
        ]
        
        for case in test_cases:
            result = get_lorentzian_distance(
                case['i'], case['feature_count'], 
                case['feature_series'], case['feature_arrays']
            )
            
            test_result = ParityTestResult(
                f"lorentzian_distance_{case['i']}",
                case['expected'],
                result,
                self.tolerance
            )
            self.results.append(test_result)
            
            if not test_result.passed:
                logger.error(f"Lorentzian distance test failed: {test_result.to_dict()}")
    
    def test_feature_calculations(self, ohlc_data: Dict[str, float]):
        """
        Test feature calculations (RSI, WT, CCI, ADX)
        
        These must match Pine Script indicator calculations exactly.
        """
        feature_series = self.classifier.calculate_features(ohlc_data)
        
        # Get Pine Script feature calculations for comparison (using persistent classifier)
        
        # Convert single OHLC data to arrays for Pine Script reference
        ohlc_arrays = {
            'open': [ohlc_data['open']],
            'high': [ohlc_data['high']],
            'low': [ohlc_data['low']],
            'close': [ohlc_data['close']]
        }
        
        # Get Pine Script feature calculations (using persistent classifier)
        pine_features = self.pine_classifier.calculate_features(ohlc_arrays)
        
        # Test cases with actual Pine Script values
        test_cases = [
            ('f1_rsi', pine_features['f1'], feature_series.f1),
            ('f2_wt', pine_features['f2'], feature_series.f2),
            ('f3_cci', pine_features['f3'], feature_series.f3),
            ('f4_adx', pine_features['f4'], feature_series.f4),
            ('f5_rsi', pine_features['f5'], feature_series.f5)
        ]
        
        for variable_name, expected, actual in test_cases:
            test_result = ParityTestResult(
                variable_name,
                expected,
                actual,
                self.tolerance
            )
            self.results.append(test_result)
            
            if not test_result.passed:
                logger.error(f"Feature calculation test failed: {test_result.to_dict()}")
    
    def test_ml_prediction(self, ohlc_data: Dict[str, float], historical_data: List[Dict[str, float]]):
        """
        Test ML prediction calculation
        
        The core prediction logic must match Pine Script exactly.
        """
        # Convert single OHLC data to arrays for Pine Script reference
        ohlc_arrays = {
            'open': [ohlc_data['open']],
            'high': [ohlc_data['high']],
            'low': [ohlc_data['low']],
            'close': [ohlc_data['close']]
        }
        
        # Get Python implementation result
        python_signal_data = self.classifier.generate_signal(ohlc_data, historical_data)
        
        # Get Pine Script reference result (using persistent classifier)
        pine_signal_data = self.pine_classifier.generate_signal(ohlc_arrays, historical_data)
        
        # Test prediction components against Pine Script reference
        test_cases = [
            ('prediction', pine_signal_data['prediction'], python_signal_data['prediction']),
            ('confidence', pine_signal_data['confidence'], python_signal_data['confidence']),
            ('signal', pine_signal_data['signal'], python_signal_data['signal']),
            ('neighbors_count', pine_signal_data['neighbors_count'], python_signal_data['neighbors_count'])
        ]
        
        for variable_name, expected, actual in test_cases:
            test_result = ParityTestResult(
                variable_name,
                expected,
                actual,
                self.tolerance
            )
            self.results.append(test_result)
            
            if not test_result.passed:
                logger.error(f"ML prediction test failed: {test_result.to_dict()}")
                logger.error(f"Pine Script: {variable_name}={expected}, Python: {variable_name}={actual}")
    
    def test_risk_calculations(self, symbol: str, entry_price: float, stop_loss: float, confidence: float):
        """
        Test risk management calculations
        
        Position sizing and risk calculations must be consistent.
        """
        from core.risk import AccountInfo
        
        # Mock account info
        account_info = AccountInfo(
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            currency='USD'
        )
        
        self.risk_manager.update_account_info(account_info)
        
        lot_size, risk_amount = self.risk_manager.calculate_position_size(
            symbol, entry_price, stop_loss, confidence
        )
        
        # Test risk calculations
        test_cases = [
            ('lot_size_min', 0.01, lot_size),  # Should be at least minimum
            ('risk_amount_positive', 0.0, risk_amount),  # Should be positive
        ]
        
        for variable_name, expected, actual in test_cases:
            if variable_name == 'lot_size_min':
                passed = actual >= expected
            elif variable_name == 'risk_amount_positive':
                passed = actual > expected
            else:
                passed = abs(expected - actual) <= self.tolerance
            
            test_result = ParityTestResult(
                variable_name,
                expected,
                actual,
                self.tolerance
            )
            test_result.passed = passed
            self.results.append(test_result)
            
            if not test_result.passed:
                logger.error(f"Risk calculation test failed: {test_result.to_dict()}")
    
    def test_session_management(self, symbol: str):
        """
        Test session management logic
        
        Weekend blocking and session detection must work correctly.
        """
        can_trade, reason = self.session_manager.can_trade_symbol(symbol)
        current_session = self.session_manager.get_current_session()
        
        # Test session logic
        test_cases = [
            ('can_trade_boolean', True, can_trade),  # Should be boolean
            ('reason_string', 'OK', reason),         # Should be string
            ('current_session_string', 'london', current_session)  # Should be string
        ]
        
        for variable_name, expected, actual in test_cases:
            if variable_name == 'can_trade_boolean':
                passed = isinstance(actual, bool)
            elif variable_name == 'reason_string':
                passed = isinstance(actual, str)
            elif variable_name == 'current_session_string':
                passed = isinstance(actual, str) and actual in ['london', 'new_york', 'asia', 'closed']
            else:
                passed = expected == actual
            
            test_result = ParityTestResult(
                variable_name,
                expected if isinstance(expected, (int, float)) else 0,
                actual if isinstance(actual, (int, float)) else 0,
                self.tolerance
            )
            test_result.passed = passed
            self.results.append(test_result)
            
            if not test_result.passed:
                logger.error(f"Session management test failed: {test_result.to_dict()}")
    
    def run_comprehensive_test(self, symbol: str = 'EURUSD', timeframe: str = 'M5'):
        """
        Run comprehensive parity test suite
        
        Args:
            symbol: Trading symbol to test
            timeframe: Timeframe to test
        """
        logger.info(f"Starting comprehensive parity test for {symbol} {timeframe}")
        
        # Generate test data
        test_data = self.generate_test_data(symbol, timeframe, 100)
        
        # Test feature calculations
        for i, row in test_data.iterrows():
            ohlc_data = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            
            historical_data = test_data.iloc[:i+1].to_dict('records')
            
            # Test feature calculations
            self.test_feature_calculations(ohlc_data)
            
            # Test ML prediction (only if we have enough data)
            if len(historical_data) >= 10:
                self.test_ml_prediction(ohlc_data, historical_data)
        
        # Test risk calculations
        self.test_risk_calculations(symbol, 1.1000, 1.0950, 0.5)
        
        # Test session management
        self.test_session_management(symbol)
        
        # Generate report
        self.generate_parity_report()
        
        # Check if any tests failed
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            logger.error(f"Parity test failed: {len(failed_tests)} tests failed")
            for test in failed_tests:
                logger.error(f"Failed test: {test.to_dict()}")
            return False
        else:
            logger.info("All parity tests passed!")
            return True
    
    def generate_parity_report(self, output_file: str = "parity_report.csv"):
        """
        Generate detailed parity report
        
        Args:
            output_file: Output CSV file path
        """
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'variable', 'pine_value', 'python_value', 
                         'delta', 'tolerance', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        logger.info(f"Parity report generated: {output_file}")
    
    def get_test_summary(self) -> Dict[str, any]:
        """Get test summary statistics"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.passed])
        failed_tests = total_tests - passed_tests
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'failed_tests_details': [r.to_dict() for r in self.results if not r.passed]
        }

# Test fixtures for pytest
@pytest.fixture
def parity_test_suite():
    """Fixture for parity test suite"""
    return ParityTestSuite()

@pytest.fixture
def sample_ohlc_data():
    """Fixture for sample OHLC data"""
    return {
        'open': 1.1000,
        'high': 1.1010,
        'low': 1.0990,
        'close': 1.1005
    }

@pytest.fixture
def sample_historical_data():
    """Fixture for sample historical data"""
    return [
        {'open': 1.0995, 'high': 1.1005, 'low': 1.0990, 'close': 1.1000},
        {'open': 1.1000, 'high': 1.1010, 'low': 1.0995, 'close': 1.1005},
        {'open': 1.1005, 'high': 1.1015, 'low': 1.1000, 'close': 1.1010}
    ]

# Pytest test functions
def test_lorentzian_distance_parity(parity_test_suite, sample_ohlc_data):
    """Test Lorentzian distance calculation parity"""
    from core.signals import FeatureSeries, FeatureArrays
    
    feature_series = FeatureSeries(f1=50.0, f2=50.0, f3=0.0, f4=25.0, f5=50.0)
    feature_arrays = FeatureArrays(
        f1=[50.0, 51.0, 49.0],
        f2=[50.0, 51.0, 49.0],
        f3=[0.0, 1.0, -1.0],
        f4=[25.0, 26.0, 24.0],
        f5=[50.0, 51.0, 49.0]
    )
    
    parity_test_suite.test_lorentzian_distance(feature_series, feature_arrays, 5)
    
    # Check that no tests failed
    failed_tests = [r for r in parity_test_suite.results if not r.passed]
    assert len(failed_tests) == 0, f"Lorentzian distance tests failed: {failed_tests}"

def test_feature_calculations_parity(parity_test_suite, sample_ohlc_data):
    """Test feature calculations parity"""
    parity_test_suite.test_feature_calculations(sample_ohlc_data)
    
    # Check that no tests failed
    failed_tests = [r for r in parity_test_suite.results if not r.passed]
    assert len(failed_tests) == 0, f"Feature calculation tests failed: {failed_tests}"

def test_risk_calculations_parity(parity_test_suite):
    """Test risk calculations parity"""
    parity_test_suite.test_risk_calculations('EURUSD', 1.1000, 1.0950, 0.5)
    
    # Check that no tests failed
    failed_tests = [r for r in parity_test_suite.results if not r.passed]
    assert len(failed_tests) == 0, f"Risk calculation tests failed: {failed_tests}"

def test_session_management_parity(parity_test_suite):
    """Test session management parity"""
    parity_test_suite.test_session_management('EURUSD')
    
    # Check that no tests failed
    failed_tests = [r for r in parity_test_suite.results if not r.passed]
    assert len(failed_tests) == 0, f"Session management tests failed: {failed_tests}"

def test_comprehensive_parity(parity_test_suite):
    """Test comprehensive parity suite"""
    success = parity_test_suite.run_comprehensive_test('EURUSD', 'M5')
    assert success, "Comprehensive parity test failed"

if __name__ == "__main__":
    # Run parity tests directly
    suite = ParityTestSuite()
    success = suite.run_comprehensive_test()
    
    if success:
        print("✅ All parity tests passed!")
    else:
        print("❌ Parity tests failed!")
        summary = suite.get_test_summary()
        print(f"Failed tests: {summary['failed_tests']}")
        for test in summary['failed_tests_details']:
            print(f"  - {test['variable']}: Pine={test['pine_value']}, Python={test['python_value']}, Delta={test['delta']}")
