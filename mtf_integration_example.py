"""
Multi-Timeframe Validator Integration Example

This file shows how to integrate the MTF Validator into your existing trading bot
with minimal changes to the main algorithm.

The integration is designed to be non-intrusive - it only adds one call before
trade execution without modifying any existing core logic.
"""

from mtf_validator import MultiTimeframeValidator, MTFValidationResult
from typing import Dict, Any

class MTFIntegrationExample:
    """
    Example of how to integrate MTF Validator into your existing trading bot
    """
    
    def __init__(self, broker_adapter):
        # Initialize the MTF Validator
        self.mtf_validator = MultiTimeframeValidator(broker_adapter)
        
    def process_signal_with_mtf_validation(self, symbol: str, signal_data: Dict, 
                                         historical_data: list, symbol_info: Any) -> Dict:
        """
        Process a trading signal with MTF validation
        
        This is the only function you need to add to your main bot.
        It wraps around your existing signal processing logic.
        
        Args:
            symbol: Trading symbol
            signal_data: Signal data from your existing Lorentzian ML algorithm
            historical_data: Historical data from your existing data pipeline
            symbol_info: Symbol information from your broker adapter
            
        Returns:
            Dictionary with trade decision and modifiers
        """
        try:
            # Extract signal from your existing algorithm
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.0)
            
            # If no signal, return as-is
            if signal == 0:
                return {
                    'execute_trade': False,
                    'original_signal': signal_data,
                    'mtf_validation': None,
                    'reason': 'No signal from main algorithm'
                }
            
            # Get current 1m data (use the most recent bar from historical data)
            current_1m_data = historical_data[-1] if historical_data else {}
            
            # Run MTF validation
            mtf_result = self.mtf_validator.validate_trade_signal(
                symbol=symbol,
                signal=signal,
                current_1m_data=current_1m_data,
                historical_data=historical_data
            )
            
            # Apply MTF validation results
            if not mtf_result.allow_trade:
                return {
                    'execute_trade': False,
                    'original_signal': signal_data,
                    'mtf_validation': mtf_result,
                    'reason': f'MTF validation blocked: {mtf_result.reasoning}'
                }
            
            # Apply trade modifiers
            modified_lot_size = self._apply_lot_size_modifier(symbol_info, mtf_result)
            modified_tp_distance = self._apply_tp_modifier(mtf_result)
            
            return {
                'execute_trade': True,
                'original_signal': signal_data,
                'mtf_validation': mtf_result,
                'modified_lot_size': modified_lot_size,
                'modified_tp_multiplier': modified_tp_distance,
                'confidence_boost': mtf_result.confidence_boost,
                'reason': f'MTF validation passed: {mtf_result.reasoning}'
            }
            
        except Exception as e:
            # Fail-safe: if MTF validation fails, proceed with original algorithm
            return {
                'execute_trade': True,
                'original_signal': signal_data,
                'mtf_validation': None,
                'reason': f'MTF validation failed, using original algorithm: {e}'
            }
    
    def _apply_lot_size_modifier(self, symbol_info: Any, mtf_result: MTFValidationResult) -> float:
        """
        Apply lot size modifier based on MTF validation
        
        Args:
            symbol_info: Symbol information from broker
            mtf_result: MTF validation result
            
        Returns:
            Modified lot size
        """
        # Get original lot size from your existing position sizing logic
        original_lot_size = 0.01  # This would come from your existing risk management
        
        # Apply MTF multiplier
        modified_lot_size = original_lot_size * mtf_result.lot_multiplier
        
        # Ensure lot size is within broker limits
        min_lot = getattr(symbol_info, 'min_lot', 0.01)
        max_lot = getattr(symbol_info, 'max_lot', 100.0)
        
        return max(min_lot, min(modified_lot_size, max_lot))
    
    def _apply_tp_modifier(self, mtf_result: MTFValidationResult) -> float:
        """
        Apply take-profit modifier based on MTF validation
        
        Args:
            mtf_result: MTF validation result
            
        Returns:
            Modified TP multiplier
        """
        return mtf_result.tp_multiplier

# Example of how to integrate this into your main.py
def example_integration_in_main_py():
    """
    Example of how to modify your main.py to use MTF validation
    
    This shows the minimal changes needed in your existing _process_signal method
    """
    
    # In your LorentzianTradingBot.__init__ method, add:
    # self.mtf_integration = MTFIntegrationExample(self.broker_adapter)
    
    # In your _process_signal method, replace the trade execution logic with:
    """
    # Original signal processing (unchanged)
    signal = signal_data['signal']
    confidence = signal_data['confidence']
    # ... all your existing logic ...
    
    # NEW: Add MTF validation before trade execution
    mtf_decision = self.mtf_integration.process_signal_with_mtf_validation(
        symbol=symbol,
        signal_data=signal_data,
        historical_data=historical_data,
        symbol_info=symbol_info
    )
    
    # Check if trade should be executed
    if not mtf_decision['execute_trade']:
        logger.info(f"   [SKIP] MTF Validation: {mtf_decision['reason']}")
        return
    
    # Apply MTF modifiers to your existing trade execution
    # Use mtf_decision['modified_lot_size'] instead of original lot size
    # Use mtf_decision['modified_tp_multiplier'] to adjust TP distance
    # Add mtf_decision['confidence_boost'] to your confidence calculation
    
    # Rest of your existing trade execution logic remains unchanged
    """
    
    pass

if __name__ == "__main__":
    # Example usage
    print("MTF Validator Integration Example")
    print("This shows how to integrate the Multi-Timeframe Validator into your existing bot.")
    print("The integration is designed to be non-intrusive and fail-safe.")
