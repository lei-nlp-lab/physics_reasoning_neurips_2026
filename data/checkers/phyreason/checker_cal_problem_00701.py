import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all expected results and inputs without direct attribute access
        self.results = {}
        self.inputs = {}
        
        # Extract results using safe getattr with fallback to results dictionary
        self.results['A_1'] = getattr(solution_instance, 'A_1', 
                                      getattr(solution_instance, 'results', {}).get('A_1', None))
        self.results['A_2'] = getattr(solution_instance, 'A_2',
                                      getattr(solution_instance, 'results', {}).get('A_2', None))
        self.results['fracA_1A_2'] = getattr(solution_instance, 'fracA_1A_2',
                                             getattr(solution_instance, 'results', {}).get('fracA_1A_2', None))
        
        # If symbolic values are stored as strings, convert to sympy expressions
        for key in self.results:
            if isinstance(self.results[key], str):
                try:
                    self.results[key] = sp.sympify(self.results[key])
                except:
                    pass
        
        # Extract inputs if available (x might be stored as input)
        self.inputs['x'] = getattr(solution_instance, 'x',
                                   getattr(solution_instance, 'inputs', {}).get('x', None))
        
        # Initialize logs
        self.logs = []
        self.success = True
        
    def _log(self, msg, is_error=False):
        self.logs.append(msg)
        if is_error:
            self.success = False
            self.logs.append(f"ERROR: {msg}")
    
    def _is_close(self, a, b):
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        
        # Handle symbolic expressions
        if isinstance(a, sp.Expr) and isinstance(b, sp.Expr):
            return sp.simplify(a - b) == 0
        
        # Convert to float if possible
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            # Fallback to symbolic equality
            return str(a) == str(b)
    
    def test_invariants(self):
        """Verify physical invariants from KB."""
        try:
            # 1. Check ratio is exactly 1/2
            expected_ratio = sp.Rational(1, 2)
            if self.results['fracA_1A_2'] is not None:
                if not self._is_close(self.results['fracA_1A_2'], expected_ratio):
                    self._log(f"Ratio mismatch: {self.results['fracA_1A_2']} != 1/2", True)
                else:
                    self._log("Ratio test passed: A1/A2 = 1/2")
            
            # 2. Check A2 = 2*A1
            if self.results['A_1'] is not None and self.results['A_2'] is not None:
                if not self._is_close(self.results['A_2'], 2 * self.results['A_1']):
                    self._log(f"A2 != 2*A1: {self.results['A_2']} != 2*{self.results['A_1']}", True)
                else:
                    self._log("Amplitude relationship test passed: A2 = 2*A1")
            
            # 3. Check consistency with KB formulas
            # Using amplitude formula: A = |initial displacement| when v0=0
            if self.inputs['x'] is not None and self.results['A_1'] is not None:
                x_val = self.inputs['x']
                if not self._is_close(self.results['A_1'], abs(x_val)):
                    self._log(f"A1 != |x|: {self.results['A_1']} != {abs(x_val)}", True)
                else:
                    self._log(f"A1 consistency test passed: A1 = |x| = {abs(x_val)}")
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {traceback.format_exc()}", True)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Check that ratio can be computed from A1 and A2
            if (self.results['A_1'] is not None and 
                self.results['A_2'] is not None and 
                self.results['fracA_1A_2'] is not None):
                
                # Compute ratio from amplitudes
                computed_ratio = self.results['A_1'] / self.results['A_2']
                
                if not self._is_close(computed_ratio, self.results['fracA_1A_2']):
                    self._log(f"Trace inconsistency: {computed_ratio} != {self.results['fracA_1A_2']}", True)
                else:
                    self._log("Trace consistency test passed")
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {traceback.format_exc()}", True)
    
    def test_monotonicity(self):
        """Test that ratio is invariant under scaling of x."""
        try:
            # Monotonicity: The ratio A1/A2 should be independent of x
            # We'll test by checking that if we scale x, the ratio remains the same
            
            if self.inputs['x'] is not None and self.results['fracA_1A_2'] is not None:
                x_val = self.inputs['x']
                
                # Test with scaled x values (if x is numeric)
                if isinstance(x_val, (int, float)):
                    # Test 1: Scale by factor 2
                    x_scaled = 2 * x_val
                    A1_scaled = abs(x_scaled)  # From physics: A1 = |x|
                    A2_scaled = abs(2 * x_scaled)  # A2 = |2x|
                    ratio_scaled = A1_scaled / A2_scaled
                    
                    if not self._is_close(ratio_scaled, self.results['fracA_1A_2']):
                        self._log(f"Monotonicity failed: ratio changed from {self.results['fracA_1A_2']} to {ratio_scaled} when scaling x", True)
                    else:
                        self._log("Monotonicity test passed: ratio is scale-invariant")
                
                # Symbolic test
                else:
                    # Define symbolic x
                    x_sym = sp.symbols('x', positive=True)
                    A1_sym = x_sym
                    A2_sym = 2 * x_sym
                    ratio_sym = A1_sym / A2_sym
                    
                    if ratio_sym != sp.Rational(1, 2):
                        self._log(f"Symbolic monotonicity failed: {ratio_sym} != 1/2", True)
                    else:
                        self._log("Symbolic monotonicity test passed")
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {traceback.format_exc()}", True)
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        self.success = True
        
        self.logs.append("Starting PhysicsChecker tests...")
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Final summary
        if self.success:
            self.logs.append("All tests passed successfully!")
        else:
            self.logs.append("Some tests failed.")
        
        return self.success, self.logs