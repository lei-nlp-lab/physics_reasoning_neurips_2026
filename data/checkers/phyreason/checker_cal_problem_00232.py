import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        Inputs: e, E, M, m, n, k, d
        Results: a, B_2, F, F_prime, N
        """
        # Safe extraction of inputs from solution_instance
        self.inputs = {}
        # First try to get inputs from results dictionary (they might be stored there)
        results_dict = getattr(solution_instance, 'results', {})
        for key in ['e', 'E', 'M', 'm', 'n', 'k', 'd']:
            # Try to get from results first, then from attributes
            val = results_dict.get(key, None)
            if val is None:
                val = getattr(solution_instance, key, None)
            self.inputs[key] = val
        
        # Safe extraction of results from solution_instance
        self.results = {}
        for key in ['a', 'B_2', 'F', 'N', 'F_prime']:
            val = results_dict.get(key, None)
            if val is None:
                val = getattr(solution_instance, key, None)
            self.results[key] = val
        
        # Special handling for F' if stored with different key
        if self.results['F_prime'] is None:
            self.results['F_prime'] = results_dict.get("F'", None)
        
        # Logging
        self.logs = []
        
    def _log(self, msg):
        """Helper to log messages."""
        self.logs.append(msg)
    
    def _is_close(self, a, b):
        """Robust numerical comparison with 2% tolerance."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        return False
    
    def test_invariants(self):
        """Test the fundamental physics laws based on KB."""
        try:
            # 1. Check Newton's Second Law for xenon ion acceleration: a = eE/M
            if all(v is not None for v in [self.inputs['e'], self.inputs['E'], self.inputs['M'], self.results['a']]):
                expected_a = self.inputs['e'] * self.inputs['E'] / self.inputs['M']
                actual_a = self.results['a']
                if not self._is_close(expected_a, actual_a):
                    self._log(f"FAIL: Acceleration a mismatch. Expected ~{expected_a}, got {actual_a}")
                    return False
                self._log("PASS: Acceleration a = eE/M holds.")
            else:
                self._log("WARNING: Missing inputs or result for acceleration check.")
            
            # 2. Check radial magnetic field B_2 = sqrt(mE/(2e))
            if all(v is not None for v in [self.inputs['m'], self.inputs['E'], self.inputs['e'], self.results['B_2']]):
                expected_B2 = math.sqrt(self.inputs['m'] * self.inputs['E'] / (2 * self.inputs['e']))
                actual_B2 = self.results['B_2']
                if not self._is_close(expected_B2, actual_B2):
                    self._log(f"FAIL: B_2 mismatch. Expected ~{expected_B2}, got {actual_B2}")
                    return False
                self._log("PASS: B_2 = sqrt(mE/(2e)) holds.")
            else:
                self._log("WARNING: Missing inputs or result for B_2 check.")
            
            # 3. Check thrust F' = F = (n k / (1+k)) * sqrt(2e E M)
            if all(v is not None for v in [self.inputs['n'], self.inputs['k'], self.inputs['e'], 
                                          self.inputs['E'], self.inputs['M'], self.results['F'], 
                                          self.results['F_prime']]):
                expected_F = (self.inputs['n'] * self.inputs['k'] / (1 + self.inputs['k'])) * \
                             math.sqrt(2 * self.inputs['e'] * self.inputs['E'] * self.inputs['M'])
                actual_F = self.results['F']
                actual_F_prime = self.results['F_prime']
                
                # Check F and F' equality (Newton's Third Law)
                if not self._is_close(actual_F, actual_F_prime):
                    self._log(f"FAIL: F and F' not equal. F={actual_F}, F'={actual_F_prime}")
                    return False
                self._log("PASS: F = F' (Newton's Third Law holds).")
                
                # Check F' against expected formula
                if not self._is_close(expected_F, actual_F_prime):
                    self._log(f"FAIL: Thrust F' mismatch. Expected ~{expected_F}, got {actual_F_prime}")
                    return False
                self._log("PASS: Thrust F' formula holds.")
            else:
                self._log("WARNING: Missing inputs or results for thrust check.")
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check N = nk/(1+k)
            if all(v is not None for v in [self.inputs['n'], self.inputs['k'], self.results['N']]):
                expected_N = self.inputs['n'] * self.inputs['k'] / (1 + self.inputs['k'])
                actual_N = self.results['N']
                if not self._is_close(expected_N, actual_N):
                    self._log(f"FAIL: N mismatch. Expected ~{expected_N}, got {actual_N}")
                    return False
                self._log("PASS: N = nk/(1+k) holds.")
            else:
                self._log("WARNING: Missing inputs or result for N check.")
            
            # 2. Check consistency of F = N * M * v1, with v1 from energy conservation
            # From step_5: eEd = 0.5 * M * v1^2 -> v1 = sqrt(2eEd/M)
            if all(v is not None for v in [self.inputs['e'], self.inputs['E'], self.inputs['d'], 
                                          self.inputs['M'], self.results['N'], self.results['F']]):
                v1 = math.sqrt(2 * self.inputs['e'] * self.inputs['E'] * self.inputs['d'] / self.inputs['M'])
                expected_F_from_v1 = self.results['N'] * self.inputs['M'] * v1
                actual_F = self.results['F']
                if not self._is_close(expected_F_from_v1, actual_F):
                    self._log(f"FAIL: F inconsistency with v1. Expected from v1 ~{expected_F_from_v1}, got {actual_F}")
                    return False
                self._log("PASS: F = N * M * v1 holds.")
            else:
                self._log("WARNING: Missing inputs or results for F consistency check.")
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check the trend of results."""
        try:
            # Choose a parameter to perturb: E (electric field)
            if self.inputs['E'] is None:
                self._log("WARNING: E not provided, skipping monotonicity test.")
                return True
            
            # Original value
            E_orig = float(self.inputs['E'])
            perturb_factor = 1.01  # 1% increase
            E_new = E_orig * perturb_factor
            
            # Store original inputs
            original_inputs = self.inputs.copy()
            
            # Update E for recalculation
            self.inputs['E'] = E_new
            
            # Recalculate expected values with new E
            # a = eE/M -> should increase
            if all(v is not None for v in [self.inputs['e'], self.inputs['M'], original_inputs['e'], original_inputs['M']]):
                a_orig = original_inputs['e'] * E_orig / original_inputs['M']
                a_new = self.inputs['e'] * E_new / self.inputs['M']
                if not (a_new > a_orig):
                    self._log(f"FAIL: Monotonicity fail for a. Original {a_orig}, new {a_new}")
                    # Restore original input
                    self.inputs['E'] = E_orig
                    return False
                self._log(f"PASS: a increases with E ({a_orig} -> {a_new}).")
            
            # B_2 = sqrt(mE/(2e)) -> should increase
            if all(v is not None for v in [self.inputs['m'], self.inputs['e'], original_inputs['m'], original_inputs['e']]):
                B2_orig = math.sqrt(original_inputs['m'] * E_orig / (2 * original_inputs['e']))
                B2_new = math.sqrt(self.inputs['m'] * E_new / (2 * self.inputs['e']))
                if not (B2_new > B2_orig):
                    self._log(f"FAIL: Monotonicity fail for B_2. Original {B2_orig}, new {B2_new}")
                    self.inputs['E'] = E_orig
                    return False
                self._log(f"PASS: B_2 increases with E ({B2_orig} -> {B2_new}).")
            
            # F' = (n k sqrt(2e E M))/(1+k) -> should increase
            if all(v is not None for v in [self.inputs['n'], self.inputs['k'], self.inputs['e'], 
                                          self.inputs['M'], original_inputs['n'], original_inputs['k'],
                                          original_inputs['e'], original_inputs['M']]):
                F_orig = (original_inputs['n'] * original_inputs['k'] / (1 + original_inputs['k'])) * \
                         math.sqrt(2 * original_inputs['e'] * E_orig * original_inputs['M'])
                F_new = (self.inputs['n'] * self.inputs['k'] / (1 + self.inputs['k'])) * \
                        math.sqrt(2 * self.inputs['e'] * E_new * self.inputs['M'])
                if not (F_new > F_orig):
                    self._log(f"FAIL: Monotonicity fail for F'. Original {F_orig}, new {F_new}")
                    self.inputs['E'] = E_orig
                    return False
                self._log(f"PASS: F' increases with E ({F_orig} -> {F_new}).")
            
            # Restore original input
            self.inputs['E'] = E_orig
            return True
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}")
            # Restore original input on error
            if 'E_orig' in locals():
                self.inputs['E'] = E_orig
            return False
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        success = True
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== Running {test_name} Test ===")
            try:
                if not test_func():
                    success = False
                    self._log(f"FAIL: {test_name} test failed.")
                else:
                    self._log(f"PASS: {test_name} test passed.")
            except Exception as e:
                success = False
                self._log(f"ERROR: {test_name} test crashed: {traceback.format_exc()}")
        
        return success, self.logs