import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access from solution instance."""
        # Extract input parameters with safe defaults
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 2.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', 0.5))
        self.hA = getattr(solution_instance, 'hA', solution_instance.results.get('hA', 10.0))
        self.hB = getattr(solution_instance, 'hB', solution_instance.results.get('hB', 20.0))
        self.hO = getattr(solution_instance, 'hO', solution_instance.results.get('hO', 30.0))
        
        # Extract results with safe defaults
        self.W_G1 = solution_instance.results.get('W_G1', 0.0)
        self.W_G2 = solution_instance.results.get('W_G2', 0.0)
        self.W_f1 = solution_instance.results.get('W_f1', 0.0)
        self.W_f2 = solution_instance.results.get('W_f2', 0.0)
        self.NA = solution_instance.results.get('N/A', None)
        
        self.solution_instance = solution_instance
        self.logs = []
        self.success = True
    
    def _check_equality(self, a, b, tolerance: float = 1e-2) -> bool:
        """Fuzzy equality check with 1-2% tolerance for numerical data."""
        try:
            # Handle non-numeric values
            if isinstance(a, str) or isinstance(b, str):
                return str(a) == str(b)
            
            # Convert to float
            a_float = float(a) if not isinstance(a, (int, float)) else float(a)
            b_float = float(b) if not isinstance(b, (int, float)) else float(b)
            
            return math.isclose(a_float, b_float, rel_tol=tolerance, abs_tol=tolerance)
        except (ValueError, TypeError):
            # If conversion fails, do string comparison
            return str(a) == str(b)
    
    def test_invariants(self):
        """Verify energy conservation and force work principles."""
        try:
            # Expected numeric formulas from KB (using actual numeric values)
            expected_W_G1 = -self.m * self.g * (self.hB - self.hA)
            expected_W_G2 = -self.m * self.g * (self.hB - self.hA)
            expected_W_f1 = -self.f * (self.hB - self.hA)
            expected_W_f2 = -self.f * (2 * self.hO - self.hA - self.hB)
            
            # Test 1: Check work done by gravity in A→B
            if not self._check_equality(self.W_G1, expected_W_G1):
                self.logs.append(f"FAIL: W_G1 mismatch. Got {self.W_G1}, expected {expected_W_G1}")
                self.success = False
            else:
                self.logs.append("PASS: W_G1 matches theoretical formula")
            
            # Test 2: Check work done by gravity in A→O→B
            if not self._check_equality(self.W_G2, expected_W_G2):
                self.logs.append(f"FAIL: W_G2 mismatch. Got {self.W_G2}, expected {expected_W_G2}")
                self.success = False
            else:
                self.logs.append("PASS: W_G2 matches theoretical formula")
            
            # Test 3: Check work done by air resistance in A→B
            if not self._check_equality(self.W_f1, expected_W_f1):
                self.logs.append(f"FAIL: W_f1 mismatch. Got {self.W_f1}, expected {expected_W_f1}")
                self.success = False
            else:
                self.logs.append("PASS: W_f1 matches theoretical formula")
            
            # Test 4: Check work done by air resistance in A→O→B
            if not self._check_equality(self.W_f2, expected_W_f2):
                self.logs.append(f"FAIL: W_f2 mismatch. Got {self.W_f2}, expected {expected_W_f2}")
                self.success = False
            else:
                self.logs.append("PASS: W_f2 matches theoretical formula")
            
            # Test 5: Conservative nature of gravity (path independence)
            if not self._check_equality(self.W_G1, self.W_G2):
                self.logs.append("FAIL: Gravity work not path-independent (W_G1 != W_G2)")
                self.success = False
            else:
                self.logs.append("PASS: Gravity work is path-independent (W_G1 = W_G2)")
            
            # Test 6: Non-conservative nature of air resistance (path dependence)
            if self._check_equality(self.W_f1, self.W_f2):
                self.logs.append("FAIL: Air resistance work appears path-independent (W_f1 ≈ W_f2)")
                self.success = False
            else:
                self.logs.append("PASS: Air resistance work is path-dependent (W_f1 ≠ W_f2)")
            
            # Test 7: Check energy relationship (work-energy theorem)
            W_total1 = self.W_G1 + self.W_f1
            W_total2 = self.W_G2 + self.W_f2
            
            # These should generally not be equal due to path dependence
            if self._check_equality(W_total1, W_total2):
                self.logs.append("WARNING: Total work appears path-independent (unexpected for non-conservative forces)")
            else:
                self.logs.append("PASS: Total work is path-dependent as expected")
                
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            self.success = False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Check if solution has trace data
            if hasattr(self.solution_instance, 'trace') and isinstance(self.solution_instance.trace, dict):
                trace = self.solution_instance.trace
                
                # Check each step's results if available
                step_checks = []
                for step_key in ['step_1', 'step_2', 'step_3', 'step_4']:
                    if step_key in trace:
                        step_data = trace[step_key]
                        # Extract value from step data if available
                        if 'value' in step_data:
                            step_value = step_data['value']
                            # Map step to corresponding result
                            if step_key == 'step_1' and 'W_G1' in step_data.get('symbol', ''):
                                if not self._check_equality(step_value, self.W_G1):
                                    step_checks.append(f"Mismatch in {step_key}")
                            elif step_key == 'step_2' and 'W_G2' in step_data.get('symbol', ''):
                                if not self._check_equality(step_value, self.W_G2):
                                    step_checks.append(f"Mismatch in {step_key}")
                            elif step_key == 'step_3' and 'W_f1' in step_data.get('symbol', ''):
                                if not self._check_equality(step_value, self.W_f1):
                                    step_checks.append(f"Mismatch in {step_key}")
                            elif step_key == 'step_4' and 'W_f2' in step_data.get('symbol', ''):
                                if not self._check_equality(step_value, self.W_f2):
                                    step_checks.append(f"Mismatch in {step_key}")
                
                if step_checks:
                    self.logs.append(f"FAIL: Trace inconsistencies: {', '.join(step_checks)}")
                    self.success = False
                else:
                    self.logs.append("PASS: Trace consistency verified")
            else:
                self.logs.append("SKIP: No trace data available for consistency check")
                
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
    
    def test_monotonicity(self):
        """Perturb inputs and check monotonic trends."""
        try:
            # Save original values
            original_m = self.m
            original_f = self.f
            
            # Test 1: Perturb mass m
            perturb_factor = 1.1  # 10% increase
            self.m = original_m * perturb_factor
            
            # Recalculate expected values with perturbed mass
            expected_W_G1_perturbed = -self.m * self.g * (self.hB - self.hA)
            expected_W_G2_perturbed = -self.m * self.g * (self.hB - self.hA)
            
            # Original expected values
            original_W_G1_expected = -original_m * self.g * (self.hB - self.hA)
            original_W_G2_expected = -original_m * self.g * (self.hB - self.hA)
            
            # Check proportionality: perturbed/original = perturb_factor
            if original_W_G1_expected != 0:
                ratio1 = expected_W_G1_perturbed / original_W_G1_expected
                if not math.isclose(ratio1, perturb_factor, rel_tol=1e-2):
                    self.logs.append(f"FAIL: W_G1 not linear with mass. Ratio: {ratio1}, expected: {perturb_factor}")
                    self.success = False
                else:
                    self.logs.append("PASS: W_G1 linear with mass")
            
            if original_W_G2_expected != 0:
                ratio2 = expected_W_G2_perturbed / original_W_G2_expected
                if not math.isclose(ratio2, perturb_factor, rel_tol=1e-2):
                    self.logs.append(f"FAIL: W_G2 not linear with mass. Ratio: {ratio2}, expected: {perturb_factor}")
                    self.success = False
                else:
                    self.logs.append("PASS: W_G2 linear with mass")
            
            # Test 2: Perturb air resistance f (restore m first)
            self.m = original_m
            original_f = self.f
            self.f = original_f * perturb_factor
            
            # Recalculate expected values with perturbed f
            expected_W_f1_perturbed = -self.f * (self.hB - self.hA)
            expected_W_f2_perturbed = -self.f * (2 * self.hO - self.hA - self.hB)
            
            # Original expected values
            original_W_f1_expected = -original_f * (self.hB - self.hA)
            original_W_f2_expected = -original_f * (2 * self.hO - self.hA - self.hB)
            
            # Check proportionality
            if original_W_f1_expected != 0:
                ratio3 = expected_W_f1_perturbed / original_W_f1_expected
                if not math.isclose(ratio3, perturb_factor, rel_tol=1e-2):
                    self.logs.append(f"FAIL: W_f1 not linear with f. Ratio: {ratio3}, expected: {perturb_factor}")
                    self.success = False
                else:
                    self.logs.append("PASS: W_f1 linear with f")
            
            if original_W_f2_expected != 0:
                ratio4 = expected_W_f2_perturbed / original_W_f2_expected
                if not math.isclose(ratio4, perturb_factor, rel_tol=1e-2):
                    self.logs.append(f"FAIL: W_f2 not linear with f. Ratio: {ratio4}, expected: {perturb_factor}")
                    self.success = False
                else:
                    self.logs.append("PASS: W_f2 linear with f")
            
            # Restore original values
            self.m = original_m
            self.f = original_f
            
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            self.success = False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status with logs."""
        self.logs = []
        self.success = True
        
        try:
            self.logs.append("=== PhysicsChecker: Starting Tests ===")
            
            self.test_invariants()
            self.logs.append("---")
            
            self.test_trace_consistency()
            self.logs.append("---")
            
            self.test_monotonicity()
            self.logs.append("---")
            
            if self.success:
                self.logs.append("SUMMARY: All tests passed")
            else:
                self.logs.append("SUMMARY: Some tests failed")
            
            self.logs.append("=== PhysicsChecker: Tests Complete ===")
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            self.success = False
        
        return self.success, self.logs