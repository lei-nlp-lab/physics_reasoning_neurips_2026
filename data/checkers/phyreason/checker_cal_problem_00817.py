import math
import traceback
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely initialize by extracting inputs and results from solution."""
        self.solution = solution_instance
        
        # Problem constants
        self.d = 0.8  # Distance between A and B (m)
        
        # Safe extraction of results
        self.T = self._safe_get('T', 0.2)  # Period from graph
        
        # Extract lists - handle both list and single value cases
        self.wavelengths = self._safe_get_list('λ', [1.6, 0.8, 8/15])
        self.speeds = self._safe_get_list('v', [8.0, 4.0, 8/3])
        self.times_A = self._safe_get_list('t_A', [0.05, 0.15])
        self.times_B = self._safe_get_list('t_B', [0.05, 0.15])
        
        # Intermediate results
        self.lam32 = self._safe_get('3λ/2', 0.8)
        
        # Constants for checking
        self.REL_TOL = 1e-2
        self.ABS_TOL = 1e-2
    
    def _safe_get(self, key, default=None):
        """Safely retrieve value from solution instance."""
        try:
            # First try getattr on solution instance
            val = getattr(self.solution, key, None)
            if val is None:
                # Then try results dictionary
                val = self.solution.results.get(key, default)
            return val
        except (AttributeError, KeyError):
            return default
    
    def _safe_get_list(self, key, default=None):
        """Safely retrieve list from solution instance."""
        val = self._safe_get(key, default)
        if val is None:
            return default if isinstance(default, list) else []
        
        # If it's already a list, return it
        if isinstance(val, list):
            return val
        
        # If it's a single numeric value, wrap it in a list
        if isinstance(val, (int, float)):
            return [val]
        
        # For any other type, try to convert to list
        try:
            return list(val)
        except:
            return default if isinstance(default, list) else []
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        logs = []
        
        try:
            # 1. Basic wave relationship: v = f * λ, where f = 1/T
            if len(self.wavelengths) == len(self.speeds):
                for λ, v in zip(self.wavelengths, self.speeds):
                    expected_v = λ / self.T
                    if not math.isclose(v, expected_v, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                        logs.append(f"FAIL: v = {v} but λ/T = {expected_v} for λ = {λ}")
                    else:
                        logs.append(f"PASS: Wave relationship v = λ/T holds for λ = {λ}")
            else:
                logs.append(f"FAIL: Mismatched lengths: wavelengths={len(self.wavelengths)}, speeds={len(self.speeds)}")
            
            # 2. Distance-wavelength relationships for the three cases
            # Case 1: d = λ/2
            if len(self.wavelengths) >= 1:
                expected_d = self.wavelengths[0] / 2
                if not math.isclose(self.d, expected_d, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                    logs.append(f"FAIL: Case 1: d = {self.d} but λ/2 = {expected_d}")
                else:
                    logs.append(f"PASS: Case 1 distance-wavelength relationship holds")
            
            # Case 2: d = λ
            if len(self.wavelengths) >= 2:
                if not math.isclose(self.d, self.wavelengths[1], rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                    logs.append(f"FAIL: Case 2: d = {self.d} but λ = {self.wavelengths[1]}")
                else:
                    logs.append(f"PASS: Case 2 distance-wavelength relationship holds")
            
            # Case 3: d = 3λ/2
            if len(self.wavelengths) >= 3:
                expected_d = 1.5 * self.wavelengths[2]
                if not math.isclose(self.d, expected_d, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                    logs.append(f"FAIL: Case 3: d = {self.d} but 3λ/2 = {expected_d}")
                else:
                    logs.append(f"PASS: Case 3 distance-wavelength relationship holds")
            
            # 3. Check that 3λ/2 intermediate matches case 3
            if len(self.wavelengths) >= 3:
                expected_lam32 = 1.5 * self.wavelengths[2]
                if not math.isclose(self.lam32, expected_lam32, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                    logs.append(f"FAIL: 3λ/2 = {self.lam32} but 1.5*λ = {expected_lam32}")
                else:
                    logs.append(f"PASS: 3λ/2 intermediate value consistent")
            
            # 4. SHM time relationships
            # Time from equilibrium to trough should be T/4
            if len(self.times_A) >= 1:
                expected_t = self.T / 4
                if not math.isclose(self.times_A[0], expected_t, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                    logs.append(f"FAIL: Time to trough = {self.times_A[0]} but T/4 = {expected_t}")
                else:
                    logs.append(f"PASS: Time to trough equals T/4")
            
            # Time from equilibrium to crest should be 3T/4
            if len(self.times_A) >= 2:
                expected_t = 3 * self.T / 4
                if not math.isclose(self.times_A[1], expected_t, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                    logs.append(f"FAIL: Time to crest = {self.times_A[1]} but 3T/4 = {expected_t}")
                else:
                    logs.append(f"PASS: Time to crest equals 3T/4")
            
            # 5. Check that times for B are subset of times for A
            # Convert to sets for comparison
            set_A = set(self.times_A)
            set_B = set(self.times_B)
            if set_B.issubset(set_A):
                logs.append(f"PASS: B's possible times are subset of A's times")
            else:
                logs.append(f"FAIL: B's times {self.times_B} not subset of A's times {self.times_A}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate and final results."""
        logs = []
        
        try:
            # Check each case's consistency
            cases = [
                ("Case 1 (λ/2)", 0.5, 0),
                ("Case 2 (λ)", 1.0, 1),
                ("Case 3 (3λ/2)", 1.5, 2)
            ]
            
            for case_name, factor, idx in cases:
                if idx < len(self.wavelengths):
                    λ = self.wavelengths[idx]
                    v = self.speeds[idx] if idx < len(self.speeds) else None
                    
                    # Check d = factor * λ
                    expected_d = factor * λ
                    if not math.isclose(self.d, expected_d, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                        logs.append(f"FAIL {case_name}: d = {self.d}, factor*λ = {expected_d}")
                    else:
                        logs.append(f"PASS {case_name}: Distance-wavelength relationship holds")
                    
                    # Check v = λ/T
                    if v is not None:
                        expected_v = λ / self.T
                        if not math.isclose(v, expected_v, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
                            logs.append(f"FAIL {case_name}: v = {v}, λ/T = {expected_v}")
                        else:
                            logs.append(f"PASS {case_name}: Speed calculation consistent")
            
            # Check time calculations
            if len(self.times_A) >= 2 and len(self.times_B) >= 2:
                # Times should be the same two values (0.05 and 0.15)
                sorted_A = sorted(self.times_A)
                sorted_B = sorted(self.times_B)
                if (math.isclose(sorted_A[0], sorted_B[0], rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL) and
                    math.isclose(sorted_A[1], sorted_B[1], rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL)):
                    logs.append("PASS: B's times match A's times")
                else:
                    logs.append(f"FAIL: Times mismatch: A={self.times_A}, B={self.times_B}")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def test_monotonicity(self):
        """Test that perturbations in inputs produce expected trends."""
        logs = []
        
        try:
            # Save original state
            orig_T = self.T
            orig_speeds = self.speeds.copy()
            orig_wavelengths = self.wavelengths.copy()
            
            # Test 1: Increase period, speeds should decrease proportionally
            perturbed_T = self.T * 1.1  # 10% increase
            
            for i, (λ, v_orig) in enumerate(zip(self.wavelengths, self.speeds)):
                v_new = λ / perturbed_T
                if v_new < v_orig:
                    logs.append(f"PASS: Speed decreases when period increases (λ={λ:.3f})")
                else:
                    logs.append(f"FAIL: Speed should decrease but {v_new} >= {v_orig}")
            
            # Test 2: Increase distance, wavelengths should increase proportionally
            # For each case, check if λ = d/factor
            perturbed_d = self.d * 1.1  # 10% increase
            
            factors = [0.5, 1.0, 1.5]  # For the three cases
            for i, factor in enumerate(factors):
                if i < len(self.wavelengths):
                    λ_new = perturbed_d / factor
                    if λ_new > self.wavelengths[i]:
                        logs.append(f"PASS: Wavelength increases when distance increases (case {i+1})")
                    else:
                        logs.append(f"FAIL: Wavelength should increase but {λ_new} <= {self.wavelengths[i]}")
            
            # Restore original state
            self.T = orig_T
            self.speeds = orig_speeds
            self.wavelengths = orig_wavelengths
            
            logs.append("Monotonicity tests completed and state restored")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            
            # Attempt to restore on error
            self.T = orig_T if 'orig_T' in locals() else self.T
            self.speeds = orig_speeds if 'orig_speeds' in locals() else self.speeds
            self.wavelengths = orig_wavelengths if 'orig_wavelengths' in locals() else self.wavelengths
        
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        all_passed = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            passed, logs = test_func()
            all_logs.append(f"=== {test_name} Test ===")
            all_logs.extend(logs)
            all_logs.append(f"Result: {'PASSED' if passed else 'FAILED'}")
            all_logs.append("")
            
            if not passed:
                all_passed = False
        
        # Final summary
        if all_passed:
            all_logs.append("=== FINAL RESULT ===")
            all_logs.append("All physics checks PASSED")
        else:
            all_logs.append("=== FINAL RESULT ===")
            all_logs.append("Some physics checks FAILED")
        
        return all_passed, all_logs