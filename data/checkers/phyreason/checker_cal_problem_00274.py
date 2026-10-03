import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        SAFE initialization: use getattr or .get() to avoid AttributeError.
        Extract all needed variables from the solution instance.
        """
        self.solution = solution_instance
        
        # Constants from problem statement
        self.H = 5.4  # m (height of suspension point)
        self.g = 10.0  # m/s² (gravitational acceleration)
        
        # Extract results with safe fallbacks
        self.l = self._safe_get('l', 0.0)            # string length
        self.t = self._safe_get('t', 0.0)            # free fall time until taut
        self.v = self._safe_get('v', 0.0)            # velocity when string becomes taut
        self.v_prime = self._safe_get('v_prime', 0.0)  # velocity after break
        self.l_prime = self._safe_get('l_prime', 0.0)  # distance fallen after break
        self.t_total = self._safe_get('t_total', 0.0)  # total time
        
        # Initialize symbolic variables for invariant checking
        self.sym_l, self.sym_t, self.sym_v, self.sym_vp = sp.symbols('l t v vp', positive=True)
        self.sym_g, self.sym_H = sp.symbols('g H', positive=True)
        
        # Build logs
        self.logs = []
    
    def _safe_get(self, key, default):
        """Safely retrieve attribute from solution instance."""
        try:
            # First try direct attribute
            val = getattr(self.solution, key, None)
            if val is not None:
                return float(val)
            # Then try results dictionary
            if hasattr(self.solution, 'results'):
                val = self.solution.results.get(key, default)
                return float(val)
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            self.logs.append(f"Warning: Could not retrieve {key}: {str(e)}")
        return default
    
    def _log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
        return success
    
    def test_invariants(self):
        """Verify fundamental physical laws."""
        checks = []
        
        # 1. Check free-fall kinematics: v^2 = 2*g*l (when string becomes taut)
        lhs = self.v**2
        rhs = 2 * self.g * self.l
        check1 = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"v² = 2gl: {lhs:.3f} ≈ {rhs:.3f}", check1))
        
        # 2. Check time-distance relation: l = (1/2)*g*t² (free fall until taut)
        lhs = self.l
        rhs = 0.5 * self.g * self.t**2
        check2 = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"l = ½gt²: {lhs:.3f} ≈ {rhs:.3f}", check2))
        
        # 3. Check velocity ratio: v' = v/2 (given condition)
        lhs = self.v_prime
        rhs = self.v / 2
        check3 = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"v' = v/2: {lhs:.3f} ≈ {rhs:.3f}", check3))
        
        # 4. Check geometry: l + l' = H (total vertical distance)
        total_dist = self.l + self.l_prime
        check4 = math.isclose(total_dist, self.H, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"l + l' = H: {total_dist:.3f} ≈ {self.H:.3f}", check4))
        
        # 5. Check kinematics after break: l' = v'*t + (1/2)*g*t² (same t as before break)
        lhs = self.l_prime
        rhs = self.v_prime * self.t + 0.5 * self.g * self.t**2
        check5 = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"l' = v't + ½gt²: {lhs:.3f} ≈ {rhs:.3f}", check5))
        
        # 6. Check total time: t_total = 2*t (given)
        lhs = self.t_total
        rhs = 2 * self.t
        check6 = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"t_total = 2t: {lhs:.3f} ≈ {rhs:.3f}", check6))
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        checks = []
        
        # 1. Derive l from H using step 7: l = H/3
        l_from_H = self.H / 3
        check1 = math.isclose(self.l, l_from_H, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"l = H/3: {self.l:.3f} ≈ {l_from_H:.3f}", check1))
        
        # 2. Derive v from l: v = sqrt(2*g*l)
        v_from_l = math.sqrt(2 * self.g * self.l)
        check2 = math.isclose(self.v, v_from_l, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"v = √(2gl): {self.v:.3f} ≈ {v_from_l:.3f}", check2))
        
        # 3. Derive t from l: t = sqrt(2*l/g)
        t_from_l = math.sqrt(2 * self.l / self.g)
        check3 = math.isclose(self.t, t_from_l, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"t = √(2l/g): {self.t:.3f} ≈ {t_from_l:.3f}", check3))
        
        # 4. Derive l' from v' and t: l' = v'*t + (1/2)*g*t²
        l_prime_calc = self.v_prime * self.t + 0.5 * self.g * self.t**2
        check4 = math.isclose(self.l_prime, l_prime_calc, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"l' from kinematics: {self.l_prime:.3f} ≈ {l_prime_calc:.3f}", check4))
        
        # 5. Alternative: check that l' = 2*l (from step 6)
        check5 = math.isclose(self.l_prime, 2 * self.l, rel_tol=1e-2, abs_tol=1e-2)
        checks.append(self._log(f"l' = 2l: {self.l_prime:.3f} ≈ {2*self.l:.3f}", check5))
        
        return all(checks)
    
    def test_monotonicity(self):
        """Perturb H and verify expected trends in l and t_total."""
        original_H = self.H
        original_l = self.l
        original_t_total = self.t_total
        
        try:
            # Perturb H by +10%
            perturbed_H = original_H * 1.10
            
            # Expected behavior: l ∝ H, t_total ∝ √H
            expected_l = perturbed_H / 3
            expected_t = math.sqrt(2 * expected_l / self.g)
            expected_t_total = 2 * expected_t
            
            # Check monotonicity: increasing H should increase l and t_total
            check_l = expected_l > original_l
            check_t_total = expected_t_total > original_t_total
            
            self._log(f"Perturbed H from {original_H:.2f} to {perturbed_H:.2f}", None)
            self._log(f"Expected l increases: {original_l:.3f} → {expected_l:.3f}", check_l)
            self._log(f"Expected t_total increases: {original_t_total:.3f} → {expected_t_total:.3f}", check_t_total)
            
            # Also verify the proportional relationships hold
            ratio_H = perturbed_H / original_H
            ratio_l = expected_l / original_l
            check_ratio_l = math.isclose(ratio_l, ratio_H, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"l scales linearly with H: ratio {ratio_l:.3f} ≈ {ratio_H:.3f}", check_ratio_l)
            
            # t_total ∝ √H
            ratio_t_total = expected_t_total / original_t_total
            expected_sqrt_ratio = math.sqrt(ratio_H)
            check_ratio_t = math.isclose(ratio_t_total, expected_sqrt_ratio, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"t_total scales as √H: {ratio_t_total:.3f} ≈ √{ratio_H:.3f}={expected_sqrt_ratio:.3f}", check_ratio_t)
            
            return all([check_l, check_t_total, check_ratio_l, check_ratio_t])
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
        finally:
            # Restore original H (though it's not modified in solution_instance)
            pass
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        # Collect individual test results
        results = []
        try:
            self.logs.append("=== Physics Checker Start ===")
            
            # Test 1: Invariants
            self.logs.append("\n--- Testing Physical Invariants ---")
            results.append(self.test_invariants())
            
            # Test 2: Trace Consistency
            self.logs.append("\n--- Testing Trace Consistency ---")
            results.append(self.test_trace_consistency())
            
            # Test 3: Monotonicity
            self.logs.append("\n--- Testing Monotonicity ---")
            results.append(self.test_monotonicity())
            
            # Final summary
            success = all(results)
            self.logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ({sum(results)}/3 tests passed) ===")
            
        except Exception as e:
            self.logs.append(f"ERROR in test execution: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        return success, self.logs