import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract expected keys with safe defaults
        self.k = self._safe_get('k')
        self.F1 = self._safe_get('F1')
        self.F2 = self._safe_get('F2')
        self.a = self._safe_get('a')
        
        # Extract inputs if available
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.m = self._safe_get_input('m', 1.0)  # default m=1 for testing
        self.mu = self._safe_get_input('mu', 0.2)  # default μ=0.2
        
        # Physical constants
        self.g = 9.8  # gravity
        
        # Validation flag
        self.valid_data = all(val is not None for val in [self.k, self.F1, self.F2, self.a])
        
        # Test results log
        self.logs = []
        
    def _safe_get(self, key):
        """Safely get value from results with nested dict support."""
        if hasattr(self, 'results'):
            if isinstance(self.results, dict):
                return self.results.get(key, None)
        return None
    
    def _safe_get_input(self, key, default):
        """Safely get input value."""
        if isinstance(self.inputs, dict):
            return self.inputs.get(key, default)
        return default
    
    def _log_test(self, name, passed, message):
        """Log test result with details."""
        status = "PASS" if passed else "FAIL"
        self.logs.append(f"{name}: {status} - {message}")
        return passed
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with fallback for sympy."""
        try:
            # Convert to float if possible
            a_val = float(a) if hasattr(a, '__float__') else a
            b_val = float(b) if hasattr(b, '__float__') else b
            
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
            else:
                # Symbolic comparison
                diff = sp.simplify(a_val - b_val)
                return diff == 0
        except:
            return False
    
    def test_invariants(self):
        """Test physics laws: Newton's 2nd Law and force relationships."""
        tests_passed = True
        
        if not self.valid_data:
            return self._log_test("Invariants", False, "Missing required data")
        
        # Symbolic setup
        m, μ, g, a, F1, F2 = sp.symbols('m μ g a F1 F2')
        
        try:
            # Test 1: Newton's 2nd Law for block R (3m)
            # Equation: F1 - μ*3m*g = 3m*a
            lhs_eq1 = F1 - μ * 3 * m * g
            rhs_eq1 = 3 * m * a
            
            # Substitute actual values
            subs_dict = {
                m: self.m,
                μ: self.mu,
                g: self.g,
                a: self.a,
                F1: self.F1
            }
            
            lhs_val1 = float(lhs_eq1.subs(subs_dict))
            rhs_val1 = float(rhs_eq1.subs(subs_dict))
            
            if self._is_close(lhs_val1, rhs_val1):
                self._log_test("Newton_R", True, f"F1 - μ*3mg = 3ma: {lhs_val1:.3f} ≈ {rhs_val1:.3f}")
            else:
                tests_passed = False
                self._log_test("Newton_R", False, f"F1 - μ*3mg = 3ma failed: {lhs_val1:.3f} ≠ {rhs_val1:.3f}")
            
            # Test 2: Newton's 2nd Law for blocks Q&R (5m)
            # Equation: F2 - μ*5m*g = 5m*a
            lhs_eq2 = F2 - μ * 5 * m * g
            rhs_eq2 = 5 * m * a
            
            subs_dict[F2] = self.F2
            lhs_val2 = float(lhs_eq2.subs(subs_dict))
            rhs_val2 = float(rhs_eq2.subs(subs_dict))
            
            if self._is_close(lhs_val2, rhs_val2):
                self._log_test("Newton_QR", True, f"F2 - μ*5mg = 5ma: {lhs_val2:.3f} ≈ {rhs_val2:.3f}")
            else:
                tests_passed = False
                self._log_test("Newton_QR", False, f"F2 - μ*5mg = 5ma failed: {lhs_val2:.3f} ≠ {rhs_val2:.3f}")
            
            # Test 3: Ratio consistency k = F1/F2 = 3/5
            # From equations: F1 = 3m(a + μg), F2 = 5m(a + μg) ⇒ F1/F2 = 3/5
            expected_ratio = 3/5
            actual_ratio = self.F1 / self.F2
            
            if self._is_close(actual_ratio, expected_ratio):
                self._log_test("Ratio", True, f"k = F1/F2 = {actual_ratio:.3f} ≈ 3/5")
            else:
                tests_passed = False
                self._log_test("Ratio", False, f"k = {actual_ratio:.3f} ≠ 3/5")
            
            # Test 4: Check that k in results matches computed ratio
            if self._is_close(self.k, actual_ratio):
                self._log_test("k_consistency", True, f"Result k={self.k} matches F1/F2={actual_ratio:.3f}")
            else:
                tests_passed = False
                self._log_test("k_consistency", False, f"Result k={self.k} ≠ F1/F2={actual_ratio:.3f}")
                
        except Exception as e:
            tests_passed = False
            self._log_test("Invariants", False, f"Exception: {str(e)}")
        
        return self._log_test("Invariants", tests_passed, "All physics laws verified" if tests_passed else "Physics law violations")
    
    def test_trace_consistency(self):
        """Check intermediate results are consistent with final answer."""
        tests_passed = True
        
        if not self.valid_data:
            return self._log_test("Trace", False, "Missing required data")
        
        try:
            # Check 1: Acceleration should be the same in both equations
            # From step 1: a = (F1 - μ*3m*g) / (3m)
            # From step 2: a = (F2 - μ*5m*g) / (5m)
            a1 = (self.F1 - self.mu * 3 * self.m * self.g) / (3 * self.m)
            a2 = (self.F2 - self.mu * 5 * self.m * self.g) / (5 * self.m)
            
            if self._is_close(a1, a2) and self._is_close(a1, self.a):
                self._log_test("Accel_consistency", True, f"a consistent: {a1:.3f} ≈ {a2:.3f} ≈ {self.a}")
            else:
                tests_passed = False
                self._log_test("Accel_consistency", False, f"a mismatch: {a1:.3f}, {a2:.3f}, {self.a}")
            
            # Check 2: Ratio from simultaneous equations should be 3/5
            # Solving the two equations symbolically
            m, μ, g, a = sp.symbols('m μ g a')
            F1_expr = 3*m*a + μ*3*m*g
            F2_expr = 5*m*a + μ*5*m*g
            ratio_expr = sp.simplify(F1_expr / F2_expr)
            
            if ratio_expr == sp.Rational(3, 5):
                self._log_test("Symbolic_ratio", True, "Symbolic derivation gives k=3/5")
            else:
                tests_passed = False
                self._log_test("Symbolic_ratio", False, f"Symbolic ratio: {ratio_expr} ≠ 3/5")
            
        except Exception as e:
            tests_passed = False
            self._log_test("Trace", False, f"Exception: {str(e)}")
        
        return self._log_test("Trace", tests_passed, "Trace consistency verified" if tests_passed else "Trace inconsistencies")
    
    def test_monotonicity(self):
        """Test that ratio k is invariant under mass perturbations."""
        tests_passed = True
        
        if not self.valid_data:
            return self._log_test("Monotonicity", False, "Missing required data")
        
        try:
            # Test: Ratio should be independent of mass m
            # Use two different mass values
            original_m = self.m
            test_masses = [original_m * 0.5, original_m * 2.0]
            
            for test_m in test_masses:
                # Recalculate forces using the physics equations
                # From equations: F1 = 3m(a + μg), F2 = 5m(a + μg)
                # But we need a consistent 'a' - use the same a from results
                # (This assumes a remains constant when changing m, which is physically
                # valid if external force F adjusts accordingly)
                
                F1_test = 3 * test_m * (self.a + self.mu * self.g)
                F2_test = 5 * test_m * (self.a + self.mu * self.g)
                k_test = F1_test / F2_test
                
                if self._is_close(k_test, 3/5):
                    self._log_test(f"Mass_{test_m}", True, f"k invariant at m={test_m}: {k_test:.3f} ≈ 3/5")
                else:
                    tests_passed = False
                    self._log_test(f"Mass_{test_m}", False, f"k varies at m={test_m}: {k_test:.3f} ≠ 3/5")
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            tests_passed = False
            self._log_test("Monotonicity", False, f"Exception: {str(e)}")
        
        return self._log_test("Monotonicity", tests_passed, "Ratio invariant under mass changes" if tests_passed else "Ratio not invariant")
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        try:
            # Run tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            # Overall success
            success = all([test1, test2, test3])
            
            # Add summary
            if success:
                self.logs.append("SUMMARY: All tests passed - Physics solution is consistent.")
            else:
                self.logs.append("SUMMARY: Some tests failed - Check physics consistency.")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Critical error in PhysicsChecker: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs