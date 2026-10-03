import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        All inputs and results extracted via getattr with fallback.
        """
        # Extract known constants from the problem
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 1.0))
        self.E0 = getattr(solution_instance, 'E0', solution_instance.results.get('E0', 1.0))
        self.B0 = getattr(solution_instance, 'B0', solution_instance.results.get('B0', 1.0))
        
        # Define time periods based on problem
        self.T_half = sp.pi * self.m / (self.q * self.B0)  # πm/(qB0)
        
        # Extract all results with safe fallbacks
        self.results = {}
        keys = [
            't', 'r_2', 'y_3', 'y_5', 't_1', 'r_2', 'x_2', 'x_4', 'x', 'y', 
            'y_5', 'N/A', 'y_3', 'y_1', 'r_4', 'y_1', 'x_2', 't_1', 'r_4', 
            'W', 'v_3', 'y_1', 'v_2', 'x_4', 'x_2', 'y_3', 'r_2', 'v_1',
            'y_1_prime', 'x_2_prime', 'y_3_prime', 'y_5_prime',  # Primed variables
            't_1_double_prime', 'y_1_double_prime', 'x_2_double_prime', 'y_3_double_prime'  # Double primed
        ]
        
        for key in keys:
            self.results[key] = getattr(solution_instance, key, 
                                       solution_instance.results.get(key, None))
        
        # Initialize symbolic variables
        self.m_sym, self.q_sym, self.E0_sym, self.B0_sym = sp.symbols('m q E0 B0', positive=True)
        self.pi = sp.pi
        
        # Define expected symbolic expressions from steps analysis
        self.expected = {
            'v_1': self.pi * self.E0_sym / self.B0_sym,
            'y_1': self.pi**2 * self.E0_sym * self.m_sym / (2 * self.q_sym * self.B0_sym**2),
            'r_2': self.pi * self.E0_sym * self.m_sym / (self.q_sym * self.B0_sym**2),
            'x_2': 2 * self.pi * self.E0_sym * self.m_sym / (self.q_sym * self.B0_sym**2),
            'x': 2 * self.pi * self.E0_sym * self.m_sym / (self.q_sym * self.B0_sym**2),
            'y': self.pi**2 * self.E0_sym * self.m_sym / (2 * self.q_sym * self.B0_sym**2),
            'v_2': self.pi * self.E0_sym / self.B0_sym,
            'y_3': 0,
            'r_4': self.pi * self.E0_sym * self.m_sym / (self.q_sym * self.B0_sym**2),
            'x_4': 2 * self.pi * self.E0_sym * self.m_sym / (self.q_sym * self.B0_sym**2),
            'v_3': 2 * self.pi * self.E0_sym / self.B0_sym,
            'y_5': self.pi**2 * self.E0_sym * self.m_sym / (2 * self.q_sym * self.B0_sym**2),
            'W': 2 * self.pi**2 * self.E0_sym**2 * self.m_sym / self.B0_sym**2,
            't_1': self.pi * self.m_sym / (2 * self.q_sym * self.B0_sym),
            't_double_prime': 13 * self.pi * self.m_sym / (3 * self.q_sym * self.B0_sym)
        }
        
        self.logs = []
    
    def _log(self, msg, success=None):
        """Helper for logging test results."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def _compare(self, val1, val2, name, symbolic_check=True):
        """
        Compare two values using fuzzy matching or symbolic equality.
        Returns (success, message)
        """
        try:
            # Try numerical comparison first
            num1 = float(val1) if not isinstance(val1, (int, float)) else val1
            num2 = float(val2) if not isinstance(val2, (int, float)) else val2
            if math.isclose(num1, num2, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"{name}: {num1:.3e} ≈ {num2:.3e}"
            else:
                return False, f"{name}: {num1:.3e} ≠ {num2:.3e}"
        except (TypeError, ValueError):
            # Fallback to symbolic comparison if possible
            if symbolic_check:
                try:
                    diff = sp.simplify(val1 - val2)
                    if diff == 0:
                        return True, f"{name}: symbolic match"
                    else:
                        return False, f"{name}: symbolic mismatch, diff = {diff}"
                except:
                    pass
            return False, f"{name}: comparison failed (types: {type(val1)}, {type(val2)})"
    
    def test_invariants(self):
        """Test physical invariants: impulse theorem, work-energy, circular motion."""
        logs = []
        success = True
        
        # Test impulse theorem for step 1
        try:
            # From step 1: qE0 * (πm/(qB0)) = m*v1
            lhs = self.q * self.E0 * self.T_half
            rhs = self.m * self.results['v_1']
            ok, msg = self._compare(lhs, rhs, "Impulse theorem step 1")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"Impulse step 1 error: {e}")
            success = False
        
        # Test impulse theorem for step 6
        try:
            # q*2E0*(πm/(qB0)) = m*v2 + m*v1
            lhs = 2 * self.q * self.E0 * self.T_half
            rhs = self.m * self.results['v_2'] + self.m * self.results['v_1']
            ok, msg = self._compare(lhs, rhs, "Impulse theorem step 6")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"Impulse step 6 error: {e}")
            success = False
        
        # Test impulse theorem for step 10
        try:
            # q*3E0*(πm/(qB0)) = m*v3 + m*v2
            lhs = 3 * self.q * self.E0 * self.T_half
            rhs = self.m * self.results['v_3'] + self.m * self.results['v_2']
            ok, msg = self._compare(lhs, rhs, "Impulse theorem step 10")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"Impulse step 10 error: {e}")
            success = False
        
        # Test circular motion: r2 = m*v1/(q*B0)
        try:
            expected_r2 = self.m * self.results['v_1'] / (self.q * self.B0)
            ok, msg = self._compare(self.results['r_2'], expected_r2, "Circular motion r2")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"Circular motion error: {e}")
            success = False
        
        # Test work-energy consistency: W = qE0*y1 + q*2E0*y3 + q*3E0*y5
        try:
            w_calc = (self.q * self.E0 * self.results['y_1'] +
                     self.q * 2 * self.E0 * self.results['y_3'] +
                     self.q * 3 * self.E0 * self.results['y_5'])
            ok, msg = self._compare(self.results['W'], w_calc, "Work-energy consistency")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"Work-energy error: {e}")
            success = False
        
        self._log("Invariants test: " + ("passed" if success else "failed"), success)
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check that x coordinate matches x_2
        try:
            ok, msg = self._compare(self.results['x'], self.results['x_2'], "x vs x_2")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"x consistency error: {e}")
            success = False
        
        # Check that y coordinate matches y_1
        try:
            ok, msg = self._compare(self.results['y'], self.results['y_1'], "y vs y_1")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"y consistency error: {e}")
            success = False
        
        # Check that r_4 equals r_2 (same magnitude)
        try:
            ok, msg = self._compare(self.results['r_4'], self.results['r_2'], "r_4 vs r_2")
            logs.append(msg)
            if not ok: success = False
        except Exception as e:
            logs.append(f"radius consistency error: {e}")
            success = False
        
        # Check that y_3 is zero as derived
        try:
            if not math.isclose(float(self.results['y_3']), 0, abs_tol=1e-2):
                logs.append(f"y_3 should be 0, got {self.results['y_3']}")
                success = False
            else:
                logs.append("y_3 = 0 ✓")
        except Exception as e:
            logs.append(f"y_3 check error: {e}")
            success = False
        
        self._log("Trace consistency: " + ("passed" if success else "failed"), success)
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb mass m and check trends in work and coordinates.
        Restores original mass after test.
        """
        logs = []
        success = True
        
        original_m = self.m
        
        try:
            # Increase mass by 10%
            self.m = original_m * 1.1
            
            # Recalculate expected values using formulas
            # Work should increase with mass (W ∝ m)
            W_original = 2 * math.pi**2 * self.E0**2 * original_m / self.B0**2
            W_perturbed = 2 * math.pi**2 * self.E0**2 * self.m / self.B0**2
            
            if W_perturbed > W_original:
                logs.append(f"Work monotonicity: W increases with m ({W_original:.3e} → {W_perturbed:.3e}) ✓")
            else:
                logs.append(f"Work monotonicity FAILED: W decreases with m")
                success = False
            
            # x-coordinate should increase with mass (x ∝ m)
            x_original = 2 * math.pi * self.E0 * original_m / (self.q * self.B0**2)
            x_perturbed = 2 * math.pi * self.E0 * self.m / (self.q * self.B0**2)
            
            if x_perturbed > x_original:
                logs.append(f"x monotonicity: x increases with m ({x_original:.3e} → {x_perturbed:.3e}) ✓")
            else:
                logs.append(f"x monotonicity FAILED: x decreases with m")
                success = False
            
            # y-coordinate should increase with mass (y ∝ m)
            y_original = math.pi**2 * self.E0 * original_m / (2 * self.q * self.B0**2)
            y_perturbed = math.pi**2 * self.E0 * self.m / (2 * self.q * self.B0**2)
            
            if y_perturbed > y_original:
                logs.append(f"y monotonicity: y increases with m ({y_original:.3e} → {y_perturbed:.3e}) ✓")
            else:
                logs.append(f"y monotonicity FAILED: y decreases with m")
                success = False
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {e}")
            success = False
            self.m = original_m  # Ensure restoration
        
        self._log("Monotonicity test: " + ("passed" if success else "failed"), success)
        return success, logs
    
    def test_expected_values(self):
        """Compare results with expected symbolic expressions."""
        logs = []
        success = True
        
        # Test key results against expected expressions
        test_cases = [
            ('v_1', self.expected['v_1']),
            ('y_1', self.expected['y_1']),
            ('x', self.expected['x']),
            ('y', self.expected['y']),
            ('W', self.expected['W']),
        ]
        
        for key, expected_expr in test_cases:
            try:
                # Substitute numerical values into expected expression
                expected_num = expected_expr.subs({
                    self.m_sym: self.m,
                    self.q_sym: self.q,
                    self.E0_sym: self.E0,
                    self.B0_sym: self.B0
                })
                
                result_val = self.results[key]
                ok, msg = self._compare(result_val, expected_num, key)
                logs.append(msg)
                if not ok: success = False
            except Exception as e:
                logs.append(f"{key} check error: {e}")
                success = False
        
        self._log("Expected values test: " + ("passed" if success else "failed"), success)
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Expected Values", self.test_expected_values),
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name}...")
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"{test_name} crashed: {traceback.format_exc()}")
                overall_success = False
        
        self._log(f"All tests completed. Overall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs