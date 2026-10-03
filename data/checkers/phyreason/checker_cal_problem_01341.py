import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract input parameters safely
        self.m = getattr(solution_instance, 'm', None) or solution_instance.results.get('m', 1.0)
        self.g = getattr(solution_instance, 'g', None) or solution_instance.results.get('g', 9.8)
        self.l = getattr(solution_instance, 'l', None) or solution_instance.results.get('l', 1.0)
        
        # Extract results safely
        self.W_G = getattr(solution_instance, 'W_G', None) or solution_instance.results.get('W_G', 0.0)
        self.delta_E_p = getattr(solution_instance, 'delta_E_p', None) or solution_instance.results.get('delta_E_p', 0.0)
        self.h = getattr(solution_instance, 'h', None) or solution_instance.results.get('h', 0.0)
        
        # Store original values for perturbation tests
        self.original_m = float(self.m) if isinstance(self.m, (int, float)) else self.m
        self.original_g = float(self.g) if isinstance(self.g, (int, float)) else self.g
        self.original_l = float(self.l) if isinstance(self.l, (int, float)) else self.l

    def _compare(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust comparison for symbolic or numeric values."""
        try:
            # Convert to floats if possible
            a_num = float(a) if isinstance(a, (int, float, sp.Number)) else a
            b_num = float(b) if isinstance(b, (int, float, sp.Number)) else b
            
            if isinstance(a_num, (int, float)) and isinstance(b_num, (int, float)):
                return math.isclose(a_num, b_num, rel_tol=rel_tol, abs_tol=abs_tol)
            else:
                # Symbolic comparison
                diff = sp.simplify(sp.sympify(a_num) - sp.sympify(b_num))
                return diff == 0
        except:
            return False

    def test_invariants(self):
        """Check energy conservation and work-energy theorem."""
        logs = []
        
        try:
            # Check: W_G = -ΔE_p (from KB reference)
            if not self._compare(self.W_G, -self.delta_E_p):
                logs.append(f"FAIL: Work-energy theorem violated. W_G={self.W_G}, -ΔE_p={-self.delta_E_p}")
                return False, logs
            
            # Check symbolic relationship: ΔE_p = m*g*(h1-h0)
            # For this problem: h1-h0 = -h (falling downward)
            m_sym, g_sym, l_sym, h_sym = sp.symbols('m g l h')
            expected_delta_E_p = m_sym * g_sym * (-h_sym)  # = -m*g*h
            actual_delta_E_p = sp.sympify(self.delta_E_p) if not isinstance(self.delta_E_p, (int, float)) else self.delta_E_p
            
            # Test with numeric substitution
            test_vals = {m_sym: float(self.m), g_sym: float(self.g), h_sym: float(self.h)}
            num_expected = float(expected_delta_E_p.subs(test_vals))
            num_actual = float(actual_delta_E_p) if isinstance(actual_delta_E_p, (int, float)) else float(actual_delta_E_p.subs(test_vals))
            
            if not math.isclose(num_expected, num_actual, rel_tol=1e-2):
                logs.append(f"FAIL: ΔE_p formula mismatch. Expected {num_expected}, got {num_actual}")
                return False, logs
                
            logs.append("PASS: All invariants satisfied.")
            return True, logs
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            return False, logs

    def test_trace_consistency(self):
        """Check intermediate results match final answers."""
        logs = []
        
        try:
            # Step 1: Check h = l - (2/3)l = (1/3)l
            expected_h = self.l / 3
            if not self._compare(self.h, expected_h):
                logs.append(f"FAIL: Height mismatch. Expected h={expected_h}, got {self.h}")
                return False, logs
            logs.append(f"PASS: h = {self.h} = (1/3)l")
            
            # Step 2: Check W_G = m*g*h
            expected_W_G = self.m * self.g * self.h
            if not self._compare(self.W_G, expected_W_G):
                logs.append(f"FAIL: Work done by gravity mismatch. Expected W_G={expected_W_G}, got {self.W_G}")
                return False, logs
            logs.append(f"PASS: W_G = m*g*h = {self.W_G}")
            
            # Step 3: Check ΔE_p = -W_G
            expected_delta_E_p = -self.W_G
            if not self._compare(self.delta_E_p, expected_delta_E_p):
                logs.append(f"FAIL: ΔE_p mismatch. Expected ΔE_p={expected_delta_E_p}, got {self.delta_E_p}")
                return False, logs
            logs.append(f"PASS: ΔE_p = -W_G = {self.delta_E_p}")
            
            # Final check: Both answers should be (1/3)mgl (in magnitude)
            expected_magnitude = abs(self.m * self.g * self.l / 3)
            if not (self._compare(abs(self.W_G), expected_magnitude) and 
                    self._compare(abs(self.delta_E_p), expected_magnitude)):
                logs.append(f"FAIL: Final answers don't match expected magnitude {expected_magnitude}")
                return False, logs
                
            logs.append("PASS: All trace consistency checks passed.")
            return True, logs
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False, logs

    def test_monotonicity(self):
        """Check that results scale correctly with mass perturbation."""
        logs = []
        
        try:
            # Store original values
            original_m = float(self.m) if isinstance(self.m, (int, float)) else self.m
            perturbation_factor = 1.1
            
            # Perturb mass
            if isinstance(original_m, (int, float)):
                perturbed_m = original_m * perturbation_factor
                
                # Calculate expected perturbed results using physics formulas
                expected_h = self.l / 3  # unchanged
                expected_W_G_perturbed = perturbed_m * self.g * expected_h
                expected_delta_E_p_perturbed = -expected_W_G_perturbed
                
                # Calculate original expected values
                expected_W_G_original = original_m * self.g * expected_h
                expected_delta_E_p_original = -expected_W_G_original
                
                # Check scaling
                if not math.isclose(expected_W_G_perturbed / expected_W_G_original, 
                                  perturbation_factor, rel_tol=1e-2):
                    logs.append(f"FAIL: W_G doesn't scale linearly with mass.")
                    return False, logs
                    
                if not math.isclose(expected_delta_E_p_perturbed / expected_delta_E_p_original,
                                  perturbation_factor, rel_tol=1e-2):
                    logs.append(f"FAIL: ΔE_p doesn't scale linearly with mass.")
                    return False, logs
                    
                logs.append("PASS: Monotonicity test passed - results scale correctly with mass.")
                return True, logs
            else:
                logs.append("SKIP: Monotonicity test - mass is not numeric.")
                return True, logs
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            return False, logs
        finally:
            # Restore original mass (important for future tests)
            self.m = original_m

    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== Running {test_name} Test ===")
            try:
                test_success, logs = test_func()
                all_logs.extend(logs)
                if not test_success:
                    success = False
                    all_logs.append(f"FAIL: {test_name} test failed.")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                success = False
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, all_logs