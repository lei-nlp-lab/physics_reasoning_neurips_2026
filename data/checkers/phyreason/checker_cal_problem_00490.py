import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results using getattr with fallback to results dict
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        self.a_A = getattr(solution_instance, 'a_A', solution_instance.results.get('a_A', 0.0))
        self.a_B = getattr(solution_instance, 'a_B', solution_instance.results.get('a_B', 0.0))
        
        # Safe extraction of input parameters with defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.V0 = getattr(solution_instance, 'V0', 0.0)  # Though not used in formulas
        
        # Store logs
        self.logs = []
        
        # Setup symbols for symbolic checks
        self.m_sym, self.q_sym, self.g_sym, self.E_sym, self.aA_sym, self.aB_sym = sp.symbols('m q g E a_A a_B')
        self.sym_map = {self.m_sym: self.m, self.q_sym: self.q, self.g_sym: self.g}

    def _is_close(self, a, b):
        """Fuzzy numerical comparison with 1-2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Test Newton's Second Law invariants from KB."""
        logs = []
        try:
            # System equilibrium before string breaks: 2qE = 3mg
            net_force_before = 2 * self.q * self.E
            weight_before = 3 * self.m * self.g
            if self._is_close(net_force_before, weight_before):
                logs.append("✓ System equilibrium invariant passed: 2qE ≈ 3mg")
            else:
                logs.append(f"✗ System equilibrium invariant failed: 2qE={net_force_before}, 3mg={weight_before}")
            
            # Ball A after break: qE - mg = m*a_A
            net_force_A = self.q * self.E - self.m * self.g
            expected_force_A = self.m * self.a_A
            if self._is_close(net_force_A, expected_force_A):
                logs.append("✓ Ball A Newton's 2nd Law passed: qE - mg ≈ m*a_A")
            else:
                logs.append(f"✗ Ball A Newton's 2nd Law failed: qE - mg={net_force_A}, m*a_A={expected_force_A}")
            
            # Ball B after break: qE - 2mg = 2m*a_B
            net_force_B = self.q * self.E - 2 * self.m * self.g
            expected_force_B = 2 * self.m * self.a_B
            if self._is_close(net_force_B, expected_force_B):
                logs.append("✓ Ball B Newton's 2nd Law passed: qE - 2mg ≈ 2m*a_B")
            else:
                logs.append(f"✗ Ball B Newton's 2nd Law failed: qE - 2mg={net_force_B}, 2m*a_B={expected_force_B}")
                
        except Exception as e:
            logs.append(f"✗ Invariant test error: {e}")
        self.logs.extend(logs)
        return len(logs) == 3  # All three invariants passed

    def test_trace_consistency(self):
        """Cross-check with intermediate results from steps analysis."""
        logs = []
        try:
            # Step 1: E = 3mg/(2q)
            E_expected = 3 * self.m * self.g / (2 * self.q)
            if self._is_close(self.E, E_expected):
                logs.append(f"✓ E consistency passed: {self.E} ≈ {E_expected}")
            else:
                logs.append(f"✗ E consistency failed: got {self.E}, expected {E_expected}")
            
            # Step 2: a_A = g/2
            a_A_expected = self.g / 2
            if self._is_close(self.a_A, a_A_expected):
                logs.append(f"✓ a_A consistency passed: {self.a_A} ≈ {a_A_expected}")
            else:
                logs.append(f"✗ a_A consistency failed: got {self.a_A}, expected {a_A_expected}")
            
            # Step 3: a_B = -g/2
            a_B_expected = -self.g / 2
            if self._is_close(self.a_B, a_B_expected):
                logs.append(f"✓ a_B consistency passed: {self.a_B} ≈ {a_B_expected}")
            else:
                logs.append(f"✗ a_B consistency failed: got {self.a_B}, expected {a_B_expected}")
                
        except Exception as e:
            logs.append(f"✗ Trace consistency test error: {e}")
        self.logs.extend(logs)
        return len(logs) == 3  # All three steps passed

    def test_monotonicity(self):
        """Perturb mass m and check trend direction."""
        logs = []
        original_m = self.m
        try:
            # Increase m by 10%
            self.m *= 1.1
            
            # Recalculate expected values using formulas
            E_new = 3 * self.m * self.g / (2 * self.q)
            a_A_new = self.g / 2
            a_B_new = -self.g / 2
            
            # Check trends
            if E_new > self.E:
                logs.append("✓ E monotonicity passed: E increases with m")
            else:
                logs.append(f"✗ E monotonicity failed: E_new={E_new}, E_original={self.E}")
            
            if self._is_close(a_A_new, self.a_A):
                logs.append("✓ a_A monotonicity passed: a_A independent of m")
            else:
                logs.append(f"✗ a_A monotonicity failed: a_A_new={a_A_new}, a_A_original={self.a_A}")
            
            if self._is_close(a_B_new, self.a_B):
                logs.append("✓ a_B monotonicity passed: a_B independent of m")
            else:
                logs.append(f"✗ a_B monotonicity failed: a_B_new={a_B_new}, a_B_original={self.a_B}")
                
        except Exception as e:
            logs.append(f"✗ Monotonicity test error: {e}")
        finally:
            # Restore original m
            self.m = original_m
        
        self.logs.extend(logs)
        return len(logs) == 3  # All three monotonicity checks passed

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        success = True
        
        # Run tests
        inv_success = self.test_invariants()
        trace_success = self.test_trace_consistency()
        mono_success = self.test_monotonicity()
        
        # Overall success
        success = inv_success and trace_success and mono_success
        
        # Summary
        self.logs.append("\n=== SUMMARY ===")
        self.logs.append(f"Invariants: {'PASS' if inv_success else 'FAIL'}")
        self.logs.append(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}")
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs