import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Follows robust access pattern to prevent AttributeError.
        """
        # Get results dict with safe fallback
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract computed values with safe defaults
        self.a_computed = self.results.get('a', 0.0)  # Average acceleration from solution
        self.t_computed = self.results.get('t', 0.0)  # Time for ordinary car from solution
        
        # Extract problem constants (could also be from solution_instance if available)
        self.dt_porsche = 4.2  # s - Porsche acceleration time (given in problem)
        self.a_ordinary = 3.0  # m/s² - Ordinary car acceleration (given in problem)
        
        # Convert velocity units: 100 km/h to m/s
        self.v0 = 0.0  # m/s - initial velocity (standstill)
        self.v1 = 100.0 * (1000.0 / 3600.0)  # 100 km/h → m/s = 27.777... m/s
        self.delta_v = self.v1 - self.v0  # Δv = 27.777... m/s
        
        # Initialize sympy symbols for symbolic checks
        self.v0_sym = sp.Symbol('v0', real=True)
        self.v1_sym = sp.Symbol('v1', real=True)
        self.dt_sym = sp.Symbol('dt', real=True)
        self.a_sym = sp.Symbol('a', real=True)
        
        # Store validation logs
        self.logs = []

    def _log(self, message, success):
        """Helper to append log messages with success flag."""
        self.logs.append({
            'test': message,
            'success': success
        })
        return success

    def test_invariants(self):
        """
        Verify physical invariants based on definition of average acceleration.
        """
        tests_passed = True
        
        # Test 1: Check Porsche acceleration using a = Δv / Δt
        a_expected = self.delta_v / self.dt_porsche
        if not math.isclose(self.a_computed, a_expected, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Porsche acceleration invariant failed: computed {self.a_computed:.3f}, expected {a_expected:.3f}", False)
            tests_passed = False
        else:
            self._log(f"Porsche acceleration check passed: {self.a_computed:.3f} ≈ {a_expected:.3f} m/s²", True)
        
        # Test 2: Check ordinary car time using t = Δv / a
        t_expected = self.delta_v / self.a_ordinary
        if not math.isclose(self.t_computed, t_expected, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Ordinary car time invariant failed: computed {self.t_computed:.3f}, expected {t_expected:.3f}", False)
            tests_passed = False
        else:
            self._log(f"Ordinary car time check passed: {self.t_computed:.3f} ≈ {t_expected:.3f} s", True)
        
        # Symbolic check: a = (v1 - v0)/dt
        symbolic_eq = sp.Eq(self.a_sym, (self.v1_sym - self.v0_sym) / self.dt_sym)
        self._log(f"Symbolic invariant: {sp.latex(symbolic_eq)}", True)
        
        return tests_passed

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        Ensure both steps use consistent Δv.
        """
        tests_passed = True
        
        # Compute Δv from Porsche step: Δv = a_computed * dt_porsche
        delta_v_from_porsche = self.a_computed * self.dt_porsche
        
        # Compute Δv from ordinary car step: Δv = a_ordinary * t_computed
        delta_v_from_ordinary = self.a_ordinary * self.t_computed
        
        # Both should match the known Δv (27.777... m/s)
        delta_v_actual = self.delta_v
        
        # Check Porsche step consistency
        if not math.isclose(delta_v_from_porsche, delta_v_actual, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Porsche step inconsistency: a*dt = {delta_v_from_porsche:.3f}, Δv = {delta_v_actual:.3f}", False)
            tests_passed = False
        else:
            self._log(f"Porsche step consistent: a*dt = Δv = {delta_v_actual:.3f} m/s", True)
        
        # Check ordinary car step consistency
        if not math.isclose(delta_v_from_ordinary, delta_v_actual, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Ordinary car step inconsistency: a_ordinary*t = {delta_v_from_ordinary:.3f}, Δv = {delta_v_actual:.3f}", False)
            tests_passed = False
        else:
            self._log(f"Ordinary car step consistent: a_ordinary*t = Δv = {delta_v_actual:.3f} m/s", True)
        
        # Verify both give same Δv (cross-check between steps)
        if not math.isclose(delta_v_from_porsche, delta_v_from_ordinary, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Cross-step Δv mismatch: Porsche gives {delta_v_from_porsche:.3f}, ordinary gives {delta_v_from_ordinary:.3f}", False)
            tests_passed = False
        else:
            self._log(f"Cross-step Δv consistent: {delta_v_from_porsche:.3f} m/s", True)
        
        return tests_passed

    def test_monotonicity(self):
        """
        Perturb inputs and verify expected monotonic trends.
        Based on KB theorem: a increases with Δv (fixed dt), decreases with dt (fixed Δv).
        """
        tests_passed = True
        
        # Test 1: Perturb Porsche time (dt_porsche) and check acceleration trend
        original_dt = self.dt_porsche
        perturbed_dt = original_dt * 1.1  # Increase by 10%
        
        # Compute expected acceleration with perturbed time
        a_perturbed_expected = self.delta_v / perturbed_dt
        
        # Since dt increased, acceleration should decrease
        if a_perturbed_expected < self.a_computed:
            self._log(f"Monotonicity trend correct: increasing dt ({original_dt:.1f}→{perturbed_dt:.1f}s) decreases a", True)
        else:
            self._log(f"Monotonicity trend violation: increasing dt should decrease a", False)
            tests_passed = False
        
        # Restore original value
        self.dt_porsche = original_dt
        
        # Test 2: Perturb ordinary car acceleration and check time trend
        original_a = self.a_ordinary
        perturbed_a = original_a * 1.1  # Increase by 10%
        
        # Compute expected time with perturbed acceleration
        t_perturbed_expected = self.delta_v / perturbed_a
        
        # Since a increased, time should decrease
        if t_perturbed_expected < self.t_computed:
            self._log(f"Monotonicity trend correct: increasing a ({original_a:.1f}→{perturbed_a:.1f}m/s²) decreases t", True)
        else:
            self._log(f"Monotonicity trend violation: increasing a should decrease t", False)
            tests_passed = False
        
        # Restore original value
        self.a_ordinary = original_a
        
        # Test 3: Perturb velocity change (Δv) and check acceleration trend (for Porsche)
        original_dv = self.delta_v
        perturbed_dv = original_dv * 1.1  # Increase by 10%
        
        # Compute expected acceleration with perturbed Δv
        a_perturbed_dv_expected = perturbed_dv / self.dt_porsche
        
        # Since Δv increased, acceleration should increase
        if a_perturbed_dv_expected > self.a_computed:
            self._log(f"Monotonicity trend correct: increasing Δv ({original_dv:.1f}→{perturbed_dv:.1f}m/s) increases a", True)
        else:
            self._log(f"Monotonicity trend violation: increasing Δv should increase a", False)
            tests_passed = False
        
        return tests_passed

    def run_all_tests(self):
        """
        Execute all audit tests and return overall success with logs.
        """
        self.logs = []  # Reset logs
        
        try:
            # Run individual tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            # Determine overall success
            all_passed = test1 and test2 and test3
            
            # Add summary log
            summary = f"All tests passed: {all_passed} (invariants:{test1}, trace:{test2}, monotonicity:{test3})"
            self._log(summary, all_passed)
            
            return all_passed, self.logs
            
        except Exception as e:
            # Catch any unexpected errors
            error_msg = f"PhysicsChecker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append({'test': error_msg, 'success': False})
            return False, self.logs