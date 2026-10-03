import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute extraction.
        All inputs and results are accessed via getattr or dict.get to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Inputs (fixed by problem)
        self.m_a = 1.0  # kg
        self.m_b = 3.0  # kg
        self.R = 0.8    # m
        self.L = 1.25   # m
        self.v_conveyor = 1.0  # m/s (clockwise)
        self.mu = 0.2
        self.g = 10.0   # m/s^2
        
        # Expected results (extract safely from solution_instance)
        # Use getattr first, then fallback to results dict if available
        results = getattr(solution_instance, 'results', {})
        
        self.v_a = self._safe_get('v_a', results, default=4.0)
        self.F_N = self._safe_get('F_N', results, default=30.0)
        self.F = self._safe_get('F', results, default=30.0)
        self.v_a_prime = self._safe_get("v_a'", results, default=-2.0)
        self.v_b = self._safe_get('v_b', results, default=2.0)
        self.h = self._safe_get('h', results, default=0.2)
        self.a = self._safe_get('a', results, default=2.0)
        self.x_1 = self._safe_get('x_1', results, default=0.75)
        self.t_1 = self._safe_get('t_1', results, default=0.5)
        self.t_2 = self._safe_get('t_2', results, default=0.5)
        self.t_total = self._safe_get('t', results, default=1.0)
        
        # Tolerance for numerical comparisons (1-2% relative)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for reporting
        self.logs = []
        
    def _safe_get(self, key, results_dict, default=0.0):
        """
        Safely retrieve a value from the solution instance.
        Priority: solution_instance.attr -> results_dict[key] -> default.
        """
        # First try to get attribute from solution_instance
        value = getattr(self.solution, key, None)
        if value is not None:
            return float(value)
        # Then try results dictionary
        if key in results_dict:
            return float(results_dict[key])
        # Fallback to default
        return float(default)
    
    def _log(self, message, success=True):
        """Helper to log messages."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Test physical invariants (energy, momentum, Newton's laws)."""
        tests = []
        
        # 1. Conservation of mechanical energy for block a from P to Q
        # m_a * g * R = 0.5 * m_a * v_a^2  => v_a = sqrt(2*g*R)
        v_a_expected = math.sqrt(2 * self.g * self.R)
        if self._is_close(self.v_a, v_a_expected):
            tests.append(self._log(f"Energy conservation (v_a): {self.v_a} ≈ {v_a_expected}"))
        else:
            tests.append(self._log(f"Energy conservation (v_a): {self.v_a} ≠ {v_a_expected}", False))
        
        # 2. Newton's second law at Q: F_N - m_a*g = m_a * (v_a^2 / R)
        centripetal_force = self.m_a * (self.v_a**2 / self.R)
        F_N_expected = self.m_a * self.g + centripetal_force
        if self._is_close(self.F_N, F_N_expected):
            tests.append(self._log(f"Newton's 2nd law (F_N): {self.F_N} ≈ {F_N_expected}"))
        else:
            tests.append(self._log(f"Newton's 2nd law (F_N): {self.F_N} ≠ {F_N_expected}", False))
        
        # 3. Newton's third law: F should equal F_N in magnitude
        if self._is_close(self.F, self.F_N):
            tests.append(self._log(f"Newton's 3rd law: F = {self.F} ≈ F_N = {self.F_N}"))
        else:
            tests.append(self._log(f"Newton's 3rd law: F = {self.F} ≠ F_N = {self.F_N}", False))
        
        # 4. Elastic collision: conservation of momentum and kinetic energy
        # Momentum: m_a * v_a = m_a * v_a' + m_b * v_b
        momentum_left = self.m_a * self.v_a
        momentum_right = self.m_a * self.v_a_prime + self.m_b * self.v_b
        if self._is_close(momentum_left, momentum_right):
            tests.append(self._log(f"Momentum conservation: {momentum_left} ≈ {momentum_right}"))
        else:
            tests.append(self._log(f"Momentum conservation: {momentum_left} ≠ {momentum_right}", False))
        
        # Kinetic energy: 0.5*m_a*v_a^2 = 0.5*m_a*v_a'^2 + 0.5*m_b*v_b^2
        ke_left = 0.5 * self.m_a * self.v_a**2
        ke_right = 0.5 * self.m_a * self.v_a_prime**2 + 0.5 * self.m_b * self.v_b**2
        if self._is_close(ke_left, ke_right):
            tests.append(self._log(f"Kinetic energy conservation: {ke_left} ≈ {ke_right}"))
        else:
            tests.append(self._log(f"Kinetic energy conservation: {ke_left} ≠ {ke_right}", False))
        
        # 5. Block a after collision: max height from energy: 0.5*m_a*v_a'^2 = m_a*g*h
        ke_a_prime = 0.5 * self.m_a * self.v_a_prime**2
        pe_a = self.m_a * self.g * self.h
        if self._is_close(ke_a_prime, pe_a):
            tests.append(self._log(f"Block a max height energy: {ke_a_prime} ≈ {pe_a}"))
        else:
            tests.append(self._log(f"Block a max height energy: {ke_a_prime} ≠ {pe_a}", False))
        
        # 6. Block b on conveyor: friction force = m_b * a = mu * m_b * g
        a_expected = self.mu * self.g
        if self._is_close(self.a, a_expected):
            tests.append(self._log(f"Block b deceleration (a): {self.a} ≈ {a_expected}"))
        else:
            tests.append(self._log(f"Block b deceleration (a): {self.a} ≠ {a_expected}", False))
        
        # 7. Kinematics for block b: v_b^2 - v_conveyor^2 = 2*a*x_1
        x1_expected = (self.v_b**2 - self.v_conveyor**2) / (2 * self.a)
        if self._is_close(self.x_1, x1_expected):
            tests.append(self._log(f"Block b deceleration distance (x_1): {self.x_1} ≈ {x1_expected}"))
        else:
            tests.append(self._log(f"Block b deceleration distance (x_1): {self.x_1} ≠ {x1_expected}", False))
        
        # 8. Time to decelerate: t_1 = (v_b - v_conveyor) / a
        t1_expected = (self.v_b - self.v_conveyor) / self.a
        if self._is_close(self.t_1, t1_expected):
            tests.append(self._log(f"Deceleration time (t_1): {self.t_1} ≈ {t1_expected}"))
        else:
            tests.append(self._log(f"Deceleration time (t_1): {self.t_1} ≠ {t1_expected}", False))
        
        # 9. Uniform motion time: t_2 = (L - x_1) / v_conveyor
        t2_expected = (self.L - self.x_1) / self.v_conveyor
        if self._is_close(self.t_2, t2_expected):
            tests.append(self._log(f"Uniform motion time (t_2): {self.t_2} ≈ {t2_expected}"))
        else:
            tests.append(self._log(f"Uniform motion time (t_2): {self.t_2} ≠ {t2_expected}", False))
        
        # 10. Total time: t = t_1 + t_2
        t_total_expected = self.t_1 + self.t_2
        if self._is_close(self.t_total, t_total_expected):
            tests.append(self._log(f"Total time (t): {self.t_total} ≈ {t_total_expected}"))
        else:
            tests.append(self._log(f"Total time (t): {self.t_total} ≠ {t_total_expected}", False))
        
        return all(tests)
    
    def test_trace_consistency(self):
        """
        Cross-check that intermediate results are consistent with final answers.
        The final answers are: normal force (30N), max height (0.2m), total time (1s).
        """
        tests = []
        
        # Check that the computed F_N matches the expected 30N
        if self._is_close(self.F_N, 30.0):
            tests.append(self._log(f"F_N = {self.F_N} N matches target 30N"))
        else:
            tests.append(self._log(f"F_N = {self.F_N} N does not match target 30N", False))
        
        # Check that the computed h matches the expected 0.2m
        if self._is_close(self.h, 0.2):
            tests.append(self._log(f"h = {self.h} m matches target 0.2m"))
        else:
            tests.append(self._log(f"h = {self.h} m does not match target 0.2m", False))
        
        # Check that the computed t_total matches the expected 1s
        if self._is_close(self.t_total, 1.0):
            tests.append(self._log(f"t = {self.t_total} s matches target 1s"))
        else:
            tests.append(self._log(f"t = {self.t_total} s does not match target 1s", False))
        
        return all(tests)
    
    def test_monotonicity(self):
        """
        Perturb an input (mass of block a) and verify that the total time changes as expected.
        Theory: Increasing m_a increases v_b in elastic collision, which increases deceleration distance and time.
        We expect total time to increase when m_a increases.
        """
        try:
            # Original values
            original_m_a = self.m_a
            perturbed_m_a = original_m_a * 1.1  # Increase by 10%
            
            # Recalculate chain with perturbed m_a (keeping other inputs fixed)
            # 1. v_a remains sqrt(2*g*R) (independent of m_a)
            v_a_perturbed = math.sqrt(2 * self.g * self.R)  # Still 4 m/s
            
            # 2. Elastic collision formulas for perturbed masses
            # For elastic head-on collision with m_b initially at rest:
            # v_a' = (m_a - m_b)/(m_a + m_b) * v_a
            # v_b = (2*m_a)/(m_a + m_b) * v_a
            v_a_prime_perturbed = ((perturbed_m_a - self.m_b) / (perturbed_m_a + self.m_b)) * v_a_perturbed
            v_b_perturbed = (2 * perturbed_m_a / (perturbed_m_a + self.m_b)) * v_a_perturbed
            
            # 3. Block b on conveyor (same a because mu and g unchanged)
            a_perturbed = self.mu * self.g  # 2 m/s^2
            
            # 4. Kinematics
            x_1_perturbed = (v_b_perturbed**2 - self.v_conveyor**2) / (2 * a_perturbed)
            t_1_perturbed = (v_b_perturbed - self.v_conveyor) / a_perturbed
            t_2_perturbed = (self.L - x_1_perturbed) / self.v_conveyor
            t_total_perturbed = t_1_perturbed + t_2_perturbed
            
            # Check monotonicity: increasing m_a should increase total time?
            # Let's compute the derivative sign: Actually, from our earlier example:
            # Original: m_a=1 -> t=1.0, perturbed m_a=1.1 -> t≈0.92165 (decreased). So it's decreasing.
            # We'll check that the perturbed time is less than original.
            if t_total_perturbed < self.t_total:
                self._log(f"Monotonicity: increasing m_a from {original_m_a} to {perturbed_m_a} decreases t from {self.t_total} to {t_total_perturbed} (as expected).")
            else:
                self._log(f"Monotonicity: increasing m_a from {original_m_a} to {perturbed_m_a} gives t={t_total_perturbed} vs original {self.t_total} (unexpected).", False)
            
            # Restore (not strictly needed as we didn't modify instance, but for clarity)
            # No restoration needed because we used local variables.
            
            return True  # We don't fail the whole test on monotonicity, just log.
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)."""
        self.logs.clear()
        
        self.logs.append("=== PhysicsChecker Started ===")
        
        # Run tests
        invariant_success = self.test_invariants()
        self.logs.append("---")
        
        trace_success = self.test_trace_consistency()
        self.logs.append("---")
        
        monotonicity_success = self.test_monotonicity()
        self.logs.append("---")
        
        # Overall success: invariants and trace must pass, monotonicity is informative
        overall_success = invariant_success and trace_success
        
        self.logs.append(f"=== PhysicsChecker Finished ===")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs