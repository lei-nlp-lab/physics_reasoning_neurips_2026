import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Uses getattr and dict.get to prevent AttributeError.
        """
        # Input parameters
        self.theta = 53  # degrees
        self.m_A = 2.0
        self.mu = 0.5
        self.k1 = 140.0
        self.k2 = 100.0
        self.m_B = 1.6
        self.m_C = 1.0
        self.m_D = 0.4
        self.alpha = 74  # degrees
        self.g = 10.0
        
        # Trigonometric values (given)
        self.sin53 = 0.8  # sin(53°)
        self.cos53 = 0.6  # cos(53°)
        self.cos37 = 0.8  # cos(37°)
        
        # Safely extract results from solution_instance
        # Using getattr with fallback to results dictionary
        self.T = self._safe_get(solution_instance, 'T', 0.0)
        self.T_all = self._safe_get(solution_instance, 'T_all', 0.0)
        self.F = self._safe_get(solution_instance, 'F', 0.0)
        self.G_B = self._safe_get(solution_instance, 'G_B', 0.0)
        self.T1 = self._safe_get(solution_instance, 'T1', 0.0)
        self.delta_x1 = self._safe_get(solution_instance, 'delta_x1', 0.0)
        self.delta_x2 = self._safe_get(solution_instance, 'delta_x2', 0.0)
        self.delta_x3 = self._safe_get(solution_instance, 'delta_x3', 0.0)
        self.h = self._safe_get(solution_instance, 'h', 0.0)
        
        # Logs for reporting
        self.logs = []
        
    def _safe_get(self, solution_instance, key, default):
        """
        Safely retrieve attribute from solution_instance.
        First tries direct attribute, then results dictionary.
        """
        value = getattr(solution_instance, key, None)
        if value is None:
            results = getattr(solution_instance, 'results', {})
            value = results.get(key, default)
        return float(value) if isinstance(value, (int, float)) else default

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success

    def test_invariants(self):
        """
        Verify physical invariants:
        1. Block A equilibrium (T = m_A*g*sinθ - μ*m_A*g*cosθ)
        2. Movable pulley equilibrium (T_all = 2*T*cos37)
        3. Weight of B (G_B = m_B*g)
        4. Spring compressions/extensions from Hooke's Law
        5. Height consistency (h = Δx1 + Δx2 + Δx3)
        6. Second tension (T1 = (m_B+m_C+m_D)*g)
        7. Force F equilibrium (corrected formula)
        """
        tests_passed = []
        
        # 1. Block A equilibrium (about to slide down)
        T_expected = self.m_A * self.g * self.sin53 - self.mu * self.m_A * self.g * self.cos53
        tests_passed.append(self._log(
            f"Block A equilibrium: T = {self.T:.2f} N, expected {T_expected:.2f} N",
            math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 2. Movable pulley equilibrium (vector addition)
        T_all_expected = 2 * self.T * self.cos37
        tests_passed.append(self._log(
            f"Movable pulley: T_all = {self.T_all:.2f} N, expected {T_all_expected:.2f} N",
            math.isclose(self.T_all, T_all_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 3. Weight of B
        G_B_expected = self.m_B * self.g
        tests_passed.append(self._log(
            f"Weight of B: G_B = {self.G_B:.2f} N, expected {G_B_expected:.2f} N",
            math.isclose(self.G_B, G_B_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 4. Spring compressions/extensions (Hooke's Law)
        # Spring 2 compression due to C (Δx1 = m_C*g/k2)
        delta_x1_expected = self.m_C * self.g / self.k2
        tests_passed.append(self._log(
            f"Spring 2 compression: Δx1 = {self.delta_x1:.3f} m, expected {delta_x1_expected:.3f} m",
            math.isclose(self.delta_x1, delta_x1_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Spring 1 elongation (Δx2 = (m_C+m_D)*g/k1)
        delta_x2_expected = (self.m_C + self.m_D) * self.g / self.k1
        tests_passed.append(self._log(
            f"Spring 1 elongation: Δx2 = {self.delta_x2:.3f} m, expected {delta_x2_expected:.3f} m",
            math.isclose(self.delta_x2, delta_x2_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Spring 2 extension due to D (Δx3 = m_D*g/k2)
        delta_x3_expected = self.m_D * self.g / self.k2
        tests_passed.append(self._log(
            f"Spring 2 extension: Δx3 = {self.delta_x3:.3f} m, expected {delta_x3_expected:.3f} m",
            math.isclose(self.delta_x3, delta_x3_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 5. Height consistency
        h_expected = self.delta_x1 + self.delta_x2 + self.delta_x3
        tests_passed.append(self._log(
            f"Height h: {self.h:.3f} m, expected {h_expected:.3f} m",
            math.isclose(self.h, h_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 6. Second tension (vertical rope)
        T1_expected = (self.m_B + self.m_C + self.m_D) * self.g
        tests_passed.append(self._log(
            f"Second tension T1: {self.T1:.2f} N, expected {T1_expected:.2f} N",
            math.isclose(self.T1, T1_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # 7. Force F equilibrium (CORRECTED FORMULA)
        # From equilibrium: T1 + μ*(m_A*g + F)*cosθ = (m_A*g + F)*sinθ
        # Rearranged: F = (T1 - m_A*g*sinθ + μ*m_A*g*cosθ) / (sinθ - μ*cosθ)
        numerator = self.T1 - self.m_A * self.g * self.sin53 + self.mu * self.m_A * self.g * self.cos53
        denominator = self.sin53 - self.mu * self.cos53
        F_expected = numerator / denominator
        tests_passed.append(self._log(
            f"Force F: {self.F:.2f} N, expected {F_expected:.2f} N",
            math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        return all(tests_passed)

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        tests_passed = []
        
        # Check that T_all and G_B are equal (both 16 N from steps)
        tests_passed.append(self._log(
            f"T_all vs G_B: {self.T_all:.2f} N vs {self.G_B:.2f} N",
            math.isclose(self.T_all, self.G_B, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Check that h matches target answer (0.24 m)
        tests_passed.append(self._log(
            f"Height h vs target 0.24 m: {self.h:.3f} m",
            math.isclose(self.h, 0.24, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Check that F matches target answer (40 N)
        tests_passed.append(self._log(
            f"Force F vs target 40 N: {self.F:.2f} N",
            math.isclose(self.F, 40.0, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Check that T matches target answer (10 N)
        tests_passed.append(self._log(
            f"Tension T vs target 10 N: {self.T:.2f} N",
            math.isclose(self.T, 10.0, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        return all(tests_passed)

    def test_monotonicity(self):
        """
        Perturb an input parameter (m_A) and verify expected trend.
        Then restore original value.
        """
        try:
            # Store original value
            m_A_original = self.m_A
            
            # Perturb m_A by +10%
            self.m_A *= 1.1
            
            # Recompute expected T using equilibrium formula
            T_perturbed = self.m_A * self.g * self.sin53 - self.mu * self.m_A * self.g * self.cos53
            
            # Check that T increases with m_A (since derivative positive)
            # Derivative: dT/dm_A = g*(sin53 - μ*cos53) = 10*(0.8 - 0.5*0.6) = 5 N/kg
            expected_increase = (self.m_A - m_A_original) * self.g * (self.sin53 - self.mu * self.cos53)
            actual_increase = T_perturbed - self.T
            
            # Restore original value
            self.m_A = m_A_original
            
            # Verify trend (allow small tolerance)
            success = math.isclose(actual_increase, expected_increase, rel_tol=1e-2, abs_tol=1e-2)
            return self._log(
                f"Monotonicity test: dT/dm_A positive, actual ΔT = {actual_increase:.3f} N, expected ΔT = {expected_increase:.3f} N",
                success
            )
            
        except Exception as e:
            return self._log(f"Monotonicity test failed with error: {str(e)}", False)

    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []  # Reset logs
        
        # Run tests in sequence
        results = []
        try:
            results.append(("Invariants", self.test_invariants()))
            results.append(("Trace Consistency", self.test_trace_consistency()))
            results.append(("Monotonicity", self.test_monotonicity()))
        except Exception as e:
            self.logs.append(f"ERROR: Test suite crashed: {traceback.format_exc()}")
            return False, self.logs
        
        # Check overall success
        success = all(r[1] for r in results)
        self.logs.insert(0, f"Overall result: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs