import math
import sympy as sp
from sympy.abc import x, y
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Uses getattr and .get() to avoid AttributeErrors in sandbox.
        """
        try:
            self.solution = solution_instance
            self.results = getattr(solution_instance, 'results', {})

            # Extract input parameters with safe defaults (SI units)
            self.m = getattr(solution_instance, 'm', self.results.get('m', 0.04))
            self.v = getattr(solution_instance, 'v', self.results.get('v', 10.0))
            self.k = getattr(solution_instance, 'k', self.results.get('k', 3e-3))
            self.g = getattr(solution_instance, 'g', self.results.get('g', 10.0))

            # Extract result quantities with safe fallbacks
            self.F1 = getattr(solution_instance, 'F_1', self.results.get('F_1', None))
            self.F2 = getattr(solution_instance, 'F_2', self.results.get('F_2', None))
            self.T = getattr(solution_instance, 'T', self.results.get('T', None))
            self.T2 = getattr(solution_instance, 'T2', self.results.get('T2', None))
            self.costheta = getattr(solution_instance, 'costheta', self.results.get('costheta', None))
            self.theta = getattr(solution_instance, 'theta', self.results.get('theta', None))

            # Store expected target answers for comparison
            self.target_T = 0.5   # N
            self.target_theta = 37.0  # degrees

            self.logs = []
            self.all_tests_passed = True
        except Exception as e:
            self.logs.append(f"ERROR in __init__: {e}")
            self.all_tests_passed = False

    def _log(self, message, success):
        """Helper to log test results and update overall success."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        if not success:
            self.all_tests_passed = False

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify fundamental physical laws: equilibrium, wind force law, Newton's third law."""
        try:
            # 1. Equilibrium of concurrent forces: net force zero in x and y
            if self.F1 is not None and self.T is not None and self.theta is not None:
                # Horizontal: F_wind = T * sinθ
                theta_rad = math.radians(self.theta)
                expected_F1_horiz = self.T * math.sin(theta_rad)
                horiz_ok = self._is_close(self.F1, expected_F1_horiz)
                self._log(f"Horizontal equilibrium: F1={self.F1}, T*sinθ={expected_F1_horiz}", horiz_ok)

                # Vertical: m*g = T * cosθ
                weight = self.m * self.g
                expected_F2_vert = self.T * math.cos(theta_rad)
                vert_ok = self._is_close(weight, expected_F2_vert)
                self._log(f"Vertical equilibrium: mg={weight}, T*cosθ={expected_F2_vert}", vert_ok)

                # Alternative check using Pythagorean theorem
                if self.T2 is not None:
                    expected_T2 = weight**2 + self.F1**2
                    pythag_ok = self._is_close(self.T2, expected_T2)
                    self._log(f"Pythagorean check: T^2={self.T2}, (mg)^2+F1^2={expected_T2}", pythag_ok)
            else:
                self._log("Missing data for equilibrium checks", False)

            # 2. Wind force equation: F1 = k * v^2
            if self.F1 is not None:
                expected_F1_wind = self.k * self.v**2
                wind_ok = self._is_close(self.F1, expected_F1_wind)
                self._log(f"Wind force law: F1={self.F1}, k*v^2={expected_F1_wind}", wind_ok)

            # 3. Newton's third law consistency (tension in wire is equal magnitude at both ends)
            # For the ball, tension T balances weight and wind force. For the support, reaction is -T.
            # Since we only have T from the ball, we assume it's consistent.
            # We can check that T is positive and matches the vector sum.
            if self.T is not None:
                self._log(f"Tension T={self.T} is positive (magnitude)", self.T > 0)
        except Exception as e:
            self._log(f"Exception in test_invariants: {e}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate results and final answers for internal consistency."""
        try:
            # 1. Check F2 = m*g
            if self.F2 is not None:
                weight = self.m * self.g
                f2_ok = self._is_close(self.F2, weight)
                self._log(f"F2 = mg: {self.F2} vs {weight}", f2_ok)

            # 2. Check T^2 = (mg)^2 + F1^2
            if self.T2 is not None and self.F1 is not None:
                expected_T2 = (self.m * self.g)**2 + self.F1**2
                t2_ok = self._is_close(self.T2, expected_T2)
                self._log(f"T^2 = (mg)^2 + F1^2: {self.T2} vs {expected_T2}", t2_ok)

            # 3. Check T = sqrt(T^2)
            if self.T is not None and self.T2 is not None:
                expected_T = math.sqrt(self.T2)
                t_ok = self._is_close(self.T, expected_T)
                self._log(f"T = sqrt(T^2): {self.T} vs {expected_T}", t_ok)

            # 4. Check cosθ = mg / T
            if self.costheta is not None and self.T is not None:
                expected_costheta = (self.m * self.g) / self.T
                cos_ok = self._is_close(self.costheta, expected_costheta)
                self._log(f"cosθ = mg/T: {self.costheta} vs {expected_costheta}", cos_ok)

            # 5. Check θ = arccos(cosθ) in degrees
            if self.theta is not None and self.costheta is not None:
                expected_theta = math.degrees(math.acos(self.costheta))
                theta_ok = self._is_close(self.theta, expected_theta)
                self._log(f"θ = arccos(cosθ): {self.theta} vs {expected_theta}", theta_ok)

            # 6. Compare final answers to target values
            if self.T is not None:
                t_target_ok = self._is_close(self.T, self.target_T)
                self._log(f"T matches target 0.5 N: {self.T}", t_target_ok)
            if self.theta is not None:
                theta_target_ok = self._is_close(self.theta, self.target_theta)
                self._log(f"θ matches target 37°: {self.theta}", theta_target_ok)
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {e}", False)

    def test_monotonicity(self):
        """
        Perturb inputs and verify expected trends.
        We do NOT modify the solution instance, only use local formulas.
        """
        try:
            # Perturb wind speed v (increase by 10%)
            v_original = self.v
            v_perturbed = v_original * 1.1
            F1_perturbed = self.k * v_perturbed**2
            T_perturbed = math.sqrt((self.m * self.g)**2 + F1_perturbed**2)
            theta_perturbed_rad = math.atan(F1_perturbed / (self.m * self.g))
            theta_perturbed = math.degrees(theta_perturbed_rad)

            # Check trends: if v increases, T and θ should increase
            if self.T is not None and self.theta is not None:
                t_trend = T_perturbed > self.T or self._is_close(T_perturbed, self.T)
                self._log(f"T increases with v: {self.T} -> {T_perturbed}", t_trend)
                theta_trend = theta_perturbed > self.theta or self._is_close(theta_perturbed, self.theta)
                self._log(f"θ increases with v: {self.theta} -> {theta_perturbed}", theta_trend)
            else:
                self._log("Missing T or θ for v perturbation test", False)

            # Perturb mass m (increase by 10%)
            m_original = self.m
            m_perturbed = m_original * 1.1
            # Recompute with original v
            F1_original = self.k * self.v**2
            T_perturbed_m = math.sqrt((m_perturbed * self.g)**2 + F1_original**2)
            theta_perturbed_rad_m = math.atan(F1_original / (m_perturbed * self.g))
            theta_perturbed_m = math.degrees(theta_perturbed_rad_m)

            # Check trends: if m increases, T should increase, θ should decrease
            if self.T is not None and self.theta is not None:
                t_trend_m = T_perturbed_m > self.T or self._is_close(T_perturbed_m, self.T)
                self._log(f"T increases with m: {self.T} -> {T_perturbed_m}", t_trend_m)
                theta_trend_m = theta_perturbed_m < self.theta or self._is_close(theta_perturbed_m, self.theta)
                self._log(f"θ decreases with m: {self.theta} -> {theta_perturbed_m}", theta_trend_m)
            else:
                self._log("Missing T or θ for m perturbation test", False)

        except Exception as e:
            self._log(f"Exception in test_monotonicity: {e}", False)

    def run_all_tests(self):
        """Execute all test suites and return (success_bool, logs_list)."""
        self.logs = ["=== Physics Checker Start ==="]
        self.all_tests_passed = True

        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()

        self.logs.append(f"=== All Tests {'PASSED' if self.all_tests_passed else 'FAILED'} ===")
        return self.all_tests_passed, self.logs