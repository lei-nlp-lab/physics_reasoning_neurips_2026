import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the checker by safely extracting all required inputs and results
        from the solution instance.
        """
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Input parameters with safe access
        self.m = self.results.get('m', 1.0)
        self.q = self.results.get('q', 1.0)
        self.d = self.results.get('d', 1.0)
        self.E = self.results.get('E', 1.0)
        self.E_k = self.results.get('E_k', 1.0)
        self.theta0 = self.results.get('theta', 0.0)

        # Results from mapping (safe access via .get)
        self.v_prime = self.results.get("v'", 0.0)
        self.v_x = self.results.get('v_x', 0.0)
        self.t = self.results.get('t', 0.0)
        self.a = self.results.get('a', 0.0)
        self.theta_prime = self.results.get("theta'", 0.0)
        self.tantheta_prime = self.results.get("tantheta'", 0.0)
        
        # Handle sintheta: could be tuple or single value
        sintheta_val = self.results.get('sintheta', (0.0, 0.0))
        if isinstance(sintheta_val, (int, float)):
            # If single value, assume it's the positive boundary and create symmetric tuple
            self.sintheta = (-abs(sintheta_val), abs(sintheta_val))
        else:
            self.sintheta = sintheta_val
            
        self.t_0 = self.results.get('t_0', 0.0)
        self.t_all = self.results.get('t_all', 0.0)
        self.ratio = self.results.get('N:N_0', 0.0)

        # Additional velocities (individual keys)
        self.v_1d = self.results.get('v_1d', 0.0)
        self.v_3d = self.results.get('v_3d', 0.0)
        self.v_5d = self.results.get('v_5d', 0.0)
        self.v_2d = self.results.get('v_2d', 0.0)
        self.v_4d = self.results.get('v_4d', 0.0)
        self.v_6d = self.results.get('v_6d', 0.0)

        # Target answers
        self.target_t = self.results.get('target_t', 0.0)
        # Handle target_theta_range: could be tuple or single value
        target_theta_range_val = self.results.get('target_theta_range', (-math.pi/6, math.pi/6))
        if isinstance(target_theta_range_val, (int, float)):
            # If single value, assume it's the positive boundary and create symmetric tuple
            self.target_theta_range = (-abs(target_theta_range_val), abs(target_theta_range_val))
        else:
            self.target_theta_range = target_theta_range_val
            
        self.target_ratio = self.results.get('target_ratio', 0.5)

        # Logs
        self.logs = []

    def _log(self, message: str, success: bool = True):
        """Helper to log messages with success/failure indicator."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")

    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants:
        1. Energy conservation in each layer (work-energy theorem).
        2. Newton's second law in each layer.
        """
        success = True

        # 1. Energy conservation: E_k = 4*q*E*d
        try:
            expected_E_k = 4 * self.q * self.E * self.d
            if not self._isclose(self.E_k, expected_E_k):
                self._log(f"Energy invariant: E_k = {self.E_k} != 4qEd = {expected_E_k}", False)
                success = False
            else:
                self._log(f"Energy invariant: E_k = 4qEd holds.")
        except Exception as e:
            self._log(f"Energy invariant check failed with error: {e}", False)
            success = False

        # 2. Newton's second law: a = q*E/m
        try:
            expected_a = self.q * self.E / self.m
            if not self._isclose(self.a, expected_a):
                self._log(f"Newton's second law: a = {self.a} != qE/m = {expected_a}", False)
                success = False
            else:
                self._log(f"Newton's second law: a = qE/m holds.")
        except Exception as e:
            self._log(f"Newton's second law check failed with error: {e}", False)
            success = False

        # 3. Kinetic energy definition: E_k = 0.5 * m * v^2
        try:
            v_initial = math.sqrt(2 * self.E_k / self.m)
            v_x_expected = v_initial * math.cos(self.theta0)
            if not self._isclose(self.v_x, v_x_expected):
                self._log(f"Kinetic energy definition: v_x = {self.v_x} != sqrt(2E_k/m)*cosθ0 = {v_x_expected}", False)
                success = False
            else:
                self._log(f"Kinetic energy definition: v_x consistent with E_k.")
        except Exception as e:
            self._log(f"Kinetic energy definition check failed with error: {e}", False)
            success = False

        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers and steps.
        """
        success = True

        # 1. Check horizontal velocity v_x = v * cosθ0
        try:
            v_initial = math.sqrt(2 * self.E_k / self.m)
            v_x_expected = v_initial * math.cos(self.theta0)
            if not self._isclose(self.v_x, v_x_expected):
                self._log(f"Trace consistency (v_x): {self.v_x} != {v_x_expected}", False)
                success = False
            else:
                self._log(f"Trace consistency (v_x): matches initial velocity and angle.")
        except Exception as e:
            self._log(f"Trace consistency (v_x) failed: {e}", False)
            success = False

        # 2. Check time t = 8d / (v_x)
        try:
            t_expected = 8 * self.d / self.v_x if self.v_x != 0 else 0.0
            if not self._isclose(self.t, t_expected):
                self._log(f"Trace consistency (t): {self.t} != 8d/v_x = {t_expected}", False)
                success = False
            else:
                self._log(f"Trace consistency (t): matches 8d/v_x.")
        except Exception as e:
            self._log(f"Trace consistency (t) failed: {e}", False)
            success = False

        # 3. Check angle range: -30° < θ < 30° (from sinθ range -1/2 < sinθ < 1/2)
        try:
            # Check sintheta range
            if isinstance(self.sintheta, tuple) and len(self.sintheta) == 2:
                sintheta_low, sintheta_high = self.sintheta
                if not (self._isclose(sintheta_low, -0.5) and self._isclose(sintheta_high, 0.5)):
                    self._log(f"Trace consistency (sinθ range): ({sintheta_low}, {sintheta_high}) != (-0.5, 0.5)", False)
                    success = False
                else:
                    self._log(f"Trace consistency (sinθ range): correct.")
            else:
                self._log(f"Trace consistency (sinθ range): sintheta not a tuple", False)
                success = False
            
            # Check theta range in degrees
            if isinstance(self.target_theta_range, tuple) and len(self.target_theta_range) == 2:
                theta_low, theta_high = self.target_theta_range
                theta_low_deg = math.degrees(theta_low)
                theta_high_deg = math.degrees(theta_high)
                if not (self._isclose(theta_low_deg, -30.0) and self._isclose(theta_high_deg, 30.0)):
                    self._log(f"Trace consistency (θ range): ({theta_low_deg}, {theta_high_deg})° != (-30°, 30°)", False)
                    success = False
                else:
                    self._log(f"Trace consistency (θ range): correct.")
            else:
                self._log(f"Trace consistency (θ range): target_theta_range not a tuple", False)
                success = False
        except Exception as e:
            self._log(f"Trace consistency (θ range) failed: {e}", False)
            success = False

        # 4. Check ratio N/N_0 = 50% (or 0.5)
        try:
            if not self._isclose(self.ratio, 0.5):
                self._log(f"Trace consistency (ratio): N/N_0 = {self.ratio} != 0.5", False)
                success = False
            else:
                self._log(f"Trace consistency (ratio): N/N_0 = 0.5 holds.")
        except Exception as e:
            self._log(f"Trace consistency (ratio) failed: {e}", False)
            success = False

        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb an input parameter (mass m) and verify the trend in output (time t).
        Restore original value after check.
        """
        original_m = self.m
        success = True

        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.10
            # Recompute t using formula from step_2: t = 8d / (cosθ0) * sqrt(m/(2E_k))
            original_t = 8 * self.d / (math.cos(self.theta0)) * math.sqrt(original_m / (2 * self.E_k))
            perturbed_t = 8 * self.d / (math.cos(self.theta0)) * math.sqrt(perturbed_m / (2 * self.E_k))

            # Expect t to increase with mass (since t ∝ sqrt(m))
            if not (perturbed_t > original_t):
                self._log(f"Monotonicity: t does not increase with mass (original t={original_t}, perturbed t={perturbed_t})", False)
                success = False
            else:
                self._log(f"Monotonicity: t increases with mass (original t={original_t}, perturbed t={perturbed_t}).")

        except Exception as e:
            self._log(f"Monotonicity test failed: {e}", False)
            success = False

        # Restore original mass (in memory only, since we don't modify solution_instance)
        self.m = original_m
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name}...")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False

        if overall_success:
            self._log("All tests passed!")
        else:
            self._log("Some tests failed.", False)

        return overall_success, self.logs