import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        self.logs = []

        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 1.0))
        self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'results', {}).get('q', 1.0))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'results', {}).get('L', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 9.8))
        self.theta = getattr(solution_instance, 'theta', getattr(solution_instance, 'results', {}).get('theta', math.radians(30)))

        # Extract results with safe defaults
        results_dict = getattr(solution_instance, 'results', {})
        
        # Numeric values
        self.E = self._safe_float(results_dict.get('E', 0.0))
        self.v = self._safe_float(results_dict.get('v', 0.0))
        self.T = self._safe_float(results_dict.get('T', 0.0))
        self.frac12mv2 = self._safe_float(results_dict.get('frac12mv2', 0.0))
        self.F_net_c = self._safe_float(results_dict.get('F_net_c', 0.0))
        
        # String value for direction
        self.E_direction = results_dict.get('E_direction', "")
        # If it's a numeric code, convert to string representation
        if isinstance(self.E_direction, (int, float)):
            self.E_direction = "right" if self.E_direction > 0 else "left"

        # Convert theta to radians if it's in degrees (assuming > 2 indicates degrees)
        if abs(self.theta) > 2:
            self.theta_rad = math.radians(self.theta)
        else:
            self.theta_rad = self.theta

    def _safe_float(self, value):
        """Safely convert to float, returning 0.0 on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        try:
            a_val = float(a)
            b_val = float(b)
            return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Check fundamental physics laws: equilibrium, Newton's 2nd law, energy conservation."""
        try:
            # 1. Equilibrium at point A: T0*sinθ = qE, T0*cosθ = mg => E = (mg tanθ)/q
            expected_E = self.m * self.g * math.tan(self.theta_rad) / self.q
            if not self._is_close(self.E, expected_E):
                self._log(f"Equilibrium E mismatch: got {self.E}, expected {expected_E}", False)
                return False
            self._log("Equilibrium at point A holds (E = mg tanθ / q)", True)

            # 2. Newton's 2nd law at point C: T - mg = m*v^2/L (centripetal)
            centripetal = self.m * self.v**2 / self.L
            if not self._is_close(self.F_net_c, centripetal):
                self._log(f"Centripetal force mismatch: F_net_c={self.F_net_c}, m*v^2/L={centripetal}", False)
                return False
            self._log("Newton's 2nd law at point C holds (T - mg = m*v^2/L)", True)

            # 3. Energy conservation: mgL - qEL = 1/2 m v^2 (from steps analysis)
            left_energy = self.m * self.g * self.L - self.q * self.E * self.L
            right_energy = 0.5 * self.m * self.v**2
            if not self._is_close(left_energy, right_energy):
                self._log(f"Energy conservation mismatch: mgL - qEL={left_energy}, 1/2 m v^2={right_energy}", False)
                return False
            self._log("Work-energy theorem holds (mgL - qEL = 1/2 m v^2)", True)

            return True
        except Exception as e:
            self._log(f"Invariant test failed with error: {traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Step 1: E magnitude
            expected_E = self.m * self.g * math.tan(self.theta_rad) / self.q
            if not self._is_close(self.E, expected_E):
                self._log(f"E magnitude inconsistent: got {self.E}, expected {expected_E}", False)
                return False
            self._log(f"E magnitude {self.E} matches expression", True)

            # Step 2: E direction
            if "right" not in str(self.E_direction).lower():
                self._log(f"E direction mismatch: got '{self.E_direction}', expected 'horizontally to the right'", False)
                return False
            self._log(f"E direction '{self.E_direction}' matches", True)

            # Step 5: v magnitude
            expected_v = math.sqrt(2 * self.g * self.L * (1 - math.sqrt(3)/3))
            if not self._is_close(self.v, expected_v):
                self._log(f"v magnitude inconsistent: got {self.v}, expected {expected_v}", False)
                return False
            self._log(f"v magnitude {self.v} matches expression", True)

            # Step 6: T magnitude
            expected_T = 3 * self.m * self.g - (2 * math.sqrt(3)/3) * self.m * self.g
            if not self._is_close(self.T, expected_T):
                self._log(f"T magnitude inconsistent: got {self.T}, expected {expected_T}", False)
                return False
            self._log(f"T magnitude {self.T} matches expression", True)

            # Consistency between T, v, and centripetal force
            centripetal = self.m * self.v**2 / self.L
            expected_T_from_centripetal = self.m * self.g + centripetal
            if not self._is_close(self.T, expected_T_from_centripetal):
                self._log(f"T from centripetal mismatch: got {self.T}, expected {expected_T_from_centripetal}", False)
                return False
            self._log("T consistent with centripetal force (T = mg + m*v^2/L)", True)

            return True
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """Perturb mass m and check trends, then restore."""
        try:
            original_m = self.m
            perturbed_m = original_m * 1.01  # +1% perturbation

            # Recalculate expected values with perturbed m
            # E ∝ m (from E = mg tanθ / q)
            expected_E_perturbed = perturbed_m * self.g * math.tan(self.theta_rad) / self.q
            # v independent of m (from v = sqrt(2gL(1-√3/3)))
            expected_v_perturbed = math.sqrt(2 * self.g * self.L * (1 - math.sqrt(3)/3))
            # T ∝ m (from T = mg(3 - 2√3/3))
            expected_T_perturbed = 3 * perturbed_m * self.g - (2 * math.sqrt(3)/3) * perturbed_m * self.g

            # Check monotonic trends
            if not (expected_E_perturbed > self.E and self._is_close(expected_v_perturbed, self.v) and expected_T_perturbed > self.T):
                self._log(f"Monotonicity failed: E and T should increase with m, v should remain constant", False)
                return False

            # Restore original m
            self.m = original_m
            self._log("Monotonicity test passed: E and T increase with m, v constant", True)
            return True
        except Exception as e:
            self.m = original_m  # Ensure restoration even on error
            self._log(f"Monotonicity test failed with error: {traceback.format_exc()}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        overall_success = True

        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self._log(f"Starting {test_name}...")
            if not test_func():
                overall_success = False
                self._log(f"{test_name} failed.", False)
            else:
                self._log(f"{test_name} passed.", True)

        return overall_success, self.logs