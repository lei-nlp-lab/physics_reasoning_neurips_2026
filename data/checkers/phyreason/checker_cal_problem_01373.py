import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all input parameters and results from the solution instance.
        Uses getattr and .get() to avoid AttributeError in sandbox environments.
        """
        # Extract results from solution_instance.results with fallbacks
        self.v = getattr(solution_instance, 'v', None) or solution_instance.results.get('v')
        # Handle T' which may be stored as "T'" or "T_prime"
        self.T_prime = getattr(solution_instance, 'T_prime', None) or \
                       solution_instance.results.get("T'") or \
                       solution_instance.results.get('T_prime')
        self.a = getattr(solution_instance, 'a', None) or solution_instance.results.get('a')
        self.h = getattr(solution_instance, 'h', None) or solution_instance.results.get('h')
        
        # Use standard Earth constants for checking (since they're known)
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        self.M = 5.972e24     # kg (mass of Earth)
        self.R = 6.371e6      # m (radius of Earth)
        self.T = 86164        # s (Earth's sidereal rotation period)

        # Store original values for monotonicity test restoration
        self._original_M = self.M

    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Gravitational force = centripetal force for satellite A.
        2. Period-velocity-radius relationship for satellite A.
        3. Centripetal acceleration formula.
        4. Geostationary orbit condition.
        """
        logs = []
        success = True

        # 1. Satellite A: F_grav = m*v^2/R  -> v^2 = G*M/R
        if self.v is not None:
            expected_v2 = self.G * self.M / self.R
            actual_v2 = self.v ** 2
            if not math.isclose(actual_v2, expected_v2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant failed: v² = {actual_v2:.3e}, expected G*M/R = {expected_v2:.3e}")
                success = False
            else:
                logs.append("Invariant (v² = G*M/R) passed.")
        else:
            logs.append("Missing data for v² = G*M/R check.")
            success = False

        # 2. Satellite A: T' = 2πR/v
        if all(val is not None for val in [self.T_prime, self.v]):
            expected_T_prime = 2 * math.pi * self.R / self.v
            if not math.isclose(self.T_prime, expected_T_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant failed: T' = {self.T_prime:.3e}, expected 2πR/v = {expected_T_prime:.3e}")
                success = False
            else:
                logs.append("Invariant (T' = 2πR/v) passed.")
        else:
            logs.append("Missing data for T' = 2πR/v check.")
            success = False

        # 3. Satellite A: a = v²/R = G*M/R²
        if all(val is not None for val in [self.a, self.v]):
            expected_a_from_v = self.v ** 2 / self.R
            expected_a_from_GM = self.G * self.M / self.R ** 2
            # Check against v²/R
            if not math.isclose(self.a, expected_a_from_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant failed: a = {self.a:.3e}, expected v²/R = {expected_a_from_v:.3e}")
                success = False
            else:
                logs.append("Invariant (a = v²/R) passed.")
            # Cross-check with G*M/R²
            if not math.isclose(self.a, expected_a_from_GM, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Cross-check failed: a = {self.a:.3e} ≠ G*M/R² = {expected_a_from_GM:.3e}")
                success = False
        else:
            logs.append("Missing data for a = v²/R check.")
            success = False

        # 4. Geostationary satellite: (R+h)³ = G*M*T²/(4π²)
        if self.h is not None:
            expected_h = (self.G * self.M * self.T ** 2 / (4 * math.pi ** 2)) ** (1/3) - self.R
            if not math.isclose(self.h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant failed: h = {self.h:.3e}, expected (G*M*T²/(4π²))^(1/3)-R = {expected_h:.3e}")
                success = False
            else:
                logs.append("Invariant (geostationary condition) passed.")
        else:
            logs.append("Missing data for geostationary invariant check.")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers using formulas from steps analysis.
        """
        logs = []
        success = True

        # Check v = sqrt(G*M/R)
        if self.v is not None:
            expected_v = math.sqrt(self.G * self.M / self.R)
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: v = {self.v:.3e} ≠ sqrt(G*M/R) = {expected_v:.3e}")
                success = False
            else:
                logs.append("Trace consistency for v passed.")

        # Check T' = sqrt(4π²R³/(G*M))
        if self.T_prime is not None:
            expected_T_prime = math.sqrt(4 * math.pi ** 2 * self.R ** 3 / (self.G * self.M))
            if not math.isclose(self.T_prime, expected_T_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: T' = {self.T_prime:.3e} ≠ sqrt(4π²R³/(G*M)) = {expected_T_prime:.3e}")
                success = False
            else:
                logs.append("Trace consistency for T' passed.")

        # Check a = G*M/R²
        if self.a is not None:
            expected_a = self.G * self.M / self.R ** 2
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: a = {self.a:.3e} ≠ G*M/R² = {expected_a:.3e}")
                success = False
            else:
                logs.append("Trace consistency for a passed.")

        # Check h = (G*M*T²/(4π²))^(1/3) - R
        if self.h is not None:
            expected_h = (self.G * self.M * self.T ** 2 / (4 * math.pi ** 2)) ** (1/3) - self.R
            if not math.isclose(self.h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: h = {self.h:.3e} ≠ (G*M*T²/(4π²))^(1/3)-R = {expected_h:.3e}")
                success = False
            else:
                logs.append("Trace consistency for h passed.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb Earth's mass M by +1%, recalculate using formulas, check expected trends, then restore.
        Expected trends (from KB):
        - v increases with M
        - T' decreases with M
        - a increases with M
        - h increases with M
        """
        logs = []
        success = True

        if None in [self.v, self.T_prime, self.a, self.h]:
            logs.append("Missing data for monotonicity test.")
            return False, logs

        # Perturb M by +1%
        perturbed_M = self.M * 1.01

        # Recalculate using formulas (do not modify instance attributes)
        v_pert = math.sqrt(self.G * perturbed_M / self.R)
        T_prime_pert = math.sqrt(4 * math.pi ** 2 * self.R ** 3 / (self.G * perturbed_M))
        a_pert = self.G * perturbed_M / self.R ** 2
        h_pert = (self.G * perturbed_M * self.T ** 2 / (4 * math.pi ** 2)) ** (1/3) - self.R

        # Check trends
        if not v_pert > self.v:
            logs.append(f"Monotonicity failed: v does not increase with M (original {self.v:.3e}, perturbed {v_pert:.3e})")
            success = False
        else:
            logs.append("Monotonicity for v: passed (increases with M).")

        if not T_prime_pert < self.T_prime:
            logs.append(f"Monotonicity failed: T' does not decrease with M (original {self.T_prime:.3e}, perturbed {T_prime_pert:.3e})")
            success = False
        else:
            logs.append("Monotonicity for T': passed (decreases with M).")

        if not a_pert > self.a:
            logs.append(f"Monotonicity failed: a does not increase with M (original {self.a:.3e}, perturbed {a_pert:.3e})")
            success = False
        else:
            logs.append("Monotonicity for a: passed (increases with M).")

        if not h_pert > self.h:
            logs.append(f"Monotonicity failed: h does not increase with M (original {self.h:.3e}, perturbed {h_pert:.3e})")
            success = False
        else:
            logs.append("Monotonicity for h: passed (increases with M).")

        # Restore original M (though we never changed it, just for completeness)
        self.M = self._original_M

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and consolidated logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"{test_name} raised exception: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs