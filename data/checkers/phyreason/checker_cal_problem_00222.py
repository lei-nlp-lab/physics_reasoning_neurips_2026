import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Input parameters (with safe defaults)
        self.h = 2.0  # mercury column height (cm)
        self.h1 = 2.0 # distance from mercury top to tube opening (cm)
        self.p0 = 76.0 # atmospheric pressure (cmHg)
        self.T0 = 296.0 # ambient temperature (K)
        self.S = 1.0   # cross-sectional area (arbitrary units, will cancel)

        # Results from solution (safe extraction with fallbacks)
        self.p = float(getattr(solution_instance, 'p', self.results.get('p', 0.0)))
        self.p1 = float(getattr(solution_instance, 'p_1', self.results.get('p_1', self.results.get('p1', 0.0))))
        self.L = float(getattr(solution_instance, 'L', self.results.get('L', 0.0)))
        self.T = float(getattr(solution_instance, 'T', self.results.get('T', 0.0)))

        # Calculate volumes from L (using formulas from step 4)
        self.V = self.S * (self.L - self.h1 - self.h) if self.L > 0 else 0.0
        self.V1 = self.S * (self.L - self.h) if self.L > 0 else 0.0

        # Log storage
        self.logs = []

    def _log(self, msg, success=None):
        """Helper to log messages and optionally track success."""
        self.logs.append(msg)
        return success

    def test_invariants(self):
        """
        Test physical invariants: Force Equilibrium and Gas Laws.
        """
        logs = []
        try:
            # 1. Force Equilibrium: p = p0 + ρgh (with ρgh = 2 cmHg)
            expected_p = self.p0 + self.h
            if not math.isclose(self.p, expected_p, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Initial pressure p={self.p} cmHg, expected {expected_p} cmHg")
                return self._log("Force equilibrium (initial) violated", False)
            logs.append(f"PASS: Initial pressure p={self.p} ≈ {expected_p} cmHg")

            # 2. Force Equilibrium: p1 = p0 - ρgh
            expected_p1 = self.p0 - self.h
            if not math.isclose(self.p1, expected_p1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final pressure p1={self.p1} cmHg, expected {expected_p1} cmHg")
                return self._log("Force equilibrium (final) violated", False)
            logs.append(f"PASS: Final pressure p1={self.p1} ≈ {expected_p1} cmHg")

            # 3. Boyle's Law (isothermal inversion): p * V = p1 * V1
            if self.V > 0 and self.V1 > 0:
                left = self.p * self.V
                right = self.p1 * self.V1
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Boyle's law: p*V={left}, p1*V1={right}")
                    return self._log("Boyle's law violated", False)
                logs.append(f"PASS: Boyle's law holds: p*V ≈ p1*V1")

            # 4. Charles/Gay-Lussac Law (step 6): V/T0 = V1/T  (constant pressure during heating?)
            if self.T > 0 and self.V > 0 and self.V1 > 0:
                left = self.V / self.T0
                right = self.V1 / self.T
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Volume-temperature law: V/T0={left}, V1/T={right}")
                    return self._log("Volume-temperature law violated", False)
                logs.append(f"PASS: Volume-temperature law: V/T0 ≈ V1/T")

            return self._log("\n".join(logs), True)

        except Exception as e:
            return self._log(f"Error in test_invariants: {traceback.format_exc()}", False)

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        try:
            # 1. Recalculate L from Boyle's law using pressures and geometry
            # Equation: p*(L - h1 - h) = p1*(L - h)
            if self.p > 0 and self.p1 > 0:
                numerator = self.p * (self.h1 + self.h) - self.p1 * self.h
                denominator = self.p - self.p1
                if abs(denominator) > 1e-9:
                    L_calc = numerator / denominator
                    if not math.isclose(self.L, L_calc, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: L from Boyle's law: calculated {L_calc}, given {self.L}")
                        return self._log("Trace inconsistency in L", False)
                    logs.append(f"PASS: L={self.L} consistent with Boyle's law")
                else:
                    logs.append("WARN: Denominator zero in L calculation")

            # 2. Recalculate T from volume-temperature law
            # Equation: V/T0 = V1/T  -> T = T0 * (V1 / V)
            if self.V > 0:
                T_calc = self.T0 * (self.V1 / self.V)
                if not math.isclose(self.T, T_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T from volume law: calculated {T_calc}, given {self.T}")
                    return self._log("Trace inconsistency in T", False)
                logs.append(f"PASS: T={self.T} consistent with volume-temperature law")

            return self._log("\n".join(logs), True)

        except Exception as e:
            return self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)

    def test_monotonicity(self):
        """
        Perturb mercury height h and check expected trends.
        """
        logs = []
        original_h = self.h
        try:
            # Perturbation: increase h by 5%
            perturbed_h = original_h * 1.05
            logs.append(f"Perturbation: h from {original_h} cm to {perturbed_h} cm")

            # Recalculate pressures
            p_perturbed = self.p0 + perturbed_h
            p1_perturbed = self.p0 - perturbed_h

            # Recalculate L from Boyle's law (using same formula as in trace consistency)
            numerator = p_perturbed * (self.h1 + perturbed_h) - p1_perturbed * perturbed_h
            denominator = p_perturbed - p1_perturbed
            if abs(denominator) < 1e-9:
                return self._log("Denominator zero in perturbed L calculation", False)

            L_perturbed = numerator / denominator

            # Recalculate volumes
            V_perturbed = self.S * (L_perturbed - self.h1 - perturbed_h)
            V1_perturbed = self.S * (L_perturbed - perturbed_h)

            # Recalculate T from volume-temperature law
            T_perturbed = self.T0 * (V1_perturbed / V_perturbed)

            # Expected trends (derived from physical reasoning):
            # - If h increases, the pressure difference increases, so L should decrease.
            # - If h increases, the volume ratio V1/V changes, and T should increase.
            if not (L_perturbed < self.L):
                logs.append(f"FAIL: L did not decrease when h increased (L_perturbed={L_perturbed}, original={self.L})")
                return self._log("Monotonicity trend for L violated", False)
            logs.append(f"PASS: L decreases when h increases (as expected)")

            if not (T_perturbed > self.T):
                logs.append(f"FAIL: T did not increase when h increased (T_perturbed={T_perturbed}, original={self.T})")
                return self._log("Monotonicity trend for T violated", False)
            logs.append(f"PASS: T increases when h increases (as expected)")

            return self._log("\n".join(logs), True)

        except Exception as e:
            return self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        success = True

        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self.logs.append(f"\n=== {test_name} ===")
            test_success = test_func()
            if isinstance(test_success, tuple):
                # Some tests return (bool, logs)
                test_success, extra_logs = test_success
                if extra_logs:
                    self.logs.append(extra_logs)
            success = success and (test_success is True)

        self.logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs