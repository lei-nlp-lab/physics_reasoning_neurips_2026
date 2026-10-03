import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Never assume direct attribute access; use getattr with fallbacks.
        """
        self.solution = solution_instance

        # Extract inputs with safe fallbacks
        self.m = self._safe_get('m', 0.0)
        self.S = self._safe_get('S', 0.0)
        self.V1 = self._safe_get('V1', 0.0)
        self.T1 = self._safe_get('T1', 0.0)
        self.g = self._safe_get('g', 9.8)
        self.p0 = self._safe_get('p0', 0.0)
        self.V3 = self._safe_get('V3', 0.0)  # Needed for T3 calculation

        # Extract results with safe fallbacks
        self.p1 = self._safe_get('p_1', 0.0)
        self.p2 = self._safe_get('p_2', 0.0)
        self.V2 = self._safe_get('V_2', 0.0)
        self.p3 = self._safe_get('p_3', 0.0)
        self.T3 = self._safe_get('T_3', 0.0)

        # For symbolic computations
        self.m_sym = sp.symbols('m', positive=True)
        self.S_sym = sp.symbols('S', positive=True)
        self.V1_sym = sp.symbols('V1', positive=True)
        self.T1_sym = sp.symbols('T1', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        self.p0_sym = sp.symbols('p0', positive=True)
        self.V3_sym = sp.symbols('V3', positive=True)
        self.p1_sym = sp.symbols('p1', positive=True)
        self.p2_sym = sp.symbols('p2', positive=True)
        self.V2_sym = sp.symbols('V2', positive=True)
        self.p3_sym = sp.symbols('p3', positive=True)
        self.T3_sym = sp.symbols('T3', positive=True)

    def _safe_get(self, key, default):
        """Safely retrieve attribute from solution instance or its results."""
        # Try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float)) else val

        # Try results dictionary
        results = getattr(self.solution, 'results', {})
        val = results.get(key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float)) else val

        # Try inputs dictionary
        inputs = getattr(self.solution, 'inputs', {})
        val = inputs.get(key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float)) else val

        return default

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Boyle's Law between state (a) and (b) (assumed isothermal).
        2. Ideal Gas Law between state (a) and (c) (same amount of gas).
        """
        logs = []
        success = True

        # 1. Boyle's Law: p1 * V1 = p2 * V2 (constant temperature)
        if all(v != 0.0 for v in [self.p1, self.V1, self.p2, self.V2]):
            left = self.p1 * self.V1
            right = self.p2 * self.V2
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Boyle's Law violated. p1*V1={left}, p2*V2={right}")
                success = False
            else:
                logs.append("PASS: Boyle's Law holds between state (a) and (b).")
        else:
            logs.append("SKIP: Boyle's Law check due to missing data.")

        # 2. Ideal Gas Law: p1*V1/T1 = p3*V3/T3 (same n)
        if all(v != 0.0 for v in [self.p1, self.V1, self.T1, self.p3, self.V3, self.T3]):
            left = self.p1 * self.V1 / self.T1
            right = self.p3 * self.V3 / self.T3
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Ideal Gas Law violated. p1*V1/T1={left}, p3*V3/T3={right}")
                success = False
            else:
                logs.append("PASS: Ideal Gas Law holds between state (a) and (c).")
        else:
            logs.append("SKIP: Ideal Gas Law check due to missing data.")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the derived formulas from steps.
        """
        logs = []
        success = True

        # Expected formulas from steps analysis
        # p1 = p0 + m*g/S
        if all(v != 0.0 for v in [self.p0, self.m, self.g, self.S]):
            expected_p1 = self.p0 + self.m * self.g / self.S
            if not math.isclose(self.p1, expected_p1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: p1 mismatch. Expected {expected_p1}, got {self.p1}")
                success = False
            else:
                logs.append(f"PASS: p1 = p0 + m*g/S holds.")
        else:
            logs.append("SKIP: p1 check due to missing inputs.")

        # p2 = p0 - m*g/S
        if all(v != 0.0 for v in [self.p0, self.m, self.g, self.S]):
            expected_p2 = self.p0 - self.m * self.g / self.S
            if not math.isclose(self.p2, expected_p2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: p2 mismatch. Expected {expected_p2}, got {self.p2}")
                success = False
            else:
                logs.append(f"PASS: p2 = p0 - m*g/S holds.")
        else:
            logs.append("SKIP: p2 check due to missing inputs.")

        # p3 = p0
        if self.p0 != 0.0:
            if not math.isclose(self.p3, self.p0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: p3 mismatch. Expected {self.p0}, got {self.p3}")
                success = False
            else:
                logs.append("PASS: p3 = p0 holds.")
        else:
            logs.append("SKIP: p3 check due to missing p0.")

        # V2 = (p1/p2) * V1
        if all(v != 0.0 for v in [self.p1, self.p2, self.V1]):
            expected_V2 = (self.p1 / self.p2) * self.V1
            if not math.isclose(self.V2, expected_V2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: V2 mismatch. Expected {expected_V2}, got {self.V2}")
                success = False
            else:
                logs.append(f"PASS: V2 = (p1/p2)*V1 holds.")
        else:
            logs.append("SKIP: V2 check due to missing data.")

        # T3 = (p3 * V3 * T1) / (p1 * V1)
        if all(v != 0.0 for v in [self.p3, self.V3, self.T1, self.p1, self.V1]):
            expected_T3 = (self.p3 * self.V3 * self.T1) / (self.p1 * self.V1)
            if not math.isclose(self.T3, expected_T3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T3 mismatch. Expected {expected_T3}, got {self.T3}")
                success = False
            else:
                logs.append(f"PASS: T3 = (p3*V3*T1)/(p1*V1) holds.")
        else:
            logs.append("SKIP: T3 check due to missing data.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (mass m) and check if the results change in the expected direction.
        We will compute the expected change using the derived formulas, then restore the original values.
        """
        logs = []
        success = True

        original_m = self.m
        if original_m == 0.0:
            logs.append("SKIP: Monotonicity test because m=0.")
            return success, logs

        # Perturbation: increase m by 1%
        perturbed_m = original_m * 1.01
        self.m = perturbed_m

        # Recompute expected values using formulas
        perturbed_p1 = self.p0 + perturbed_m * self.g / self.S
        perturbed_p2 = self.p0 - perturbed_m * self.g / self.S
        perturbed_V2 = (perturbed_p1 / perturbed_p2) * self.V1
        perturbed_T3 = (self.p3 * self.V3 * self.T1) / (perturbed_p1 * self.V1)

        # Restore original m
        self.m = original_m

        # Check trends
        # p1 should increase with m
        if perturbed_p1 > self.p1:
            logs.append("PASS: p1 increases with m (as expected).")
        else:
            logs.append("FAIL: p1 does not increase with m.")
            success = False

        # p2 should decrease with m
        if perturbed_p2 < self.p2:
            logs.append("PASS: p2 decreases with m (as expected).")
        else:
            logs.append("FAIL: p2 does not decrease with m.")
            success = False

        # V2 should increase because p1/p2 increases (since p1 increases and p2 decreases)
        if perturbed_V2 > self.V2:
            logs.append("PASS: V2 increases with m (as expected).")
        else:
            logs.append("FAIL: V2 does not increase with m.")
            success = False

        # T3 should decrease because p1 increases in the denominator
        if perturbed_T3 < self.T3:
            logs.append("PASS: T3 decreases with m (as expected).")
        else:
            logs.append("FAIL: T3 does not decrease with m.")
            success = False

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
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
                success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs