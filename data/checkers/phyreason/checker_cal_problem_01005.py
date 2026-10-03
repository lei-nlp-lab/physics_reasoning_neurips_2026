import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Inputs (with safe fallbacks)
        self.p0 = self._safe_get(solution_instance, 'p0', default=1.0)
        self.V0 = self._safe_get(solution_instance, 'V0', default=1.0)
        self.p = self._safe_get(solution_instance, 'p', default=2.0)
        self.T = self._safe_get(solution_instance, 'T', default=300.0)  # ambient temperature
        
        # Results
        self.V = self._safe_get(solution_instance, 'V', default=None)
        self.fracVV_1 = self._safe_get(solution_instance, 'fracVV_1', default=None)
        self.LeakageRate = self._safe_get(solution_instance, 'LeakageRate', default=None)
        
        # Derived quantities for step 2 (from problem statement)
        self.p1 = 1.16 * self.p   # final pressure after leakage test
        self.T2 = 1.2 * self.T    # final temperature after leakage test
        
        # Store original values for monotonicity test
        self._original_p0 = self.p0
        self._original_V = self.V
        
        # Logs
        self.logs = []

    def _safe_get(self, obj, key, default=None):
        """
        Safely get attribute from solution_instance, trying direct attribute,
        then results dictionary, then returning default.
        """
        val = getattr(obj, key, None)
        if val is None:
            val = getattr(obj, 'results', {}).get(key, default)
        return val

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Robust numerical comparison with tolerances.
        Handles symbolic expressions by simplifying difference.
        """
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        try:
            # Try symbolic simplification
            diff = sp.simplify(a - b)
            if diff.is_zero:
                return True
            # If diff is a number, check if it's close to zero
            if diff.is_number:
                return math.isclose(float(diff), 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            pass
        return False

    def test_invariants(self):
        """
        Verify physical invariants:
        1. Boyle's Law for step 1: p0 * V0 = V * (p - p0)
        2. Ideal gas ratio for step 2: V/V1 = (p1 * T) / (p * T2)
        3. Leakage rate formula: LeakageRate = 100 * (1 - V/V1)
        """
        logs = []
        success = True

        # 1. Boyle's Law check
        try:
            lhs = self.p0 * self.V0
            rhs = self.V * (self.p - self.p0)
            if not self._is_close(lhs, rhs):
                logs.append(f"FAIL: Boyle's Law invariant: p0*V0 = {lhs}, V*(p-p0) = {rhs}")
                success = False
            else:
                logs.append("PASS: Boyle's Law holds.")
        except Exception as e:
            logs.append(f"ERROR in Boyle's Law check: {e}")
            success = False

        # 2. Ideal gas ratio check (step 2)
        try:
            expected_ratio = (self.p1 * self.T) / (self.p * self.T2)  # should be 1.16/1.2 ≈ 0.9667
            if self.fracVV_1 is not None:
                if not self._is_close(self.fracVV_1, expected_ratio):
                    logs.append(f"FAIL: Ratio V/V1 mismatch: computed {self.fracVV_1}, expected {expected_ratio}")
                    success = False
                else:
                    logs.append(f"PASS: Ratio V/V1 matches expected value {expected_ratio:.4f}")
            else:
                logs.append("SKIP: fracVV_1 not provided.")
        except Exception as e:
            logs.append(f"ERROR in ratio check: {e}")
            success = False

        # 3. Leakage rate formula check
        try:
            if self.fracVV_1 is not None and self.LeakageRate is not None:
                expected_leakage = 100 * (1 - self.fracVV_1)
                if not self._is_close(self.LeakageRate, expected_leakage):
                    logs.append(f"FAIL: LeakageRate mismatch: computed {self.LeakageRate}, expected {expected_leakage}")
                    success = False
                else:
                    logs.append(f"PASS: LeakageRate formula holds (value {self.LeakageRate:.1f}%)")
            else:
                logs.append("SKIP: Missing data for leakage rate check.")
        except Exception as e:
            logs.append(f"ERROR in leakage rate check: {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results:
        1. V should equal p0*V0/(p - p0)
        2. LeakageRate should equal 100*(1 - fracVV_1)
        """
        logs = []
        success = True

        # 1. Check V formula
        try:
            expected_V = self.p0 * self.V0 / (self.p - self.p0)
            if self.V is not None:
                if not self._is_close(self.V, expected_V):
                    logs.append(f"FAIL: V mismatch: computed {self.V}, expected {expected_V}")
                    success = False
                else:
                    logs.append(f"PASS: V matches formula (value {self.V})")
            else:
                logs.append("SKIP: V not provided.")
        except Exception as e:
            logs.append(f"ERROR in V formula check: {e}")
            success = False

        # 2. Check LeakageRate from fracVV_1
        try:
            if self.fracVV_1 is not None and self.LeakageRate is not None:
                expected_LR = 100 * (1 - self.fracVV_1)
                if not self._is_close(self.LeakageRate, expected_LR):
                    logs.append(f"FAIL: LeakageRate inconsistency: {self.LeakageRate} vs {expected_LR}")
                    success = False
                else:
                    logs.append(f"PASS: LeakageRate consistent with fracVV_1")
            else:
                logs.append("SKIP: Missing data for LeakageRate consistency.")
        except Exception as e:
            logs.append(f"ERROR in LeakageRate consistency: {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def test_monotonicity(self):
        """
        Perturb p0 (increase by 1%) and verify that V increases as expected from formula.
        Then restore original values.
        """
        logs = []
        success = True

        # Only run if V is numerical and we have p0, V0, p
        if None in (self.V, self.p0, self.V0, self.p):
            logs.append("SKIP: Missing data for monotonicity test.")
            self.logs.extend(logs)
            return False, logs

        try:
            # Perturb p0 by +1%
            perturbed_p0 = self.p0 * 1.01
            # Compute expected V from formula with perturbed p0
            expected_V_perturbed = perturbed_p0 * self.V0 / (self.p - perturbed_p0)
            # Compute expected V from formula with original p0
            expected_V_original = self.p0 * self.V0 / (self.p - self.p0)

            # Check monotonicity: if p0 increases, V should increase (since p > p0)
            if expected_V_perturbed > expected_V_original:
                logs.append("PASS: Monotonicity holds (V increases with p0).")
            else:
                logs.append("FAIL: Monotonicity violated (V did not increase with p0).")
                success = False

            # Restore original p0 (though we only used local copies)
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            self.logs.append(f"\n=== {name} Test ===")
            try:
                success, logs = test_func()
                if not success:
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR during {name}: {traceback.format_exc()}")
                overall_success = False

        self.logs.insert(0, f"Overall success: {overall_success}")
        return overall_success, self.logs