import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Do not directly access solution_instance.attr; use getattr with fallback.
        """
        # Inputs (with default values from the problem statement)
        self.p0 = float(getattr(solution_instance, 'p0', 75.0))  # cmHg
        self.l1 = float(getattr(solution_instance, 'l1', 18.0))  # cm
        self.l2 = float(getattr(solution_instance, 'l2', 10.0))  # cm
        self.g = float(getattr(solution_instance, 'g', 10.0))    # m/s^2

        # Results - safely extract from solution_instance or its results dictionary
        results = getattr(solution_instance, 'results', {})
        self.p1 = self._safe_extract(solution_instance, results, 'p_1')
        self.p2 = self._safe_extract(solution_instance, results, 'p_2')
        self.l = self._safe_extract(solution_instance, results, 'l')

        # Compute acceleration ratio a/g from results if available
        self.a_g = None
        if self.p1 is not None and self.p2 is not None:
            try:
                self.a_g = (self.p2 - self.p1) / self.l2
            except (TypeError, ZeroDivisionError):
                pass

    def _safe_extract(self, instance, results_dict, key):
        """
        Try to get the value from the instance attribute, then from results_dict.
        Return as float if possible, else None.
        """
        val = getattr(instance, key, None)
        if val is None:
            val = results_dict.get(key, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        return None

    def test_invariants(self):
        """
        Verify fundamental physical laws:
        1. Initial force equilibrium: p1 = p0 + l2.
        2. Boyle's Law for the transformation: p1 * l1 = p2 * l.
        """
        logs = []
        success = True

        # Check 1: Initial equilibrium (force balance on mercury column)
        if self.p1 is not None:
            expected_p1 = self.p0 + self.l2
            if not math.isclose(self.p1, expected_p1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Initial pressure p1 = {self.p1} cmHg, "
                            f"but expected p0 + l2 = {expected_p1} cmHg.")
                success = False
            else:
                logs.append(f"PASS: Initial equilibrium p1 = p0 + l2 = {expected_p1} cmHg.")
        else:
            logs.append("SKIP: p1 not available for initial equilibrium check.")
            success = False

        # Check 2: Boyle's Law (constant temperature process)
        if all(v is not None for v in [self.p1, self.l1, self.p2, self.l]):
            left = self.p1 * self.l1
            right = self.p2 * self.l
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Boyle's Law violated: p1*l1 = {left}, p2*l = {right}.")
                success = False
            else:
                logs.append(f"PASS: Boyle's Law holds: p1*l1 = p2*l = {left}.")
        else:
            missing = [k for k, v in {'p1': self.p1, 'l1': self.l1, 'p2': self.p2, 'l': self.l} if v is None]
            logs.append(f"SKIP: Boyle's Law check missing {missing}.")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        Specifically, verify that using p1 and p2 from steps, Boyle's Law gives the reported l.
        """
        logs = []
        success = True

        if all(v is not None for v in [self.p1, self.l1, self.p2, self.l]):
            # Recompute l from Boyle's Law
            l_computed = (self.p1 * self.l1) / self.p2
            if not math.isclose(l_computed, self.l, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Inconsistent trace: computed l = {l_computed} cm, "
                            f"but reported l = {self.l} cm.")
                success = False
            else:
                logs.append(f"PASS: Trace consistency: Boyle's Law yields l = {l_computed} cm ≈ reported {self.l} cm.")
        else:
            logs.append("SKIP: Trace consistency check missing data.")
            success = False

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (l2) and verify the expected trend in the output (l).
        Steps:
          1. Record original l2, p1, p2, l.
          2. Compute a/g from original results: a_g = (p2 - p1) / l2.
          3. Increase l2 by 1 cm.
          4. Compute new p1' = p0 + l2'.
          5. Compute new p2' = p0 + l2' * (1 + a_g).
          6. Compute new l' = (p1' * l1) / p2'.
          7. Check that l' < l (since increasing l2 should decrease l).
          8. Restore (no actual modification to instance attributes).
        """
        logs = []
        success = True

        # Ensure we have all needed data
        if any(v is None for v in [self.p1, self.p2, self.l, self.a_g]):
            logs.append("SKIP: Monotonicity check missing data (p1, p2, l, or a/g).")
            return False, logs

        original_l2 = self.l2
        perturbed_l2 = original_l2 + 1.0

        # Step 4 & 5: Compute new pressures
        p1_perturbed = self.p0 + perturbed_l2
        p2_perturbed = self.p0 + perturbed_l2 * (1 + self.a_g)

        # Step 6: Compute new length from Boyle's Law
        l_perturbed = (p1_perturbed * self.l1) / p2_perturbed

        # Step 7: Check monotonicity (increasing l2 should decrease l)
        if l_perturbed >= self.l and not math.isclose(l_perturbed, self.l, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Monotonicity violated. "
                        f"Original l = {self.l} cm, after increasing l2 by 1 cm, l' = {l_perturbed} cm (expected decrease).")
            success = False
        else:
            logs.append(f"PASS: Monotonicity holds. Increasing l2 decreases l: "
                        f"original l = {self.l} cm, perturbed l' = {l_perturbed} cm.")

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and log messages.
        """
        all_logs = ["=== PhysicsChecker Report ==="]
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"\n--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"\n--- {test_name} ---")
                all_logs.append(f"ERROR: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs