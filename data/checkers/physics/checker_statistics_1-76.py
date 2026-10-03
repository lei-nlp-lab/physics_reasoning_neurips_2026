import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the problem:
    The difference between cp and cv for ideal gas, and the physical reason.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.delta_c = results.get('delta_c', None)
        self.reason = results.get('reason', None)

        # Extract givens safely (e.g., molar mass M)
        givens = getattr(solution_instance, 'givens', {})
        self.M = givens.get('M', None)

        # Universal gas constant (J/(mol·K))
        self.R = 8.314

    # ── interface tests ──────────────────────────────────────────────
    def test_interface_delta_c_exists(self):
        if self.delta_c is None:
            return (False, "delta_c is missing (None)")
        return (True, "delta_c present")

    def test_interface_reason_exists(self):
        if self.reason is None:
            return (False, "reason is missing (None)")
        return (True, "reason present")

    def test_interface_delta_c_is_numeric(self):
        if self.delta_c is None:
            return (False, "SKIP: delta_c missing")
        if not isinstance(self.delta_c, (int, float)):
            return (False, f"delta_c is not numeric, got {type(self.delta_c)}")
        return (True, "delta_c is numeric")

    def test_interface_reason_is_string(self):
        if self.reason is None:
            return (False, "SKIP: reason missing")
        if not isinstance(self.reason, str):
            return (False, f"reason is not string, got {type(self.reason)}")
        return (True, "reason is string")

    # ── invariant / sanity tests ─────────────────────────────────────
    def test_invariant_positive_delta_c(self):
        if self.delta_c is None:
            return (False, "SKIP: delta_c missing")
        if not isinstance(self.delta_c, (int, float)):
            return (False, "delta_c not numeric, cannot check positivity")
        if self.delta_c > 0:
            return (True, "delta_c > 0")
        else:
            return (False, f"delta_c should be positive, got {self.delta_c}")

    def test_invariant_finite_delta_c(self):
        if self.delta_c is None:
            return (False, "SKIP: delta_c missing")
        if not isinstance(self.delta_c, (int, float)):
            return (False, "delta_c not numeric, cannot check finiteness")
        if math.isfinite(self.delta_c):
            return (True, "delta_c is finite")
        else:
            return (False, "delta_c is not finite (inf or NaN)")

    # ── modeling / trace tests (at least two) ────────────────────────
    def test_modeling_delta_c_equals_R_over_M(self):
        """
        If molar mass M is provided, check that delta_c = R / M (ideal gas).
        Exact value derived from physics inside the checker.
        """
        if self.M is None or self.delta_c is None:
            return (False, "SKIP: M or delta_c not available")
        if not isinstance(self.M, (int, float)) or self.M <= 0:
            return (False, f"Invalid M: {self.M}")
        expected = self.R / self.M
        # exact values allowed because derived here from physics
        if math.isclose(self.delta_c, expected, rel_tol=1e-9):
            return (True, f"delta_c ≈ {expected}")
        else:
            return (False, f"delta_c = {self.delta_c}, expected R/M = {expected}")

    def test_modeling_reason_references_ideal_gas_law(self):
        """
        Check that the reasoning mentions the ideal gas law or the
        relation Cp - Cv = R (key modeling step).
        """
        if self.reason is None:
            return (False, "SKIP: reason missing")
        keywords = ['ideal gas', 'cp - cv = r', 'cp–cv = r',
                     'c_p - c_v = r', 'r = cp - cv', 'molar specific',
                     'first law', 'internal energy', 'work']
        if any(kw in self.reason.lower() for kw in keywords):
            return (True, "reason references ideal gas law or first law")
        else:
            return (False, "reason does not contain relevant physical concepts")

    # ── monotonicity test ────────────────────────────────────────────
    def test_monotonicity_delta_c_decreases_with_increasing_M(self):
        """
        If molar mass increases, delta_c must decrease (inverse proportionality).
        Simulate perturbation: increase M by 10% and compute expected new delta_c
        using R/M. Compare to original delta_c (must be larger than new).
        """
        if self.M is None or self.delta_c is None:
            return (False, "SKIP: M or delta_c not available")
        if not isinstance(self.delta_c, (int, float)) or not isinstance(self.M, (int, float)):
            return (False, "delta_c or M not numeric")
        # Baseline expected delta_c (should match if model holds)
        base_expected = self.R / self.M
        if not math.isclose(self.delta_c, base_expected, rel_tol=1e-9):
            # If the baseline model does not hold, skip (cannot test monotonicity)
            return (False, "SKIP: baseline delta_c does not match R/M")
        new_M = self.M * 1.1
        new_expected = self.R / new_M
        if new_expected < self.delta_c:
            return (True, "delta_c decreases when M increases (inverse relationship)")
        else:
            return (False, "Monotonicity failed: expected new delta_c < old, but got not")

    # ── weak answer sanity (at most one) ─────────────────────────────
    def test_answer_sanity_reason_contains_work_terms(self):
        """
        Weak sanity: reason should mention work or expansion.
        """
        if self.reason is None:
            return (False, "SKIP: reason missing")
        keywords = ['work', 'expansion', 'expand', 'pdv', 'pressure-volume']
        if any(kw in self.reason.lower() for kw in keywords):
            return (True, "reason mentions work/expansion")
        else:
            return (False, "reason does not contain work-related keywords")

    # ── run_all_tests ────────────────────────────────────────────────
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)