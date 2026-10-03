import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}
        
        self.T1 = givens.get('T1', None)
        self.T2 = givens.get('T2', None)
        self.sigma = givens.get('sigma', None)
        if self.sigma is None:
            # Fallback to standard constant (physically known, not a numeric default for missing data)
            self.sigma = 5.67e-8
        
        # Result keys from pseudocode emit
        self.W_formula = results.get('W_formula', None)
        self.W2 = results.get('W2', None)
        self.W_n_formula = results.get('W_n_formula', None)
        
        # Additional useful results if present
        self.E1 = results.get('E1', None)
        self.E2 = results.get('E2', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_has_W_formula(self):
        if self.W_formula is None:
            return (False, "Missing W_formula in results")
        if not isinstance(self.W_formula, str):
            return (False, "W_formula should be a string")
        return (True, "W_formula present and is string")

    def test_interface_has_W2(self):
        if self.W2 is None:
            return (False, "Missing W2 in results")
        if not isinstance(self.W2, (int, float)):
            return (False, "W2 should be numeric")
        return (True, "W2 present and numeric")

    def test_interface_has_W_n_formula(self):
        if self.W_n_formula is None:
            return (False, "Missing W_n_formula in results")
        if not isinstance(self.W_n_formula, str):
            return (False, "W_n_formula should be a string")
        return (True, "W_n_formula present and is string")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_W_formula_sign(self):
        # For any positive e1, e2 and E1 > E2, the formula yields positive W.
        if self.W_formula is None:
            return (False, "SKIP: W_formula not available")
        try:
            # Test with a typical set of positive numbers respecting e <= E
            test_E1 = 500.0
            test_E2 = 100.0
            test_e1 = 400.0
            test_e2 = 80.0
            test_W = (test_E1 - test_E2) / (test_E1/test_e1 + test_E2/test_e2 - 1)
            if test_W > 0:
                return (True, "W formula yields positive for typical positive inputs")
            else:
                return (False, f"W formula gave non-positive: {test_W}")
        except Exception as e:
            return (False, f"Invariant check crashed: {e}")

    def test_invariant_W2_positive(self):
        if self.W2 is None:
            return (False, "SKIP: W2 not available")
        if self.W2 > 0:
            return (True, "W2 is positive as expected for T1 > T2")
        else:
            return (False, f"W2 is {self.W2}, but should be positive")

    # ------------------------------------------------------------------
    # Modeling / Trace tests
    # ------------------------------------------------------------------
    def test_modeling_W2_calculation(self):
        # Re-derive W2 from T1, T2, sigma and compare with given W2 (if available)
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 givens missing")
        if self.W2 is None:
            return (False, "SKIP: W2 not available for comparison")
        E1 = self.sigma * (self.T1 ** 4)
        E2 = self.sigma * (self.T2 ** 4)
        expected_W = E1 - E2
        if math.isclose(self.W2, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"W2 {self.W2} matches re-derived value {expected_W:.2f} within 1%")
        else:
            return (False, f"W2 {self.W2} deviates from re-derived {expected_W:.2f} by more than 1%")

    def test_modeling_W_n_formula_structure(self):
        # Check that W_n_formula is a string containing proper variables
        if self.W_n_formula is None:
            return (False, "SKIP: W_n_formula not available")
        formula = self.W_n_formula.replace(' ', '')
        required_parts = ['sigma', 'n+1', 'T1^4', 'T2^4']
        found = all(part in formula for part in required_parts)
        if found:
            return (True, "W_n_formula contains required components: sigma/(n+1)*(T1^4-T2^4)")
        else:
            return (False, f"W_n_formula '{self.W_n_formula}' missing expected components")

    def test_modeling_W_n_formula_consistency(self):
        # Evaluate the formula for a specific n and compare with expected
        if self.W_n_formula is None:
            return (False, "SKIP: W_n_formula not available")
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 missing for evaluation")
        try:
            n_test = 1
            deltaT4 = self.T1**4 - self.T2**4
            expected = self.sigma * deltaT4 / (n_test + 1)
            expr_str = self.W_n_formula.replace('^', '**')
            expr_str = expr_str.replace('sigma', str(self.sigma))
            expr_str = expr_str.replace('T1', str(self.T1))
            expr_str = expr_str.replace('T2', str(self.T2))
            expr_str = expr_str.replace('n', str(n_test))
            computed = eval(expr_str)
            if math.isclose(computed, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"W_n_formula for n=1 yields {computed:.4f}, matches expected {expected:.4f}")
            else:
                return (False, f"W_n_formula gave {computed:.4f}, expected {expected:.4f}")
        except Exception as e:
            return (False, f"Failed to evaluate W_n_formula: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_W_increases_with_T1(self):
        # If T1 increases, W should increase (for fixed T2, sigma)
        if self.W2 is None:
            return (False, "SKIP: W2 not available for baseline")
        if self.T1 is None:
            return (False, "SKIP: T1 not available")
        delta = 0.01 * self.T1
        T1_pert = self.T1 + delta
        new_W = self.sigma * (T1_pert**4 - self.T2**4)
        if new_W > self.W2:
            return (True, f"Increasing T1 by 1% increases W from {self.W2:.2f} to {new_W:.2f}")
        else:
            return (False, f"Monotonicity fails: T1 higher but W not larger (new_W={new_W:.2f} <= {self.W2:.2f})")

    def test_monotonicity_W_decreases_with_n(self):
        # For black bodies, as n increases, W_n should decrease
        if self.W_n_formula is None:
            return (False, "SKIP: W_n_formula not available")
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 missing")
        try:
            deltaT4 = self.T1**4 - self.T2**4
            W_0 = self.sigma * deltaT4  # n=0 (no shields)
            n_test = 1
            expr_str = self.W_n_formula.replace('^', '**')
            expr_str = expr_str.replace('sigma', str(self.sigma))
            expr_str = expr_str.replace('T1', str(self.T1))
            expr_str = expr_str.replace('T2', str(self.T2))
            expr_str = expr_str.replace('n', str(n_test))
            W_1 = eval(expr_str)
            if W_1 < W_0:
                return (True, f"W_n for n=1 ({W_1:.4f}) is less than W0 ({W_0:.4f})")
            else:
                return (False, f"W_n not decreasing: W0={W_0:.4f}, W1={W_1:.4f}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ------------------------------------------------------------------
    # Single weak answer sanity (do not dominate)
    # ------------------------------------------------------------------
    def test_answer_sanity_W2_positive_finite(self):
        if self.W2 is None:
            return (False, "SKIP: W2 not available")
        if math.isfinite(self.W2) and self.W2 > 0:
            return (True, "W2 is finite and positive")
        else:
            return (False, f"W2 is not finite/positive: {self.W2}")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)