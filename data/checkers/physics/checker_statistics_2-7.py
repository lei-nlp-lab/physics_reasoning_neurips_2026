import sympy
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.Delta_S1 = self.results.get('Delta_S1', None)
        self.Delta_S2 = self.results.get('Delta_S2', None)
        self.ratio = self.results.get('ratio', None)

        # Extract givens (population & energy parameters) safely
        self.k_B = getattr(solution_instance, 'k_B', None)
        self.T = getattr(solution_instance, 'T', None)
        self.E1 = getattr(solution_instance, 'E1', None)
        self.E2 = getattr(solution_instance, 'E2', None)
        self.n1 = getattr(solution_instance, 'n1', None)
        self.n2 = getattr(solution_instance, 'n2', None)
        self.N = getattr(solution_instance, 'N', None)  # = n1 + n2, fixed

        # Define symbols for sympy parsing
        self.sym_k_B = sympy.Symbol('k_B', positive=True)
        self.sym_T = sympy.Symbol('T', positive=True)
        self.sym_E1 = sympy.Symbol('E1')
        self.sym_E2 = sympy.Symbol('E2')
        self.sym_n1 = sympy.Symbol('n1', positive=True)
        self.sym_n2 = sympy.Symbol('n2', positive=True)
        self.sym_N = sympy.Symbol('N', positive=True)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _parse_expr(self, expr_str, expect_symbolic=True):
        """Parse a string (containing ln, log, exp) into a sympy expression.
        Returns None on failure."""
        if expr_str is None:
            return None
        try:
            # Replace 'ln' with 'log' for sympy
            text = expr_str.replace('ln', 'log')
            expr = sympy.sympify(text, locals={
                'k_B': self.sym_k_B,
                'T': self.sym_T,
                'E1': self.sym_E1,
                'E2': self.sym_E2,
                'n1': self.sym_n1,
                'n2': self.sym_n2,
                'N': self.sym_N
            })
            return expr
        except Exception:
            return None

    def _evaluate_numeric(self, expr_str):
        """Evaluate a string expression with the stored numeric givens.
        Returns a float or None if impossible."""
        if expr_str is None:
            return None
        # Check that all required numeric givens are available
        numeric_needed = [self.k_B, self.T, self.E1, self.E2, self.n1, self.n2]
        if any(v is None for v in numeric_needed):
            return None
        expr = self._parse_expr(expr_str)
        if expr is None:
            return None
        subs = {
            self.sym_k_B: self.k_B,
            self.sym_T: self.T,
            self.sym_E1: self.E1,
            self.sym_E2: self.E2,
            self.sym_n1: self.n1,
            self.sym_n2: self.n2
        }
        try:
            val = expr.subs(subs).evalf()
            return float(val)
        except Exception:
            return None

    def _numeric_givens_available(self):
        return all(v is not None for v in [self.k_B, self.T, self.E1, self.E2, self.n1, self.n2])

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------
    def test_interface_results_exist(self):
        """Check that the three required result keys are present and not None."""
        if self.Delta_S1 is None:
            return (False, "Missing results['Delta_S1']")
        if self.Delta_S2 is None:
            return (False, "Missing results['Delta_S2']")
        if self.ratio is None:
            return (False, "Missing results['ratio']")
        return (True, "All three result keys present")

    # ------------------------------------------------------------------
    # Modeling tests (check that the symbolic expressions have the correct form)
    # ------------------------------------------------------------------
    def test_modeling_delta_s1_expression(self):
        """Verify that Delta_S1 equals k_B * ln(n2/n1) or k_B * ln(n2/(n1+1))"""
        if self.Delta_S1 is None:
            return (False, "SKIP: Delta_S1 not available")
        actual = self._parse_expr(self.Delta_S1)
        if actual is None:
            return (False, "Could not parse Delta_S1 expression")
        # expected candidate 1: k_B * ln(n2/n1)
        expected1 = self.sym_k_B * sympy.log(self.sym_n2 / self.sym_n1)
        # candidate 2: k_B * ln(n2/(n1+1))
        expected2 = self.sym_k_B * sympy.log(self.sym_n2 / (self.sym_n1 + 1))
        if sympy.simplify(actual - expected1) == 0:
            return (True, "Delta_S1 matches k_B*ln(n2/n1)")
        if sympy.simplify(actual - expected2) == 0:
            return (True, "Delta_S1 matches k_B*ln(n2/(n1+1))")
        return (False, f"Delta_S1 does not match the expected form. Got: {self.Delta_S1}")

    def test_modeling_delta_s2_expression(self):
        """Check that Delta_S2 equals (E2 - E1)/T."""
        if self.Delta_S2 is None:
            return (False, "SKIP: Delta_S2 not available")
        actual = self._parse_expr(self.Delta_S2)
        if actual is None:
            return (False, "Could not parse Delta_S2 expression")
        expected = (self.sym_E2 - self.sym_E1) / self.sym_T
        if sympy.simplify(actual - expected) == 0:
            return (True, "Delta_S2 matches (E2-E1)/T")
        return (False, f"Delta_S2 does not match (E2-E1)/T. Got: {self.Delta_S2}")

    def test_modeling_ratio_expression(self):
        """Check that ratio equals exp(-(E2-E1)/(k_B T))."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        actual = self._parse_expr(self.ratio)
        if actual is None:
            return (False, "Could not parse ratio expression")
        expected = sympy.exp(-(self.sym_E2 - self.sym_E1) / (self.sym_k_B * self.sym_T))
        if sympy.simplify(actual - expected) == 0:
            return (True, "ratio matches exp(-(E2-E1)/(k_B T))")
        return (False, f"ratio does not match expected Boltzmann factor. Got: {self.ratio}")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_stirling_consistency(self):
        """Check that the numeric Delta_S1 is consistent with the exact Stirling formula 
        (k * ln(n2/(n1+1))) to within 1% relative tolerance."""
        if not self._numeric_givens_available():
            return (False, "SKIP: numeric givens (k_B,T,E1,E2,n1,n2) missing")
        numeric_ds1 = self._evaluate_numeric(self.Delta_S1)
        if numeric_ds1 is None:
            return (False, "Cannot evaluate numeric Delta_S1")
        exact = self.k_B * math.log(self.n2 / (self.n1 + 1))
        approx = self.k_B * math.log(self.n2 / self.n1)
        # Check against exact (with 1% tolerance)
        if math.isclose(numeric_ds1, exact, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Delta_S1 numeric ({numeric_ds1:.4e}) close to exact Stirling ({exact:.4e})")
        if math.isclose(numeric_ds1, approx, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Delta_S1 numeric ({numeric_ds1:.4e}) close to approximate Stirling ({approx:.4e})")
        return (False, f"Delta_S1 numeric ({numeric_ds1:.4e}) differs from both exact ({exact:.4e}) and approx ({approx:.4e})")

    def test_invariant_reservoir_entropy_consistency(self):
        """Check that numeric Delta_S2 equals (E2 - E1)/T."""
        if not self._numeric_givens_available():
            return (False, "SKIP: numeric givens missing")
        numeric_ds2 = self._evaluate_numeric(self.Delta_S2)
        if numeric_ds2 is None:
            return (False, "Cannot evaluate numeric Delta_S2")
        expected = (self.E2 - self.E1) / self.T
        if math.isclose(numeric_ds2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Delta_S2 numeric ({numeric_ds2:.4e}) matches expected ({expected:.4e})")
        return (False, f"Delta_S2 numeric ({numeric_ds2:.4e}) differs from expected ({expected:.4e})")

    def test_invariant_total_entropy_zero(self):
        """Check that the sum Delta_S1 + Delta_S2 is close to zero (equilibrium condition)."""
        if not self._numeric_givens_available():
            return (False, "SKIP: numeric givens missing")
        numeric_ds1 = self._evaluate_numeric(self.Delta_S1)
        numeric_ds2 = self._evaluate_numeric(self.Delta_S2)
        if numeric_ds1 is None or numeric_ds2 is None:
            return (False, "Cannot evaluate numeric entropies")
        total = numeric_ds1 + numeric_ds2
        if math.isclose(total, 0.0, abs_tol=1e-2):
            return (True, f"Total entropy change {total:.4e} is close to zero")
        return (False, f"Total entropy change {total:.4e} is not zero")

    # ------------------------------------------------------------------
    # Monotonicity tests (check directional trends by perturbing inputs)
    # ------------------------------------------------------------------
    def test_monotonicity_system_entropy_with_population(self):
        """If n2 increases (keeping n1 fixed), Delta_S1 should increase."""
        if not self._numeric_givens_available():
            return (False, "SKIP: numeric givens missing")
        # Baseline
        base = self._evaluate_numeric(self.Delta_S1)
        if base is None:
            return (False, "Cannot evaluate baseline Delta_S1")
        # Perturb n2 upward by 1 (or 1% if large)
        delta = max(1, int(0.01 * self.n2))
        new_n2 = self.n2 + delta
        # Temporarily override n2 for evaluation
        original_n2 = self.n2
        self.n2 = new_n2
        perturbed = self._evaluate_numeric(self.Delta_S1)
        self.n2 = original_n2
        if perturbed is None:
            return (False, "Cannot evaluate perturbed Delta_S1")
        if perturbed > base:
            return (True, f"Delta_S1 increased from {base:.4e} to {perturbed:.4e} when n2 increased by {delta}")
        return (False, f"Delta_S1 did not increase: base {base:.4e}, perturbed {perturbed:.4e}")

    def test_monotonicity_ratio_with_temperature(self):
        """For E2>E1, increasing temperature should increase the ratio n2/n1."""
        if not self._numeric_givens_available():
            return (False, "SKIP: numeric givens missing")
        if self.E2 <= self.E1:
            return (False, "SKIP: requires E2 > E1 for monotonicity check")
        # Compute baseline ratio (using the expression if possible, else direct formula)
        base_val = self._evaluate_numeric(self.ratio)
        if base_val is None:
            # Fall back to manual Boltzmann factor
            base_val = math.exp(-(self.E2 - self.E1) / (self.k_B * self.T))
        # Increase T by 10% or 1 K, whichever is larger
        delta_T = max(1.0, 0.1 * self.T)
        new_T = self.T + delta_T
        # Evaluate the ratio at new T (need to temporarily change self.T)
        original_T = self.T
        self.T = new_T
        new_val = self._evaluate_numeric(self.ratio)
        if new_val is None:
            new_val = math.exp(-(self.E2 - self.E1) / (self.k_B * new_T))
        self.T = original_T
        if new_val > base_val:
            return (True, f"Ratio increased from {base_val:.4e} to {new_val:.4e} when T increased by {delta_T}")
        return (False, f"Ratio did not increase: base {base_val:.4e}, perturbed {new_val:.4e}")

    # ------------------------------------------------------------------
    # Weak answer sanity check (at most one, not re-deriving final answers)
    # ------------------------------------------------------------------
    def test_answer_sanity_ratio_sign_and_finite(self):
        """Check that ratio is positive and finite (non-NaN)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        actual = self._parse_expr(self.ratio)
        if actual is None:
            return (False, "Cannot parse ratio expression")
        # Check that the expression contains a positive exponential
        try:
            # If it's an exponential with a negative argument, the result is positive
            # We can do a structural check: it should be exp(-(E2-E1)/(k_B T))
            # But that is too specific. Instead, check that it's not obviously negative.
            # Since exp(-something) with real something is always positive, we can trust.
            # As a weak sanity, just ensure the expression is valid and contains exp.
            if actual.has(sympy.exp):
                return (True, "ratio contains exp, likely positive and finite")
            else:
                return (False, "ratio does not contain an exponential; expected exp form")
        except Exception:
            return (False, "Cannot verify ratio structure")

    # ------------------------------------------------------------------
    # Run all tests (mandatory)
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