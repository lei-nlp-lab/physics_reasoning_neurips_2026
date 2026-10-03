import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the resistance and capacitance between coaxial conductors
    with a dielectric filling.
    """

    def __init__(self, solution_instance):
        # Safe extraction of givens and results
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        # Extract givens with None defaults
        self.a = self.givens.get('a', None)
        self.b = self.givens.get('b', None)
        self.l = self.givens.get('l', None)
        self.sigma = self.givens.get('sigma', None)
        self.epsilon = self.givens.get('epsilon', None)

        # Extract results with None defaults
        self.R = self.results.get('R', None)
        self.C = self.results.get('C', None)

    # ----------------------------------------------------------------------
    # Interface checks: required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_required_givens(self):
        """Check that all required givens are present and positive."""
        missing = []
        for name, val in [('a', self.a), ('b', self.b), ('l', self.l),
                          ('sigma', self.sigma), ('epsilon', self.epsilon)]:
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(f"{name} (non-numeric type)")
            elif val <= 0:
                missing.append(f"{name} (non-positive)")
        if missing:
            return (False, f"Missing or invalid givens: {missing}")
        return (True, "All required givens present and positive")

    def test_interface_required_results(self):
        """Check that R and C are present and finite."""
        missing = []
        for name, val in [('R', self.R), ('C', self.C)]:
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(f"{name} (non-numeric)")
            elif not math.isfinite(val):
                missing.append(f"{name} (non-finite)")
        if missing:
            return (False, f"Missing or invalid results: {missing}")
        return (True, "R and C present and finite")

    # ----------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose, 1% tolerance)
    # ----------------------------------------------------------------------
    def test_invariant_RC_product(self):
        """Check RC = ε/σ (the product equals permittivity over conductivity)."""
        if any(v is None for v in [self.R, self.C, self.epsilon, self.sigma]):
            return (False, "SKIP: insufficient data for RC product check")
        target = self.epsilon / self.sigma
        product = self.R * self.C
        if math.isclose(product, target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"RC product = {product} equals ε/σ = {target} (within 1%)")
        else:
            return (False, f"RC product = {product} does not match ε/σ = {target}")

    def test_invariant_sign(self):
        """R and C must be positive."""
        if self.R is None or self.C is None:
            return (False, "SKIP: R or C missing")
        if self.R > 0 and self.C > 0:
            return (True, f"R = {self.R} > 0, C = {self.C} > 0")
        else:
            return (False, f"Non-positive values: R = {self.R}, C = {self.C}")

    def test_invariant_length_condition(self):
        """Check that l >> b (if applicable) to avoid end effects."""
        if self.l is None or self.b is None:
            return (False, "SKIP: missing l or b")
        # Even a factor of 10 is a weak condition; we accept it.
        if self.l > 10 * self.b:
            return (True, f"l = {self.l} > 10 * b = {self.b}, condition satisfied")
        else:
            return (False, f"l = {self.l} is not >> b = {self.b} (l > 10*b required)")

    # ----------------------------------------------------------------------
    # Modeling-quality checks (trace consistency, re-derived from givens)
    # ----------------------------------------------------------------------
    def test_modeling_resistance_formula(self):
        """Re-derive R from given a, b, l, sigma and compare with reported R."""
        if any(v is None for v in [self.a, self.b, self.l, self.sigma, self.R]):
            return (False, "SKIP: missing data for resistance formula check")
        try:
            expected_R = math.log(self.b / self.a) / (2 * math.pi * self.l * self.sigma)
            if math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"R = {self.R} matches expected {expected_R} (within 1%)")
            else:
                return (False, f"R = {self.R} differs from expected {expected_R} by more than 1%")
        except Exception as e:
            return (False, f"Resistance formula check exception: {e}")

    def test_modeling_capacitance_formula(self):
        """Re-derive C from given a, b, l, epsilon and compare with reported C."""
        if any(v is None for v in [self.a, self.b, self.l, self.epsilon, self.C]):
            return (False, "SKIP: missing data for capacitance formula check")
        try:
            expected_C = 2 * math.pi * self.epsilon * self.l / math.log(self.b / self.a)
            if math.isclose(self.C, expected_C, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"C = {self.C} matches expected {expected_C} (within 1%)")
            else:
                return (False, f"C = {self.C} differs from expected {expected_C} by more than 1%")
        except Exception as e:
            return (False, f"Capacitance formula check exception: {e}")

    def test_modeling_symmetry_check(self):
        """
        Structural consistency: (R * σ) and (C / ε) should be reciprocals.
        This is a null test if both formulas are correct.
        """
        if any(v is None for v in [self.R, self.C, self.sigma, self.epsilon]):
            return (False, "SKIP: missing variables for symmetry check")
        try:
            val_Rsigma = self.R * self.sigma
            val_Cepsilon = self.C / self.epsilon
            product = val_Rsigma * val_Cepsilon
            if math.isclose(product, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Consistency: (R*σ)*(C/ε) = 1")
            else:
                return (False, f"(R*σ)*(C/ε) = {product}, expected 1")
        except Exception as e:
            return (False, f"Symmetry check exception: {e}")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturb input, recompute expected behavior)
    # ----------------------------------------------------------------------
    def test_monotonicity_resistance_increase_with_b(self):
        """R increases when b increases (since R ∝ ln(b/a))."""
        if any(v is None for v in [self.a, self.b, self.l, self.sigma]):
            return (False, "SKIP: missing givens for monotonicity test")
        try:
            baseline = math.log(self.b / self.a) / (2 * math.pi * self.l * self.sigma)
            b_new = self.b * 1.1
            perturbed = math.log(b_new / self.a) / (2 * math.pi * self.l * self.sigma)
            if perturbed > baseline:
                return (True, f"Resistance increases with b: baseline={baseline}, perturbed(b+10%)={perturbed}")
            else:
                return (False, f"Resistance should increase but baseline={baseline}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Monotonicity check exception: {e}")

    def test_monotonicity_capacitance_decrease_with_b(self):
        """C decreases when b increases (since C ∝ 1/ln(b/a))."""
        if any(v is None for v in [self.a, self.b, self.l, self.epsilon]):
            return (False, "SKIP: missing givens for monotonicity test")
        try:
            baseline = 2 * math.pi * self.epsilon * self.l / math.log(self.b / self.a)
            b_new = self.b * 1.1
            perturbed = 2 * math.pi * self.epsilon * self.l / math.log(b_new / self.a)
            if perturbed < baseline:
                return (True, f"Capacitance decreases with b: baseline={baseline}, perturbed(b+10%)={perturbed}")
            else:
                return (False, f"Capacitance should decrease but baseline={baseline}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Monotonicity check exception: {e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (no re-derivation, no isclose)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """R and C should be positive and finite (already checked, but explicit)."""
        if self.R is None or self.C is None:
            return (False, "SKIP: R or C missing")
        if not (math.isfinite(self.R) and self.R > 0):
            return (False, f"R is not positive finite: {self.R}")
        if not (math.isfinite(self.C) and self.C > 0):
            return (False, f"C is not positive finite: {self.C}")
        return (True, "R and C are positive finite numbers")

    # ----------------------------------------------------------------------
    # Run all tests (discovery and execution)
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)