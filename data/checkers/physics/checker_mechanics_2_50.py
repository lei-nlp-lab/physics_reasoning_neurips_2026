import math
import traceback

class PhysicsChecker:
    """
    Auditor for normal modes of a suspended bar.
    Extracts results and givens from solution_instance; runs physics sanity checks.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.omega1 = results.get('omega1', None)
        self.omega2 = results.get('omega2', None)
        self.ratio1 = results.get('ratio1', None)
        self.ratio2 = results.get('ratio2', None)

        # Try to extract g and l from solution_instance (attributes or givens dict)
        self.g = getattr(solution_instance, 'g', None)
        self.l = getattr(solution_instance, 'l', None)
        givens = getattr(solution_instance, 'givens', None)
        if givens and isinstance(givens, dict):
            if self.g is None:
                self.g = givens.get('g', None)
            if self.l is None:
                self.l = givens.get('l', None)

    def run_all_tests(self):
        """Discover all test_ methods and execute them in sorted order."""
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

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that the four required keys exist."""
        if self.omega1 is None:
            return (False, "Missing 'omega1' in results")
        if self.omega2 is None:
            return (False, "Missing 'omega2' in results")
        if self.ratio1 is None:
            return (False, "Missing 'ratio1' in results")
        if self.ratio2 is None:
            return (False, "Missing 'ratio2' in results")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Ensure values are numeric (int or float) and ratios are real."""
        for name, val in [('omega1', self.omega1), ('omega2', self.omega2),
                          ('ratio1', self.ratio1), ('ratio2', self.ratio2)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not a number (type {type(val).__name__})")
        # Check ratios are real (not complex)
        if isinstance(self.ratio1, complex) or isinstance(self.ratio2, complex):
            return (False, "Ratios must be real numbers")
        return (True, "All values are numeric and real")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_quadratic_relation(self):
        """Verify that omegas satisfy l^2 ω^4 - 8 g l ω^2 + 4 g^2 = 0.
           Uses g and l; skips if either missing."""
        if self.omega1 is None or self.omega2 is None:
            return (False, "SKIP: omega values missing")
        if self.g is None or self.l is None:
            return (False, "SKIP: g or l not available")
        try:
            x1 = (self.omega1 ** 2) * self.l / self.g
            x2 = (self.omega2 ** 2) * self.l / self.g
        except ZeroDivisionError:
            return (False, "g or l is zero")
        # Characteristic equation in x: x^2 - 8 x + 4 = 0
        lhs1 = x1**2 - 8*x1 + 4
        lhs2 = x2**2 - 8*x2 + 4
        if not (math.isclose(lhs1, 0.0, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(lhs2, 0.0, rel_tol=1e-2, abs_tol=1e-2)):
            return (False, f"Quadratic relation not satisfied: lhs1={lhs1}, lhs2={lhs2}")
        # Also check sum and product of roots
        if not math.isclose(x1 + x2, 8.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Sum of x1+x2 = {x1+x2:.4f}, expected ~8")
        if not math.isclose(x1 * x2, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Product x1*x2 = {x1*x2:.4f}, expected ~4")
        return (True, f"Omegas satisfy quadratic: x1={x1:.4f}, x2={x2:.4f}")

    def test_invariant_determinant(self):
        """For each omega, check (g - ω^2 l)^2 - (3/4) l^2 ω^4 ≈ 0."""
        if self.omega1 is None or self.omega2 is None:
            return (False, "SKIP: omega values missing")
        if self.g is None or self.l is None:
            return (False, "SKIP: g or l not available")
        try:
            det1 = (self.g - self.omega1**2 * self.l)**2 - (3.0/4.0) * self.l**2 * self.omega1**4
            det2 = (self.g - self.omega2**2 * self.l)**2 - (3.0/4.0) * self.l**2 * self.omega2**4
        except Exception:
            return (False, "Error computing determinant")
        if not math.isclose(det1, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Determinant for ω1 = {det1} (≈0 expected)")
        if not math.isclose(det2, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Determinant for ω2 = {det2} (≈0 expected)")
        return (True, "Determinant equations satisfied for both omegas")

    def test_invariant_ratio_consistency(self):
        """Check that each ratio satisfies B/A = (g - ω^2 l) / (l ω^2)."""
        if self.omega1 is None or self.omega2 is None or self.ratio1 is None or self.ratio2 is None:
            return (False, "SKIP: required values missing")
        if self.g is None or self.l is None:
            return (False, "SKIP: g or l not available")
        try:
            exp_ratio1 = (self.g - self.omega1**2 * self.l) / (self.l * self.omega1**2)
            exp_ratio2 = (self.g - self.omega2**2 * self.l) / (self.l * self.omega2**2)
        except ZeroDivisionError:
            return (False, "omega or l is zero")
        if not math.isclose(self.ratio1, exp_ratio1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ratio1 mismatch: given {self.ratio1}, expected {exp_ratio1}")
        if not math.isclose(self.ratio2, exp_ratio2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ratio2 mismatch: given {self.ratio2}, expected {exp_ratio2}")
        return (True, "Amplitude ratios consistent with eigenvalue equation")

    # ------------------------------------------------------------------
    # Modeling / Trace checks
    # ------------------------------------------------------------------
    def test_trace_normal_mode_equation(self):
        """
        Verify that for each mode, the first linearized equation is satisfied:
        -(3/4) l ω^2 * 1 + (g - l ω^2) * (B/A) ≈ 0.
        This traces the substitution of the mode shape into the EoM.
        """
        if self.omega1 is None or self.omega2 is None or self.ratio1 is None or self.ratio2 is None:
            return (False, "SKIP: required results missing")
        if self.g is None or self.l is None:
            return (False, "SKIP: g or l not available")
        # Use first row: - (3/4) l ω^2 * A + (g - l ω^2) * B = 0
        # With A=1 (normalized), B = ratio, we check expression.
        try:
            eq1 = - (3.0/4.0) * self.l * self.omega1**2 * 1.0 + (self.g - self.l * self.omega1**2) * self.ratio1
            eq2 = - (3.0/4.0) * self.l * self.omega2**2 * 1.0 + (self.g - self.l * self.omega2**2) * self.ratio2
        except Exception:
            return (False, "Error computing normal-mode equations")
        if not math.isclose(eq1, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Equation for ω1 not satisfied: {eq1}")
        if not math.isclose(eq2, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Equation for ω2 not satisfied: {eq2}")
        return (True, "Normal-mode equations satisfied for both modes")

    def test_trace_ratio_product(self):
        """
        The product of the two ratios should be -3/4? Check: 
        ratio1 * ratio2 = (-√3/2)*(√3/2) = -3/4.
        This is a derived consistency condition from the eigenvalue problem.
        """
        if self.ratio1 is None or self.ratio2 is None:
            return (False, "SKIP: ratio values missing")
        product = self.ratio1 * self.ratio2
        if not math.isclose(product, -0.75, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Product of ratios = {product}, expected -3/4")
        return (True, f"ratio1*ratio2 = {product} ≈ -3/4")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of input)
    # ------------------------------------------------------------------
    def test_monotonicity_omega_decreases_with_length(self):
        """
        Perturb l upward by 10% and check that the expected omega becomes smaller.
        Expected omega = sqrt(g/l)*(√3 ± 1). Since we have the actual omega value,
        we use the formula to compute perturbed and compare directions.
        """
        if self.omega1 is None or self.omega2 is None:
            return (False, "SKIP: omega values missing")
        if self.g is None or self.l is None:
            return (False, "SKIP: g or l not available")
        # Original lengths
        l_orig = self.l
        delta_factor = 1.1   # 10% increase
        l_new = l_orig * delta_factor
        # Compute expected frequencies for both modes using the theoretical formula
        # Since we only have the actual omegas, we compute expected ones from given g,l
        try:
            expected_omega1 = math.sqrt(self.g / l_orig) * (math.sqrt(3) + 1)
            expected_omega2 = math.sqrt(self.g / l_orig) * (math.sqrt(3) - 1)
            perturbed_omega1 = math.sqrt(self.g / l_new) * (math.sqrt(3) + 1)
            perturbed_omega2 = math.sqrt(self.g / l_new) * (math.sqrt(3) - 1)
        except Exception:
            return (False, "Error computing perturbed omegas")
        if not expected_omega1 > perturbed_omega1:
            return (False, f"ω1 should decrease when l increases: {expected_omega1} > {perturbed_omega1}")
        if not expected_omega2 > perturbed_omega2:
            return (False, f"ω2 should decrease when l increases: {expected_omega2} > {perturbed_omega2}")
        # Also verify that the given omega values are close to expected
        if not math.isclose(self.omega1, expected_omega1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Given ω1={self.omega1} deviates from expected {expected_omega1}")
        if not math.isclose(self.omega2, expected_omega2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Given ω2={self.omega2} deviates from expected {expected_omega2}")
        return (True, "Monotonicity: omegas decrease with increasing length")

    def test_monotonicity_omega_increases_with_g(self):
        """
        Perturb g upward by 10% and check that expected omegas increase.
        """
        if self.omega1 is None or self.omega2 is None:
            return (False, "SKIP: omega values missing")
        if self.g is None or self.l is None:
            return (False, "SKIP: g or l not available")
        g_orig = self.g
        delta_factor = 1.1
        g_new = g_orig * delta_factor
        try:
            expected_omega1 = math.sqrt(g_orig / self.l) * (math.sqrt(3) + 1)
            expected_omega2 = math.sqrt(g_orig / self.l) * (math.sqrt(3) - 1)
            perturbed_omega1 = math.sqrt(g_new / self.l) * (math.sqrt(3) + 1)
            perturbed_omega2 = math.sqrt(g_new / self.l) * (math.sqrt(3) - 1)
        except Exception:
            return (False, "Error computing perturbed omegas")
        if not expected_omega1 < perturbed_omega1:
            return (False, f"ω1 should increase when g increases: {expected_omega1} < {perturbed_omega1}")
        if not expected_omega2 < perturbed_omega2:
            return (False, f"ω2 should increase when g increases: {expected_omega2} < {perturbed_omega2}")
        if not math.isclose(self.omega1, expected_omega1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Given ω1={self.omega1} deviates from expected {expected_omega1}")
        if not math.isclose(self.omega2, expected_omega2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Given ω2={self.omega2} deviates from expected {expected_omega2}")
        return (True, "Monotonicity: omegas increase with increasing g")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive(self):
        """Omegas should be positive finite numbers; ratios real and finite."""
        for name, val in [('omega1', self.omega1), ('omega2', self.omega2)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is non-numeric")
            if val <= 0 or math.isinf(val) or math.isnan(val):
                return (False, f"{name} invalid: {val}")
        for name, val in [('ratio1', self.ratio1), ('ratio2', self.ratio2)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is non-numeric")
            if math.isinf(val) or math.isnan(val):
                return (False, f"{name} invalid: {val}")
        return (True, "All output values are positive finite real numbers")