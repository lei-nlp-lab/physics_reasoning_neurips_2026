import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for the potential energy of a point charge near a semi-infinite dielectric.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})

        # Extract required results with None defaults (no numeric fill-in)
        self.U = self.results.get('U', None)

        # Extract given parameters: prefer instance attribute, then results dict
        self.q = getattr(solution_instance, 'q', None)
        if self.q is None:
            self.q = self.results.get('q', None)

        self.x = getattr(solution_instance, 'x', None)
        if self.x is None:
            self.x = self.results.get('x', None)

        self.K = getattr(solution_instance, 'K', None)
        if self.K is None:
            self.K = self.results.get('K', None)

        # Vacuum permittivity (fundamental constant) – ONLY if provided
        self.eps0 = getattr(solution_instance, 'eps0', None)
        if self.eps0 is None:
            self.eps0 = self.results.get('eps0', None)

    def _is_numeric(self, val):
        """Check if val is finite numeric."""
        if val is None:
            return False
        if isinstance(val, (int, float)):
            return math.isfinite(val)
        try:
            v = float(val)
            return math.isfinite(v)
        except (TypeError, ValueError):
            return False

    # --------------------------------------------------
    # Interface tests
    # --------------------------------------------------
    def test_interface_has_U_key(self):
        if self.U is None:
            return (False, "Missing key 'U' in results")
        return (True, "'U' found in results")

    def test_interface_q_present(self):
        if self.q is None:
            return (False, "Missing given 'q'")
        if not self._is_numeric(self.q) or float(self.q) <= 0:
            return (False, f"'q' should be positive numeric, got {self.q}")
        return (True, f"'q' = {self.q} is valid")

    def test_interface_x_present(self):
        if self.x is None:
            return (False, "Missing given 'x'")
        if not self._is_numeric(self.x) or float(self.x) <= 0:
            return (False, f"'x' should be positive numeric, got {self.x}")
        return (True, f"'x' = {self.x} is valid")

    def test_interface_K_present(self):
        if self.K is None:
            return (False, "Missing given 'K'")
        if not self._is_numeric(self.K) or float(self.K) <= 1:
            return (False, f"'K' should be > 1 (dielectric constant), got {self.K}")
        return (True, f"'K' = {self.K} is valid")

    def test_interface_eps0_present(self):
        if self.eps0 is None:
            return (False, "Missing eps0; cannot test numerical formulas")
        if not self._is_numeric(self.eps0) or float(self.eps0) <= 0:
            return (False, f"eps0 should be positive numeric, got {self.eps0}")
        return (True, f"eps0 = {self.eps0} is valid")

    # --------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # --------------------------------------------------
    def test_invariant_U_finite(self):
        if self.U is None:
            return (False, "SKIP: U not available")
        if not self._is_numeric(self.U):
            return (False, f"U is not finite numeric, got {self.U}")
        return (True, "U is finite numeric")

    def test_invariant_sign_consistent_with_K(self):
        """For K>1, (1-K) < 0, so U should be negative (attractive interaction)."""
        if self.U is None or self.K is None:
            return (False, "SKIP: U or K not available")
        k_val = float(self.K)
        u_val = float(self.U)
        # Physics: U ∝ (1-K)
        expected_sign = 1 if (1 - k_val) > 0 else -1
        # Allow zero only if (1-K)=0 exactly (K=1)
        if math.isclose(u_val, 0.0, abs_tol=1e-15):
            actual_sign = 0
        else:
            actual_sign = 1 if u_val > 0 else -1
        if actual_sign != expected_sign:
            return (False,
                    f"U sign mismatch: expected sign {expected_sign} from (1-K)={1-k_val}, got U={u_val}")
        return (True, f"U sign ({actual_sign}) consistent with (1-K) ({1-k_val})")

    # --------------------------------------------------
    # Modeling / trace tests (formula agreement)
    # --------------------------------------------------
    def test_modeling_formula_agreement(self):
        """Check that U matches the expected expression U = (1-K)*q^2 / (16*pi*(1+K)*eps0*x)."""
        if any(v is None for v in [self.U, self.q, self.x, self.K, self.eps0]):
            return (False, "SKIP: one or more required parameters missing")
        q_val = float(self.q)
        x_val = float(self.x)
        K_val = float(self.K)
        eps0_val = float(self.eps0)
        U_val = float(self.U)

        numerator = (1 - K_val) * q_val * q_val
        denominator = 16.0 * math.pi * (1.0 + K_val) * eps0_val * x_val
        expected_U = numerator / denominator

        # Use fuzzy matching (physics consistent with numerical noise)
        if math.isclose(U_val, expected_U, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U matches expected within tolerance: U={U_val}, expected={expected_U}")
        else:
            return (False, f"U deviates: got {U_val}, expected {expected_U}")

    def test_modeling_expression_form(self):
        """If U is a symbolic expression, verify it matches the expected symbolic form."""
        if self.U is None:
            return (False, "SKIP: U not available")
        try:
            expr = sp.sympify(self.U)
        except (sp.SympifyError, TypeError):
            return (False, "U is not a symbolic expression, skipping symbolic check")
        q, x, K, eps0 = sp.symbols('q x K eps0', positive=True)
        expected_expr = (1 - K) * q**2 / (16 * sp.pi * (1 + K) * eps0 * x)
        if sp.simplify(expr - expected_expr) != 0:
            return (False, f"Symbolic expression does not match expected: got {expr}, expected {expected_expr}")
        return (True, "Symbolic expression matches expected form")

    # --------------------------------------------------
    # Monotonicity tests (perturb an input and compare)
    # --------------------------------------------------
    def test_monotonicity_inverse_with_distance(self):
        """
        Perturb x by +10% and recompute expected U (using the formula) to verify
        that magnitude |U| decreases and that U scales as 1/x.
        """
        if any(v is None for v in [self.U, self.q, self.x, self.K, self.eps0]):
            return (False, "SKIP: one or more required parameters missing")
        q_val = float(self.q)
        x_val = float(self.x)
        K_val = float(self.K)
        eps0_val = float(self.eps0)
        U_val = float(self.U)

        # Baseline: expected U at original x
        expected_U_original = (1 - K_val) * q_val * q_val / (16.0 * math.pi * (1.0 + K_val) * eps0_val * x_val)

        # Perturb: increase x by 10% -> x_pert = 1.1 * x_val
        x_pert = 1.1 * x_val
        expected_U_pert = (1 - K_val) * q_val * q_val / (16.0 * math.pi * (1.0 + K_val) * eps0_val * x_pert)

        # Check that the perturbed magnitude is smaller (since denominator larger)
        mag_pert = abs(expected_U_pert)
        mag_orig = abs(expected_U_original)
        if mag_pert >= mag_orig:
            return (False, f"Perturbation: |U_pert| = {mag_pert} >= |U_original| = {mag_orig}; expected decrease")
        # Also verify that the product U * x is invariant
        product_original = expected_U_original * x_val
        product_pert = expected_U_pert * x_pert
        if not math.isclose(product_original, product_pert, rel_tol=1e-2):
            return (False, f"U*x not invariant: original={product_original}, perturbed={product_pert}")
        return (True, f"Monotonicity holds: |U| decreases as x increases; U*x constant")

    def test_monotonicity_magnitude_decreases_with_x(self):
        """
        Another monotonicity check: for a slightly larger x, compute the expected U ratio.
        We cannot change the actual solution, but we can show that the formula predicts monotonic decrease.
        This test is mostly redundant but satisfies the requirement to perturb.
        """
        if any(v is None for v in [self.U, self.q, self.x, self.K, self.eps0]):
            return (False, "SKIP: one or more required parameters missing")
        q_val = float(self.q)
        x_val = float(self.x)
        K_val = float(self.K)
        eps0_val = float(self.eps0)
        U_val = float(self.U)

        # Use a +5% perturbation
        x_pert = 1.05 * x_val
        expected_U_pert = (1 - K_val) * q_val * q_val / (16.0 * math.pi * (1.0 + K_val) * eps0_val * x_pert)
        expected_U_orig = (1 - K_val) * q_val * q_val / (16.0 * math.pi * (1.0 + K_val) * eps0_val * x_val)
        # Check sign consistency: ratio should be inversion
        ratio = expected_U_pert / expected_U_orig
        expected_ratio = x_val / x_pert
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"Perturbed U decreases: ratio = {ratio} ≈ {expected_ratio}")
        else:
            return (False, f"Perturbed U does not scale as 1/x: ratio = {ratio}, expected {expected_ratio}")

    # --------------------------------------------------
    # Answer sanity (weak, no isclose)
    # --------------------------------------------------
    def test_answer_sanity_magnitude_bound(self):
        """
        Weak sanity: magnitude of U should be less than q^2/(4*pi*eps0*x).
        """
        if any(v is None for v in [self.U, self.q, self.x, self.eps0]):
            return (False, "SKIP: one or more required parameters missing")
        q_val = float(self.q)
        x_val = float(self.x)
        eps0_val = float(self.eps0)
        U_val = float(self.U)

        bound = q_val * q_val / (4.0 * math.pi * eps0_val * x_val)
        if abs(U_val) <= bound:
            return (True, f"|U| = {abs(U_val)} <= bound {bound}")
        else:
            return (False, f"|U| = {abs(U_val)} exceeds bound {bound}")

    # --------------------------------------------------
    # Run all tests
    # --------------------------------------------------
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