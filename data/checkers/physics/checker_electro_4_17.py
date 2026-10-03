import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for the self-trapped beam threshold power problem.
    Checks interface, invariants, modeling consistency, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Safely extract givens (expected keys from pseudocode [givens])
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.D = givens.get('D', None)
        self.lmbda = givens.get('lambda', None)          # vacuum wavelength
        self.n0 = givens.get('n0', None)
        self.n2 = givens.get('n2', None)
        self.c = givens.get('c', None)
        self.epsilon0 = givens.get('epsilon0', None)

        # Safely extract results (expected key: 'P_th')
        results = getattr(solution_instance, 'results', {}) or {}
        self.P_th = results.get('P_th', None)

        # Explanation steps for trace‑consistency checks (not directly used here)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        # Pseudocode / solution plan (not used as final authority)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface tests – check required keys exist and are not None
    # ------------------------------------------------------------------
    def test_interface_P_th_present(self):
        """Check that the result key 'P_th' exists and is not None."""
        if self.P_th is None:
            return (False, "Key 'P_th' missing or None in results.")
        return (True, "P_th is present.")

    def test_interface_P_th_numeric(self):
        """P_th must be a numeric type (int or float)."""
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")
        if not isinstance(self.P_th, (int, float)):
            return (False, f"P_th has wrong type: {type(self.P_th)}")
        return (True, "P_th is numeric.")

    # ------------------------------------------------------------------
    # Invariant tests – physical sanity using theorem‑based relations
    # ------------------------------------------------------------------
    def test_invariant_P_th_positive(self):
        """Threshold power must be positive."""
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")
        if self.P_th <= 0:
            return (False, f"P_th = {self.P_th} is not positive.")
        return (True, "P_th > 0.")

    def test_invariant_P_th_finite(self):
        """Threshold power must be finite (not inf or NaN)."""
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")
        if not math.isfinite(self.P_th):
            return (False, f"P_th is not finite: {self.P_th}")
        return (True, "P_th is finite.")

    # ------------------------------------------------------------------
    # Modeling/trace tests – consistency with physical formulas and scaling
    # ------------------------------------------------------------------
    def test_modeling_threshold_formula(self):
        """
        If all required givens (λ, n₂, c, ε₀) are present, re‑derive P_th
        and compare with the provided result using math.isclose.
        Tolerances: rel_tol=1e-2, abs_tol=1e-2 (allowing for rounding).
        """
        if any(x is None for x in (self.lmbda, self.n2, self.c, self.epsilon0)):
            return (False, "SKIP: not all required givens (lambda, n2, c, epsilon0) present.")
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")

        # Derived formula: P_th = (π c ε₀ (1.22 λ)²) / (16 n₂)
        expected_P = (math.pi * self.c * self.epsilon0 * (1.22 * self.lmbda) ** 2) / (16 * self.n2)
        if math.isclose(self.P_th, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_th matches derived value {expected_P:.6e}.")
        else:
            return (False, f"P_th = {self.P_th} differs from expected {expected_P:.6e}.")

    def test_modeling_proportionality_lambda_squared(self):
        """
        Verify that P_th ∝ λ². Compute the ratio P_th / λ² and compare with
        the expected constant: (π c ε₀ (1.22)²) / (16 n₂).
        """
        if any(x is None for x in (self.lmbda, self.n2, self.c, self.epsilon0)):
            return (False, "SKIP: not all required givens (lambda, n2, c, epsilon0) present.")
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")
        if self.lmbda == 0:
            return (False, "Lambda is zero, cannot compute ratio.")

        expected_ratio = (math.pi * self.c * self.epsilon0 * (1.22 ** 2)) / (16 * self.n2)
        actual_ratio = self.P_th / (self.lmbda ** 2)
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_th/λ² = {actual_ratio:.6e} matches expected {expected_ratio:.6e}.")
        else:
            return (False, f"P_th/λ² = {actual_ratio:.6e} differs from expected {expected_ratio:.6e}.")

    def test_modeling_proportionality_n2_inverse(self):
        """
        Verify that P_th ∝ 1/n₂. Compute the product P_th * n₂ and compare with
        the expected constant: (π c ε₀ (1.22 λ)²) / 16.
        """
        if any(x is None for x in (self.lmbda, self.n2, self.c, self.epsilon0)):
            return (False, "SKIP: not all required givens (lambda, n2, c, epsilon0) present.")
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")
        if self.n2 == 0:
            return (False, "n₂ is zero, cannot compute product.")

        expected_product = (math.pi * self.c * self.epsilon0 * (1.22 * self.lmbda) ** 2) / 16
        actual_product = self.P_th * self.n2
        if math.isclose(actual_product, expected_product, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_th * n₂ = {actual_product:.6e} matches expected {expected_product:.6e}.")
        else:
            return (False, f"P_th * n₂ = {actual_product:.6e} differs from expected {expected_product:.6e}.")

    # ------------------------------------------------------------------
    # Monotonicity test – input perturbation checks directional trends
    # ------------------------------------------------------------------
    def test_monotonicity_n2_increase(self):
        """
        If n₂ increases, the threshold power must decrease.
        Perturb n₂ by +5% and recompute P_th from the formula (if givens available).
        Use the baseline formula to avoid dependence on the answer itself.
        """
        if any(x is None for x in (self.lmbda, self.n2, self.c, self.epsilon0)):
            return (False, "SKIP: not all required givens present for monotonicity.")
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")

        # Baseline from givens (same as modeling test)
        baseline = (math.pi * self.c * self.epsilon0 * (1.22 * self.lmbda) ** 2) / (16 * self.n2)

        # Perturb n₂ upward by 5%
        n2_perturbed = self.n2 * 1.05
        perturbed = (math.pi * self.c * self.epsilon0 * (1.22 * self.lmbda) ** 2) / (16 * n2_perturbed)

        if perturbed < baseline:
            return (True, f"Increasing n₂ by 5% reduces P_th (baseline {baseline:.6e}, perturbed {perturbed:.6e}).")
        else:
            return (False, f"Monotonicity violated: perturbed {perturbed:.6e} not less than baseline {baseline:.6e}.")

    # ------------------------------------------------------------------
    # Optional weak answer‑sanity test (secondary, no math.isclose)
    # Only checks sign and broad magnitude range.
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_check(self):
        """Weak sanity: P_th must be positive and within a reasonable range (1e-20 to 1e10)."""
        if self.P_th is None:
            return (False, "SKIP: P_th not available.")
        if self.P_th <= 0:
            return (False, "P_th is not positive.")
        if self.P_th > 1e10 or self.P_th < 1e-20:   # extremely broad range
            return (False, f"P_th = {self.P_th} lies outside typical expected range [1e-20, 1e10].")
        return (True, "P_th is positive and within a plausible magnitude range.")

    # ------------------------------------------------------------------
    # Runner – discover and execute all test_* methods
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all bound method names starting with "test_"
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED with exception: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)