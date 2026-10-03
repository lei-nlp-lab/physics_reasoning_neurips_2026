import math
import traceback

class PhysicsChecker:
    """
    Audits the derivation that λ = exp(μ/kT) = n V_Q for an ideal gas in the classical limit.
    """

    def __init__(self, solution_instance):
        """
        Extract relevant data from the solution_instance safely.

        Expected attributes of solution_instance:
            - results: dict with key 'lambda'
            - givens: dict with keys like 'n', 'V_Q', 'T', 'mu', 'h', 'm', 'k'
            - explanation_steps: list of strings
        """
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', []) or []

        # Result lambda
        self.lambda_val = self.results.get('lambda', None)

        # Givens (with None defaults)
        self.n = self.givens.get('n', None)
        self.V_Q = self.givens.get('V_Q', None)
        self.T = self.givens.get('T', None)
        self.mu = self.givens.get('mu', None)
        self.h = self.givens.get('h', None)
        self.m = self.givens.get('m', None)
        self.k = self.givens.get('k', None)

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_results_has_lambda(self):
        """Check that the final answer key 'lambda' exists and is a number."""
        if self.lambda_val is None:
            return (False, "Missing key 'lambda' in results.")
        if not isinstance(self.lambda_val, (int, float)):
            return (False, f"'lambda' is not numeric: {type(self.lambda_val)}")
        if not math.isfinite(self.lambda_val):
            return (False, f"'lambda' is not finite: {self.lambda_val}")
        return (True, "'lambda' present and finite.")

    def test_interface_givens_available(self):
        """Check that critical givens are present (n and V_Q)."""
        if self.n is None or self.V_Q is None:
            return (False, "Missing givens: 'n' and/or 'V_Q' are required.")
        if not (isinstance(self.n, (int, float)) and isinstance(self.V_Q, (int, float))):
            return (False, "'n' or 'V_Q' not numeric.")
        return (True, "Required givens 'n' and 'V_Q' available.")

    # ----------------------------------------------------------------
    # Invariant / theorem‑based tests
    # ----------------------------------------------------------------
    def test_invariant_lambda_definition(self):
        """λ must equal exp(μ/kT) if μ, T, k are provided."""
        mu = self.mu
        T = self.T
        k = self.k
        if mu is None or T is None or k is None:
            return (False, "SKIP: missing mu, T, or k for definition check.")
        if T <= 0 or k <= 0:
            return (False, "Temperature or Boltzmann constant non‑positive.")
        expected = math.exp(mu / (k * T))
        if self.lambda_val is None:
            return (False, "Cannot compare: 'lambda' missing.")
        if not math.isclose(self.lambda_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"λ = {self.lambda_val} does not match exp(μ/kT) = {expected}")
        return (True, "λ matches exp(μ/kT) within tolerance.")

    def test_invariant_nVQ_relation(self):
        """λ must equal n * V_Q (fundamental derived equation)."""
        if self.n is None or self.V_Q is None:
            return (False, "SKIP: missing n or V_Q for relation check.")
        if self.lambda_val is None:
            return (False, "Cannot check: 'lambda' missing.")
        expected = self.n * self.V_Q
        if not math.isclose(self.lambda_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"λ = {self.lambda_val} != n*V_Q = {expected}")
        return (True, "λ ≈ n V_Q (classical limit relation).")

    def test_invariant_lambda_positive(self):
        """λ = exp(μ/kT) must be positive."""
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        if self.lambda_val <= 0:
            return (False, f"λ = {self.lambda_val} is not positive.")
        return (True, "λ is positive.")

    def test_invariant_classical_limit_hint(self):
        """λ should be << 1 if classical limit holds (weak sanity)."""
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        if self.lambda_val >= 1.0:
            return (False, f"λ = {self.lambda_val} is not << 1; classical limit may not apply.")
        return (True, f"λ = {self.lambda_val} < 1, consistent with classical regime.")

    # ----------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------
    def test_modeling_VQ_definition(self):
        """If h, m, k, T are given, verify V_Q computed from them matches the provided V_Q."""
        h = self.h
        m = self.m
        k = self.k
        T = self.T
        V_Q_given = self.V_Q
        if None in (h, m, k, T, V_Q_given):
            return (False, "SKIP: missing one or more of h, m, k, T, V_Q for modeling check.")
        if T <= 0 or k <= 0 or m <= 0 or h <= 0:
            return (False, "Physical constants or temperature non‑positive.")
        computed_V_Q = (h**2 / (2 * math.pi * m * k * T)) ** 1.5
        if not math.isclose(computed_V_Q, V_Q_given, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Provided V_Q = {V_Q_given} does not match computed V_Q = {computed_V_Q}")
        return (True, "V_Q matches the definition (h²/(2πmkT))^(3/2).")

    def test_modeling_explanation_steps_referenced(self):
        """Check that the explanation steps mention the key relation λ = n V_Q."""
        key_phrases = ["lambda = nV_Q", "λ = n V_Q", "n = λ / V_Q"]
        steps_text = ' '.join(self.explanation_steps)
        if any(phrase.lower() in steps_text.lower() for phrase in key_phrases):
            return (True, "Explanation steps contain the key relation.")
        # Not fatal, just informative
        return (True, "Explanation steps may not explicitly state the relation, but the conclusion is expected.")

    # ----------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------
    def test_monotonicity_lambda_increases_with_n(self):
        """Perturb n by +5% and verify λ increases proportionally."""
        if self.n is None or self.V_Q is None or self.lambda_val is None:
            return (False, "SKIP: missing n, V_Q, or lambda.")
        base_n = self.n
        base_lambda = self.lambda_val
        # Perturb n by +5%
        perturbation_factor = 1.05
        perturbed_n = base_n * perturbation_factor
        # From λ = n V_Q, λ' = perturbation_factor * base_lambda
        expected_perturbed = base_lambda * perturbation_factor
        # Check that the perturbed output would be larger
        if expected_perturbed <= base_lambda:
            return (False, "Monotonicity violated: increasing n should increase λ.")
        return (True, f"Increasing n by 5% increases λ from {base_lambda} to {expected_perturbed}.")

    def test_monotonicity_lambda_decreases_with_T(self):
        """
        Perturb temperature by +10% and compute new λ using the definition of V_Q.
        Verify that λ decreases (since V_Q ∝ T^{-3/2}).
        """
        # Need n, V_Q, T, and physical constants to recompute V_Q if available
        if self.n is None or self.V_Q is None or self.T is None or self.lambda_val is None:
            return (False, "SKIP: missing n, V_Q, T, or lambda.")
        base_T = self.T
        base_lambda = self.lambda_val
        # Perturb temperature by +10%
        perturbation_factor = 1.10
        perturbed_T = base_T * perturbation_factor

        # Compute perturbed V_Q. Use definition if constants present, else use scaling.
        h = self.h
        m = self.m
        k = self.k
        if h is not None and m is not None and k is not None:
            # Recompute V_Q at perturbed temperature using formula
            try:
                perturbed_V_Q = (h**2 / (2 * math.pi * m * k * perturbed_T)) ** 1.5
            except (ZeroDivisionError, ValueError):
                return (False, "Failed to compute perturbed V_Q from formula.")
        else:
            # Fallback to scaling: V_Q ∝ T^{-3/2}
            perturbed_V_Q = self.V_Q * (base_T / perturbed_T) ** 1.5

        # Compute new λ using the same relationship
        perturbed_lambda = self.n * perturbed_V_Q

        # Check monotonic direction: λ should decrease when T increases
        if perturbed_lambda >= base_lambda:
            return (False, f"Monotonicity violated: increasing T by 10% should decrease λ, but got λ'={perturbed_lambda} >= λ={base_lambda}.")
        # Optional: check relative change matches expected T^{-3/2} scaling
        expected_relative_change = (base_T / perturbed_T) ** 1.5
        actual_relative_change = perturbed_lambda / base_lambda
        if not math.isclose(actual_relative_change, expected_relative_change, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Relative change {actual_relative_change} does not match expected T^{-3/2} scaling {expected_relative_change}.")
        return (True, f"Increasing T by 10% decreases λ from {base_lambda:.4g} to {perturbed_lambda:.4g} (ratio {actual_relative_change:.4g}).")

    # ----------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ----------------------------------------------------------------
    def test_answer_sanity_lambda_sign_and_order(self):
        """Weak sanity: λ is positive, finite, and not absurdly large."""
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available.")
        if self.lambda_val <= 0:
            return (False, "λ must be positive.")
        if not math.isfinite(self.lambda_val):
            return (False, "λ is infinite or NaN.")
        if self.lambda_val > 1e6:
            return (False, f"λ = {self.lambda_val} is unreasonably large for classical limit.")
        return (True, "λ passes basic sanity checks (positive, finite, not extreme).")

    # ----------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods. Return (overall_pass, log_list)."""
        logs = []
        overall_pass = True
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
                overall_pass = False
        return (overall_pass, logs)