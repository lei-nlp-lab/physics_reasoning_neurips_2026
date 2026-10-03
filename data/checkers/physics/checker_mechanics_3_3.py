import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics checker for the relativistic Doppler effect problem.
    Checks interface, invariants, modeling consistency, monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens, default to None for all
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}

        self.tau = givens.get('tau', None)          # emission interval in S (s)
        self.beta_given = givens.get('beta', None)  # symbolic β from parts (a)-(c)
        self.lambda_val = givens.get('lambda', None)  # Hβ wavelength (Å)
        self.V = givens.get('V', None)              # accelerating potential (kV)
        self.m_p_MeV = givens.get('m_p', None)      # proton rest mass (MeV/c^2)

        # Extract results, default to None
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}

        self.tau_prime = results.get('tau_prime', None)
        self.lambda_ratio = results.get('lambda_ratio', None)  # sqrt((1+β)/(1-β))
        self.lambda_prime = results.get('lambda_prime', None)
        self.beta = results.get('beta', None)          # numeric β from part (c)
        self.delta_lambda_1 = results.get('delta_lambda_1', None)
        self.delta_lambda_2 = results.get('delta_lambda_2', None)

        # Helper variable for unit scaling
        self.c = 1.0  # natural units c=1 for ratio checks

    # ---------- Helper for safe numeric conversion ----------
    def _to_float(self, val):
        """Convert a sympy expression or number to float. Return None on failure."""
        if val is None:
            return None
        try:
            if hasattr(val, 'evalf'):
                return float(val.evalf())
            else:
                return float(val)
        except (TypeError, ValueError):
            return None

    # ---------- Interface checks ----------
    def test_interface_required_keys_exist(self):
        """Check that all required result keys are present and not None."""
        required_keys = [
            'tau_prime', 'lambda_ratio', 'lambda_prime',
            'beta', 'delta_lambda_1', 'delta_lambda_2'
        ]
        for key in required_keys:
            if getattr(self, key, None) is None:
                return (False, f"Missing or None result key: '{key}'")
        return (True, "All required keys present")

    # ---------- Invariant checks (use math.isclose) ----------
    def test_invariant_tau_prime_formula(self):
        """Check τ' = τ * sqrt((1+β)/(1-β))."""
        if self.tau_prime is None or self.tau is None or self.beta_given is None:
            return (False, "SKIP: missing τ, β_given, or τ'")
        try:
            τ, β = sp.symbols('τ β')
            expected_expr = τ * sp.sqrt((1+β)/(1-β))
            # Try to evaluate numerically
            tau_num = self._to_float(self.tau)
            beta_num = self._to_float(self.beta_given)
            tau_prime_num = self._to_float(self.tau_prime)
            if None in (tau_num, beta_num, tau_prime_num):
                # Use symbolic substitution if possible
                if hasattr(self.tau_prime, 'subs'):
                    sub = {τ: self.tau, β: self.beta_given}
                    expected = self.tau_prime.subs(sub).simplify()  # should be 0 if correct
                    if expected == 0 or (hasattr(expected, 'evalf') and float(expected.evalf()) < 1e-6):
                        return (True, "τ' expression matches formula symbolically")
                    else:
                        return (False, f"Symbolic mismatch, residual: {expected}")
                else:
                    return (False, "Cannot perform numerical evaluation")
            else:
                # Numerical check
                expected_val = tau_num * math.sqrt((1+beta_num)/(1-beta_num))
                if math.isclose(tau_prime_num, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"τ' matches formula: {tau_prime_num} ≈ {expected_val}")
                else:
                    return (False, f"τ' mismatch: {tau_prime_num} vs {expected_val}")
        except Exception as e:
            return (False, f"Error in tau_prime check: {e}")

    def test_invariant_lambda_ratio_formula(self):
        """Check λ_ratio = sqrt((1+β)/(1-β))."""
        if self.lambda_ratio is None or self.beta_given is None:
            return (False, "SKIP: missing λ_ratio or β_given")
        try:
            β = sp.symbols('β')
            expected_expr = sp.sqrt((1+β)/(1-β))
            ratio_num = self._to_float(self.lambda_ratio)
            beta_num = self._to_float(self.beta_given)
            if None in (ratio_num, beta_num):
                # Symbolic check
                if hasattr(self.lambda_ratio, 'subs'):
                    sub = {β: self.beta_given}
                    diff = self.lambda_ratio.subs(sub) - expected_expr.subs(sub)
                    if hasattr(diff, 'evalf') and float(diff.evalf()) < 1e-6:
                        return (True, "λ_ratio matches formula symbolically")
                    else:
                        return (False, f"Symbolic mismatch, residual: {diff}")
                else:
                    return (False, "Cannot perform numerical evaluation")
            else:
                expected_val = math.sqrt((1+beta_num)/(1-beta_num))
                if math.isclose(ratio_num, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"λ_ratio matches formula: {ratio_num} ≈ {expected_val}")
                else:
                    return (False, f"λ_ratio mismatch: {ratio_num} vs {expected_val}")
        except Exception as e:
            return (False, f"Error in lambda_ratio check: {e}")

    def test_invariant_lambda_prime_via_tau(self):
        """Check λ_ratio ≈ τ_ratio (τ'/τ)."""
        if self.tau_prime is None or self.tau is None or self.lambda_ratio is None:
            return (False, "SKIP: missing τ', τ, or λ_ratio")
        try:
            tau_ratio = self._to_float(self.tau_prime) / self._to_float(self.tau) if self._to_float(self.tau) else None
            if tau_ratio is None:
                return (False, "Cannot compute τ ratio")
            num_lambda_ratio = self._to_float(self.lambda_ratio)
            if num_lambda_ratio is None:
                return (False, "Cannot convert λ_ratio to float")
            if math.isclose(tau_ratio, num_lambda_ratio, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"τ'/τ ≈ λ_ratio: {tau_ratio} ≈ {num_lambda_ratio}")
            else:
                return (False, f"τ'/τ = {tau_ratio} but λ_ratio = {num_lambda_ratio}")
        except Exception as e:
            return (False, f"Error in tau/lambda relation: {e}")

    def test_invariant_lambda_prime_ratio(self):
        """Check λ_prime / λ = λ_ratio."""
        if self.lambda_prime is None or self.lambda_val is None or self.lambda_ratio is None:
            return (False, "SKIP: missing λ', λ, or λ_ratio")
        try:
            computed_ratio = self._to_float(self.lambda_prime) / self._to_float(self.lambda_val)
            expected_ratio = self._to_float(self.lambda_ratio)
            if None in (computed_ratio, expected_ratio):
                return (False, "Cannot convert to float for comparison")
            if math.isclose(computed_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"λ'/λ ≈ λ_ratio: {computed_ratio} ≈ {expected_ratio}")
            else:
                return (False, f"λ'/λ = {computed_ratio} but λ_ratio = {expected_ratio}")
        except Exception as e:
            return (False, f"Error in lambda_prime ratio: {e}")

    # ---------- Modeling/trace checks ----------
    def test_modeling_consistency_delta_shifts(self):
        """
        Check that δλ2 = (δλ1)^2 / (2λ) (derived from δλ1 = λ β, δλ2 = λ β²/2).
        This uses only result values, not hardcoded constants.
        """
        if None in (self.delta_lambda_1, self.delta_lambda_2, self.lambda_val):
            return (False, "SKIP: missing delta_lambda_1, delta_lambda_2, or lambda")
        try:
            d1 = self._to_float(self.delta_lambda_1)
            d2 = self._to_float(self.delta_lambda_2)
            lam = self._to_float(self.lambda_val)
            if None in (d1, d2, lam):
                return (False, "Cannot convert to float")
            expected_d2 = (d1 ** 2) / (2 * lam)
            if math.isclose(d2, expected_d2, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"δλ2 ≈ δλ1²/(2λ): {d2} ≈ {expected_d2}")
            else:
                return (False, f"δλ2 = {d2} but expected {expected_d2}")
        except Exception as e:
            return (False, f"Error in shift consistency: {e}")

    def test_modeling_beta_from_gamma(self):
        """
        Check that beta (part c) is consistent with γ = 1 + eV/(m_p c^2).
        Only if givens are present.
        """
        if self.beta is None or self.V is None or self.m_p_MeV is None:
            return (False, "SKIP: missing beta, V, or m_p")
        try:
            beta_num = self._to_float(self.beta)
            V_num = self._to_float(self.V)
            m_num = self._to_float(self.m_p_MeV)
            if None in (beta_num, V_num, m_num):
                return (False, "Cannot convert to float")
            # Convert V from kV to MeV
            gamma = 1 + (V_num * 1e-3) / m_num
            beta_expected = math.sqrt(1 - 1 / (gamma * gamma))
            if math.isclose(beta_num, beta_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"β = {beta_num} ≈ expected {beta_expected}")
            else:
                return (False, f"β = {beta_num} but expected {beta_expected}")
        except Exception as e:
            return (False, f"Error in beta check: {e}")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_tau_prime_increases_with_beta(self):
        """
        For τ' = τ sqrt((1+β)/(1-β)), increasing β should increase τ'.
        Perturb the available numeric beta (from results) by 10% and recompute.
        """
        if self.tau_prime is None or self.tau is None or self.beta is None:
            return (False, "SKIP: missing numeric values for monotonicity")
        try:
            tau_num = self._to_float(self.tau)
            beta_num = self._to_float(self.beta)
            tau_prime_num = self._to_float(self.tau_prime)
            if None in (tau_num, beta_num, tau_prime_num):
                return (False, "Cannot convert to float")
            # Perturb β upward by 10% (cap at 0.99)
            beta_pert = beta_num * 1.1
            if beta_pert >= 1:
                beta_pert = 0.99
                if beta_pert <= beta_num:
                    return (False, "SKIP: cannot perturb β sufficiently")
            # Compute new τ' using same τ (should be same value from results)
            tau_prime_pert = tau_num * math.sqrt((1+beta_pert)/(1-beta_pert))
            if tau_prime_pert > tau_prime_num:
                return (True, f"τ' increases with β: {tau_prime_num} < {tau_prime_pert}")
            else:
                return (False, f"τ' does not increase: {tau_prime_num} >= {tau_prime_pert}")
        except Exception as e:
            return (False, f"Error in monotonicity: {e}")

    def test_monotonicity_lambda_ratio_increases_with_beta(self):
        """Same for λ_ratio = sqrt((1+β)/(1-β))."""
        if self.lambda_ratio is None or self.beta is None:
            return (False, "SKIP: missing numeric lambda_ratio or beta")
        try:
            beta_num = self._to_float(self.beta)
            ratio_num = self._to_float(self.lambda_ratio)
            if None in (beta_num, ratio_num):
                return (False, "Cannot convert to float")
            beta_pert = beta_num * 1.1
            if beta_pert >= 1:
                beta_pert = 0.99
                if beta_pert <= beta_num:
                    return (False, "SKIP: cannot perturb β sufficiently")
            ratio_pert = math.sqrt((1+beta_pert)/(1-beta_pert))
            if ratio_pert > ratio_num:
                return (True, f"λ_ratio increases with β: {ratio_num} < {ratio_pert}")
            else:
                return (False, f"λ_ratio does not increase: {ratio_num} >= {ratio_pert}")
        except Exception as e:
            return (False, f"Error in monotonicity: {e}")

    # ---------- Answer sanity (weak, no isclose, at most one method) ----------
    def test_answer_sanity_positive_shifts(self):
        """
        Weak check: delta_lambda_1 and delta_lambda_2 must be positive and not too large.
        Provide plausible ranges based on the problem context.
        """
        if self.delta_lambda_1 is None or self.delta_lambda_2 is None:
            return (False, "SKIP: missing shift values")
        try:
            d1 = self._to_float(self.delta_lambda_1)
            d2 = self._to_float(self.delta_lambda_2)
            if None in (d1, d2):
                return (False, "Cannot convert to float")
            if d1 <= 0:
                return (False, f"First-order shift must be positive, got {d1}")
            if d1 > 100:  # 100 Å is a large but still plausible upper bound for this context
                return (False, f"First-order shift too large: {d1} > 100")
            if d2 <= 0:
                return (False, f"Second-order shift must be positive, got {d2}")
            if d2 > 1:  # second order shift should be sub-Å
                return (False, f"Second-order shift too large: {d2} > 1")
            return (True, f"Shifts are positive and plausible: δλ1={d1:.3g}, δλ2={d2:.3g}")
        except Exception as e:
            return (False, f"Error in answer sanity: {e}")

    # ---------- Run all tests ----------
    def run_all_tests(self):
        """
        Discover all methods starting with 'test_', run them, collect logs.
        Returns (all_pass, logs_list).
        """
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