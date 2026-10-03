import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits solutions to the adiabatic demagnetization problem.
    Checks interface completeness, invariant relations, modeling quality,
    monotonic trends, and a weak final answer sanity test.
    """

    def __init__(self, solution_instance):
        # Extract result keys safely
        res = getattr(solution_instance, 'results', {})
        self.S_func = res.get('S_func', None)

        # Extract optional givens for numerical checks
        givens = getattr(solution_instance, 'givens', {})
        self.k = givens.get('k', None)
        self.mu = givens.get('mu', None)
        self.H = givens.get('H', None)
        self.T = givens.get('T', None)
        self.beta = givens.get('beta', None)

        # Optional explanation steps for textual checks
        self.explanation = getattr(solution_instance, 'explanation_steps', None) or []

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_S_func_exists(self):
        if self.S_func is None:
            return (False, "SKIP: 'S_func' not provided in results")
        return (True, "S_func key present")

    def test_interface_S_func_is_string(self):
        if self.S_func is None:
            return (False, "SKIP: 'S_func' is missing")
        if not isinstance(self.S_func, str):
            return (False, f"S_func is not a string, got {type(self.S_func).__name__}")
        return (True, "S_func is a string")

    def test_interface_givens_partial(self):
        # Checks that at least some givens are supplied for numerical tests
        if self.k is not None and self.mu is not None:
            return (True, "k and mu are available for numerical tests")
        return (True, "SKIP: Numerical givens not fully provided; proceeding with textual tests")

    # ------------------------------------------------------------------
    # Invariant tests (numeric + fuzzy)
    # ------------------------------------------------------------------
    def test_invariant_entropy_numerical_H_T_ratio(self):
        """
        Verify that for two different (H,T) pairs with the same H/T ratio,
        the computed entropy (using the standard formula) is the same.
        """
        if None in (self.k, self.mu, self.H, self.T):
            return (True, "SKIP: Insufficient givens; cannot run numerical invariant test")

        beta = 1.0 / (self.k * self.T)
        x = self.mu * self.H * beta
        # Compute S using the formula
        S1 = self.k * (math.log(2 * math.cosh(x)) - x * math.tanh(x))

        # Create a second pair with same H/T ratio, e.g., double both H and T
        H2 = 2.0 * self.H
        T2 = 2.0 * self.T
        beta2 = 1.0 / (self.k * T2)
        x2 = self.mu * H2 * beta2
        S2 = self.k * (math.log(2 * math.cosh(x2)) - x2 * math.tanh(x2))

        if math.isclose(S1, S2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S same for same H/T ratio: S1={S1:.4f}, S2={S2:.4f}")
        else:
            return (False, f"S differs for same H/T ratio: S1={S1:.4f}, S2={S2:.4f}")

    def test_invariant_entropy_function_contains_cosh_tanh(self):
        """
        Check that the string representation of S_func includes the
        hyperbolic functions that naturally appear in the spin-1/2
        partition function.
        """
        if self.S_func is None:
            return (False, "SKIP: S_func missing")
        s = self.S_func.lower()
        has_cosh = 'cosh' in s
        has_tanh = 'tanh' in s
        if has_cosh and has_tanh:
            return (True, "S_func contains cosh and tanh as expected")
        else:
            msg = []
            if not has_cosh:
                msg.append("missing cosh")
            if not has_tanh:
                msg.append("missing tanh")
            return (False, f"S_func {', '.join(msg)}")

    def test_invariant_adiabatic_constant_H_over_T_mentioned(self):
        """
        Check that the explanation steps (if present) mention that
        H/T is constant during an adiabatic process.
        """
        if not self.explanation:
            return (True, "SKIP: no explanation steps to check")
        combined = ' '.join(self.explanation).lower()
        # Look for key phrases
        key_phrases = ['h/t constant', 'h/t conserved', 'constant h/t',
                       'h/t remains constant', 'adiabatic invariant h/t']
        if any(phrase in combined for phrase in key_phrases):
            return (True, "Explanation mentions constant H/T during adiabatic process")
        else:
            # Broader check: look for "H/T" together with "constant" or "invariant"
            if ('h/t' in combined or 'h/t' in combined) and \
               ('constant' in combined or 'invariant' in combined):
                return (True, "Broad check: H/T and constant/invariant found")
            return (False, "No mention of H/T being constant in adiabatic process")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_cooling_mentioned(self):
        """
        Check that the explanation states that reducing H after
        isolation leads to a decrease in temperature.
        """
        if not self.explanation:
            return (True, "SKIP: no explanation steps to check")
        combined = ' '.join(self.explanation).lower()
        cooling_phrases = ['cooling', 'temperature decrease', 'temperature drops',
                           't decreases', 'drops to maintain', 'adiabatic demagnetization cooling']
        if any(phrase in combined for phrase in cooling_phrases):
            return (True, "Cooling effect is mentioned in explanation")
        return (False, "No mention of cooling due to adiabatic demagnetization")

    def test_modeling_negative_temperature_discussed(self):
        """
        Check that the explanation discusses negative temperature and
        the condition for it (population inversion, field reversal, etc.)
        """
        if not self.explanation:
            return (True, "SKIP: no explanation steps to check")
        combined = ' '.join(self.explanation).lower()
        neg_phrases = ['negative temperature', 't < 0', 'population inversion',
                       'spin inversion', 'field reversal', 'higher energy state']
        if any(phrase in combined for phrase in neg_phrases):
            return (True, "Negative temperature is discussed")
        return (False, "Negative temperature not discussed in explanation")

    def test_modeling_low_temperature_use_explained(self):
        """
        Check that the explanation addresses why adiabatic demagnetization
        is used only at low temperatures (lattice entropy dominates).
        """
        if not self.explanation:
            return (True, "SKIP: no explanation steps to check")
        combined = ' '.join(self.explanation).lower()
        lowT_phrases = ['low temperature', 'lattice entropy', 'dominates',
                        'low t', 'below 1 k', 'very low temperature']
        if any(phrase in combined for phrase in lowT_phrases):
            return (True, "Low-temperature use justification present")
        return (False, "Explanation lacks discussion on low-temperature regime")

    def test_modeling_third_law_mentioned(self):
        """
        Check that the impossibility of reaching T=0 is acknowledged
        (third law or infinite population inversion).
        """
        if not self.explanation:
            return (True, "SKIP: no explanation steps to check")
        combined = ' '.join(self.explanation).lower()
        thirdlaw_phrases = ['third law', 'cannot reach zero', 'impossible to reach 0',
                            'infinite population inversion', 't = 0 not possible',
                            'absolute zero not attainable']
        if any(phrase in combined for phrase in thirdlaw_phrases):
            return (True, "Impossibility of reaching T=0 is mentioned")
        return (False, "No mention about impossibility of T=0")

    # ------------------------------------------------------------------
    # Monotonicity tests (numeric perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_entropy_decreases_with_field(self):
        """
        At fixed T, increasing H should decrease entropy (more alignment).
        Use the standard formula if givens are available.
        """
        if None in (self.k, self.mu, self.H, self.T):
            return (True, "SKIP: Cannot compute monotonicity without givens")
        # Base entropy
        beta = 1.0 / (self.k * self.T)
        x0 = self.mu * self.H * beta
        S0 = self.k * (math.log(2 * math.cosh(x0)) - x0 * math.tanh(x0))

        # Perturb H upward by 10%
        H_pert = 1.1 * self.H
        x1 = self.mu * H_pert * beta
        S1 = self.k * (math.log(2 * math.cosh(x1)) - x1 * math.tanh(x1))

        if S1 < S0:
            return (True, f"Entropy decreases when H increases: S0={S0:.6f}, S1={S1:.6f}")
        else:
            return (False, f"Entropy did not decrease (S0={S0:.6f}, S1={S1:.6f})")

    def test_monotonicity_adiabatic_cooling(self):
        """
        Under constant S (adiabatic), if H decreases then T must decrease.
        We simulate by reducing H and computing the T that keeps S constant,
        then check that T_new < T_initial.
        """
        if None in (self.k, self.mu, self.H, self.T):
            return (True, "SKIP: Cannot compute without givens")
        # Compute initial entropy
        beta0 = 1.0 / (self.k * self.T)
        x0 = self.mu * self.H * beta0
        S_const = self.k * (math.log(2 * math.cosh(x0)) - x0 * math.tanh(x0))
        # Reduce H to 90% of original
        H_new = 0.9 * self.H
        # Find T_new such that S is the same (solve transcendental equation)
        # We'll search numerically by scanning T or use simple relation:
        # H/T constant => T_new = T * (H_new/H)
        T_new_approx = self.T * (H_new / self.H)
        # Verify that with this T_new, entropy is (approximately) S_const
        beta_new = 1.0 / (self.k * T_new_approx)
        x_new = self.mu * H_new * beta_new
        S_new = self.k * (math.log(2 * math.cosh(x_new)) - x_new * math.tanh(x_new))
        if math.isclose(S_new, S_const, rel_tol=1e-2, abs_tol=1e-2):
            if T_new_approx < self.T:
                return (True, f"Adiabatic cooling: H reduced, T decreases to {T_new_approx:.4f} (initial T={self.T:.4f})")
            else:
                return (False, "T did not decrease despite H reduction")
        else:
            return (False, f"Constant S condition not satisfied: S_const={S_const:.6f}, S_new={S_new:.6f}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (no re-derivation, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Very light check on the final S_func string: should not contain
        obviously forbidden functions (sin, cos, polynomial, etc.) and
        should contain log (or ln) because the entropy formula includes ln Z.
        """
        if self.S_func is None:
            return (False, "SKIP: S_func not present to santiy-check")
        s = self.S_func.lower()
        # Must contain natural log
        if 'ln' not in s and 'log' not in s:
            return (False, "S_func should contain ln or log (entropy formula)")
        # Should not contain functions that don't appear in the physics:
        # e.g., sin, cos, tan, sqrt (sqrt might appear but unlikely)
        forbidden = ['sin', 'cos', 'tan', 'sqrt', 'exp']
        found_bad = [f for f in forbidden if f in s and f not in ['cosh', 'tanh']]
        if found_bad:
            return (False, f"S_func contains unexpected functions: {', '.join(found_bad)}")
        # Check that it mentions H/T or mu*H/(k*T) indirectly (presence of 'h' and 't')
        if 'h' in self.S_func and 't' in self.S_func:
            return (True, "Weak sanity passed: S_func mentions H and T")
        return (True, "Weak sanity passed (no obvious errors)")

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)