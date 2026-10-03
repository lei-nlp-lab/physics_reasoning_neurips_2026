import math
import traceback

class PhysicsChecker:
    """
    Audits a solution_instance for a breeder reactor problem.
    Checks:
      - Interface: required keys exist with correct types
      - Invariants: eta_threshold == 2, moderator_allowed == False
      - Modeling: explanation mentions eta > 2 and water unsuitable
      - Monotonicity: eta decreases when alpha (capture-to-fission) increases
      - Weak answer sanity: plausible range for eta_threshold
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.eta_threshold = results.get('eta_threshold', None)
        self.moderator_allowed = results.get('moderator_allowed', None)

        # Attempt to extract alpha and eta for monotonicity test (if present)
        self.alpha = None
        self.eta = None
        # Try from results first
        self.alpha = results.get('alpha', None)
        self.eta = results.get('eta', None)
        # Also try direct attribute on solution_instance
        if self.alpha is None:
            self.alpha = getattr(solution_instance, 'alpha', None)
        if self.eta is None:
            self.eta = getattr(solution_instance, 'eta', None)

        # Extract explanation_steps
        exp = getattr(solution_instance, 'explanation_steps', None)
        if exp is None:
            self.explanation_text = ""
        elif isinstance(exp, str):
            self.explanation_text = exp
        elif isinstance(exp, list):
            self.explanation_text = " ".join(str(step) for step in exp)
        else:
            self.explanation_text = str(exp)

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

    # --- Interface Tests ---
    def test_interface_eta_threshold_exists(self):
        if self.eta_threshold is None:
            return (False, "SKIP: eta_threshold not present")
        return (True, "eta_threshold key found")

    def test_interface_moderator_allowed_exists(self):
        if self.moderator_allowed is None:
            return (False, "SKIP: moderator_allowed not present")
        return (True, "moderator_allowed key found")

    def test_interface_eta_threshold_type(self):
        if self.eta_threshold is None:
            return (False, "SKIP: eta_threshold not available")
        if not isinstance(self.eta_threshold, (int, float)):
            return (False, f"eta_threshold should be numeric, got {type(self.eta_threshold)}")
        return (True, "eta_threshold is numeric")

    def test_interface_moderator_allowed_type(self):
        if self.moderator_allowed is None:
            return (False, "SKIP: moderator_allowed not available")
        if not isinstance(self.moderator_allowed, (bool, int, str)):
            return (False, f"moderator_allowed unexpected type {type(self.moderator_allowed)}")
        return (True, "moderator_allowed acceptable type")

    # --- Invariant / Value Tests ---
    def test_invariant_eta_threshold_value(self):
        if self.eta_threshold is None:
            return (False, "SKIP: eta_threshold not available")
        expected = 2.0
        if not math.isclose(self.eta_threshold, expected, rel_tol=1e-9, abs_tol=1e-9):
            return (False, f"eta_threshold should be {expected}, got {self.eta_threshold}")
        return (True, f"eta_threshold equals {expected}")

    def test_invariant_moderator_allowed_false(self):
        if self.moderator_allowed is None:
            return (False, "SKIP: moderator_allowed not available")
        allowed = self.moderator_allowed
        if isinstance(allowed, bool):
            if allowed is not False:
                return (False, f"moderator_allowed should be False, got {allowed}")
        elif isinstance(allowed, int):
            if allowed != 0:
                return (False, f"moderator_allowed should be 0, got {allowed}")
        elif isinstance(allowed, str):
            if allowed.strip().lower() not in ('false', 'no', '0'):
                return (False, f"moderator_allowed string should indicate false, got {allowed}")
        else:
            return (False, f"moderator_allowed unrecognized type {type(allowed)}")
        return (True, "moderator_allowed indicates breeding not possible with water")

    # --- Modeling / Trace Tests ---
    def test_modeling_explanation_mentions_eta_gt_2(self):
        if not self.explanation_text:
            return (False, "SKIP: explanation_text empty")
        phrases = ["eta > 2", "η > 2", "eta > 2.0", "η > 2.0"]
        found = any(phrase in self.explanation_text for phrase in phrases)
        if not found:
            return (False, "Explanation does not mention 'eta > 2' condition")
        return (True, "Explanation mentions eta > 2 condition")

    def test_modeling_explanation_mentions_water_unsuitable(self):
        if not self.explanation_text:
            return (False, "SKIP: explanation_text empty")
        negations = ["water cannot", "water can't", "water not", "no water",
                     "water is unsuitable", "water cannot be used",
                     "water is not suitable", "not be used as a moderator"]
        found = any(phrase in self.explanation_text.lower() for phrase in negations)
        if not found:
            return (False, "Explanation does not state that water cannot be used")
        return (True, "Explanation states water unsuitable for breeder")

    def test_modeling_explanation_mentions_neutron_balance(self):
        if not self.explanation_text:
            return (False, "SKIP: explanation_text empty")
        phrases = ["neutron balance", "neutrons per fission", "chain reaction", "one neutron"]
        found = any(phrase in self.explanation_text.lower() for phrase in phrases)
        if not found:
            return (False, "Explanation does not reference neutron balance or chain reaction")
        return (True, "Explanation references neutron balance concepts")

    # --- Monotonicity Tests ---
    def test_monotonicity_eta_decreases_with_alpha(self):
        """
        If the solution provides alpha and eta values, check that increasing
        alpha (capture-to-fission ratio) causes eta to decrease.
        Since we cannot modify the solution, we simply verify the relationship
        eta = nu/(1+alpha) is decreasing in alpha.  If nu is not available,
        we check that for any plausible alpha > 0, eta decreases with alpha.
        We use the provided alpha and eta to check the inverse relationship:
        if alpha increases by a small relative amount, the new eta (computed from
        the same formula) should be less than the original eta.
        We assume the solution's eta was computed using the same formula.
        """
        if self.alpha is None or self.eta is None:
            return (False, "SKIP: alpha or eta not available for monotonicity check")
        # Ensure numeric
        if not isinstance(self.alpha, (int, float)) or not isinstance(self.eta, (int, float)):
            return (False, "alpha or eta not numeric")
        if self.alpha <= 0:
            return (False, "alpha must be positive")
        # We need nu to compute the relationship, but we can derive nu from eta and alpha:
        # eta = nu / (1 + alpha) => nu = eta * (1 + alpha)
        nu = self.eta * (1.0 + self.alpha)
        # Now perturb alpha by +5%
        alpha_perturbed = self.alpha * 1.05
        eta_perturbed = nu / (1.0 + alpha_perturbed)
        if eta_perturbed >= self.eta - 1e-12:  # should be strictly less
            return (False, f"Increasing alpha from {self.alpha} to {alpha_perturbed} did not decrease eta (eta_orig={self.eta}, eta_perturbed={eta_perturbed})")
        return (True, "eta decreases monotonically with alpha (perturbed +5%)")

    # --- Weak answer sanity (only one) ---
    def test_answer_sanity_eta_threshold_range(self):
        if self.eta_threshold is None:
            return (False, "SKIP: eta_threshold not available")
        if not (1 < self.eta_threshold < 5):
            return (False, f"eta_threshold {self.eta_threshold} outside plausible range [1,5]")
        return (True, "eta_threshold in plausible range")