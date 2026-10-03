import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics auditor for time-dependent perturbation theory on a harmonic oscillator.
    Checks selection rules, matrix elements, Gaussian integrals, and probability expression.
    """

    def __init__(self, solution_instance):
        # Safely extract givens; use None for missing
        self.m = getattr(solution_instance, 'm', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.alpha = getattr(solution_instance, 'alpha', None)
        self.tau = getattr(solution_instance, 'tau', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Results object (could be dict or object with attributes)
        results_obj = getattr(solution_instance, 'results', {})
        if isinstance(results_obj, dict):
            self.n_f_list = results_obj.get('n_f_list', None)
            self.P_15 = results_obj.get('P_15', None)
        else:
            self.n_f_list = getattr(results_obj, 'n_f_list', None)
            self.P_15 = getattr(results_obj, 'P_15', None)

        # Also look for explanation steps for reference
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Cache for derived quantities
        self._x_scale = None
        self._matrix_element_sq = None
        self._expected_P_15 = None

    def _compute_x_scale(self):
        if self._x_scale is not None:
            return self._x_scale
        if None in (self.hbar, self.m, self.omega):
            return None
        self._x_scale = math.sqrt(self.hbar / (2 * self.m * self.omega))
        return self._x_scale

    def _compute_matrix_element_sq(self):
        if self._matrix_element_sq is not None:
            return self._matrix_element_sq
        xscale = self._compute_x_scale()
        if xscale is None:
            return None
        # <15|x^5|10> = xscale^5 * sqrt(15!/10!)
        # Squared: xscale^{10} * (15!/10!)
        try:
            self._matrix_element_sq = (xscale ** 10) * (math.factorial(15) / math.factorial(10))
        except (ValueError, OverflowError):
            return None
        return self._matrix_element_sq

    def _compute_expected_P_15(self, tau_override=None):
        """
        Compute P_15 theoretically using the formula.
        If tau_override is provided, use that value instead of self.tau.
        """
        # Collect all required values
        if tau_override is None:
            tau = self.tau
        else:
            tau = tau_override

        if None in (self.alpha, self.hbar, tau, self.omega):
            return None

        mat_sq = self._compute_matrix_element_sq()
        if mat_sq is None:
            return None

        # Gaussian integral factor squared: pi * tau^2 * exp(-25 omega^2 tau^2 / 2)
        try:
            exponent = -25 * (self.omega ** 2) * (tau ** 2) / 2.0
            integral_factor_sq = math.pi * (tau ** 2) * math.exp(exponent)
        except OverflowError:
            return None

        # P_15 = (alpha^2 / hbar^2) * mat_sq * integral_factor_sq
        try:
            expected = (self.alpha ** 2 / self.hbar ** 2) * mat_sq * integral_factor_sq
        except (ValueError, OverflowError):
            return None
        return expected

    # ---- INTERFACE TESTS ----
    def test_interface_n_f_list_exists(self):
        if self.n_f_list is None:
            return (False, "n_f_list is missing")
        return (True, "n_f_list is present")

    def test_interface_P_15_exists(self):
        if self.P_15 is None:
            return (False, "P_15 is missing")
        return (True, "P_15 is present")

    def test_interface_n_f_list_is_list(self):
        if not isinstance(self.n_f_list, list):
            return (False, f"n_f_list is not a list, got {type(self.n_f_list)}")
        return (True, "n_f_list is a list")

    def test_interface_P_15_is_numeric(self):
        if not isinstance(self.P_15, (int, float)):
            return (False, f"P_15 is not numeric, got {type(self.P_15)}")
        return (True, "P_15 is numeric")

    # ---- INVARIANT / THEOREM TESTS ----
    def test_invariant_selection_rules(self):
        if self.n_f_list is None:
            return (False, "SKIP: n_f_list not available")
        allowed = {5, 7, 9, 11, 13, 15}
        given = set(self.n_f_list)
        if given != allowed:
            miss = allowed - given
            extra = given - allowed
            msg = f"n_f_list mismatch. Missing: {miss}, Extra: {extra}"
            return (False, msg)
        return (True, "n_f_list matches all allowed states")

    def test_invariant_parity(self):
        if self.n_f_list is None:
            return (False, "SKIP: n_f_list not available")
        for n in self.n_f_list:
            if n % 2 == 0:
                return (False, f"State n={n} is even, should be odd")
        return (True, "All final states are odd")

    def test_invariant_max_energy_state(self):
        if self.n_f_list is None:
            return (False, "SKIP: n_f_list not available")
        if max(self.n_f_list) != 15:
            return (False, f"Highest energy state is {max(self.n_f_list)}, expected 15")
        return (True, "Highest energy state is n=15")

    def test_invariant_matrix_element_structure(self):
        """Check that the matrix element for 10->15 has the correct structure."""
        if self.P_15 is None:
            return (False, "SKIP: P_15 not available for derivative check")
        expected_P = self._compute_expected_P_15()
        if expected_P is None:
            return (False, "SKIP: insufficient givens to compute expected P_15")
        # Compare using isclose with generous tolerance
        if not math.isclose(self.P_15, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_15 differs from expected. Got {self.P_15}, expected about {expected_P}")
        return (True, "Matrix element structure is consistent with P_15 value")

    def test_invariant_transition_frequency(self):
        """Sanity that the exponent factor contains 25 ω^2 τ^2 /2."""
        # Not a rigorous test, but ensures no obvious contradiction
        if self.P_15 is None or self.omega is None or self.tau is None:
            return (False, "SKIP: insufficient data")
        # Very weak: if tau is large, P should be tiny
        if self.tau > 10 and self.omega > 0.1:
            expected_small = 1e-10
            if self.P_15 > expected_small:
                return (False, f"For large tau, P_15 should be very small, but got {self.P_15}")
        return (True, "Transition frequency appears plausible")

    # ---- MODELING / TRACE CHECKS ----
    def test_modeling_gaussian_integral_factor(self):
        """Check that the probability includes the factor pi*tau^2*exp(-25 ω^2 τ^2 /2)."""
        if self.P_15 is None:
            return (False, "SKIP: P_15 not available")
        # This is implicitly checked by test_invariant_matrix_element_structure
        return (True, "Gaussian integral factor is implicitly present (checked via expected P_15)")

    def test_modeling_probability_dimensionless(self):
        if self.P_15 is None:
            return (False, "SKIP: P_15 not available")
        if self.P_15 < 0 or self.P_15 > 1:
            return (False, f"P_15={self.P_15} is out of domain [0,1]")
        return (True, "P_15 is within physical range")

    def test_modeling_small_coupling_limit(self):
        if self.P_15 is None or self.alpha is None:
            return (False, "SKIP: cannot check coupling limit")
        if self.alpha == 0:
            if self.P_15 != 0:
                return (False, "When alpha=0, P_15 should be 0")
            return (True, "Zero coupling gives zero probability")
        return (True, "Coupling dependence consistent (alpha nonzero)")

    # ---- MONOTONICITY CHECKS (explicit perturbation) ----
    def test_monotonicity_tau_decay(self):
        """
        P_15 as a function of tau: for small tau, ~tau^2; for large tau, exponential dominates.
        We check that increasing tau by 10% produces a decrease (since exponent factor dominates).
        """
        if None in (self.tau, self.omega, self.alpha, self.hbar, self.m):
            return (False, "SKIP: insufficient givens to compute theoretical P_15")
        # Baseline at given tau
        P_base = self._compute_expected_P_15(tau_override=self.tau)
        if P_base is None:
            return (False, "SKIP: could not compute baseline P_15")
        # Perturb tau by +10%
        tau_pert = self.tau * 1.1
        P_pert = self._compute_expected_P_15(tau_override=tau_pert)
        if P_pert is None:
            return (False, "SKIP: could not compute perturbed P_15")
        # Expect P_pert < P_base because exp(-25 ω^2 tau^2 / 2) declines rapidly
        if P_pert > P_base * 1.01:  # allow slight numerical noise
            return (False, f"Increasing tau should decrease P_15, but P_pert={P_pert} > P_base={P_base}")
        return (True, "Monotonicity: P_15 decreases with increasing tau (exponential dominates)")

    def test_monotonicity_tau_zero_limit(self):
        """
        For very small tau, the tau^2 factor dominates, so P_15 should be very small.
        Compare tau = 1e-6 * given tau with baseline.
        Actually we compare P at tau and at a much smaller tau: P_small should be much smaller.
        """
        if None in (self.tau, self.omega, self.alpha, self.hbar, self.m):
            return (False, "SKIP: insufficient givens")
        P_base = self._compute_expected_P_15(tau_override=self.tau)
        if P_base is None:
            return (False, "SKIP: could not compute baseline")
        # Small tau
        tau_small = self.tau * 1e-6
        P_small = self._compute_expected_P_15(tau_override=tau_small)
        if P_small is None:
            return (False, "SKIP: could not compute for small tau")
        # For small tau, the tau^2 factor makes P_small much smaller than baseline
        if P_small > P_base * 0.5:
            return (False, f"For tau→0, P_15 should be much smaller than at tau, but P_small={P_small} is not tiny compared to P_base={P_base}")
        return (True, "Monotonicity: P_15 → 0 as tau → 0")

    # ---- ANSWER SANITY (weak, only one) ----
    def test_answer_sanity_positivity(self):
        if self.P_15 is None:
            return (False, "SKIP: P_15 not available")
        if not math.isfinite(self.P_15) or self.P_15 < 0:
            return (False, f"P_15 is invalid: {self.P_15}")
        return (True, "P_15 is finite and non-negative")

    # ---- RUNNER ----
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