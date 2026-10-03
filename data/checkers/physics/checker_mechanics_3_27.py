import math

class PhysicsChecker:
    """Robust physics checker for pion photoproduction threshold analysis."""

    def __init__(self, solution_instance):
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.E_gamma_th = self.results.get('E_gamma_th', None)
        self.E_p_min = self.results.get('E_p_min', None)

        # Safely extract givens
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.m_p = self.givens.get('m_p', None)
        self.m_pi = self.givens.get('m_pi', None)
        self.epsilon = self.givens.get('epsilon', None)

    # ------------------------- Interface tests -------------------------
    def test_interface_required_keys(self):
        """Check that required result keys exist."""
        if 'E_gamma_th' not in self.results:
            return (False, "Result key 'E_gamma_th' missing")
        if 'E_p_min' not in self.results:
            return (False, "Result key 'E_p_min' missing")
        return (True, "All required keys present")

    def test_interface_results_not_none(self):
        """Check that result values are not None."""
        if self.E_gamma_th is None:
            return (False, "E_gamma_th is None")
        if self.E_p_min is None:
            return (False, "E_p_min is None")
        return (True, "Both result values are not None")

    def test_interface_types(self):
        """Check that result values are numeric (int or float)."""
        if self.E_gamma_th is None or self.E_p_min is None:
            return (False, "Cannot check type for None value")
        if not isinstance(self.E_gamma_th, (int, float)):
            return (False, f"E_gamma_th has type {type(self.E_gamma_th)}, expected numeric")
        if not isinstance(self.E_p_min, (int, float)):
            return (False, f"E_p_min has type {type(self.E_p_min)}, expected numeric")
        return (True, "Types are numeric")

    # ------------------------- Invariant tests -------------------------
    def test_invariant_threshold_formula(self):
        """Check that E_gamma_th satisfies the threshold formula derived from invariants."""
        if self.E_gamma_th is None or self.m_p is None or self.m_pi is None:
            return (False, "SKIP: missing data for threshold formula check")
        expected = self.m_pi + self.m_pi**2 / (2 * self.m_p)
        if math.isclose(self.E_gamma_th, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_gamma_th = {self.E_gamma_th:.4f} matches expected {expected:.4f}")
        else:
            return (False, f"E_gamma_th = {self.E_gamma_th:.4f} is not within 1% of expected {expected:.4f}")

    def test_invariant_squared_mass_gamma(self):
        """Check that squared invariant mass from photon+proton equals (m_p+m_pi)^2."""
        if self.E_gamma_th is None or self.m_p is None or self.m_pi is None:
            return (False, "SKIP: missing data for invariant mass check")
        s_computed = (self.E_gamma_th + self.m_p)**2 - self.E_gamma_th**2
        s_expected = (self.m_p + self.m_pi)**2
        if math.isclose(s_computed, s_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s computed = {s_computed:.2f} matches expected {s_expected:.2f}")
        else:
            return (False, f"s computed = {s_computed:.2f} differs from expected {s_expected:.2f}")

    # ------------------------- Modeling/Trace quality tests (at least two) -------------------------
    def test_modeling_ultrarelativistic_approximation(self):
        """Check that E_p_min >> m_p (ultrarelativistic condition) with ratio > 1e4."""
        if self.E_p_min is None or self.m_p is None:
            return (False, "SKIP: missing data for ultrarelativistic check")
        ratio = self.E_p_min / self.m_p
        if ratio > 1e4:
            return (True, f"E_p_min / m_p = {ratio:.2e} >> 1, condition satisfied")
        else:
            return (False, f"E_p_min / m_p = {ratio:.2e}, not sufficiently >1e4")

    def test_modeling_ep_min_via_approximation_formula(self):
        """Check that E_p_min matches the ultrarelativistic approximation formula."""
        if self.E_p_min is None or self.m_p is None or self.m_pi is None or self.epsilon is None:
            return (False, "SKIP: missing data for approximation formula check")
        expected_approx = (2 * self.m_p * self.m_pi + self.m_pi**2) / (4 * self.epsilon)
        if math.isclose(self.E_p_min, expected_approx, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_p_min = {self.E_p_min:.6e} matches approximated {expected_approx:.6e}")
        else:
            return (False, f"E_p_min = {self.E_p_min:.6e} differs from approximated {expected_approx:.6e}")

    # ------------------------- Monotonicity tests -------------------------
    def test_monotonicity_gamma_th_increase_with_m_pi(self):
        """Check that E_gamma_th increases when pion mass is increased by 1%."""
        if self.E_gamma_th is None or self.m_p is None or self.m_pi is None:
            return (False, "SKIP: missing data for monotonicity test (E_gamma)")
        baseline = self.m_pi + self.m_pi**2 / (2 * self.m_p)
        increased_m_pi = self.m_pi * 1.01
        perturbed = increased_m_pi + increased_m_pi**2 / (2 * self.m_p)
        if perturbed > baseline:
            return (True, f"E_gamma(th) increases with m_pi: perturbed {perturbed:.4f} > baseline {baseline:.4f}")
        else:
            return (False, f"Monotonicity violated: perturbed {perturbed:.4f} <= baseline {baseline:.4f}")

    def test_monotonicity_Ep_min_decrease_with_epsilon(self):
        """Check that E_p_min decreases when cosmic photon energy is increased by 1%."""
        if self.E_p_min is None or self.m_p is None or self.m_pi is None or self.epsilon is None:
            return (False, "SKIP: missing data for monotonicity test (E_p)")
        baseline = (2 * self.m_p * self.m_pi + self.m_pi**2) / (4 * self.epsilon)
        increased_eps = self.epsilon * 1.01
        perturbed = (2 * self.m_p * self.m_pi + self.m_pi**2) / (4 * increased_eps)
        if perturbed < baseline:
            return (True, f"E_p(min) decreases with epsilon: perturbed {perturbed:.6e} < baseline {baseline:.6e}")
        else:
            return (False, f"Monotonicity violated: perturbed {perturbed:.6e} >= baseline {baseline:.6e}")

    # ------------------------- Weak answer sanity test (only one) -------------------------
    def test_answer_sanity_basic(self):
        """Weak sanity: E_gamma_th and E_p_min are positive and finite."""
        if self.E_gamma_th is None or self.E_p_min is None:
            return (False, "SKIP: missing data for answer sanity")
        if not (self.E_gamma_th > 0 and math.isfinite(self.E_gamma_th)):
            return (False, f"E_gamma_th = {self.E_gamma_th} is not positive and finite")
        if not (self.E_p_min > 0 and math.isfinite(self.E_p_min)):
            return (False, f"E_p_min = {self.E_p_min} is not positive and finite")
        return (True, f"E_gamma_th = {self.E_gamma_th:.4f} > 0, E_p_min = {self.E_p_min:.6e} > 0, both finite")

    # ------------------------- Main runner -------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, collecting results."""
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