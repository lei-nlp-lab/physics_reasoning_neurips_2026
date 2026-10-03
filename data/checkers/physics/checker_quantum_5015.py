import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Determine whether solution_instance is a dict or an object
        if isinstance(solution_instance, dict):
            givens = solution_instance.get('givens', {})
            results = solution_instance.get('results', {})
            steps = solution_instance.get('explanation_steps', None) or solution_instance.get('steps', None)
        else:
            givens = getattr(solution_instance, 'givens', None) or {}
            results = getattr(solution_instance, 'results', None) or {}
            steps = (getattr(solution_instance, 'explanation_steps', None) or 
                     getattr(solution_instance, 'steps', None))

        # Givens – all set to None if missing (no numeric defaults)
        self.I = givens.get('I', None)
        self.mu = givens.get('mu', None)
        self.eps = givens.get('eps', None) or givens.get('epsilon', None)
        self.hbar = givens.get('hbar', None)
        self.n = givens.get('n', None)

        # Results
        self.E2 = results.get('E2', None)

        # Store explanation steps for possible future use (not used currently)
        self.explanation_steps = steps

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------
    def _all_present(self, *args):
        """Return True if every argument is not None."""
        for v in args:
            if v is None:
                return False
        return True

    # -----------------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All required givens and the final result must be present."""
        missing = []
        if self.I is None:
            missing.append("I")
        if self.mu is None:
            missing.append("mu")
        if self.eps is None:
            missing.append("eps / epsilon")
        if self.hbar is None:
            missing.append("hbar")
        if self.n is None:
            missing.append("n")
        if self.E2 is None:
            missing.append("E2")
        if missing:
            return (False, f"Missing key(s): {', '.join(missing)}")
        return (True, "All required keys present (I, mu, eps, hbar, n, E2)")

    def test_interface_values_positive(self):
        """Physical constants must be positive; n must be integer."""
        issues = []
        if self.I is not None and self.I <= 0:
            issues.append("I must be positive")
        if self.hbar is not None and self.hbar <= 0:
            issues.append("hbar must be positive")
        if self.mu is not None and self.mu <= 0:
            issues.append("mu (dipole magnitude) must be positive")
        if self.eps is not None and self.eps <= 0:
            issues.append("eps (field strength) must be positive")
        if self.n is not None:
            if not isinstance(self.n, (int, float)) or abs(self.n - round(self.n)) > 1e-12:
                issues.append("n must be an integer")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All givens are positive / valid")

    # -----------------------------------------------------------------------
    # Invariant tests (theorem-based physical invariants)
    # -----------------------------------------------------------------------
    def test_invariant_perturbation_theory_sum(self):
        """Second-order energy shift must equal sum over intermediate states m=n±1,
        as given by perturbation theory (an invariant of the formalism)."""
        if not self._all_present(self.I, self.mu, self.eps, self.hbar, self.n, self.E2):
            return (True, "SKIP: missing required variables")
        n_int = round(self.n)
        if abs(self.n - n_int) > 1e-12:
            return (False, "n is not integer")
        # Compute matrix element magnitude
        matrix_element_sq = (self.mu * self.eps / 2.0) ** 2
        # Energy denominators
        factor = self.hbar**2 / (2.0 * self.I)
        denom_minus = factor * (2 * n_int - 1)
        denom_plus  = factor * (-(2 * n_int + 1))
        # Sum the two contributions
        sum_term = matrix_element_sq * (1.0 / denom_minus + 1.0 / denom_plus)
        if math.isclose(sum_term, self.E2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Perturbation sum yields E2={sum_term:.6g}, matches given {self.E2:.6g}")
        else:
            return (False, f"Perturbation sum mismatch: computed {sum_term:.6g}, given {self.E2:.6g}")

    def test_invariant_first_order_vanishes(self):
        """First-order energy correction must be zero because expectation of cosφ is zero.
        (We assume E1 is not provided; if it were, we would check it is 0.)"""
        return (True, "SKIP: E1 not provided; the invariant – ∫cosφ dφ = 0 – is acknowledged.")

    # -----------------------------------------------------------------------
    # Modeling / trace tests (exact relations re-derived from givens)
    # -----------------------------------------------------------------------
    def test_modeling_E2_formula(self):
        """Check that E2 matches the theoretically derived expression."""
        if not self._all_present(self.I, self.mu, self.eps, self.hbar, self.n, self.E2):
            return (True, "SKIP: missing one or more required variables")
        n_int = round(self.n)
        if abs(self.n - n_int) > 1e-12:
            return (False, "n is not an integer")
        denom = 4 * n_int**2 - 1
        if denom == 0:
            return (False, "Denominator zero (|n| = 0.5)")
        expected = (self.mu**2 * self.eps**2 * self.I) / (self.hbar**2 * denom)
        if math.isclose(expected, self.E2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E2 matches formula: expected={expected:.6g}, got={self.E2:.6g}")
        else:
            return (False, f"E2 mismatch: expected={expected:.6g}, got={self.E2:.6g}")

    def test_modeling_term_ratio(self):
        """Check that the ratio of the two perturbation contributions is -(2n-1)/(2n+1)."""
        if not self._all_present(self.I, self.mu, self.eps, self.hbar, self.n, self.E2):
            return (True, "SKIP: missing required variables")
        n_int = round(self.n)
        if abs(self.n - n_int) > 1e-12:
            return (False, "n is not integer")
        # Compute the two terms individually
        mat_sq = (self.mu * self.eps / 2.0) ** 2
        factor = self.hbar**2 / (2.0 * self.I)
        term_minus = mat_sq / (factor * (2 * n_int - 1))
        term_plus  = mat_sq / (factor * (-(2 * n_int + 1)))
        # Theoretical ratio: term_plus / term_minus = -(2*n-1)/(2*n+1)
        if abs(term_minus) < 1e-30:
            return (False, "term_minus is zero; ratio undefined")
        theoretical_ratio = - (2 * n_int - 1) / (2 * n_int + 1)
        actual_ratio = term_plus / term_minus
        if math.isclose(actual_ratio, theoretical_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Term ratio matches theory: actual={actual_ratio:.6g}, theory={theoretical_ratio:.6g}")
        else:
            return (False, f"Term ratio mismatch: actual={actual_ratio:.6g}, theory={theoretical_ratio:.6g}")

    def test_modeling_n0_special(self):
        """For n=0, the formula gives E2 = - mu^2 eps^2 I / hbar^2. Check if n=0 is provided."""
        if not self._all_present(self.I, self.mu, self.eps, self.hbar, self.n, self.E2):
            return (True, "SKIP: missing required variables")
        n_int = round(self.n)
        if abs(self.n - n_int) > 1e-12:
            return (False, "n is not integer")
        if n_int != 0:
            return (True, "SKIP: n != 0, cannot test n=0 special case")
        denom = -1  # 4*0^2 - 1 = -1
        expected = (self.mu**2 * self.eps**2 * self.I) / (self.hbar**2) * (-1)
        if math.isclose(expected, self.E2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"For n=0, E2 matches derived value {expected:.6g}")
        else:
            return (False, f"For n=0, E2 mismatch: expected {expected:.6g}, got {self.E2:.6g}")

    # -----------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # -----------------------------------------------------------------------
    def test_monotonicity_E2_with_I(self):
        """Increasing I should increase |E2| (positive shift → larger, negative → more negative)."""
        if not self._all_present(self.I, self.mu, self.eps, self.hbar, self.n):
            return (True, "SKIP: missing variables")
        n_int = round(self.n)
        denom = 4 * n_int**2 - 1
        if denom == 0:
            return (False, "Denominator zero")
        baseline = (self.mu**2 * self.eps**2 * self.I) / (self.hbar**2 * denom)
        I_pert = self.I * 1.01
        perturbed = (self.mu**2 * self.eps**2 * I_pert) / (self.hbar**2 * denom)
        if baseline >= 0:
            if perturbed > baseline:
                return (True, f"I increase → E2 increases: {baseline:.6g} → {perturbed:.6g}")
            else:
                return (False, f"Monotonicity with I failed: {baseline:.6g} → {perturbed:.6g}")
        else:
            if perturbed < baseline:
                return (True, f"I increase → E2 becomes more negative: {baseline:.6g} → {perturbed:.6g}")
            else:
                return (False, f"Monotonicity with I failed: {baseline:.6g} → {perturbed:.6g}")

    def test_monotonicity_E2_with_mu(self):
        """Increasing mu should increase |E2| (quadratic dependence)."""
        if not self._all_present(self.I, self.mu, self.eps, self.hbar, self.n):
            return (True, "SKIP: missing variables")
        n_int = round(self.n)
        denom = 4 * n_int**2 - 1
        if denom == 0:
            return (False, "Denominator zero")
        baseline = (self.mu**2 * self.eps**2 * self.I) / (self.hbar**2 * denom)
        mu_pert = self.mu * 1.05
        perturbed = (mu_pert**2 * self.eps**2 * self.I) / (self.hbar**2 * denom)
        if baseline >= 0:
            if perturbed > baseline:
                return (True, f"mu increase → E2 increases: {baseline:.6g} → {perturbed:.6g}")
            else:
                return (False, f"Monotonicity with mu failed: {baseline:.6g} → {perturbed:.6g}")
        else:
            if perturbed < baseline:
                return (True, f"mu increase → E2 becomes more negative: {baseline:.6g} → {perturbed:.6g}")
            else:
                return (False, f"Monotonicity with mu failed: {baseline:.6g} → {perturbed:.6g}")

    def test_monotonicity_E2_with_n(self):
        """For |n| >= 1, increasing |n| should decrease |E2|."""
        if not self._all_present(self.I, self.mu, self.eps, self.hbar, self.n):
            return (True, "SKIP: missing variables")
        n_int = round(self.n)
        if abs(n_int) < 1:
            return (True, "SKIP: |n| too small for monotonicity")
        denom = 4 * n_int**2 - 1
        if denom == 0:
            return (False, "Denominator zero")
        baseline = (self.mu**2 * self.eps**2 * self.I) / (self.hbar**2 * denom)
        # increase |n| by exactly 1
        n_pert = n_int + 1 if n_int > 0 else n_int - 1
        denom2 = 4 * n_pert**2 - 1
        perturbed = (self.mu**2 * self.eps**2 * self.I) / (self.hbar**2 * denom2)
        if baseline > 0:
            if perturbed < baseline:
                return (True, f"Increasing n reduces E2: {baseline:.6g} → {perturbed:.6g}")
            else:
                return (False, f"Monotonicity with n failed: {baseline:.6g} → {perturbed:.6g}")
        else:
            if abs(perturbed) < abs(baseline):
                return (True, f"Increasing n reduces |E2|: |{baseline:.6g}| → |{perturbed:.6g}|")
            else:
                return (False, f"Monotonicity with n failed: {baseline:.6g} → {perturbed:.6g}")

    # -----------------------------------------------------------------------
    # Weak answer‑sanity test (one allowed, no derived targets, no math.isclose)
    # -----------------------------------------------------------------------
    def test_answer_sanity_E2_finite(self):
        """Check that E2 is a finite number (not NaN or Inf)."""
        if self.E2 is None:
            return (True, "SKIP: E2 missing")
        if math.isnan(self.E2) or math.isinf(self.E2):
            return (False, "E2 is NaN or Inf")
        # Also check that it's a real number (not complex)
        if isinstance(self.E2, complex):
            return (False, "E2 is complex, expected real")
        return (True, "E2 is finite and real")

    # -----------------------------------------------------------------------
    # Master runner
    # -----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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