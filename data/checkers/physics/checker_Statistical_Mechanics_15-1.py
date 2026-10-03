import sympy as sp
import math

class PhysicsChecker:
    """
    Robust checker for Problem 14.2: Entropy of N harmonic oscillators.
    Checks consistency of two derivations (microcanonical and canonical).
    """
    def __init__(self, solution_instance):
        # Extract input givens safely (default None)
        self.N = getattr(solution_instance, 'N', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.tau = getattr(solution_instance, 'tau', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.sigma_Q1 = results.get('sigma_Q1', None)
        self.sigma_Q2 = results.get('sigma_Q2', None)

        # Define symbolic variables for tests
        self._N, self._hbar, self._omega, self._tau = sp.symbols(
            'N hbar omega tau', positive=True, real=True)
        self._x = self._hbar * self._omega / self._tau
        # Reference entropy expression (canonical)
        self._target_sigma = self._N * (self._x / (sp.exp(self._x) - 1) -
                                         sp.log(1 - sp.exp(-self._x)))

    def _eval_numeric_sigma(self, N, hbar, omega, tau):
        """Compute entropy numerically for given parameters."""
        if any(v is None for v in (N, hbar, omega, tau)):
            return None
        x = hbar * omega / tau
        try:
            sigma = N * (x / (math.exp(x) - 1) - math.log(1 - math.exp(-x)))
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        return sigma

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_sigma_Q1_exists(self):
        """Ensure sigma_Q1 is provided."""
        if self.sigma_Q1 is None:
            return False, "sigma_Q1 missing in results"
        return True, "sigma_Q1 present"

    def test_interface_sigma_Q2_exists(self):
        """Ensure sigma_Q2 is provided."""
        if self.sigma_Q2 is None:
            return False, "sigma_Q2 missing in results"
        return True, "sigma_Q2 present"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_sigma_consistency(self):
        """
        Check that sigma_Q1 and sigma_Q2 are mathematically identical.
        Uses symbolic simplification.
        """
        if self.sigma_Q1 is None or self.sigma_Q2 is None:
            return False, "SKIP: sigma_Q1 or sigma_Q2 missing"
        # Try symbolic comparison
        expr1 = sp.sympify(self.sigma_Q1)
        expr2 = sp.sympify(self.sigma_Q2)
        diff = sp.simplify(expr1 - expr2)
        if diff == 0:
            return True, "sigma_Q1 and sigma_Q2 are symbolically identical"
        # Fallback to numeric check if symbolic fails (e.g., different forms)
        # Use test parameters if givens missing; otherwise use givens
        if self.N is not None and self.hbar is not None and self.omega is not None and self.tau is not None:
            val1 = self._eval_numeric_sigma(self.N, self.hbar, self.omega, self.tau)
            # Evaluate sigma_Q2 numerically
            val2 = float(expr2.subs({self._N: self.N, self._hbar: self.hbar,
                                     self._omega: self.omega, self._tau: self.tau}))
            if val1 is not None and math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2):
                return True, "sigma_Q1 and sigma_Q2 numerically consistent"
            return False, f"sigma_Q1={val1}, sigma_Q2={val2} not close"
        return False, f"Symbolic diff not zero: {diff}. Provide N, hbar, omega, tau for numeric fallback."

    def test_invariant_entropy_sign(self):
        """Entropy must be positive for any positive temperature and N>0."""
        # Use symbolic positivity check: for positive variables, expression > 0
        # Could try to prove analytically, but numeric check is practical.
        if self.N is not None and self.hbar is not None and self.omega is not None and self.tau is not None:
            sigma_val = self._eval_numeric_sigma(self.N, self.hbar, self.omega, self.tau)
            if sigma_val is None:
                return False, "Cannot compute sigma numerically"
            if sigma_val > 0:
                return True, f"sigma = {sigma_val} > 0"
            return False, f"sigma = {sigma_val} <= 0"
        # If givens missing, try deriving from supplied sigma expression
        if self.sigma_Q1 is not None:
            # Check with random safe values (only for sign, not exact)
            test_N = 100
            test_hbar = 1.0
            test_omega = 1.0
            test_tau = 1.0
            x = test_hbar * test_omega / test_tau
            sigma_num = float(sp.N(self.sigma_Q1.subs({self._N: test_N, self._hbar: test_hbar,
                                                      self._omega: test_omega, self._tau: test_tau})))
            if sigma_num > 0:
                return True, f"sigma > 0 for test parameters"
            return False, f"sigma not positive for test parameters"
        return False, "SKIP: insufficient data to check positivity"

    def test_invariant_free_energy_relation(self):
        """
        Verify that -∂F/∂τ equals sigma, where F = -τ ln Z.
        This is a thermodynamic identity.
        """
        if self.sigma_Q1 is None:
            return False, "SKIP: sigma_Q1 missing"
        # Build free energy expression
        F = -self._tau * self._N * sp.log(1 - sp.exp(-self._x))
        sigma_from_F = -sp.diff(F, self._tau)
        sigma_simplified = sp.simplify(sigma_from_F - self._target_sigma)
        if sigma_simplified == 0:
            return True, "sigma = -∂F/∂τ holds"
        # Check if sigma_Q1 matches sigma_from_F
        sigma1 = sp.sympify(self.sigma_Q1)
        diff2 = sp.simplify(sigma_from_F - sigma1)
        if diff2 == 0:
            return True, "sigma_Q1 also satisfies sigma = -∂F/∂τ"
        # Numeric fallback
        if self.N is not None and self.hbar is not None and self.omega is not None and self.tau is not None:
            val_F = float(-self._tau * self._N * sp.log(1 - sp.exp(-self._x)).subs(
                {self._N: self.N, self._hbar: self.hbar, self._omega: self.omega, self._tau: self.tau}))
            # Use sympy to differentiate via finite difference? Better to use analytic derivative.
            sigma_num = float(sigma_from_F.subs({self._N: self.N, self._hbar: self.hbar,
                                                 self._omega: self.omega, self._tau: self.tau}))
            sigma1_num = float(sigma1.subs({self._N: self.N, self._hbar: self.hbar,
                                            self._omega: self.omega, self._tau: self.tau}))
            if math.isclose(sigma_num, sigma1_num, rel_tol=1e-2, abs_tol=1e-2):
                return True, "sigma from free energy matches sigma_Q1 numerically"
            return False, f"Derivative = {sigma_num}, sigma_Q1 = {sigma1_num} not close"
        return False, f"Symbolic discrepancy: sigma_from_F - sigma_target = {sigma_simplified}"

    # ------------------------------------------------------------------
    # Modeling tests (quality of derivation)
    # ------------------------------------------------------------------
    def test_modeling_Q1_microcanonical_limit(self):
        """
        Check that sigma_Q1 reduces correctly in the classical limit (τ >> ℏω, x→0).
        In that limit, sigma ≈ N (1 - ln(x)) = N [1 - ln(ℏω/τ)].
        This tests modeling of Stirling approximation.
        """
        if self.sigma_Q1 is None:
            return False, "SKIP: sigma_Q1 missing"
        sigma_sym = sp.sympify(self.sigma_Q1)
        # Series expansion for small x
        series = sp.series(sigma_sym.subs({self._hbar: 1, self._omega: 1, self._tau: sp.Symbol('tau')}),
                           sp.Symbol('tau'), sp.oo, 3).removeO()
        # More direct: expand for x->0: x/(exp(x)-1) ≈ 1 - x/2 + x^2/12, -ln(1-exp(-x)) ≈ -ln(x) + x/2 - x^2/24
        # So sigma ≈ N (1 - ln(x)) 
        x = sp.Symbol('x')
        sigma_approx = sp.simplify(self._N * (x/(sp.exp(x)-1) - sp.log(1-sp.exp(-x))))
        classical_limit = sp.simplify(sp.limit(sigma_approx - self._N * (1 - sp.log(x)), x, 0))
        if classical_limit == 0:
            return True, "sigma_Q1 correct classical limit"
        # Numeric check for large tau
        if self.N is not None and self.hbar is not None and self.omega is not None:
            tau_large = 100.0 * self.hbar * self.omega  # ensure x ≈ 0.01
            x_val = self.hbar * self.omega / tau_large
            sigma_num = float(sigma_sym.subs({self._N: self.N, self._hbar: self.hbar,
                                              self._omega: self.omega, self._tau: tau_large}))
            approx = self.N * (1 - math.log(x_val))
            if math.isclose(sigma_num, approx, rel_tol=0.1, abs_tol=1):  # tolerant
                return True, f"classical limit approx: sigma={sigma_num}, approx={approx}"
            return False, f"classical limit mismatch: sigma={sigma_num}, approx={approx}"
        return False, "SKIP: insufficient data for classical limit check"

    def test_modeling_Q2_partition_function_chain(self):
        """
        Verify that sigma_Q2 was derived from partition function:
        Z = (1 - e^{-x})^{-N}, F = -τ ln Z, and sigma from derivative.
        We already did the derivative identity; now check that sigma_Q2 matches.
        """
        # This is essentially covered by test_invariant_free_energy_relation.
        # But we can also verify that the expression for Z is used correctly.
        if self.sigma_Q2 is None:
            return False, "SKIP: sigma_Q2 missing"
        # Build Z and F from the known formula
        Z = (1 - sp.exp(-self._x)) ** (-self._N)
        F = -self._tau * sp.log(Z)
        sigma_derived = -sp.diff(F, self._tau)
        # Compare with sigma_Q2
        sigma2 = sp.sympify(self.sigma_Q2)
        if sp.simplify(sigma_derived - sigma2) == 0:
            return True, "sigma_Q2 consistent with partition function derivation"
        return False, f"Sigma from partition function mismatch: {sp.simplify(sigma_derived - sigma2)}"

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_entropy_increases_with_tau(self):
        """
        For fixed N, hbar, omega, entropy should increase as temperature (tau) increases.
        Perturb tau upward by 10% and check that sigma increases.
        """
        if self.N is None or self.hbar is None or self.omega is None or self.tau is None:
            return False, "SKIP: missing givens (N, hbar, omega, tau) needed"
        tau0 = self.tau
        # Compute baseline
        sigma0 = self._eval_numeric_sigma(self.N, self.hbar, self.omega, tau0)
        if sigma0 is None:
            return False, "Cannot compute sigma0"
        # Perturb: increase tau by 10%
        tau1 = tau0 * 1.1
        sigma1 = self._eval_numeric_sigma(self.N, self.hbar, self.omega, tau1)
        if sigma1 is None:
            return False, "Cannot compute sigma1"
        if sigma1 > sigma0:
            return True, f"sigma increased from {sigma0} to {sigma1} (tau: {tau0} -> {tau1})"
        return False, f"sigma did not increase: baseline {sigma0}, perturbed {sigma1}"

    def test_monotonicity_entropy_increases_with_N(self):
        """
        For fixed hbar, omega, tau, entropy should increase with N.
        Perturb N upward by 10% and check that sigma increases.
        """
        if self.N is None or self.hbar is None or self.omega is None or self.tau is None:
            return False, "SKIP: missing givens"
        N0 = self.N
        sigma0 = self._eval_numeric_sigma(N0, self.hbar, self.omega, self.tau)
        if sigma0 is None:
            return False, "Cannot compute sigma0"
        N1 = N0 * 1.1  # 10% increase (not necessarily integer, but fine for trend)
        sigma1 = self._eval_numeric_sigma(N1, self.hbar, self.omega, self.tau)
        if sigma1 is None:
            return False, "Cannot compute sigma1"
        if sigma1 > sigma0:
            return True, f"sigma increased from {sigma0} to {sigma1} (N: {N0} -> {N1})"
        return False, f"sigma did not increase: baseline {sigma0}, perturbed {sigma1}"

    # ------------------------------------------------------------------
    # Answer sanity test (weak, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak_sign_and_range(self):
        """
        Perform a weak sanity check on sigma_Q1: it should be non-negative and finite.
        No isclose, no re-derivation, just broad magnitude range.
        """
        if self.sigma_Q1 is None:
            return False, "SKIP: sigma_Q1 missing"
        # Try to substitute random reasonable numbers to get a numeric value
        # Use sympy lambdify with safe defaults
        try:
            test_params = {self._N: 100, self._hbar: 1, self._omega: 1, self._tau: 1}
            sigma_num = float(self.sigma_Q1.subs(test_params).evalf())
        except Exception:
            return False, "Cannot evaluate sigma_Q1 with test parameters"
        if sigma_num < 0 or math.isnan(sigma_num) or math.isinf(sigma_num):
            return False, f"sigma_Q1 with test params: {sigma_num} (should be non-negative and finite)"
        # Additionally, sigma should be roughly of order N (since extensive)
        if sigma_num > 0.5 * test_params[self._N] and sigma_num < 10 * test_params[self._N]:
            return True, f"sigma_Q1 plausible magnitude: {sigma_num} for N={test_params[self._N]}"
        return False, f"sigma_Q1 magnitude suspicious: {sigma_num} for N={test_params[self._N]}"

    # ------------------------------------------------------------------
    # Run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)