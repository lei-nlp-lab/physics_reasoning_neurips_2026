import math
import sys

class PhysicsChecker:
    """
    Audits the solution for the quantum oscillator entropy problem.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract results and givens
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Extracting emitted results (all optional, default to None)
        self.sigma_n = self.results.get('sigma_n', None)
        self.sigma_U = self.results.get('sigma_U', None)
        self.U = self.results.get('U', None)

        # Extracting input parameters (from givens, results, or direct attribute)
        self.N = (self.givens.get('N', None) or
                  self.results.get('N', None) or
                  getattr(solution_instance, 'N', None))

        self.n = (self.givens.get('n', None) or
                  self.results.get('n', None) or
                  getattr(solution_instance, 'n', None))

        self.omega = (self.givens.get('omega', None) or
                      self.results.get('omega', None) or
                      getattr(solution_instance, 'omega', None))

        self.hbar = (self.givens.get('hbar', None) or
                     self.results.get('hbar', None) or
                     getattr(solution_instance, 'hbar', None))

        self.tau = (self.givens.get('tau', None) or
                    self.results.get('tau', None) or
                    getattr(solution_instance, 'tau', None))

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_sigma_n_available(self):
        if self.sigma_n is None:
            return False, "sigma_n is missing from results"
        return True, "sigma_n is present"

    def test_interface_sigma_U_available(self):
        if self.sigma_U is None:
            return False, "sigma_U is missing from results"
        return True, "sigma_U is present"

    def test_interface_U_available(self):
        if self.U is None:
            return False, "U is missing from results"
        return True, "U is present"

    def test_interface_required_givens_available(self):
        missing = []
        if self.N is None:
            missing.append("N")
        if self.n is None:
            missing.append("n")
        if self.omega is None:
            missing.append("omega")
        if self.hbar is None:
            missing.append("hbar")
        if self.tau is None:
            missing.append("tau")
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All required givens (N, n, omega, hbar, tau) are available"

    # ----------------------------------------------------------------------
    # Invariant checks (using isclose)
    # ----------------------------------------------------------------------
    def test_invariant_sigma_n_via_combinatorial_count(self):
        """Check that sigma_n is consistent with the combinatorial multiplicity."""
        if self.sigma_n is None or self.n is None or self.N is None:
            return False, "SKIP: missing sigma_n, n, or N"
        try:
            # Use lgamma to avoid overflow: ln(Ω) with Ω = (n+N choose n)
            ln_omega = (math.lgamma(self.n + self.N + 1) -
                        math.lgamma(self.n + 1) -
                        math.lgamma(self.N + 1))
            if not math.isfinite(ln_omega):
                return False, "ln(Ω) computed as non‑finite (overflow?)"
            # The derived sigma_n should match ln(omega) within 1% tolerance
            if math.isclose(self.sigma_n, ln_omega, rel_tol=1e-2):
                return True, f"sigma_n matches ln(Ω) within 1%: {self.sigma_n} vs {ln_omega}"
            else:
                return False, (f"sigma_n ({self.sigma_n}) deviates from ln(Ω) "
                               f"({ln_omega}) by more than 1%")
        except Exception as e:
            return False, f"Computation error in ln(Ω): {e}"

    def test_invariant_sigma_U_substitution_consistency(self):
        """Check that sigma_U follows from sigma_n by n = U/(ħω)."""
        if (self.sigma_U is None or self.sigma_n is None or
            self.U is None or self.N is None or
            self.hbar is None or self.omega is None):
            return False, "SKIP: missing required quantities"
        n_from_U = self.U / (self.hbar * self.omega)
        # Compute sigma_n from the analytic expression (Stirling formula)
        ratio = n_from_U / self.N
        if ratio <= 0:
            return False, "n/N must be positive for the logarithm"
        sigma_n_from_U = self.N * ((1 + ratio) * math.log(1 + ratio) -
                                   ratio * math.log(ratio))
        if math.isclose(self.sigma_U, sigma_n_from_U, rel_tol=1e-2):
            return True, "sigma_U matches sigma_n after substitution (n = U/(ħω))"
        else:
            return False, (f"sigma_U ({self.sigma_U}) differs from sigma_n via substitution "
                           f"({sigma_n_from_U}) by >1%")

    def test_invariant_thermodynamic_identity(self):
        """
        Verify that 1/τ = (∂σ/∂U)_N.
        For Planck distribution, ∂σ/∂U = (1/(ħω)) ln(1 + Nħω/U).
        """
        if (self.sigma_U is None or self.U is None or self.N is None or
            self.hbar is None or self.omega is None or self.tau is None):
            return False, "SKIP: missing required quantities"
        if self.U <= 0:
            return False, "U must be positive for the logarithmic derivative"
        dsigma_dU = (1.0 / (self.hbar * self.omega)) * \
                    math.log(1.0 + self.N * self.hbar * self.omega / self.U)
        inv_tau = 1.0 / self.tau
        if math.isclose(dsigma_dU, inv_tau, rel_tol=1e-2):
            return True, f"1/τ = {inv_tau} matches (∂σ/∂U) = {dsigma_dU}"
        else:
            return False, (f"1/τ = {inv_tau} differs from (∂σ/∂U) = {dsigma_dU} "
                           "by more than 1%")

    # ----------------------------------------------------------------------
    # Modeling quality checks (using isclose)
    # ----------------------------------------------------------------------
    def test_modeling_stirling_approximation_valid(self):
        """
        For the given values of n and N, the Stirling approximation should
        yield a result close to the exact combinatorial logarithm.
        (Quality of the model.)
        """
        return self.test_invariant_sigma_n_via_combinatorial_count()

    def test_modeling_log_arguments_positive(self):
        """
        Check that the arguments of all logarithms in the entropy expressions
        are positive, as required by the physical model (positive entropy).
        """
        # Check sigma_n arguments
        if self.n is not None and self.N is not None and self.N > 0:
            ratio = self.n / self.N
            if ratio <= 0:
                return False, f"n/N = {ratio} is not positive (log argument invalid)"
            if 1 + ratio <= 0:
                return False, f"1 + n/N = {1+ratio} is not positive (log argument invalid)"
        # If sigma_U is present, check its arguments
        if self.U is not None and self.N is not None and self.hbar is not None and self.omega is not None:
            arg1 = 1 + self.U / (self.N * self.hbar * self.omega)
            arg2 = self.U / (self.N * self.hbar * self.omega)
            if arg1 <= 0:
                return False, f"Argument (1+U/(Nħω)) = {arg1} is not positive"
            if arg2 <= 0:
                return False, f"Argument U/(Nħω) = {arg2} is not positive"
        # If tau is present and U is present, check the denominator in Planck formula
        if (self.hbar is not None and self.omega is not None and
            self.tau is not None and self.tau > 0):
            exp_arg = self.hbar * self.omega / self.tau
            if math.isinf(math.exp(exp_arg)) or math.isnan(math.exp(exp_arg)):
                return False, f"exp(ħω/τ) = {math.exp(exp_arg)} is not finite"
        return True, "All logarithm arguments in entropy expressions are positive/valid"

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_sigma_n_increases_with_n(self):
        """
        For fixed N, sigma_n should increase when n increases.
        Re‑derive sigma_n from the Stirling formula.
        """
        if self.n is None or self.N is None or self.hbar is None or self.omega is None:
            return False, "SKIP: missing n, N, hbar, or omega"
        # Baseline sigma_n using the analytic expression
        ratio = self.n / self.N
        if ratio <= 0:
            return False, "n/N must be positive"
        sigma_base = self.N * ((1 + ratio) * math.log(1 + ratio) - ratio * math.log(ratio))

        # Perturb n by +10%
        n_pert = self.n * 1.1
        ratio_pert = n_pert / self.N
        sigma_pert = self.N * ((1 + ratio_pert) * math.log(1 + ratio_pert) -
                               ratio_pert * math.log(ratio_pert))

        if sigma_pert > sigma_base:
            return True, f"sigma_n increases when n rises from {self.n} to {n_pert}: {sigma_pert} > {sigma_base}"
        else:
            return False, f"sigma_n is not monotonic in n: sigma({self.n}) = {sigma_base}, sigma({n_pert}) = {sigma_pert}"

    def test_monotonicity_U_increases_with_tau(self):
        """
        For fixed N, hbar, omega, U should increase when τ increases.
        Use the Planck distribution formula.
        """
        if (self.N is None or self.hbar is None or self.omega is None or
            self.tau is None):
            return False, "SKIP: missing N, hbar, omega, or tau"
        base_U = (self.N * self.hbar * self.omega) / \
                 (math.exp(self.hbar * self.omega / self.tau) - 1.0)
        tau_pert = self.tau * 1.1
        pert_U = (self.N * self.hbar * self.omega) / \
                 (math.exp(self.hbar * self.omega / tau_pert) - 1.0)
        if pert_U > base_U:
            return True, f"U increases with τ: U({self.tau}) = {base_U}, U({tau_pert}) = {pert_U}"
        else:
            return False, f"U does not increase monotonically with τ: base {base_U}, perturbed {pert_U}"

    # ----------------------------------------------------------------------
    # Weak answer sanity (no isclose, no re‑derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_U_positive_finite(self):
        if self.U is None:
            return False, "SKIP: U not available"
        if not math.isfinite(self.U):
            return False, f"U is not finite: {self.U}"
        if self.U <= 0:
            return False, f"U must be positive, got {self.U}"
        return True, f"U is positive and finite: {self.U}"

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
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
        return all_pass, logs