import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for a spin-1/2 system coupled to a phonon bath.
    Extracts all relevant quantities from the solution instance and
    provides a battery of checks: interface, invariants, modeling, monotonicity,
    and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Givens (expect them in solution_instance.givens or direct attributes)
        self.Ns = getattr(solution_instance, 'Ns', None) or solution_instance.givens.get('Ns', None)
        self.mu = getattr(solution_instance, 'mu', None) or solution_instance.givens.get('mu', None)
        self.B = getattr(solution_instance, 'B', None) or solution_instance.givens.get('B', None)
        self.tau = getattr(solution_instance, 'tau', None) or solution_instance.givens.get('tau', None)
        self.Np = getattr(solution_instance, 'Np', None) or solution_instance.givens.get('Np', None)
        self.tau_D = getattr(solution_instance, 'tau_D', None) or solution_instance.givens.get('tau_D', None)
        self.tau_i = getattr(solution_instance, 'tau_i', None)  # may be same as tau

        # Results (expect them in solution_instance.results dictionary)
        results = getattr(solution_instance, 'results', {})
        self.Z1 = results.get('Z1', None)
        self.P_plus = results.get('P_plus', None)
        self.P_minus = results.get('P_minus', None)
        self.M = results.get('M', None)
        self.chi = results.get('chi', None)
        self.sigma_s = results.get('sigma_s', None)
        self.tau_f = results.get('tau_f', None)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present and not None."""
        required = ['Z1', 'P_plus', 'P_minus', 'M', 'chi', 'sigma_s', 'tau_f']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens(self):
        """Check that all necessary givens are provided."""
        givens = {'Ns': self.Ns, 'mu': self.mu, 'B': self.B, 'tau': self.tau,
                  'Np': self.Np, 'tau_D': self.tau_D}
        missing = [k for k, v in givens.items() if v is None]
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All necessary givens provided")

    def test_interface_types(self):
        """Check that numeric results are numeric."""
        for attr in ['Z1', 'P_plus', 'P_minus', 'M', 'chi', 'sigma_s', 'tau_f']:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"{attr} is not numeric (type {type(val).__name__})")
        return (True, "All numeric types OK")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem‑based)
    # ----------------------------------------------------------------------
    def test_invariant_partition_function_form(self):
        """Z1 should equal 2*cosh(mu*B/tau) up to tolerance."""
        if self.Z1 is None or self.mu is None or self.B is None or self.tau is None:
            return (False, "SKIP: missing data")
        expected = 2.0 * math.cosh(self.mu * self.B / self.tau)
        if math.isclose(self.Z1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Z1 consistent with 2*cosh(mu*B/tau)")
        else:
            return (False, f"Z1 = {self.Z1}, expected {expected}")

    def test_invariant_probability_sum(self):
        """P_plus + P_minus should be 1."""
        if self.P_plus is None or self.P_minus is None:
            return (False, "SKIP: missing P_plus or P_minus")
        total = self.P_plus + self.P_minus
        if math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Probabilities sum to 1")
        else:
            return (False, f"P_plus + P_minus = {total} != 1")

    def test_invariant_magnetization_from_probabilities(self):
        """M should equal Ns*mu*(P_plus - P_minus)."""
        if any(x is None for x in [self.Ns, self.mu, self.P_plus, self.P_minus, self.M]):
            return (False, "SKIP: missing data")
        expected = self.Ns * self.mu * (self.P_plus - self.P_minus)
        if math.isclose(self.M, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "M derived from probabilities")
        else:
            return (False, f"M = {self.M}, expected from probabilities = {expected}")

    def test_invariant_entropy_conservation(self):
        """Check total entropy conservation (spin+phonon) between initial and final state."""
        if any(x is None for x in [self.Ns, self.mu, self.B, self.tau, self.Np, self.tau_D, self.tau_f,
                                   self.sigma_s, self.M, self.chi]):
            return (False, "SKIP: missing data for entropy conservation")
        # Initial sigma_s from given (already high-T expression)
        # Phonon initial: sigma_p_i = (4/5)*pi^4 * Np * (tau/tau_D)^3
        sigma_p_i = (4.0 / 5.0) * math.pi**4 * self.Np * (self.tau / self.tau_D)**3
        total_i = self.sigma_s + sigma_p_i

        # Final spin entropy (B=0): sigma_s_f = Ns*ln2
        sigma_s_f = self.Ns * math.log(2)
        # Final phonon entropy at tau_f
        sigma_p_f = (4.0 / 5.0) * math.pi**4 * self.Np * (self.tau_f / self.tau_D)**3
        total_f = sigma_s_f + sigma_p_f

        if math.isclose(total_i, total_f, rel_tol=1e-1, abs_tol=1e-1):
            return (True, f"Total entropy conserved: initial {total_i:.3e} final {total_f:.3e}")
        else:
            return (False, f"Entropy not conserved: initial {total_i:.3e} final {total_f:.3e}")

    def test_invariant_high_t_susceptibility(self):
        """In high-T limit, chi should be Ns*mu^2/tau (from M/B)."""
        if any(x is None for x in [self.chi, self.Ns, self.mu, self.tau]):
            return (False, "SKIP: missing data")
        expected_chi = self.Ns * self.mu**2 / self.tau
        if math.isclose(self.chi, expected_chi, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "chi consistent with Curie law")
        else:
            return (False, f"chi = {self.chi}, expected high-T {expected_chi}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_magnetization_tanh(self):
        """M should equal Ns*mu*tanh(mu*B/tau)."""
        if any(x is None for x in [self.M, self.Ns, self.mu, self.B, self.tau]):
            return (False, "SKIP: missing data")
        x = self.mu * self.B / self.tau
        expected = self.Ns * self.mu * math.tanh(x)
        if math.isclose(self.M, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "M = Ns*mu*tanh(mu*B/tau)")
        else:
            return (False, f"M = {self.M}, expected tanh {expected}")

    def test_modeling_spin_entropy_highT(self):
        """sigma_s should be Ns*(ln2 - 0.5*(mu*B/tau)^2)."""
        if any(x is None for x in [self.sigma_s, self.Ns, self.mu, self.B, self.tau]):
            return (False, "SKIP: missing data")
        x = self.mu * self.B / self.tau
        expected = self.Ns * (math.log(2) - 0.5 * x**2)
        if math.isclose(self.sigma_s, expected, rel_tol=1e-1, abs_tol=1e-1):
            return (True, "sigma_s = Ns*(ln2 - 0.5*(muB/tau)^2)")
        else:
            return (False, f"sigma_s = {self.sigma_s}, expected {expected}")

    def test_modeling_tau_f_relation(self):
        """Check that tau_f satisfies the derived cubic relation."""
        if any(x is None for x in [self.tau_f, self.Ns, self.Np, self.mu, self.B, self.tau, self.tau_D]):
            return (False, "SKIP: missing data for tau_f")
        C = (5.0 / (8.0 * math.pi**2)) * (self.Ns / self.Np) * (self.mu * self.B / self.tau)**2 * (self.tau_D / self.tau)**3
        if C >= 1:  # unphysical, but formula expects small C
            return (False, "C >= 1, formula may be invalid")
        expected_tau_f = self.tau * (1 - C)**(1.0/3.0)
        if math.isclose(self.tau_f, expected_tau_f, rel_tol=1e-1, abs_tol=1e-1):
            return (True, "tau_f consistent with adiabatic cooling formula")
        else:
            return (False, f"tau_f = {self.tau_f}, expected {expected_tau_f}")

    # ----------------------------------------------------------------------
    # Monotonicity tests (each perturbs an input and compares perturbed to baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_M_increases_with_B(self):
        """If we perturb B slightly higher, M should increase (hold other fixed)."""
        if any(x is None for x in [self.M, self.Ns, self.mu, self.B, self.tau]):
            return (False, "SKIP: missing data")
        # Perturb B by +5%
        B_perturbed = self.B * 1.05
        x_pert = self.mu * B_perturbed / self.tau
        M_pert = self.Ns * self.mu * math.tanh(x_pert)
        if M_pert > self.M:
            return (True, "M increases with B (positive perturbation)")
        else:
            return (False, f"M = {self.M}, perturbed M = {M_pert} (should increase)")

    def test_monotonicity_chi_decreases_with_tau(self):
        """If we perturb tau slightly higher, chi should decrease (positive perturbation)."""
        if any(x is None for x in [self.chi, self.Ns, self.mu, self.tau]):
            return (False, "SKIP: missing data")
        # Perturb tau by +5%
        tau_pert = self.tau * 1.05
        chi_pert = self.Ns * self.mu**2 / tau_pert
        if chi_pert < self.chi:
            return (True, "chi decreases with increasing tau")
        else:
            return (False, f"chi = {self.chi}, perturbed chi = {chi_pert} (should decrease)")

    def test_monotonicity_sigma_s_decreases_with_B(self):
        """At fixed tau, sigma_s should decrease with increasing B."""
        if any(x is None for x in [self.sigma_s, self.Ns, self.mu, self.B, self.tau]):
            return (False, "SKIP: missing data")
        B_perturbed = self.B * 1.1
        x_pert = self.mu * B_perturbed / self.tau
        sigma_pert = self.Ns * (math.log(2) - 0.5 * x_pert**2)
        if sigma_pert < self.sigma_s:
            return (True, "sigma_s decreases with increasing B")
        else:
            return (False, f"sigma_s = {self.sigma_s}, perturbed = {sigma_pert} (should decrease)")

    def test_monotonicity_tau_f_decreases_with_B(self):
        """For larger initial B, final temperature should be lower."""
        if any(x is None for x in [self.tau_f, self.Ns, self.Np, self.mu, self.B, self.tau, self.tau_D]):
            return (False, "SKIP: missing data")
        B_perturbed = self.B * 1.2
        C = (5.0 / (8.0 * math.pi**2)) * (self.Ns / self.Np) * (self.mu * B_perturbed / self.tau)**2 * (self.tau_D / self.tau)**3
        tau_f_pert = self.tau * (1 - C)**(1.0/3.0)
        if tau_f_pert < self.tau_f:
            return (True, "Larger B gives lower tau_f")
        else:
            return (False, f"tau_f = {self.tau_f}, perturbed B gives {tau_f_pert} (should be lower)")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one allowed, no math.isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Sanity checks: all answers finite, signs reasonable, positive where expected."""
        if self.Z1 is None or self.P_plus is None or self.M is None or self.chi is None:
            return (False, "SKIP: missing results for sanity")
        issues = []
        if not math.isfinite(self.Z1):
            issues.append("Z1 not finite")
        if not (0 <= self.P_plus <= 1):
            issues.append("P_plus out of [0,1]")
        if not (0 <= self.P_minus <= 1):
            issues.append("P_minus out of [0,1]")
        if not math.isfinite(self.M):
            issues.append("M not finite")
        if self.chi <= 0:
            issues.append("chi not positive")
        if not math.isfinite(self.sigma_s):
            issues.append("sigma_s not finite")
        if self.tau_f is not None:
            if not math.isfinite(self.tau_f):
                issues.append("tau_f not finite")
            if self.tau_f <= 0 or self.tau_f > self.tau:
                issues.append("tau_f out of physical range (0 < tau_f <= tau)")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Basic sanity checks passed")

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
        return (all_pass, logs)