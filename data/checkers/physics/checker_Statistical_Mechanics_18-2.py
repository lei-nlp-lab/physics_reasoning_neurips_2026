import math
import sympy as sym

class PhysicsChecker:
    def __init__(self, solution_instance):
        # ---------- Inputs (givens) ----------
        def safe_get(key, fallback=None):
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                return solution_instance.results.get(key, fallback)
            return getattr(solution_instance, key, fallback)

        self.N = safe_get('N')
        self.V = safe_get('V')
        self.tau = safe_get('tau')
        self.epsilon = safe_get('epsilon')
        self.h = safe_get('h')
        self.m = safe_get('m')

        # ---------- Results (outputs) ----------
        self.C_int = safe_get('C_int')
        self.C_trans = safe_get('C_trans')
        self.mu = safe_get('mu')
        self.F = safe_get('F')
        self.S = safe_get('S')
        self.p = safe_get('p')
        self.C_p = safe_get('C_p')

        # Derived quantities for internal checks
        self._n = None
        self._n_Q = None
        self._Z_int = None
        self._lambda = None
        self._compute_derived()

    # ---------- Helper to compute derived quantities from inputs ----------
    def _compute_derived(self):
        if None in (self.tau, self.epsilon, self.h, self.m, self.N, self.V):
            return
        self._lambda = self.h / math.sqrt(2 * math.pi * self.m * self.tau)
        self._n_Q = 1.0 / (self._lambda ** 3)
        self._n = self.N / self.V
        self._Z_int = 1.0 + math.exp(-self.epsilon / self.tau)

    # ==================== INTERFACE TESTS ====================
    def test_interface_required_keys_present(self):
        """Check that all required result keys exist and are not None."""
        keys = ['C_int', 'C_trans', 'mu', 'F', 'S', 'p', 'C_p']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_inputs_present(self):
        """Check that all givens are present and numeric."""
        required = ['N', 'V', 'tau', 'epsilon', 'h', 'm']
        missing = [r for r in required if getattr(self, r) is None]
        if missing:
            return (False, f"Missing inputs: {missing}")
        for r in required:
            val = getattr(self, r)
            if not isinstance(val, (int, float)):
                return (False, f"Input {r} is not numeric (type: {type(val).__name__})")
        return (True, "All inputs present and numeric")

    # ==================== INVARIANT / PHYSICS CHECKS ====================
    def test_invariant_ideal_gas_law(self):
        """Check pV = Nτ (ideal gas law)."""
        if None in (self.p, self.V, self.N, self.tau):
            return (False, "SKIP: missing p, V, N, or tau")
        lhs = self.p * self.V
        rhs = self.N * self.tau
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"pV = {lhs:.5e}, Nτ = {rhs:.5e} — match")
        else:
            return (False, f"pV = {lhs:.5e}, Nτ = {rhs:.5e} — mismatch")

    def test_invariant_Cp_minus_CV(self):
        """Check C_p - (C_trans + C_int) ≈ N."""
        if None in (self.C_p, self.C_trans, self.C_int, self.N):
            return (False, "SKIP: missing C_p, C_trans, C_int, or N")
        delta = self.C_p - (self.C_trans + self.C_int)
        if math.isclose(delta, self.N, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_p - (C_trans+C_int) = {delta:.5e}, N = {self.N:.5e} — match")
        else:
            return (False, f"C_p - (C_trans+C_int) = {delta:.5e}, N = {self.N:.5e} — mismatch")

    def test_invariant_chemical_potential_expression(self):
        """Check mu = -τ ln(V/(N λ^3)) - τ ln(1+e^{-ε/τ})."""
        if None in (self.mu, self.N, self.V, self.tau, self.epsilon, self.h, self.m):
            return (False, "SKIP: missing mu or inputs")
        if self._Z_int is None:
            return (False, "SKIP: derived quantities not computed")
        expected_mu = -self.tau * math.log(self.V / (self.N * (self._lambda**3))) - self.tau * math.log(self._Z_int)
        if math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"mu = {self.mu:.5e}, expected = {expected_mu:.5e} — match")
        else:
            return (False, f"mu = {self.mu:.5e}, expected = {expected_mu:.5e} — mismatch")

    # ==================== MODELING / TRACE CHECKS ====================
    def test_modeling_free_energy_expression(self):
        """Check F = -Nτ[ln(V/(N λ^3)) + 1 + ln(1+e^{-ε/τ})]."""
        if None in (self.F, self.N, self.V, self.tau, self.epsilon, self.h, self.m):
            return (False, "SKIP: missing F or inputs")
        if self._Z_int is None:
            return (False, "SKIP: derived quantities not computed")
        expected_F = -self.N * self.tau * (
            math.log(self.V / (self.N * (self._lambda ** 3))) + 1 + math.log(self._Z_int)
        )
        if math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F = {self.F:.5e}, expected = {expected_F:.5e} — match")
        else:
            return (False, f"F = {self.F:.5e}, expected = {expected_F:.5e} — mismatch")

    def test_modeling_entropy_expression(self):
        """Check S = N[ln(V/(Nλ^3)) + 5/2 + ln(1+e^{-ε/τ}) + (ε/τ)/(e^{ε/τ}+1)]."""
        if None in (self.S, self.N, self.V, self.tau, self.epsilon, self.h, self.m):
            return (False, "SKIP: missing S or inputs")
        if self._Z_int is None:
            return (False, "SKIP: derived quantities not computed")
        exp_term = math.exp(-self.epsilon / self.tau)
        numerator_eps_tau = self.epsilon / self.tau
        fraction = numerator_eps_tau * exp_term / (1 + exp_term)
        expected_S = self.N * (
            math.log(self.V / (self.N * (self._lambda ** 3)))
            + 5/2
            + math.log(self._Z_int)
            + fraction
        )
        if math.isclose(self.S, expected_S, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S = {self.S:.5e}, expected = {expected_S:.5e} — match")
        else:
            return (False, f"S = {self.S:.5e}, expected = {expected_S:.5e} — mismatch")

    def test_modeling_C_trans(self):
        """Check C_trans ≈ (3/2)N."""
        if None in (self.C_trans, self.N):
            return (False, "SKIP: missing C_trans or N")
        expected = 1.5 * self.N
        if math.isclose(self.C_trans, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_trans = {self.C_trans:.5e}, expected = {expected:.5e} — match")
        else:
            return (False, f"C_trans = {self.C_trans:.5e}, expected = {expected:.5e} — mismatch")

    def test_modeling_C_int_expression(self):
        """Check C_int = N (ε/τ)^2 e^{ε/τ} / (e^{ε/τ}+1)^2."""
        if None in (self.C_int, self.N, self.tau, self.epsilon):
            return (False, "SKIP: missing C_int or inputs")
        x = self.epsilon / self.tau
        exp_minus_x = math.exp(-x)
        numerator_exp = x**2 * exp_minus_x
        denominator = (1 + exp_minus_x)**2
        expected = self.N * numerator_exp / denominator
        if math.isclose(self.C_int, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_int = {self.C_int:.5e}, expected = {expected:.5e} — match")
        else:
            return (False, f"C_int = {self.C_int:.5e}, expected = {expected:.5e} — mismatch")

    # ==================== MONOTONICITY / TREND CHECKS (explicit perturbation) ====================
    def test_monotonicity_pressure_with_N(self):
        """Perturb N by +10% and check p scales proportionally: baseline p vs. p_pert = N_pert * τ / V."""
        if None in (self.p, self.N, self.V, self.tau):
            return (False, "SKIP: missing p, N, V, or tau")
        # baseline output: p (as given)
        baseline_p = self.p
        # perturb input: increase N by 10%
        N_pert = 1.1 * self.N
        # compute expected perturbed output using ideal gas law
        p_perturbed = N_pert * self.tau / self.V
        # Compare perturbed output to baseline (ratio should be 1.1)
        ratio = p_perturbed / baseline_p
        if math.isclose(ratio, 1.1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Baseline p = {baseline_p:.5e}, perturbed p (N→1.1N) = {p_perturbed:.5e}, ratio = {ratio:.5f} ≈ 1.1")
        else:
            return (False, f"Baseline p = {baseline_p:.5e}, perturbed p = {p_perturbed:.5e}, ratio = {ratio:.5f} ≠ 1.1")

    def test_monotonicity_C_int_with_temperature(self):
        """Perturb τ upward by +10% and check C_int changes as expected (for typical ε/τ values, C_int increases)."""
        if None in (self.C_int, self.N, self.tau, self.epsilon):
            return (False, "SKIP: missing C_int or inputs")
        # store baseline C_int
        baseline_C_int = self.C_int
        # perturb τ upward by +10%
        tau_pert = 1.1 * self.tau
        # recompute expected C_int at perturbed τ using internal formula
        x_pert = self.epsilon / tau_pert
        exp_minus_x = math.exp(-x_pert)
        numerator = self.N * (x_pert**2) * exp_minus_x
        denominator = (1 + exp_minus_x)**2
        C_int_pert = numerator / denominator
        # check that perturbed C_int is larger (since for typical values C_int increases with τ up to a peak)
        # Use isclose for ratio; for a true monotonicity we expect C_int_pert > baseline_C_int (if we are in increasing region)
        if C_int_pert > baseline_C_int:
            return (True, f"Baseline C_int = {baseline_C_int:.5e}, perturbed (τ→1.1τ) C_int = {C_int_pert:.5e} — increased")
        else:
            return (False, f"Baseline C_int = {baseline_C_int:.5e}, perturbed C_int = {C_int_pert:.5e} — did not increase (monotonicity assumption failed)")

    # ==================== WEAK ANSWER SANITY (ONE METHOD ONLY) ====================
    def test_answer_sanity_sign_and_finite(self):
        """Weak sanity: all results are finite and physically non-negative where required."""
        result_names = ['C_int', 'C_trans', 'mu', 'F', 'S', 'p', 'C_p']
        for name in result_names:
            val = getattr(self, name, None)
            if val is None:
                return (False, f"SKIP: {name} is None, cannot check")
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or Inf: {val}")
        nonneg = ['C_int', 'C_trans', 'C_p', 'S', 'p']
        for name in nonneg:
            val = getattr(self, name)
            if val < 0:
                return (False, f"{name} is negative: {val}")
        return (True, "All results finite, non-negative where required")

    # ==================== RUN ALL TESTS ====================
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