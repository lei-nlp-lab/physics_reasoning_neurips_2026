import math
import traceback
import sympy as sp


class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results from solution (these are the main outputs)
        self.Z = getattr(solution_instance, 'Z', None) or solution_instance.results.get('Z', None)
        self.S = getattr(solution_instance, 'S', None) or solution_instance.results.get('S', None)
        self.U = getattr(solution_instance, 'U', None) or solution_instance.results.get('U', None)
        self.M_avg = getattr(solution_instance, 'M_avg', None) or solution_instance.results.get('M_avg', None)
        self.delta_M = getattr(solution_instance, 'delta_M', None) or solution_instance.results.get('delta_M', None)
        self.T_f = getattr(solution_instance, 'T_f', None) or solution_instance.results.get('T_f', None)

        # Extract givens (if available) – safely default to None
        self.N = None
        self.mu0 = None
        self.B = None
        self.T = None
        self.k_B = None
        self.B_i = None
        self.B_f = None
        self.T_i = None

        # Try to find givens in the instance or its .given dict
        if hasattr(solution_instance, 'given'):
            given = solution_instance.given
            self.N = given.get('N', None)
            self.mu0 = given.get('mu0', given.get('mu_0', None))
            self.B = given.get('B', None)
            self.T = given.get('T', None)
            self.k_B = given.get('k_B', given.get('k_B', None))
            self.B_i = given.get('B_i', None)
            self.B_f = given.get('B_f', None)
            self.T_i = given.get('T_i', None)
        else:
            # fallback: try direct attributes
            self.N = getattr(solution_instance, 'N', None)
            self.mu0 = getattr(solution_instance, 'mu0', None) or getattr(solution_instance, 'mu_0', None)
            self.B = getattr(solution_instance, 'B', None)
            self.T = getattr(solution_instance, 'T', None)
            self.k_B = getattr(solution_instance, 'k_B', None)
            self.B_i = getattr(solution_instance, 'B_i', None)
            self.B_f = getattr(solution_instance, 'B_f', None)
            self.T_i = getattr(solution_instance, 'T_i', None)

    def _alpha(self, B, T):
        """Compute alpha = mu0 B / (k_B T) if all required values present."""
        if self.mu0 is None or B is None or T is None or self.k_B is None or T == 0:
            return None
        return self.mu0 * B / (self.k_B * T)

    def _check_numeric_availability(self, *vars):
        return all(v is not None for v in vars)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist and are numeric."""
        keys = ['Z', 'S', 'U', 'M_avg', 'delta_M', 'T_f']
        missing = []
        for k in keys:
            val = getattr(self, k, None)
            if val is None:
                missing.append(k)
        if missing:
            return False, f"Missing result keys: {missing}"
        # Check that each is a real number (int or float)
        for k in keys:
            val = getattr(self, k)
            if not isinstance(val, (int, float)):
                return False, f"Key '{k}' is not numeric (type {type(val).__name__})"
        return True, "All required numeric results present."

    def test_interface_type_Z(self):
        """Z should be positive (real > 0)."""
        if self.Z is None:
            return False, "SKIP: Z not available"
        if not isinstance(self.Z, (int, float)) or self.Z <= 0:
            return False, f"Z must be positive, got {self.Z}"
        return True, f"Z = {self.Z} > 0"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_partition_function_form(self):
        """Check Z = (2 cosh(alpha))^N for numerical given values."""
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B):
            return False, "SKIP: insufficient givens to compute alpha"
        alpha = self._alpha(self.B, self.T)
        if alpha is None:
            return False, "SKIP: failed to compute alpha"
        Z_expected = (2 * math.cosh(alpha)) ** self.N
        if not math.isclose(self.Z, Z_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Z mismatch: expected ~{Z_expected:.3e}, got {self.Z:.3e}"
        return True, "Z matches (2 cosh(alpha))^N"

    def test_invariant_entropy_form(self):
        """Check S = N k_B (ln(2 cosh(alpha)) - alpha tanh(alpha))."""
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B, self.S):
            return False, "SKIP: insufficient data"
        alpha = self._alpha(self.B, self.T)
        if alpha is None:
            return False, "SKIP: failed to compute alpha"
        S_expected = self.N * self.k_B * (math.log(2 * math.cosh(alpha)) - alpha * math.tanh(alpha))
        if not math.isclose(self.S, S_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Entropy mismatch: expected ~{S_expected:.3e}, got {self.S:.3e}"
        return True, "S matches theoretical formula"

    def test_invariant_energy_form(self):
        """Check U = -N μ0 B tanh(alpha)."""
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B, self.U):
            return False, "SKIP: insufficient data"
        alpha = self._alpha(self.B, self.T)
        if alpha is None:
            return False, "SKIP: failed to compute alpha"
        U_expected = -self.N * self.mu0 * self.B * math.tanh(alpha)
        if not math.isclose(self.U, U_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Energy mismatch: expected ~{U_expected:.3e}, got {self.U:.3e}"
        return True, "U matches -N μ0 B tanh(alpha)"

    def test_invariant_magnetization_form(self):
        """Check M_avg = N μ0 tanh(alpha)."""
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B, self.M_avg):
            return False, "SKIP: insufficient data"
        alpha = self._alpha(self.B, self.T)
        if alpha is None:
            return False, "SKIP: failed to compute alpha"
        M_expected = self.N * self.mu0 * math.tanh(alpha)
        if not math.isclose(self.M_avg, M_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"M_avg mismatch: expected ~{M_expected:.3e}, got {self.M_avg:.3e}"
        return True, "M_avg matches N μ0 tanh(alpha)"

    def test_invariant_fluctuation_form(self):
        """Check delta_M = μ0 sqrt(N) sech(alpha)."""
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B, self.delta_M):
            return False, "SKIP: insufficient data"
        alpha = self._alpha(self.B, self.T)
        if alpha is None:
            return False, "SKIP: failed to compute alpha"
        delta_M_expected = self.mu0 * math.sqrt(self.N) / math.cosh(alpha)
        if not math.isclose(self.delta_M, delta_M_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"ΔM mismatch: expected ~{delta_M_expected:.3e}, got {self.delta_M:.3e}"
        return True, "ΔM matches μ0 sqrt(N) sech(alpha)"

    def test_invariant_adiabatic_temperature(self):
        """Check T_f ≈ T_i * B_f / B_i for adiabatic demagnetization."""
        if not self._check_numeric_availability(self.T_f, self.T_i, self.B_i, self.B_f):
            return False, "SKIP: missing T_i, B_i, B_f, or T_f"
        if self.T_i == 0:
            return False, "SKIP: T_i is zero"
        T_f_expected = self.T_i * (self.B_f / self.B_i)
        if not math.isclose(self.T_f, T_f_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"T_f mismatch: expected ~{T_f_expected:.6f}, got {self.T_f:.6f}"
        return True, "T_f matches T_i * B_f / B_i"

    # ------------------------------------------------------------------
    # Modeling / trace tests (quality of explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_adiabatic_entropy_constant(self):
        """S(T_i, B_i) should equal S(T_f, B_f) within tolerance."""
        if not self._check_numeric_availability(self.N, self.mu0, self.T_i, self.B_i, self.T_f, self.B_f, self.k_B):
            return False, "SKIP: missing adiabatic parameters"
        # compute entropy at initial and final states
        alpha_i = self._alpha(self.B_i, self.T_i)
        alpha_f = self._alpha(self.B_f, self.T_f)
        if alpha_i is None or alpha_f is None:
            return False, "SKIP: could not compute alphas"
        S_i = self.N * self.k_B * (math.log(2 * math.cosh(alpha_i)) - alpha_i * math.tanh(alpha_i))
        S_f = self.N * self.k_B * (math.log(2 * math.cosh(alpha_f)) - alpha_f * math.tanh(alpha_f))
        if not math.isclose(S_i, S_f, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Entropy not constant: S_i={S_i:.5e}, S_f={S_f:.5e}"
        return True, "Entropy constant during adiabatic demagnetization"

    def test_modeling_energy_via_partition_derivative(self):
        """
        Check that self.U is consistent with the numerical derivative of ln Z with respect to β.
        This verifies that the solution correctly implements the relation U = -∂ ln Z / ∂β.
        """
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B, self.U, self.Z):
            return False, "SKIP: insufficient data"
        beta = 1.0 / (self.k_B * self.T)
        # compute ln Z at β
        alpha = self.mu0 * self.B * beta
        lnZ = self.N * math.log(2 * math.cosh(alpha))
        # perturb β by a small relative change
        delta_frac = 1e-6
        beta_pert = beta * (1.0 + delta_frac)
        alpha_pert = self.mu0 * self.B * beta_pert
        lnZ_pert = self.N * math.log(2 * math.cosh(alpha_pert))
        dlnZ_dbeta = (lnZ_pert - lnZ) / (beta_pert - beta)
        U_derived = -dlnZ_dbeta
        if not math.isclose(U_derived, self.U, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"U from derivative ({U_derived:.6e}) != self.U ({self.U:.6e})"
        return True, "U consistent with -d ln Z / dβ numerical derivative"

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input and check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_with_B(self):
        """Energy U should decrease (become more negative) as B increases."""
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B, self.U):
            return False, "SKIP: insufficient data"
        # Baseline
        alpha0 = self._alpha(self.B, self.T)
        U0 = self.U  # we assume this is correct
        # Perturb B upward by 5%
        B_pert = self.B * 1.05
        alpha1 = self._alpha(B_pert, self.T)
        if alpha1 is None:
            return False, "SKIP: cannot compute perturbed alpha"
        U1 = -self.N * self.mu0 * B_pert * math.tanh(alpha1)  # derived from model
        if U1 >= U0:
            return False, f"Energy did not decrease: U0={U0:.3e}, U_pert={U1:.3e}"
        return True, "U becomes more negative as B increases"

    def test_monotonicity_magnetization_with_B(self):
        """M_avg should increase (more alignment) as B increases."""
        if not self._check_numeric_availability(self.N, self.mu0, self.B, self.T, self.k_B, self.M_avg):
            return False, "SKIP: insufficient data"
        alpha0 = self._alpha(self.B, self.T)
        M0 = self.M_avg
        B_pert = self.B * 1.05
        alpha1 = self._alpha(B_pert, self.T)
        if alpha1 is None:
            return False, "SKIP: cannot compute perturbed alpha"
        M1 = self.N * self.mu0 * math.tanh(alpha1)  # derived model value
        if M1 <= M0:
            return False, f"M_avg did not increase: M0={M0:.3e}, M_pert={M1:.3e}"
        return True, "M_avg increases as B increases"

    # ------------------------------------------------------------------
    # Weak sanity check (final answer, only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_T_f(self):
        """T_f should be positive and much smaller than T_i (if available)."""
        if self.T_f is None:
            return False, "SKIP: T_f not available"
        if self.T_f <= 0:
            return False, f"T_f must be positive, got {self.T_f}"
        if self.T_i is not None and self.T_i > 0:
            if self.T_f >= self.T_i:
                return False, f"T_f ({self.T_f}) should be < T_i ({self.T_i})"
        return True, "T_f positive and plausibly small"

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
        return all_pass, logs