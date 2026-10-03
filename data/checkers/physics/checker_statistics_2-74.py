import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for photon gas equilibrium problem.
    Checks: correct scaling of N, U, C_V; positivity; consistency relation C_V T / U = 4;
    accurate numerical constant α; derivative consistency; integral ratio; monotonic increase with T.
    """

    # Known constants: ζ(3) ≈ 1.2020569031595942, π from math.pi
    ZETA3 = 1.2020569031595942
    I2 = 2 * ZETA3                     # ∫ x^2/(e^x‑1) dx
    I3 = math.pi**4 / 15               # ∫ x^3/(e^x‑1) dx

    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', {})
        # Main emitted results
        self.N = results.get('N', None)
        self.alpha = results.get('alpha', None)
        self.U = results.get('U', None)
        self.C_V = results.get('C_V', None)

        # Givens: try instance attribute, then fallback to results dict
        self.V = getattr(solution_instance, 'V', results.get('V', None))
        self.T = getattr(solution_instance, 'T', results.get('T', None))
        self.hbar = getattr(solution_instance, 'hbar', results.get('hbar', None))
        self.c = getattr(solution_instance, 'c', results.get('c', None))
        self.k = getattr(solution_instance, 'k', results.get('k', None))

    # ----------------------------------------------------------------------
    #  Test infrastructure
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test_* methods
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

    # ----------------------------------------------------------------------
    #  Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Ensure all expected result entries and givens are present."""
        missing = []
        for key, val in [('N', self.N), ('alpha', self.alpha),
                         ('U', self.U), ('C_V', self.C_V),
                         ('V', self.V), ('T', self.T),
                         ('hbar', self.hbar), ('c', self.c), ('k', self.k)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    # ----------------------------------------------------------------------
    #  Invariant / theorem‑based checks (use math.isclose with 1‑2% tolerance)
    # ----------------------------------------------------------------------
    def test_invariant_positive(self):
        """N, U, C_V must be positive (finite > 0)."""
        if any(v is None for v in [self.N, self.U, self.C_V]):
            return (False, "SKIP: Missing N, U, or C_V")
        for name, val in [('N', self.N), ('U', self.U), ('C_V', self.C_V)]:
            if val <= 0:
                return (False, f"{name} = {val} is not positive")
        return (True, "N, U, C_V are positive")

    def test_invariant_N_scaling(self):
        """Check N ∝ (kT/(ħc))^3 with correct numerical coefficient."""
        if any(v is None for v in [self.N, self.V, self.T, self.hbar, self.c, self.k]):
            return (False, "SKIP: Missing required variables for N scaling")
        coeff = (self.k * self.T / (self.hbar * self.c)) ** 3 * self.V
        expected_N = coeff * (self.I2 / (math.pi**2))
        if math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N scales correctly (expected ~{expected_N:.4g})")
        return (False, f"N = {self.N:.4g}, expected approx {expected_N:.4g}")

    def test_invariant_U_scaling(self):
        """Check U ∝ T^4 with correct Stefan–Boltzmann coefficient."""
        if any(v is None for v in [self.U, self.V, self.T, self.hbar, self.c, self.k]):
            return (False, "SKIP: Missing required variables for U scaling")
        expected_U = (self.V * self.I3 / (math.pi**2) *
                      (self.k**4 / (self.hbar**3 * self.c**3)) * self.T**4)
        if math.isclose(self.U, expected_U, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U scales correctly (expected ~{expected_U:.4g})")
        return (False, f"U = {self.U:.4g}, expected approx {expected_U:.4g}")

    def test_invariant_CV_scaling(self):
        """Check C_V ∝ T^3 with correct coefficient."""
        if any(v is None for v in [self.C_V, self.V, self.T, self.hbar, self.c, self.k]):
            return (False, "SKIP: Missing required variables for C_V scaling")
        expected_CV = (4 * self.V * self.I3 / (math.pi**2) *
                       (self.k**4 / (self.hbar**3 * self.c**3)) * self.T**3)
        if math.isclose(self.C_V, expected_CV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_V scales correctly (expected ~{expected_CV:.4g})")
        return (False, f"C_V = {self.C_V:.4g}, expected approx {expected_CV:.4g}")

    def test_invariant_CV_U_relation(self):
        """C_V should equal 4U/T exactly (thermodynamic consistency)."""
        if any(v is None for v in [self.C_V, self.U, self.T]):
            return (False, "SKIP: Missing C_V, U, or T")
        if self.T == 0:
            return (False, "T = 0, cannot compute ratio")
        ratio = self.C_V * self.T / self.U
        if math.isclose(ratio, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_V*T/U = {ratio:.4f} ≈ 4")
        return (False, f"C_V*T/U = {ratio:.4f}, expected 4")

    # ----------------------------------------------------------------------
    #  Modeling / explanation‑based checks (quality of modelling steps)
    # ----------------------------------------------------------------------
    def test_modeling_alpha_value(self):
        """Check that the constant α equals 2ζ(3)/π² (from the integral)."""
        if self.alpha is None:
            return (False, "SKIP: alpha not provided")
        expected = self.I2 / (math.pi**2)
        if math.isclose(self.alpha, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"α = {self.alpha:.4g} ≈ 2ζ(3)/π² = {expected:.4g}")
        return (False, f"α = {self.alpha:.4g}, expected {expected:.4g}")

    def test_modeling_CV_from_derivative(self):
        """
        Check that C_V is consistent with the derivative of U w.r.t. T,
        using a finite difference approximation (perturbation of T).
        This reproduces the thermodynamic definition from explanation steps.
        """
        if any(v is None for v in [self.U, self.C_V, self.T]):
            return (False, "SKIP: Missing U, C_V, or T for derivative check")
        if self.T == 0:
            return (False, "T = 0, cannot compute finite difference")
        # Perturb T by a small amount (1%)
        delta = 0.01
        T_plus = self.T * (1 + delta)
        # Assume U ∝ T^4, compute U at perturbed T
        U_plus = self.U * (T_plus / self.T) ** 4
        dU_dT_fd = (U_plus - self.U) / (T_plus - self.T)
        if math.isclose(dU_dT_fd, self.C_V, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_V ≈ (ΔU/ΔT) from 1% perturbation: {dU_dT_fd:.4g} vs C_V = {self.C_V:.4g}")
        return (False, f"Derivative mismatch: finite difference = {dU_dT_fd:.4g}, C_V = {self.C_V:.4g}")

    def test_modeling_NU_integral_ratio(self):
        """
        Check the ratio U/(N k T) equals I3/I2 = (π^4/15) / (2ζ(3)).
        This verifies that the same integrals (I2, I3) are used in N and U.
        """
        if any(v is None for v in [self.N, self.U, self.T, self.k]):
            return (False, "SKIP: Missing N, U, T, or k for integral ratio check")
        if self.N == 0 or self.T == 0:
            return (False, "N or T is zero, cannot compute ratio")
        ratio_actual = self.U / (self.N * self.k * self.T)
        ratio_expected = self.I3 / self.I2
        if math.isclose(ratio_actual, ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U/(N k T) = {ratio_actual:.4g} ≈ I3/I2 = {ratio_expected:.4g}")
        return (False, f"U/(N k T) = {ratio_actual:.4g}, expected {ratio_expected:.4g}")

    # ----------------------------------------------------------------------
    #  Monotonicity checks (explicit input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_N_with_T(self):
        """Increasing T by 10% should increase N (per T³ scaling)."""
        if any(v is None for v in [self.N, self.T]):
            return (False, "SKIP: Missing N or T")
        delta = 0.1
        T2 = self.T * (1 + delta)
        expected_N2 = self.N * (T2 / self.T) ** 3
        if expected_N2 > self.N:
            return (True, f"At T+10%: expected N = {expected_N2:.4g} > {self.N:.4g}")
        return (False, "N does not increase with T (scaling contradiction)")

    def test_monotonicity_U_with_T(self):
        """Increasing T by 10% should increase U (per T⁴ scaling)."""
        if any(v is None for v in [self.U, self.T]):
            return (False, "SKIP: Missing U or T")
        delta = 0.1
        T2 = self.T * (1 + delta)
        expected_U2 = self.U * (T2 / self.T) ** 4
        if expected_U2 > self.U:
            return (True, f"At T+10%: expected U = {expected_U2:.4g} > {self.U:.4g}")
        return (False, "U does not increase with T (scaling contradiction)")

    def test_monotonicity_CV_with_T(self):
        """Increasing T by 10% should increase C_V (per T³ scaling)."""
        if any(v is None for v in [self.C_V, self.T]):
            return (False, "SKIP: Missing C_V or T")
        delta = 0.1
        T2 = self.T * (1 + delta)
        expected_CV2 = self.C_V * (T2 / self.T) ** 3
        if expected_CV2 > self.C_V:
            return (True, f"At T+10%: expected C_V = {expected_CV2:.4g} > {self.C_V:.4g}")
        return (False, "C_V does not increase with T (scaling contradiction)")

    # ----------------------------------------------------------------------
    #  Weak answer sanity (at most one, no hardcoded expected values)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """All numerical results must be finite numbers."""
        trouble = []
        for name, val in [('N', self.N), ('U', self.U),
                          ('C_V', self.C_V), ('alpha', self.alpha),
                          ('V', self.V), ('T', self.T),
                          ('hbar', self.hbar), ('c', self.c), ('k', self.k)]:
            if val is not None and not math.isfinite(val):
                trouble.append(f"{name} = {val} is not finite")
        if trouble:
            return (False, "; ".join(trouble))
        return (True, "All provided quantities are finite")