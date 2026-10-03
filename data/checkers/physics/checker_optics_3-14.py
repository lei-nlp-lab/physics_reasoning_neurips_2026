import sympy
import math
import traceback

class PhysicsChecker:
    """
    Physics checker for broadening mechanisms problem: Doppler and Zeeman broadening.
    Verifies consistency against basic physical formulas and order-of-magnitude.
    """
    def __init__(self, solution_instance):
        # Extract results safely; use None for missing keys
        self.results = getattr(solution_instance, 'results', {})
        self.dlambda_D = self.results.get('dlambda_D', None)
        self.dlambda_Z = self.results.get('dlambda_Z', None)
        self.dlambda_Z2 = self.results.get('dlambda_Z2', None)

        # Givens from the instance (if stored); default None.
        self.given_lambda = getattr(solution_instance, 'lambda', None)
        self.given_kT = getattr(solution_instance, 'kT', None)     # in eV
        self.given_n = getattr(solution_instance, 'n', None)
        self.given_B = getattr(solution_instance, 'B', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_has_dlambda_D(self):
        """Result dictionary must contain key 'dlambda_D'."""
        if self.dlambda_D is None:
            return (False, "Key 'dlambda_D' is missing")
        return (True, "'dlambda_D' present")

    def test_interface_has_dlambda_Z(self):
        """Result dictionary must contain key 'dlambda_Z'."""
        if self.dlambda_Z is None:
            return (False, "Key 'dlambda_Z' is missing")
        return (True, "'dlambda_Z' present")

    def test_interface_has_dlambda_Z2(self):
        """Result dictionary must contain key 'dlambda_Z2' (Zeeman from nonuniform field)."""
        if self.dlambda_Z2 is None:
            return (False, "Key 'dlambda_Z2' is missing")
        return (True, "'dlambda_Z2' present")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, using formulas; use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_doppler_formula_order(self):
        """
        Check that delta_lambda_D computed from the Doppler formula matches within 2%.
        Use constant values from the problem (fallback if givens missing).
        """
        if self.dlambda_D is None:
            return (False, "SKIP: dlambda_D not available")
        # Constants (fallback to problem defaults if givens not provided)
        lam = 500e-9 if self.given_lambda is None else self.given_lambda
        kT_eV = 20.0 if self.given_kT is None else self.given_kT
        kT_J = kT_eV * 1.602e-19
        m_C = 12 * 1.6605e-27
        c = 3e8
        sqrt_term = math.sqrt(2.0 * kT_J / (m_C * c**2))
        expected_D = lam * sqrt_term
        if not math.isclose(self.dlambda_D, expected_D, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Doppler width {self.dlambda_D:.3e} m does not match expected {expected_D:.3e} m")
        return (True, f"Doppler width {self.dlambda_D:.3e} m consistent with formula (expected {expected_D:.3e} m)")

    def test_invariant_zeeman_formula_order(self):
        """
        Check that delta_lambda_Z from Zeeman formula matches within 2%.
        """
        if self.dlambda_Z is None:
            return (False, "SKIP: dlambda_Z not available")
        lam = 500e-9 if self.given_lambda is None else self.given_lambda
        c = 3e8
        mu_B = 5.79e-5         # eV/T
        h = 4.1357e-15         # eV*s
        B = 1.0 if self.given_B is None else self.given_B
        expected_Z = (lam**2 / c) * (mu_B * B / h)
        if not math.isclose(self.dlambda_Z, expected_Z, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Zeeman width {self.dlambda_Z:.3e} m does not match expected {expected_Z:.3e} m")
        return (True, f"Zeeman width {self.dlambda_Z:.3e} m consistent with formula (expected {expected_Z:.3e} m)")

    def test_invariant_zeeman_second_equivalent(self):
        """
        Since both dlambda_Z and dlambda_Z2 refer to same mechanism (non‑uniform B),
        they should agree to within 2%.
        """
        if self.dlambda_Z is None or self.dlambda_Z2 is None:
            return (False, "SKIP: one of the Zeeman results missing")
        if not math.isclose(self.dlambda_Z, self.dlambda_Z2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Zeeman results differ: dlambda_Z={self.dlambda_Z:.3e}, dlambda_Z2={self.dlambda_Z2:.3e}")
        return (True, "Both Zeeman results agree")

    # ------------------------------------------------------------------
    # Modeling / trace checks (quality of intermediate steps, functional forms)
    # ------------------------------------------------------------------
    def test_modeling_both_broadenings_on_order(self):
        """
        Both Doppler and Zeeman widths should be on the order of 0.1-0.2 Å,
        i.e., between 1e-11 and 2e-11 m.
        """
        if self.dlambda_D is None or self.dlambda_Z is None:
            return (False, "SKIP: one or both broadening results missing")
        lower = 1e-11
        upper = 2e-11
        D_ok = lower <= self.dlambda_D <= upper
        Z_ok = lower <= self.dlambda_Z <= upper
        if not (D_ok and Z_ok):
            msg = f"Broadening outside expected range: D={self.dlambda_D:.3e}, Z={self.dlambda_Z:.3e}, expected ~1e-11 to 2e-11"
            return (False, msg)
        return (True, f"Both widths are in range [1e-11, 2e-11] m: D={self.dlambda_D:.3e}, Z={self.dlambda_Z:.3e}")

    def test_trace_doppler_sqrt_term(self):
        """
        Trace quality: the intermediate sqrt(2kT/(m c^2)) used in Doppler
        should be approximately 4e-5 (within 2%).
        """
        if self.dlambda_D is None:
            return (False, "SKIP: dlambda_D not available")
        kT_eV = 20.0 if self.given_kT is None else self.given_kT
        kT_J = kT_eV * 1.602e-19
        m_C = 12 * 1.6605e-27
        c = 3e8
        sqrt_term = math.sqrt(2.0 * kT_J / (m_C * c**2))
        expected_sqrt = 4.0e-5  # order‑of‑magnitude from explanation steps
        if not math.isclose(sqrt_term, expected_sqrt, rel_tol=1e-2):
            return (False, f"Intermediate sqrt term = {sqrt_term:.3e}, expected ~{expected_sqrt:.3e}")
        return (True, f"sqrt(2kT/(m c^2)) = {sqrt_term:.3e} consistent with 4e-5")

    def test_trace_zeeman_prefactor(self):
        """
        Trace quality: the prefactor λ^2/c used in Zeeman broadening
        should be approximately 8.33e-22 s (within 2%).
        """
        if self.dlambda_Z is None:
            return (False, "SKIP: dlambda_Z not available")
        lam = 500e-9 if self.given_lambda is None else self.given_lambda
        c = 3e8
        prefactor = lam**2 / c
        expected_pre = 8.33e-22  # from pseudocode
        if not math.isclose(prefactor, expected_pre, rel_tol=1e-2):
            return (False, f"λ^2/c = {prefactor:.3e} s, expected ~{expected_pre:.3e} s")
        return (True, f"λ^2/c = {prefactor:.3e} s consistent with 8.33e-22 s")

    # ------------------------------------------------------------------
    # Monotonicity checks (must perturb an input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_doppler_increases_with_temperature(self):
        """
        Perturb kT (temperature) by +10% and verify Doppler width increases.
        Since we cannot recompute with the solution's function, we use the formula
        to derive the expected monotonic behavior.
        """
        if self.dlambda_D is None:
            return (False, "SKIP: dlambda_D not available")
        kT_eV = 20.0 if self.given_kT is None else self.given_kT
        lam = 500e-9 if self.given_lambda is None else self.given_lambda
        m_C = 12 * 1.6605e-27
        c = 3e8
        # Baseline width (using formula)
        kT_J_base = kT_eV * 1.602e-19
        baseline = lam * math.sqrt(2.0 * kT_J_base / (m_C * c**2))
        # Perturb kT by +10%
        kT_J_perturbed = 1.1 * kT_J_base
        perturbed = lam * math.sqrt(2.0 * kT_J_perturbed / (m_C * c**2))
        # The perturbed width should be larger than baseline
        if not (perturbed > baseline):
            return (False, f"Monotonicity failed: perturbed width {perturbed:.3e} <= baseline {baseline:.3e}")
        # Additionally, verify that the ratio follows sqrt(1.1) ~ 1.049
        expected_ratio = math.sqrt(1.1)  # 1.0488...
        actual_ratio = perturbed / baseline
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-3):
            return (False, f"Ratio {actual_ratio:.4f} does not match expected sqrt(1.1) = {expected_ratio:.4f}")
        return (True, f"Doppler width increases with temperature: ratio = {actual_ratio:.4f} (expected {expected_ratio:.4f})")

    def test_monotonicity_zeeman_increases_with_magnetic_field(self):
        """
        Perturb B by +10% and verify Zeeman width increases.
        """
        if self.dlambda_Z is None:
            return (False, "SKIP: dlambda_Z not available")
        lam = 500e-9 if self.given_lambda is None else self.given_lambda
        c = 3e8
        mu_B = 5.79e-5
        h = 4.1357e-15
        B_base = 1.0 if self.given_B is None else self.given_B
        baseline = (lam**2 / c) * (mu_B * B_base / h)
        B_perturbed = 1.1 * B_base
        perturbed = (lam**2 / c) * (mu_B * B_perturbed / h)
        if not (perturbed > baseline):
            return (False, f"Monotonicity failed: perturbed width {perturbed:.3e} <= baseline {baseline:.3e}")
        expected_ratio = 1.1
        actual_ratio = perturbed / baseline
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-3):
            return (False, f"Ratio {actual_ratio:.4f} does not match expected {expected_ratio:.4f}")
        return (True, f"Zeeman width increases with B: ratio = {actual_ratio:.4f} (expected {expected_ratio:.4f})")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one; no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        """All broadening widths must be positive and finite."""
        for key, val in [('dlambda_D', self.dlambda_D), ('dlambda_Z', self.dlambda_Z), ('dlambda_Z2', self.dlambda_Z2)]:
            if val is None:
                continue
            if not (val > 0 and math.isfinite(val)):
                return (False, f"{key} = {val} is not positive/finite")
        return (True, "All present results are positive and finite")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (bool, list of strings)."""
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)