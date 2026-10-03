import math

class PhysicsChecker:
    """
    Audits a solution for the expansion of 1 mole of monatomic ideal gas:
    (a) isothermal expansion from V0 to 2V0
    (b) isobaric expansion from V0 to 2V0
    """

    def __init__(self, solution_instance):
        """Safely extract givens and results. Never use numeric defaults."""
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}

        # Givens (may be missing; default to None)
        self.n = self.givens.get('n', None)          # moles
        self.R = self.givens.get('R', None)          # gas constant
        self.T0 = self.givens.get('T0', None)        # initial temperature
        self.V0 = self.givens.get('V0', None)        # initial volume
        self.Vf = self.givens.get('Vf', None)        # final volume (should be 2*V0)

        # Derived constants (only if R is available)
        if self.R is not None:
            self.Cv = 1.5 * self.R
            self.Cp = 2.5 * self.R
        else:
            self.Cv = None
            self.Cp = None

        # Results (emitted keys)
        self.W_iso = self.results.get('W_iso', None)
        self.Q_iso = self.results.get('Q_iso', None)
        self.W_ib = self.results.get('W_ib', None)
        self.Q_ib = self.results.get('Q_ib', None)

        self.ln2 = math.log(2)

    # ----- interface tests ------------------------------------------------
    def test_interface_W_iso_exists(self):
        if self.W_iso is None:
            return (False, "W_iso is missing or None")
        return (True, "W_iso present")

    def test_interface_Q_iso_exists(self):
        if self.Q_iso is None:
            return (False, "Q_iso is missing or None")
        return (True, "Q_iso present")

    def test_interface_W_ib_exists(self):
        if self.W_ib is None:
            return (False, "W_ib is missing or None")
        return (True, "W_ib present")

    def test_interface_Q_ib_exists(self):
        if self.Q_ib is None:
            return (False, "Q_ib is missing or None")
        return (True, "Q_ib present")

    def test_interface_numerical_type(self):
        for key, val in [('W_iso', self.W_iso), ('Q_iso', self.Q_iso),
                         ('W_ib', self.W_ib), ('Q_ib', self.Q_ib)]:
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"{key} is not numeric (type {type(val).__name__})")
        return (True, "All results are numerical")

    # ----- invariant tests (theorem-based, use math.isclose) --------------
    def test_invariant_isothermal_deltaU_zero(self):
        if self.W_iso is None or self.Q_iso is None:
            return (False, "SKIP: missing W_iso or Q_iso")
        if not math.isclose(self.Q_iso, self.W_iso, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_iso ({self.Q_iso}) != W_iso ({self.W_iso})")
        return (True, "Isothermal Q = W (ΔU=0)")

    def test_invariant_isothermal_work_formula(self):
        if None in (self.W_iso, self.n, self.R, self.T0):
            return (False, "SKIP: missing n, R, T0, or W_iso")
        expected = self.n * self.R * self.T0 * self.ln2
        if not math.isclose(self.W_iso, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W_iso ({self.W_iso}) != nRT0ln2 ({expected})")
        return (True, "Isothermal work matches nRT0ln2")

    def test_invariant_isobaric_work_formula(self):
        if None in (self.W_ib, self.n, self.R, self.T0):
            return (False, "SKIP: missing n, R, T0, or W_ib")
        expected = self.n * self.R * self.T0
        if not math.isclose(self.W_ib, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W_ib ({self.W_ib}) != nRT0 ({expected})")
        return (True, "Isobaric work matches nRT0")

    def test_invariant_isobaric_first_law(self):
        if None in (self.R, self.T0, self.Q_ib, self.W_ib, self.n):
            return (False, "SKIP: missing required quantities for first law")
        delta_T = self.T0          # V doubles at constant P => T doubles
        delta_U = self.n * self.Cv * delta_T
        expected_Q = delta_U + self.W_ib
        if not math.isclose(self.Q_ib, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_ib ({self.Q_ib}) != ΔU+W_ib ({expected_Q})")
        return (True, "Isobaric first law satisfied")

    def test_invariant_isobaric_heat_formula(self):
        if None in (self.Q_ib, self.n, self.R, self.T0):
            return (False, "SKIP: missing Q_ib, n, R, T0")
        expected = self.n * self.Cp * self.T0
        if not math.isclose(self.Q_ib, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_ib ({self.Q_ib}) != nCpΔT ({expected})")
        return (True, "Isobaric heat matches nCpΔT")

    # ----- modeling / trace consistency tests -----------------------------
    def test_modeling_positive_work(self):
        if self.W_iso is not None and self.W_iso <= 0:
            return (False, f"Isothermal work non-positive: {self.W_iso}")
        if self.W_ib is not None and self.W_ib <= 0:
            return (False, f"Isobaric work non-positive: {self.W_ib}")
        return (True, "All works are positive")

    def test_modeling_heat_positive(self):
        if self.Q_iso is not None and self.Q_iso <= 0:
            return (False, f"Isothermal heat non-positive: {self.Q_iso}")
        if self.Q_ib is not None and self.Q_ib <= 0:
            return (False, f"Isobaric heat non-positive: {self.Q_ib}")
        return (True, "All heats are positive")

    def test_modeling_isobaric_work_less_than_heat(self):
        if self.W_ib is None or self.Q_ib is None:
            return (False, "SKIP: missing W_ib or Q_ib")
        if not (self.W_ib < self.Q_ib):
            return (False, f"W_ib ({self.W_ib}) is not less than Q_ib ({self.Q_ib})")
        return (True, "Isobaric work < heat (ΔU>0)")

    def test_modeling_isothermal_vs_isobaric_work(self):
        if self.W_iso is None or self.W_ib is None:
            return (False, "SKIP: missing both work values")
        if not (self.W_iso < self.W_ib):
            return (False, f"Isothermal work ({self.W_iso}) > isobaric work ({self.W_ib})")
        return (True, "Isothermal work < isobaric work (ln2 < 1)")

    # ----- monotonicity tests ---------------------------------------------
    def test_monotonicity_isothermal_work_increases_with_T0(self):
        """
        Verify that isothermal work increases with T0.
        Using the given T0, compute the expected perturbed work for T0*1.1
        and confirm it is larger than the baseline.
        """
        if None in (self.W_iso, self.n, self.R, self.T0):
            return (False, "SKIP: missing data for monotonicity check")
        # Baseline expected (already close to W_iso)
        base_expected = self.n * self.R * self.T0 * self.ln2
        # Perturb T0 by +10%
        T0_pert = self.T0 * 1.1
        perturbed_expected = self.n * self.R * T0_pert * self.ln2
        # Monotonicity: perturbed > baseline
        if not (perturbed_expected > base_expected):
            return (False, f"Perturbed work ({perturbed_expected}) not > baseline ({base_expected})")
        return (True, "Isothermal work increases with T0")

    def test_monotonicity_isobaric_work_increases_with_T0(self):
        """
        Verify that isobaric work increases with T0.
        """
        if None in (self.W_ib, self.n, self.R, self.T0):
            return (False, "SKIP: missing data for monotonicity check")
        base_expected = self.n * self.R * self.T0
        T0_pert = self.T0 * 1.1
        perturbed_expected = self.n * self.R * T0_pert
        if not (perturbed_expected > base_expected):
            return (False, f"Perturbed work ({perturbed_expected}) not > baseline ({base_expected})")
        return (True, "Isobaric work increases with T0")

    # ----- weak answer sanity (single method) -----------------------------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: all results are positive and finite.
        No isclose, no re-derivation of target values.
        """
        for name, val in [('W_iso', self.W_iso), ('Q_iso', self.Q_iso),
                          ('W_ib', self.W_ib), ('Q_ib', self.Q_ib)]:
            if val is not None:
                if val <= 0:
                    return (False, f"{name} is non-positive: {val}")
                if not math.isfinite(val):
                    return (False, f"{name} is not finite: {val}")
        return (True, "All results positive and finite")

    # ----- runner ---------------------------------------------------------
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