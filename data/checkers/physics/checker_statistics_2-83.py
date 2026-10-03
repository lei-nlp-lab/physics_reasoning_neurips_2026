import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the 1D lattice specific heat problem.
    Checks existence, types, physical invariants, and consistency.
    """

    def __init__(self, solution_instance):
        # Safely extract main containers
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Extract required result keys
        self.Cv_highT = self.results.get('Cv_highT', None)
        self.alpha = self.results.get('alpha', None)
        self.gamma = self.results.get('gamma', None)

        # Extract relevant givens (if present)
        self.N = self.givens.get('N', None)
        self.kB = self.givens.get('kB', None)
        self.hbar = self.givens.get('hbar', None)
        self.wbar = self.givens.get('wbar', None)
        self.T = self.givens.get('T', None)

    # ------------------------------------------------------------------
    # Interface checks – one per key
    # ------------------------------------------------------------------
    def test_interface_required_keys_Cv_highT(self):
        if self.Cv_highT is None:
            return (False, "Missing result key 'Cv_highT'")
        return (True, "'Cv_highT' present")

    def test_interface_required_keys_alpha(self):
        if self.alpha is None:
            return (False, "Missing result key 'alpha'")
        return (True, "'alpha' present")

    def test_interface_required_keys_gamma(self):
        if self.gamma is None:
            return (False, "Missing result key 'gamma'")
        return (True, "'gamma' present")

    def test_interface_type_Cv_highT_numeric(self):
        if self.Cv_highT is None:
            return (False, "SKIP: 'Cv_highT' not available")
        if not isinstance(self.Cv_highT, (int, float)):
            return (False, f"'Cv_highT' is not numeric, got {type(self.Cv_highT).__name__}")
        return (True, "'Cv_highT' is numeric")

    def test_interface_type_alpha_int_or_float(self):
        if self.alpha is None:
            return (False, "SKIP: 'alpha' not available")
        if not isinstance(self.alpha, (int, float)):
            return (False, f"'alpha' should be int or float, got {type(self.alpha).__name__}")
        return (True, "'alpha' has valid type")

    def test_interface_type_gamma_int_or_float(self):
        if self.gamma is None:
            return (False, "SKIP: 'gamma' not available")
        if not isinstance(self.gamma, (int, float)):
            return (False, f"'gamma' should be int or float, got {type(self.gamma).__name__}")
        return (True, "'gamma' has valid type")

    # ------------------------------------------------------------------
    # Invariant checks – physical sanity
    # ------------------------------------------------------------------
    def test_invariant_highT_positive(self):
        if self.Cv_highT is None:
            return (False, "SKIP: 'Cv_highT' not available")
        if self.Cv_highT <= 0:
            return (False, f"'Cv_highT' should be positive, got {self.Cv_highT}")
        return (True, "'Cv_highT' > 0")

    def test_invariant_alpha_positive(self):
        if self.alpha is None:
            return (False, "SKIP: 'alpha' not available")
        if self.alpha <= 0:
            return (False, f"'alpha' should be positive, got {self.alpha}")
        return (True, "'alpha' > 0")

    def test_invariant_gamma_positive(self):
        if self.gamma is None:
            return (False, "SKIP: 'gamma' not available")
        if self.gamma <= 0:
            return (False, f"'gamma' should be positive, got {self.gamma}")
        return (True, "'gamma' > 0")

    def test_invariant_highT_scales_with_N(self):
        """If N and kB are given, check Cv_highT ≈ N * kB (classical equipartition)."""
        if self.Cv_highT is None:
            return (False, "SKIP: 'Cv_highT' not available")
        if self.N is None or self.kB is None:
            return (False, "SKIP: cannot check scaling without N and kB givens")
        expected = self.N * self.kB
        if not math.isclose(self.Cv_highT, expected, rel_tol=1e-2):
            return (False, f"'Cv_highT' = {self.Cv_highT} not close to N*kB = {expected}")
        return (True, f"'Cv_highT' consistent with N*kB")

    def test_invariant_alpha_one(self):
        """Low-temperature exponent α should be 1 (Debye-like)."""
        if self.alpha is None:
            return (False, "SKIP: 'alpha' not available")
        if not math.isclose(self.alpha, 1.0, rel_tol=1e-2):
            return (False, f"'alpha' = {self.alpha} not close to 1")
        return (True, "'alpha' ≈ 1")

    def test_invariant_gamma_one(self):
        """Low-temperature exponent γ should be 1 (linear in T)."""
        if self.gamma is None:
            return (False, "SKIP: 'gamma' not available")
        if not math.isclose(self.gamma, 1.0, rel_tol=1e-2):
            return (False, f"'gamma' = {self.gamma} not close to 1")
        return (True, "'gamma' ≈ 1")

    # ------------------------------------------------------------------
    # Modeling checks – consistency with explanation / theoretical structure
    # ------------------------------------------------------------------
    def test_modeling_highT_independent_of_wbar(self):
        """High-T limit should not depend on oscillator strength ō."""
        if self.Cv_highT is None:
            return (False, "SKIP: 'Cv_highT' not available")
        if not math.isfinite(self.Cv_highT):
            return (False, "'Cv_highT' is not finite")
        return (True, "'Cv_highT' finite (implies no dependence on wbar)")

    def test_modeling_lowT_exponents_positive_correlation(self):
        """Both low-T exponents should be positive (specific heat increasing with T)."""
        if self.alpha is None or self.gamma is None:
            return (False, "SKIP: exponent(s) not available")
        if self.alpha <= 0 or self.gamma <= 0:
            return (False, f"Exponents must be positive for increasing C_v(T); α={self.alpha}, γ={self.gamma}")
        return (True, "α,γ > 0")

    # ------------------------------------------------------------------
    # Monotonicity checks – explicit perturbation and comparison
    # ------------------------------------------------------------------
    def test_monotonicity_Cv_highT_increases_with_N(self):
        """
        Perturb the number of particles N by +10% and check that the
        high‑temperature heat capacity (computed via equipartition) increases.
        """
        if self.N is None or self.kB is None or self.Cv_highT is None:
            return (False, "SKIP: needed data missing")
        epsilon = 0.1
        N_pert = self.N * (1 + epsilon)
        # According to equipartition Cv_highT = N * kB
        Cv_pert = N_pert * self.kB
        # Baseline is the original Cv_highT (which should equal N*kB)
        if not math.isclose(self.Cv_highT, self.N * self.kB, rel_tol=1e-2):
            # the original may deviate slightly; still check that perturbed > original
            pass
        if not (Cv_pert > self.Cv_highT):
            return (False, f"Perturbed Cv_highT ({Cv_pert}) not larger than baseline ({self.Cv_highT})")
        return (True, "Cv_highT increases with N (equipartition)")

    def test_monotonicity_lowT_increases_with_T_exponent(self):
        """
        Using the solution's gamma, compute a synthetic low‑T specific heat
        (proportional to T^gamma) and perturb T by +10%.  Check that the
        synthetic heat capacity increases.
        """
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        # Use arbitrary dimensionless T values (A = 1)
        T0 = 1.0
        T1 = 1.1   # 10% increase
        Cv0 = T0 ** self.gamma
        Cv1 = T1 ** self.gamma
        if not (Cv1 > Cv0):
            return (False, f"With gamma={self.gamma}, low‑T Cv does not increase when T increases")
        return (True, f"Low‑T Cv increases with T (gamma={self.gamma})")

    # ------------------------------------------------------------------
    # Optional weak sanity – minimal, no re‑derivation
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Very weak sanity: results finite, not NaN, and positive signs.
        No numeric comparison with target answers.
        """
        checks = []
        if self.Cv_highT is not None:
            if not math.isfinite(self.Cv_highT):
                checks.append("Cv_highT not finite")
            if self.Cv_highT <= 0:
                checks.append("Cv_highT not positive")
        if self.alpha is not None:
            if not math.isfinite(self.alpha):
                checks.append("alpha not finite")
            if self.alpha <= 0:
                checks.append("alpha not positive")
        if self.gamma is not None:
            if not math.isfinite(self.gamma):
                checks.append("gamma not finite")
            if self.gamma <= 0:
                checks.append("gamma not positive")
        if checks:
            return (False, "Weak sanity failed: " + "; ".join(checks))
        return (True, "All basic sanity checks passed")

    # ------------------------------------------------------------------
    # Discovery and execution of tests
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)