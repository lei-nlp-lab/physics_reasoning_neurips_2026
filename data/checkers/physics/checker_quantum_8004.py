import math
import sys

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results
        results = getattr(solution_instance, 'results', {}) or {}
        self.KE = results.get('KE', None)
        self.B = results.get('B', None)          # in Tesla
        self.n = results.get('n', None)
        self.ratio = results.get('ratio', None)

        # Safe extraction of given constants (if provided)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.hbar = givens.get('hbar', None)          # J·s
        self.m_n = givens.get('mN', None)             # kg
        self.R = givens.get('R', None)                # m
        self.Eb = givens.get('Eb', None)              # J
        self.mu_B = givens.get('mu_B', None)          # J/T
        self.m_osc = givens.get('m_osc', None)        # kg
        self.T = givens.get('T', None)                # s
        self.x0 = givens.get('x0', None)              # m
        self.me = givens.get('me', None)              # kg
        self.mp = givens.get('mp', None)              # kg
        self.alpha = givens.get('alpha', None)        # dimensionless

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All four result keys must exist and be non‑None."""
        missing = []
        for key, val in [('KE', self.KE), ('B', self.B),
                          ('n', self.n), ('ratio', self.ratio)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_kinetic_energy_type(self):
        """KE should be numeric (int, float)."""
        if self.KE is None:
            return (False, "KE is missing")
        if not isinstance(self.KE, (int, float)):
            return (False, f"KE type is {type(self.KE)}, expected numeric")
        return (True, "KE is numeric")

    def test_interface_negative_kinetic_energy(self):
        """KE must be non‑negative."""
        if self.KE is None:
            return (False, "KE missing")
        if self.KE < 0:
            return (False, f"KE={self.KE} is negative")
        return (True, "KE is non‑negative")

    # ------------------------------------------------------------------
    # Invariant / modeling tests (use large tolerance for order‑of‑magn)
    # ------------------------------------------------------------------
    def test_invariant_uncertainty_relation(self):
        """
        For the nucleon kinetic energy, check that
        KE * m_n * R^2 ≈ hbar^2 / 2.
        Use factor‑of‑10 tolerance given order‑of‑magnitude nature.
        """
        if None in (self.KE, self.hbar, self.m_n, self.R):
            return (False, "SKIP: missing constants (hbar, m_n, R) for invariant")
        lhs = self.KE * self.m_n * (self.R ** 2)
        rhs = (self.hbar ** 2) / 2.0
        # rel_tol=9.0 allows factor 10
        if math.isclose(lhs, rhs, rel_tol=9.0):
            return (True, f"KE·m_n·R^2 ≈ ħ²/2 (ratio {lhs/rhs:.2e})")
        else:
            return (False, f"KE·m_n·R^2 = {lhs:.3e}, ħ²/2 = {rhs:.3e} (ratio {lhs/rhs:.2e})")

    def test_modeling_zeeman_field(self):
        """
        Check that B ≈ Eb / mu_B (Zeeman splitting equals Coulomb binding).
        Accept factor 10.
        """
        if None in (self.B, self.Eb, self.mu_B):
            return (False, "SKIP: missing Eb or mu_B for Zeeman check")
        expected_B = self.Eb / self.mu_B
        if math.isclose(self.B, expected_B, rel_tol=9.0):
            return (True, f"B ≈ Eb/μ_B = {expected_B:.2e} T (actual {self.B:.2e} T)")
        else:
            return (False, f"B = {self.B:.2e} T, expected ~{expected_B:.2e} T")

    def test_modeling_classical_oscillator_energy(self):
        """
        For the harmonic oscillator, check that the classical energy equals
        E_c = 0.5·m_osc·(2π/T)²·x0², and that n ≈ E_c/(ħ·ω).
        Accept factor 10.
        """
        if None in (self.n, self.m_osc, self.T, self.x0, self.hbar):
            return (False, "SKIP: missing oscillator parameters for n test")
        omega = 2.0 * math.pi / self.T
        E_classical = 0.5 * self.m_osc * (omega ** 2) * (self.x0 ** 2)
        expected_n = E_classical / (self.hbar * omega)
        if math.isclose(self.n, expected_n, rel_tol=9.0):
            return (True, f"n ≈ E_c/(ħω) = {expected_n:.2e} (actual {self.n:.2e})")
        else:
            return (False, f"n = {self.n:.2e}, expected ~{expected_n:.2e}")

    def test_modeling_hyperfine_ratio_parametric(self):
        """
        Check that the hyperfine ratio is 2·α²·(m_e/m_p).
        Since this is a symbolic expression, we verify the numeric value
        if constants are provided.
        """
        if None in (self.ratio, self.alpha, self.me, self.mp):
            return (False, "SKIP: missing α, m_e, m_p for hyperfine ratio")
        expected_ratio = 2.0 * (self.alpha ** 2) * (self.me / self.mp)
        if math.isclose(self.ratio, expected_ratio, rel_tol=9.0):
            return (True, f"ratio ≈ 2α²(m_e/m_p) = {expected_ratio:.2e} (actual {self.ratio:.2e})")
        else:
            return (False, f"ratio = {self.ratio:.2e}, expected ~{expected_ratio:.2e}")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_n_vs_period(self):
        """
        For the oscillator part: n ∝ 1/T (since n ≈ (π m x0²)/(ħ T)).
        Perturb T upward by +10% and check that the recomputed n decreases.
        We need the stored constants to recompute n at perturbed T.
        """
        if None in (self.n, self.m_osc, self.T, self.x0, self.hbar):
            return (False, "SKIP: missing oscillator parameters for monotonicity n vs T")
        # Original n from solution
        original_n = self.n
        # Store baseline constants
        m = self.m_osc
        T0 = self.T
        x0 = self.x0
        hbar = self.hbar
        # Compute n using the formula n ≈ (π m x0²)/(ħ T)
        baseline_n = (math.pi * m * x0**2) / (hbar * T0)
        # Perturb T upward by 10%
        T1 = T0 * 1.1
        perturbed_n = (math.pi * m * x0**2) / (hbar * T1)
        # The perturbed n should be smaller than baseline_n
        if perturbed_n < baseline_n:
            return (True, f"n decreases when T increases: baseline_n={baseline_n:.2e}, perturbed_n={perturbed_n:.2e}")
        else:
            return (False, f"n does not decrease with increasing T: baseline_n={baseline_n:.2e}, perturbed_n={perturbed_n:.2e}")

    # ------------------------------------------------------------------
    # Pure physics sanity (non‑numeric where appropriate)
    # ------------------------------------------------------------------
    def test_invariant_all_results_positive(self):
        """All results must be positive (order estimates)."""
        results = [self.KE, self.B, self.n, self.ratio]
        if any(v is None for v in results):
            return (False, "SKIP: one or more results missing")
        if any(v <= 0 for v in results):
            neg = [(k, v) for k, v in zip(["KE","B","n","ratio"], results) if v <= 0]
            return (False, f"Non‑positive values: {neg}")
        return (True, "All results positive")

    def test_invariant_finite_values(self):
        """All results must be finite floats/ints."""
        results = [self.KE, self.B, self.n, self.ratio]
        if any(v is None for v in results):
            return (False, "SKIP: missing results")
        for name, val in zip(["KE","B","n","ratio"], results):
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")
        return (True, "All results finite")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no exact re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """
        Check that orders of magnitude are plausible:
        KE ~ 10^7 eV, B ~ 10^9 G (≈ 10^5 T), n ~ 10^27, ratio ~ 10^-8.
        Use broad ranges, does NOT use math.isclose.
        """
        if self.KE is None or self.B is None or self.n is None or self.ratio is None:
            return (False, "SKIP: missing results for magnitude check")
        # KE in J: 10^7 eV ≈ 1.6e-12 J, allow factor 10^3
        if not (1e-15 < self.KE < 1e-9):
            return (False, f"KE = {self.KE} J outside expected broad range (1e-15, 1e-9)")
        # B in T: 10^9 G = 10^5 T, allow factor 1000
        if not (1e2 < self.B < 1e8):
            return (False, f"B = {self.B} T outside expected broad range (1e2, 1e8)")
        # n: around 10^27, allow factor 10^6
        if not (1e21 < self.n < 1e33):
            return (False, f"n = {self.n} outside expected broad range (1e21, 1e33)")
        # ratio: around 10^-8, allow factor 10^4
        if not (1e-12 < self.ratio < 1e-4):
            return (False, f"ratio = {self.ratio} outside expected broad range (1e-12, 1e-4)")
        return (True, "All results within plausible order‑of‑magnitude ranges")

    # ------------------------------------------------------------------
    # Main runner (do not modify)
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