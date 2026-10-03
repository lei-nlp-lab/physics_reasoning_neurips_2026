import math
import traceback

class PhysicsChecker:
    """
    Auditor for adiabatic compression problem.
    Checks interface, invariants, modeling quality, and weak sanity.
    """

    # Physics constants (from problem statement) – used only for re-derivation when solution provides no gamma
    GAMMA_AIR = 7/5
    GAMMA_HE = 5/3

    def __init__(self, solution_instance):
        # Safely extract results
        self.results = getattr(solution_instance, 'results', None) or {}
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

        # Target output keys (expected from pseudocode [emit])
        self.T2_air = self.results.get('T2_air', None)
        self.T2_he = self.results.get('T2_he', None)

        # Givens – use None if missing; never assign numeric defaults for missing data
        self.P1 = (self.results.get('P1') or self.givens.get('P1') or None)
        self.T1 = (self.results.get('T1') or self.givens.get('T1') or None)
        self.P2 = (self.results.get('P2') or self.givens.get('P2') or None)

        # Gammas – prefer solution's values, else use constants (only for re-derivation)
        self.gamma_air = (self.results.get('gamma_air') or
                          self.givens.get('gamma_air') or
                          self.GAMMA_AIR)
        self.gamma_he = (self.results.get('gamma_he') or
                         self.givens.get('gamma_he') or
                         self.GAMMA_HE)

        # Derived quantities (used in invariants)
        self.ratio = None
        if self.P2 is not None and self.P1 is not None and self.P1 != 0:
            self.ratio = self.P2 / self.P1

    # ------------------------------------------------------------------
    # Interface tests – ensure required result keys exist and are numeric
    # ------------------------------------------------------------------
    def test_interface_T2_air_exists(self):
        if self.T2_air is None:
            return (False, "Missing results['T2_air']")
        return (True, "T2_air present")

    def test_interface_T2_he_exists(self):
        if self.T2_he is None:
            return (False, "Missing results['T2_he']")
        return (True, "T2_he present")

    def test_interface_T2_air_numeric(self):
        if self.T2_air is None:
            return (False, "SKIP: T2_air missing")
        if not isinstance(self.T2_air, (int, float)):
            return (False, f"T2_air is not numeric, type={type(self.T2_air).__name__}")
        return (True, "T2_air is numeric")

    def test_interface_T2_he_numeric(self):
        if self.T2_he is None:
            return (False, "SKIP: T2_he missing")
        if not isinstance(self.T2_he, (int, float)):
            return (False, f"T2_he is not numeric, type={type(self.T2_he).__name__}")
        return (True, "T2_he is numeric")

    # ------------------------------------------------------------------
    # Invariant tests – check physical laws (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_adiabatic_air(self):
        """Check T2_air ≈ T1 * (P2/P1)^{((γ-1)/γ)}"""
        if self.T2_air is None:
            return (False, "SKIP: T2_air not available")
        if self.ratio is None or self.T1 is None:
            return (False, "SKIP: missing P1, P2 or T1")
        exponent = (self.gamma_air - 1) / self.gamma_air
        expected = self.T1 * (self.ratio ** exponent)
        if math.isclose(self.T2_air, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T2_air ({self.T2_air:.2f}) ≈ expected ({expected:.2f}) within 1%")
        else:
            return (False, f"T2_air ({self.T2_air:.2f}) ≠ expected ({expected:.2f}) – deviation >1%")

    def test_invariant_adiabatic_he(self):
        """Check T2_he ≈ T1 * (P2/P1)^{((γ-1)/γ)}"""
        if self.T2_he is None:
            return (False, "SKIP: T2_he not available")
        if self.ratio is None or self.T1 is None:
            return (False, "SKIP: missing P1, P2 or T1")
        exponent = (self.gamma_he - 1) / self.gamma_he
        expected = self.T1 * (self.ratio ** exponent)
        if math.isclose(self.T2_he, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T2_he ({self.T2_he:.2f}) ≈ expected ({expected:.2f}) within 1%")
        else:
            return (False, f"T2_he ({self.T2_he:.2f}) ≠ expected ({expected:.2f}) – deviation >1%")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check quality of implementation
    # ------------------------------------------------------------------
    def test_modeling_compression_heating_air(self):
        """Adiabatic compression heats gas: T2_air > T1"""
        if self.T2_air is None or self.T1 is None:
            return (False, "SKIP: missing T2_air or T1")
        if self.T2_air > self.T1:
            return (True, f"T2_air ({self.T2_air}) > T1 ({self.T1}) – correct")
        else:
            return (False, f"T2_air ({self.T2_air}) ≯ T1 ({self.T1}) – compression should increase temperature")

    def test_modeling_compression_heating_he(self):
        """Adiabatic compression heats gas: T2_he > T1"""
        if self.T2_he is None or self.T1 is None:
            return (False, "SKIP: missing T2_he or T1")
        if self.T2_he > self.T1:
            return (True, f"T2_he ({self.T2_he}) > T1 ({self.T1}) – correct")
        else:
            return (False, f"T2_he ({self.T2_he}) ≯ T1 ({self.T1}) – compression should increase temperature")

    def test_modeling_helium_higher_than_air(self):
        """Because gamma_he > gamma_air, T2_he should be greater than T2_air (same pressure ratio)"""
        if self.T2_air is None or self.T2_he is None:
            return (False, "SKIP: missing T2 values")
        if self.gamma_he <= self.gamma_air:
            return (False, "Gamma values reversed? γ_he should be > γ_air (5/3 > 7/5)")
        if self.T2_he > self.T2_air:
            return (True, f"T2_he ({self.T2_he}) > T2_air ({self.T2_air}) – consistent with higher gamma")
        else:
            return (False, f"T2_he ({self.T2_he}) ≯ T2_air ({self.T2_air}) – should be larger for helium")

    def test_modeling_ratio_near_ten(self):
        """Implicit check that pressure ratio is approximately 10 (as given in problem)"""
        if self.ratio is None:
            return (False, "SKIP: cannot compute ratio")
        if math.isclose(self.ratio, 10.0, rel_tol=0.05):
            return (True, f"Pressure ratio ≈ 10 (={self.ratio:.2f})")
        else:
            return (False, f"Pressure ratio = {self.ratio:.2f}, expected ≈ 10")

    # ------------------------------------------------------------------
    # Monotonicity test – explicitly perturb an input and compare perturbed output to baseline
    # ------------------------------------------------------------------
    def test_monotonicity_gamma(self):
        """
        Perturb gamma from air value (1.4) to helium value (5/3) and
        check that the final temperature predicted by the adiabatic relation increases.
        Baseline: T2 with gamma_air  →  perturbed: T2 with gamma_he.
        We re-derive both from givens (T1, ratio) to isolate the effect of gamma.
        """
        if self.T1 is None or self.ratio is None:
            return (False, "SKIP: missing T1 or ratio (needed to re-derive baseline)")
        # Baseline output with gamma_air
        exponent_air = (self.gamma_air - 1) / self.gamma_air
        baseline = self.T1 * (self.ratio ** exponent_air)
        # Perturbed output with gamma_he
        exponent_he = (self.gamma_he - 1) / self.gamma_he
        perturbed = self.T1 * (self.ratio ** exponent_he)
        # Check monotonicity: perturbed > baseline because gamma_he > gamma_air
        if perturbed > baseline:
            return (True, f"Perturbed T2 (γ=5/3) = {perturbed:.2f} > baseline T2 (γ=7/5) = {baseline:.2f} – monotonic")
        else:
            return (False, f"Perturbed T2 ({perturbed:.2f}) ≤ baseline T2 ({baseline:.2f}) – monotonicity violated")

    # ------------------------------------------------------------------
    # Answer sanity – weak, no isclose, no re-derivation of final answer
    # ------------------------------------------------------------------
    def test_answer_sanity_range(self):
        """
        Weak sanity: T2 values should be between 300 K and 2000 K,
        finite, and within a plausible range for 10:1 compression.
        """
        if self.T2_air is None or self.T2_he is None:
            return (False, "SKIP: T2 values missing")
        # Check finiteness
        if not math.isfinite(self.T2_air) or not math.isfinite(self.T2_he):
            return (False, "One of T2 values is not finite")
        # Check reasonable range: compression from 300K to 10atm -> T2 should be a few hundred K
        min_val = 300
        max_val = 2000
        if not (min_val < self.T2_air < max_val):
            return (False, f"T2_air={self.T2_air} out of reasonable range ({min_val}, {max_val})")
        if not (min_val < self.T2_he < max_val):
            return (False, f"T2_he={self.T2_he} out of reasonable range ({min_val}, {max_val})")
        # Order of magnitude: target answers are ~5.8e2 and 7.5e2, so within 300-1000 is fine
        if not (300 < self.T2_air < 1000 and 300 < self.T2_he < 1000):
            return (False, "T2 values are unexpectedly high (>1000 K) for 10:1 compression")
        return (True, "T2 values are finite, positive, and within plausible range")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all test methods
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
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)