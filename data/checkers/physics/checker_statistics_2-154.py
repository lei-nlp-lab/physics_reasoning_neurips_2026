import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the two-container helium gas problem.
    Checks regime identification and mean free path ratios under different flow regimes.
    """

    def __init__(self, solution_instance):
        # Safely extract all expected keys with None defaults
        self.results = getattr(solution_instance, 'results', {}) if hasattr(solution_instance, 'results') else {}
        self.givens = getattr(solution_instance, 'givens', {})

        # Expected emitted keys from pseudocode
        self.regime = self.results.get('regime', None)
        self.l1_over_l2_effusion = self.results.get('l1_over_l2_effusion', None)
        self.l1_over_l2_continuum = self.results.get('l1_over_l2_continuum', None)

        # Givens (if provided separately)
        self.T1 = self.givens.get('T1', None) if hasattr(self.givens, 'get') else None
        self.T2 = self.givens.get('T2', None) if hasattr(self.givens, 'get') else None

        # Also try to get from results if givens missing
        if self.T1 is None:
            self.T1 = self.results.get('T1', None)
        if self.T2 is None:
            self.T2 = self.results.get('T2', None)

    # ==========================================================
    # Interface tests
    # ==========================================================
    def test_interface_required_keys(self):
        """All three expected result keys must exist and be non-None."""
        missing = []
        if self.regime is None:
            missing.append('regime')
        if self.l1_over_l2_effusion is None:
            missing.append('l1_over_l2_effusion')
        if self.l1_over_l2_continuum is None:
            missing.append('l1_over_l2_continuum')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present.")

    def test_interface_regime_string(self):
        """Regime must be a non-empty string containing 'effusion' and 'continuum'."""
        if self.regime is None:
            return (False, "SKIP: regime not available")
        if not isinstance(self.regime, str):
            return (False, f"regime must be string, got {type(self.regime).__name__}")
        if not self.regime.strip():
            return (False, "regime string is empty")
        if 'effusion' not in self.regime.lower() and 'continuum' not in self.regime.lower():
            return (False, "regime description should mention effusion and continuum")
        return (True, "Regime string seems reasonable.")

    # ==========================================================
    # Invariant tests (theorem-based, use isclose)
    # ==========================================================
    def test_invariant_effusion_ratio_formula(self):
        """In effusion regime, l1/l2 = sqrt(T1/T2)."""
        if self.l1_over_l2_effusion is None:
            return (False, "SKIP: l1_over_l2_effusion not available")
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 not available for reference calculation")
        try:
            expected = math.sqrt(self.T1 / self.T2)
        except (ZeroDivisionError, ValueError):
            return (False, "Invalid T1/T2")
        if abs(self.T2) < 1e-12:
            return (False, "T2 too close to zero")
        actual = self.l1_over_l2_effusion
        if not math.isfinite(expected) or not math.isfinite(actual):
            return (False, "Non-finite value encountered")
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Effusion ratio {actual:.4f} ~ sqrt(T1/T2) = {expected:.4f}")
        else:
            return (False, f"Effusion ratio {actual:.4f} differs from sqrt(T1/T2) = {expected:.4f}")

    def test_invariant_continuum_ratio_formula(self):
        """In continuum regime, l1/l2 = T1/T2."""
        if self.l1_over_l2_continuum is None:
            return (False, "SKIP: l1_over_l2_continuum not available")
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 not available")
        try:
            expected = self.T1 / self.T2
        except ZeroDivisionError:
            return (False, "T2 is zero")
        if abs(self.T2) < 1e-12:
            return (False, "T2 too close to zero")
        actual = self.l1_over_l2_continuum
        if not math.isfinite(expected) or not math.isfinite(actual):
            return (False, "Non-finite value encountered")
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Continuum ratio {actual:.4f} ~ T1/T2 = {expected:.4f}")
        else:
            return (False, f"Continuum ratio {actual:.4f} differs from T1/T2 = {expected:.4f}")

    # ==========================================================
    # Modeling / trace tests
    # ==========================================================
    def test_modeling_effusion_ratio_less_than_one(self):
        """Since T1=150K, T2=300K, effusion ratio should be <1."""
        if self.l1_over_l2_effusion is None:
            return (False, "SKIP: l1_over_l2_effusion not available")
        if not isinstance(self.l1_over_l2_effusion, (int, float)):
            return (False, "Effusion ratio is not numeric")
        if self.l1_over_l2_effusion < 1.0:
            return (True, f"Effusion ratio {self.l1_over_l2_effusion:.4f} < 1 as expected")
        else:
            return (False, f"Effusion ratio {self.l1_over_l2_effusion:.4f} should be <1")

    def test_modeling_continuum_ratio_less_than_effusion(self):
        """Continuum ratio = 0.5, effusion ~0.707; continuum should be smaller."""
        if self.l1_over_l2_continuum is None or self.l1_over_l2_effusion is None:
            return (False, "SKIP: one of the ratios missing")
        if not isinstance(self.l1_over_l2_continuum, (int, float)) or not isinstance(self.l1_over_l2_effusion, (int, float)):
            return (False, "Ratios are not numeric")
        if self.l1_over_l2_continuum < self.l1_over_l2_effusion:
            return (True, "Continuum ratio < effusion ratio as expected")
        else:
            return (False, "Continuum ratio should be smaller than effusion ratio")

    def test_modeling_regime_knudsen_logic(self):
        """Check that regime description correctly contrasts D<<l vs D>>l."""
        if self.regime is None:
            return (False, "SKIP: regime not available")
        has_large_D = 'D >>' in self.regime or 'macroscopic' in self.regime
        has_small_D = 'D <<' in self.regime or 'leakage' in self.regime or 'pinhole' in self.regime
        if has_large_D and has_small_D:
            return (True, "Regime description mentions both macroscopic flow and leakage/effusion")
        else:
            return (False, "Regime description should contrast D>>l and D<<l cases")

    # ==========================================================
    # Monotonicity tests (perturb an input and compare)
    # ==========================================================
    def test_monotonicity_effusion_increases_with_T1(self):
        """In effusion regime, l1/l2 = sqrt(T1/T2) increases if T1 increases."""
        if self.l1_over_l2_effusion is None:
            return (False, "SKIP: l1_over_l2_effusion not available")
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 not available")
        if not math.isfinite(self.T1) or not math.isfinite(self.T2) or abs(self.T2) < 1e-12:
            return (False, "Invalid T1/T2 values")
        # Baseline ratio computed from formula (should match solution)
        baseline = math.sqrt(self.T1 / self.T2)
        # Perturb T1 upward by 10%
        perturbed_T1 = self.T1 * 1.10
        perturbed_ratio = math.sqrt(perturbed_T1 / self.T2)
        # Since perturbed_T1 > self.T1, perturbed_ratio should be > baseline
        if perturbed_ratio > baseline:
            return (True, f"Effusion ratio increases when T1 increases: baseline={baseline:.4f}, perturbed={perturbed_ratio:.4f}")
        else:
            return (False, "Effusion ratio should increase with T1 (perturbed ratio not larger)")

    def test_monotonicity_continuum_increases_with_T1(self):
        """In continuum regime, l1/l2 = T1/T2 increases if T1 increases."""
        if self.l1_over_l2_continuum is None:
            return (False, "SKIP: l1_over_l2_continuum not available")
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 not available")
        if not math.isfinite(self.T1) or not math.isfinite(self.T2) or abs(self.T2) < 1e-12:
            return (False, "Invalid T1/T2 values")
        baseline = self.T1 / self.T2
        perturbed_T1 = self.T1 * 1.10
        perturbed_ratio = perturbed_T1 / self.T2
        if perturbed_ratio > baseline:
            return (True, f"Continuum ratio increases when T1 increases: baseline={baseline:.4f}, perturbed={perturbed_ratio:.4f}")
        else:
            return (False, "Continuum ratio should increase with T1 (perturbed ratio not larger)")

    # ==========================================================
    # Answer sanity tests (weak, no isclose, no re-derivation)
    # ==========================================================
    def test_answer_sanity_ratio_ranges(self):
        """Both ratios should be in plausible ranges, given T1<T2."""
        issues = []
        # Effusion ratio should be between 0.5 and 1.0 (since sqrt(150/300)=0.707)
        if self.l1_over_l2_effusion is not None and isinstance(self.l1_over_l2_effusion, (int, float)):
            if not (0.5 <= self.l1_over_l2_effusion <= 1.0):
                issues.append(f"effusion ratio {self.l1_over_l2_effusion:.4f} out of [0.5,1.0]")
        else:
            issues.append("effusion ratio missing or non-numeric")
        # Continuum ratio should be between 0 and 1, and likely around 0.5
        if self.l1_over_l2_continuum is not None and isinstance(self.l1_over_l2_continuum, (int, float)):
            if not (0.0 < self.l1_over_l2_continuum < 1.0):
                issues.append(f"continuum ratio {self.l1_over_l2_continuum:.4f} out of (0,1)")
        else:
            issues.append("continuum ratio missing or non-numeric")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Both ratios in plausible ranges")

    # ==========================================================
    # Main test runner (do not modify)
    # ==========================================================
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