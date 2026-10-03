import math
import traceback

class PhysicsChecker:
    """
    Autograder for canonical ensemble problem.
    Checks interface, invariants (theorem-based), modeling/trace, monotonicity,
    and one weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safe extraction of required result keys
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}

        self.E_bar = results.get('E_bar', None)
        self.E2_bar = results.get('E2_bar', None)
        self.DeltaE2_bar = results.get('DeltaE2_bar', None)
        self.C_V = results.get('C_V', None)
        self.DeltaE_rms = results.get('DeltaE_rms', None)
        self.ratio = results.get('ratio', None)

        # Additional parameters (optional)
        givens = getattr(solution_instance, 'givens', None) or {}
        self.N = results.get('N', None) or givens.get('N', None)
        self.k = results.get('k', None) or givens.get('k', None)
        self.T = results.get('T', None) or givens.get('T', None)

        self.solution_instance = solution_instance

    # ------------------------------------------------------------------
    # Interface checks (one per key)
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """All six required result keys must exist and be not None."""
        keys = ['E_bar', 'E2_bar', 'DeltaE2_bar', 'C_V', 'DeltaE_rms', 'ratio']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_E_bar_numeric(self):
        if self.E_bar is None:
            return (False, "SKIP: E_bar missing")
        if not isinstance(self.E_bar, (int, float)):
            return (False, f"E_bar not numeric: {type(self.E_bar)}")
        return (True, "E_bar is numeric")

    def test_interface_E2_bar_numeric(self):
        if self.E2_bar is None:
            return (False, "SKIP: E2_bar missing")
        if not isinstance(self.E2_bar, (int, float)):
            return (False, f"E2_bar not numeric: {type(self.E2_bar)}")
        return (True, "E2_bar is numeric")

    def test_interface_DeltaE2_bar_numeric(self):
        if self.DeltaE2_bar is None:
            return (False, "SKIP: DeltaE2_bar missing")
        if not isinstance(self.DeltaE2_bar, (int, float)):
            return (False, f"DeltaE2_bar not numeric: {type(self.DeltaE2_bar)}")
        return (True, "DeltaE2_bar is numeric")

    def test_interface_C_V_numeric(self):
        if self.C_V is None:
            return (False, "SKIP: C_V missing")
        if not isinstance(self.C_V, (int, float)):
            return (False, f"C_V not numeric: {type(self.C_V)}")
        return (True, "C_V is numeric")

    def test_interface_DeltaE_rms_numeric(self):
        if self.DeltaE_rms is None:
            return (False, "SKIP: DeltaE_rms missing")
        if not isinstance(self.DeltaE_rms, (int, float)):
            return (False, f"DeltaE_rms not numeric: {type(self.DeltaE_rms)}")
        return (True, "DeltaE_rms is numeric")

    def test_interface_ratio_numeric(self):
        if self.ratio is None:
            return (False, "SKIP: ratio missing")
        if not isinstance(self.ratio, (int, float)):
            return (False, f"ratio not numeric: {type(self.ratio)}")
        return (True, "ratio is numeric")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_dispersion_definition(self):
        """DeltaE2_bar must equal E2_bar - E_bar^2 (rel/abs 1e-5)."""
        if any(v is None for v in [self.E_bar, self.E2_bar, self.DeltaE2_bar]):
            return (False, "SKIP: missing E_bar, E2_bar, or DeltaE2_bar")
        expected = self.E2_bar - self.E_bar**2
        if not math.isclose(self.DeltaE2_bar, expected, rel_tol=1e-5, abs_tol=1e-5):
            return (False,
                    f"DeltaE2_bar = {self.DeltaE2_bar}, "
                    f"E2_bar - E_bar^2 = {expected}")
        return (True, "Dispersion satisfies DeltaE2_bar = E2_bar - E_bar^2")

    def test_invariant_rms_vs_variance(self):
        """DeltaE_rms must be sqrt(DeltaE2_bar) (rel/abs 1e-2)."""
        if self.DeltaE_rms is None or self.DeltaE2_bar is None:
            return (False, "SKIP: missing DeltaE_rms or DeltaE2_bar")
        expected = math.sqrt(self.DeltaE2_bar)
        if not math.isclose(self.DeltaE_rms, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"DeltaE_rms = {self.DeltaE_rms}, "
                    f"sqrt(DeltaE2_bar) = {expected}")
        return (True, "RMS dispersion equals sqrt of variance")

    def test_invariant_C_V_positive(self):
        """Heat capacity must be positive."""
        if self.C_V is None:
            return (False, "SKIP: C_V missing")
        if self.C_V <= 0:
            return (False, f"C_V = {self.C_V} is not > 0")
        return (True, "C_V > 0")

    def test_invariant_ratio_ideal_gas(self):
        """If N is available, ratio must equal sqrt(2/(3N)) (rel/abs 1e-2)."""
        if self.ratio is None:
            return (False, "SKIP: ratio missing")
        if self.N is None:
            return (False, "SKIP: N not available")
        N = self.N
        expected = math.sqrt(2.0 / (3.0 * N))
        if not math.isclose(self.ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"ratio = {self.ratio}, expected sqrt(2/(3N)) = {expected} "
                    f"with N={N}")
        return (True, f"Ratio matches ideal gas formula with N={N}")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_energy_positive(self):
        """Average energy should be positive."""
        if self.E_bar is None:
            return (False, "SKIP: E_bar missing")
        if self.E_bar <= 0:
            return (False, f"E_bar = {self.E_bar} is not > 0")
        return (True, "E_bar > 0")

    def test_modeling_second_moment_positive(self):
        """Second moment E2_bar should be positive."""
        if self.E2_bar is None:
            return (False, "SKIP: E2_bar missing")
        if self.E2_bar <= 0:
            return (False, f"E2_bar = {self.E2_bar} is not > 0")
        return (True, "E2_bar > 0")

    def test_modeling_dispersion_nonnegative(self):
        """Dispersion must be nonnegative."""
        if self.DeltaE2_bar is None:
            return (False, "SKIP: DeltaE2_bar missing")
        if self.DeltaE2_bar < 0:
            return (False, f"DeltaE2_bar = {self.DeltaE2_bar} < 0")
        return (True, "DeltaE2_bar >= 0")

    # ------------------------------------------------------------------
    # Monotonicity check (perturb N, verify ratio decreases)
    # ------------------------------------------------------------------
    def test_monotonicity_ratio_decreases_with_N(self):
        """
        For ideal gas, ratio ~ sqrt(2/(3N)).
        Perturb N by +5% and verify that the resulting ratio (computed from
        the formula) is smaller than the solution's ratio (baseline).
        """
        if self.ratio is None or self.N is None:
            return (False, "SKIP: ratio or N missing")
        N0 = self.N
        # Baseline ratio from solution (or compute from formula; here we use
        # the solution's value as the baseline output)
        ratio0 = self.ratio
        # Perturb N upward by 5%
        eps = 0.05
        N1 = N0 * (1 + eps)
        # Recompute expected ratio for perturbed N using ideal gas formula
        ratio1 = math.sqrt(2.0 / (3.0 * N1))
        # Since N increased, ratio should decrease
        if ratio1 >= ratio0:
            return (False,
                    f"Perturbing N from {N0} to {N1} gives ratio {ratio1}, "
                    f"which is not less than baseline {ratio0}. "
                    f"Expected monotonic decrease.")
        return (True, f"Ratio decreases from {ratio0:.4f} to {ratio1:.4f} "
                      f"when N increases by {eps*100:.0f}%")

    # ------------------------------------------------------------------
    # Single weak answer sanity (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausibility(self):
        """All numeric results must be finite and physically plausible."""
        quantities = {
            'E_bar': self.E_bar,
            'E2_bar': self.E2_bar,
            'DeltaE2_bar': self.DeltaE2_bar,
            'C_V': self.C_V,
            'DeltaE_rms': self.DeltaE_rms,
            'ratio': self.ratio
        }
        for name, val in quantities.items():
            if val is None:
                continue
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")
            # Basic sign checks (non‑negative for variance, C_V, rms, ratio)
            if name in ('DeltaE2_bar', 'C_V', 'DeltaE_rms', 'ratio'):
                if val < 0:
                    return (False, f"{name} = {val} is negative")
            # For E_bar, E2_bar we already have positivity in modeling tests
        # Also check that ratio is less than 1 (for N>=1)
        if self.ratio is not None and self.ratio >= 1:
            return (False, f"ratio = {self.ratio} should be < 1 for N>=1")
        return (True, "All quantities are finite and pass basic plausibility checks")

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with test_
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