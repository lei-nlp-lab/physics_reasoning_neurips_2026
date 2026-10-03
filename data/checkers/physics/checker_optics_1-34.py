import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely, defaulting to None for missing keys
        results = getattr(solution_instance, 'results', {}) or {}
        self.L_K = results.get('L_K', None)
        self.L_G = results.get('L_G', None)
        self.ratio = results.get('ratio', None)

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

    # ---------- Interface Tests ----------
    def test_interface_required_keys_exist(self):
        if self.L_K is None:
            return (False, "Missing key: L_K")
        if self.L_G is None:
            return (False, "Missing key: L_G")
        if self.ratio is None:
            return (False, "Missing key: ratio")
        return (True, "All required keys present")

    def test_interface_values_are_numeric(self):
        for name, val in [("L_K", self.L_K), ("L_G", self.L_G), ("ratio", self.ratio)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric, got {type(val).__name__}")
            if not math.isfinite(val):
                return (False, f"{name} is not finite (nan or inf)")
        return (True, "All values are numeric and finite")

    # ---------- Invariant Tests (theorem-based, use math.isclose) ----------
    def test_invariant_ratio_consistency(self):
        if any(v is None for v in [self.L_K, self.L_G, self.ratio]):
            return (False, "SKIP: missing one or more of L_K, L_G, ratio")
        if self.L_G == 0:
            return (False, "L_G is zero, cannot compute ratio")
        computed_ratio = self.L_K / self.L_G
        if not math.isclose(computed_ratio, self.ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ratio {self.ratio} diverges from L_K/L_G = {computed_ratio}")
        return (True, "ratio matches L_K/L_G within tolerance")

    def test_invariant_keplerian_longer(self):
        if self.L_K is None or self.L_G is None:
            return (False, "SKIP: missing L_K or L_G")
        if not (self.L_K > self.L_G):
            return (False, f"L_K ({self.L_K}) is not > L_G ({self.L_G})")
        return (True, f"L_K = {self.L_K} > L_G = {self.L_G} (consistent with M>1)")

    def test_invariant_ratio_greater_than_one(self):
        if self.ratio is None:
            return (False, "SKIP: ratio missing")
        if not (self.ratio > 1.0):
            return (False, f"ratio = {self.ratio} is not > 1")
        return (True, f"ratio = {self.ratio} > 1 (consistent with M>1)")

    def test_invariant_lengths_positive(self):
        if self.L_K is None or self.L_G is None:
            return (False, "SKIP: missing lengths")
        if self.L_K <= 0:
            return (False, f"L_K = {self.L_K} is not positive")
        if self.L_G <= 0:
            return (False, f"L_G = {self.L_G} is not positive (requires M>1)")
        return (True, "Both lengths are positive")

    def test_invariant_magnification_derivation(self):
        if any(v is None for v in [self.L_K, self.L_G, self.ratio]):
            return (False, "SKIP: missing data for derivation")
        if self.L_K == self.L_G:
            return (False, "L_K equals L_G, cannot derive M (would be infinite)")
        M_derived = (self.L_K + self.L_G) / (self.L_K - self.L_G)
        if M_derived <= 1:
            return (False, f"Derived M = {M_derived} is not > 1")
        expected_ratio = (M_derived + 1) / (M_derived - 1)
        if not math.isclose(self.ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ratio {self.ratio} differs from (M+1)/(M-1) = {expected_ratio} "
                           f"using M = {M_derived}")
        return (True, f"ratio consistent with derived M = {M_derived:.3f}")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_afocal_condition_keplerian(self):
        if any(v is None for v in [self.L_K, self.L_G]):
            return (False, "SKIP: missing lengths")
        if self.L_K == self.L_G:
            return (False, "Cannot compute f_o when L_K = L_G")
        f_o = (self.L_K + self.L_G) / 2.0
        f_e = (self.L_K - self.L_G) / 2.0
        if f_o <= 0 or f_e <= 0:
            return (False, f"Derived f_o = {f_o} or f_e = {f_e} is not positive")
        L_K_from_afocal = f_o + f_e
        if not math.isclose(self.L_K, L_K_from_afocal, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"L_K {self.L_K} does not equal f_o+f_e = {L_K_from_afocal}")
        return (True, f"L_K = f_o + f_e = {L_K_from_afocal} holds (derived f_o={f_o}, f_e={f_e})")

    def test_modeling_afocal_condition_galilean(self):
        if any(v is None for v in [self.L_K, self.L_G]):
            return (False, "SKIP: missing lengths")
        if self.L_K == self.L_G:
            return (False, "Cannot compute f_o when L_K = L_G")
        f_o = (self.L_K + self.L_G) / 2.0
        f_e = (self.L_K - self.L_G) / 2.0
        if f_o <= 0 or f_e <= 0:
            return (False, f"Derived f_o = {f_o} or f_e = {f_e} is not positive")
        L_G_from_afocal = f_o - f_e  # f_d = f_e
        if L_G_from_afocal <= 0:
            return (False, "Derived L_G from afocal condition is non-positive")
        if not math.isclose(self.L_G, L_G_from_afocal, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"L_G {self.L_G} does not equal f_o - f_d = {L_G_from_afocal}")
        return (True, f"L_G = f_o - f_d = {L_G_from_afocal} holds (derived f_o={f_o}, f_d=f_e={f_e})")

    # ---------- Monotonicity Test ----------
    def test_monotonicity_ratio_decreases_with_magnification(self):
        """
        Verify monotonic behavior: as M increases, ratio = (M+1)/(M-1) decreases.
        We derive M from L_K and L_G, then perturb M upward and check that the
        theoretical ratio decreases.
        """
        if any(v is None for v in [self.L_K, self.L_G, self.ratio]):
            return (False, "SKIP: missing L_K, L_G, or ratio for monotonicity check")
        if self.L_K == self.L_G:
            return (False, "L_K equals L_G, cannot derive M")
        # Baseline M from given lengths
        M_base = (self.L_K + self.L_G) / (self.L_K - self.L_G)
        if M_base <= 1:
            return (False, f"Derived baseline M = {M_base} is not > 1")
        # Compute baseline ratio using formula (should match self.ratio within tolerance)
        ratio_base = (M_base + 1) / (M_base - 1)
        if not math.isclose(ratio_base, self.ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Internal inconsistency: derived ratio {ratio_base} does not match stored ratio {self.ratio}")
        # Perturb M upward (e.g., 10% increase)
        factor = 1.1
        M_perturbed = M_base * factor
        ratio_perturbed = (M_perturbed + 1) / (M_perturbed - 1)
        # Check that perturbed ratio is smaller (monotonic decreasing)
        if not (ratio_perturbed < ratio_base):
            return (False, f"Expected ratio to decrease when M increases, but ratio_perturbed={ratio_perturbed:.4f} >= ratio_base={ratio_base:.4f}")
        # Also check that perturbed ratio > 1 (still positive)
        if not (ratio_perturbed > 1.0):
            return (False, f"Perturbed ratio = {ratio_perturbed:.4f} is not > 1 (unphysical)")
        return (True, f"Monotonicity holds: M increases from {M_base:.4f} to {M_perturbed:.4f}, ratio decreases from {ratio_base:.4f} to {ratio_perturbed:.4f}")

    # ---------- Weak Answer Sanity (no math.isclose, no re-derivation) ----------
    def test_answer_sanity_range(self):
        if self.ratio is None:
            return (False, "SKIP: ratio missing")
        if self.ratio <= 0:
            return (False, f"ratio = {self.ratio} is not positive")
        if self.ratio > 1e4 or not math.isfinite(self.ratio):
            return (False, f"ratio = {self.ratio} is unreasonably large or infinite")
        # Also check derived M if lengths available
        if self.L_K is not None and self.L_G is not None and self.L_K != self.L_G:
            M_test = (self.L_K + self.L_G) / (self.L_K - self.L_G)
            if not (M_test > 1):
                return (False, f"Derived M = {M_test} is not > 1 (sanity fail)")
        return (True, f"ratio = {self.ratio} is plausible (positive, finite, moderate)")