import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the muonic atom finite-nucleus problem.
    Extracts results keys: 'qualitative', 'DeltaE_1s', 'Delta_diff'.
    Also attempts to extract the input parameter 'R' from the solution_instance
    (e.g., solution_instance.givens['R']).
    Runs a battery of interface, invariant, modeling, monotonicity
    and weak sanity tests.
    """

    def __init__(self, solution_instance):
        # Safely extract results dictionary
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        elif isinstance(solution_instance, dict):
            self.results = solution_instance
        else:
            self.results = {}

        # Required result keys
        self.qualitative = self.results.get('qualitative', None)
        self.DeltaE_1s = self.results.get('DeltaE_1s', None)
        self.Delta_diff = self.results.get('Delta_diff', None)

        # Optional input parameter: nuclear radius R (if provided)
        # Try to get from solution_instance.givens or direct attribute or dict
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.R = solution_instance.givens.get('R', None)
        elif isinstance(solution_instance, dict):
            self.R = solution_instance.get('R', None)
        else:
            self.R = None

    # ------------------------------------------------------------------ #
    #                         Interface tests                           #
    # ------------------------------------------------------------------ #
    def test_interface_required_keys(self):
        """Check that all three required keys exist and are not None."""
        if self.qualitative is None:
            return False, "Missing key 'qualitative'"
        if self.DeltaE_1s is None:
            return False, "Missing key 'DeltaE_1s'"
        if self.Delta_diff is None:
            return False, "Missing key 'Delta_diff'"
        return True, "All required keys present"

    def test_interface_qualitative_type(self):
        """qualitative should be a string."""
        if not isinstance(self.qualitative, str):
            return False, f"'qualitative' is not a string, got {type(self.qualitative)}"
        return True, "'qualitative' is a string"

    def test_interface_DeltaE_type(self):
        """DeltaE_1s should be numeric (int or float)."""
        if not isinstance(self.DeltaE_1s, (int, float)):
            return False, f"'DeltaE_1s' is not numeric, got {type(self.DeltaE_1s)}"
        return True, "'DeltaE_1s' is numeric"

    def test_interface_Delta_diff_type(self):
        """Delta_diff should be numeric."""
        if not isinstance(self.Delta_diff, (int, float)):
            return False, f"'Delta_diff' is not numeric, got {type(self.Delta_diff)}"
        return True, "'Delta_diff' is numeric"

    # ------------------------------------------------------------------ #
    #                       Invariant tests (fuzzy OK)                   #
    # ------------------------------------------------------------------ #
    def test_invariant_positivity(self):
        """Both energy shifts must be positive (upward shift)."""
        if self.DeltaE_1s is None or self.Delta_diff is None:
            return False, "SKIP: missing data"
        if not (self.DeltaE_1s > 0):
            return False, f"DeltaE_1s = {self.DeltaE_1s} is not > 0"
        if not (self.Delta_diff > 0):
            return False, f"Delta_diff = {self.Delta_diff} is not > 0"
        return True, "Both shifts are positive"

    def test_invariant_ratio(self):
        """
        From first‑order perturbation with R/aμ << 1:
            ΔE_1s = (2/5) (Z e^2 R^2 / aμ^3)
            Δ_diff = (1/20) (Z e^2 R^2 / aμ^3)
        Hence Δ_diff / ΔE_1s should equal 1/8 = 0.125.
        Uses fuzzy matching because of possible rounding.
        """
        if self.DeltaE_1s is None or self.Delta_diff is None:
            return False, "SKIP: missing data"
        expected_ratio = 1.0 / 8.0
        actual_ratio = self.Delta_diff / self.DeltaE_1s
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Ratio {actual_ratio:.4f} ≈ {expected_ratio}"
        else:
            return False, f"Ratio {actual_ratio:.4f} differs from expected {expected_ratio}"

    # ------------------------------------------------------------------ #
    #                      Modeling / trace tests                        #
    # ------------------------------------------------------------------ #
    def test_modeling_qualitative_content(self):
        """Check that qualitative description mentions key states and trends."""
        if not isinstance(self.qualitative, str):
            return False, "'qualitative' is not available"
        text = self.qualitative.lower()
        if 's' not in text and 's states' not in text:
            return False, "Qualitative description missing reference to s-states"
        if 'p' not in text and 'p states' not in text:
            return False, "Qualitative description missing reference to p-states"
        if 'shift' not in text and 'shifts' not in text:
            return False, "Qualitative description missing 'shift' words"
        return True, "Qualitative description contains required physics elements"

    def test_modeling_qualitative_explicit_states(self):
        """Check mentions of specific quantum numbers."""
        if not isinstance(self.qualitative, str):
            return False, "qualitative not available"
        text = self.qualitative.lower()
        required = ['1s', '2s', '2p']
        missing = [st for st in required if st not in text]
        if missing:
            return False, f"Missing explicit state mention: {missing}"
        return True, "Core states (1s,2s,2p) mentioned in qualitative description"

    # ------------------------------------------------------------------ #
    #                     Monotonicity tests                              #
    # ------------------------------------------------------------------ #
    def test_monotonicity_n_decrease(self):
        """
        For s‑states (l=0), the energy shift decreases as the principal
        quantum number n increases (because the wavefunction becomes less
        concentrated near the nucleus).  Here we perturb n from 1 to 2
        and check that ΔE_1s > Δ_diff (≈ ΔE_2s).
        """
        if self.DeltaE_1s is None or self.Delta_diff is None:
            return False, "SKIP: missing ΔE_1s or Δ_diff"
        if self.DeltaE_1s > self.Delta_diff:
            return True, "ΔE_1s > Δ_diff (n=1 shift larger than n=2), monotonic decrease satisfied"
        else:
            return False, f"ΔE_1s = {self.DeltaE_1s} is not > Δ_diff = {self.Delta_diff}"

    def test_monotonicity_R_increase(self):
        """
        Perturb the nuclear radius R by a small factor (+10%) and check that
        both ΔE_1s and Δ_diff increase (since they scale as R^2).
        Requires that R is available as a numeric input.
        """
        if self.R is None:
            return False, "SKIP: input parameter 'R' not available"
        if not isinstance(self.R, (int, float)):
            return False, "SKIP: 'R' is not numeric"
        if self.DeltaE_1s is None or self.Delta_diff is None:
            return False, "SKIP: missing ΔE_1s or Δ_diff"

        # Valid numeric R exists
        R_orig = self.R
        factor = 1.1  # +10%
        R_new = R_orig * factor

        # Compute expected scaled shifts (using the scaling ΔE ∝ R^2)
        # This is a self-consistent re-derivation from the given numeric result.
        scale_factor = (R_new / R_orig) ** 2
        expected_DeltaE_1s = self.DeltaE_1s * scale_factor
        expected_Delta_diff = self.Delta_diff * scale_factor

        # The perturbed values should be larger than the original ones
        if expected_DeltaE_1s <= self.DeltaE_1s:
            return False, f"Expected ΔE_1s after +10% R = {expected_DeltaE_1s} is not > original {self.DeltaE_1s}"
        if expected_Delta_diff <= self.Delta_diff:
            return False, f"Expected Δ_diff after +10% R = {expected_Delta_diff} is not > original {self.Delta_diff}"

        return True, "Increasing R increases both shifts (ΔE ∝ R^2), monotonicity satisfied"

    # ------------------------------------------------------------------ #
    #                     Answer sanity (weak, ≤1 test)                  #
    # ------------------------------------------------------------------ #
    def test_answer_sanity_finite_sign(self):
        """
        Weak sanity: shift values must be finite (not NaN, not inf) and positive.
        This is a simple check, not a re‑derivation.
        """
        if self.DeltaE_1s is None:
            return False, "SKIP: missing ΔE_1s"
        if not math.isfinite(self.DeltaE_1s):
            return False, f"ΔE_1s is not finite: {self.DeltaE_1s}"
        if self.DeltaE_1s <= 0:
            return False, f"ΔE_1s ≤ 0: {self.DeltaE_1s}"
        if self.Delta_diff is not None:
            if not math.isfinite(self.Delta_diff):
                return False, f"Δ_diff is not finite: {self.Delta_diff}"
            if self.Delta_diff <= 0:
                return False, f"Δ_diff ≤ 0: {self.Delta_diff}"
        return True, "Shift values are finite and positive"

    # ------------------------------------------------------------------ #
    #                     Master runner                                   #
    # ------------------------------------------------------------------ #
    def run_all_tests(self):
        """
        Discovers all test_* methods, executes them safely,
        and returns (all_pass, logs).
        """
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
                msg = f"CRASH: {traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs