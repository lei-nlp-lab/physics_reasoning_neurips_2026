import math
import traceback
from typing import Any, Tuple

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Extract givens safely – try attributes first, then fallback to .givens dict
        self.I = getattr(solution_instance, 'I', None)
        if self.I is None and hasattr(solution_instance, 'givens'):
            self.I = solution_instance.givens.get('I', None)

        self.d = getattr(solution_instance, 'd', None)
        if self.d is None and hasattr(solution_instance, 'givens'):
            self.d = solution_instance.givens.get('d', None)

        self.r = getattr(solution_instance, 'r', None)
        if self.r is None and hasattr(solution_instance, 'givens'):
            self.r = solution_instance.givens.get('r', None)

        self.n = getattr(solution_instance, 'n', None)
        if self.n is None and hasattr(solution_instance, 'givens'):
            self.n = solution_instance.givens.get('n', None)

        self.e = getattr(solution_instance, 'e', None)
        if self.e is None and hasattr(solution_instance, 'givens'):
            self.e = solution_instance.givens.get('e', None)

        # If d is given, compute r
        if self.d is not None and self.r is None:
            self.r = self.d / 2.0

        # Extract result keys
        self.results = getattr(solution_instance, 'results', {})
        self.v_d_cm = self.results.get('v_d_cm', None)

        # Derived cross-sectional area (will be computed if r available)
        self.A = None
        if self.r is not None:
            self.A = math.pi * (float(self.r) ** 2)

    # ------------------------------------------------------------------
    # Interface check
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the result key `v_d_cm` exists and is a number."""
        if self.v_d_cm is None:
            return (False, "Missing required output: results['v_d_cm']")
        if not isinstance(self.v_d_cm, (int, float)):
            return (False, f"Invalid type for v_d_cm: {type(self.v_d_cm).__name__}")
        if math.isnan(self.v_d_cm) or math.isinf(self.v_d_cm):
            return (False, "v_d_cm is NaN or infinite")
        return (True, "All required keys present and valid")

    # ------------------------------------------------------------------
    # Invariant / theorem checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_drift_velocity_formula(self) -> Tuple[bool, str]:
        """Check that v_d_cm = (I * 100) / (n * A * e) with 1% tolerance."""
        if self.v_d_cm is None:
            return (False, "SKIP: v_d_cm not available")
        if self.I is None or self.n is None or self.e is None or self.A is None:
            return (False, "SKIP: missing givens (I, n, e, A) to verify formula")
        try:
            computed_v_d_ms = float(self.I) / (float(self.n) * float(self.A) * float(self.e))
            computed_v_d_cm = computed_v_d_ms * 100.0
            if not math.isclose(computed_v_d_cm, float(self.v_d_cm), rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"v_d_cm={self.v_d_cm:.6e} does not match expected {computed_v_d_cm:.6e}")
            return (True, "Drift velocity formula holds")
        except Exception as e:
            return (False, f"Error during formula verification: {e}")

    def test_invariant_positive_drift(self) -> Tuple[bool, str]:
        """Drift velocity must be positive (conventional direction)."""
        if self.v_d_cm is None:
            return (False, "SKIP: v_d_cm not available")
        if self.v_d_cm <= 0:
            return (False, f"v_d_cm is non-positive: {self.v_d_cm}")
        return (True, "Drift velocity is positive")

    # ------------------------------------------------------------------
    # Modeling quality checks (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_current_dependence(self) -> Tuple[bool, str]:
        """
        Verify that the model relation v_d = I/(n A e) reproduces the stored v_d_cm.
        This is a consistency check between the given parameters and the output.
        """
        if self.v_d_cm is None:
            return (False, "SKIP: no v_d_cm to check")
        if self.I is None or self.n is None or self.e is None or self.A is None:
            return (False, "SKIP: missing givens for recomputation")
        try:
            recalc_v_d_ms = float(self.I) / (float(self.n) * float(self.A) * float(self.e))
            recalc_v_d_cm = recalc_v_d_ms * 100.0
            if not math.isclose(recalc_v_d_cm, float(self.v_d_cm), rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Recomputed v_d_cm={recalc_v_d_cm:.6e} differs from {self.v_d_cm:.6e}")
            return (True, "Modeling consistent: current dependence correct")
        except Exception as e:
            return (False, f"Error in recomputation: {e}")

    def test_modeling_conversion_factor(self) -> Tuple[bool, str]:
        """
        Check that v_d_cm = 100 * v_d_ms, where v_d_ms is derived from the formula.
        This validates the unit conversion step in the modeling trace.
        """
        if self.v_d_cm is None:
            return (False, "SKIP: no v_d_cm to check")
        if self.I is None or self.n is None or self.e is None or self.A is None:
            return (False, "SKIP: missing givens for computing v_d in m/s")
        try:
            v_d_ms = float(self.I) / (float(self.n) * float(self.A) * float(self.e))
            expected_v_d_cm = v_d_ms * 100.0
            if not math.isclose(expected_v_d_cm, float(self.v_d_cm), rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Conversion factor incorrect: expected v_d_cm={expected_v_d_cm:.6e}, got {self.v_d_cm:.6e}")
            return (True, "Unit conversion from m/s to cm/s is correct (factor 100)")
        except Exception as e:
            return (False, f"Error during conversion check: {e}")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_current_increase_increases_v(self) -> Tuple[bool, str]:
        """
        Increase I by 5% and check that v_d_cm increases.
        """
        if self.I is None or self.n is None or self.e is None or self.A is None:
            return (False, "SKIP: missing givens for monotonicity check")
        if self.v_d_cm is None:
            return (False, "SKIP: no baseline v_d_cm")
        try:
            baseline = float(self.v_d_cm)
            perturbation_factor = 1.05
            I_pert = float(self.I) * perturbation_factor
            v_pert_ms = I_pert / (float(self.n) * float(self.A) * float(self.e))
            v_pert_cm = v_pert_ms * 100.0
            if v_pert_cm <= baseline:
                return (False, f"Increasing I by 5% should increase v_d, but v_pert={v_pert_cm:.6e} <= baseline={baseline:.6e}")
            return (True, "Monotonicity: increasing I increases v_d")
        except Exception as e:
            return (False, f"Error in monotonicity check: {e}")

    def test_monotonicity_radius_decrease_increases_v(self) -> Tuple[bool, str]:
        """
        Decrease radius by 10% (increases denominator A) should increase v_d.
        """
        if self.r is None or self.I is None or self.n is None or self.e is None:
            return (False, "SKIP: missing radius or other givens")
        if self.v_d_cm is None:
            return (False, "SKIP: no baseline v_d_cm")
        try:
            baseline = float(self.v_d_cm)
            r_pert = float(self.r) * 0.9
            A_pert = math.pi * r_pert ** 2
            v_pert_ms = float(self.I) / (float(self.n) * A_pert * float(self.e))
            v_pert_cm = v_pert_ms * 100.0
            if v_pert_cm <= baseline:
                return (False, f"Decreasing radius (r->0.9r) should increase v_d, but v_pert={v_pert_cm:.6e} <= baseline={baseline:.6e}")
            return (True, "Monotonicity: decreasing radius increases v_d")
        except Exception as e:
            return (False, f"Error in monotonicity check: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (ONLY ONE)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: v_d_cm must be finite, positive, and within a very broad range
        (here between 1e-6 and 1e6 cm/s). This does NOT compute expected value.
        """
        if self.v_d_cm is None:
            return (False, "SKIP: no v_d_cm to check")
        if math.isnan(self.v_d_cm) or math.isinf(self.v_d_cm):
            return (False, "v_d_cm is NaN or infinite")
        if self.v_d_cm <= 0:
            return (False, "v_d_cm is not positive")
        # Broad range: orders of magnitude away from plausible answers may indicate error
        if not (1e-6 <= self.v_d_cm <= 1e6):
            return (False, f"v_d_cm={self.v_d_cm:.6e} lies outside broad plausible range [1e-6, 1e6] cm/s")
        return (True, "Answer sanity: passes broad range and positivity")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
        logs = []
        all_pass = True
        test_methods = [name for name in dir(self) if name.startswith("test_") and callable(getattr(self, name))]
        for name in sorted(test_methods):
            method = getattr(self, name)
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)