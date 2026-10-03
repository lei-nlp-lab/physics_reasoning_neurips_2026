import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract the emitted result
        self.v_over_c = None
        results = getattr(solution_instance, 'results', None)
        if isinstance(results, dict):
            self.v_over_c = results.get('v_over_c', None)
        elif hasattr(solution_instance, 'v_over_c'):
            self.v_over_c = getattr(solution_instance, 'v_over_c', None)

        # Safely extract the atomic number Z (from givens, attribute, or fallback None)
        self.Z = None
        givens = getattr(solution_instance, 'givens', {})
        if isinstance(givens, dict):
            self.Z = givens.get('Z', None)
        elif hasattr(givens, 'Z'):
            self.Z = getattr(givens, 'Z', None)
        else:
            self.Z = getattr(solution_instance, 'Z', None)

        # Fine‑structure constant (well‑known physical constant, not taken from solution)
        self.alpha = 1.0 / 137.0

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_result_exists(self):
        if self.v_over_c is None:
            return (False, "SKIP: v_over_c not available in results")
        if not isinstance(self.v_over_c, (int, float)):
            return (False, f"v_over_c has type {type(self.v_over_c).__name__}, expected numeric")
        return (True, "v_over_c exists and is numeric")

    def test_interface_result_finite(self):
        if self.v_over_c is None:
            return (False, "SKIP: v_over_c not available")
        if not math.isfinite(self.v_over_c):
            return (False, f"v_over_c is not finite: {self.v_over_c}")
        return (True, "v_over_c is finite")

    def test_interface_Z_available(self):
        if self.Z is None:
            return (False, "SKIP: Z not available in givens or as attribute")
        if not isinstance(self.Z, (int, float)):
            return (False, f"Z has type {type(self.Z).__name__}, expected numeric")
        return (True, "Z is available and numeric")

    # ------------------------------------------------------------------
    # Invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_speed_less_than_c(self):
        if self.v_over_c is None:
            return (False, "SKIP: v_over_c not available")
        if self.v_over_c >= 1.0:
            return (False, f"v_over_c = {self.v_over_c} >= 1 (should be < 1)")
        return (True, f"v_over_c = {self.v_over_c} < 1, non‑relativistic")

    def test_invariant_fine_structure_relation(self):
        if self.v_over_c is None or self.Z is None:
            return (False, "SKIP: v_over_c or Z not available")
        expected = self.Z * self.alpha
        if math.isclose(self.v_over_c, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_over_c = {self.v_over_c} ≈ Z*alpha = {expected}")
        else:
            return (False, f"v_over_c = {self.v_over_c} does not match expected Z*alpha = {expected}")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_positive_speed(self):
        if self.v_over_c is None:
            return (False, "SKIP: v_over_c not available")
        if self.v_over_c <= 0:
            return (False, f"v_over_c = {self.v_over_c} is not positive")
        return (True, "v_over_c is positive")

    def test_modeling_scaling_with_Z(self):
        if self.v_over_c is None or self.Z is None:
            return (False, "SKIP: v_over_c or Z not available")
        if self.Z == 0:
            if self.v_over_c != 0:
                return (False, "For Z=0, v_over_c should be 0")
            else:
                return (True, "For Z=0, v_over_c=0")
        ratio = self.v_over_c / self.Z
        if math.isclose(ratio, self.alpha, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_over_c / Z = {ratio} ≈ alpha = {self.alpha}")
        else:
            return (False, f"v_over_c / Z = {ratio}, expected ≈ {self.alpha}")

    # ------------------------------------------------------------------
    # Monotonicity check (explicit perturbation of Z)
    # ------------------------------------------------------------------
    def test_monotonicity_increase_with_Z(self):
        """
        Because v/c = Z * alpha, increasing Z should increase v/c.
        We perturb Z to Z+1 and predict the new v/c.
        Since we cannot re‑run the solver, we use the derived proportionality.
        """
        if self.v_over_c is None or self.Z is None:
            return (False, "SKIP: v_over_c or Z not available")
        if self.Z <= 0:
            # For Z=0, v/c=0; monotonic increase still holds for Z>0.
            return (True, "Z non‑positive, monotonicity test not applicable")
        # Predict v/c for Z+1 using the same alpha
        v_over_c_increased = (self.Z + 1) * self.alpha
        # Check that the predicted value is greater than the current value
        if v_over_c_increased > self.v_over_c:
            return (True, f"Increasing Z from {self.Z} to {self.Z+1} increases v/c from {self.v_over_c} to {v_over_c_increased} (predicted)")
        else:
            return (False, f"Expected v/c to increase with Z but current v/c = {self.v_over_c} and predicted for Z+1 = {v_over_c_increased} is not larger")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_range(self):
        if self.v_over_c is None:
            return (False, "SKIP: v_over_c not available")
        if math.isnan(self.v_over_c):
            return (False, "v_over_c is NaN")
        if not (0.0 < self.v_over_c < 1.0):
            return (False, f"v_over_c = {self.v_over_c} not in (0,1)")
        return (True, "v_over_c lies in (0,1) and is finite")

    # ------------------------------------------------------------------
    # Test runner
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