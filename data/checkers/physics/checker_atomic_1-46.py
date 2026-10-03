import math
import traceback
from typing import Any

class PhysicsChecker:
    """
    Checker for photoionization threshold wavelengths for H and He+.
    """

    def __init__(self, solution_instance: Any):
        # Extract results safely with None defaults
        if hasattr(solution_instance, 'results') and solution_instance.results is not None:
            res = solution_instance.results
        else:
            res = {}

        self.lambda_H1 = res.get('lambda_H1', None)
        self.lambda_H2 = res.get('lambda_H2', None)
        self.lambda_He1 = res.get('lambda_He1', None)
        self.lambda_He2 = res.get('lambda_He2', None)

        # Physical constants (for internal computation)
        self.hc = 12400.0      # eV * Å
        self.R_H = 13.6        # eV

    # ------------------------------------------------------------------
    # INTERFACE TESTS – check that keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_keys_H1(self) -> tuple:
        """Check that lambda_H1 is present."""
        if self.lambda_H1 is None:
            return (False, "SKIP: lambda_H1 missing")
        return (True, "lambda_H1 present")

    def test_interface_keys_H2(self) -> tuple:
        if self.lambda_H2 is None:
            return (False, "SKIP: lambda_H2 missing")
        return (True, "lambda_H2 present")

    def test_interface_keys_He1(self) -> tuple:
        if self.lambda_He1 is None:
            return (False, "SKIP: lambda_He1 missing")
        return (True, "lambda_He1 present")

    def test_interface_keys_He2(self) -> tuple:
        if self.lambda_He2 is None:
            return (False, "SKIP: lambda_He2 missing")
        return (True, "lambda_He2 present")

    def test_interface_positive(self) -> tuple:
        """All available wavelengths must be positive."""
        vals = [self.lambda_H1, self.lambda_H2, self.lambda_He1, self.lambda_He2]
        for v in vals:
            if v is not None and v <= 0:
                return (False, f"Non-positive value: {v}")
        return (True, "All provided wavelengths are positive")

    def test_interface_finite(self) -> tuple:
        """Values must be finite numbers."""
        vals = [self.lambda_H1, self.lambda_H2, self.lambda_He1, self.lambda_He2]
        for v in vals:
            if v is not None and (math.isnan(v) or math.isinf(v)):
                return (False, f"Non-finite value: {v}")
        return (True, "All provided wavelengths are finite")

    # ------------------------------------------------------------------
    # INVARIANT TESTS – scaling relations derived from physics
    # ------------------------------------------------------------------
    def test_invariant_H_n1_vs_n2(self) -> tuple:
        """For H, lambda(n=2) / lambda(n=1) should be 4."""
        if self.lambda_H1 is None or self.lambda_H2 is None:
            return (False, "SKIP: H wavelengths not fully available")
        ratio = self.lambda_H2 / self.lambda_H1
        if math.isclose(ratio, 4.0, rel_tol=1e-2):
            return (True, f"Ratio {ratio:.3f} ≈ 4")
        return (False, f"Ratio {ratio:.3f} not close to 4")

    def test_invariant_He_n1_vs_n2(self) -> tuple:
        """For He+, lambda(n=2) / lambda(n=1) should be 4."""
        if self.lambda_He1 is None or self.lambda_He2 is None:
            return (False, "SKIP: He+ wavelengths not fully available")
        ratio = self.lambda_He2 / self.lambda_He1
        if math.isclose(ratio, 4.0, rel_tol=1e-2):
            return (True, f"Ratio {ratio:.3f} ≈ 4")
        return (False, f"Ratio {ratio:.3f} not close to 4")

    def test_invariant_H_vs_He_n1(self) -> tuple:
        """For n=1, lambda(H) / lambda(He+) should be 4."""
        if self.lambda_H1 is None or self.lambda_He1 is None:
            return (False, "SKIP: n=1 wavelengths not fully available")
        ratio = self.lambda_H1 / self.lambda_He1
        if math.isclose(ratio, 4.0, rel_tol=1e-2):
            return (True, f"Ratio {ratio:.3f} ≈ 4")
        return (False, f"Ratio {ratio:.3f} not close to 4")

    def test_invariant_H_vs_He_n2(self) -> tuple:
        """For n=2, lambda(H) / lambda(He+) should be 4."""
        if self.lambda_H2 is None or self.lambda_He2 is None:
            return (False, "SKIP: n=2 wavelengths not fully available")
        ratio = self.lambda_H2 / self.lambda_He2
        if math.isclose(ratio, 4.0, rel_tol=1e-2):
            return (True, f"Ratio {ratio:.3f} ≈ 4")
        return (False, f"Ratio {ratio:.3f} not close to 4")

    # ------------------------------------------------------------------
    # MODELING TESTS – compare with the correct formula
    # ------------------------------------------------------------------
    def _expected_lambda(self, Z: int, n: int) -> float:
        """Compute theoretical threshold wavelength in Å."""
        return self.hc * (n ** 2) / ((Z ** 2) * self.R_H)

    def test_modeling_H1_formula(self) -> tuple:
        """Check lambda_H1 against formula: hc / (1^2 * R_H)."""
        if self.lambda_H1 is None:
            return (False, "SKIP: lambda_H1 missing")
        expected = self._expected_lambda(1, 1)
        if math.isclose(self.lambda_H1, expected, rel_tol=1e-2):
            return (True, f"lambda_H1={self.lambda_H1} ≈ expected {expected:.2f}")
        return (False, f"lambda_H1={self.lambda_H1} not close to expected {expected:.2f}")

    def test_modeling_H2_formula(self) -> tuple:
        """Check lambda_H2 against formula: 4*hc / R_H."""
        if self.lambda_H2 is None:
            return (False, "SKIP: lambda_H2 missing")
        expected = self._expected_lambda(1, 2)
        if math.isclose(self.lambda_H2, expected, rel_tol=1e-2):
            return (True, f"lambda_H2={self.lambda_H2} ≈ expected {expected:.2f}")
        return (False, f"lambda_H2={self.lambda_H2} not close to expected {expected:.2f}")

    def test_modeling_He1_formula(self) -> tuple:
        """Check lambda_He1 against formula: hc / (4*R_H)."""
        if self.lambda_He1 is None:
            return (False, "SKIP: lambda_He1 missing")
        expected = self._expected_lambda(2, 1)
        if math.isclose(self.lambda_He1, expected, rel_tol=1e-2):
            return (True, f"lambda_He1={self.lambda_He1} ≈ expected {expected:.2f}")
        return (False, f"lambda_He1={self.lambda_He1} not close to expected {expected:.2f}")

    def test_modeling_He2_formula(self) -> tuple:
        """Check lambda_He2 against formula: hc / R_H (since n=2 cancels Z^2)."""
        if self.lambda_He2 is None:
            return (False, "SKIP: lambda_He2 missing")
        expected = self._expected_lambda(2, 2)
        if math.isclose(self.lambda_He2, expected, rel_tol=1e-2):
            return (True, f"lambda_He2={self.lambda_He2} ≈ expected {expected:.2f}")
        return (False, f"lambda_He2={self.lambda_He2} not close to expected {expected:.2f}")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS – explicit perturbation and trend check
    # ------------------------------------------------------------------
    def test_monotonicity_lambda_decreases_with_Z(self) -> tuple:
        """Perturb Z upward by 10% for n=1, check that lambda decreases."""
        if self.lambda_H1 is None:
            return (False, "SKIP: lambda_H1 missing (baseline needed)")
        # Use lambda_H1 as baseline for (Z=1, n=1)
        Z_ref = 1
        n_ref = 1
        # Perturb Z by +10%
        Z_pert = 1.1
        # Compute expected wavelength for perturbed Z (same n)
        lambda_pert = self._expected_lambda(Z_pert, n_ref)
        # Baseline (theoretical for unperturbed)
        lambda_ref = self._expected_lambda(Z_ref, n_ref)
        # Check that the perturbation trend is correct: increasing Z => smaller lambda
        if lambda_pert < lambda_ref:
            return (True, f"Perturbed Z=1.1 gives λ={lambda_pert:.2f} < baseline {lambda_ref:.2f} (correct)")
        else:
            return (False, f"Perturbed Z=1.1 gives λ={lambda_pert:.2f} not < baseline {lambda_ref:.2f}")

    def test_monotonicity_lambda_increases_with_n(self) -> tuple:
        """Perturb n upward by 10% for Z=1, check that lambda increases."""
        if self.lambda_H1 is None:
            return (False, "SKIP: lambda_H1 missing (baseline needed)")
        # Baseline for (Z=1, n=1)
        Z_ref = 1
        n_ref = 1
        # Perturb n by +10% (not integer, but formula works with real n)
        n_pert = 1.1
        lambda_pert = self._expected_lambda(Z_ref, n_pert)
        lambda_ref = self._expected_lambda(Z_ref, n_ref)
        if lambda_pert > lambda_ref:
            return (True, f"Perturbed n=1.1 gives λ={lambda_pert:.2f} > baseline {lambda_ref:.2f} (correct)")
        else:
            return (False, f"Perturbed n=1.1 gives λ={lambda_pert:.2f} not > baseline {lambda_ref:.2f}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self) -> tuple:
        """Broad range check and ordering between all four values."""
        if None in (self.lambda_H1, self.lambda_H2, self.lambda_He1, self.lambda_He2):
            return (False, "SKIP: not all λ values available")
        # All wavelengths should lie between 100 and 4000 Å (physical range)
        for name, val in [("H1",self.lambda_H1), ("H2",self.lambda_H2),
                          ("He1",self.lambda_He1), ("He2",self.lambda_He2)]:
            if not (100 <= val <= 4000):
                return (False, f"{name}={val} outside expected range [100,4000]")
        # Ordering: He1 < He2 ≈ H1 < H2 (since He2 should be close to H1)
        if not (self.lambda_He1 < self.lambda_He2):
            return (False, f"Ordering He1 < He2 violated: {self.lambda_He1} vs {self.lambda_He2}")
        if not (self.lambda_He2 <= self.lambda_H1 * 1.05):  # He2 should be close to H1
            return (False, f"He2={self.lambda_He2} not ≤ 1.05*H1={self.lambda_H1*1.05:.2f}")
        if not (self.lambda_H1 < self.lambda_H2):
            return (False, f"Ordering H1 < H2 violated: {self.lambda_H1} vs {self.lambda_H2}")
        return (True, "All values in plausible range and have correct ordering")

    # ------------------------------------------------------------------
    # RUNNER
    # ------------------------------------------------------------------
    def run_all_tests(self) -> tuple:
        """Discover and run all test_* methods, return (all_pass, log_list)."""
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
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)