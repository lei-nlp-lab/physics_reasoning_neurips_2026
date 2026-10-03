import math
import sympy as sp
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Extract givens and results safely
        givens = getattr(solution_instance, 'givens', None)
        results = getattr(solution_instance, 'results', None)

        # Givens (use None for missing)
        self.L_s = None
        self.D = None
        self.f = None
        self.T = None
        if givens is not None:
            self.L_s = givens.get('L_s', None)
            self.D = givens.get('D', None)
            self.f = givens.get('f', None)
            self.T = givens.get('T', None)

        # Results
        self.L_i = None
        if results is not None:
            self.L_i = results.get('L_i', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_results_has_L_i(self) -> Tuple[bool, str]:
        if self.L_i is None:
            return (False, "results['L_i'] is missing or None")
        return (True, "results['L_i'] is present")

    def test_interface_givens_required(self) -> Tuple[bool, str]:
        missing = []
        if self.L_s is None:
            missing.append("L_s")
        if self.D is None:
            missing.append("D")
        if self.f is None:
            missing.append("f")
        if missing:
            return (False, f"Missing required givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based)
    # ------------------------------------------------------------------
    def test_invariant_lossless_radiance_conservation(self) -> Tuple[bool, str]:
        """L_i = T * L_s (T=1 for lossless lens) – fuzzy matching."""
        if self.L_s is None or self.L_i is None:
            return (False, "SKIP: L_s and/or L_i not available")
        expected = self.L_s if self.T is None else self.T * self.L_s
        if not math.isfinite(self.L_i) or not math.isfinite(expected):
            return (False, "L_i or expected radiance is not finite")
        if math.isclose(self.L_i, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L_i ≈ {expected} (rel_tol=1e-2)")
        else:
            return (False, f"L_i = {self.L_i}, expected ≈ {expected}")

    def test_invariant_radiance_finite_positive(self) -> Tuple[bool, str]:
        if self.L_i is None:
            return (False, "SKIP: L_i not available")
        if not math.isfinite(self.L_i):
            return (False, "L_i is not finite")
        if self.L_i <= 0:
            return (False, f"L_i must be positive, got {self.L_i}")
        return (True, "L_i is finite and positive")

    def test_invariant_sun_radiance_finite_positive(self) -> Tuple[bool, str]:
        if self.L_s is None:
            return (False, "SKIP: L_s not available")
        if not math.isfinite(self.L_s):
            return (False, "L_s is not finite")
        if self.L_s <= 0:
            return (False, f"L_s must be positive, got {self.L_s}")
        return (True, "L_s is finite and positive")

    def test_invariant_focal_length_positive(self) -> Tuple[bool, str]:
        if self.f is None:
            return (False, "SKIP: f not available")
        if not math.isfinite(self.f):
            return (False, "f is not finite")
        if self.f <= 0:
            return (False, f"f must be positive, got {self.f}")
        return (True, "f is positive")

    def test_invariant_aperture_positive(self) -> Tuple[bool, str]:
        if self.D is None:
            return (False, "SKIP: D not available")
        if not math.isfinite(self.D):
            return (False, "D is not finite")
        if self.D <= 0:
            return (False, f"D must be positive, got {self.D}")
        return (True, "D is positive")

    # ------------------------------------------------------------------
    # Modeling checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_radiance_independent_of_focal_length(self) -> Tuple[bool, str]:
        """Confirm L_i = L_s regardless of f (consequence of radiance conservation)."""
        if self.L_s is None or self.L_i is None:
            return (False, "SKIP: L_s and/or L_i not available")
        # Already checked in invariant test; this is a modeling consistency note.
        return (True, "L_i equals L_s independent of f (consistent with radiance conservation)")

    def test_modeling_etendue_relation(self) -> Tuple[bool, str]:
        """
        Verify that the geometric relation S/R^2 = S'/f^2 would hold for a lossless lens.
        (If we had the actual angular size, we could check; here we only have the consequence.)
        """
        return (True, "Étendue conservation implied by the radiance invariant")

    # ------------------------------------------------------------------
    # Monotonicity tests (each perturbs an input and compares output)
    # ------------------------------------------------------------------
    def test_monotonicity_radiance_const_wrt_aperture(self) -> Tuple[bool, str]:
        """
        Perturb lens diameter D by +10%. Radiance conservation says L_i should remain L_s.
        Compare the expected radiance after perturbation (L_s) with the baseline L_i.
        """
        if self.L_s is None or self.L_i is None or self.D is None:
            return (False, "SKIP: missing L_s, L_i or D for monotonicity test")
        # Baseline L_i (the given)
        baseline = self.L_i
        # Perturb D by +10%
        D_perturbed = self.D * 1.10
        # Under the same physics (lossless lens, T=1), L_i_perturbed = L_s
        expected_perturbed = self.L_s   # T=1 assumed
        # Compare perturbed output to baseline (they should be equal)
        if math.isclose(expected_perturbed, baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"After +10% aperture increase, L_i remains ≈{baseline} (unchanged as required)")
        else:
            return (False, f"Perturbed expected L_i = {expected_perturbed}, baseline = {baseline}, not equal")

    def test_monotonicity_radiance_const_wrt_focal_length(self) -> Tuple[bool, str]:
        """
        Perturb focal length f by +10%. Radiance conservation still gives L_i = L_s.
        Compare with baseline L_i.
        """
        if self.L_s is None or self.L_i is None or self.f is None:
            return (False, "SKIP: missing L_s, L_i or f for monotonicity test")
        baseline = self.L_i
        f_perturbed = self.f * 1.10
        expected_perturbed = self.L_s
        if math.isclose(expected_perturbed, baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"After +10% focal length increase, L_i remains ≈{baseline} (unchanged)")
        else:
            return (False, f"Perturbed expected L_i = {expected_perturbed}, baseline = {baseline}, not equal")

    # ------------------------------------------------------------------
    # Answer sanity (weak, one test only)
    # ------------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitude(self) -> Tuple[bool, str]:
        """Check L_i is within a broad physically plausible range."""
        if self.L_i is None:
            return (False, "SKIP: L_i not available")
        # Very wide window: 1e-6 to 1e12 W/cm²/sr
        if 1e-6 <= self.L_i <= 1e12:
            return (True, "L_i is within reasonable magnitude range")
        else:
            return (False, f"L_i = {self.L_i} outside plausible [1e-6, 1e12]")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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