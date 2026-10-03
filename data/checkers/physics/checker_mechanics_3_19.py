import math
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Auditor for the two-particle relativistic velocity problem.
    Checks interface, invariants, modeling consistency, and monotonicity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens and results with None defaults.
        self.m = getattr(solution_instance, 'm', None)
        self.c = getattr(solution_instance, 'c', None)  # never numeric fallback

        # Results (expected keys from pseudocode [emit] contract)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.v_A = solution_instance.results.get('v_A', None)
            self.v_B = solution_instance.results.get('v_B', None)
            self.w_fs = solution_instance.results.get('w_fs', None)
            self.w_sf = solution_instance.results.get('w_sf', None)
        else:
            self.v_A = None
            self.v_B = None
            self.w_fs = None
            self.w_sf = None

        # Convert string representations to float (e.g., "0.595c" -> 0.595)
        for attr in ['v_A', 'v_B', 'w_fs', 'w_sf']:
            val = getattr(self, attr)
            if isinstance(val, str):
                try:
                    cleaned = val.replace('c', '').strip()
                    setattr(self, attr, float(cleaned))
                except (ValueError, TypeError):
                    pass  # keep as is (likely None)
            elif val is not None:
                try:
                    setattr(self, attr, float(val))
                except (ValueError, TypeError):
                    pass

        # Problem constants (not from solution instance)
        self.p_over_mc_A = 5.0   # p_A = 5 m c
        self.p_over_mc_B = 10.0  # p_B = 10 m c

    # ----------------------------------------------------------------------
    # Helper: effective c (1.0 if not provided)
    # ----------------------------------------------------------------------
    def _c_effective(self) -> float:
        return self.c if self.c is not None else 1.0

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """All four result keys must exist and be non-None."""
        if None in (self.v_A, self.v_B, self.w_fs, self.w_sf):
            missing = []
            if self.v_A is None:
                missing.append('v_A')
            if self.v_B is None:
                missing.append('v_B')
            if self.w_fs is None:
                missing.append('w_fs')
            if self.w_sf is None:
                missing.append('w_sf')
            return False, f"Missing result keys: {', '.join(missing)}"
        return True, "All required result keys present"

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ----------------------------------------------------------------------
    def test_invariant_velocities_less_than_c(self) -> Tuple[bool, str]:
        """All velocities must be less than c (magnitude < 0.9999c)."""
        if any(v is None for v in (self.v_A, self.v_B, self.w_fs, self.w_sf)):
            return False, "SKIP: velocities not all available"
        c_eff = self._c_effective()
        for name, val in [('v_A', self.v_A), ('v_B', self.v_B),
                          ('w_fs', self.w_fs), ('w_sf', self.w_sf)]:
            if abs(val) >= 0.9999 * c_eff:
                return False, f"{name} = {val} is not less than c (≥ 0.9999c)"
        return True, "All velocities are < c"

    def test_invariant_relative_velocity_negation(self) -> Tuple[bool, str]:
        """Relativistic relative speeds are exact negatives: w_sf = -w_fs."""
        if self.w_fs is None or self.w_sf is None:
            return False, "SKIP: w_fs or w_sf not available"
        if not math.isclose(self.w_fs, -self.w_sf, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"w_fs = {self.w_fs}, w_sf = {self.w_sf}, expected w_sf ≈ {-self.w_fs}"
        return True, "w_sf = -w_fs within tolerance"

    # ----------------------------------------------------------------------
    # Modeling (derived consistency) checks
    # ----------------------------------------------------------------------
    def test_modeling_v_A_from_momentum(self) -> Tuple[bool, str]:
        """v_A should equal c * (p/(mc)) / sqrt(1 + (p/(mc))^2) = 5/sqrt(26) * c."""
        if self.v_A is None:
            return False, "SKIP: v_A not available"
        c_eff = self._c_effective()
        expected = 5.0 / math.sqrt(26.0) * c_eff
        if not math.isclose(self.v_A, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"v_A = {self.v_A}, expected {expected}"
        return True, "v_A consistent with momentum p=5mc"

    def test_modeling_v_B_from_momentum(self) -> Tuple[bool, str]:
        """v_B should equal 10/sqrt(101) * c."""
        if self.v_B is None:
            return False, "SKIP: v_B not available"
        c_eff = self._c_effective()
        expected = 10.0 / math.sqrt(101.0) * c_eff
        if not math.isclose(self.v_B, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"v_B = {self.v_B}, expected {expected}"
        return True, "v_B consistent with momentum p=10mc"

    def test_modeling_relativistic_addition_w_fs(self) -> Tuple[bool, str]:
        """Compute w_fs from v_A and v_B using relativistic addition formula and compare."""
        if None in (self.v_A, self.v_B, self.w_fs):
            return False, "SKIP: v_A, v_B or w_fs missing"
        c_eff = self._c_effective()
        numerator = self.v_B - self.v_A
        denominator = 1.0 - (self.v_B * self.v_A) / (c_eff ** 2)
        if denominator == 0:
            return False, "Denominator is zero in w_fs formula"
        expected = numerator / denominator
        if not math.isclose(self.w_fs, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"w_fs = {self.w_fs}, expected {expected}"
        return True, "w_fs consistent with relativistic addition formula"

    def test_modeling_relativistic_addition_w_sf(self) -> Tuple[bool, str]:
        """Compute w_sf from v_A and v_B using relativistic addition formula and compare."""
        if None in (self.v_A, self.v_B, self.w_sf):
            return False, "SKIP: v_A, v_B or w_sf missing"
        c_eff = self._c_effective()
        numerator = self.v_A - self.v_B
        denominator = 1.0 - (self.v_A * self.v_B) / (c_eff ** 2)
        if denominator == 0:
            return False, "Denominator is zero in w_sf formula"
        expected = numerator / denominator
        if not math.isclose(self.w_sf, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"w_sf = {self.w_sf}, expected {expected}"
        return True, "w_sf consistent with relativistic addition formula"

    # ----------------------------------------------------------------------
    # Monotonicity / trend checks
    # ----------------------------------------------------------------------
    def test_monotonicity_w_fs_increases_with_v_B(self) -> Tuple[bool, str]:
        """
        For fixed v_A, the relative velocity w_fs is monotonically increasing
        with v_B. Perturb v_B by +0.5% and verify that w_fs increases.
        """
        if None in (self.v_A, self.v_B, self.w_fs):
            return False, "SKIP: v_A, v_B or w_fs missing"
        c_eff = self._c_effective()
        # Small positive perturbation of 0.5% of v_B
        epsilon = 0.005 * self.v_B
        v_B_pert = self.v_B + epsilon
        numerator_pert = v_B_pert - self.v_A
        denom_pert = 1.0 - (v_B_pert * self.v_A) / (c_eff ** 2)
        if denom_pert == 0:
            return False, "Denominator zero for perturbed w_fs"
        w_fs_pert = numerator_pert / denom_pert
        if w_fs_pert <= self.w_fs:
            return False, f"Perturbed w_fs = {w_fs_pert} does not exceed original {self.w_fs}"
        return True, "w_fs increases with v_B (monotonicity verified)"

    # ----------------------------------------------------------------------
    # Answer sanity (weak; only sign/finite checks)
    # ----------------------------------------------------------------------
    def test_answer_sanity_signs(self) -> Tuple[bool, str]:
        """w_fs should be positive, w_sf negative, both finite."""
        if self.w_fs is None or self.w_sf is None:
            return False, "SKIP: w_fs or w_sf not available"
        if self.w_fs <= 0:
            return False, f"w_fs = {self.w_fs} is not positive"
        if self.w_sf >= 0:
            return False, f"w_sf = {self.w_sf} is not negative"
        if not math.isfinite(self.w_fs) or not math.isfinite(self.w_sf):
            return False, "w_fs or w_sf is not finite"
        return True, "Signs and finiteness are correct"

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, execute them safely, collect logs."""
        logs: List[str] = []
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
        return all_pass, logs