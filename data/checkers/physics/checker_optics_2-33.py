import math
import traceback
from typing import Any, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits solutions for Fresnel diffraction through a circular aperture.
    Problem: plane wave λ=1cm, detector on axis at 1m, aperture radius increased from zero.
    On-axis intensity I ∝ sin²(πa²/(2λz)).
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results with safe defaults (None for missing)
        results = getattr(solution_instance, 'results', {}) or {}
        self.a1: Optional[float] = results.get('a1', None)
        self.a2: Optional[float] = results.get('a2', None)
        self.z_max_list: Optional[List[float]] = results.get('z_max_list', None)
        self.z_min_list: Optional[List[float]] = results.get('z_min_list', None)

        # Problem constants (fixed, from problem statement)
        self.lambda_ = 0.01  # 1 cm in meters
        self.z0 = 1.0        # detector distance in meters

    def _safe_list(self, lst: Any) -> List[float]:
        """Convert None or non-list to empty list for safe iteration."""
        if not isinstance(lst, (list, tuple)):
            return []
        return list(lst)

    def _compute_z_max(self, a: float, m: int) -> float:
        """z_max for given aperture radius a and integer m (0-indexed)."""
        return a**2 / (2.0 * self.lambda_ * (m + 0.5))

    def _compute_z_min(self, a: float, m: int) -> float:
        """z_min for given aperture radius a and integer m (1-indexed)."""
        return a**2 / (2.0 * self.lambda_ * m)

    # ------------------------------------------------------------------
    #  Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all expected result keys are present and non-None."""
        missing = []
        if self.a1 is None:
            missing.append('a1')
        if self.a2 is None:
            missing.append('a2')
        if self.z_max_list is None:
            missing.append('z_max_list')
        if self.z_min_list is None:
            missing.append('z_min_list')
        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "All required keys present (a1, a2, z_max_list, z_min_list)"

    # ------------------------------------------------------------------
    #  Theorem-based invariant tests (use math.isclose with 1% tolerance)
    # ------------------------------------------------------------------
    def test_invariant_a1_fresnel_first_max(self) -> Tuple[bool, str]:
        """a1 must satisfy π a1²/(2λz0) = π/2  → a1 = √(λz0) = 0.1 m"""
        if self.a1 is None:
            return False, "SKIP: a1 not available"
        expected_a1 = math.sqrt(self.lambda_ * self.z0)  # 0.1
        if math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"a1={self.a1:.4f} matches expected {expected_a1:.4f}"
        return False, f"a1={self.a1:.4f} differs from expected {expected_a1:.4f}"

    def test_invariant_a2_fresnel_second_min(self) -> Tuple[bool, str]:
        """a2 must satisfy π a2²/(2λz0) = 2π → a2 = 2√(λz0) = 0.2 m"""
        if self.a2 is None:
            return False, "SKIP: a2 not available"
        expected_a2 = 2.0 * math.sqrt(self.lambda_ * self.z0)  # 0.2
        if math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"a2={self.a2:.4f} matches expected {expected_a2:.4f}"
        return False, f"a2={self.a2:.4f} differs from expected {expected_a2:.4f}"

    def test_invariant_z_max_formula(self) -> Tuple[bool, str]:
        """
        For each z_max, π a2²/(2λ z) should equal (m+0.5)π for integer m.
        Check: z = a2²/(2λ(m+0.5)) for some integer m.
        """
        if self.a2 is None or not self.z_max_list:
            return False, "SKIP: a2 or z_max_list missing"
        a2_sq = self.a2 ** 2
        for idx, z in enumerate(self._safe_list(self.z_max_list)):
            if z <= 0:
                return False, f"z_max[{idx}]={z} must be positive"
            raw_m = a2_sq / (2.0 * self.lambda_ * z) - 0.5
            m = int(round(raw_m))
            if m < 0:
                return False, f"z_max[{idx}]={z} leads to negative m={m}"
            lhs = a2_sq / (2.0 * self.lambda_ * z)
            rhs = m + 0.5
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"z_max[{idx}]={z}: a2²/(2λz)={lhs:.4f} != {rhs:.4f} (m={m})"
        return True, f"All z_max values satisfy z = a2²/(2λ(m+0.5)) with integer m>=0"

    def test_invariant_z_min_formula(self) -> Tuple[bool, str]:
        """
        For each z_min, π a2²/(2λ z) = mπ with integer m>=1.
        Check: z = a2²/(2λ m)
        """
        if self.a2 is None or not self.z_min_list:
            return False, "SKIP: a2 or z_min_list missing"
        a2_sq = self.a2 ** 2
        for idx, z in enumerate(self._safe_list(self.z_min_list)):
            if z <= 0:
                return False, f"z_min[{idx}]={z} must be positive"
            raw_m = a2_sq / (2.0 * self.lambda_ * z)
            m = int(round(raw_m))
            if m < 1:
                return False, f"z_min[{idx}]={z} leads to m={m} (should be >=1)"
            lhs = a2_sq / (2.0 * self.lambda_ * z)
            rhs = m
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"z_min[{idx}]={z}: a2²/(2λz)={lhs:.4f} != {rhs:.4f} (m={m})"
        return True, f"All z_min values satisfy z = a2²/(2λ m) with integer m>=1"

    # ------------------------------------------------------------------
    #  Modeling / trace quality tests (re-derive structural relations)
    # ------------------------------------------------------------------
    def test_modeling_z_max_min_interleaving(self) -> Tuple[bool, str]:
        """
        The axial order (decreasing z) should be: max, min, max, min, ...
        Check interleaving: z_max[0] > z_min[0] > z_max[1] > z_min[1] > ...
        """
        z_max = self._safe_list(self.z_max_list)
        z_min = self._safe_list(self.z_min_list)
        if not z_max or not z_min:
            return False, "SKIP: missing z_max or z_min lists"
        n = min(len(z_max), len(z_min))
        if n == 0:
            return False, "SKIP: no pairs to compare"
        for i in range(n):
            if not (z_max[i] > z_min[i]):
                return False, f"Failed: z_max[{i}]={z_max[i]} should be > z_min[{i}]={z_min[i]}"
            if i + 1 < len(z_max):
                if not (z_min[i] > z_max[i + 1]):
                    return False, f"Failed: z_min[{i}]={z_min[i]} should be > z_max[{i+1}]={z_max[i+1]}"
        return True, "z_max and z_min lists are properly interleaved"

    def test_modeling_z_max_ratio(self) -> Tuple[bool, str]:
        """
        For consecutive z_max values (with m=0,1,2,...), the ratio
        z_max[m] / z_max[m+1] should equal (m+1.5)/(m+0.5).
        This is derived from the formula z_max(m) = a2²/(2λ(m+0.5)).
        """
        if self.a2 is None or not self.z_max_list:
            return False, "SKIP: a2 or z_max_list missing"
        lst = self._safe_list(self.z_max_list)
        if len(lst) < 2:
            return False, "SKIP: need at least two z_max values"
        for i in range(len(lst) - 1):
            m = i  # first corresponds to m=0, second to m=1, etc.
            expected_ratio = (m + 1.5) / (m + 0.5)
            if lst[i] <= 0 or lst[i + 1] <= 0:
                return False, f"Non-positive z_max value at index {i} or {i+1}"
            actual_ratio = lst[i] / lst[i + 1]
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"z_max[{i}]/z_max[{i+1}] = {actual_ratio:.4f}, expected {expected_ratio:.4f}"
        return True, "Consecutive z_max ratios match theoretical values"

    def test_modeling_z_min_ratio(self) -> Tuple[bool, str]:
        """
        For consecutive z_min values (with m=1,2,3,...), the ratio
        z_min[m] / z_min[m+1] should equal (m+1)/m.
        """
        if self.a2 is None or not self.z_min_list:
            return False, "SKIP: a2 or z_min_list missing"
        lst = self._safe_list(self.z_min_list)
        if len(lst) < 2:
            return False, "SKIP: need at least two z_min values"
        for i in range(len(lst) - 1):
            m = i + 1  # first corresponds to m=1, second to m=2, etc.
            expected_ratio = (m + 1) / m
            if lst[i] <= 0 or lst[i + 1] <= 0:
                return False, f"Non-positive z_min value at index {i} or {i+1}"
            actual_ratio = lst[i] / lst[i + 1]
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"z_min[{i}]/z_min[{i+1}] = {actual_ratio:.4f}, expected {expected_ratio:.4f}"
        return True, "Consecutive z_min ratios match theoretical values"

    # ------------------------------------------------------------------
    #  Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_a2_increases_axial_positions(self) -> Tuple[bool, str]:
        """
        If we increase a2 by +1%, then z_max[0] and z_min[0] should increase
        by approximately +2% (since z ∝ a²).
        We recompute using the formula with a perturbed a2.
        """
        if self.a2 is None:
            return False, "SKIP: a2 not available"
        baseline_a2 = self.a2
        perturbed_a2 = baseline_a2 * 1.01  # +1%
        # baseline z_max[0] for m=0: z = a²/(λ)
        # Actually z_max(0) = a²/(2λ*(0+0.5)) = a²/(λ)
        baseline_z_max = baseline_a2**2 / self.lambda_
        perturbed_z_max = perturbed_a2**2 / self.lambda_
        expected_increase = (1.01**2 - 1.0) * 100  # ~2.01%
        actual_increase = (perturbed_z_max / baseline_z_max - 1.0) * 100
        if not math.isclose(actual_increase, expected_increase, rel_tol=1e-2, abs_tol=0.05):
            return False, f"a2 +1% gave z_max increase {actual_increase:.3f}%, expected ~{expected_increase:.3f}%"
        return True, f"a2 +1% increases axial distances by ~{(1.01**2-1)*100:.2f}% (consistent with ∝a²)"

    def test_monotonicity_lambda_increases_decreases_z(self) -> Tuple[bool, str]:
        """
        If we increase the wavelength by +1%, then axial positions for fixed a2
        should decrease by approximately -1% (since z ∝ 1/λ).
        We test using z_max[0] formula.
        """
        if self.a2 is None:
            return False, "SKIP: a2 not available"
        baseline_lambda = self.lambda_
        perturbed_lambda = baseline_lambda * 1.01  # +1%
        # For a fixed aperture radius a2, z_max(0) = a2²/(λ)
        baseline_z = self.a2**2 / baseline_lambda
        perturbed_z = self.a2**2 / perturbed_lambda
        expected_change = (1/1.01 - 1.0) * 100  # ~ -0.99%
        actual_change = (perturbed_z / baseline_z - 1.0) * 100
        if not math.isclose(actual_change, expected_change, rel_tol=1e-2, abs_tol=0.05):
            return False, f"λ +1% gave z change {actual_change:.3f}%, expected ~{expected_change:.3f}%"
        return True, f"λ +1% decreases axial distances by ~{(1/1.01-1)*100:.2f}% (consistent with ∝1/λ)"

    # ------------------------------------------------------------------
    #  Answer sanity (weak – no re-derivation, no isclose, no target values)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self) -> Tuple[bool, str]:
        """All radii and distances must be positive, finite numbers."""
        all_vals = []
        if self.a1 is not None:
            all_vals.append(self.a1)
        if self.a2 is not None:
            all_vals.append(self.a2)
        if self.z_max_list is not None:
            all_vals.extend(self.z_max_list)
        if self.z_min_list is not None:
            all_vals.extend(self.z_min_list)
        if not all_vals:
            return False, "No numerical values to check"
        for v in all_vals:
            if not isinstance(v, (int, float)):
                return False, f"Non-numeric value encountered: {v}"
            if v <= 0 or math.isinf(v) or math.isnan(v):
                return False, f"Non-positive or non-finite value: {v}"
        return True, "All values are positive, finite numbers"

    # ------------------------------------------------------------------
    #  Master runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, execute them, collect logs."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs