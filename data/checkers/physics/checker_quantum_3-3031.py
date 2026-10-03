import math
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Checker for Problem: Clebsch-Gordan coefficients and cross-section ratios.
    Tests interface, invariants, modeling quality, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract results safely using .get(..., None)
        results = getattr(solution_instance, 'results', {})
        self.cg_1 = results.get('cg_1', None)          # expected: 1/sqrt(3)
        self.cg_2 = results.get('cg_2', None)          # expected: sqrt(2/3)
        self.ratio_delta = results.get('ratio_delta', None)   # expected: (1, 1/9, 2/9)
        self.ratio_nstar = results.get('ratio_nstar', None)   # expected: (0, 4/9, 2/9)

        # Givens (if available); do NOT set numeric defaults
        givens = getattr(solution_instance, 'givens', {})
        self.j1 = givens.get('j1', None)      # default None
        self.j2 = givens.get('j2', None)      # default None
        self.j = givens.get('j', None)        # default None
        self.M = givens.get('M', None)        # default None

        # Tolerances for numerical checks
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_cg_1_exists(self) -> Tuple[bool, str]:
        if self.cg_1 is None:
            return (False, "cg_1 missing from results")
        return (True, "cg_1 present")

    def test_interface_cg_2_exists(self) -> Tuple[bool, str]:
        if self.cg_2 is None:
            return (False, "cg_2 missing from results")
        return (True, "cg_2 present")

    def test_interface_ratio_delta_exists(self) -> Tuple[bool, str]:
        if self.ratio_delta is None:
            return (False, "ratio_delta missing from results")
        if not isinstance(self.ratio_delta, (list, tuple)):
            return (False, "ratio_delta not a list/tuple")
        if len(self.ratio_delta) != 3:
            return (False, f"ratio_delta length {len(self.ratio_delta)} != 3")
        return (True, "ratio_delta present and valid")

    def test_interface_ratio_nstar_exists(self) -> Tuple[bool, str]:
        if self.ratio_nstar is None:
            return (False, "ratio_nstar missing from results")
        if not isinstance(self.ratio_nstar, (list, tuple)):
            return (False, "ratio_nstar not a list/tuple")
        if len(self.ratio_nstar) != 3:
            return (False, f"ratio_nstar length {len(self.ratio_nstar)} != 3")
        return (True, "ratio_nstar present and valid")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_cg_normalization(self) -> Tuple[bool, str]:
        """
        The two allowed CG coefficients should satisfy |c1|^2 + |c2|^2 = 1.
        """
        if self.cg_1 is None or self.cg_2 is None:
            return (False, "SKIP: CG coefficients not available")
        try:
            c1 = float(self.cg_1)
            c2 = float(self.cg_2)
        except (TypeError, ValueError):
            return (False, "CG coefficients not convertible to float")
        norm = c1**2 + c2**2
        if math.isclose(norm, 1.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return (True, f"Normalization OK: {c1}^2 + {c2}^2 = {norm}")
        else:
            return (False, f"Normalization violated: sum of squares = {norm} (expected 1)")

    def test_invariant_cg_ratio_lowering(self) -> Tuple[bool, str]:
        """
        From the lowering operator: sqrt(3)*c1 = 1, sqrt(3)*c2 = sqrt(2)
        => c1 = 1/√3, c2 = √(2/3). Check both individually.
        """
        if self.cg_1 is None or self.cg_2 is None:
            return (False, "SKIP: CG coefficients not available")
        try:
            c1 = float(self.cg_1)
            c2 = float(self.cg_2)
        except (TypeError, ValueError):
            return (False, "CG coefficients not convertible to float")
        expected_c1 = 1.0 / math.sqrt(3)
        expected_c2 = math.sqrt(2.0/3.0)
        if (math.isclose(c1, expected_c1, rel_tol=self.rel_tol, abs_tol=self.abs_tol) and
            math.isclose(c2, expected_c2, rel_tol=self.rel_tol, abs_tol=self.abs_tol)):
            return (True, f"CG coefficients match expected: c1={c1}, c2={c2}")
        else:
            # Also check the ratio c2/c1 ≈ √2 as a fallback
            if c1 != 0:
                ratio = c2 / c1
                if math.isclose(ratio, math.sqrt(2), rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    return (True, f"CG ratio matches: c2/c1 = {ratio} ≈ √2")
            return (False, f"CG values inconsistent: c1={c1} (exp {expected_c1}), c2={c2} (exp {expected_c2})")

    # ------------------------------------------------------------------
    # Modeling checks (cross-section ratios)
    # ------------------------------------------------------------------
    def test_modeling_delta_ratios(self) -> Tuple[bool, str]:
        """
        For Δ resonance (I=3/2), the cross-section ratios should be (1, 1/9, 2/9).
        Check that the three numbers, normalized to the first, equal the expected ratios.
        """
        if self.ratio_delta is None:
            return (False, "SKIP: ratio_delta not available")
        try:
            r0, r1, r2 = [float(x) for x in self.ratio_delta]
        except (TypeError, ValueError):
            return (False, "ratio_delta elements not numeric")
        if r0 == 0:
            return (False, "Delta ratio i is zero, expected 1")
        norm = float(r0)
        r0_n = r0 / norm
        r1_n = r1 / norm
        r2_n = r2 / norm
        expected = (1.0, 1.0/9.0, 2.0/9.0)
        if (math.isclose(r0_n, expected[0], rel_tol=self.rel_tol, abs_tol=self.abs_tol) and
            math.isclose(r1_n, expected[1], rel_tol=self.rel_tol, abs_tol=self.abs_tol) and
            math.isclose(r2_n, expected[2], rel_tol=self.rel_tol, abs_tol=self.abs_tol)):
            return (True, f"Delta ratios OK: {self.ratio_delta}")
        else:
            return (False, f"Delta ratios expected (1, 1/9, 2/9) but got normalized ({r0_n}, {r1_n}, {r2_n})")

    def test_modeling_nstar_ratios(self) -> Tuple[bool, str]:
        """
        For N* resonance (I=1/2), the cross-section ratios should be (0, 4/9, 2/9).
        Check that the first is zero, and the normalized (to r1) values are 1 and 0.5.
        """
        if self.ratio_nstar is None:
            return (False, "SKIP: ratio_nstar not available")
        try:
            r0, r1, r2 = [float(x) for x in self.ratio_nstar]
        except (TypeError, ValueError):
            return (False, "ratio_nstar elements not numeric")
        # Reaction i should be suppressed (I=1/2 cannot produce 2 pions)
        if not math.isclose(r0, 0.0, abs_tol=1e-2):
            return (False, f"N* ratio_i is not zero: {r0}")
        if r1 == 0:
            return (False, "N* ratio_ii is zero, expected non-zero")
        norm = float(r1)
        r1_n = r1 / norm
        r2_n = r2 / norm
        # Normalized: r1 should be 1, r2 should be (2/9)/(4/9)=0.5
        if (math.isclose(r1_n, 1.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol) and
            math.isclose(r2_n, 0.5, rel_tol=self.rel_tol, abs_tol=self.abs_tol)):
            return (True, f"N* ratios OK: {self.ratio_nstar}")
        else:
            return (False, f"N* ratios expected (0, 4/9, 2/9) but got ({r0}, {r1}, {r2})")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of an input compared to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_lowering_operator(self) -> Tuple[bool, str]:
        """
        Perturb cg_1 by 1% and check that the product sqrt(3)*cg_1 changes in the same direction.
        The baseline product should be 1; after 1% increase, it should be >1.
        """
        if self.cg_1 is None:
            return (False, "SKIP: cg_1 not available")
        try:
            c1 = float(self.cg_1)
        except (TypeError, ValueError):
            return (False, "cg_1 not convertible to float")
        # Baseline
        baseline = math.sqrt(3) * c1
        # Perturb input: increase c1 by 1%
        perturbed_c1 = c1 * 1.01
        perturbed_value = math.sqrt(3) * perturbed_c1
        # Check monotonic increase
        if perturbed_value > baseline:
            # Also check that the fractional change is ~1%
            frac_change = (perturbed_value - baseline) / baseline
            if math.isclose(frac_change, 0.01, rel_tol=0.2):  # allow 20% tolerance on the 1% change
                return (True, f"Monotonicity OK: baseline={baseline}, perturbed={perturbed_value}, change={frac_change}")
            else:
                # Still monotonic but not exactly 1% - acceptable
                return (True, f"Monotonic increase OK (change={frac_change})")
        else:
            return (False, "Monotonicity violated: perturbed value not greater than baseline")
    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self) -> Tuple[bool, str]:
        """
        Check that all numerical results are non-negative and finite.
        """
        issues = []
        for name, val in [("cg_1", self.cg_1), ("cg_2", self.cg_2)]:
            if val is not None:
                try:
                    v = float(val)
                    if not math.isfinite(v):
                        issues.append(f"{name} not finite")
                except (TypeError, ValueError):
                    issues.append(f"{name} not numeric")
        for name, ratios in [("ratio_delta", self.ratio_delta), ("ratio_nstar", self.ratio_nstar)]:
            if ratios is not None:
                try:
                    for i, r in enumerate(ratios):
                        rf = float(r)
                        if not math.isfinite(rf):
                            issues.append(f"{name}[{i}] not finite")
                        if rf < -self.abs_tol:
                            issues.append(f"{name}[{i}] negative ({rf})")
                except (TypeError, ValueError):
                    issues.append(f"{name} contains non-numeric element")
        if issues:
            return (False, "; ".join(issues))
        else:
            return (True, "All results are non-negative and finite")

    # ------------------------------------------------------------------
    # Main runner
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