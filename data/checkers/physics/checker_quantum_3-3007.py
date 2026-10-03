import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits solution for spin-orbit coupling problem: l=2 ħ, s=1 ħ, H_so = A L·S.
    Checks interface completeness, theorem-based invariants, modeling quality,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """
        Safely extract all required fields. Use None defaults for missing data.
        """
        # Extract results dictionary; if missing, use empty dict.
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Result keys from pseudocode [emit] contract
        self.j_values = results.get('j_values', None)      # list of integers
        self.deltaE = results.get('deltaE', None)          # list of floats
        self.deg = results.get('deg', None)                # list of ints

        # Optional explanation steps (if provided)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Problem constants (may not be present in results; handle missing)
        self.l = 2      # fixed
        self.s = 1      # fixed
        self.hbar = 1   # simplified units; if given later, override
        # Try to extract hbar if solution provides it
        givens = getattr(solution_instance, 'givens', {})
        if givens and 'hbar' in givens:
            self.hbar = givens['hbar']

        # Constant A might be given; not needed for most checks
        self.A = None
        if givens and 'A' in givens:
            self.A = givens['A']

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_j_values(self) -> Tuple[bool, str]:
        """Check that j_values is present, a list of length 3 containing ints."""
        if self.j_values is None:
            return (False, "Missing j_values in results")
        if not isinstance(self.j_values, list):
            return (False, f"j_values should be a list, got {type(self.j_values)}")
        if len(self.j_values) != 3:
            return (False, f"j_values must have exactly 3 entries, got {len(self.j_values)}")
        for j in self.j_values:
            if not isinstance(j, (int, float)):
                return (False, f"j_values element {j} is not numeric")
            if j != int(j):
                return (False, f"j_values element {j} should be an integer")
        # Optional: check they are 1,2,3 in some order (usually sorted)
        # We'll check sorted after casting to int
        sorted_js = sorted(int(x) for x in self.j_values)
        if sorted_js != [1,2,3]:
            return (False, f"j_values should contain {1,2,3}, got {sorted_js}")
        return (True, "j_values valid")

    def test_interface_deltaE(self) -> Tuple[bool, str]:
        """Check that deltaE is present and a list of numeric values."""
        if self.deltaE is None:
            return (False, "Missing deltaE in results")
        if not isinstance(self.deltaE, list):
            return (False, f"deltaE should be a list, got {type(self.deltaE)}")
        if len(self.deltaE) != 3:
            return (False, f"deltaE must have exactly 3 entries, got {len(self.deltaE)}")
        for val in self.deltaE:
            if not isinstance(val, (int, float)):
                return (False, f"deltaE element {val} is not numeric")
        return (True, "deltaE list valid (length 3)")

    def test_interface_deg(self) -> Tuple[bool, str]:
        """Check that deg is present and a list of positive integers."""
        if self.deg is None:
            return (False, "Missing deg in results")
        if not isinstance(self.deg, list):
            return (False, f"deg should be a list, got {type(self.deg)}")
        if len(self.deg) != 3:
            return (False, f"deg must have exactly 3 entries, got {len(self.deg)}")
        for d in self.deg:
            if not isinstance(d, (int, float)) or d != int(d):
                return (False, f"deg element {d} is not an integer")
            if d <= 0:
                return (False, f"deg element {d} is not positive")
        return (True, "deg list valid (length 3, positive integers)")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_deltaE_formula(self) -> Tuple[bool, str]:
        """
        For each j in j_values, verify that deltaE satisfies:
            ΔE = (A ħ²/2) [ j(j+1) - l(l+1) - s(s+1) ]
        Since A may not be known, we check consistency of ratios across js.
        i.e., for any two js, the ratio of ΔE values should equal ratio of
        [j(j+1)-l(l+1)-s(s+1)].
        """
        if any(x is None for x in [self.j_values, self.deltaE]):
            return (False, "SKIP: j_values or deltaE missing")
        if len(self.j_values) != len(self.deltaE):
            return (False, "Mismatch between j_values and deltaE lengths")
        l, s = 2, 1
        L2 = l*(l+1)
        S2 = s*(s+1)

        # Compute the dimensionless factor for each j
        factors = {}
        for j in self.j_values:
            j = int(j)
            factor = j*(j+1) - L2 - S2
            factors[j] = factor

        # Compare ratios between pairs (skip if factor=0 for all? but we have three distinct)
        # Use j=1 and j=3 as reference (factors: j=1:2-6-2=-6; j=2:6-6-2=-2; j=3:12-6-2=4)
        # Ratio ΔE_j / factor_j should be constant
        # Use math.isclose for fuzzy matching (float noise)
        ref_j = self.j_values[0]
        ref_idx = 0
        ref_factor = int(ref_j)*(int(ref_j)+1) - L2 - S2
        ref_delta = self.deltaE[ref_idx]
        for idx, j in enumerate(self.j_values[1:], start=1):
            factor = int(j)*(int(j)+1) - L2 - S2
            delta = self.deltaE[idx]
            # Compare (delta / factor) vs (ref_delta / ref_factor)
            # To avoid division by zero: if either factor or ref_factor is zero, skip.
            # In our problem none are zero.
            ratio1 = delta / factor
            ratio2 = ref_delta / ref_factor
            if not math.isclose(ratio1, ratio2, rel_tol=1e-2, abs_tol=1e-2):
                return (False, 
                    f"Ratio mismatch for j={j}: ΔE/factor={ratio1:.4f}, "
                    f"expected ~{ratio2:.4f} from j={ref_j}")
        return (True, "ΔE ratios consistent with L·S eigenvalue formula")

    def test_invariant_degeneracy_formula(self) -> Tuple[bool, str]:
        """For each j in j_values, deg should equal 2j+1."""
        if any(x is None for x in [self.j_values, self.deg]):
            return (False, "SKIP: j_values or deg missing")
        if len(self.j_values) != len(self.deg):
            return (False, "Mismatch between j_values and deg lengths")
        for j, d in zip(self.j_values, self.deg):
            expected = 2*int(j) + 1
            if d != expected:
                return (False, f"j={j}: deg={d}, expected {expected}")
        return (True, "All degeneracies satisfy d=2j+1")

    # ------------------------------------------------------------------
    # Modeling/trace tests (based on explanation steps structure)
    # ------------------------------------------------------------------
    def test_modeling_ordering(self) -> Tuple[bool, str]:
        """
        Check that deltaE values are in strictly increasing order with j,
        and deg values are also increasing.
        """
        if any(x is None for x in [self.j_values, self.deltaE, self.deg]):
            return (False, "SKIP: required lists missing")
        # pair by sorted j
        pairs = sorted(zip(self.j_values, self.deltaE, self.deg), key=lambda x: x[0])
        # check deltaE monotonic increasing
        for i in range(len(pairs)-1):
            if pairs[i][1] >= pairs[i+1][1]:
                return (False, 
                    f"deltaE not strictly increasing: {pairs[i][1]} >= {pairs[i+1][1]} "
                    f"for j={pairs[i][0]} vs j={pairs[i+1][0]}")
            if pairs[i][2] >= pairs[i+1][2]:
                return (False, 
                    f"deg not strictly increasing: {pairs[i][2]} >= {pairs[i+1][2]} "
                    f"for j={pairs[i][0]} vs j={pairs[i+1][0]}")
        return (True, "Ordering of deltaE and deg is correct (increasing with j)")

    def test_modeling_symmetry(self) -> Tuple[bool, str]:
        """
        For l=2, s=1, the sum over j of (2j+1) should equal (2l+1)(2s+1) = 5*3=15.
        This checks completeness of basis.
        """
        if self.deg is None:
            return (False, "SKIP: deg missing")
        total_deg = sum(self.deg)
        expected_total = (2*self.l+1)*(2*self.s+1)  # 5*3=15
        if total_deg != expected_total:
            return (False, f"Total degeneracy {total_deg} != expected {expected_total}")
        return (True, f"Total degeneracy = {total_deg} matches (2l+1)(2s+1)=15")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation on input parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_scaling_with_A(self) -> Tuple[bool, str]:
        """
        If coefficient A is provided in givens, scaling A by a factor (e.g. +10%) 
        should scale all deltaE by the same factor. Since we only have one solution,
        we simulate by creating a scaled deltaE list from the existing one.
        (This tests that the output is linear in A.)
        """
        if self.deltaE is None:
            return (False, "SKIP: deltaE missing")
        # We'll perturb by a factor 1.1 relative to the reported values.
        # Since A is not in results, we just check linearity i.e. 
        # deltaE_new = factor * deltaE_old as inferred from physical law.
        # We simulate by taking the reported deltaE and checking that if we 
        # multiply by a factor, the ratios among elements remain the same.
        # Actually, we already tested ratio consistency in invariant test.
        # Instead, we can check that if we treat A as unknown, the ratios 
        # between elements are fixed. We already did that.
        # For monotonicity, we can check that sign of deltaE correlates with A sign.
        # But A not known. Alternatively, check that deltaE for j=3 is positive 
        # if A>0 (could be negative if A negative). Not reliable.
        # Let's check that multiplying all deltaE by a constant factor preserves ordering.
        # That's trivial. We'll skip this monotonicity test if no A provided.
        if self.A is None:
            return (False, "SKIP: A not available for monotonicity test")
        # We'll create a hypothetical alternative solution where A is 10% larger.
        # Since we only have one solution, we can't test that; we would need to call 
        # the solver again. So skip.
        return (False, "SKIP: monotonicity test requires solving with perturbed A; not implemented")

    # Alternative monotonicity: deg increases with j – already covered in ordering.

    # ------------------------------------------------------------------
    # Weak answer sanity (do not dominate; only sign/finite checks)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """
        Weak sanity: all deltaE finite, all deg positive, ordering preserved.
        Already largely covered by other tests. Here just ensure no NaN.
        """
        if any(x is None for x in [self.deltaE, self.deg]):
            return (False, "SKIP: deltaE or deg missing")
        for val in self.deltaE:
            if math.isnan(val) or math.isinf(val):
                return (False, f"deltaE contains NaN or Inf: {val}")
        for d in self.deg:
            if d <= 0 or math.isnan(d) or math.isinf(d):
                return (False, f"deg contains invalid value: {d}")
        return (True, "All deltaE finite, all deg positive integers")

    # ------------------------------------------------------------------
    # Master test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, run them in try/except, collect results.
        Returns (all_passed, logs).
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)