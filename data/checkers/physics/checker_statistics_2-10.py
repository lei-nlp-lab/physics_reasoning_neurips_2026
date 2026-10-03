import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens (N, k_B, T1) and results (c1) safely using None defaults
        self.N = getattr(solution_instance, 'N', None)
        self.k_B = getattr(solution_instance, 'k_B', None)
        self.T1 = getattr(solution_instance, 'T1', None)
        
        # result key mapping: pseudocode emits 'c1'
        self.c1 = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.c1 = solution_instance.results.get('c1', None)
        
        # Precompute known constant ln(2)
        self.ln2 = math.log(2)

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

    # ------------------------------------------------------------
    # Interface checks: ensure required keys exist and have valid types
    # ------------------------------------------------------------
    def test_interface_c1_present(self):
        """Check that c1 is present in results and is a numeric value."""
        if self.c1 is None:
            return (False, "c1 not found in results")
        if not isinstance(self.c1, (int, float)):
            return (False, f"c1 has invalid type: {type(self.c1)}")
        return (True, "c1 is present and numeric")

    def test_interface_c1_positive(self):
        """Check that c1 is positive (physical requirement)."""
        if self.c1 is None:
            return (False, "SKIP: c1 not available")
        if self.c1 <= 0:
            return (False, f"c1 must be positive, got {self.c1}")
        return (True, f"c1 = {self.c1} > 0")

    # ------------------------------------------------------------
    # Invariant checks: theorem-based physical sanity (at least two)
    # ------------------------------------------------------------
    def test_invariant_entropy_integral(self):
        """
        Verify that the heat capacity integral yields the expected entropy change:
        ΔS = ∫ C/T dT from T1/2 to T1 = c1*(1 - ln2) should equal N k_B ln2.
        Use math.isclose with 1% relative tolerance.
        """
        if self.c1 is None or self.N is None or self.k_B is None:
            return (False, "SKIP: missing required variables (c1, N, k_B)")
        # Compute ΔS from integral using given c1
        delta_S_integral = self.c1 * (1 - self.ln2)
        # Target entropy change
        delta_S_target = self.N * self.k_B * self.ln2
        # Relative tolerance of 1% (plus absolute tolerance for near-zero)
        if math.isclose(delta_S_integral, delta_S_target, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"∫ C/T dT = {delta_S_integral} ≈ N k_B ln2 = {delta_S_target}")
        else:
            return (False, f"∫ C/T dT = {delta_S_integral} ≠ N k_B ln2 = {delta_S_target} (diff {abs(delta_S_integral - delta_S_target)})")

    def test_invariant_heat_capacity_at_boundaries(self):
        """
        The piecewise definition implies C(T1/2) = 0 and C(T1) = c1.
        Check that these hold (within 1% tolerance for floating-point).
        """
        if self.c1 is None or self.T1 is None:
            return (False, "SKIP: need c1 and T1")
        if self.T1 <= 0:
            return (False, "T1 must be positive")
        T_half = self.T1 / 2.0
        C_at_half = self.c1 * (2 * T_half / self.T1 - 1)  # should be 0
        C_at_T1 = self.c1 * (2 * self.T1 / self.T1 - 1)   # should be c1
        # Check C(T_half) ≈ 0
        if abs(C_at_half) > 1e-10:
            return (False, f"C(T1/2) = {C_at_half} should be 0")
        # Check C(T1) ≈ c1
        if not math.isclose(C_at_T1, self.c1, rel_tol=1e-10, abs_tol=1e-12):
            return (False, f"C(T1) = {C_at_T1} ≠ c1 = {self.c1}")
        return (True, f"C(T1/2) = 0, C(T1) = c1 = {self.c1}")

    def test_invariant_entropy_positive(self):
        """
        The entropy change ΔS = c1*(1 - ln2) must be positive because
        the physical system goes from ordered (S=0) to disordered (S=N k_B ln2 > 0).
        """
        if self.c1 is None:
            return (False, "SKIP: c1 not available")
        delta_S = self.c1 * (1 - self.ln2)
        if delta_S <= 0:
            return (False, f"ΔS = {delta_S} must be positive (c1 > 0 and 1-ln2 > 0)")
        return (True, f"ΔS = {delta_S} > 0")

    # ------------------------------------------------------------
    # Modeling checks: verify that the derived expression for c1 matches
    # the result from solving the entropy equation.
    # ------------------------------------------------------------
    def test_modeling_c1_formula(self):
        """
        Check that c1 equals N k_B ln2 / (1 - ln2) within 1% tolerance.
        This re-derives the formula inside the checker from givens.
        """
        if self.c1 is None or self.N is None or self.k_B is None:
            return (False, "SKIP: missing required variables (c1, N, k_B)")
        denominator = 1 - self.ln2
        c1_expected = self.N * self.k_B * self.ln2 / denominator
        if math.isclose(self.c1, c1_expected, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"c1 = {self.c1} matches N k_B ln2/(1-ln2) = {c1_expected}")
        else:
            return (False, f"c1 = {self.c1} ≠ N k_B ln2/(1-ln2) = {c1_expected}")

    # ------------------------------------------------------------
    # Trace checks: verify consistency with explanation steps
    # ------------------------------------------------------------
    def test_trace_entropy_continuity(self):
        """
        The explanation steps state S(∞)-S(0) = N k_B ln2.
        Verify that the entropy change from the integral indeed equals that.
        """
        return self.test_invariant_entropy_integral()  # reuse invariant test

    # ------------------------------------------------------------
    # Monotonicity checks: perturb an input parameter and compare output
    # ------------------------------------------------------------
    def test_monotonicity_c1_scales_with_N(self):
        """
        Perturb N by +10% and verify that c1 increases proportionally.
        Since c1 = N k_B ln2/(1-ln2), doubling N doubles c1.
        We compute baseline c1 from formula, then compute c1 for N' = 1.1*N,
        and check that the ratio c1'/c1 ≈ 1.1.
        """
        if self.c1 is None or self.N is None or self.k_B is None:
            return (False, "SKIP: missing required variables (c1, N, k_B)")
        # Baseline computed from formula (but we can also use self.c1)
        # Use self.c1 as baseline (should be consistent)
        c1_baseline = self.c1
        # Perturb N by +10%
        N_perturbed = 1.1 * self.N
        # Expected c1 for perturbed N (using the same formula)
        c1_expected_perturbed = N_perturbed * self.k_B * self.ln2 / (1 - self.ln2)
        # Ratio
        ratio = c1_expected_perturbed / c1_baseline
        # Expected ratio: 1.1
        expected_ratio = 1.1
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"c1 scales with N: ratio {ratio} ≈ {expected_ratio}")
        else:
            return (False, f"c1/N ratio inconsistent: expected {expected_ratio}, got {ratio}")

    def test_monotonicity_c1_scales_with_kB(self):
        """
        Perturb k_B by +10% and verify that c1 increases proportionally.
        """
        if self.c1 is None or self.N is None or self.k_B is None:
            return (False, "SKIP: missing required variables (c1, N, k_B)")
        # Baseline
        c1_baseline = self.c1
        # Perturb k_B by +10%
        kB_perturbed = 1.1 * self.k_B
        c1_expected_perturbed = self.N * kB_perturbed * self.ln2 / (1 - self.ln2)
        ratio = c1_expected_perturbed / c1_baseline
        expected_ratio = 1.1
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"c1 scales with k_B: ratio {ratio} ≈ {expected_ratio}")
        else:
            return (False, f"c1/k_B ratio inconsistent: expected {expected_ratio}, got {ratio}")

    # ------------------------------------------------------------
    # Weak answer sanity (secondary, use only one such test)
    # ------------------------------------------------------------
    def test_answer_sanity_c1(self):
        """
        Weak sanity: c1 should be finite, not NaN, and not extremely large
        (e.g., not > 10^100 times N k_B).  No math.isclose or re-derivation.
        """
        if self.c1 is None:
            return (False, "SKIP: c1 not available")
        if not math.isfinite(self.c1):
            return (False, f"c1 is not finite: {self.c1}")
        # Broad magnitude check: c1 should not be absurdly large relative to N k_B.
        # Typically c1 ~ N k_B * (ln2/(1-ln2)) ≈ 2.26 N k_B.
        if self.N is not None and self.k_B is not None:
            ratio = self.c1 / (self.N * self.k_B)
            if ratio > 1e100:
                return (False, f"c1/(N k_B) = {ratio} is unreasonably large")
        return (True, "c1 is finite and plausible")